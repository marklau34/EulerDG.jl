# =====================================
# Semidiscretization: the DG right-hand side
#
# Downstream of the mesh contract. The weak form, evaluated by explicit quadrature on every
# element:
#
#     |K| du_i/dt = ∫_K ∇φ_i · F(U) dx - ∮_∂K φ_i F̂(U⁻, U⁺, n) ds
#
# The solver data -- quadrature points and weights, basis values, face vectors per quadrature
# point -- is built here from a `PolygonMesh`, not stored in the contract, so curved faces later
# change only this data. Each face quadrature point carries its weight times the face vector, and
# Trixi-style fluxes scale with the length of their normal direction, so a face integral is a sum
# of plain flux calls.
#
# The state and the residual are 4 x 3 x elements arrays: variable, mode, element.

"""
    EulerDG.Semidiscretization

The weak-form modal DG right-hand side, P1 on every element.

- [`DGSemidiscretization`](@ref): element and face solver data built from a mesh, with the
  equations, the surface flux, one boundary condition per named boundary and optional source
  terms.
- [`rhs!`](@ref) in two threaded passes: the flux at every face Gauss point, then each
  element's volume integrals, sources and face integrals, divided by |K|.
- [`allocate_state`](@ref), [`project`](@ref), [`evaluate`](@ref), [`cell_average`](@ref).
- The plan's step formula: [`local_dt!`](@ref), each element's own step (Phase 3), and
  [`max_dt`](@ref), the smallest of them. [`assert_positive`](@ref) checks the state.

The loops over faces and elements are KernelAbstractions kernels, threaded on the CPU backend. No
two threads ever write to the same element, and each element's sums run in the order of a serial
loop over its faces, so the results do not depend on the number of threads.
"""
module Semidiscretization

using KernelAbstractions: @index, @kernel, CPU
using StaticArrays: SMatrix, SVector
using ..EulerDG: PolygonMesh, nelements, nfaces, element_nodes, interior_faces, boundary_faces
using ..Quadrature: GAUSS2, polygon_rule
using ..Basis: ElementBasis, element_basis, basis_values, basis_gradients
using ..Equations: cons2prim, flux, flux_lax_friedrichs, max_abs_speed

export DGSemidiscretization, allocate_state, project, evaluate, cell_average, rhs!, max_dt
export local_dt!, assert_positive, face_trace

"""
    DGSemidiscretization{T,M,EQ,SF,BC,ST,B}

The P1 DG discretization of `equations` on a `PolygonMesh`, as struct-of-arrays:

- per element: its basis, the gradients of its modes, and its mass (the mass matrix is `mass[e]`
  times the identity);
- per volume quadrature point, in CSR order by element (`qptr`): position, weight and mode values;
- per face, at its two Gauss points on the left copy: position, weight times face vector, and the
  mode values of the left and right elements (the right ones at the point moved by the face's
  periodic shift);
- per element, in the mesh's `elem_ptr` layout: its faces in increasing order (`sides`), as `+f`
  where it is face `f`'s left element and `-f` where it is the right one.

It also holds the kernels' work buffers: the face fluxes, the face and element wave speeds, and
the positivity flags. Two solves at once need two semidiscretizations.

Build it with `DGSemidiscretization(mesh, equations; ...)`.
"""
mutable struct DGSemidiscretization{T,M<:PolygonMesh{T},EQ,SF,BC,ST,B}
    # Mutable, with every field constant, so that a kernel launch passes it by reference: the CPU
    # backend boxes its launch arguments, and boxing a large immutable struct copies it each time.
    const mesh::M
    const equations::EQ
    const surface_flux::SF
    const boundary_conditions::BC
    const source_terms::ST
    const backend::B
    const grainsize::Int
    const basis::Vector{ElementBasis{T}}
    const gradients::Vector{SMatrix{2,3,T,6}}
    const mass::Vector{T}
    const qptr::Vector{Int32}
    const qpoint::Vector{SVector{2,T}}
    const qweight::Vector{T}
    const qphi::Vector{SVector{3,T}}
    const fpoint::Vector{NTuple{2,SVector{2,T}}}
    const fnormal::Vector{NTuple{2,SVector{2,T}}}
    const fphi_left::Vector{NTuple{2,SVector{3,T}}}
    const fphi_right::Vector{NTuple{2,SVector{3,T}}}
    const sides::Vector{Int32}
    const fhat::Vector{NTuple{2,SVector{4,T}}}
    const fspeed::Vector{T}
    const speed::Vector{T}
    const positivity::Vector{Int32}
end

"""
    DGSemidiscretization(mesh, equations; surface_flux=flux_lax_friedrichs,
                         boundary_conditions=(;), source_terms=nothing, volume_degree=2,
                         backend=CPU(), grainsize=64)

Build the discretization.

- `boundary_conditions` is a NamedTuple with one condition per name in `mesh.boundary_names`,
  each called as `bc(u_inner, normal_direction, x, t, surface_flux, equations)` like Trixi's.
- `source_terms`, if given, is called as `source_terms(u, x, t, equations)` like Trixi's, at every
  volume quadrature point, and added to the right-hand side. A manufactured solution enters this
  way.
- `volume_degree` is the degree of the element rules: 2 by default, 4 to over-integrate.
- `backend` is the KernelAbstractions backend the kernels run on. Only `CPU()` for now, since the
  solver data are host arrays. It uses all of Julia's threads (`julia -t`).
- `grainsize` is the fewest elements or faces a thread takes in one kernel launch. A launch over
  at most `grainsize` of them runs on the calling thread without spawning tasks, and
  `typemax(Int)` runs everything serially. The results are the same either way.

Boundary conditions and source terms are called from many threads at once, so they must not
mutate shared state.
"""
function DGSemidiscretization(mesh::PolygonMesh{T}, equations; surface_flux=flux_lax_friedrichs,
                              boundary_conditions::NamedTuple=(;), source_terms=nothing,
                              volume_degree::Integer=2, backend=CPU(),
                              grainsize::Integer=64) where {T}
    grainsize >= 1 || throw(ArgumentError("grainsize must be at least 1, not $grainsize"))
    names = mesh.boundary_names
    for name in names
        haskey(boundary_conditions, name) ||
            throw(ArgumentError("no boundary condition for boundary :$name; the mesh's " *
                                "boundaries are $names"))
    end
    for name in keys(boundary_conditions)
        name in names || throw(ArgumentError("a boundary condition for :$name, but the mesh " *
                                             "has no such boundary; its boundaries are $names"))
    end
    bcs = Tuple(boundary_conditions[name] for name in names)

    # Rule points and mode values are computed relative to each element's box centre, so they
    # keep their full precision on a sliver far from the origin. Absolute positions are kept only
    # for boundary conditions and exact solutions.
    ne = nelements(mesh)
    basis = Vector{ElementBasis{T}}(undef, ne)
    qptr = Int32[1]
    qpoint, qweight, qphi = SVector{2,T}[], T[], SVector{3,T}[]
    for e in 1:ne
        lo, hi = mesh.elements.lo[e], mesh.elements.hi[e]
        c = (lo + hi) / 2
        x, w = polygon_rule(mesh.points, element_nodes(mesh, e); degree=volume_degree, origin=c)
        b = try
            element_basis(lo, hi, x, w; origin=c)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError("element $e: $(err.msg)"))
        end
        basis[e] = b
        append!(qpoint, (c + xq for xq in x))
        append!(qweight, w)
        append!(qphi, (basis_values(b, xq, c) for xq in x))
        push!(qptr, length(qweight) + 1)
    end

    nf = nfaces(mesh)
    F = mesh.faces
    s, ws = T.(GAUSS2.nodes), T.(GAUSS2.weights)
    fpoint = Vector{NTuple{2,SVector{2,T}}}(undef, nf)
    fnormal = similar(fpoint)
    fphi_left = Vector{NTuple{2,SVector{3,T}}}(undef, nf)
    fphi_right = similar(fphi_left)
    # The modes of `bs` at the face's Gauss points, given the face's start relative to its centre.
    face_values(bs, o, d) = (basis_values(bs, o + s[1] * d, bs.center),
                             basis_values(bs, o + s[2] * d, bs.center))
    for f in 1:nf
        a, b = F.nodes[f]
        pa, d = mesh.points[a], mesh.points[b] - mesh.points[a]
        L, R = F.left[f], F.right[f]
        fpoint[f] = (pa + s[1] * d, pa + s[2] * d)
        fnormal[f] = (ws[1] * F.vector[f], ws[2] * F.vector[f])
        fphi_left[f] = face_values(basis[L], pa - basis[L].center, d)
        fphi_right[f] = R == 0 ? (zero(SVector{3,T}), zero(SVector{3,T})) :
                        face_values(basis[R], (pa - basis[R].center) + F.shift[f], d)
    end

    # Each element's faces in increasing order, the order in which a serial loop over the faces
    # reaches them, signed by side.
    sides = Vector{Int32}(undef, length(mesh.elem_faces))
    for e in 1:ne
        r = Int(mesh.elem_ptr[e]):(Int(mesh.elem_ptr[e + 1]) - 1)
        sides[r] = sort!([F.left[f] == e ? f : -f for f in mesh.elem_faces[r]]; by=abs)
    end
    return DGSemidiscretization(mesh, equations, surface_flux, bcs, source_terms, backend,
                                Int(grainsize), basis, [basis_gradients(b) for b in basis],
                                [b.mass for b in basis], qptr, qpoint, qweight, qphi, fpoint,
                                fnormal, fphi_left, fphi_right, sides,
                                Vector{NTuple{2,SVector{4,T}}}(undef, nf), zeros(T, nf),
                                zeros(T, ne), zeros(Int32, ne))
end

function Base.show(io::IO, s::DGSemidiscretization)
    print(io, "DGSemidiscretization(", nelements(s.mesh), " elements, P1, ", s.surface_flux, ")")
end

# =====================================
# States

"""
    allocate_state(semi) -> Array{T,3}

A zero state: variable x mode x element.
"""
allocate_state(semi::DGSemidiscretization{T}) where {T} = zeros(T, 4, 3, nelements(semi.mesh))

@inline _mode(u, i, e) = @inbounds SVector(u[1, i, e], u[2, i, e], u[3, i, e], u[4, i, e])
@inline _value(u, e, φ) = φ[1] * _mode(u, 1, e) + φ[2] * _mode(u, 2, e) + φ[3] * _mode(u, 3, e)
@inline function _add!(du, i, e, v)
    @inbounds for k in 1:4
        du[k, i, e] += v[k]
    end
    return nothing
end

"""
    cell_average(u, e) -> SVector{4}

The average of the state over element `e`: its mode-1 coefficients.
"""
@inline cell_average(u, e) = _mode(u, 1, e)

"""
    evaluate(semi, u, e, x) -> SVector{4}
    evaluate(semi, u, e, x, origin) -> SVector{4}

The state of element `e` at the point `x`, or at the point `origin + x`.
"""
@inline evaluate(semi::DGSemidiscretization, u, e, x) =
    _value(u, e, basis_values(semi.basis[e], x))
@inline evaluate(semi::DGSemidiscretization, u, e, x, origin) =
    _value(u, e, basis_values(semi.basis[e], x, origin))

"""
    face_trace(semi, u, f, q) -> SVector{4}

The state of face `f`'s left element at the face's Gauss point `q` (1 or 2): the interior trace
on a boundary face. The point is `semi.fpoint[f][q]`, and its weight times the face vector is
`semi.fnormal[f][q]`.
"""
@inline face_trace(semi::DGSemidiscretization, u, f, q) =
    _value(u, semi.mesh.faces.left[f], semi.fphi_left[f][q])

"""
    project(f, semi, t=0; degree=4) -> Array{T,3}

The L2 projection of `f(x, t, equations)` onto the P1 modes, with element rules of `degree`.
Since the mass matrix is diagonal, each coefficient is one weighted sum.
"""
function project(f, semi::DGSemidiscretization{T}, t=zero(T); degree::Integer=4) where {T}
    u = allocate_state(semi)
    mesh = semi.mesh
    for e in 1:nelements(mesh)
        b = semi.basis[e]
        x, w = polygon_rule(mesh.points, element_nodes(mesh, e); degree, origin=b.center)
        acc = zero(SMatrix{4,3,T,12})
        for (xq, wq) in zip(x, w)
            acc += (wq * f(b.center + xq, t, semi.equations)) * basis_values(b, xq, b.center)'
        end
        u[:, :, e] .= acc / b.mass
    end
    return u
end

# =====================================
# Kernel launches

# Run `kernel` over the items 1:n, on the semidiscretization's backend. The CPU backend hands each
# thread a contiguous run of whole workgroups, 1024 items each by default, which would leave most
# threads idle on a mesh of a few thousand elements. So the workgroups split the items evenly over
# the threads, but hold at least `grainsize` items: a launch over at most `grainsize` items runs on
# the calling thread. The size is capped at `n`, since the backend steps through every slot of a
# workgroup.
@inline function _launch!(kernel::K, semi::DGSemidiscretization, n,
                          args::Vararg{Any,N}) where {K,N}
    n > 0 || return nothing
    ws = min(n, max(semi.grainsize, cld(n, Threads.nthreads())))
    kernel(semi.backend)(args...; ndrange=n, workgroupsize=ws)
    return nothing
end

# =====================================
# The right-hand side
#
# Two kernels, so that no two threads ever add into the same element. The face kernel evaluates F̂
# once at each face Gauss point, into `semi.fhat`. The element kernel then sums each element's
# volume integral, source terms and face integrals, taking its faces in increasing order: the order
# of a serial loop over the faces, which scattered into both neighbours.

"""
    rhs!(du, u, semi, t) -> du

The DG time derivative of the state `u` at time `t`: the flux at every face Gauss point, then
each element's volume integrals, source terms and face integrals, divided by its mass. Threaded
over faces and then elements, with the same result on any number of threads. The kernels are
allocation-free; each launch allocates KernelAbstractions' bookkeeping, about 150 bytes, plus its
tasks when threaded.
"""
function rhs!(du::AbstractArray{T,3}, u::AbstractArray{T,3}, semi::DGSemidiscretization{T},
              t) where {T}
    _launch!(_face_flux_kernel!, semi, nfaces(semi.mesh), semi, u, t)
    _launch!(_element_rhs_kernel!, semi, nelements(semi.mesh), du, u, semi, t)
    return du
end

@kernel function _face_flux_kernel!(semi, u, t)
    f = @index(Global, Linear)
    @inbounds semi.fhat[f] = _face_flux(u, semi, t, f)
end

@kernel function _element_rhs_kernel!(du, u, semi, t)
    e = @index(Global, Linear)
    _element_rhs!(du, u, semi, t, e)
end

# F̂ at face `f`'s two Gauss points: the surface flux between the two traces on an interior face,
# and the boundary's condition on a boundary face.
@inline function _face_flux(u, semi::DGSemidiscretization, t, f)
    eq, sf, F = semi.equations, semi.surface_flux, semi.mesh.faces
    @inbounds begin
        L, k = F.left[f], F.boundary[f]
        φL, n = semi.fphi_left[f], semi.fnormal[f]
        if k == 0
            R, φR = F.right[f], semi.fphi_right[f]
            return (sf(_value(u, L, φL[1]), _value(u, R, φR[1]), n[1], eq),
                    sf(_value(u, L, φL[2]), _value(u, R, φR[2]), n[2], eq))
        else
            x, bcs = semi.fpoint[f], semi.boundary_conditions
            return (_boundary_flux(bcs, k, _value(u, L, φL[1]), n[1], x[1], t, sf, eq),
                    _boundary_flux(bcs, k, _value(u, L, φL[2]), n[2], x[2], t, sf, eq))
        end
    end
end

# Boundary `k`'s condition, picked from the tuple by unrolled recursion, so the call is type-stable
# whatever the mix of conditions.
@inline _boundary_flux(::Tuple{}, k, u, n, x, t, sf, eq) = _no_boundary()
@inline function _boundary_flux(bcs::Tuple, k, u, n, x, t, sf, eq)
    k == 1 && return first(bcs)(u, n, x, t, sf, eq)
    return _boundary_flux(Base.tail(bcs), k - 1, u, n, x, t, sf, eq)
end

@noinline _no_boundary() = throw(ArgumentError("no boundary condition for a face's boundary"))

# Element `e`'s rows of the right-hand side, into `du`:
#
#     ( ∫ ∇φ_i · F(U) + ∫ φ_i S - ∮ φ_i F̂ ) / |K|
#
# The gradients are constant, so the weighted x and y fluxes are summed first; mode 1 has no
# gradient. The face fluxes come from `semi.fhat`.
@inline function _element_rhs!(du, u, semi::DGSemidiscretization{T}, t, e) where {T}
    eq = semi.equations
    @inbounds begin
        Fx = Fy = zero(SVector{4,T})
        for q in semi.qptr[e]:(semi.qptr[e + 1] - 1)
            U = _value(u, e, semi.qphi[q])
            w = semi.qweight[q]
            Fx += w * flux(U, 1, eq)
            Fy += w * flux(U, 2, eq)
        end
        g = semi.gradients[e]
        r = zero(SMatrix{4,3,T,12}) +
            hcat(zero(SVector{4,T}), g[1, 2] * Fx + g[2, 2] * Fy, g[1, 3] * Fx + g[2, 3] * Fy)
        r = _element_sources(r, u, semi, t, e, semi.source_terms)
        for j in semi.mesh.elem_ptr[e]:(semi.mesh.elem_ptr[e + 1] - 1)
            f = semi.sides[j]
            if f > 0
                φ, Fhat = semi.fphi_left[f], semi.fhat[f]
                r = r - Fhat[1] * φ[1]' - Fhat[2] * φ[2]'
            else
                φ, Fhat = semi.fphi_right[-f], semi.fhat[-f]
                r = r + Fhat[1] * φ[1]' + Fhat[2] * φ[2]'
            end
        end
        m = inv(semi.mass[e])
        for i in 1:3, k in 1:4
            du[k, i, e] = r[k, i] * m
        end
    end
    return nothing
end

# + ∫ φ_i S(U, x, t): the source terms, at the volume quadrature points; none by default.
@inline _element_sources(r, u, semi, t, e, ::Nothing) = r
@inline function _element_sources(r, u, semi, t, e, source)
    eq = semi.equations
    @inbounds for q in semi.qptr[e]:(semi.qptr[e + 1] - 1)
        φ = semi.qphi[q]
        S = semi.qweight[q] * source(_value(u, e, φ), semi.qpoint[q], t, eq)
        r += S * φ'
    end
    return r
end

# =====================================
# Time step and positivity

# Each element's `Σ_f λ_f |f|`, into `semi.speed`, with `λ_f` the largest `|v · n| + c` over the
# face's Gauss points and both its sides. Boundary faces count too: a thin sliver along a long wall
# face gets a step that scales with its thickness. With a vector `dt`, also each element's step at
# `cfl`.
function _face_speeds!(dt, u, semi::DGSemidiscretization; cfl)
    _launch!(_face_speed_kernel!, semi, nfaces(semi.mesh), semi, u)
    _launch!(_element_speed_kernel!, semi, nelements(semi.mesh), dt, semi, cfl)
    return semi.speed
end

@kernel function _face_speed_kernel!(semi, u)
    f = @index(Global, Linear)
    @inbounds semi.fspeed[f] = _face_speed(u, semi, f)
end

@kernel function _element_speed_kernel!(dt, semi, cfl)
    e = @index(Global, Linear)
    _element_speed!(dt, semi, cfl, e)
end

# `λ_f |f|` for face `f`.
@inline function _face_speed(u, semi::DGSemidiscretization{T}, f) where {T}
    F, eq = semi.mesh.faces, semi.equations
    L, R = @inbounds F.left[f], F.right[f]
    λ = zero(T)
    @inbounds for q in 1:2
        UL = _value(u, L, semi.fphi_left[f][q])
        UR = R == 0 ? UL : _value(u, R, semi.fphi_right[f][q])
        λ = max(λ, max_abs_speed(UL, UR, F.normal[f], eq))
    end
    return @inbounds λ * F.length[f]
end

@inline function _element_speed!(dt, semi::DGSemidiscretization{T}, cfl, e) where {T}
    s = zero(T)
    @inbounds for j in semi.mesh.elem_ptr[e]:(semi.mesh.elem_ptr[e + 1] - 1)
        s += semi.fspeed[abs(semi.sides[j])]
    end
    @inbounds semi.speed[e] = s
    if dt !== nothing
        @inbounds dt[e] = cfl * semi.mass[e] / (3 * s)
    end
    return nothing
end

"""
    local_dt!(dt, u, semi; cfl) -> dt

Each element's own step, the plan's local-step formula with p = 1:

    Δt_K = cfl |K| / ((2p + 1) Σ_f λ_f |f|),   λ_f = max over the face's Gauss points and both
                                               sides of |v · n| + c

The face sum includes boundary faces. Threaded over faces and then elements.
"""
function local_dt!(dt::AbstractVector, u, semi::DGSemidiscretization; cfl)
    _face_speeds!(dt, u, semi; cfl)
    return dt
end

"""
    max_dt(u, semi; cfl) -> T

The global time step: the smallest of the elements' [`local_dt!`](@ref) steps.
"""
function max_dt(u, semi::DGSemidiscretization{T}; cfl) where {T}
    speed = _face_speeds!(nothing, u, semi; cfl)
    dt = typemax(T)
    @inbounds for e in eachindex(speed)
        dt = min(dt, semi.mass[e] / (3 * speed[e]))
    end
    return cfl * dt
end

"""
    assert_positive(u, semi)

Throw a `DomainError` naming the element and the point if the density or the pressure is not
positive at any volume quadrature point. NaNs fail too. Threaded over elements; the error names
the first such element, and its first such point.
"""
function assert_positive(u, semi::DGSemidiscretization)
    _launch!(_positivity_kernel!, semi, nelements(semi.mesh), semi, u)
    e = findfirst(!iszero, semi.positivity)
    e === nothing && return nothing
    q = semi.positivity[e]
    ρ, _, _, p = cons2prim(_value(u, e, semi.qphi[q]), semi.equations)
    return _positivity_error(e, semi.qpoint[q], ρ, p)
end

@kernel function _positivity_kernel!(semi, u)
    e = @index(Global, Linear)
    @inbounds semi.positivity[e] = _first_nonpositive(u, semi, e)
end

# The first volume quadrature point of element `e` where the density or the pressure is not
# positive, or 0.
@inline function _first_nonpositive(u, semi::DGSemidiscretization, e)
    eq = semi.equations
    @inbounds for q in semi.qptr[e]:(semi.qptr[e + 1] - 1)
        ρ, _, _, p = cons2prim(_value(u, e, semi.qphi[q]), eq)
        ρ > 0 && p > 0 || return Int32(q)
    end
    return Int32(0)
end

@noinline _positivity_error(e, x, ρ, p) =
    throw(DomainError((e, ρ, p), "element $e: density $ρ and pressure $p at $(Tuple(x))"))

end # module Semidiscretization
