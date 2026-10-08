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
  equations, the surface flux and one boundary condition per named boundary.
- [`rhs!`](@ref) in four passes: element volume integrals, interior faces, boundary faces (one
  type-stable loop per boundary), then division by |K|.
- [`allocate_state`](@ref), [`project`](@ref), [`evaluate`](@ref), [`cell_average`](@ref).
- [`max_dt`](@ref), the time step from the plan's local-step formula, and
  [`assert_positive`](@ref).
"""
module Semidiscretization

using StaticArrays: SMatrix, SVector
using ..EulerDG: PolygonMesh, nelements, nfaces, element_nodes, interior_faces, boundary_faces
using ..Quadrature: GAUSS2, polygon_rule
using ..Basis: ElementBasis, element_basis, basis_values, basis_gradients
using ..Equations: cons2prim, flux, flux_lax_friedrichs, max_abs_speed

export DGSemidiscretization, allocate_state, project, evaluate, cell_average, rhs!, max_dt
export assert_positive

"""
    DGSemidiscretization{T,M,EQ,SF,BC}

The P1 DG discretization of `equations` on a `PolygonMesh`, as struct-of-arrays:

- per element: its basis, the gradients of its modes, and its mass (the mass matrix is `mass[e]`
  times the identity);
- per volume quadrature point, in CSR order by element (`qptr`): position, weight and mode values;
- per face, at its two Gauss points on the left copy: position, weight times face vector, and the
  mode values of the left and right elements (the right ones at the point moved by the face's
  periodic shift).

Build it with `DGSemidiscretization(mesh, equations; ...)`.
"""
struct DGSemidiscretization{T,M<:PolygonMesh{T},EQ,SF,BC}
    mesh::M
    equations::EQ
    surface_flux::SF
    boundary_conditions::BC
    basis::Vector{ElementBasis{T}}
    gradients::Vector{SMatrix{2,3,T,6}}
    mass::Vector{T}
    qptr::Vector{Int32}
    qpoint::Vector{SVector{2,T}}
    qweight::Vector{T}
    qphi::Vector{SVector{3,T}}
    fpoint::Vector{NTuple{2,SVector{2,T}}}
    fnormal::Vector{NTuple{2,SVector{2,T}}}
    fphi_left::Vector{NTuple{2,SVector{3,T}}}
    fphi_right::Vector{NTuple{2,SVector{3,T}}}
    speed::Vector{T}
end

"""
    DGSemidiscretization(mesh, equations; surface_flux=flux_lax_friedrichs,
                         boundary_conditions=(;), volume_degree=2)

Build the discretization. `boundary_conditions` is a NamedTuple with one condition per name in
`mesh.boundary_names`, each called as `bc(u_inner, normal_direction, x, t, surface_flux,
equations)` like Trixi's. `volume_degree` is the degree of the element rules: 2 by default, 4 to
over-integrate.
"""
function DGSemidiscretization(mesh::PolygonMesh{T}, equations; surface_flux=flux_lax_friedrichs,
                              boundary_conditions::NamedTuple=(;),
                              volume_degree::Integer=2) where {T}
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
    return DGSemidiscretization(mesh, equations, surface_flux, bcs, basis,
                                [basis_gradients(b) for b in basis], [b.mass for b in basis],
                                qptr, qpoint, qweight, qphi, fpoint, fnormal, fphi_left,
                                fphi_right, zeros(T, ne))
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
# The right-hand side

"""
    rhs!(du, u, semi, t) -> du

The DG time derivative of the state `u` at time `t`: element volume integrals, then interior
faces, then boundary faces, then division by each element's mass. Allocation-free.
"""
function rhs!(du::AbstractArray{T,3}, u::AbstractArray{T,3}, semi::DGSemidiscretization{T},
              t) where {T}
    fill!(du, zero(T))
    _volume_integrals!(du, u, semi)
    _interior_faces!(du, u, semi)
    _boundary_passes!(du, u, semi, t, semi.boundary_conditions, 1)
    _inverse_mass!(du, semi)
    return du
end

# ∫ ∇φ_i · F(U): the gradients are constant, so sum the weighted x and y fluxes first. Mode 1 has
# no gradient.
function _volume_integrals!(du, u, semi::DGSemidiscretization{T}) where {T}
    eq = semi.equations
    @inbounds for e in eachindex(semi.mass)
        Fx = Fy = zero(SVector{4,T})
        for q in semi.qptr[e]:(semi.qptr[e + 1] - 1)
            U = _value(u, e, semi.qphi[q])
            w = semi.qweight[q]
            Fx += w * flux(U, 1, eq)
            Fy += w * flux(U, 2, eq)
        end
        g = semi.gradients[e]
        _add!(du, 2, e, g[1, 2] * Fx + g[2, 2] * Fy)
        _add!(du, 3, e, g[1, 3] * Fx + g[2, 3] * Fy)
    end
    return nothing
end

# -∮ φ_i F̂ on interior faces: each face flux is computed once and scattered to both sides.
function _interior_faces!(du, u, semi::DGSemidiscretization)
    eq, F = semi.equations, semi.mesh.faces
    @inbounds for f in interior_faces(semi.mesh)
        L, R = F.left[f], F.right[f]
        φL, φR, n = semi.fphi_left[f], semi.fphi_right[f], semi.fnormal[f]
        for q in 1:2
            Fhat = semi.surface_flux(_value(u, L, φL[q]), _value(u, R, φR[q]), n[q], eq)
            for i in 1:3
                _add!(du, i, L, -φL[q][i] * Fhat)
                _add!(du, i, R, φR[q][i] * Fhat)
            end
        end
    end
    return nothing
end

# One boundary at a time, so each loop is type-stable for its own condition.
@inline _boundary_passes!(du, u, semi, t, ::Tuple{}, k) = nothing
@inline function _boundary_passes!(du, u, semi, t, bcs::Tuple, k)
    _boundary_pass!(du, u, semi, t, first(bcs), boundary_faces(semi.mesh, k))
    return _boundary_passes!(du, u, semi, t, Base.tail(bcs), k + 1)
end

function _boundary_pass!(du, u, semi::DGSemidiscretization, t, bc, faces)
    eq, F = semi.equations, semi.mesh.faces
    @inbounds for f in faces
        L = F.left[f]
        φL, n, x = semi.fphi_left[f], semi.fnormal[f], semi.fpoint[f]
        for q in 1:2
            Fhat = bc(_value(u, L, φL[q]), n[q], x[q], t, semi.surface_flux, eq)
            for i in 1:3
                _add!(du, i, L, -φL[q][i] * Fhat)
            end
        end
    end
    return nothing
end

function _inverse_mass!(du, semi::DGSemidiscretization)
    @inbounds for e in eachindex(semi.mass)
        m = inv(semi.mass[e])
        for i in 1:3, k in 1:4
            du[k, i, e] *= m
        end
    end
    return nothing
end

# =====================================
# Time step and positivity

"""
    max_dt(u, semi; cfl) -> T

The global time step: the smallest over the elements of the plan's local step

    Δt_K = cfl |K| / ((2p + 1) Σ_f λ_f |f|),   λ_f = max over the face's Gauss points and both
                                               sides of |v · n| + c

with p = 1. The face sum includes boundary faces.
"""
function max_dt(u, semi::DGSemidiscretization{T}; cfl) where {T}
    speed, F, eq = semi.speed, semi.mesh.faces, semi.equations
    fill!(speed, zero(T))
    @inbounds for f in eachindex(semi.fnormal)
        L, R = F.left[f], F.right[f]
        λ = zero(T)
        for q in 1:2
            UL = _value(u, L, semi.fphi_left[f][q])
            UR = R == 0 ? UL : _value(u, R, semi.fphi_right[f][q])
            λ = max(λ, max_abs_speed(UL, UR, F.normal[f], eq))
        end
        speed[L] += λ * F.length[f]
        R == 0 || (speed[R] += λ * F.length[f])
    end
    dt = typemax(T)
    @inbounds for e in eachindex(speed)
        dt = min(dt, semi.mass[e] / (3 * speed[e]))
    end
    return cfl * dt
end

"""
    assert_positive(u, semi)

Throw a `DomainError` naming the element and the point if the density or the pressure is not
positive at any volume quadrature point. NaNs fail too.
"""
function assert_positive(u, semi::DGSemidiscretization)
    eq = semi.equations
    @inbounds for e in eachindex(semi.mass), q in semi.qptr[e]:(semi.qptr[e + 1] - 1)
        ρ, _, _, p = cons2prim(_value(u, e, semi.qphi[q]), eq)
        ρ > 0 && p > 0 || _positivity_error(e, semi.qpoint[q], ρ, p)
    end
    return nothing
end

@noinline _positivity_error(e, x, ρ, p) =
    throw(DomainError((e, ρ, p), "element $e: density $ρ and pressure $p at $(Tuple(x))"))

end # module Semidiscretization
