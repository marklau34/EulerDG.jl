# =====================================
# Post: forces, accuracy measures and output
#
# Downstream of the mesh contract. Wall quantities are taken at the wall faces' Gauss points from
# the interior trace, with each point's weight times the face vector -- the same face vectors the
# closure check covers, so a uniform pressure gives zero force to round-off. Errors are integrated
# with the degree-4 element rules, finer than the operator's.

"""
    EulerDG.Post

Postprocessing.

- Phase 1: [`l2_error`](@ref) against an exact solution, and [`write_solution_vtk`](@ref), the
  cell averages on the element polygons.
- Phase 4: [`wall_forces`](@ref) (lift, drag and moment), [`surface_cp`](@ref),
  [`entropy_error`](@ref), and [`RunLog`](@ref), the CSV run log of residual and force histories.
- Phase 5: each element's P1 values at its vertices in the field output.
"""
module Post

using Printf: @printf
using StaticArrays: SVector
using WriteVTK: MeshCell, PolyData, VTKCellData, vtk_grid, vtk_save
using ..EulerDG: nelements, npoints, element_nodes, boundary_faces
using ..Quadrature: polygon_rule
using ..Semidiscretization: DGSemidiscretization, cell_average, evaluate, face_trace
using ..Equations: cons2prim, pressure

export l2_error, entropy_error, wall_forces, surface_cp, RunLog, write_solution_vtk

"""
    l2_error(semi, u, exact, t=0; degree=4) -> (; l2, linf)

The error of `u` against `exact(x, t, equations)`, per conservative variable: `l2`, the L2 norm
over the domain, and `linf`, the largest pointwise error at the quadrature points. Integrated with
element rules of `degree`.
"""
function l2_error(semi::DGSemidiscretization{T}, u, exact, t=zero(T); degree::Integer=4) where {T}
    mesh = semi.mesh
    sq = linf = zero(SVector{4,T})
    for e in 1:nelements(mesh)
        c = semi.basis[e].center
        x, w = polygon_rule(mesh.points, element_nodes(mesh, e); degree, origin=c)
        for (xq, wq) in zip(x, w)
            d = evaluate(semi, u, e, xq, c) - exact(c + xq, t, semi.equations)
            sq += wq * d .^ 2
            linf = max.(linf, abs.(d))
        end
    end
    return (; l2=sqrt.(sq), linf)
end

# Freestream reference quantities: pressure, dynamic pressure, entropy function, and the drag and
# lift directions.
function _reference(uinf, eq)
    ρ, v1, v2, p = cons2prim(uinf, eq)
    speed = hypot(v1, v2)
    d = SVector(v1, v2) / speed
    return (; p, q=ρ * speed^2 / 2, s=p / ρ^eq.gamma, drag=d, lift=SVector(-d[2], d[1]))
end

"""
    entropy_error(semi, u, u_infinity; degree=4) -> (; l2, max)

The entropy error `(s - s∞)/s∞`, with `s = p/ρ^γ` and `s∞` the freestream's: its L2 norm over the
domain, and its largest magnitude at the quadrature points. It is zero in the continuous problem,
so it measures the discretization alone; the plan's primary accuracy metric.
"""
function entropy_error(semi::DGSemidiscretization{T}, u, uinf; degree::Integer=4) where {T}
    mesh, eq = semi.mesh, semi.equations
    s∞ = _reference(uinf, eq).s
    sq, big = zero(T), zero(T)
    for e in 1:nelements(mesh)
        c = semi.basis[e].center
        x, w = polygon_rule(mesh.points, element_nodes(mesh, e); degree, origin=c)
        for (xq, wq) in zip(x, w)
            ρ, _, _, p = cons2prim(evaluate(semi, u, e, xq, c), eq)
            δ = (p / ρ^eq.gamma - s∞) / s∞
            sq += wq * δ^2
            big = max(big, abs(δ))
        end
    end
    return (; l2=sqrt(sq), max=big)
end

"""
    wall_forces(semi, u, u_infinity; boundary=:wall, length=1, point=(0, 0))
        -> (; force, moment, cd, cl, cm)

The pressure force on the body, `∮ (p - p∞) n ds` with `n` into the body, from the interior trace
at the wall faces' Gauss points, and its moment about `point`, counter-clockwise positive. The
coefficients are scaled by the freestream's dynamic pressure and the reference `length` (squared
for the moment). Drag is along the freestream and lift a quarter-turn counter-clockwise from it.
Subtracting `p∞` does not change the force on a closed body; it keeps round-off down.
"""
function wall_forces(semi::DGSemidiscretization{T}, u, uinf; boundary::Symbol=:wall, length=1,
                     point=(0, 0)) where {T}
    eq = semi.equations
    ref = _reference(uinf, eq)
    x0 = SVector{2,T}(point)
    F, M = zero(SVector{2,T}), zero(T)
    for f in boundary_faces(semi.mesh, boundary), q in 1:2
        dF = (pressure(face_trace(semi, u, f, q), eq) - ref.p) * semi.fnormal[f][q]
        F += dF
        r = semi.fpoint[f][q] - x0
        M += r[1] * dF[2] - r[2] * dF[1]
    end
    return (; force=F, moment=M, cd=F' * ref.drag / (ref.q * length),
            cl=F' * ref.lift / (ref.q * length), cm=M / (ref.q * length^2))
end

"""
    surface_cp(semi, u, u_infinity; boundary=:wall) -> (; x, cp)

The pressure coefficient `(p - p∞)/q∞` at every Gauss point of the boundary's faces, from the
interior trace, with the points `x`. Cut cells make it scatter from cell to cell; plotting every
point shows that honestly.
"""
function surface_cp(semi::DGSemidiscretization{T}, u, uinf; boundary::Symbol=:wall) where {T}
    eq = semi.equations
    ref = _reference(uinf, eq)
    x, cp = SVector{2,T}[], T[]
    for f in boundary_faces(semi.mesh, boundary), q in 1:2
        push!(x, semi.fpoint[f][q])
        push!(cp, (pressure(face_trace(semi, u, f, q), eq) - ref.p) / ref.q)
    end
    return (; x, cp)
end

"""
    RunLog(path, semi, u_infinity; every=10, length=1, point=(0, 0))

A [`solve_steady!`](@ref EulerDG.PseudoTime.solve_steady!) callback that writes the run log to the
CSV file `path`: one row on the first iteration and every `every` iterations after, with the
iteration, the wall time, the L2 density residual relative to the first, the largest residual and
its element, and -- if the mesh has a `:wall` -- the lift, drag and moment coefficients
(`length` and `point` as in [`wall_forces`](@ref)). Close it with `close`.
"""
mutable struct RunLog{S,V}
    io::IOStream
    semi::S
    uinf::V
    every::Int
    length::Float64
    point::SVector{2,Float64}
    t0::Float64
    r0::Float64
end

function RunLog(path::AbstractString, semi::DGSemidiscretization, uinf; every::Integer=10,
                length=1.0, point=(0.0, 0.0))
    io = open(path, "w")
    println(io, "iteration,walltime,residual,residual_max,residual_element,cl,cd,cm")
    return RunLog(io, semi, uinf, Int(every), Float64(length), SVector{2,Float64}(point), time(),
                  0.0)
end

function (log::RunLog)(u, it, norms)
    it == 1 && (log.r0 = norms.l2)
    (it == 1 || it % log.every == 0) || return nothing
    f = :wall in log.semi.mesh.boundary_names ?
        wall_forces(log.semi, u, log.uinf; length=log.length, point=log.point) :
        (; cl=NaN, cd=NaN, cm=NaN)
    @printf(log.io, "%d,%.6f,%.12e,%.12e,%d,%.12e,%.12e,%.12e\n", it, time() - log.t0,
            log.r0 > 0 ? norms.l2 / log.r0 : 0.0, norms.max, norms.element, f.cl, f.cd, f.cm)
    flush(log.io)
    return nothing
end

Base.close(log::RunLog) = close(log.io)

"""
    write_solution_vtk(prefix, semi, u; freestream=nothing) -> Vector{String}

Write `prefix.vtp`: one polygon per element, with the cell averages of the density, velocity,
pressure, Mach number and entropy function `p / ρ^γ` as cell data, and -- given the `freestream`
state -- the pressure coefficient `cp` and the entropy error `entropy_error`. Returns the file
names.
"""
function write_solution_vtk(prefix::AbstractString, semi::DGSemidiscretization, u;
                            freestream=nothing)
    mesh, eq = semi.mesh, semi.equations
    ne = nelements(mesh)
    pts = zeros(Float64, 3, npoints(mesh))
    for (i, p) in enumerate(mesh.points)
        pts[1, i], pts[2, i] = p[1], p[2]
    end
    polys = [MeshCell(PolyData.Polys(), collect(Int, element_nodes(mesh, e))) for e in 1:ne]
    prim = [cons2prim(cell_average(u, e), eq) for e in 1:ne]
    velocity = zeros(Float64, 3, ne)
    for (e, p) in enumerate(prim)
        velocity[1, e], velocity[2, e] = p[2], p[3]
    end
    vtk = vtk_grid(prefix, pts, polys)
    vtk["density", VTKCellData()] = Float64[p[1] for p in prim]
    vtk["velocity", VTKCellData()] = velocity
    vtk["pressure", VTKCellData()] = Float64[p[4] for p in prim]
    vtk["mach", VTKCellData()] = Float64[hypot(p[2], p[3]) / sqrt(eq.gamma * p[4] / p[1])
                                         for p in prim]
    vtk["entropy", VTKCellData()] = Float64[p[4] / p[1]^eq.gamma for p in prim]
    if freestream !== nothing
        ref = _reference(freestream, eq)
        vtk["cp", VTKCellData()] = Float64[(p[4] - ref.p) / ref.q for p in prim]
        vtk["entropy_error", VTKCellData()] = Float64[(p[4] / p[1]^eq.gamma - ref.s) / ref.s
                                                      for p in prim]
    end
    return vtk_save(vtk)
end

end # module Post
