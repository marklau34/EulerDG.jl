# =====================================
# Post: accuracy measures and output
#
# Downstream of the mesh contract. Errors are integrated with the degree-4 element rules, finer
# than the operator's, so they measure the discretization rather than the quadrature.

"""
    EulerDG.Post

Postprocessing.

- Phase 1: [`l2_error`](@ref) against an exact solution, and [`write_solution_vtk`](@ref), the
  cell averages on the element polygons.
- Phase 4: lift, drag and moment from the wall faces; entropy error and spurious drag; the CSV
  run log of residual and force history.
- Phase 5: surface Cp at the wall quadrature points, sorted by x/c; each element's P1 values at
  its vertices in the field output.
"""
module Post

using StaticArrays: SVector
using WriteVTK: MeshCell, PolyData, VTKCellData, vtk_grid, vtk_save
using ..EulerDG: nelements, npoints, element_nodes
using ..Quadrature: polygon_rule
using ..Semidiscretization: DGSemidiscretization, cell_average, evaluate
using ..Equations: cons2prim

export l2_error, write_solution_vtk

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

"""
    write_solution_vtk(prefix, semi, u) -> Vector{String}

Write `prefix.vtp`: one polygon per element, with the cell averages of the density, velocity,
pressure, Mach number and entropy function `p / ρ^γ` as cell data. Returns the file names.
"""
function write_solution_vtk(prefix::AbstractString, semi::DGSemidiscretization, u)
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
    return vtk_save(vtk)
end

end # module Post
