# =====================================
# Cut meshes: a circle and NACA 0012, for ParaView
#
# Run with `julia --project=examples examples/cut.jl [n]`. Cuts an n x n grid (default 64) against
# a circle of radius 0.5 and against NACA 0012, prints each preprocessor report, and writes the
# meshes to out/cut/ with `write_mesh_vtk`: the elements as polygons, with their volume fractions
# and closure errors, and the faces as lines, with their boundary names.

using EulerDG
using CartesianMeshes: CartesianGrid

n = isempty(ARGS) ? 64 : parse(Int, ARGS[1])
outdir = mkpath(normpath(joinpath(@__DIR__, "..", "out", "cut")))

bodies = (; circle=((-1.0, -1.0), (1.0, 1.0), h -> circle_points(; radius=0.5, h)),
          naca0012=((-0.5, -0.6), (1.5, 0.6), h -> naca4_points("0012"; h)))
for (name, (lo, hi, points)) in pairs(bodies)
    grid = CartesianGrid(Float64; mincorner=lo, maxcorner=hi, n=(n, n))
    # A sub-cell shift, so no body vertex sits on a grid line.
    shift = grid_shifts(2)[2] .* grid.d
    c = cut_mesh(grid, polygon_body(points(minimum(grid.d)) .+ Ref(shift); name=string(name)))
    println("== $name, $n x $n")
    cut_report(c)
    write_mesh_vtk(joinpath(outdir, "$(name)_n$n"), c.mesh)
end
println("Meshes written to ", outdir)
