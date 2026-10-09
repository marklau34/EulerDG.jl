# =====================================
# NACA 0012: the cut mesh round the airfoil, and its VTK output
#
# Run with `julia --project=examples examples/naca0012.jl [cells_per_chord] [alpha_deg]`, for
# example `julia --project=examples examples/naca0012.jl 64 2`. The script:
#
# 1. builds NACA 0012 with a sharp trailing edge, as a polygon with 4 segments per cell, turned
#    nose-up by alpha_deg about its leading edge;
# 2. cuts a uniform grid round it, placed so the trailing edge sits 0.3 h into its cell and the
#    chord line runs along the middle of a row, and prints the preprocessor report and the cells
#    round the trailing edge;
# 3. writes the mesh to out/naca0012/: `*_elements.vtp`, one polygon per element with its volume
#    fraction, parent cell, fragment number and closure error, and `*_faces.vtp`, one line per face
#    with its boundary index (1 is the wall) and normal;
# 4. builds the P1 DG discretization with the freestream at Mach 0.3, checks that uniform flow is a
#    steady state of the discrete operator on the cut mesh, and writes that state to
#    `*_solution.vtp`.
#
# This is not yet a flow solution: the wall needs Phase 3's slip wall and the outer boundary
# Phase 4's farfield, so every boundary gets the freestream here. What it shows is the mesh, and the
# discrete operator's consistency on it.
#
# In ParaView: open the three .vtp files; colour the elements by `volume_fraction` on a log scale;
# keep only the faces with `boundary` = 1 (Threshold) to see the wall; zoom on the trailing edge.

using EulerDG
using CartesianMeshes: CartesianGrid
using Printf: @printf

cpc = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64           # cells per chord
alpha = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.0     # degrees, nose up
h = 1 / cpc
outdir = mkpath(normpath(joinpath(@__DIR__, "..", "out", "naca0012")))
# No dot in the file name, or WriteVTK takes the rest for an extension: 2.5 degrees is "a2p5".
prefix = joinpath(outdir, "naca0012_c$(cpc)_a" *
                          (isinteger(alpha) ? string(Int(alpha)) : replace(string(alpha), "." => "p")))

# 1. The body.
points = naca4_points("0012"; h, alpha_deg=alpha)
body = polygon_body(points; name="naca0012")
te = points[1]                                               # the trailing edge

# 2. The grid: from 1.5 chords ahead of the leading edge to 2.5 behind it, and 1.5 above and
#    below, with the trailing edge 0.3 h into its cell and on the middle line of its row.
mx, my = round(Int, (te[1] + 1.5) / h), round(Int, (te[2] + 1.5) / h)
x0, y0 = te[1] - 0.3h - mx * h, te[2] - 0.5h - my * h
nx, ny = 4cpc, 3cpc
grid = CartesianGrid(Float64; mincorner=(x0, y0), maxcorner=(x0 + nx * h, y0 + ny * h),
                     n=(nx, ny))
cut = cut_mesh(grid, body)
mesh = cut.mesh
println("NACA 0012 at $alpha degrees, $cpc cells per chord, on a $nx x $ny grid")
cut_report(cut)

# The 3 x 3 cells round the trailing edge, top row first: their fragments. A split cell has one
# fragment above the trailing edge and one below, with no face between them.
tecell = CartesianIndex(mx + 1, my + 1)
lin = LinearIndices(grid)
wall = boundary_index(mesh, :wall)
println("Cells round the trailing edge, cell $(Tuple(tecell)) in the middle:")
for dj in 1:-1:-1, di in -1:1
    ci = tecell + CartesianIndex(di, dj)
    elements = findall(==(lin[ci]), mesh.elements.parent)
    isempty(elements) && (@printf("  cell %-12s solid\n", string(Tuple(ci))); continue)
    for e in elements
        nwall = count(f -> mesh.faces.boundary[f] == wall, element_faces(mesh, e))
        @printf("  cell %-12s element %6d  fragment %d  volume fraction %.4f  %2d vertices  %d wall faces\n",
                string(Tuple(ci)), e, mesh.elements.fragment[e], mesh.elements.volume_fraction[e],
                length(element_nodes(mesh, e)), nwall)
    end
end

# 3. The mesh, for ParaView.
files = write_mesh_vtk(prefix, mesh)

# 4. The discretization, and the freestream as its steady state.
eq = CompressibleEulerEquations2D(1.4)
uinf = freestream(eq; mach=0.3)
bc = BoundaryConditionDirichlet((x, t, eq) -> uinf)
semi = DGSemidiscretization(mesh, eq;
                            boundary_conditions=NamedTuple{Tuple(mesh.boundary_names)}(ntuple(_ -> bc,
                                                                                              nboundaries(mesh))))
u = project((x, t, eq) -> uinf, semi)
du = rhs!(similar(u), u, semi, 0.0)
residual = maximum(1:nelements(mesh)) do e
    perimeter = sum(mesh.faces.length[f] for f in element_faces(mesh, e))
    maximum(abs, du[:, :, e]) * semi.mass[e] / perimeter
end
@printf("Freestream residual |K| du, relative to each element's perimeter: at most %.1e\n",
        residual)
# A full cell's step: |K| = h^2, and λ = |u| + c on its two vertical faces and c on the others.
full = 2 * h^2 / (3 * h * (2 * (0.3 + 1) + 2 * 1))
@printf("Global time step at CFL 2: %.2e, against %.2e for a full cell: the smallest fragments set it\n",
        max_dt(u, semi; cfl=2), full)
append!(files, write_solution_vtk(prefix * "_solution", semi, u))

println("Written:")
foreach(f -> println("  ", f), files)
