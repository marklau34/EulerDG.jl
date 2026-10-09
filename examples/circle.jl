# =====================================
# The inviscid circle at Mach 0.3, at low resolution
#
# Run with `julia --project=examples -t auto examples/circle.jl [cells_per_diameter]` (default 8);
# `-t` sets the threads the solver's kernels run on, and the results are the same on any. Cuts a
# uniform grid over [-3, 3]^2 round a circle of diameter 1 centred on a grid line, marches the P1
# DG discretization to a steady state with local steps -- HLLC, the mirror-state slip wall and the
# characteristic farfield -- and writes to out/circle/:
#
# - `*_run.csv`, the run log: residual and force histories;
# - `*_cp.csv`, the surface pressure coefficient at every wall Gauss point, with its angle;
# - `*_solution.vtp`, the cell averages with Cp and the entropy error, for ParaView.
#
# The circle has no lift and, in the continuous problem, no drag and no entropy change, so all
# three measure the discretization. At this resolution expect a visible entropy wake behind the
# circle and a spurious drag of a few hundredths; both fall as the cells shrink. The far boundary at
# 6 radii also raises the stagnation Cp several per cent above the isentropic 1.023.

using EulerDG
using CartesianMeshes: CartesianGrid
using Printf: @printf

cpd = isempty(ARGS) ? 8 : parse(Int, ARGS[1])                  # cells per diameter
outdir = mkpath(normpath(joinpath(@__DIR__, "..", "out", "circle")))
prefix = joinpath(outdir, "circle_d$cpd")

eq = CompressibleEulerEquations2D(1.4)
uinf = freestream(eq; mach=0.3)
grid = CartesianGrid(Float64; mincorner=(-3.0, -3.0), maxcorner=(3.0, 3.0), n=(6cpd, 6cpd))
cut = cut_mesh(grid, polygon_body(circle_points(; radius=0.5, h=minimum(grid.d)); name="circle"))
cut_report(cut)
mesh = cut.mesh

far = BoundaryConditionFarfield(uinf)
bcs = NamedTuple{Tuple(mesh.boundary_names)}(Tuple(n == :wall ? boundary_condition_mirror_wall : far
                                                   for n in mesh.boundary_names))
# The degree-4 volume rules: ear clipping does not triangulate mirrored cut fragments alike, and
# with the default degree-2 rules that alone gives the circle a lift of about 1e-3.
semi = DGSemidiscretization(mesh, eq; surface_flux=flux_hllc, boundary_conditions=bcs,
                            volume_degree=4)
u = project((x, t, eq) -> uinf, semi)

runlog = RunLog(prefix * "_run.csv", semi, uinf; every=100)
seconds = @elapsed result = solve_steady!(u, semi; tol=1e-10, maxiter=200_000, callback=runlog)
close(runlog)
@printf("%s after %d iterations (%.1f s): residual down %.1e\n",
        result.converged ? "Converged" : "Not converged", result.iterations, seconds, result.residual)

f = wall_forces(semi, u, uinf; length=1)
s = entropy_error(semi, u, uinf)
cp = surface_cp(semi, u, uinf)
cp0 = (2 / (1.4 * 0.3^2)) * ((1 + 0.2 * 0.3^2)^3.5 - 1)
@printf("cl %+.2e, cd %.3e (spurious), entropy error L2 %.3e and max %.3e\n", f.cl, f.cd, s.l2,
        s.max)
@printf("Cp: largest %.4f against the isentropic stagnation value %.4f; smallest %.4f\n",
        maximum(cp.cp), cp0, minimum(cp.cp))

open(prefix * "_cp.csv", "w") do io
    println(io, "x,y,angle_deg,cp")
    for k in sortperm([atand(p[2], p[1]) for p in cp.x])
        p = cp.x[k]
        @printf(io, "%.12e,%.12e,%.6f,%.12e\n", p[1], p[2], atand(p[2], p[1]), cp.cp[k])
    end
end
write_solution_vtk(prefix * "_solution", semi, u; freestream=uinf)
println("Written to ", outdir)
