# =====================================
# NACA 0012 in subsonic inviscid flow, at low resolution
#
# Run with `julia --project=examples -t auto examples/naca0012_flow.jl [cells_per_chord] [mach]
# [alpha_deg]` (default 16, 0.5 and 1.25). Cuts a uniform grid round NACA 0012 with a sharp
# trailing edge, from a chord ahead of the leading edge to 1.5 behind the trailing edge and a chord
# above and below, wraps it in coarser levels out to 50 chords or more (`nested_cut_mesh`), and
# marches the P1 DG discretization to a steady state with local steps -- HLLC, the mirror-state
# slip wall and the characteristic farfield. Writes to out/naca0012_flow/:
#
# - `*_run.csv`, the run log: residual and force histories, the moment about the quarter chord;
# - `*_cp.csv`, the surface pressure coefficient at every wall Gauss point, with x/c and the side;
# - `*_solution.vtp`, the cell averages with Cp and the entropy error, for ParaView.
#
# The section is Vassberg and Jameson's sharp-trailing-edge NACA 0012 (J. Aircraft, 2010): the
# standard thickness with its chord extended to x = 1.00893, reference chord 1 and moment about
# x = 0.25. At M 0.5 and 1.25 degrees, the default case, their continuum estimates are printed
# for comparison. At low resolution the leading-edge radius of 1.6% chord spans a fraction of a
# cell, so the nose makes spurious entropy and the lift comes out low.

using EulerDG
using CartesianMeshes: CartesianGrid
using Printf: @printf

cpc = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 16                # cells per chord
mach = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
alpha = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.25         # degrees, nose up
h = 1 / cpc
outdir = mkpath(normpath(joinpath(@__DIR__, "..", "out", "naca0012_flow")))
# No dot in the file name, or WriteVTK takes the rest for an extension: 1.25 becomes "1p25".
tag(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
prefix = joinpath(outdir, "naca0012_c$(cpc)_m$(tag(mach))_a$(tag(alpha))")

# The body, turned nose-up by alpha about its leading edge; the freestream runs along x.
points = naca4_points("0012"; h, alpha_deg=alpha, trailing_edge=:extended)
te = points[1]
le = points[argmax([sum(abs2, p - te) for p in points])]
# Along the chord line, with the reference chord of 1, not the extended length.
chordline = (te - le) / hypot((te - le)...)
quarter = le + 0.25 * chordline

# The finest grid, with the trailing edge 0.3 h into its cell and on the middle line of its row, and
# an even number of cells each way, as the nested builder needs.
mx, my = round(Int, (te[1] - le[1] + 1) / h), round(Int, 1 / h)
x0, y0 = te[1] - 0.3h - mx * h, te[2] - 0.5h - my * h
nx, ny = mx + round(Int, 1.5 / h), 2my + 2
nx += isodd(nx)
grid = CartesianGrid(Float64; mincorner=(x0, y0), maxcorner=(x0 + nx * h, y0 + ny * h),
                     n=(nx, ny))
cut = nested_cut_mesh(grid, polygon_body(points; name="naca0012");
                      farfield=((-50.0, -50.0), (50.0, 50.0)))
cut_report(cut)
mesh = cut.mesh

eq = CompressibleEulerEquations2D(1.4)
uinf = freestream(eq; mach)
far = BoundaryConditionFarfield(uinf)
bcs = NamedTuple{Tuple(mesh.boundary_names)}(Tuple(n == :wall ? boundary_condition_mirror_wall : far
                                                   for n in mesh.boundary_names))
semi = DGSemidiscretization(mesh, eq; surface_flux=flux_hllc, boundary_conditions=bcs,
                            volume_degree=4)
u = project((x, t, eq) -> uinf, semi)

runlog = RunLog(prefix * "_run.csv", semi, uinf; every=100, point=Tuple(quarter))
seconds = @elapsed result = solve_steady!(u, semi; tol=1e-10, maxiter=400_000, callback=runlog)
close(runlog)
@printf("%s after %d iterations (%.1f s): residual down %.1e\n",
        result.converged ? "Converged" : "Not converged", result.iterations, seconds, result.residual)

f = wall_forces(semi, u, uinf; length=1, point=Tuple(quarter))
s = entropy_error(semi, u, uinf)
cp = surface_cp(semi, u, uinf)
cp0 = (2 / (1.4 * mach^2)) * ((1 + 0.2 * mach^2)^3.5 - 1)
@printf("cl %.4f, cd %.2e (spurious), cm %+.4f about the quarter chord\n", f.cl, f.cd, f.cm)
if mach == 0.5 && alpha == 1.25
    # Vassberg and Jameson's Table 2, FLO82 extrapolated to the continuum.
    @printf("Reference (Vassberg and Jameson): cl %.4f, cd 0, cm %+.5f; here cl is %+.1f%% off\n",
            0.180346, -0.002269, 100 * (f.cl - 0.180346) / 0.180346)
end
@printf("Cp: largest %.4f against the isentropic stagnation value %.4f; smallest %.4f\n",
        maximum(cp.cp), cp0, minimum(cp.cp))
@printf("Entropy error: L2 %.3e, max %.3e\n", s.l2, s.max)

# Cp along the chord: x/c from the leading edge, and the side from the chord line.
c = chordline
open(prefix * "_cp.csv", "w") do io
    println(io, "x,y,x_over_c,side,cp")
    for (p, v) in zip(cp.x, cp.cp)
        r = p - le
        side = c[1] * r[2] - c[2] * r[1] >= 0 ? "upper" : "lower"
        @printf(io, "%.12e,%.12e,%.12e,%s,%.12e\n", p[1], p[2], r' * c, side, v)
    end
end
write_solution_vtk(prefix * "_solution", semi, u; freestream=uinf)
println("Written to ", outdir)
