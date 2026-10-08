# =====================================
# The isentropic vortex: a convergence study of the P1 DG core on uniform periodic grids
#
# Run with `julia --project=examples examples/vortex.jl [n ...]`, for example
# `julia --project=examples examples/vortex.jl 16 32 64 128`. For each n it marches the vortex on
# an n x n periodic grid over [-5, 5]^2 to t = 1, and prints the L2 density error and the observed
# order. The finest solution's cell averages go to out/vortex/ for ParaView.

using EulerDG
using CartesianMeshes: CartesianGrid
using Printf: @printf, @sprintf

ns = isempty(ARGS) ? [16, 32, 64] : parse.(Int, ARGS)
eq = CompressibleEulerEquations2D(1.4)
vortex = IsentropicVortex()
outdir = mkpath(normpath(joinpath(@__DIR__, "..", "out", "vortex")))

global previous = NaN
for n in ns
    grid = CartesianGrid(Float64; mincorner=(-5, -5), maxcorner=(5, 5), n=(n, n))
    semi = DGSemidiscretization(uncut_mesh(grid; periodic=(true, true)), eq)
    u = project(vortex, semi)
    seconds = @elapsed r = integrate!(u, semi, (0.0, 1.0))
    err = l2_error(semi, u, vortex, 1.0).l2[1]
    order = isnan(previous) ? "" : @sprintf("order %.2f", log2(previous / err))
    @printf("%4d x %-4d  L2(rho) = %.4e  %-11s  %5d steps  %6.2f s\n", n, n, err, order,
            r.steps, seconds)
    global previous = err
    n == last(ns) && write_solution_vtk(joinpath(outdir, "vortex_n$n"), semi, u)
end
println("The n = $(last(ns)) solution is in ", outdir)
