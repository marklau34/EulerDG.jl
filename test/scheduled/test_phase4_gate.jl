# =====================================
# The Phase 4 gate, at low resolution: the inviscid circle at Mach 0.3
#
# The circle of diameter 1 sits on a grid line of a uniform grid over [-3, 3]^2, with HLLC, the
# mirror wall, the characteristic farfield and the degree-4 volume rules, marched with local steps
# from the freestream. Low resolution by decision (Oct 8): 8 and 16 cells per diameter. The plan's
# full circle gate needs finer grids and a far boundary much further out, which a uniform grid
# makes expensive; this gate checks what must already hold at this resolution and logs the rest.
#
# 1. Both grids converge 10 orders. Lift is zero to 1e-6, and Cp is symmetric top to bottom to
#    1e-6. Both are round-off in the continuous symmetry, apart from the triangulation of mirrored
#    fragments.
# 2. The spurious drag, the entropy error's L2 norm and the entropy wake fall with h, at order 1.5
#    or better. The stagnation Cp is logged against the isentropic 1.0227: the boundary at 6 radii
#    raises it by several per cent, scaling like (R/L)^2.
# 3. A grid-shift sweep at 8 cells per diameter -- four sub-cell shifts and three sliver shifts (a
#    grid node 1e-3, 1e-6 and 1e-8 h outside the circle) -- converges 10 orders every time. Its
#    table logs the smallest volume fraction, the iterations, the entropy error and the forces, for
#    the plan's decision on safeguard rungs 2 and 3.

@testset "Phase 4 gate (low resolution)" begin
    cp0 = (2 / (1.4 * 0.3^2)) * ((1 + 0.2 * 0.3^2)^3.5 - 1)

    runs = [solve_circle(cpd) for cpd in (8, 16)]
    println("Phase 4 gate, the circle at M 0.3 on [-3, 3]^2 (isentropic stagnation Cp $(round(cp0; digits=4))):")
    for (cpd, c) in zip((8, 16), runs)
        @printf("  %2d cells/D: %6d iterations, %5.0f s; cl %+.1e, cd %.3e; stagnation Cp %.4f; Cp asymmetry %.1e; entropy L2 %.3e, wake %.3e\n",
                cpd, c.run.iterations, c.seconds, c.forces.cl, c.forces.cd, maximum(c.cp.cp),
                cp_mirror_asymmetry(c), c.entropy.l2, wake_entropy(c))
        @test c.run.converged
        @test abs(c.forces.cl) <= 1e-6
        @test cp_mirror_asymmetry(c) <= 1e-6
        @test abs(maximum(c.cp.cp) - cp0) <= 0.1 * cp0
    end
    order(a, b) = log2(a / b)
    a, b = runs
    @printf("  orders: drag %.2f, entropy L2 %.2f, entropy wake %.2f\n", order(a.forces.cd, b.forces.cd),
            order(a.entropy.l2, b.entropy.l2), order(wake_entropy(a), wake_entropy(b)))
    @test order(a.forces.cd, b.forces.cd) >= 1.5
    @test order(a.entropy.l2, b.entropy.l2) >= 1.5
    @test order(wake_entropy(a), wake_entropy(b)) >= 1.5

    println("Phase 4 gate, grid-shift sweep at 8 cells/D:")
    sweeps = vcat([(; label=@sprintf("shift (%.2f, %.2f)", s...), shift=Tuple(s), gap=nothing)
                   for s in grid_shifts(5)[2:end]],
                  [(; label=@sprintf("node %.0e h outside", gap), shift=(0.0, 0.0), gap)
                   for gap in (1e-3, 1e-6, 1e-8)])
    for s in sweeps
        c = solve_circle(8; shift=s.shift, gap=s.gap)
        @printf("  %-22s smallest volume fraction %.1e: %6d iterations, entropy L2 %.3e, cd %.3e, cl %+.1e\n",
                s.label, c.minvf, c.run.iterations, c.entropy.l2, c.forces.cd, c.forces.cl)
        @test c.run.converged
    end
end
