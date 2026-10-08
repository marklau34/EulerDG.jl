# =====================================
# The Phase 1 gate: the DG core on uncut grids
#
# 1. Isentropic vortex convergence. The L2 density error at t = 1 falls at second order on uniform
#    periodic grids, with square cells and with dx = 2 dy: every observed order at least 1.9.
# 2. Freestream preservation over time. With the freestream imposed on every boundary, uniform
#    flow stays uniform to 1e-12 through t = 5 on an uncut grid and on two Phase 0 fixtures, one
#    with a hanging node and one with cut cells.

@testset "Phase 1 gate" begin
    eq = CompressibleEulerEquations2D(1.4)
    vortex = IsentropicVortex()

    function vortex_error(n)
        g = CartesianGrid(Float64; mincorner=(-5, -5), maxcorner=(5, 5), n)
        semi = DGSemidiscretization(uncut_mesh(g; periodic=(true, true)), eq)
        u = project(vortex, semi)
        integrate!(u, semi, (0.0, 1.0))
        return l2_error(semi, u, vortex, 1.0).l2[1]
    end

    @testset "vortex convergence, $label" for (label, grids) in
                                               (("square cells", [(n, n) for n in (16, 32, 64, 128)]),
                                                ("dx = 2dy", [(n, n ÷ 2) for n in (32, 64, 128)]))
        errors = [vortex_error(n) for n in grids]
        orders = log2.(errors[1:(end - 1)] ./ errors[2:end])
        println("Phase 1 gate, vortex convergence ($label):")
        for (k, n) in enumerate(grids)
            @printf("  %4d x %-4d  L2(rho) = %.4e  %s\n", n..., errors[k],
                    k == 1 ? "" : @sprintf("order %.3f", orders[k - 1]))
        end
        @test all(>=(1.9), orders)
    end

    uinf = freestream(eq; mach=0.5, alpha_deg=30)
    grid = CartesianGrid(Float64; mincorner=(-0.7, -0.3), maxcorner=(1.3, 0.9), n=(24, 24))
    meshes = (; uncut=uncut_mesh(grid), hanging_node=hanging_node(), tilted_wall=tilted_wall())
    @testset "freestream preservation over time, $name" for (name, m) in pairs(meshes)
        semi = DGSemidiscretization(m, eq; boundary_conditions=freestream_conditions(m, uinf))
        u = allocate_state(semi)
        u[:, 1, :] .= uinf
        r = integrate!(u, semi, (0.0, 5.0))
        deviation = max(maximum(abs, u[:, 1, :] .- uinf), maximum(abs, u[:, 2:3, :]))
        @printf("Phase 1 gate, freestream on %-12s %6d steps, largest deviation %.2e\n", name,
                r.steps, deviation)
        @test deviation <= 1e-12
    end
end
