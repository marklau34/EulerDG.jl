@testset "equations" begin
    eq = CompressibleEulerEquations2D(1.4)
    @test (eq.gamma, eq.inv_gamma_minus_one) == (1.4, inv(1.4 - 1))
    @test nvariables(eq) == 4 && ndims(eq) == 2

    prim = SVector(1.2, 0.3, -0.4, 0.9)
    u = prim2cons(prim, eq)
    @test cons2prim(u, eq) ≈ prim
    @test (density(u, eq), velocity(u, eq), pressure(u, eq)) == (u[1], SVector(u[2], u[3]) / u[1],
                                                                 pressure(u, eq))
    @test pressure(u, eq) ≈ prim[4]

    # Fluxes: consistency, the normal-direction flux from the Cartesian ones, homogeneity in the
    # normal direction (weighted face vectors rely on it), and conservation.
    v = prim2cons(SVector(0.8, -0.2, 0.5, 1.1), eq)
    n = SVector(0.6, -0.8)
    @test flux(u, n, eq) ≈ n[1] * flux(u, 1, eq) + n[2] * flux(u, 2, eq)
    @test flux_lax_friedrichs(u, u, n, eq) ≈ flux(u, n, eq)
    @test flux_lax_friedrichs(u, v, 0.5 * n, eq) ≈ 0.5 * flux_lax_friedrichs(u, v, n, eq)
    @test flux_lax_friedrichs(u, v, n, eq) ≈ -flux_lax_friedrichs(v, u, -n, eq)
    @test flux_lax_friedrichs === FluxLaxFriedrichs()
    @test sprint(show, flux_lax_friedrichs) == "FluxLaxFriedrichs(max_abs_speed)"

    # Wave speeds.
    c(w) = sqrt(eq.gamma * pressure(w, eq) / w[1])
    vn(w) = velocity(w, eq)' * n
    @test max_abs_speed(u, v, n, eq) ≈ max(abs(vn(u)) + c(u), abs(vn(v)) + c(v))
    @test max_abs_speed_naive(u, v, n, eq) ≈ max(abs(vn(u)), abs(vn(v))) + max(c(u), c(v))
    @test max_abs_speeds(u, eq)[1] ≈ abs(velocity(u, eq)[1]) + c(u)

    # A Dirichlet condition calls the surface flux with the boundary state outside.
    bc = BoundaryConditionDirichlet((x, t, eq) -> v)
    @test bc(u, n, SVector(0.0, 0.0), 0.0, flux_lax_friedrichs, eq) ==
          flux_lax_friedrichs(u, v, n, eq)

    # The freestream: density 1, sound speed 1, Mach 0.5 at 30 degrees.
    uinf = freestream(eq; mach=0.5, alpha_deg=30)
    @test uinf[1] == 1 && c(uinf) ≈ 1
    @test velocity(uinf, eq) ≈ 0.5 * SVector(cosd(30), sind(30))

    # The vortex: uniform entropy, the freestream far away, advected and periodic.
    vortex = IsentropicVortex()
    for x in (SVector(0.3, -0.2), SVector(1.0, 1.0), SVector(-4.0, 2.5))
        w = vortex(x, 0.0, eq)
        @test pressure(w, eq) / w[1]^eq.gamma ≈ 1
        @test vortex(x + SVector(1.5, 1.5), 1.5, eq) ≈ w
        @test vortex(x + SVector(10.0, 0.0), 0.0, eq) ≈ w
    end
    far = vortex(SVector(4.9, 4.9), 0.0, eq)
    @test cons2prim(far, eq) ≈ SVector(1, 1, 1, 1) atol = 1e-8
    @test cons2prim(vortex(SVector(0.0, 0.0), 0.0, eq), eq)[1] < 0.7
end
