@testset "walls and local steps" begin
    eq = CompressibleEulerEquations2D(1.4)

    @testset "HLLC" begin
        u = prim2cons(SVector(1.2, 0.3, -0.4, 0.9), eq)
        v = prim2cons(SVector(0.8, -0.2, 0.5, 1.1), eq)
        n = SVector(0.6, -0.8)
        @test flux_hllc(u, u, n, eq) ≈ flux(u, n, eq)
        @test flux_hllc(u, v, 0.5 * n, eq) ≈ 0.5 * flux_hllc(u, v, n, eq)
        @test flux_hllc(u, v, n, eq) ≈ -flux_hllc(v, u, -n, eq)
        # Supersonic to the right: the upwind flux.
        fast = prim2cons(SVector(1.0, 3.0, 0.0, 1.0), eq)
        @test flux_hllc(fast, v, SVector(1.0, 0.0), eq) == flux(fast, SVector(1.0, 0.0), eq)
        @test rotate_from_x(rotate_to_x(u, n, eq), n, eq) ≈ u
    end

    @testset "slip walls" begin
        S = SVector(0.0, 2.0)                              # a wall face of length 2, normal +y
        x = SVector(0.0, 0.0)
        along = prim2cons(SVector(1.1, 0.5, 0.0, 0.8), eq)  # tangent to the wall
        into = prim2cons(SVector(1.1, 0.5, 0.3, 0.8), eq)   # moving into the wall
        p = pressure(along, eq)
        for sf in (flux_hllc, flux_lax_friedrichs)
            # Flow along the wall: both walls give the pressure flux (0, p S, 0).
            @test boundary_condition_mirror_wall(along, S, x, 0.0, sf, eq) ≈ SVector(0, p * S[1], p * S[2], 0)
            @test boundary_condition_slip_wall(along, S, x, 0.0, sf, eq) ≈ SVector(0, p * S[1], p * S[2], 0)
            # Flow into the wall: no mass or energy flux, and a pressure above the inner one.
            for wall in (boundary_condition_mirror_wall, boundary_condition_slip_wall)
                F = wall(into, S, x, 0.0, sf, eq)
                @test abs(F[1]) <= 1e-15 && abs(F[4]) <= 1e-15
                @test F[3] / S[2] > pressure(into, eq)
                @test wall(into, 0.5 * S, x, 0.0, sf, eq) ≈ 0.5 * F
            end
        end
    end

    @testset "half-plane mesh" begin
        g = CartesianGrid(Float64; mincorner=(0.0, 0.0), maxcorner=(1.0, 1.0), n=(16, 16))
        θ = deg2rad(17)
        n = PV(-sin(θ), cos(θ))
        m = halfplane_mesh(g; point=(0.0, 0.3), normal=n)
        @test check_mesh(m).ok
        @test m.boundary_names == [:wall, :x_neg, :x_pos, :y_neg, :y_pos]
        @test isempty(boundary_faces(m, :y_neg))           # the bottom side is all solid
        # The fluid is the box above the line, the wall the line's chord across it.
        y1 = 0.3 + tan(θ)
        @test sum(m.elements.area) ≈ 1 - (0.3 + y1) / 2
        @test sum(m.faces.length[f] for f in boundary_faces(m, :wall)) ≈ hypot(1, y1 - 0.3)
        @test all(f -> m.faces.normal[f] ≈ -n, boundary_faces(m, :wall))   # out of the fluid
        @test throws_with(() -> halfplane_mesh(g; point=(0.0, 0.3), normal=(0.0, 1.0)),
                          "tilted off the grid axes")
        # A node 1e-6 h above the line leaves a corner fragment of volume fraction about 1e-12;
        # 1e-8 h, one below 1e-12, which is dropped and closed onto its neighbours.
        h = 1 / 16
        node = PV(CartesianMeshes.get_node(g, CartesianIndex(8, 7)))
        tiny = halfplane_mesh(g; point=node - 1e-6 * h * n, normal=n)
        @test 1e-13 < minimum(tiny.elements.volume_fraction) < 1e-11
        gone = halfplane_mesh(g; point=node - 1e-8 * h * n, normal=n)
        @test minimum(gone.elements.volume_fraction) > 1e-11
        @test check_mesh(gone).ok
        @test nelements(gone) == nelements(tiny) - 1
    end

    @testset "local steps" begin
        g = CartesianGrid(Float64; mincorner=(0.0, 0.0), maxcorner=(1.0, 1.0), n=(8, 8))
        semi = DGSemidiscretization(uncut_mesh(g; periodic=(true, true)), eq)
        rest(x, t, eq) = prim2cons(SVector(1.0, 0.0, 0.0, 1 / eq.gamma), eq)
        u = project(rest, semi)
        dt = local_dt!(zeros(64), u, semi; cfl=2)
        @test all(≈(2 * (1 / 8) / 12), dt)                 # cfl h² / (3 · 4h), sound speed 1
        @test max_dt(u, semi; cfl=2) == minimum(dt)
        @test allocs((dt, u, semi) -> local_dt!(dt, u, semi; cfl=2), dt, u, semi) == 0
        # A constant source adds itself to every cell average's time derivative.
        S = SVector(0.1, -0.2, 0.3, 0.4)
        sourced = DGSemidiscretization(semi.mesh, eq; source_terms=(u, x, t, eq) -> S)
        du = rhs!(similar(u), u, sourced, 0.0)
        @test all(e -> du[:, 1, e] ≈ S && maximum(abs, du[:, 2:3, e]) < 1e-14, 1:64)

        # Uniform flow along a tilted wall stays uniform under local steps, slivers included:
        # to round-off on regular elements, and within round-off scaled by h over the shortest
        # face where a wall face only 1e-7 long rounds its own direction.
        θ = deg2rad(17)
        n = PV(-sin(θ), cos(θ))
        g = CartesianGrid(Float64; mincorner=(0.0, 0.0), maxcorner=(1.0, 1.0), n=(16, 16))
        node = PV(CartesianMeshes.get_node(g, CartesianIndex(8, 7)))
        for m in (halfplane_mesh(g; point=(0.0, 0.3), normal=n),
                  halfplane_mesh(g; point=node - 1e-6 * n / 16, normal=n))
            d = wall_aligned_deviation(m, θ; wall=boundary_condition_mirror_wall,
                                       surface_flux=flux_hllc, iterations=100)
            @test d.regular <= 1e-12 && d.scaled <= 1e-12
        end
    end

    @testset "steady manufactured solution" begin
        # The steady state on a coarse grid, from the exact solution and from uniform flow with the
        # CFL ramp: both converge, to the same state.
        a = solve_mms(8)
        @test a.run.converged && a.run.history[end] <= 1e-9
        @test a.error < 5e-3
        b = solve_mms(8; start=:uniform, tol=1e-11)
        @test b.run.converged && b.run.residual <= 1e-11
        @test b.run.iterations > a.run.iterations
        @test maximum(abs, a.u - b.u) < 1e-8
        # The monitors: the integrated density residual, and where it is largest.
        r = residual_norms(rhs!(similar(b.u), b.u, b.semi, 0.0), b.semi)
        @test 0 < r.max <= r.l2 && 1 <= r.element <= nelements(b.mesh)
    end
end
