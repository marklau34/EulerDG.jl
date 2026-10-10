@testset "farfield, forces and the circle" begin
    eq = CompressibleEulerEquations2D(1.4)
    uinf = freestream(eq; mach=0.3, alpha_deg=10)
    far = BoundaryConditionFarfield(uinf)
    x = SVector(0.0, 0.0)

    @testset "characteristic farfield" begin
        # The freestream passes through unchanged, entering or leaving, and the flux scales with
        # the normal direction.
        for n in (SVector(1.0, 0.0), SVector(-1.0, 0.0), SVector(0.0, 1.0), SVector(0.6, -0.8))
            @test far(uinf, n, x, 0.0, flux_hllc, eq) ≈ flux(uinf, n, eq)
            @test far(uinf, 2.5 * n, x, 0.0, flux_hllc, eq) ≈ 2.5 * far(uinf, n, x, 0.0, flux_hllc, eq)
        end
        # Supersonic: inflow takes the freestream, outflow the interior state.
        fast_in = prim2cons(SVector(1.0, 2.0, 0.0, 1 / 1.4), eq)
        n = SVector(-1.0, 0.0)                              # the flow enters through this face
        @test far(fast_in, n, x, 0.0, flux_hllc, eq) == flux_hllc(fast_in, uinf, n, eq)
        @test far(fast_in, -n, x, 0.0, flux_hllc, eq) == flux_hllc(fast_in, fast_in, -n, eq)
        # Subsonic outflow keeps the interior's entropy and tangential velocity, and the pressure
        # moves towards the freestream's: an interior overpressure leaves.
        hot = prim2cons(SVector(1.0, 0.3cosd(10), 0.3sind(10), 1.2 / 1.4), eq)
        F = far(hot, SVector(1.0, 0.0), x, 0.0, flux_lax_friedrichs, eq)
        @test F[1] > flux(uinf, SVector(1.0, 0.0), eq)[1]
    end

    @testset "forces, Cp and entropy on a uniform state" begin
        g = CartesianGrid(Float64; mincorner=(-2.0, -2.0), maxcorner=(2.0, 2.0), n=(16, 16))
        c = cut_mesh(g, polygon_body(circle_points(; radius=0.5, h=0.25)))
        m = c.mesh
        semi = DGSemidiscretization(m, eq; surface_flux=flux_hllc,
                                    boundary_conditions=wall_conditions(m, boundary_condition_mirror_wall, far))
        u = project((x, t, eq) -> uinf, semi)
        f = wall_forces(semi, u, uinf; length=1, point=(0.25, 0))
        @test maximum(abs, f.force) < 1e-14 && abs(f.moment) < 1e-14
        @test abs(f.cl) < 1e-13 && abs(f.cd) < 1e-13 && abs(f.cm) < 1e-13
        cp = surface_cp(semi, u, uinf)
        @test length(cp.x) == 2 * length(boundary_faces(m, :wall))
        @test maximum(abs, cp.cp) < 1e-13
        @test all(p -> 0.49 < hypot(p...) < 0.51, cp.x)
        s = entropy_error(semi, u, uinf)
        @test s.l2 < 1e-13 && s.max < 1e-13

        # The pitching moment is nose-up positive. A pressure rising with height pushes the circle
        # down through its centre; about a point at its front, that is behind the point and pitches
        # the nose up. The raw moment is counter-clockwise positive, so it is negative.
        rising = project((x, t, eq) -> prim2cons(SVector(1.0, 0.3cosd(10), 0.3sind(10), 1 / 1.4 + 0.01x[2]), eq),
                         semi)
        f = wall_forces(semi, rising, uinf; length=1, point=(-0.5, 0))
        @test f.force[2] < 0 && f.moment < 0 && f.cm > 0
        @test f.cm ≈ -f.moment / (0.5 * 0.3^2)

        # A uniform pressure 0.1 above the freestream's pushes on a closed body with no net force.
        up = project((x, t, eq) -> prim2cons(SVector(1.0, 0.3cosd(10), 0.3sind(10), 1 / 1.4 + 0.1), eq),
                     semi)
        @test maximum(abs, wall_forces(semi, up, uinf).force) < 1e-14
        @test all(≈(0.1 / (0.5 * 0.3^2)), surface_cp(semi, up, uinf).cp)

        # The run log and the solution output.
        dir = mktempdir()
        log = RunLog(joinpath(dir, "run.csv"), semi, uinf; every=5)
        r = solve_steady!(u, semi; maxiter=20, callback=log)
        close(log)
        lines = readlines(joinpath(dir, "run.csv"))
        @test lines[1] == "iteration,walltime,residual,residual_max,residual_element,cl,cd,cm"
        @test length(lines) == 1 + 1 + 20 ÷ 5
        @test startswith(lines[2], "1,") && parse(Float64, split(lines[2], ',')[3]) == 1
        files = write_solution_vtk(joinpath(dir, "circle"), semi, u; freestream=uinf)
        text = read(files[1], String)
        @test occursin("Name=\"cp\"", text) && occursin("Name=\"entropy_error\"", text)
    end

    @testset "the circle's symmetry" begin
        # The lower half of the polygon mirrors the upper bitwise.
        p = circle_points(; radius=0.5, n=51)
        @test all(p[k + 1] == SVector(p[51 - k + 1][1], -p[51 - k + 1][2]) for k in 1:50)
        # A circle vertex 6e-17 inside a cell, beside a crossing (240 degrees on 51 points, with
        # h = R/2): the loop point merges into the crossing instead of folding back.
        g = CartesianGrid(Float64; mincorner=(-4.0, -4.0), maxcorner=(4.0, 4.0), n=(32, 32))
        c = cut_mesh(g, polygon_body(circle_points(; radius=0.5, h=0.25)))
        @test check_mesh(c.mesh).ok && abs(c.report.area_defect) < 1e-13
    end
end
