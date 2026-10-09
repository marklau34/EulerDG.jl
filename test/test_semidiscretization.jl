@testset "semidiscretization" begin
    eq = CompressibleEulerEquations2D(1.4)
    uinf = freestream(eq; mach=0.5, alpha_deg=30)

    @testset "boundary conditions by name" begin
        m = two_triangles()
        @test throws_with(() -> DGSemidiscretization(m, eq), "no boundary condition for boundary :y_neg")
        bcs = merge(freestream_conditions(m, uinf), (; wall=BoundaryConditionDirichlet(nothing)))
        @test throws_with(() -> DGSemidiscretization(m, eq; boundary_conditions=bcs),
                          "a boundary condition for :wall, but the mesh has no such boundary")
    end

    # Uniform flow stays uniform to round-off on every mesh, cut fixtures included, with the
    # freestream imposed on all boundaries (the wall too, until Phase 3's slip wall). The residual
    # |K| du is measured against the element's perimeter; du itself is large on the 1e-11 sliver
    # only because it is divided by the sliver's area.
    δ = 1e-11
    grid = CartesianGrid(Float64; mincorner=(-0.7, -0.3), maxcorner=(1.3, 0.9), n=(37, 23))
    meshes = merge(fixtures(), (; sliver=tilted_wall(; y0=1 - 2δ, y1=1 - δ, offset=(1000.0, 1000.0)),
                                 uncut=uncut_mesh(grid),
                                 periodic=uncut_mesh(grid; periodic=(true, true))))
    @testset "freestream preservation on $name" for (name, m) in pairs(meshes)
        semi = DGSemidiscretization(m, eq; boundary_conditions=freestream_conditions(m, uinf))
        u = allocate_state(semi)
        for e in 1:nelements(m)
            u[:, 1, e] .= uinf
        end
        du = rhs!(similar(u), u, semi, 0.0)
        worst = maximum(1:nelements(m)) do e
            perimeter = sum(m.faces.length[f] for f in element_faces(m, e))
            maximum(abs, du[:, :, e]) * semi.mass[e] / perimeter
        end
        @test worst <= 1e-14
    end

    g = CartesianGrid(Float64; mincorner=(-5, -5), maxcorner=(5, 5), n=(12, 12))
    semi = DGSemidiscretization(uncut_mesh(g; periodic=(true, true)), eq)
    vortex = IsentropicVortex()
    u = project(vortex, semi)
    du = similar(u)

    @testset "conservation" begin
        # Perturb the gradient modes too, so every term is exercised.
        u2 = copy(u)
        ne = nelements(semi.mesh)
        u2[:, 2:3, :] .+= 0.01 .* sin.(reshape(1:(8 * ne), 4, 2, ne))
        rhs!(du, u2, semi, 0.0)
        total = sum(semi.mass[e] * du[:, 1, e] for e in 1:nelements(semi.mesh))
        scale = sum(semi.mass[e] * abs.(du[:, 1, e]) for e in 1:nelements(semi.mesh))
        @test all(abs.(total) .<= 1e-13 .* scale)
    end

    @testset "projection and evaluation" begin
        # A linear field is reproduced exactly; mode 1 is the cell average.
        lin(x, t, eq) = SVector(1 + 0.1x[1], 0.2 - 0.05x[2], 0.3, 2 + 0.01 * (x[1] + x[2]))
        ul = project(lin, semi)
        m = semi.mesh
        for e in (1, 50, nelements(m))
            for x in (m.elements.centroid[e], m.points[element_nodes(m, e)[1]])
                @test evaluate(semi, ul, e, x) ≈ lin(x, 0, eq)
            end
            @test cell_average(ul, e) ≈ lin(m.elements.centroid[e], 0, eq)
        end
    end

    @testset "time step" begin
        # At rest with sound speed 1, λ = 1 on every face: Δt = cfl h² / (3 · 4h).
        rest(x, t, eq) = prim2cons(SVector(1.0, 0.0, 0.0, 1 / eq.gamma), eq)
        h = 10 / 12
        @test max_dt(project(rest, semi), semi; cfl=2) ≈ 2 * h / 12
    end

    @testset "positivity" begin
        bad = copy(u)
        bad[1, 1, 7] = -1
        @test_throws DomainError assert_positive(bad, semi)
        err = try
            assert_positive(bad, semi)
        catch e
            e
        end
        @test occursin("element 7", err.msg)
        @test assert_positive(u, semi) === nothing
    end

    @testset "a short vortex run" begin
        uv = copy(u)
        calls = Ref(0)
        r = integrate!(uv, semi, (0.0, 0.5); callback=(u, t, step) -> (calls[] += 1))
        @test r.t == 0.5 && r.steps == calls[] && r.steps > 5
        @test l2_error(semi, uv, vortex, 0.5).l2[1] < 0.15
        @test_throws ErrorException integrate!(copy(u), semi, (0.0, 0.5); maxsteps=2)
    end
end
