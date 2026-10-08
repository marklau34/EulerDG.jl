@testset "basis" begin
    δ = 1e-11
    meshes = (; two_triangles=two_triangles(), hanging_node=hanging_node(), tilted_wall=tilted_wall(),
              sliver=tilted_wall(; y0=1 - 2δ, y1=1 - δ, offset=(1000.0, 1000.0)),
              ell=PolygonMesh(PV[(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], [1:6],
                              Dict((i, mod1(i + 1, 6)) => :side for i in 1:6)))
    @testset "$name" for (name, m) in pairs(meshes), e in 1:nelements(m)
        ids = element_nodes(m, e)
        lo, hi = m.elements.lo[e], m.elements.hi[e]
        c = (lo + hi) / 2
        x, w = polygon_rule(m.points, ids; degree=2, origin=c)
        b = element_basis(lo, hi, x, w; origin=c)
        @test b.center == c
        # The mass matrix is |K| times the identity, under both the operator's rule and the finer
        # one, and |K| is the element's area -- also on the sliver 1000 cells out, because the
        # rules are taken relative to the box centre.
        @test b.mass ≈ m.elements.area[e]
        for degree in (2, 4)
            xd, wd = polygon_rule(m.points, ids; degree, origin=c)
            M = mass_matrix(b, xd, wd; origin=c)
            @test maximum(abs, M - b.mass * one(M)) <= 1e-13 * b.mass
        end
        # Mode 1 is the constant 1; the modes are linear, with the gradients `basis_gradients`
        # gives; the absolute and relative forms agree.
        @test all(basis_values(b, xq, c)[1] == 1 for xq in x)
        G = basis_gradients(b)
        @test G[:, 1] == PV(0, 0)
        p, q = x[1], x[end]
        @test basis_values(b, q, c) - basis_values(b, p, c) ≈ G' * (q - p) atol = 1e-12
        name == :sliver || @test basis_values(b, c + p) ≈ basis_values(b, p, c)
    end

    # A sliver along the diagonal, too thin for its x and y modes to stay independent.
    pts = PV[(0, 0), (1, 1), (1, 1 + 1e-13)]
    x, w = polygon_rule(pts, 1:3)
    lo, hi = polygon_bounds(pts, 1:3)
    @test throws_with(() -> element_basis(lo, hi, x, w), "degenerate")
end
