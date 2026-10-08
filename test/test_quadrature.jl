# The exact integral of x^a y^b over the polygon through `pts`, by Green's theorem: the boundary
# integral of x^(a+1) y^b / (a+1) dy, with 3-point Gauss on each edge (exact to degree 5).
function monomial_integral(pts, a, b)
    n = length(pts)
    total = 0.0
    for k in 1:n
        p, q = pts[k], pts[mod1(k + 1, n)]
        for (s, w) in zip(GAUSS3.nodes, GAUSS3.weights)
            x = p + s * (q - p)
            total += w * x[1]^(a + 1) * x[2]^b / (a + 1) * (q[2] - p[2])
        end
    end
    return total
end

@testset "quadrature" begin
    # Reference rules.
    for (g, degree) in ((GAUSS2, 3), (GAUSS3, 5)), k in 0:degree
        @test sum(w * s^k for (s, w) in zip(g.nodes, g.weights)) ≈ 1 / (k + 1)
    end
    for (rule, degree) in ((TRIANGLE3, 2), (TRIANGLE6, 4)), a in 0:degree, b in 0:(degree - a)
        # The unit right triangle: the integral of x^a y^b is a! b! / (a + b + 2)!.
        exact = factorial(a) * factorial(b) / factorial(a + b + 2)
        @test isapprox(sum(w * λ[2]^a * λ[3]^b / 2 for (λ, w) in zip(rule.points, rule.weights)),
                       exact; rtol=1e-14)
    end

    square = PV[(0, 0), (1, 0), (1, 1), (0, 1)]
    box = PV[(1, 2), (1.5, 2), (1.5, 2.25), (1, 2.25)]                # a full cell off the origin
    ell = PV[(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]          # nonconvex
    pentagon = PV[(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]             # hanging node at (1, 1)
    hexagon = PV[(cos(θ), sin(θ)) for θ in range(0, 2π; length=7)[1:6]]
    fragment = PV[(0, 0.25), (1, 0.55), (1, 1), (0, 1)]               # a cut cell
    polygons = (; square, box, ell, pentagon, hexagon, fragment)

    @testset "triangulate $name" for (name, pts) in pairs(polygons)
        ids = collect(Int32, eachindex(pts))
        tris = triangulate(pts, ids)
        @test length(tris) == length(pts) - 2
        areas = [polygon_area(pts, collect(t)) for t in tris]
        @test all(>(0), areas)
        @test sum(areas) ≈ polygon_area(pts, ids)
    end

    @testset "polygon_rule $name" for (name, pts) in pairs(polygons)
        ids = collect(eachindex(pts))
        isbox = name in (:square, :box)
        for degree in (isbox ? (3, 5) : (2, 4))
            x, w = polygon_rule(pts, ids; degree)
            @test length(x) == (isbox ? (degree == 3 ? 4 : 9) : (degree == 2 ? 3 : 6) * (length(pts) - 2))
            @test sum(w) ≈ polygon_area(pts, ids)
            for a in 0:degree, b in 0:(degree - a)
                @test isapprox(sum(wq * xq[1]^a * xq[2]^b for (xq, wq) in zip(x, w)),
                               monomial_integral(pts, a, b); rtol=1e-12, atol=1e-14)
            end
        end
    end

    @test throws_with(() -> polygon_rule(square, 1:4; degree=6), "reach degree 5")
    @test throws_with(() -> polygon_rule(ell, 1:6; degree=5), "reach degree 4")
    @test throws_with(() -> triangulate(PV[(0, 0), (0, 1), (1, 1), (1, 0)], 1:4), "no ear")
end
