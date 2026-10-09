@testset "bodies" begin
    # Polygon bodies: counter-clockwise, no repeated points.
    tri = PV[(0.3, 0.3), (0.6, 0.3), (0.4, 0.6)]
    b = polygon_body(tri; name="triangle")
    @test b.points == tri && b.name == "triangle"
    @test sprint(show, b) == "Body(\"triangle\", 3 points)"
    @test throws_with(() -> polygon_body(reverse(tri)), "must run counter-clockwise")
    @test throws_with(() -> polygon_body(PV[(0, 0), (1, 0), (1, 0), (0, 1)]), "coincide")
    @test throws_with(() -> polygon_body(tri[1:2]), "at least 3 points")
    @test polygon_perimeter(PV[(0, 0), (1, 0), (1, 1), (0, 1)]) == 4

    # The circle: enough segments for `per_cell` per cell of size h, counter-clockwise.
    c = circle_points(; center=(0.1, -0.2), radius=0.5, h=0.05)
    @test length(c) == ceil(Int, 4 * π / 0.05)
    @test all(p -> hypot((p - PV(0.1, -0.2))...) ≈ 0.5, c)
    @test maximum(hypot((c[mod1(k + 1, end)] - c[k])...) for k in eachindex(c)) <= 0.05 / 4
    @test length(circle_points(; radius=1, n=7)) == 7
    @test throws_with(() -> circle_points(; radius=1), "either the number of segments")

    # NACA sections: a sharp trailing edge, counter-clockwise, the leading edge at `le`, the widest
    # segment at most h / per_cell.
    h = 0.02
    p = naca4_points("0012"; h)
    k = ceil(Int, 4 * π / 2 / h)
    @test length(p) == 2k
    @test p[1] == PV(1, 0)                                         # the trailing edge, once
    @test p[k + 1] == PV(0, 0)                                     # the leading edge
    @test all(q -> q[2] > 0, p[2:k]) && all(q -> q[2] < 0, p[(k + 2):end])
    @test maximum(hypot((p[mod1(j + 1, end)] - p[j])...) for j in eachindex(p)) <= h / 4
    area = polygon_body(p).points |> pts -> sum(pts[j][1] * pts[mod1(j + 1, end)][2] -
                                                pts[mod1(j + 1, end)][1] * pts[j][2]
                                                for j in eachindex(pts)) / 2
    # The exact area of NACA 0012 is 10 t times the integral of the thickness polynomial.
    exact = 10 * 0.12 * (0.2969 * 2 / 3 - 0.1260 / 2 - 0.3516 / 3 + 0.2843 / 4 - 0.1036 / 5)
    @test isapprox(area, exact; rtol=1e-4)
    cambered = naca4_points("2412"; n=40, chord=2, le=(1, 1))
    @test cambered[1] ≈ PV(3, 1) && cambered[41] ≈ PV(1, 1)
    @test polygon_body(cambered).name == "body"
    nose_up = naca4_points("0012"; n=40, alpha_deg=10)
    @test nose_up[1] ≈ PV(cosd(10), -sind(10))
    @test throws_with(() -> naca4_points("012"; n=10), "needs 4 digits")
    @test throws_with(() -> naca4_points("2012"; n=10), "camber without its position")

    # Grid shifts: (0, 0) first, then distinct offsets inside the cell.
    s = grid_shifts(32)
    @test s[1] == PV(0, 0) && length(unique(s)) == 32
    @test all(q -> all(0 .<= q .< 1), s)
    # Both coordinates are spread over the cell.
    @test all(count(q -> q[d] < 0.5, s) in 12:20 for d in 1:2)
end
