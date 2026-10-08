# Bytes allocated by a call, after a first call to compile it.
allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))

@testset "geometry" begin
    square = PV[(0, 0), (1, 0), (1, 1), (0, 1)]
    triangle = PV[(0, 0), (1, 0), (0, 1)]
    ell = PV[(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]          # nonconvex at (1, 1)
    for (pts, area, centroid) in ((square, 1.0, PV(0.5, 0.5)), (triangle, 0.5, PV(1, 1) / 3),
                                  (ell, 3.0, PV(5, 5) / 6))
        ids = Int32.(eachindex(pts))
        @test polygon_area(pts, ids) ≈ area
        @test polygon_area(pts, reverse(ids)) ≈ -area
        @test polygon_centroid(pts, ids) ≈ centroid
        @test polygon_bounds(pts, ids) == (reduce((a, b) -> min.(a, b), pts),
                                           reduce((a, b) -> max.(a, b), pts))

        # Scaled to h = 1e-3 and moved to (1, 1).
        moved = [PV(1, 1) + 1e-3 * p for p in pts]
        @test isapprox(polygon_area(moved, ids), 1e-6 * area; rtol=1e-12)
        @test isapprox(polygon_centroid(moved, ids), PV(1, 1) + 1e-3 * centroid; rtol=1e-14)

        # Any integer vector works, views included; nothing allocates, and everything infers.
        v = view(ids, :)
        @test @inferred(polygon_area(pts, v)) ≈ area
        @test @inferred(polygon_centroid(pts, v)) ≈ centroid
        @inferred polygon_bounds(pts, v)
        @test polygon_area(pts, collect(eachindex(pts))) ≈ area
        @test allocs(polygon_area, pts, v) == 0
        @test allocs(polygon_centroid, pts, v) == 0
        @test allocs(polygon_bounds, pts, v) == 0
    end

    # Face vectors point out of a counter-clockwise polygon, with the edge's length.
    @test face_vector(PV(0, 0), PV(1, 0)) == PV(0, -1)
    @test face_vector(PV(1, 0), PV(1, 1)) == PV(1, 0)
    @test @inferred(face_vector(PV(0, 0), PV(2, 1))) == PV(1, -2)
    @test allocs(face_vector, PV(0, 0), PV(1, 1)) == 0
    pts = ell
    n = length(pts)
    @test sum(face_vector(pts[k], pts[mod1(k + 1, n)]) for k in 1:n) == PV(0, 0)
end
