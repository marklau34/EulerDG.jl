@testset "uncut mesh" begin
    # A grid whose spacing is not a power of two: get_elem_area and the nodes disagree in the last
    # bit there, yet every element has volume fraction exactly 1.
    g = CartesianGrid(Float64; mincorner=(-0.7, -0.3), maxcorner=(1.3, 0.9), n=(37, 23))
    m = uncut_mesh(g)
    @test (nelements(m), npoints(m)) == (37 * 23, 38 * 24)
    @test m.boundary_names == [:x_neg, :x_pos, :y_neg, :y_pos]
    @test [length(boundary_faces(m, n)) for n in m.boundary_names] == [23, 23, 37, 37]
    @test length(interior_faces(m)) == 36 * 23 + 37 * 22
    @test all(==(1), m.elements.volume_fraction)
    @test m.elements.parent == 1:(37 * 23)
    @test check_mesh(m).ok && check_mesh(m).max_closure == 0
    # Element e is the cell LinearIndices(g)[e].
    @test all(m.elements.centroid[LinearIndices(g)[I]] ≈ CartesianMeshes.get_elem_centroid(g, I)
              for I in CartesianIndices(g))

    mx = uncut_mesh(g; periodic=(true, false))
    @test mx.boundary_names == [:y_neg, :y_pos]
    @test count(f -> !iszero(mx.faces.shift[f]), interior_faces(mx)) == 23
    mxy = uncut_mesh(g; periodic=(true, true))
    @test nboundaries(mxy) == 0
    @test count(f -> !iszero(mxy.faces.shift[f]), interior_faces(mxy)) == 23 + 37
    @test all(==(1), mxy.elements.volume_fraction)

    thin = CartesianGrid(Float64; mincorner=(0, 0), maxcorner=(1, 1), n=(1, 4))
    @test throws_with(() -> uncut_mesh(thin; periodic=(true, false)), "own element")
end
