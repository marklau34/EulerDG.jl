@testset "mesh" begin
    @testset "two triangles" begin
        m = two_triangles()
        @test (nelements(m), nfaces(m), npoints(m), nboundaries(m)) == (2, 5, 4, 4)
        @test interior_faces(m) == 1:1
        # Boundaries in the order their names first appear along the elements' edges.
        @test m.boundary_names == [:y_neg, :x_pos, :y_pos, :x_neg]
        @test [boundary_faces(m, n) for n in m.boundary_names] == [2:2, 3:3, 4:4, 5:5]
        @test boundary_index(m, :x_pos) == 2
        @test throws_with(() -> boundary_faces(m, :wall), "no boundary :wall")

        @test element_nodes(m, 2) == [1, 3, 4]
        @test element_faces(m, 1) == [2, 3, 1]
        @test element_faces(m, 2) == [1, 4, 5]
        @test m.elements.area == [0.5, 0.5]
        @test m.elements.volume_fraction == [1.0, 1.0]
        @test m.elements.centroid[1] ≈ PV(2, 1) / 3

        # The diagonal: element 1 runs it 3 -> 1, and its vector points out of element 1.
        f = 1
        @test (m.faces.left[f], m.faces.right[f], m.faces.boundary[f]) == (1, 2, 0)
        @test m.faces.nodes[f] == (3, 1)
        @test m.faces.vector[f] == PV(-1, 1)
        @test m.faces.length[f] ≈ sqrt(2)
        @test m.faces.normal[f] ≈ PV(-1, 1) / sqrt(2)
        @test (face_sign(m, 1, f), face_sign(m, 2, f), face_sign(m, 2, 2)) == (1, -1, 0)
        @test m.faces.boundary[2:5] == 1:4 && all(iszero, m.faces.right[2:5])

        c = check_mesh(m)
        @test c.ok && c.max_closure == 0
        @test all(c[k] == 0 for k in MESH_INVARIANTS)
        @test sprint(show, m) == "PolygonMesh{Float64}(2 elements, 5 faces)"
        @test occursin("y_neg: 1 face", sprint(show, MIME"text/plain"(), m))

        m32 = two_triangles(Float32)
        @test m32 isa PolygonMesh{Float32}
        @test check_mesh(m32).ok
    end

    @testset "hanging node" begin
        m = hanging_node()
        @test (nelements(m), nfaces(m), length(interior_faces(m))) == (3, 10, 3)
        @test length(element_nodes(m, 1)) == length(element_faces(m, 1)) == 5
        @test m.elements.area == [2.0, 1.0, 1.0]
        @test check_mesh(m).max_closure == 0
        # The pentagon's long side is two faces, one to each square.
        right_side = [f for f in element_faces(m, 1) if m.faces.right[f] != 0]
        @test sort(m.faces.right[right_side]) == [2, 3]
    end

    @testset "periodic 2 x 2" begin
        m = periodic_2x2()
        @test (nelements(m), nfaces(m), nboundaries(m)) == (4, 8, 0)
        @test interior_faces(m) == 1:8
        glued = [f for f in 1:nfaces(m) if !iszero(m.faces.shift[f])]
        @test length(glued) == 4
        @test count(f -> m.faces.shift[f] == PV(-2, 0), glued) == 2
        @test count(f -> m.faces.shift[f] == PV(0, -2), glued) == 2
        # The :x_pos and :y_pos sides are the left copies.
        @test all(m.faces.left[f] in (m.faces.shift[f][1] != 0 ? (2, 4) : (3, 4)) for f in glued)
        for f in glued
            a, b = m.faces.nodes[f]
            r = m.faces.right[f]
            ids = element_nodes(m, r)
            k = findfirst(==(f), element_faces(m, r))
            c, d = ids[k], ids[mod1(k + 1, length(ids))]
            @test m.points[c] == m.points[b] + m.faces.shift[f]
            @test m.points[d] == m.points[a] + m.faces.shift[f]
        end
        @test check_mesh(m).max_closure == 0
    end

    @testset "tilted wall" begin
        m = tilted_wall()
        @test (nelements(m), length(interior_faces(m)), nfaces(m)) == (4, 4, 12)
        @test m.boundary_names == [:wall, :x_neg, :x_pos, :y_pos]
        @test length(boundary_faces(m, :wall)) == 2
        @test m.elements.volume_fraction[1:2] ≈ [0.6, 0.3]
        @test m.elements.volume_fraction[3:4] == [1.0, 1.0]
        @test m.elements.parent == 1:4
        # The crossing at x = 1 is one pool vertex: the bottom fragments share two vertices.
        @test length(intersect(element_nodes(m, 1), element_nodes(m, 2))) == 2
        @test check_mesh(m).max_closure <= 1e-13
    end

    @testset "sliver 1000 cells from the origin" begin
        δ = 1e-11
        m = tilted_wall(; y0=1 - 2δ, y1=1 - δ, offset=(1000.0, 1000.0))
        for e in 1:2
            pts = m.points[element_nodes(m, e)]
            exact = naive_area([big.(p) for p in pts])
            # About the first vertex the area is exact; in absolute coordinates it is lost.
            @test isapprox(m.elements.area[e], exact; rtol=1e-12)
            @test abs(naive_area(pts) - exact) > exact / 2
        end
        @test isapprox(m.elements.volume_fraction[1], 1.75δ; rtol=1e-2)
        @test isapprox(m.elements.volume_fraction[2], 1.25δ; rtol=1e-2)
        @test m.elements.volume_fraction[3:4] == [1.0, 1.0]
        @test check_mesh(m).max_closure <= 1e-13
    end

    @testset "split cell" begin
        m = split_cell()
        @test nelements(m) == 2 && isempty(interior_faces(m))
        @test (m.elements.parent, m.elements.fragment) == ([1, 1], [1, 2])
        @test m.elements.volume_fraction ≈ [0.4, 0.4]
        @test length(boundary_faces(m, :wall)) == 2
    end

    @testset "invalid inputs" begin
        sq = PV[(0, 0), (1, 0), (1, 1), (0, 1)]
        sides = Dict((1, 2) => :y_neg, (2, 3) => :x_pos, (3, 4) => :y_pos, (4, 1) => :x_neg)
        none = Dict{Tuple{Int,Int},Symbol}()

        # Polygons that are not valid elements.
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3], [1, 4, 3]], sides),
                          "element 2: its signed area is -0.5")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3, 2]], none),
                          "element 1: pool id 2 appears twice")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2]], none), "element 1: it has 2 vertices")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 7]], none), "pool id 7, outside the pool")
        bowtie = PV[(0, 0), (0, 1), (2, 0), (2, 2)]           # positive area, edges 2 and 4 cross
        @test throws_with(() -> PolygonMesh(bowtie, [[1, 2, 3, 4]], none),
                          "element 1: edges 2 and 4 cross or touch")

        # Edges that do not pair up.
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3]], Dict((1, 2) => :y_neg, (2, 3) => :x_pos)),
                          "element 1, edge 3 (3 -> 1) has no neighbour and no boundary name")
        three = PV[(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 2)]
        @test throws_with(() -> PolygonMesh(three, [[1, 2, 3], [1, 3, 4], [1, 3, 5]], sides),
                          "elements 2 and 3 both run edge 1 -> 3")
        # The same coordinates under a new pool id are not the same vertex.
        dup = PV[(0, 0), (1, 0), (1, 1), (0, 1), (1, 1)]
        dup_sides = Dict((1, 2) => :y_neg, (2, 3) => :x_pos, (5, 4) => :y_pos, (4, 1) => :x_neg)
        @test throws_with(() -> PolygonMesh(dup, [[1, 2, 3], [1, 5, 4]], dup_sides),
                          "element 1, edge 3 (3 -> 1) has no neighbour")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3], [1, 3, 4]],
                                            merge(sides, Dict((3, 1) => :wall))),
                          "borders element 2, but edge 3 -> 1 is also named :wall")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3], [1, 3, 4]],
                                            merge(sides, Dict((2, 1) => :y_neg))),
                          "boundary edge 2 -> 1 (:y_neg) is not an edge of any element")

        # Periodic sides that do not match.
        @test throws_with(() -> periodic_2x2(; shift_x=PV(-1.5, 0)), "has no partner on :x_neg")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3, 4]], sides;
                                            periodic=((:x_pos, :x_neg) => PV(-1, 0),)),
                          "is glued to its own element")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3, 4]], sides;
                                            periodic=((:x_pos, :wall) => PV(-1, 0),)),
                          "periodic side :wall has no faces")

        # Per-element inputs and boundary names.
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3, 4]], sides; parent=[1, 2]),
                          "parent has 2 entries for 1 elements")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3, 4]], sides;
                                            boundary_names=[:y_neg, :x_pos, :y_pos]),
                          "boundary :x_neg has faces but is missing")
        @test throws_with(() -> PolygonMesh(sq, [[1, 2, 3, 4]], sides; parent=[1],
                                            parent_area=[0.5]),
                          "element 1 has volume fraction 2.0")
        m = PolygonMesh(sq, [[1, 2, 3, 4]], sides;
                        boundary_names=[:x_neg, :x_pos, :y_neg, :y_pos, :wall])
        @test m.boundary_names == [:x_neg, :x_pos, :y_neg, :y_pos, :wall]
        @test isempty(boundary_faces(m, :wall))
        @test check_mesh(m).ok

        # A mesh edited after it was built: its stored geometry is stale.
        m = two_triangles()
        m.points[3] = PV(1, 1.1)
        c = check_mesh(m)
        @test !c.ok && c.geometry > 0
        @test startswith(c.first.geometry, "element 1: its area")
        @test throws_with(() -> assert_mesh(m), "geometry:")
    end
end
