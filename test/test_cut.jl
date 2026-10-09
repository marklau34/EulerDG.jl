# The largest freestream residual |K| du on `m`, relative to each element's perimeter, with the
# freestream imposed on every boundary.
function freestream_residual(m)
    eq = CompressibleEulerEquations2D(1.4)
    uinf = freestream(eq; mach=0.5, alpha_deg=30)
    semi = DGSemidiscretization(m, eq; boundary_conditions=freestream_conditions(m, uinf))
    u = allocate_state(semi)
    u[:, 1, :] .= uinf
    du = rhs!(similar(u), u, semi, 0.0)
    return maximum(1:nelements(m)) do e
        maximum(abs, du[:, :, e]) * semi.mass[e] / sum(m.faces.length[f] for f in element_faces(m, e))
    end
end

# Grid nodes as points, for `sliver_shift`.
grid_nodes(g) = [SVector{2,Float64}(CartesianMeshes.get_node(g, I))
                 for I in CartesianIndices(g.n .+ 1)]

@testset "cut mesh" begin
    grid = CartesianGrid(Float64; mincorner=(-1.03, -0.97), maxcorner=(1.01, 1.05), n=(34, 34))
    h = minimum(grid.d)

    @testset "circle" begin
        body = polygon_body(circle_points(; radius=0.5, h); name="circle")
        c = cut_mesh(grid, body)
        m, r = c.mesh, c.report
        @test m.boundary_names == [:wall, :x_neg, :x_pos, :y_neg, :y_pos]
        @test check_mesh(m).ok
        # Every cell is accounted for: fluid cells, cut cells' fragments, solid cells.
        @test r.cells.fluid + r.cells.cut + r.cells.split + r.cells.solid == prod(grid.n)
        @test r.elements == nelements(m) == r.full + r.fragments
        @test r.full == r.cells.fluid && r.fragments == r.cells.cut
        # The cut is exact for the polygon: the fluid is the domain minus the body, and the wall
        # is the body's perimeter.
        @test abs(r.area_defect) <= 1e-13
        @test r.fluid_area ≈ r.domain_area - r.body_area
        @test r.wall_length ≈ r.body_perimeter
        @test sum(m.faces.length[f] for f in boundary_faces(m, :wall)) ≈ polygon_perimeter(body.points)
        # Fragments are cut cells: below volume fraction 1, and nonconvex where a body vertex is
        # inside the cell.
        @test all(<(1), m.elements.volume_fraction[m.elements.volume_fraction .!= 1])
        @test count(==(1), m.elements.volume_fraction) == r.full
        @test r.nonconvex > 0
        @test freestream_residual(m) <= 1e-14
        io = IOBuffer()
        @test cut_report(io, c) === r
        @test occursin("lost-feature checks passed", String(take!(io)))
    end

    @testset "NACA 0012 at grid shifts" begin
        g = CartesianGrid(Float64; mincorner=(-0.5, -0.6), maxcorner=(1.5, 0.6), n=(64, 38))
        nsplit = 0
        for s in grid_shifts(8)
            pts = naca4_points("0012"; h=minimum(g.d)) .+ Ref(s .* g.d)
            c = cut_mesh(g, polygon_body(pts))
            m, r = c.mesh, c.report
            @test abs(r.area_defect) <= 1e-13
            @test r.wall_length ≈ r.body_perimeter
            # A split cell's fragments share no face: no interior face joins two elements of one
            # background cell.
            P = m.elements.parent
            @test all(P[m.faces.left[f]] != P[m.faces.right[f]] for f in interior_faces(m))
            nsplit += length(r.split_cells)
            @test freestream_residual(m) <= 1e-14
        end
        @test nsplit > 0                                   # the trailing edge splits cells
    end

    @testset "slivers" begin
        pts = circle_points(; radius=0.5, h)
        nodes = grid_nodes(grid)
        # A grid node 1e-4 h outside the body leaves a fragment of volume fraction about 1e-8.
        s = sliver_shift(polygon_body(pts), nodes; gap=1e-4 * h)
        c = cut_mesh(grid, polygon_body(pts .+ Ref(s)))
        @test 1e-9 < c.report.min_volume_fraction < 1e-7
        @test c.report.closed_intervals == 0
        @test freestream_residual(c.mesh) <= 1e-14
        # At 1e-8 h the fragment would be about 1e-16: it is dropped, its intervals are closed, and
        # the wall moves onto its neighbours.
        s = sliver_shift(polygon_body(pts), nodes; gap=1e-8 * h)
        c = cut_mesh(grid, polygon_body(pts .+ Ref(s)))
        @test c.report.min_volume_fraction > 1e-12
        @test c.report.closed_intervals > 0 && c.report.cells.snapped > 0
        @test abs(c.report.area_defect) <= 1e-13
        @test check_mesh(c.mesh).ok
        @test freestream_residual(c.mesh) <= 1e-14
    end

    @testset "lost features and bad bodies" begin
        g = CartesianGrid(Float64; mincorner=(0.0, 0.0), maxcorner=(1.0, 1.0), n=(10, 10))
        island = polygon_body(PV[(0.42, 0.42), (0.47, 0.42), (0.44, 0.47)])
        @test throws_with(() -> cut_mesh(g, island), "lies wholly inside cell (5, 5)")
        # A comb of 40 thin teeth puts more crossings on one cell than the cutter's walk tracks.
        comb = PV[(0.38, 0.32), (0.62, 0.32), (0.62, 0.39)]
        for k in 39:-1:0
            x = 0.4005 + k * 0.0024
            append!(comb, PV[(x + 0.0012, 0.39), (x + 0.0012, 0.405), (x, 0.405), (x, 0.39)])
        end
        push!(comb, PV(0.38, 0.39))
        @test @test_logs((:warn, r"could not be walked"), match_mode=:any,
                         throws_with(() -> cut_mesh(g, polygon_body(comb)), "could not walk cell"))
        edge = polygon_body(PV[(0.02, 0.3), (0.5, 0.3), (0.5, 0.6)])
        @test throws_with(() -> cut_mesh(g, edge), "reaches the edge of the grid")
    end
end
