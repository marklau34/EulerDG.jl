@testset "summary" begin
    s = mesh_summary(split_cell())
    @test (s.nsplit, s.nfull, s.ncut) == (2, 0, 0)
    @test (s.nelements, s.ninterior, s.boundaries.wall.faces) == (2, 0, 2)

    m = tilted_wall()
    s = mesh_summary(m)
    @test (s.nfull, s.ncut, s.nsplit) == (2, 2, 0)
    @test s.area ≈ 4 - 1.1                                  # the solid below the wall has area 1.1
    @test s.boundaries.wall.faces == 2
    @test s.boundaries.wall.length ≈ hypot(2, 0.6)
    @test s.boundary_closure <= 1e-14
    @test s.max_closure <= 1e-13
    @test s.vf_histogram[end] == 2 && s.vf_histogram[end - 1] == 2   # [0.1, 1) holds 0.3 and 0.6
    @test (s.min_volume_fraction, s.min_volume_fraction_element) == (m.elements.volume_fraction[2], 2)

    δ = 1e-11
    m = tilted_wall(; y0=1 - 2δ, y1=1 - δ, offset=(1000.0, 1000.0))
    s = mesh_summary(m)
    @test s.min_volume_fraction_element == 2
    @test s.min_volume_fraction == m.elements.volume_fraction[2]
    @test VF_BINS[3] == "[1e-11, 1e-10)" && s.vf_histogram[3] == 2
    @test sum(s.vf_histogram) == nelements(m)

    s = mesh_summary(periodic_2x2())
    @test (s.ninterior, s.nperiodic, s.boundaries, s.boundary_closure) == (8, 4, (;), 0.0)

    # A reflex vertex makes an element nonconvex; a collinear (hanging) vertex does not.
    ell = PV[(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    m = PolygonMesh(ell, [1:6], Dict((i, mod1(i + 1, 6)) => :side for i in 1:6))
    s = mesh_summary(m)
    @test (s.nnonconvex, s.max_vertices) == (1, 6)
    @test mesh_summary(hanging_node()).nnonconvex == 0

    # The report prints the summary and returns it.
    io = IOBuffer()
    @test mesh_report(io, tilted_wall()) == mesh_summary(tilted_wall())
    text = String(take!(io))
    @test occursin("2 full, 2 cut, 0 split", text)
    @test occursin("wall", text) && occursin("[0.1, 1)", text)
end
