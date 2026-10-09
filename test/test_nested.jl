# =====================================
# The nested builder: the cut grid wrapped in coarser levels out to a far field
#
# On the circle and on NACA 0012: a valid mesh, the levels the far field and the rings ask for,
# areas that add up to the domain less the body, coarse cells that are whole rectangles (hanging
# nodes and all) with volume fraction 1 and the tensor rule, a mesh symmetric about an origin on the
# body's axis to round-off, uniform flow preserved across the coarse-fine interfaces, and the input
# errors.

# The circle of diameter 1 centred at the origin, on a finest box 1.5 diameters round it and 3.5
# behind, with h = 1/8.
function nested_circle(; buffer=4, farfield=((-12.0, -12.0), (12.0, 12.0)), origin=(0.0, 0.0),
                       mincorner=(-1.5, -1.5), maxcorner=(3.5, 1.5), n=(40, 24))
    g = CartesianGrid(Float64; mincorner, maxcorner, n)
    body = polygon_body(circle_points(; radius=0.5, h=minimum(g.d)); name="circle")
    return nested_cut_mesh(g, body; farfield, buffer, origin)
end

@testset "nested mesh" begin
    @testset "the circle" begin
        c = nested_circle()
        m, r = c.mesh, c.report
        @test check_mesh(m).ok
        @test m.boundary_names == [:wall, :x_neg, :x_pos, :y_neg, :y_pos]
        @test all(!isempty(boundary_faces(m, k)) for k in 1:nboundaries(m))

        # The levels: cells doubling outward from h = 1/8, every element counted once.
        L = length(r.levels)
        @test L >= 3
        @test [v.h[1] for v in r.levels] == [0.125 * 2^(L - l) for l in 1:L]
        @test sum(v.elements for v in r.levels) == nelements(m) == r.elements
        lo, hi = r.domain
        @test all(lo .<= (-12, -12)) && all(hi .>= (12, 12))
        @test abs(r.fluid_area - (r.domain_area - r.body_area)) <= 1e-12 * r.domain_area
        @test r.wall_length ≈ r.body_perimeter

        # Coarse cells are whole rectangles, hanging nodes and all: volume fraction exactly 1, the
        # tensor rule, and more than four vertices where they meet finer cells.
        coarse = findall(>(prod((40, 24))), m.elements.parent)
        @test length(coarse) == nelements(m) - r.levels[end].elements
        @test all(==(1), m.elements.volume_fraction[coarse])
        @test all(e -> EulerDG.Quadrature._is_box(m.points, element_nodes(m, e)), coarse)
        @test count(e -> length(element_nodes(m, e)) > 4, coarse) > 0
        @test maximum(e -> length(element_nodes(m, e)), coarse) <= 8

        # Symmetric top to bottom about the origin: every grid node mirrors bitwise, and so every
        # point to round-off -- the cutter places body vertices from their cell's corner.
        mirror(p) = typeof(p)(p[1], -p[2])
        @test all(p -> minimum(q -> maximum(abs, q - mirror(p)), m.points) <= 1e-15, m.points)

        # Uniform flow, imposed on every boundary, is a steady state across the interfaces.
        @test freestream_residual(m) <= 1e-14

        io = IOBuffer()
        cut_report(io, c)
        text = String(take!(io))
        @test occursin("$L levels", text) && occursin("lost-feature checks passed", text)
    end

    @testset "the far field and the rings" begin
        # A wider ring needs no more levels to reach the same far field, and a farther far field
        # needs more; the domain always contains the far field.
        a = nested_circle(; buffer=4).report
        b = nested_circle(; buffer=8).report
        f = nested_circle(; buffer=4, farfield=((-48.0, -48.0), (48.0, 48.0))).report
        @test length(b.levels) <= length(a.levels) < length(f.levels)
        @test b.levels[end].elements == a.levels[end].elements
        @test all(f.domain[1] .<= (-48, -48)) && all(f.domain[2] .>= (48, 48))
    end

    @testset "NACA 0012" begin
        # The airfoil with its sharp trailing edge, the finest box a chord ahead of it and above
        # and below, two behind; the default origin, the box's corner.
        g = CartesianGrid(Float64; mincorner=(-1.0, -1.0), maxcorner=(3.0, 1.0), n=(64, 32))
        body = polygon_body(naca4_points("0012"; h=minimum(g.d), alpha_deg=2); name="naca0012")
        c = nested_cut_mesh(g, body; farfield=((-20.0, -20.0), (20.0, 20.0)), buffer=4)
        @test check_mesh(c.mesh).ok
        @test abs(c.report.fluid_area - (c.report.domain_area - c.report.body_area)) <=
              1e-12 * c.report.domain_area
        @test freestream_residual(c.mesh) <= 1e-14
    end

    @testset "input errors" begin
        @test throws_with(() -> nested_circle(; n=(41, 24), maxcorner=(3.625, 1.5)),
                          "even number along each axis")
        @test throws_with(() -> nested_circle(; origin=(0.125, 0.0)),
                          "must be an even whole number")
        @test throws_with(() -> nested_circle(; farfield=((-1.0, -1.0), (1.0, 1.0))),
                          "must contain the grid's box")
        # The body reaching the finest grid's edge: the cut itself refuses it.
        @test throws_with(() -> nested_circle(; mincorner=(-0.5, -1.5), n=(32, 24)),
                          "the body reaches the edge of the grid")
        @test throws_with(() -> nested_circle(; buffer=0), "buffer must be at least 1")
    end
end
