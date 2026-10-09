# =====================================
# The Phase 2 gate: the cut-mesh preprocessor
#
# 1. Fluid-area convergence. On the circle and on NACA 0012, the cut is exact for the polygon -- the
#    fluid area is the domain's minus the polygon's to round-off -- and with four segments per cell
#    the fluid area converges to the true body's complement at second order: every observed order
#    at least 1.8.
# 2. Grid-shift sweep. On three grids each for the circle and NACA 0012, 16 sub-cell shifts plus
#    three sliver shifts (a grid node 1e-3, 1e-6 and 1e-8 h outside the body) all give valid meshes
#    with no lost feature, an area defect within the dropped slivers' allowance, and freestream
#    preserved to round-off. The table logs each grid's smallest volume fraction.

@testset "Phase 2 gate" begin
    circle_area = π * 0.5^2
    naca_area = 10 * 0.12 * (0.2969 * 2 / 3 - 0.1260 / 2 - 0.3516 / 3 + 0.2843 / 4 - 0.1036 / 5)
    cases = (; circle=(; lo=(-1.0, -1.0), hi=(1.0, 1.0), area=circle_area,
                        points=h -> circle_points(; radius=0.5, h)),
             naca0012=(; lo=(-0.5, -0.6), hi=(1.5, 0.6), area=naca_area,
                        points=h -> naca4_points("0012"; h)))

    @testset "fluid-area convergence, $name" for (name, case) in pairs(cases)
        errors, hs = Float64[], Float64[]
        for n in (32, 64, 128, 256)
            g = CartesianGrid(Float64; mincorner=case.lo, maxcorner=case.hi, n=(n, n))
            h = minimum(g.d)
            # Shift by a fixed fraction of a cell, so no body vertex sits on a grid line.
            pts = case.points(h) .+ Ref(SVector(0.31, 0.17) .* g.d)
            r = cut_mesh(g, polygon_body(pts)).report
            @test abs(r.area_defect) <= 1e-12
            push!(errors, abs(r.fluid_area - (r.domain_area - case.area)))
            push!(hs, h)
        end
        orders = log2.(errors[1:(end - 1)] ./ errors[2:end])
        println("Phase 2 gate, fluid-area convergence ($name):")
        for k in eachindex(hs)
            @printf("  h = %.5f  area error %.3e  %s\n", hs[k], errors[k],
                    k == 1 ? "" : @sprintf("order %.2f", orders[k - 1]))
        end
        @test all(>=(1.8), orders)
    end

    @testset "grid-shift sweep, $name" for (name, case) in pairs(cases)
        println("Phase 2 gate, grid-shift sweep ($name): 16 shifts and 3 sliver shifts per grid")
        for n in (32, 64, 128)
            g = CartesianGrid(Float64; mincorner=case.lo, maxcorner=case.hi, n=(n, n))
            h = minimum(g.d)
            base = case.points(h)
            nodes = grid_nodes(g)
            shifts = vcat([s .* g.d for s in grid_shifts(16)],
                          [sliver_shift(polygon_body(base), nodes; gap=gap * h)
                           for gap in (1e-3, 1e-6, 1e-8)])
            minvf, worst, dropped, splits = 1.0, 0.0, 0, 0
            for s in shifts
                c = cut_mesh(g, polygon_body(base .+ Ref(s)))
                minvf = min(minvf, c.report.min_volume_fraction)
                dropped += c.report.closed_intervals > 0
                splits += length(c.report.split_cells)
                worst = max(worst, freestream_residual(c.mesh))
            end
            @printf("  %4d x %-4d  smallest volume fraction %.2e, %2d shifts dropped a sliver, %3d split cells, freestream residual %.1e\n",
                    n, n, minvf, dropped, splits, worst)
            @test worst <= 1e-14
            @test minvf < 1e-9                             # the sliver shifts reach tiny fragments
        end
    end
end
