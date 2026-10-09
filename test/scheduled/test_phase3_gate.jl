# =====================================
# The Phase 3 gate: cut-cell DG and walls
#
# 1. Uniform flow along a straight wall tilted off the grid axes. On half-plane meshes at three
#    angles, eight sub-cell offsets of the line and two sliver placements (a node 1e-4 and 1e-6 h
#    from the wall), both slip walls -- the mirror state with HLLC, the pressure-only flux with
#    Rusanov -- keep the flow uniform under 300 local steps. Elements of volume fraction 1e-6 and
#    above stay within 1e-12; every element stays within 1e-12 times h over its shortest face,
#    because a wall face 1e-7 long rounds its own direction by its endpoints' last bits.
# 2. A manufactured solution on the unit square cut by a wall at 17 degrees, with zero normal
#    velocity on the wall. Its steady state, with local steps, converges at second order with
#    either slip wall and either flux: every observed order at least 1.8.

@testset "Phase 3 gate" begin
    @testset "uniform flow along a tilted wall" begin
        println("Phase 3 gate, uniform flow along a tilted wall (32 x 32, 300 local steps):")
        g = CartesianGrid(Float64; mincorner=(0.0, 0.0), maxcorner=(1.0, 1.0), n=(32, 32))
        h = 1 / 32
        for deg in (17.0, 33.0, -24.0)
            θ = deg2rad(deg)
            n = PV(-sin(θ), cos(θ))
            node = PV(CartesianMeshes.get_node(g, CartesianIndex(15, 13)))
            lines = vcat([PV(0.0, 0.3 + s[2] * h) for s in grid_shifts(8)],
                         [node - gap * h * n for gap in (1e-4, 1e-6)])
            worst = (; regular=0.0, scaled=0.0, all=0.0)
            minvf = 1.0
            for (wall, sf) in ((boundary_condition_mirror_wall, flux_hllc),
                               (boundary_condition_slip_wall, flux_lax_friedrichs)), p in lines
                m = halfplane_mesh(g; point=p, normal=n)
                minvf = min(minvf, minimum(m.elements.volume_fraction))
                d = wall_aligned_deviation(m, θ; wall, surface_flux=sf)
                worst = map(max, worst, d)
            end
            @printf("  %6.1f degrees: smallest volume fraction %.1e; deviation %.1e on regular elements, %.1e on any (%.1e scaled)\n",
                    deg, minvf, worst.regular, worst.all, worst.scaled)
            @test worst.regular <= 1e-12
            @test worst.scaled <= 1e-12
        end
    end

    @testset "manufactured solution, $label" for (label, wall, sf) in
                                                  (("mirror wall, HLLC", boundary_condition_mirror_wall, flux_hllc),
                                                   ("pressure-only wall, HLLC", boundary_condition_slip_wall, flux_hllc),
                                                   ("mirror wall, Rusanov", boundary_condition_mirror_wall, flux_lax_friedrichs))
        println("Phase 3 gate, manufactured solution ($label):")
        errors = Float64[]
        for nc in (16, 32, 64)
            s = @timed solve_mms(nc; wall, surface_flux=sf)
            r = s.value
            @test r.run.converged
            push!(errors, r.error)
            @printf("  %3d x %-3d  L2(rho) = %.3e  %s  %6d iterations, %5.1f s\n", nc, nc, r.error,
                    length(errors) == 1 ? "          " : @sprintf("order %.2f", log2(errors[end - 1] / errors[end])),
                    r.run.iterations, s.time)
        end
        @test all(>=(1.8), log2.(errors[1:(end - 1)] ./ errors[2:end]))
    end
end
