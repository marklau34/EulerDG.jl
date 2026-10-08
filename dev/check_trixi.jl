# =====================================
# Cross-check EulerDG's copies of Trixi.jl's Euler functions against Trixi itself
#
# Run with `julia --project=dev dev/check_trixi.jl`. The environment pins Trixi 0.17.13, the
# version the copies come from, and keeps Trixi out of the package and its tests. The copies drop
# Trixi's fused multiply-adds, so the two agree to round-off rather than bitwise. The script prints
# the largest relative difference per function over random states and normal directions, and exits
# with an error above 1e-13.

using EulerDG: EulerDG
using Printf: @printf
using Random: Xoshiro
using StaticArrays: SVector
using Trixi: Trixi

const ours = EulerDG.CompressibleEulerEquations2D(1.4)
const theirs = Trixi.CompressibleEulerEquations2D(1.4)

rng = Xoshiro(20261008)
state() = EulerDG.prim2cons(SVector(0.2 + rand(rng), randn(rng), randn(rng), 0.2 + rand(rng)), ours)
reldiff(a, b) = maximum(abs.(collect(a) .- collect(b)) ./ max.(1, abs.(collect(b))))

worst = Dict{String,Float64}()
for _ in 1:10_000
    u, v = state(), state()
    n = SVector(randn(rng), randn(rng))
    x = SVector(randn(rng), randn(rng))
    prim = EulerDG.cons2prim(u, ours)
    bc_ours = EulerDG.BoundaryConditionDirichlet((x, t, eq) -> v)
    bc_theirs = Trixi.BoundaryConditionDirichlet((x, t, eq) -> v)
    checks = ("prim2cons" => (EulerDG.prim2cons(prim, ours), Trixi.prim2cons(prim, theirs)),
              "cons2prim" => (EulerDG.cons2prim(u, ours), Trixi.cons2prim(u, theirs)),
              "pressure" => (EulerDG.pressure(u, ours), Trixi.pressure(u, theirs)),
              "velocity" => (EulerDG.velocity(u, ours), Trixi.velocity(u, theirs)),
              "flux, x" => (EulerDG.flux(u, 1, ours), Trixi.flux(u, 1, theirs)),
              "flux, y" => (EulerDG.flux(u, 2, ours), Trixi.flux(u, 2, theirs)),
              "flux, normal" => (EulerDG.flux(u, n, ours), Trixi.flux(u, n, theirs)),
              "max_abs_speed" => (EulerDG.max_abs_speed(u, v, n, ours),
                                  Trixi.max_abs_speed(u, v, n, theirs)),
              "max_abs_speed_naive" => (EulerDG.max_abs_speed_naive(u, v, n, ours),
                                        Trixi.max_abs_speed_naive(u, v, n, theirs)),
              "max_abs_speeds" => (EulerDG.max_abs_speeds(u, ours),
                                   Trixi.max_abs_speeds(u, theirs)),
              "flux_lax_friedrichs" => (EulerDG.flux_lax_friedrichs(u, v, n, ours),
                                        Trixi.flux_lax_friedrichs(u, v, n, theirs)),
              "BoundaryConditionDirichlet" =>
                  (bc_ours(u, n, x, 0.0, EulerDG.flux_lax_friedrichs, ours),
                   bc_theirs(u, n, x, 0.0, Trixi.flux_lax_friedrichs, theirs)))
    for (name, (a, b)) in checks
        worst[name] = max(get(worst, name, 0.0), reldiff(a, b))
    end
end

println("EulerDG's copies against Trixi $(pkgversion(Trixi)), largest relative difference:")
for name in sort(collect(keys(worst)))
    @printf("  %-28s %.2e\n", name, worst[name])
end
maximum(values(worst)) <= 1e-13 || error("a copy differs from Trixi by more than 1e-13")
println("All copies agree with Trixi to round-off.")
