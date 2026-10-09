using EulerDG
using CartesianMeshes: CartesianMeshes, CartesianGrid
using Printf: @printf, @sprintf
using StaticArrays: SVector
using Test

# The test tier: `commit` runs on every commit; `scheduled` (nightly or on demand) adds the gate
# sweeps of every completed phase. Pick it with `Pkg.test(test_args=["scheduled"])` or the
# EULERDG_TEST_TIER environment variable.
const TIER = isempty(ARGS) ? get(ENV, "EULERDG_TEST_TIER", "commit") : first(ARGS)
TIER in ("commit", "scheduled") || error("unknown test tier \"$TIER\"; use commit or scheduled")

include("fixtures.jl")
include("flows.jl")

@testset "EulerDG ($TIER tier)" begin
    include("test_geometry.jl")
    include("test_mesh.jl")
    include("test_summary.jl")
    include("test_vtk.jl")
    include("test_layering.jl")
    include("test_uncut.jl")
    include("test_quadrature.jl")
    include("test_basis.jl")
    include("test_equations.jl")
    include("test_semidiscretization.jl")
    include("test_post.jl")
    include("test_bodies.jl")
    include("test_cut.jl")
    include("test_walls.jl")
    if TIER == "scheduled"
        include("scheduled/test_phase1_gate.jl")
        include("scheduled/test_phase2_gate.jl")
        include("scheduled/test_phase3_gate.jl")
    end
end
