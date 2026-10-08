using EulerDG
using StaticArrays: SVector
using Test

# The test tier: `commit` runs on every commit; `scheduled` (nightly or on demand) adds the gate
# sweeps of every completed phase, from Phase 1 on. Pick it with `Pkg.test(test_args=["scheduled"])`
# or the EULERDG_TEST_TIER environment variable.
const TIER = isempty(ARGS) ? get(ENV, "EULERDG_TEST_TIER", "commit") : first(ARGS)
TIER in ("commit", "scheduled") || error("unknown test tier \"$TIER\"; use commit or scheduled")

include("fixtures.jl")

@testset "EulerDG ($TIER tier)" begin
    include("test_geometry.jl")
    include("test_mesh.jl")
    include("test_summary.jl")
    include("test_vtk.jl")
    include("test_layering.jl")
    if TIER == "scheduled"
        # Phase gates join here from Phase 1 on, as scheduled/test_phaseN_gate.jl.
    end
end
