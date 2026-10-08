# =====================================
# The common layer: the mesh contract
#
# Everything both sides of the contract use, included straight into the top-level module: the
# polygon geometry every derived quantity comes from, the `PolygonMesh` types, the one constructor
# every builder calls, the invariant checks, the summary and the mesh VTK writer.
#
# Nothing here refers to the preprocessor's modules, its libraries or their grid types; a test
# scans these files to keep it that way.

include("geometry.jl")
include("mesh.jl")
include("build.jl")
include("check.jl")
include("summary.jl")
include("vtk.jl")
