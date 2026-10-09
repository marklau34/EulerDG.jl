# =====================================
# CutMesh: the builders
#
# Upstream of the mesh contract. Each builder hands `PolygonMesh(points, polygons, boundary_edges)`
# its vertex pool, polygons and boundary names, and the contract derives the rest. Version 1 keeps
# CartesianMeshes' uniform `CartesianGrid` as the background, the only grid CutCellMethods accepts.

"""
    EulerDG.CutMesh

Builders of the mesh contract.

- Phase 1: [`uncut_mesh`](@ref), a `PolygonMesh` straight from an uncut, uniform
  `CartesianGrid`, periodic or not.
- Phase 2: [`cut_mesh`](@ref), the cut builder over CutCellMethods' exact polyline clipping:
  crossings shared bitwise across cells, every fluid fragment its own element, slivers dropped
  below `drop_area`, and the lost-feature checks that stop the build.
- Phase 2: [`cut_report`](@ref), the preprocessor report.
- Phase 3: [`halfplane_mesh`](@ref), a grid cut by one straight line running right across it, for
  the tilted-wall verification tests.

Upstream of the mesh contract: the solver-side modules never use it.
"""
module CutMesh

using CartesianMeshes: CartesianGrid, get_node
using CutCellMethods: CutCellMethods, PolylineClippingCutCell, allocate_cache, update_cache!
using Printf: @printf
using StaticArrays: SVector
using ..EulerDG: PolygonMesh, mesh_summary, polygon_area, _polygon_problem
using ..Geometry: Body, polygon_perimeter

include("uncut.jl")
include("cut.jl")
include("halfplane.jl")

export uncut_mesh, cut_mesh, cut_report, halfplane_mesh

end # module CutMesh
