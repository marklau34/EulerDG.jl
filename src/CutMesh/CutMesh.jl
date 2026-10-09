# =====================================
# CutMesh: the builders
#
# Upstream of the mesh contract. Each builder hands `PolygonMesh(points, polygons, boundary_edges)`
# its vertex pool, polygons and boundary names, and the contract derives the rest. Every cut is on
# a uniform `CartesianGrid`, the only grid CutCellMethods' polyline clipping accepts; the nested
# builder cuts on one too, and wraps it in coarser uncut levels.

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
- Oct 9: [`nested_cut_mesh`](@ref), the cut grid wrapped in coarser levels out to a far field, a
  CartesianMeshes `NestedGrids` chain, with hanging nodes where levels meet.

Upstream of the mesh contract: the solver-side modules never use it.
"""
module CutMesh

using CartesianMeshes: CartesianMeshes, CartesianGrid, CartesianGridHierarchy, NestedGrids,
                       elementbox, get_node, is_covered, nlevels, patchids, subgrid
using CutCellMethods: CutCellMethods, PolylineClippingCutCell, allocate_cache, update_cache!
using Printf: @printf
using StaticArrays: SVector
using ..EulerDG: PolygonMesh, mesh_summary, polygon_area, _polygon_problem
using ..Geometry: Body, polygon_perimeter

include("uncut.jl")
include("cut.jl")
include("halfplane.jl")
include("nested.jl")

export uncut_mesh, cut_mesh, cut_report, halfplane_mesh, nested_cut_mesh

end # module CutMesh
