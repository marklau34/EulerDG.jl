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
- Phase 2: the cut adapter over CutCellMethods' exact polyline clipping. It builds faces from the
  cutter's arcs, so both sides of every partial edge share their split points. Every fluid fragment
  becomes its own element, and slivers are dropped below a `drop_area` keyword.
- Phase 2: the preprocessor report, which extends `mesh_summary` with lost-feature and
  truncated-tip lists.

Upstream of the mesh contract: the solver-side modules never use it.
"""
module CutMesh

using CartesianMeshes: CartesianGrid, get_node
using StaticArrays: SVector
using ..EulerDG: PolygonMesh

include("uncut.jl")

export uncut_mesh

end # module CutMesh
