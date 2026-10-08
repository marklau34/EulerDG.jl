# =====================================
# CutMesh: the builders
#
# Upstream of the mesh contract. Each builder hands `PolygonMesh(points, polygons, boundary_edges)`
# its vertex pool, polygons and boundary names, and the contract derives the rest. Version 1 keeps
# CartesianMeshes' uniform `CartesianGrid` as the background, the only grid CutCellMethods accepts.
# Every vertex comes from `get_node`, and every parent area from node differences: `get_elem_area`
# and the upper corner of `get_elem_bounds` do not agree with the nodes bitwise.

"""
    EulerDG.CutMesh

Builders of the mesh contract. Planned API:

- Phase 1: a `PolygonMesh` straight from an uncut, uniform `CartesianGrid`, periodic or not.
- Phase 2: the cut adapter over CutCellMethods' exact polyline clipping. It builds faces from the
  cutter's arcs, so both sides of every partial edge share their split points. Every fluid fragment
  becomes its own element, and slivers are dropped below a `drop_area` keyword.
- Phase 2: the preprocessor report, which extends `mesh_summary` with lost-feature and
  truncated-tip lists.

Upstream of the mesh contract: the solver-side modules never use it.
"""
module CutMesh

end # module CutMesh
