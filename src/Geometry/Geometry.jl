# =====================================
# Geometry: the bodies
#
# Upstream of the mesh contract. CutCellMethods' exact polyline clipping takes only an `SDFMesh` --
# a closed, counter-clockwise polyline -- so every body is a polygon, and its segment length must
# shrink with the cell size (the cutter's brief suggests at most h/4) for grid convergence to hold.

"""
    EulerDG.Geometry

The bodies the preprocessor cuts, as SDFLibrary polylines. Planned API, from Phase 2:

- a polygonal circle whose segment length shrinks with the cell size;
- a NACA 4-digit airfoil with a sharp trailing edge;
- sub-cell grid-shift offsets for the Phase 4 sweep.

Upstream of the mesh contract: only `CutMesh` and the top-level driver may use it.
"""
module Geometry

end # module Geometry
