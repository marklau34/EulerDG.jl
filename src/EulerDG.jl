"""
    EulerDG

Steady, inviscid, subsonic 2D Euler flow around static signed-distance geometry: a P1 modal
discontinuous Galerkin discretization on a Cartesian cut-cell mesh, converged by local pseudo-time
stepping.

After preprocessing, the solver sees only a `PolygonMesh` of polygonal elements and faces. Full
cells and cut fragments run through one code path, and no Cartesian index reaches the solver.

The development plan is in `Cut-Cell DG Euler Solver — Architecture & Development Plan.md`, with
one implementation brief per phase in `briefs/`.
"""
module EulerDG

end # module EulerDG
