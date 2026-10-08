# =====================================
# Quadrature: integration rules on segments and polygons
#
# Downstream of the mesh contract. Every rule takes its points from the mesh's vertex pool, so the
# volume and face integrals see exactly the geometry the contract checked.

"""
    EulerDG.Quadrature

Integration rules on element polygons and faces. Planned API, from Phase 1:

- 2-point Gauss on each straight face;
- a 3-point degree-2 rule per triangle, with a 6-point rule for over-integration as a switch;
- ear-clipping triangulation of element polygons, nonconvex ones included;
- a tensor 2 x 2 Gauss rule for full cells, through the same kernel as cut cells;
- exactness tests on polygons.
"""
module Quadrature

end # module Quadrature
