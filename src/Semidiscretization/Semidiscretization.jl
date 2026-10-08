# =====================================
# Semidiscretization: the DG right-hand side
#
# Downstream of the mesh contract. The solver data -- quadrature points and weights, basis values,
# face vectors per quadrature point -- is built here from a `PolygonMesh`, not stored in the
# contract, so curved faces later change only this data.

"""
    EulerDG.Semidiscretization

The weak-form modal DG right-hand side. Planned API, from Phase 1:

- element and face solver data built from the mesh, as struct-of-arrays;
- state and residual as 4 x 3 x elements arrays;
- `rhs!` in four passes: element volume integrals, interior faces, boundary faces (one
  type-stable loop per boundary name), then division by |K|.
"""
module Semidiscretization

end # module Semidiscretization
