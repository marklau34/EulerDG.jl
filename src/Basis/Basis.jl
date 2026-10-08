# =====================================
# Basis: the per-element modal basis
#
# Downstream of the mesh contract. Each element gets its own basis, built on its bounding box and
# orthogonalized against its own quadrature, so a sliver's mass matrix stays well conditioned.

"""
    EulerDG.Basis

The per-element P1 modal basis. Planned API, from Phase 1:

- the starting modes 1, (x - xc)/hx and (y - yc)/hy on the element's bounding box;
- Gram-Schmidt against the element's own quadrature, normalized so the mass matrix is |K| times
  the identity and mode 0 is the cell average;
- a unit test that the mass matrix equals |K| times the identity.
"""
module Basis

end # module Basis
