# =====================================
# Post: forces, accuracy measures and output
#
# Downstream of the mesh contract. Wall quantities are integrated at the wall faces' quadrature
# points with each face's `vector`, the same face vectors the closure check covers.

"""
    EulerDG.Post

Postprocessing. Planned API:

- Phase 1: the CSV run log of residual and force history.
- Phase 4: lift, drag and moment from the wall faces; entropy error and spurious drag.
- Phase 5: surface Cp at the wall quadrature points, sorted by x/c; field VTK output, with each
  element a polygon carrying its P1 values at its vertices plus its cell average.
"""
module Post

end # module Post
