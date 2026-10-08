# =====================================
# PseudoTime: marching to the steady state
#
# Downstream of the mesh contract. Each element advances with its own stable step, which acts as a
# diagonal preconditioner, so a sliver's tiny step never limits its neighbours.

"""
    EulerDG.PseudoTime

Time marching. Planned API:

- Phase 1: SSP-RK3 with a global step, for the unsteady vortex test.
- Phase 3: local time steps, frozen at the start of each iteration and kept across the stages,
  folded into the right-hand side; a CFL ramp.
- Phase 4: convergence monitors -- the L2 norm of the cell-average density residual, and the
  maximum residual with its element.
"""
module PseudoTime

end # module PseudoTime
