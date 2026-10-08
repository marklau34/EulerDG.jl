# =====================================
# Equations: the 2D compressible Euler equations
#
# Downstream of the mesh contract. Decided on Oct 8, 2026: the equation and flux functions are
# copied from Trixi.jl 0.17.13 rather than taken as a dependency. Trixi would add 2.5 s to every
# load and a 228-package tree, for about eight functions. The copies keep Trixi's names, its
# argument order `(u_ll, u_rr, normal_direction, equations)` and its equation type, so switching to
# the dependency later is a change of `using` lines. A script under `dev/`, with its own
# environment, cross-checks them against Trixi, so `Pkg.test` never loads it.

"""
    EulerDG.Equations

The 2D compressible Euler equations. Planned API:

- Phase 1: copied from Trixi.jl 0.17.13, with its MIT notice -- `CompressibleEulerEquations2D`,
  `prim2cons`, `cons2prim`, the physical `flux` and the Rusanov flux, all taking an unnormalized
  `normal_direction`, so a face's `vector` goes straight in.
- Phase 3: HLLC; the slip wall by mirror state, plus the pressure-only wall flux.
- Phase 4: the characteristic (Riemann-invariant) farfield, which Trixi does not have.
"""
module Equations

end # module Equations
