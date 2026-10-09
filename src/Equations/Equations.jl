# =====================================
# Equations: the 2D compressible Euler equations
#
# Downstream of the mesh contract. Decided on Oct 8, 2026: the equation and flux functions are
# copied from Trixi.jl 0.17.13 (trixi.jl) rather than taken as a dependency. Trixi would add 2.5 s
# to every load and a 228-package tree, for about eight functions. The copies keep Trixi's names,
# its argument order `(u_ll, u_rr, normal_direction, equations)` and its equation type, so
# switching to the dependency later is a change of `using` lines. `dev/check_trixi.jl`, with its
# own environment, cross-checks them against Trixi, so `Pkg.test` never loads it.

"""
    EulerDG.Equations

The 2D compressible Euler equations.

- Copied from Trixi.jl 0.17.13, with its MIT notice:
  - the equations and their variables: `CompressibleEulerEquations2D`, `prim2cons`, `cons2prim`,
    `density`, `velocity` and `pressure`;
  - the fluxes: the physical `flux`, the Rusanov flux `flux_lax_friedrichs` with its wave-speed
    estimates (Phase 1), and HLLC, `flux_hllc` (Phase 3);
  - the boundary conditions: `BoundaryConditionDirichlet`, and `boundary_condition_slip_wall`,
    the pressure-only wall flux (Phase 3).

  The fluxes take an unnormalized `normal_direction` and scale with its length, so a face's
  `vector` goes straight in.
- EulerDG's own: [`boundary_condition_mirror_wall`](@ref), the plan's default slip wall by mirror
  state (Phase 3); the reference states [`freestream`](@ref) and [`IsentropicVortex`](@ref).
- Phase 4: the characteristic (Riemann-invariant) farfield, which Trixi does not have.
"""
module Equations

using LinearAlgebra: norm
using StaticArrays: SVector

include("trixi.jl")
include("walls.jl")
include("states.jl")

export AbstractEquations, AbstractCompressibleEulerEquations, CompressibleEulerEquations2D
export nvariables, prim2cons, cons2prim, density, velocity, pressure
export flux, flux_central, FluxPlusDissipation, DissipationLocalLaxFriedrichs, FluxLaxFriedrichs
export flux_lax_friedrichs, flux_hllc, max_abs_speed, max_abs_speed_naive, max_abs_speeds
export rotate_to_x, rotate_from_x
export BoundaryConditionDirichlet, boundary_condition_slip_wall, boundary_condition_mirror_wall
export freestream, IsentropicVortex

end # module Equations
