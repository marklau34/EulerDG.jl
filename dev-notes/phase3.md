# Phase 3: cut-cell DG and walls

Companion to the plan's Phase 3 checklist. No brief was written for this phase. Part 1 records the
choices, part 2 the gate results, and part 3 what Phase 4 inherits.

New code:

- `src/Equations/trixi.jl`: `flux_hllc`, `rotate_to_x`, `rotate_from_x` and
  `boundary_condition_slip_wall`, copied from Trixi 0.17.13.
- `src/Equations/walls.jl`: `boundary_condition_mirror_wall`.
- `src/Semidiscretization`: source terms and `local_dt!`.
- `src/PseudoTime`: local-step SSP-RK3, `residual_norms` and `solve_steady!`.
- `src/CutMesh/halfplane.jl`: `halfplane_mesh`.

Tests: `test/flows.jl` (the manufactured solution and its source), `test_walls.jl` and
`scheduled/test_phase3_gate.jl`. The commit tier has 740 checks, and the scheduled tier 785.

## 1. Choices

1. **Two slip walls.**
   - The default is the mirror state through the surface flux. It reverses the normal *momentum* in
     conservative variables, leaving density and energy unchanged, so for flow tangent to the wall
     the outer state is the inner one bitwise.
   - The plan's pressure-only alternative is Trixi's `boundary_condition_slip_wall`, whose pressure
     is the exact solution of the 1D Riemann problem against the wall.
   - With HLLC the two gave the same manufactured-solution errors to four digits on every grid.
2. **HLLC** is copied from Trixi 0.17.13, as are the rotations the slip wall uses.
   `dev/check_trixi.jl` checks all the copies, with the largest difference 5.4e-15.
3. **Local steps.**
   - `local_dt!` is the plan's formula per element. It shares one face-speed pass with `max_dt`,
     which is now just the smallest local step.
   - The steps are frozen at the start of each iteration and kept across the three SSP-RK3 stages,
     with each element's step folded into the update.
   - `solve_steady!` reads the residual from the first stage's right-hand side, so monitoring costs
     no extra evaluation.
4. **Monitors.** The residual of an element is its integrated cell-average density residual,
   |K| du, the net mass-flux imbalance. Unlike du, it does not blow up on slivers. The monitors are
   its L2 norm relative to the first iteration's, and the largest value with its element. The plan
   lists these under Phase 4; the steady tests needed them now, and Phase 4 adds the run log.
5. **CFL ramp.** The ramp is linear, from 0.1 to the target (2) over the first 200 iterations, as
   the plan suggests. Started from uniform flow, the coarse manufactured solution converges with it.
6. **Source terms** are Trixi-style `source_terms(u, x, t, equations)`, evaluated at the volume
   quadrature points. With none, the pass compiles away.
7. **`halfplane_mesh`.** The plan's tests need a wall running right across the domain, and the
   polygon cutter's body must stay clear of the grid's edge. So this is a small second builder:
   - each cell is clipped against the line;
   - each crossing is computed once per edge from the edge's own coordinate, and snapped onto a
     node within round-off;
   - fragments below `drop_area` are dropped, and the faces they shared become wall, as the
     cutter does.
8. **The manufactured solution.** It is defined in the wall's frame, with the normal velocity
   proportional to the distance from the wall. Both slip walls are therefore consistent with it,
   and the test measures the wall treatment, not just the interior. Its source `∇ · F(U)` is taken
   by complex step, exact to round-off with no automatic-differentiation dependency. The box's
   sides get the exact solution as a Dirichlet condition.
9. **A finding: the discrete wall's own direction.** Uniform flow along the wall stays uniform to
   round-off, except in a fragment whose wall face is very short.
   - A wall face 1.1e-7 long has endpoints rounded to their last bits, which turns its direction by
     about 5e-10 rad. The uniform flow then crosses the discrete face by that much, and the
     fragment's steady state turns to follow it.
   - The deviation is about 1e-10, confined to the fragment, and reached within 10 iterations.
   - The vertex-derived face vectors are kept, because they give exact closure, which is what
     freestream preservation rests on. The gate bounds each element's deviation by round-off times
     h over its shortest face.
10. **Cost.** Explicit local steps carry information about one element per iteration, so the
    iteration count doubles with each refinement: about 2,000, 4,300 and 8,700 on the 16, 32 and
    64 grids. The 64 x 64 solve takes about 90 s, and the Phase 3 gate about 5.5 minutes of the
    scheduled tier's 7.5. The plan's implicit solver (Phase 6) is the cure.
11. **The cross-check environment.** `dev/` no longer loads EulerDG. Trixi 0.17.13's tree and the
    Gmsh binaries under MeshLibrary cannot be resolved together, the entanglement Phase 0 warned
    about. The script includes the copied files into a module of its own instead.
12. **A pitfall for scripts.** A source or exact solution written as a closure over non-constant
    globals was about 1000 times slower, because of dynamic dispatch at every quadrature point. The
    tests use callable structs; scripts should do the same or use `const` globals.

## 2. Gate results

**Uniform flow along a tilted wall.** The grid is 32 x 32, with 300 local steps, both
wall-and-flux pairs, and 8 line offsets plus 2 sliver placements per angle:

| Angle | Smallest volume fraction | Regular elements | Any element | Scaled by h / shortest face |
| --- | --- | --- | --- | --- |
| 17° | 1.8e-12 | 6.2e-14 | 1.1e-10 | 1.2e-14 |
| 33° | 1.1e-12 | 1.5e-13 | 2.5e-10 | 1.3e-14 |
| -24° | 1.3e-12 | 5.7e-14 | 7.0e-11 | 1.4e-14 |

**Manufactured solution, L2 density error.** The wall is at 17°, and each steady state is
converged to 1e-9:

| Grid | Mirror wall, HLLC | Pressure-only wall, HLLC | Mirror wall, Rusanov |
| --- | --- | --- | --- |
| 16 x 16 | 6.741e-4 | 6.741e-4 | 1.046e-3 |
| 32 x 32 | 1.687e-4 (2.00) | 1.687e-4 (2.00) | 2.306e-4 (2.18) |
| 64 x 64 | 4.268e-5 (1.98) | 4.268e-5 (1.98) | 5.527e-5 (2.06) |

## 3. For Phase 4

- **Farfield.** The box's sides still use Dirichlet conditions. The characteristic farfield
  comes next, with the circle at M 0.3.
- **Run log.** The monitors exist; the run log of residual and force histories, written as CSV,
  remains.
- **Iterations.** At about 2N iterations on an N x N grid, the circle's convergence studies will
  be slow on fine grids, and the grid-shift sweep slower still. Budget for it, or bring Phase 6
  forward, as the plan's last open decision allows.
- **Safeguards.** The slivers here (1e-12) needed no safeguard: no rung beyond local stepping was
  used, and nothing stalled. The circle sweep makes the plan's decision on rungs 2 and 3.
