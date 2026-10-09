# Phase 4: the inviscid circle, at low resolution

Companion to the plan's Phase 4 checklist. By decision on Oct 8, 2026, this phase runs at low
resolution only: no implicit solver yet, and no high-resolution case without a fresh decision.
Part 1 records the choices and findings, part 2 the gate results, and part 3 the decisions now
open.

New code:

- `src/Equations/farfield.jl`: `BoundaryConditionFarfield`.
- `src/Post`: `wall_forces`, `surface_cp`, `entropy_error`, `RunLog`, and Cp and entropy-error
  fields in `write_solution_vtk`.
- `face_trace`, in the semidiscretization.
- A mirrored `circle_points`, and a fix to the cut builder's canonical points.
- `examples/circle.jl`.

Tests: `test_circle.jl`, the circle helpers in `test/flows.jl`, and
`scheduled/test_phase4_gate.jl`. The commit tier has 766 checks, and the scheduled tier 829.

## 1. Choices and findings

1. **The farfield.**
   - The Riemann invariants along the outward normal give the boundary's normal velocity and sound
     speed. The entropy and the tangential velocity are taken upwind.
   - Supersonic inflow takes the freestream, and supersonic outflow the interior state.
   - The boundary state goes through the surface flux, so the freestream passes unchanged.
2. **Forces, Cp and entropy.**
   - Forces integrate `(p - p∞)` from the interior trace at the wall Gauss points, against the face
     vectors. A uniform pressure gives zero force to round-off.
   - Cp is reported at every wall Gauss point.
   - The entropy error `(s - s∞)/s∞` is integrated with the degree-4 rules.
   - `RunLog` is a `solve_steady!` callback that writes a CSV of the residual and the forces.
3. **A cut-builder fix.**
   - Vertex 35 of the 51-gon (240°, with h = R/2) landed 6e-17 inside a cell, right beside a
     crossing. Snapping to the cell's node lines then left two copies of nearly the same point,
     and the polygon folded back on itself.
   - Loop points now snap only to the cell's own perimeter points: the cutter's interval ends,
     built canonically. A vertex within round-off of a crossing merges into it, and every other
     vertex is kept as the cutter placed it.
   - A regression test covers the case.
4. **Symmetry.** `circle_points` mirrors its lower half from its upper half bitwise. A circle
   centred on a dyadic grid line then gives a mesh that is symmetric to round-off.
5. **Quadrature asymmetry.**
   - Ear clipping does not triangulate mirrored fragments alike. With the default degree-2 volume
     rules, that alone gave cl = -6.8e-4 at 8 cells per diameter.
   - The degree-4 switch brings it to 1.5e-7 (9e-9 at 16 cells per diameter), leaves drag and
     entropy unchanged, and costs about 5% at 8 cells per diameter.
   - At 16 cells per diameter it also cut the iterations from 63,000 to 20,000.
   - The circle runs use degree 4, while the plan's default stays 2: see part 3.
6. **4 cells per diameter has no steady state.** The residual fell for about 6,000 iterations,
   then grew three orders and settled into a limit cycle, with cl swinging ±0.025: a spurious
   separated wake, unsteady in pseudo-time.
7. **8 and 16 cells per diameter converge.**
   - Entropy is generated at the rear stagnation point and carried into a wake.
   - Spurious drag falls at order 1.89, the entropy error at 2.08 and the wake at 2.47.
   - With Rusanov, the drag doubles and convergence is 2.7 times faster. The two slip walls agree
     to three digits.
8. **Stagnation Cp is set by the domain, not the grid.** It is 1.112 with the far boundary at
   6 radii and 1.070 at 8 radii, against the isentropic 1.023. The excess scales like (R/L)^2, the
   blockage of a boundary too close. The plan's check needs L of 20 radii or more, or a farfield
   correction; on a uniform grid that is expensive.
9. **The sweep.**
   - Slivers down to 1e-12 converge in about the same number of iterations as the symmetric case
     (25,600 against 27,500), so safeguard rungs 2 and 3 are not needed at this resolution.
   - Shifted placements carry lift: cl from -0.15 to +0.16 at 8 cells per diameter. The
     under-resolved wake picks an asymmetric steady state, so the plan's "lift zero on every
     shift" needs finer grids.
10. **Cost.**
    - About 600N iterations on an N x N grid with the degree-2 rules; with degree 4 the 16-cell run
      took about 200N.
    - The gate's 8 and 16 cells per diameter took 140 s and 407 s, with the other gates running
      alongside. The Phase 4 gate is about 20 of the scheduled tier's 34 minutes.
11. **No run-configuration format yet.** The sweep and the example use Julia keyword arguments,
    which is enough at this scale. Raised in part 3.

## 2. Gate results

The circle at M 0.3, on [-3, 3]^2, with the degree-4 rules:

| Cells per diameter | Iterations | cl | cd | Stagnation Cp | Cp asymmetry | Entropy L2 | Wake entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 27,546 | -1.5e-7 | 5.48e-2 | 1.112 | 2.3e-7 | 1.12e-2 | 3.92e-3 |
| 16 | 20,288 | -9.2e-9 | 1.48e-2 | 1.109 | 1.3e-8 | 2.64e-3 | 7.09e-4 |
| Order | | | 1.89 | | | 2.08 | 2.47 |

Grid-shift sweep at 8 cells per diameter:

| Placement | Smallest volume fraction | Iterations | Entropy L2 | cd | cl |
| --- | --- | --- | --- | --- | --- |
| shift (0.25, 0.07) | 7.4e-6 | 23,356 | 1.07e-2 | 5.38e-2 | -3.7e-2 |
| shift (0.01, 0.64) | 1.5e-5 | 31,992 | 1.07e-2 | 6.17e-2 | -1.5e-1 |
| shift (0.76, 0.21) | 4.0e-2 | 27,769 | 1.07e-2 | 4.10e-2 | -1.8e-2 |
| shift (0.52, 0.78) | 4.6e-3 | 29,793 | 1.06e-2 | 4.51e-2 | +1.6e-1 |
| node 1e-3 h outside | 1.0e-6 | 25,659 | 1.11e-2 | 4.55e-2 | -5.5e-2 |
| node 1e-6 h outside | 1.0e-12 | 25,632 | 1.11e-2 | 4.57e-2 | -5.6e-2 |
| node 1e-8 h outside (sliver dropped) | 1.2e-7 | 25,632 | 1.11e-2 | 4.57e-2 | -5.6e-2 |

## 3. Decisions now open

- **The full circle gate** needs finer grids (32 to 64 cells per diameter) and a far boundary at
  20 radii or more. On uniform grids with explicit steps that is roughly an hour or more per case.
  The options are the implicit solver (Phase 6), stretched grids (deferred on Oct 8), or a
  farfield correction.
- **The default volume rule for cut cells.** Degree 4 on triangulated fragments removes the
  triangulation asymmetry and, at 16 cells per diameter, sped convergence; the plan's default is 2.
- **The run-configuration format**, deferred to this phase on Oct 8. It is not built yet.
