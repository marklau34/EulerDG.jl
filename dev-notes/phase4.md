# Phase 4: the inviscid circle, at low resolution

Companion to the plan's Phase 4 checklist. By decision on Oct 8, 2026, this phase runs at low
resolution only: no implicit solver yet, and no high-resolution case without a fresh decision.
Part 1 records the choices and findings, part 2 the gate results, and part 3 the decisions that
were open after them. Part 4 is the full-resolution study of Oct 9, run against the restated gate,
part 5 the decisions it leaves, and part 6 the nested far field built to settle the first of them.

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

## 4. The full-resolution study (Oct 9)

On Oct 9 the gate was restated, since a shifted grid is not symmetric and its asymmetry is
discretization error, not round-off (see the plan's Phase 4 section). The study is
`examples/circle_gate.jl`: 28 runs, 3.8 hours on 16 threads, saved with their states and run logs
to out/circle_gate/. The setup is as in part 2.

Three measurements are new:

- The stagnation Cp is read from the DG trace exactly where the ray from the centre meets the
  wall. The largest Gauss-point value misses the peak by up to 0.005 at 8 cells per diameter.
- The top-bottom and fore-aft Cp asymmetries compare each wall Gauss point with the trace at its
  mirror image on the wall. That works on shifted grids too.
- The wake entropy is the mean |s - s∞|/s∞ of the cell averages one to three diameters behind the
  centre, within half a diameter of the axis.

**A. Domain study, 8 cells per diameter, symmetric placement.** Iterations barely grow with the
domain, so larger domains cost only in proportion to their size.

| Far boundary | Elements | Iterations | Stagnation Cp | Excess | cd |
| --- | --- | --- | --- | --- | --- |
| 6 radii | 2,272 | 27,546 | 1.1173 | 0.0946 | 5.48e-2 |
| 8 radii | 4,064 | 27,609 | 1.0750 | 0.0523 | 4.12e-2 |
| 12 radii | 9,184 | 27,809 | 1.0456 | 0.0229 | 3.37e-2 |
| 24 radii | 36,832 | 29,150 | 1.0277 | 0.0050 | 3.02e-2 |

- The excess falls at order 2.00 in 1/L and extrapolates to 1.0217, 0.09% below the isentropic
  1.0227.
- The 8-cell grid reads 0.0055 high, against the 64-cell grid at 6 radii. Adding that gives
  1.0163, 0.63% low, if the grid's error does not depend on the domain.
- The far boundary also accounts for much of the drag at 8 cells per diameter.

**B. Order study, far boundary at 6 radii, symmetric placement:**

| Cells per diameter | Iterations | cl | Top-bottom | Fore-aft (order) | cd (order) | Entropy L2 (order) | Wake (order) | Stagnation Cp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 27,546 | -1.5e-7 | 2.3e-7 | 0.70 | 5.48e-2 | 1.12e-2 | 3.92e-3 | 1.1173 |
| 16 | 20,288 | -9.2e-9 | 1.3e-8 | 0.30 (1.20) | 1.48e-2 (1.89) | 2.64e-3 (2.08) | 7.09e-4 (2.47) | 1.1116 |
| 32 | 39,293 | -9.8e-9 | 1.4e-8 | 0.090 (1.76) | 1.14e-2 (0.37) | 5.70e-4 (2.21) | 1.24e-4 (2.51) | 1.1113 |
| 64 | 129,938 | -7.2e-9 | 1.0e-8 | 0.063 (0.52) | 1.35e-2 (-0.24) | 1.21e-4 (2.24) | 2.15e-5 (2.53) | 1.1119 |

- The drag stops falling at about 0.012, and the fore-aft asymmetry slows with it.
- The rear stagnation Cp climbs towards the front value: 0.48, 0.86, 1.02, 1.08 against 1.11.
- The 64-cell run took 37 minutes.

**C. Grid-shift sweep, far boundary at 6 radii.** Sliver placements agree with one another to
three digits, so sliver size does not matter. Entropy and wake on shifted grids match the symmetric
placement within 7%. Drag matches within 11% at 32 cells per diameter, and more loosely on the
coarser grids.

| Cells per diameter | Iterations | Largest \|cl\| | Top-bottom | Fore-aft | cd |
| --- | --- | --- | --- | --- | --- |
| 8 | 23,000-32,000 | 0.157 | 0.62 | 1.17 | 4.1e-2 to 6.2e-2 |
| 16 | 134,000-190,000 | 0.135 | 0.26 | 0.38 | 8.5e-3 to 1.3e-2 |
| 32 | all stopped at 400,000 | 0.079 | 0.12 | 0.13 | 1.06e-2 to 1.27e-2 |

The lift of each placement, at 8, 16 and 32 cells per diameter:

| Placement | 8 | 16 | 32 |
| --- | --- | --- | --- |
| shift 1 | -0.037 | -0.019 | -0.010 |
| shift 2 | -0.145 | +0.127 | -0.038 |
| shift 3 | -0.018 | -0.135 | -0.009 (still drifting, toward -0.009) |
| shift 4 | +0.157 | +0.011 | +0.079 (toward +0.081) |
| node 1e-3 h outside | -0.055 | +0.013 | +0.033 (toward +0.035) |

At 32 cells per diameter the residuals stopped between 1.7e-10 and 6.6e-8.

**Against the gate:**

| Criterion | Result |
| --- | --- |
| 1. Residual 10 orders on every run | Not met: 21 of 28 runs. The seven shifted runs at 32 cells per diameter stopped at the 400,000-iteration cap. |
| 2a. Symmetric lift and top-bottom asymmetry at round-off | Met: at most 1.5e-7 and 2.3e-7 on all 8 symmetric runs |
| 2b. Shifted lift and top-bottom asymmetry fall with h, \|cl\| < 1e-3 at the finest | Not met. Top-bottom falls at about first order, and the largest \|cl\| falls slowly (0.157, 0.135, 0.079), but no single placement falls steadily. \|cl\| is 0.079 at 32 cells per diameter. |
| 3. Fore-aft asymmetry falls with h | Met, but slowing (order 0.52 from 32 to 64) |
| 4. Extrapolated stagnation Cp within 0.5% | Borderline: -0.09% from the domain study alone, -0.63% with the grid's share added |
| 5. Drag and entropy L2 at about second order | Entropy met (2.08, 2.21, 2.24). Drag not met (1.89, 0.37, -0.24). |
| 6. Wake at about second order, below 1e-4 at the finest | Met: orders 2.47-2.53, 2.1e-5 at 64 cells per diameter |

**Two causes.**

1. **The far boundary.** The characteristic farfield is upwind by design: it takes three
   characteristics from the freestream at inflow and one at outflow. At 6 radii it biases the
   steady flow fore-aft by roughly (R/L)². That leaves a drag floor of about 0.012 that no grid
   refinement removes, slows the fore-aft convergence, and raises the stagnation Cp. The domain
   study shows the stagnation-Cp excess falling exactly as L^-2.
2. **The circulation.** The circle has no sharp trailing edge, so the steady Euler problem does not
   fix its circulation: every circulation gives a valid potential flow. A shifted grid picks one
   through its own asymmetric dissipation. That makes the lift erratic in h, and changing it means
   shedding vorticity through the wake, the slowest mode under explicit local steps:
   134,000-400,000+ iterations against 20,000-130,000 on the symmetric placement. The airfoil's
   trailing edge fixes the circulation by the Kutta condition, so this is particular to the circle.

## 5. Decisions now open, after the study

- **The far boundary.** Either a farfield correction, freestream plus the compressible doublet
  (and, for Phase 5, the point vortex), at the boundary state, then a rerun of B at 6 radii (about
  an hour). Or an order study on a domain of 24 radii: 16 cells per diameter is 147,000 elements,
  about 40 minutes; 32 cells per diameter is 590,000 elements, several hours.
- **The shifted-grid lift criterion.** Either report it for the circle and move the lift gate to
  the airfoil, keeping on shifted grids convergence and entropy, wake and drag close to the
  symmetric placement's. Or pursue it on finer shifted grids, which takes hours per run with
  explicit steps and probably the implicit solver.
- **The iteration cap.** At 32 cells per diameter the shifted runs need about 600,000 iterations
  to converge 10 orders, with the lift settled to about 1e-4 well before then.

## 6. The nested far field (Oct 9)

The far boundary was settled as option 3: `nested_cut_mesh`. The body is cut on a uniform finest
box by the same polyline clipping as before. The box is wrapped in a CartesianMeshes `NestedGrids`
chain of coarser levels out to the far field, each level doubling the cell size, each ring at least
`buffer` (16) of its own cells thick.

How it is built:

- **Hanging nodes.** A coarse cell meeting finer cells takes their shared node as a vertex on its
  side, so the contract's builder pairs the faces as usual.
- **One formula for every vertex.** Every vertex is a finest-level node, placed by the finest
  patch's own `get_node`, so vertices shared between levels are bitwise one point.
- **Quadrature.** `_is_box` now accepts rectangles with collinear vertices, so coarse cells keep
  the tensor Gauss rule.
- **Symmetry.** An `origin` keyword aligns every level's cells to a point. Without it the lattice
  hangs off the finest box's corner and the rings are not symmetric about the circle's axis. That
  gave lift of 1e-6 and, at 16 cells per diameter, 144,000 iterations instead of 109,000.

The circle at M 0.3, far boundary at 70 radii or more (the domain came out at about 125 radii),
centred on the origin:

| Cells per diameter | Elements | Iterations | cl | cd | Stagnation Cp | Rear Cp |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 11,620 | 42,855 | -8.2e-8 | 2.99e-2 | 1.0246 (+0.19%) | 0.434 |
| 16 | 19,392 | 108,986 | -4.1e-9 | -6.8e-4 | 1.0209 (-0.17%) | 0.787 |

- **The drag floor is gone.** At 16 cells per diameter cd is -6.8e-4, against 1.48e-2 at 6 radii.
- **Stagnation Cp needs no extrapolation.** It is within 0.2% of isentropic on both grids.
- **Lift is round-off again.**
- **Convergence is slower.** The iterations grow 1.6x at 8 cells per diameter and 5.4x at 16 over
  the 6-radius domain. The close boundary probably damped a slow mode that the far one does not.

NACA 0012 at M 0.5 and 1.25 degrees, with the far boundary at 50 chords or more
(`examples/naca0012_flow.jl`):

| Cells per chord | Elements | Iterations | cl | cl at 3.5 chords |
| --- | --- | --- | --- | --- |
| 16 | 16,515 | 81,912 | 0.1188 | 0.1149 |
| 32 | 28,079 | 215,983 | 0.1593 | 0.1487 |

The thin-airfoil estimate with the compressibility correction is about 0.158. At 32 cells per
chord the rings hold 20,800 of the 28,100 elements, so a thinner `buffer` would halve their cost.

`examples/circle_gate.jl` has the nested order study as set N (8 to 64 cells per diameter). Only
8 and 16 have been run, by hand. 32 and 64 are estimated at 20 minutes and 1.5 hours.
