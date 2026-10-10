# Cut-Cell DG Euler Solver — Architecture & Development Plan

Oct 8, 2026 · @Mark · updated Oct 10 to match the code through Phase 5

## Scope and assumptions

Version 1 delivers a converged, steady, subsonic 2D airfoil solution from a P1 modal DG discretization on a Cartesian cut-cell mesh, with local pseudo-time stepping in place of state redistribution. That goal was met in Phase 5. Phase 6 makes it fast, with an implicit pseudo-transient solver.

| Item | Version 1 choice |
| --- | --- |
| Equations | 2D compressible Euler, steady only, subsonic (local Mach < 1) |
| Discretization | Modal DG, P = 1 on every element |
| Geometry | Closed polygons (SDFLibrary `SDFMesh`), cut exactly by CutCellMethods; a fragment's wall is the polygon's own pieces in the cell (Phase 2) |
| Background mesh | Static nested Cartesian grids (Oct 9): the body is cut on a uniform finest box (CartesianMeshes' `CartesianGrid`), wrapped in coarser `NestedGrids` levels out to the far field, with hanging nodes where levels meet; no AMR |
| Small cells | Local time stepping; fragments below 1e-12 of a cell are dropped; no state redistribution, no merging |
| Runtime | Julia, CPU, FP64, threaded with KernelAbstractions kernels (Oct 9) |

Out of scope for version 1: shocks and limiting, curved cut faces, AMR, 3D and GPU. The architecture keeps a clean path to each of them. Implicit solvers were out of scope too, but Phase 6 brings one into version 1.

Assumed reuse: the SDF geometry library, the earlier cut-cell preprocessing code, and Trixi.jl's equation and flux layer and code organization.

## Core design decisions

Local time stepping is sufficient for the small-cell time-step problem because only the steady residual R(U) = 0 matters. Four further decisions make the rest of the small-cell problem manageable.

### 1. Local time stepping replaces state redistribution

The steady solution does not depend on the pseudo-time path. Each element can march with its own stable step, which acts as a diagonal preconditioner. Information still crosses about one element per iteration, so tiny cells add unknowns but do not slow global convergence.

Local time stepping does not fix three things, each handled separately:

- **Mass-matrix conditioning on slivers.** Handled by the per-element basis (decision 3).
- **Weakly controlled gradient modes on slivers.** Handled by an optional P0 fallback (decision 4).
- **Global stiffness of explicit P1 DG.** Accepted through Phase 5; the implicit solver of Phase 6 is the cure. The evidence for needing it:
  - Iterations grow 1.2–3x per refinement: 502,000 for the airfoil at 128 cells per chord.
  - The 64-cell circle on the nested far field settles into a limit cycle instead of a steady state.

### 2. A polygon-DG core; Cartesian structure lives only in the preprocessor

After preprocessing, the solver sees a general mesh of polygonal elements and faces. Full cells and cut fragments run through one code path. Several fragments in one background cell need no special case. Hanging nodes are extra faces on a polygon, so no mortars are needed; the nested far field (Oct 9) already uses them, and a later quadtree would too. The cost is losing tensor-product sum factorization, which is negligible at P1.

This also means Trixi's DGSEM kernels are not reused. They rely on tensor-product LGL nodes with summation-by-parts, which cut cells break. The solver uses the standard weak-form modal DG with explicit quadrature. It borrows Trixi's equation and flux functions and its container layout.

### 3. Per-element basis on the fragment's bounding box

Each element starts from 1, (x − xc)/hx and (y − yc)/hy on its own bounding box. Gram–Schmidt against the element's own quadrature makes the modes orthogonal. Normalizing so that the first mode is 1 gives a mass matrix of |K| times the identity. Mode 0 is then the cell average, which simplifies positivity checks, output and the P0 fallback.

### 4. A small-cell safeguard ladder, used only as needed

1. Nothing beyond local time stepping (the default).
2. Per-element CFL backoff when a cell's residual grows.
3. Drop to P0 on fragments below a volume-fraction threshold (off by default).
4. Treat fragments below round-off size, around 1e-12 of a full cell, as solid. The conservation error is negligible.

Merging stays in reserve and is not planned.

**Status (Oct 10).**

- Rung 4 is built: `drop_area`, 1e-12 of a cell by default.
- Rungs 2 and 3 are not built. Every sweep converged without them: slivers to 1e-12 on the circle and on tilted walls, and the airfoil to 1.8e-5 at 128 cells per chord.
- They wait until a case needs them.

## Software architecture

&#91;embedded content: solver architecture · 7 modules, one mesh contract\]

Everything upstream of PolygonMesh can change (quadtree, curved faces, 3D) without touching the solver, and nothing downstream ever sees a Cartesian index. Each box is one module of a single Julia package, with its own tests.

### Data layout

- State and residual are 4 × 3 × elements arrays, as in Trixi.
- Element and face data are struct-of-arrays: geometry, quadrature points and weights, normals, basis coefficients, neighbour ids, face kind.
- The right-hand side runs two passes: the flux at every face Gauss point, into a face buffer, then each element's volume integrals, face integrals and division by |K|.
- The face pass is a two-pass gather rather than a scatter: each element sums its own faces' fluxes, in increasing face order. No two threads write to the same element, and the sums are bitwise those of a serial loop whatever the thread count. **Done (Oct 9)**: the face, element, time-step, positivity and stage loops are KernelAbstractions kernels, threaded on the CPU backend; the same kernels serve a later GPU port.

## Cut-cell preprocessing requirements

The preprocessor's job is to hand the solver a watertight polygon mesh; the existing cut-cell code is adapted to that output contract rather than rewritten.

### Output contract

| Object | Fields |
| --- | --- |
| Element | Polygon vertices, area, centroid, bounding box, volume fraction, parent background cell |
| Face | Endpoints, length, unit normal, left and right element ids, boundary name (none on interior faces; for example wall, farfield) |
| Report | Counts by cell type, minimum volume fraction, volume-fraction histogram, closure error, lost-feature flags |

### Rules

- Every fluid fragment of a background cell becomes its own element.
- Each partial edge between two fluid fragments is one face, with split points computed once and shared bitwise by both sides.
- Wall crossings on cell edges come from exact intersection of the body's polygon with the edge (Phase 2), not a root-find on the SDF.
- All lengths, normals and areas derive from one vertex list per element. Then the face vectors of every element sum to zero at round-off.
- That closure property is what makes uniform flow along a straight wall preserved exactly. Check it in the report on every run.

### Trailing-edge pitfalls

The trailing edge is where a cut-cell airfoil usually goes wrong, and the errors change circulation rather than crash.

- **Lost features.** Where the airfoil is thinner than a cell, every vertex can sit in fluid while the body passes through. Vertex-based classification then misses the body and the flow leaks across the trailing edge. Flag any cell whose center SDF value is below its half-diagonal while all vertices share a sign. The preprocessor stops with an error on these rather than continuing; the fix is local refinement.
- **Split cells.** A cell straddling the trailing edge holds an upper and a lower fragment with a solid sliver between. These must be two elements with no face between them.
- **Truncated tip.** One straight segment per fragment would cut the sharp tip off, leaving a blunt base about one cell wide. **Not an issue here:** the exact polygon cut keeps the polygon's own wall pieces in each fragment, so the tip cell is a nonconvex fragment with the exact edge as a vertex. Phase 5's trailing-edge reports confirm it on every grid.

The Kutta condition needs no explicit enforcement in an Euler solver. Numerical dissipation at a sharp, non-leaking trailing edge sets the circulation.

## DG discretization

The discretization is the standard weak form, evaluated by explicit quadrature on every element:

```latex
\int_K \phi_i \, \partial_t U \, dx \;-\; \int_K \nabla \phi_i \cdot F(U) \, dx \;+\; \oint_{\partial K} \phi_i \, \hat{F}(U^-, U^+, n) \, ds \;=\; 0
```

| Component | Choice | Notes |
| --- | --- | --- |
| Basis | 3 modes per variable, built per element (Core decision 3) | State array sized 4 × 3 × elements, Trixi-style |
| Volume quadrature | Ear-clipping triangulation, 3-point degree-2 rule per triangle (`volume_degree=2`, the default) | Ear clipping also covers the nonconvex trailing-edge wedge. The circle and airfoil studies use the 6-point degree-4 rule (`volume_degree=4`): it removes the ear-clipping asymmetry of mirrored fragments (Phase 4) |
| Full-cell quadrature | Tensor 2 × 2 Gauss on every rectangle, hanging nodes included (`box_degree=3`, Oct 9) | Same kernel as cut cells; 3 × 3 is `box_degree=5`, which changed the circle's results only past their fifth digit |
| Face quadrature | 2-point Gauss per segment | Normals stored per quadrature point so curved faces drop in later |
| Interior flux | HLLC for production, Rusanov for debugging (Phase 3) | Fluxes are plain functions of (uL, uR, n), as in Trixi |
| Slip wall | Mirror state through the same Riemann flux | Pressure-only wall flux kept as an alternative to compare |
| Farfield | Characteristic (Riemann-invariant) state | Boundary at 100 chords or more via the nested far field (Oct 9); point-vortex correction not yet built |
| Limiting | None for subsonic flow | Positivity check at quadrature points aborts with the element id |

Straight cut faces are consistent with second-order accuracy at P1. They still produce spurious entropy at the wall, so entropy error is tracked from the first airfoil run. Bassi and Rebay showed boundary representation dominates DG accuracy at higher order, which is why curved faces are on the backlog.

## Pseudo-time marching with local time stepping

Each element advances with its own step, computed from its own area and face fluxes, so a sliver's tiny step never limits its neighbours.

```latex
\Delta t_K = \mathrm{CFL} \cdot \frac{|K|}{(2p+1) \sum_{f \in \partial K} \lambda_f \, |f|}, \qquad \lambda_f = \max_{\text{face quad. pts, both sides}} \left( |v \cdot n| + c \right)
```

The face sum includes wall faces. A thin sliver along a long wall face therefore gets a step that scales with its thickness, which is the correct limit. Calibrate the CFL number on full cells first. **Calibrated (Oct 9):** 3 for steady runs with local steps, held on the nested circle and airfoil and on a 1e-12 sliver; 3.5 broke a cut cell of the circle. Time-accurate runs keep 2.

### Implementation

- Freeze the step field at the start of each iteration and keep it across the Runge–Kutta stages.
- Fold the steps into the right-hand side: integrate dU/dτ = −D M⁻¹ R(U) with D = diag(Δt\_K) and dτ = 1. Any fixed-step integrator then works unchanged, including DifferentialEquations.jl.
- Hand-roll SSP-RK3 or a low-storage scheme in version 1. It is about 20 lines and keeps full control of stages and monitoring.
- Start from freestream with a CFL ramp from about 0.1 to the target over the first few hundred iterations.
- Optional startup: converge P0 first, then lift to P1. This also prototypes a P1-to-P0 multigrid. Not built.

### Convergence monitoring

- L2 norm of the cell-average density residual, normalized by its first value.
- Maximum residual over elements and the element that holds it. A stuck small cell shows here first.
- Lift, drag and moment histories.
- Stop at 10 orders of residual drop, or a force plateau to 1e-6 over 1,000 iterations.
  - Only the residual stop is built.
  - The force-plateau stop is still to do. It would have ended the shifted circle runs and the 64-cell runs long before their residual, once their lift had settled.

### Acceleration path

Explicit P1 DG is slow for steady problems; expect many thousands of iterations on airfoil grids. The planned cure is implicit pseudo-transient continuation with the same local steps, and switched evolution relaxation for CFL growth. Element and face Jacobians come from ForwardDiff, solved with block-Jacobi or ILU-preconditioned GMRES. Once implicit, small cells are no longer a time-step issue at all. P1-to-P0 multigrid is the explicit alternative.

## Postprocessing and diagnostics

For a subsonic inviscid airfoil, entropy error and spurious drag are the primary accuracy measures; lift and Cp are compared against reference data.

| Output | How | Used for |
| --- | --- | --- |
| Lift, drag, moment | Integrate p n over wall faces at their quadrature points, using the interior trace; the pitching moment nose-up positive (Oct 10) | Convergence and validation |
| Surface Cp | Values at wall quadrature points, sorted by x/c | Comparison with reference Cp |
| Entropy error | (s − s∞)/s∞ field and its domain L2 norm | Primary accuracy metric; should fall at about second order |
| Spurious drag | Drag coefficient, exactly zero in the continuous problem | Second accuracy metric |
| Field output | Each element written as a VTK polygon with its cell averages; P1 values at its vertices still to do | Visual checks of cut cells |
| Run log | Residual and force history per iteration, CSV | Convergence plots |

Cut-cell surface Cp is naturally scattered from cell to cell. Plotting per quadrature point shows that scatter honestly, which is itself a quality check.

## Development plan

Seven phases take the solver from an uncut DG core to a validated subsonic airfoil and then to fast convergence; no phase starts until the previous gate passes. Phases 4 and 5 were closed by decision on Oct 10, with their open items recorded in each section.

&#91;embedded content: development roadmap · 7 phases, each with a gate\]

Phases 1 and 2 are independent and can run in parallel. The circle comes before the airfoil because it isolates small-cell behaviour from trailing-edge problems.

### Phase 0 · Scaffolding

- [x] Create the package with modules for geometry, cut mesh, mesh types, basis, quadrature, equations, semidiscretization, pseudo-time and postprocessing
- [x] Define the PolygonMesh element and face types from the output contract
- [x] Settle the Trixi.jl dependency question: copy its equation and flux functions (Oct 8)
- [x] Set up the test harness (the run-configuration format moved to Phase 4 on Oct 8)

### Phase 1 · DG core on uncut grids

- [x] Build a PolygonMesh straight from an uncut grid: periodic and uniform (stretching deferred on Oct 8)
- [x] Basis construction, with a unit test that the mass matrix equals |K| times the identity
- [x] Quadrature: segment Gauss, triangle rule, ear clipping, with exactness tests on polygons
- [x] Right-hand side with volume, interface and boundary passes and the Rusanov flux
- [x] SSP-RK3 with a global step for the unsteady vortex test
- [x] Isentropic vortex convergence study and freestream preservation test

**Gate passed (Oct 8).** The L2 density error falls at orders 2.08–2.21 on square cells and 2.15–2.19 with dx = 2dy. Freestream stays uniform to 1e-14 through t = 5, cut fixtures included.

### Phase 2 · Cut-mesh preprocessor

- [x] Adapt the existing cut-cell code to emit PolygonMesh
- [x] Wall crossings, shared split points, multi-fragment cells. The crossings come from exact polyline intersection rather than an SDF root-find, because CutCellMethods takes polygons.
- [x] Lost-feature detection that stops with an error
- [x] Validation report and fluid-area convergence on the circle and NACA 0012
- [x] Grid-shift generator that moves the body by sub-cell offsets to create tiny fragments

**Gate passed (Oct 8).** The fluid area converges at order 2.00 on the circle and on NACA 0012, and every cut is exact for its polygon to 1e-12. A sweep of 114 cut meshes (16 shifts and 3 sliver shifts on each of 3 grids per body) gave valid meshes throughout. The smallest fragments reached about 1e-12, and freestream was preserved to 2e-15. A fragment keeps the polygon's own wall pieces, so the trailing edge is exact: a nonconvex fragment, with no truncated tip.

### Phase 3 · Cut-cell DG and walls

- [x] Slip wall by mirror state, plus the pressure-only alternative (Trixi's, with the pressure from the exact 1D Riemann problem)
- [x] Local time-step formula, frozen across Runge–Kutta stages
- [x] Uniform flow along a straight wall tilted off the grid axes
- [x] Manufactured solution on a domain cut by a tilted straight wall
- [x] Add the HLLC flux

**Gate passed (Oct 8).** The manufactured solution on the unit square cut by a wall at 17° converges at second order with both slip walls and both fluxes: orders 1.98–2.18 on the 16, 32 and 64 grids. Uniform flow along walls at three angles stays uniform under local steps, with slivers down to 1e-12. Regular elements stay within 1.5e-13. A sliver whose wall face is 1e-7 long deviates by up to 2.5e-10, because that face's own direction carries its endpoints' rounding. The explicit solve needs about 2N iterations on an N × N grid.

### Phase 4 · 2D inviscid circle

**Closed (Oct 10), by decision, at 32 cells per diameter.**

- Passed on the nested far field (70 radii or more) at 8, 16 and 32 cells per diameter:
  - every run converges 10 orders;
  - lift and the top-bottom asymmetry stay at round-off;
  - the stagnation Cp is within 0.2% of isentropic;
  - the fore-aft asymmetry falls with h;
  - the entropy error falls at orders 2.25 and 2.38;
  - the wake falls at order 2.5.
- Not closed:
  - The drag's order: it crosses zero between 8 and 16 cells per diameter.
  - The wake threshold: 1.19e-4 at 32 cells per diameter, against 1e-4.
  - Shifted-grid lift: the inviscid circle leaves its circulation free, so each shifted grid picks one. The lift gate moves to the airfoil, whose trailing edge fixes the circulation.
- At 64 cells per diameter the explicit solve does not reach a steady state on the nested far field. It settles into a symmetric, global limit cycle (drag swinging about 1, averaging -0.0005 to +0.002) with the finest box ending 1.5 or 3 diameters out. The uniform 6-radius grid, with the same cells near the circle, converges. This is recorded for Phase 6.
- Details are in `dev-notes/phase4.md`.

- [ ] Run-configuration format, moved from Phase 0 (Oct 8); the grid-shift sweep is its first user. Not built yet: the sweeps use keyword arguments. Moved to Phase 5.
- [x] Characteristic farfield boundary
- [x] Convergence monitors: L2 and maximum residual with its location, plus a CSV run log
- [x] Subsonic inviscid circle at M = 0.3, below the circle's critical Mach number, at low resolution (8 and 16 cells per diameter)
- [x] Grid-shift sweep at 8 cells per diameter, logging minimum volume fraction, iterations and entropy error
- [x] Decide from the sweep whether safeguard rungs 2 and 3 are needed: not at this resolution, since slivers to 1e-12 converge in the usual number of iterations

**Low-resolution gate passed (Oct 8).** At 8 and 16 cells per diameter, on [−3, 3]² with the degree-4 volume rules, both grids converge 10 orders. Lift (≤ 1.5e-7) and the Cp asymmetry (≤ 2.3e-7) are zero but for the triangulation of mirrored fragments. Spurious drag falls at order 1.89, the entropy error at 2.08 and the entropy wake at 2.47. Not yet met:

- The stagnation Cp is 1.11 against 1.023, from the far boundary at 6 radii; it scales like (R/L)².
- On shifted grids the lift reaches ±0.16 at 8 cells per diameter, because the under-resolved wake is asymmetric.
- Meeting the full gate needs finer grids and a larger domain. That waits for a decision on resolution, the implicit solver and stretched grids.

**Full-resolution study (Oct 9): gate not passed.** 28 runs, up to 64 cells per diameter and a far boundary at 24 radii; the details are in `dev-notes/phase4.md`.

- Met:
  - lift and top-bottom asymmetry at round-off on every symmetric run, at most 1.5e-7;
  - entropy error at order 2.1–2.2;
  - entropy wake at order 2.5, and 2.1e-5 at 64 cells per diameter;
  - fore-aft asymmetry falling, 0.70 to 0.063.
- Not met:
  - drag levels off at about 0.012 with the far boundary at 6 radii;
  - shifted-grid lift is erratic in h: largest |cl| 0.157, 0.135 and 0.079 at 8, 16 and 32 cells per diameter;
  - the seven shifted runs at 32 cells per diameter stopped at 400,000 iterations, short of 10 orders.
- Borderline: the stagnation Cp extrapolates to 0.09% below isentropic from the domain study, or 0.63% below with the grid's share added.
- Two causes:
  - the upwind characteristic farfield biases the flow fore-aft by about (R/L)²;
  - the circle's circulation is not fixed by the inviscid problem, so a shifted grid picks one.

  The open choices are a farfield correction or a larger domain, and whether the lift gate moves to the airfoil.
- **Far boundary settled (Oct 9): the nested far field** (`nested_cut_mesh`, see the open-decisions table). With the boundary at 70 radii or more, at 8 and 16 cells per diameter:
  - the drag floor is gone (cd -6.8e-4 at 16, against 1.48e-2 at 6 radii);
  - the stagnation Cp is within 0.2% of isentropic without extrapolation;
  - lift is round-off.

  The explicit solve takes up to 5x more iterations on the larger domain.
- **Nested order study (Oct 9, `buffer=8`), 8, 16 and 32 cells per diameter; 64 skipped by decision:**
  - all three converge 10 orders;
  - lift and top-bottom asymmetry stay at round-off;
  - the fore-aft asymmetry falls at orders 1.18 and 1.83;
  - the stagnation Cp is within 0.2% on every grid;
  - entropy L2 falls at orders 2.25 and 2.38;
  - the wake falls at order 2.51 (1.19e-4 at 32, just above the 1e-4 threshold).

  The drag crosses zero between 8 and 16 (+3.0e-2, -6.4e-4, -2.9e-3), so its order is not settled without the 64-cell grid. The shift sweep's lift (2b) is the remaining open criterion, which the far boundary does not affect.

**Circle gate.** The inviscid circle has no circulation and no wake, so any lift, drag or asymmetry is numerical error. As restated on Oct 9, since a shifted grid is not symmetric and its asymmetry is discretization error, not round-off:

- Residual drops 10 orders on every run.
- On the symmetric placement, lift and the top-bottom Cp asymmetry are zero to round-off. On shifted grids both fall with h, and |cl| is below 1e-3 on the finest sweep grid.
- The fore-aft Cp asymmetry falls with h.
- The stagnation Cp, extrapolated to an unbounded domain from a domain study, is within 0.5% of the isentropic value, about 1.023 at M = 0.3.
- Spurious drag and the entropy-error L2 norm fall with h, close to second order.
- The entropy wake falls at close to second order and is below 1e-4 on the finest grid; the VTK output shows it.

The runs (`examples/circle_gate.jl`): a domain study at 8 cells per diameter with the far boundary at 6, 8, 12 and 24 radii; an order study at 6 radii at 8, 16, 32 and 64 cells per diameter; and a grid-shift sweep of seven placements at 8, 16 and 32 cells per diameter.

### Phase 5 · Subsonic airfoil

**Reference case (decided Oct 10).** Vassberg and Jameson, "In Pursuit of Grid Convergence for Two-Dimensional Euler Solutions", J. Aircraft 47(4), 2010: NACA 0012 at M 0.5, α 1.25°.

- **Their geometry:** the standard thickness formula with its chord extended to the sharp trailing edge at x = 1.0089304115, reference chord 1, moment about x = 0.25.
- **Their grids:** O-meshes to 4096², about 150 chords to the far boundary.
- **FLO82 continuum estimates** (their Table 2): Cl 0.180346, Cd 5e-8, Cm -0.002269.

Checklist:

- [x] NACA 0012 with the Vassberg-Jameson sharp trailing edge (`naca4_points(...; trailing_edge=:extended)`), and a trailing-edge report on every grid
- [x] Far field at 50–100 chords: the nested far field (Oct 9), instead of a stretched grid
- [x] Forces, Cp, entropy and VTK output: `examples/naca0012_flow.jl` (Oct 9)
- [x] Three-grid study comparing lift, moment and Cp with the reference: `examples/naca0012_study.jl`
- [ ] Optional point-vortex farfield correction: not done
- [ ] Run-configuration format, moved from Phase 4: moved to the backlog (Oct 10), since keyword arguments have served every study

**Closed (Oct 10), by decision, on 32, 64 and 128 cells per chord.**

| Cells per chord | cl (error) | cm, nose-up positive (error) | cd | Entropy L2 |
| --- | --- | --- | --- | --- |
| 32 | 0.16768 (-7.0%) | -0.00136 (40%) | +1.2e-3 | 7.2e-3 |
| 64 | 0.17058 (-5.4%) | -0.00149 (34%) | -1.3e-3 | 3.5e-3 |
| 128 | 0.17730 (-1.7%) | -0.00204 (10%) | -1.4e-5 | 9.9e-4 |

- **Orders.** From 64 to 128 the lift, moment and entropy errors fall at orders 1.68, 1.77 and 1.82. From 32 to 64 they fell at 0.2–1, while the leading edge was under-resolved.
- **Pressures.** Against a published 4096² solution, Cp at 128 matches to 0.001–0.007 along the chord. The remaining gap is the trailing-edge stagnation spike, which carries little load.
- **Trailing edge.** Its cut topology is identical on all three grids, the edge is exact, and no tip is truncated.
- **Moment convention.** `wall_forces`' `cm` became nose-up positive.
- Details are in `dev-notes/phase5.md`.

### Phase 6 · Acceleration

- [ ] Element and face Jacobians with ForwardDiff
- [ ] Backward-Euler pseudo-transient continuation with local steps and switched evolution relaxation
- [ ] Block-Jacobi, then ILU-preconditioned GMRES

**Gate (proposed Oct 10).**

- **The same answers, fast.** The airfoil at 128 cells per chord converges 10 orders in a few hundred nonlinear iterations, against 502,000 explicit steps. Its forces agree with the explicit run to about 1e-8.
- **Where explicit fails.** The circle at 64 cells per diameter on the nested far field reaches a steady state, where the explicit solve settles into a limit cycle.
- **What becomes affordable.** The airfoil at 256 cells per chord, to confirm the asymptotic order and bring lift within 1% of the reference.

**Version 1 loose ends**, to finish alongside:

- [ ] The force-plateau stopping criterion, from the convergence-monitoring section
- [ ] P1 values at the vertices in the VTK output, from the postprocessing table
- [ ] The point-vortex farfield correction, carried from Phase 5: optional, worth about 0.5% in lift at 100 chords

**Backlog after version 1**, roughly in order of value:

- **Local refinement near the leading and trailing edges.** The trailing-edge singularity converges at first order, and the uniform finest box spends most cells where they are not needed. It needs CartesianMeshes' `AdaptiveMesh`, or several finest patches, and the polyline cutter extended to their cells.
- **Transonic flow.** Vassberg and Jameson's M 0.8, 1.25° case is the next benchmark, with shock capturing (Persson and Peraire) and a low-Mach flux fix.
- **Higher order, P2 and up, with curved cut faces.** Curved faces matter most at higher order (Bassi and Rebay).
- **GPU.** The kernels are KernelAbstractions already; the solver data needs Adapt, and the cutter stays on the host.
- **3D.**
- **The run-configuration format**, moved here from Phase 5.

The two-segment trailing-edge wedge is no longer needed: the exact cut keeps the tip.

## Open decisions

Eight choices were open; all are now settled. Four were decided: the two on Oct 8, the background mesh on Oct 9 and the first airfoil case on Oct 10. The other four were settled by what the phases built, the implicit solver's timing following the recommendation.

| Decision | Recommendation | Needed by |
| --- | --- | --- |
| Runtime: CPU FP64 only, or GPU kernels from the start | **Settled (Oct 9):** CPU FP64, threaded with KernelAbstractions kernels, which keep the GPU path open | Phase 0 |
| Trixi.jl as a dependency, or copy its equation and flux functions | **Decided (Oct 8): copy.** Trixi 0.17.13 adds 2.5 s to every load and a 228-package tree, for about eight functions | Phase 0 |
| Version 1 background mesh: stretched tensor-product, or quadtree | **Decided (Oct 9): nested grids.** The body is cut on a uniform finest box, the only grid CutCellMethods' polyline clipping takes. The box is wrapped in CartesianMeshes `NestedGrids` levels out to the far field, each doubling the cell size, with hanging nodes where levels meet (`nested_cut_mesh`). Square cells keep the explicit local steps growing with the cells, which stretched cells would not. `AdaptiveMesh` stays the route to surface-following or solution-adaptive refinement after version 1, once the cutter handles tree leaves | Phase 2 |
| Production flux: HLLC or Roe | **Settled by Phase 3:** HLLC, with Rusanov for debugging | Phase 3 |
| Is the P0 fallback acceptable, given state redistribution was ruled out | Yes, as an off-by-default safeguard. **Not built (Oct 10):** no case has needed it | Phase 4 |
| Trailing-edge tip: accept truncation, or two-segment wedge | **Settled by Phase 2:** the exact polyline cut keeps the polygon's own wall pieces in each fragment, so the tip is not truncated | Phase 5 |
| First airfoil case and reference data source | **Decided (Oct 10):** Vassberg and Jameson (2010), NACA 0012 at M 0.5, α 1.25°, on their sharp-trailing-edge geometry; FLO82 continuum Cl 0.180346, Cm -0.002269 | Phase 5 |
| Implicit solver timing: after Phase 5, or earlier | **Settled (Oct 10): Phase 6 follows Phase 5, as recommended.** Explicit convergence held for the airfoil to 128 cells per chord (502,000 iterations), but iterations grow 1.2–3x per refinement, and the 64-cell circle settles into a limit cycle | Phase 5 |

## References

These are cited from memory without a database check, so verify titles, venues and years before relying on them; only Vassberg and Jameson has been checked.

- Bassi and Rebay, high-order DG for the 2D Euler equations and the role of curved boundaries, J. Comput. Phys., 1997.
- Fidkowski and Darmofal, triangular cut-cell adaptive method for high-order DG with implicit steady solves, J. Comput. Phys., 2007.
- Qin and Krivodonova, DG for the Euler equations on Cartesian grids with embedded geometries, J. Comput. Sci., 2013.
- Engwer, May, Nüßing and Streitbürger, domain-of-dependence stabilization for DG on cut-cell meshes, SIAM J. Sci. Comput., 2020.
- Berger and Giuliani, state redistribution for cut cells, J. Comput. Phys., 2021.
- Ranocha et al., Trixi.jl adaptive high-order solvers in Julia, JuliaCon Proceedings, 2022.
- Vassberg and Jameson, In pursuit of grid convergence for two-dimensional Euler solutions, J. Aircraft 47(4), 2010, doi:10.2514/1.46737 (checked Oct 10: the Phase 5 reference).
- Persson and Peraire, sub-cell shock capturing for DG, AIAA paper, 2006 (for the transonic backlog).
