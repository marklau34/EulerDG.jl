# Cut-Cell DG Euler Solver — Architecture & Development Plan

Oct 8, 2026 · @Mark

## Scope and assumptions

Version 1 delivers a converged, steady, subsonic 2D airfoil solution from a P1 modal DG discretization on a Cartesian cut-cell mesh, with local pseudo-time stepping in place of state redistribution.

| Item | Version 1 choice |
| --- | --- |
| Equations | 2D compressible Euler, steady only, subsonic (local Mach < 1) |
| Discretization | Modal DG, P = 1 on every element |
| Geometry | Closed polygons (SDFLibrary `SDFMesh`), cut exactly by CutCellMethods; a fragment's wall is the polygon's own pieces in the cell (Phase 2) |
| Background mesh | Static uniform Cartesian grid (CartesianMeshes' `CartesianGrid`); stretching deferred (Oct 8); no AMR |
| Small cells | Local time stepping only; no state redistribution, no merging |
| Runtime | Julia, CPU, FP64 (assumed; see Open decisions) |

Out of scope for version 1: shocks and limiting, curved cut faces, AMR, 3D, GPU and implicit solvers. The architecture keeps a clean path to each of them.

Assumed reuse: the SDF geometry library, the earlier cut-cell preprocessing code, and Trixi.jl's equation and flux layer and code organization.

## Core design decisions

Local time stepping is sufficient for the small-cell time-step problem because only the steady residual R(U) = 0 matters. Four further decisions make the rest of the small-cell problem manageable.

### 1. Local time stepping replaces state redistribution

The steady solution does not depend on the pseudo-time path. Each element can march with its own stable step, which acts as a diagonal preconditioner. Information still crosses about one element per iteration, so tiny cells add unknowns but do not slow global convergence.

Local time stepping does not fix three things, each handled separately:

- **Mass-matrix conditioning on slivers.** Handled by the per-element basis (decision 3).
- **Weakly controlled gradient modes on slivers.** Handled by an optional P0 fallback (decision 4).
- **Global stiffness of explicit P1 DG.** Accepted in version 1; an implicit solver is the planned cure.

### 2. A polygon-DG core; Cartesian structure lives only in the preprocessor

After preprocessing, the solver sees a general mesh of polygonal elements and faces. Full cells and cut fragments run through one code path. Several fragments in one background cell need no special case. A later quadtree adds hanging nodes as extra faces on a polygon, so no mortars are needed. The cost is losing tensor-product sum factorization, which is negligible at P1.

This also means Trixi's DGSEM kernels are not reused. They rely on tensor-product LGL nodes with summation-by-parts, which cut cells break. The solver uses the standard weak-form modal DG with explicit quadrature. It borrows Trixi's equation and flux functions and its container layout.

### 3. Per-element basis on the fragment's bounding box

Each element starts from 1, (x − xc)/hx and (y − yc)/hy on its own bounding box. Gram–Schmidt against the element's own quadrature makes the modes orthogonal. Normalizing so that the first mode is 1 gives a mass matrix of |K| times the identity. Mode 0 is then the cell average, which simplifies positivity checks, output and the P0 fallback.

### 4. A small-cell safeguard ladder, used only as needed

1. Nothing beyond local time stepping (the default).
2. Per-element CFL backoff when a cell's residual grows.
3. Drop to P0 on fragments below a volume-fraction threshold (off by default).
4. Treat fragments below round-off size, around 1e-12 of a full cell, as solid. The conservation error is negligible.

Merging stays in reserve and is not planned. The grid-shift sweep in Phase 4 decides whether rungs 2 and 3 are ever needed.

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
- Wall crossings on cell edges come from a root-find on the SDF along the edge.
- All lengths, normals and areas derive from one vertex list per element. Then the face vectors of every element sum to zero at round-off.
- That closure property is what makes uniform flow along a straight wall preserved exactly. Check it in the report on every run.

### Trailing-edge pitfalls

The trailing edge is where a cut-cell airfoil usually goes wrong, and the errors change circulation rather than crash.

- **Lost features.** Where the airfoil is thinner than a cell, every vertex can sit in fluid while the body passes through. Vertex-based classification then misses the body and the flow leaks across the trailing edge. Flag any cell whose center SDF value is below its half-diagonal while all vertices share a sign. The preprocessor stops with an error on these rather than continuing; the fix is local refinement.
- **Split cells.** A cell straddling the trailing edge holds an upper and a lower fragment with a solid sliver between. These must be two elements with no face between them.
- **Truncated tip.** One straight segment per fragment cuts the sharp tip off and leaves a blunt base about one cell wide. Version 1 may accept this, but the report should list it. The option for later is a two-segment wedge wall in the tip cell, which makes that fragment nonconvex.

The Kutta condition needs no explicit enforcement in an Euler solver. Numerical dissipation at a sharp, non-leaking trailing edge sets the circulation.

## DG discretization

The discretization is the standard weak form, evaluated by explicit quadrature on every element:

```latex
\int_K \phi_i \, \partial_t U \, dx \;-\; \int_K \nabla \phi_i \cdot F(U) \, dx \;+\; \oint_{\partial K} \phi_i \, \hat{F}(U^-, U^+, n) \, ds \;=\; 0
```

| Component | Choice | Notes |
| --- | --- | --- |
| Basis | 3 modes per variable, built per element (Core decision 3) | State array sized 4 × 3 × elements, Trixi-style |
| Volume quadrature | Ear-clipping triangulation, 3-point degree-2 rule per triangle | Ear clipping also covers the nonconvex trailing-edge wedge; a 6-point rule for over-integration is a switch |
| Full-cell quadrature | Tensor 2 × 2 Gauss, shared reference data | Same kernel as cut cells |
| Face quadrature | 2-point Gauss per segment | Normals stored per quadrature point so curved faces drop in later |
| Interior flux | Rusanov for debugging; HLLC or Roe for production | Fluxes are plain functions of (uL, uR, n), as in Trixi |
| Slip wall | Mirror state through the same Riemann flux | Pressure-only wall flux kept as an alternative to compare |
| Farfield | Characteristic (Riemann-invariant) state | Boundary at 50–100 chords via stretching; point-vortex correction later |
| Limiting | None for subsonic flow | Positivity check at quadrature points aborts with the element id |

Straight cut faces are consistent with second-order accuracy at P1. They still produce spurious entropy at the wall, so entropy error is tracked from the first airfoil run. Bassi and Rebay showed boundary representation dominates DG accuracy at higher order, which is why curved faces are on the backlog.

## Pseudo-time marching with local time stepping

Each element advances with its own step, computed from its own area and face fluxes, so a sliver's tiny step never limits its neighbours.

```latex
\Delta t_K = \mathrm{CFL} \cdot \frac{|K|}{(2p+1) \sum_{f \in \partial K} \lambda_f \, |f|}, \qquad \lambda_f = \max_{\text{face quad. pts, both sides}} \left( |v \cdot n| + c \right)
```

The face sum includes wall faces. A thin sliver along a long wall face therefore gets a step that scales with its thickness, which is the correct limit. Calibrate the CFL number on full cells first.

### Implementation

- Freeze the step field at the start of each iteration and keep it across the Runge–Kutta stages.
- Fold the steps into the right-hand side: integrate dU/dτ = −D M⁻¹ R(U) with D = diag(Δt\_K) and dτ = 1. Any fixed-step integrator then works unchanged, including DifferentialEquations.jl.
- Hand-roll SSP-RK3 or a low-storage scheme in version 1. It is about 20 lines and keeps full control of stages and monitoring.
- Start from freestream with a CFL ramp from about 0.1 to the target over the first few hundred iterations.
- Optional startup: converge P0 first, then lift to P1. This also prototypes a P1-to-P0 multigrid.

### Convergence monitoring

- L2 norm of the cell-average density residual, normalized by its first value.
- Maximum residual over elements and the element that holds it. A stuck small cell shows here first.
- Lift, drag and moment histories.
- Stop at 10 orders of residual drop, or a force plateau to 1e-6 over 1,000 iterations.

### Acceleration path

Explicit P1 DG is slow for steady problems; expect many thousands of iterations on airfoil grids. The planned cure is implicit pseudo-transient continuation with the same local steps, and switched evolution relaxation for CFL growth. Element and face Jacobians come from ForwardDiff, solved with block-Jacobi or ILU-preconditioned GMRES. Once implicit, small cells are no longer a time-step issue at all. P1-to-P0 multigrid is the explicit alternative.

## Postprocessing and diagnostics

For a subsonic inviscid airfoil, entropy error and spurious drag are the primary accuracy measures; lift and Cp are compared against reference data.

| Output | How | Used for |
| --- | --- | --- |
| Lift, drag, moment | Integrate p n over wall faces at their quadrature points, using the interior trace | Convergence and validation |
| Surface Cp | Values at wall quadrature points, sorted by x/c | Comparison with reference Cp |
| Entropy error | (s − s∞)/s∞ field and its domain L2 norm | Primary accuracy metric; should fall at about second order |
| Spurious drag | Drag coefficient, exactly zero in the continuous problem | Second accuracy metric |
| Field output | Each element written as a VTK polygon with P1 values at its vertices, plus cell averages | Visual checks of cut cells |
| Run log | Residual and force history per iteration, CSV | Convergence plots |

Cut-cell surface Cp is naturally scattered from cell to cell. Plotting per quadrature point shows that scatter honestly, which is itself a quality check.

## Development plan

Seven phases take the solver from an uncut DG core to a validated subsonic airfoil and then to fast convergence; no phase starts until the previous gate passes.

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

- [ ] Run-configuration format, moved from Phase 0 (Oct 8); the grid-shift sweep is its first user. Not built yet: the low-resolution sweep uses keyword arguments.
- [x] Characteristic farfield boundary
- [x] Convergence monitors: L2 and maximum residual with its location, plus a CSV run log
- [x] Subsonic inviscid circle at M = 0.3, below the circle's critical Mach number, at low resolution (8 and 16 cells per diameter)
- [x] Grid-shift sweep at 8 cells per diameter, logging minimum volume fraction, iterations and entropy error
- [x] Decide from the sweep whether safeguard rungs 2 and 3 are needed: not at this resolution, since slivers to 1e-12 converge in the usual number of iterations

**Low-resolution gate passed (Oct 8).** At 8 and 16 cells per diameter, on [−3, 3]² with the degree-4 volume rules, both grids converge 10 orders. Lift (≤ 1.5e-7) and the Cp asymmetry (≤ 2.3e-7) are zero but for the triangulation of mirrored fragments. Spurious drag falls at order 1.89, the entropy error at 2.08 and the entropy wake at 2.47. Not yet met:

- The stagnation Cp is 1.11 against 1.023, from the far boundary at 6 radii; it scales like (R/L)².
- On shifted grids the lift reaches ±0.16 at 8 cells per diameter, because the under-resolved wake is asymmetric.
- Meeting the full gate needs finer grids and a larger domain. That waits for a decision on resolution, the implicit solver and stretched grids.

**Circle gate.** The inviscid circle has no circulation and no wake, so any lift, drag or asymmetry is numerical error. All of these must hold on every grid and every grid shift:

- Residual drops 10 orders.
- Lift is zero to round-off, and Cp is symmetric top to bottom and fore to aft.
- Stagnation Cp matches the isentropic value, about 1.023 at M = 0.3.
- Spurious drag and the entropy-error L2 norm fall with h, close to second order.
- No spurious wake of entropy behind the circle in the VTK output.

### Phase 5 · Subsonic airfoil

- [ ] NACA 0012 SDF with a sharp trailing edge, and a reviewed trailing-edge report
- [ ] Stretched grid with the farfield at 50–100 chords
- [ ] Forces, Cp, entropy and VTK output
- [ ] Three-grid study comparing lift, moment and Cp with reference data
- [ ] Optional point-vortex farfield correction

### Phase 6 · Acceleration

- [ ] Element and face Jacobians with ForwardDiff
- [ ] Backward-Euler pseudo-transient continuation with local steps and switched evolution relaxation
- [ ] Block-Jacobi, then ILU-preconditioned GMRES

Backlog after version 1: quadtree refinement, curved cut faces, a two-segment trailing-edge wedge, a low-Mach flux fix, transonic shock capturing, 3D and GPU.

## Open decisions

Eight choices were open; two were settled on Oct 8 and are marked as decided. The recommendation for each open one is the default if nothing else is decided.

| Decision | Recommendation | Needed by |
| --- | --- | --- |
| Runtime: CPU FP64 only, or GPU kernels from the start | CPU FP64 first; face-loop layout kept GPU-friendly | Phase 0 |
| Trixi.jl as a dependency, or copy its equation and flux functions | **Decided (Oct 8): copy.** Trixi 0.17.13 adds 2.5 s to every load and a 228-package tree, for about eight functions | Phase 0 |
| Version 1 background mesh: stretched tensor-product, or quadtree | **Decided (Oct 8): uniform for now**, the only grid CartesianMeshes and CutCellMethods handle. Revisit before Phase 5, whose farfield at 50–100 chords needs stretching | Phase 2 |
| Production flux: HLLC or Roe | HLLC; keep Rusanov for debugging | Phase 3 |
| Is the P0 fallback acceptable, given state redistribution was ruled out | Yes, as an off-by-default safeguard | Phase 4 |
| Trailing-edge tip: accept truncation, or two-segment wedge | Accept truncation in version 1 and report it | Phase 5 |
| First airfoil case and reference data source | A NACA 0012 subsonic case with published Euler results | Phase 5 |
| Implicit solver timing: after Phase 5, or earlier | After Phase 5, unless explicit convergence blocks airfoil grid studies | Phase 5 |

## References

These are cited from memory without a database check, so verify titles, venues and years before relying on them.

- Bassi and Rebay, high-order DG for the 2D Euler equations and the role of curved boundaries, J. Comput. Phys., 1997.
- Fidkowski and Darmofal, triangular cut-cell adaptive method for high-order DG with implicit steady solves, J. Comput. Phys., 2007.
- Qin and Krivodonova, DG for the Euler equations on Cartesian grids with embedded geometries, J. Comput. Sci., 2013.
- Engwer, May, Nüßing and Streitbürger, domain-of-dependence stabilization for DG on cut-cell meshes, SIAM J. Sci. Comput., 2020.
- Berger and Giuliani, state redistribution for cut cells, J. Comput. Phys., 2021.
- Ranocha et al., Trixi.jl adaptive high-order solvers in Julia, JuliaCon Proceedings, 2022.
- Persson and Peraire, sub-cell shock capturing for DG, AIAA paper, 2006 (for the transonic backlog).
