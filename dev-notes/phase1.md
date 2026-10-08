# Phase 1: the DG core on uncut grids

Companion to the plan's Phase 1 checklist and the decisions of Oct 8, 2026. No brief was written
for this phase. Part 1 records the choices made along the way, part 2 the gate results and
measurements, and part 3 what Phases 2 and 3 inherit.

New code:

- `src/CutMesh/uncut.jl`: `uncut_mesh`.
- `src/Quadrature/`: the reference rules, ear clipping and `polygon_rule`.
- `src/Basis/Basis.jl`: the per-element basis.
- `src/Equations/`: the Trixi copies in `trixi.jl`; `freestream` and `IsentropicVortex` in
  `states.jl`.
- `src/Semidiscretization/Semidiscretization.jl`: the solver data, `rhs!`, `project`, `max_dt` and
  `assert_positive`.
- `src/PseudoTime/PseudoTime.jl`: SSP-RK3 and `integrate!`.
- `src/Post/Post.jl`: `l2_error` and `write_solution_vtk`.
- `dev/check_trixi.jl`, the Trixi cross-check, and `examples/vortex.jl`.

Tests: `test_uncut.jl`, `test_quadrature.jl`, `test_basis.jl`, `test_equations.jl`,
`test_semidiscretization.jl` and `test_post.jl`, plus `scheduled/test_phase1_gate.jl`. The commit
tier has 608 checks, and the scheduled tier 613.

## 1. Choices

1. **Background grid.** The background is CartesianMeshes 0.2.3's `CartesianGrid`, the version
   installed and the one EBM uses. `uncut_mesh` takes every vertex from `get_node` and every parent
   area from node differences. On a 37 x 23 grid with spacing that is not a power of two, every
   volume fraction is exactly 1.
2. **Box sides.** The sides are always named in the order `:x_neg, :x_pos, :y_neg, :y_pos`. The
   builder passes this order explicitly, so it does not depend on the order of the edge walk.
3. **Trixi copies.**
   - The `@muladd` blocks are dropped, so there are no fused multiply-adds and the results are
     deterministic.
   - LinearAlgebra became a dependency, for `norm`.
   - `dev/check_trixi.jl` compares the copies with Trixi 0.17.13 over 10,000 random states. The
     largest relative difference is 3.6e-15, and most functions agree bitwise.
4. **Volume rules.**
   - Axis-aligned rectangles, detected from the polygon itself, get 2 x 2 Gauss.
   - Other polygons are ear-clipped, with the 3-point degree-2 rule per triangle.
   - The `volume_degree = 4` switch (6 points per triangle, 3 x 3 on boxes) is also what `project`
     and `l2_error` use.
5. **Basis.**
   - The coordinates are scaled to [-1, 1] on the element's bounding box.
   - Modified Gram-Schmidt runs on the rule's 3 x 3 Gram matrix, and the mass matrix is
     `Σ w` times the identity.
   - A mode whose orthogonal part has a squared norm below 64 eps of the area throws. A diagonal
     sliver 1e-13 thick trips this; the plan's P0 fallback is the cure, if Phase 4 needs it.
6. **Element-local coordinates.** This departs from the obvious implementation, and was found by
   the tests.
   - In absolute coordinates, rounding the quadrature points to the absolute coordinate's last bit
     moves them by 0.5% of the height of the Phase 0 sliver, which is 2e-11 high and 1000 cells
     from the origin.
   - Under the degree-4 rule, that sliver's mass matrix was off by 0.5%, and the y-mode's values at
     the face Gauss points by up to 0.7%. Those face errors are equal and opposite at a face's two
     points, so for linear integrands they cancel, and freestream held regardless.
   - The fix: every rule is now built relative to the element's box centre
     (`polygon_rule(...; origin)`, `element_basis(...; origin)`, `basis_values(b, x, origin)`).
     Absolute positions are kept only for boundary conditions and exact solutions, and the sliver's
     mass matrix is now |K| times the identity to 1e-13.
7. **Face data.** Each face has two Gauss points, and each stores its weight times the face vector,
   so a face integral is a plain sum of flux calls. This relies on the fluxes being homogeneous of
   degree 1 in the normal direction, which a test checks.
8. **Boundary conditions.** They are given as a NamedTuple keyed by boundary name. The constructor
   stores them as a Tuple in mesh order and walks it recursively, so each boundary's loop is
   type-stable. `rhs!` and `max_dt` allocate nothing.
9. **Time step.** The step is the plan's local-step formula, minimized over the elements, with
   (2p + 1) = 3.
   - The CFL number was calibrated on the vortex over a full period. Its stability limit is about 4
     on square cells and about 3 at aspect ratio 2, so the default is 2.
   - `integrate!` checks positivity at every volume quadrature point after every step, and a
     blow-up surfaces as a `DomainError` naming the element.
10. **Isentropic vortex.** It is Shu's vortex: strength 5 on [-5, 5]^2, moving at velocity (1, 1).
    The exact solution is measured from the nearest periodic image. The Gaussian tail leaves a
    velocity mismatch of about 2e-5 at the periodic boundary, below the gate's errors, which are at
    least 6e-4.
11. **Solution output.** `write_solution_vtk` writes the cell averages of density, velocity,
    pressure, Mach number and the entropy function. It is an early piece of the plan's Phase 5
    field output, for visual checks.

## 2. Gate results and measurements

Vortex, L2 density error at t = 1:

| Grid | L2(ρ) | Order |
| --- | --- | --- |
| 16 x 16 | 5.048e-2 | |
| 32 x 32 | 1.195e-2 | 2.08 |
| 64 x 64 | 2.578e-3 | 2.21 |
| 128 x 128 | 5.987e-4 | 2.11 |
| 32 x 16 | 3.300e-2 | |
| 64 x 32 | 7.217e-3 | 2.19 |
| 128 x 64 | 1.632e-3 | 2.15 |

**Freestream.** Mach 0.5 at 30 degrees, with the freestream imposed on every boundary.

- Through t = 5, the largest deviation is 9.6e-15 on a 24 x 24 grid, 4.4e-16 on `hanging_node`
  and 1.5e-16 on `tilted_wall`.
- In a single `rhs!` on every fixture, the sliver included, the residual |K| du is at most 6e-16
  of the element's perimeter.

**Speed.** The 128 x 128 vortex takes 208 steps at CFL 2 in 1.4 s, about 2.3 ms per right-hand
side. `using EulerDG` takes about 1.2 s, most of it loading CartesianMeshes.

## 3. For Phases 2 and 3

- **Slivers and the global step.** With freestream on the sliver, |K| du is at round-off, but du
  itself is 7e-5, because dividing by |K| = 1.75e-11 amplifies the round-off. With a global step
  the sliver also forces a tiny step on everyone. With local steps (Phase 3) the update
  `Δt_K du` is at round-off.
- **The wall.** The tilted wall is a Dirichlet freestream boundary in these tests; Phase 3's slip
  wall replaces it.
- **Orientation tests.** Ear clipping and the simplicity check use floating-point orientation
  tests. Exact predicates arrive with CutCellMethods in Phase 2, in case a real cut needs them.
- **Rectangular fragments.** A cut fragment that happens to be an axis-aligned rectangle (a cell
  cut along a grid line) also gets the tensor rule. That is correct: the rule is exact on any
  rectangle.
