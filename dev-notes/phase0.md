# Phase 0: conventions and departures from the brief

Companion to *Phase 0 Scaffolding — Implementation Brief*, including the decisions settled on
Oct 8, 2026. Part 1 records the conventions taken from EBM.jl, part 2 every place the
implementation departs from, interprets or extends the brief, and part 3 the measurements and
what Phase 1 inherits.

New code: the common layer in `src/common/` (`geometry.jl` polygon functions and validity,
`mesh.jl` types, accessors and derived geometry, `build.jl` the builder constructor, `check.jl`
the invariants, `summary.jl`, `vtk.jl`) and eight submodule stubs, one directory each. Tests:
`test/fixtures.jl`, `test_geometry.jl`, `test_mesh.jl`, `test_summary.jl`, `test_vtk.jl` and
`test_layering.jl`, 197 checks in the commit tier.

## 1. Conventions taken from EBM.jl

- **Layout.** The mesh contract is a common layer included straight into `EulerDG`, as EBM's
  `src/common/` is. The eight pipeline stages are real modules, each with a docstring listing its
  planned API as bullets.
- **Imports.** Imports are explicit everywhere (`using WriteVTK: MeshCell, PolyData, ...`), with
  no Reexport. `EulerDG.jl` keeps a hand-written export list.
- **Files and docstrings.** Every file starts with a `# =====` banner and a prose header, and
  docstrings give the signature as `f(args) -> ret`.
- **Errors.** Errors are `ArgumentError`s, thrown through `@noinline _mesh_error(msg)`.
- **Checks.** EBM's `check_invariants`/`assert_invariants` pair became `check_mesh`/`assert_mesh`:
  counts per invariant, the first offender of each, and `ok`. Its summary/report pair became
  `mesh_summary`/`mesh_report`.
- **Tests.** The two tiers are selected by `ARGS` or `EULERDG_TEST_TIER`, with one `@testset` per
  file and fixtures in `test/fixtures.jl`. An unknown tier is an error.
- **Dependencies.** `[compat]` pins the versions in use: StaticArrays 1.9.22, StructArrays 0.7.3,
  WriteVTK 1.22.0, Printf 1, with `julia = "1.11"`. `Manifest.toml` is committed, and `out/` is
  git-ignored.

## 2. Departures from the brief

1. **`MeshElement` has a `parent_area` field.** Invariant 6 recomputes the volume fraction
   bitwise, which needs the parent area, and the brief's type did not keep it. With no
   background, `parent_area` is the element's own area, so `area / area` is exactly 1.
2. **The default boundary order is the order of first appearance along the elements' edges,
   walked in element order.** The brief said "first seen in `boundary_edges`", which is undefined
   for a `Dict`. A `boundary_names` keyword sets the order explicitly, and may list boundaries
   that have no faces.
3. **Tolerances scale with the float type.**
   - Closure: 450 eps relative to the perimeter, which is 1e-13 in Float64, as in the brief.
   - Periodic matching: sqrt(eps)/100 of the face length, which is 1.5e-10 in Float64, where the
     brief said 1e-10.
   - Volume fraction: may exceed 1 by 16 eps, where the brief said at most 1. Rounding can push
     a nearly full fragment an ulp over 1. The summary counts such an element as full.
4. **`face_sign` returns 0 when the element is not on the face**, instead of throwing, so it stays
   usable in loops.
5. **The builder catches more errors than the brief listed:**
   - a named edge that no element runs in that direction;
   - a periodic side glued to its own element (a one-element-wide periodic direction);
   - a periodic side with no faces, or one glued twice;
   - a glued name that is also listed in `boundary_names`.
6. **Submodule names are not exported.** EBM exports `Preprocess, Solver, Post`, but Meshes.jl
   exports `Geometry`, so exporting ours would collide. Submodules are reached as `EulerDG.Basis`
   and so on.
7. **`MESH_INVARIANTS` and `VF_BINS` are exported constants**, the names of the checks and of the
   histogram bins.
8. **Polygon simplicity uses floating-point orientation tests, not exact predicates.** A sliver
   thin enough to fool them lies far below any volume fraction a cut keeps. ExactPredicates arrives
   with CutCellMethods in Phase 2 and can replace them if a real cut ever trips them.
9. **`nelements` is also exported by MeshLibrary.** A script that does `using MeshLibrary, EulerDG`
   must qualify it. SDFLibrary does not re-export it, so `using SDFLibrary, EulerDG` is fine.
10. **The layering scan covers comments too.** It caught a comment in `mesh.jl` that named the
    cutter library, and the comment was reworded. Prose in the downstream files avoids the
    forbidden names.

## 3. Measurements, and what Phase 1 inherits

- **Load time.** `using EulerDG` takes 0.47–0.49 s warm (3.3 s on the first load after
  precompiling).
- **Sliver precision.** For the brief's sliver case (δ = 1e-11, 1000 cells from the origin), the
  first-vertex shoelace matches a `BigFloat` reference to 1e-12. An absolute-coordinate shoelace
  loses the sliver's area entirely. Closure is exactly zero.
- **Trixi.jl.** 0.17.13 loads in 5.2 s alone and adds 2.5 s on top of the 4.2 s geometry stack;
  its manifest has 228 packages, and precompiling Trixi itself takes 31 s. Decided: copy the
  functions. The copying and a `dev/` cross-check script come in Phase 1, with the first flux.
- **Grid nodes.** On a `CartesianGrid` with spacing that is not a power of two, `get_elem_area`
  (`prod(d)`) and the `lo + d` corner of `get_elem_bounds` disagree with `get_node` bitwise. On a
  37 × 23 grid they differed in 851 and 794 of 851 cells. Phase 1's uncut builder must take every
  vertex from `get_node` and every parent area from node differences. Built that way, every full
  cell had a volume fraction of exactly 1.
- **Not done here.** The fixtures' VTK files are written to `out/phase0/`, but nobody has opened
  them in ParaView yet. The brief asks for that once, by hand.
