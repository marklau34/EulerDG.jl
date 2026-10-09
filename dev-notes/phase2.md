# Phase 2: the cut-mesh preprocessor

Companion to the plan's Phase 2 checklist and to items 1–7 of "Findings for later phases" in the
Phase 0 brief. No brief was written for this phase. Part 1 records the choices, part 2 the gate
results, and part 3 what later phases inherit.

New code:

- `src/Geometry/Geometry.jl`: `Body`, `polygon_body`, `circle_points`, `naca4_points`,
  `polygon_perimeter`, `grid_shifts` and `sliver_shift`.
- `src/CutMesh/cut.jl`: `cut_mesh` and `cut_report`.
- `examples/cut.jl`.

Tests: `test_bodies.jl`, `test_cut.jl` and `scheduled/test_phase2_gate.jl`. The commit tier has
693 checks, and the scheduled tier 720.

## 1. Choices

1. **A fragment's wall is the polygon's own pieces in the cell.** The plan assumed one straight
   wall segment per fragment. The cutter already returns each fragment's exact loop, and keeping it
   costs nothing.
   - The fluid is exactly the polygon's complement.
   - The trailing edge is exact, as a nonconvex fragment. There is no truncated tip, and the
     backlog's two-segment wedge is unnecessary.
   - With 4 segments per cell, a cut cell has a few wall faces, and most circle fragments are
     nonconvex. Ear clipping handles them.
2. **Exact crossings.** Crossings come from CutCellMethods' exact polyline intersection, not from
   a root-find on the SDF. The circle is a polygon with 4 segments per cell, and the fluid-area
   convergence the plan asks for is the polygon's convergence, at order 2.00.
3. **NACA point spacing.** `naca4_points` uses half-cosine spacing, which clusters only at the
   leading edge, and grows the segment count until every segment is at most h/4.
   - Full cosine spacing, used at first, also clustered at the trailing edge. At 64 cells per
     chord, with the trailing edge 0.3 h into its cell, that cell then held one fragment of 41
     vertices with 36 wall faces. With half-cosine spacing it holds 9 vertices and 4 wall faces,
     and the split cell beside it 8 vertices instead of 23.
   - The leading-edge cell keeps about 37 vertices at every resolution, which is where the
     curvature is resolved.
   - The area error fell by about a third at every h, still at order 2.00.
4. **Canonical points.** The cutter walks each cell in local offsets from `get_node(lo)`, and
   stores each crossing once per edge as an offset along it. Two cells sharing an edge therefore
   compute the same coordinate along the edge. Only the edge line's own coordinate, `lo + h`, can
   differ from the node value. Every loop coordinate within 4 ulp of the cell's node lines is
   snapped onto them, so shared points become bitwise equal and pool into one vertex; `-0.0` is
   normalised to `0.0` first. Pairing never failed across the gate's 114 meshes.
5. **Naming.**
   - `:wall` goes to loop edges that do not run along a side of their cell, and to the intervals
     the cutter closed.
   - Domain sides are named by their coordinates: `:x_neg`, `:x_pos`, `:y_neg` and `:y_pos`, as in
     `uncut_mesh`. The solver maps all four sides to the farfield.
   - Every other edge must pair with its neighbour's, or the builder throws, so a gap never becomes
     a wall unnoticed.
6. **Slivers.**
   - `drop_area` defaults to the plan's 1e-12. It is in units of h², with h the smaller spacing,
     which is the cutter's convention. On anisotropic cells that is a somewhat smaller fraction of
     the cell: 6e-13 at dx/dy = 1.67 in the gate.
   - The cutter closes a dropped sliver's intervals on both sides, and they become `:wall` on the
     neighbours.
   - An uncut neighbour this happens to is promoted by the cutter to status CUT with no loop, so it
     is treated as a full cell.
7. **Lost features.** The plan's vertex-sign check targets classification by vertex sign, which
   exact clipping does not use. Four checks stop the build instead:
   - a cell the cutter could not walk (more than 64 crossings, or a degenerate cut);
   - a body loop wholly inside one cell;
   - a body vertex strictly inside a cell the cutter left uncut and fluid;
   - fluid area missing more than the dropped slivers explain, `n_cut · drop_area · h²` plus
     1e-10 of the domain.

   A clockwise body, or one touching the grid's edge, is refused before the cut.
8. **Order and parents.** Elements follow the background cells' linear order, then the cutter's
   region order. Each element's parent is the cell's linear index and its fragment the region
   number. Parent areas come from node differences, so full cells have volume fraction exactly 1.
9. **Grid shifts.**
   - `grid_shifts` gives deterministic R2 low-discrepancy offsets that cover the cell evenly.
   - `sliver_shift` moves the body so that a grid node lies `gap` outside it. The node is the
     nearest one outside the body whose normal is at least 20° off both axes.
   - The gate's gaps of 1e-3, 1e-6 and 1e-8 h reach volume fractions near 1e-6 and 1e-12, and a
     dropped sliver.
   - With the half-cosine NACA spacing, the 1e-8 gap drops a sliver on every grid of both bodies.
     With full cosine spacing, it had left a fragment of 1.7e-11 on NACA 0012 at 128 x 128: on a
     thin body, the chosen node's cell is not always otherwise covered.
10. **Public API only.** Only CutCellMethods' readers are used: `region_loops`, `arcs`,
    `cache.info` and the `PL_*` constants. `region_loops` re-walks the cut cells on the host; it
    is meant "for inspection", which is fine for a one-time preprocessing step. A 128 x 128 NACA
    cut takes about 0.1 s.
11. **Dependencies and load time.** The new dependencies are SDFLibrary 0.10.6, CutCellMethods
    0.3.0 and MeshLibrary 0.4.5, the versions EBM uses. `using EulerDG` went from 1.2 s to 4.3 s
    warm, nearly all of it the geometry stack, as Phase 0 measured. If solver-only runs need to
    load fast, `Geometry` and `CutMesh` could move behind a package extension.

## 2. Gate results

Fluid-area error against the true body's complement, with 4 segments per cell:

| Body | h | Area error | Order |
| --- | --- | --- | --- |
| circle | 0.0625 | 1.27e-4 | |
| circle | 0.0313 | 3.18e-5 | 1.99 |
| circle | 0.0156 | 7.98e-6 | 2.00 |
| circle | 0.0078 | 2.00e-6 | 2.00 |
| NACA 0012 | 0.0375 | 3.23e-6 | |
| NACA 0012 | 0.0188 | 8.13e-7 | 1.99 |
| NACA 0012 | 0.0094 | 2.03e-7 | 2.00 |
| NACA 0012 | 0.0047 | 5.09e-8 | 2.00 |

**Grid-shift sweep.** For each body there were 3 grids (32, 64 and 128 cells a side), with
16 shifts and 3 sliver shifts each.

- All 114 meshes were valid and had no lost feature.
- The smallest volume fraction reached 6.4e-13 to 1.2e-12, and a sliver was dropped on all
  6 grids.
- The NACA 0012 trailing edge split 18 to 25 cells per grid.
- The freestream residual was at most 1.7e-15 of the element perimeter.

## 3. For later phases

- **The wall is real.** Every cut mesh has a `:wall` boundary. Until Phase 3 adds the slip wall,
  the tests impose the freestream on it.
- **Local steps are needed.** Fragments reach volume fractions of about 1e-12, so with a global
  step the whole mesh would march at a sliver's pace. Phase 3's local steps are what make these
  meshes usable.
- **Bases held up.** Every one of the 114 meshes built its DG basis without tripping the
  degeneracy check, including at 1e-12. The corner fragments that `sliver_shift` makes are compact
  triangles, not thin strips. A thin strip along a nearly axis-aligned wall would test the check
  harder; the Phase 4 sweep should include one.
