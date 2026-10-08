# EulerDG Phase 0 — Implementation Brief (Scaffolding and the mesh contract)

Oct 8, 2026 · @Mark

## Purpose and scope

Phase 0 turns the empty EulerDG package into the skeleton that every later phase builds on. The main deliverable is the `PolygonMesh` contract: the types, the one constructor every mesh builder calls, and a checker strict enough to serve later as the Phase 2 gate. This phase contains no numerics.

In scope:

- The submodule layout, with the plan's dependency direction enforced by a test.
- The contract: `PolygonMesh`, `MeshElement`, `MeshFace`, the polygon geometry functions, the builder constructor, `check_mesh`/`assert_mesh`, `mesh_summary`/`mesh_report` and a mesh VTK writer.
- Hand-built fixture meshes, both valid and invalid.
- The Trixi.jl decision, recorded with measurements.
- The tiered test harness.

The package already exists: `Project.toml` (no dependencies yet), an empty `src/EulerDG.jl`, a placeholder `test/runtests.jl` and `.gitignore`. It loads, and its one test passes on Julia 1.12.6.

## Decisions settled on Oct 8, 2026

- **Trixi.jl: copy its functions, don't depend on it.** See The Trixi.jl decision.
- **Boundaries are named, not given a fixed kind.** See decision 5.
- **The run-configuration format waits for Phase 4.** The grid-shift sweep is the first thing that needs it. Until then, runs are scripts with keyword arguments, as in EBM. This moves the plan's Phase 0 item to Phase 4.
- **The background grid stays uniform for now.** Builders take CartesianMeshes' `CartesianGrid`, the only grid CutCellMethods accepts; dx ≠ dy is allowed. Phase 1's uncut builder covers periodic and uniform grids, not the plan's stretched ones. See item 1 of Findings for later phases.

## Conventions

These come from EBM.jl, the closest sibling package. Follow them, and flag any point where this brief contradicts them instead of silently picking one.

- **Layout.** Shared types live in `src/common/` and are included straight into the top-level module, not wrapped in a submodule. Each pipeline stage is a real `module` with a docstring that lists its API as bullets.
- **Imports.** Imports are explicit everywhere: `using StaticArrays: SVector`, and in submodules `using ..EulerDG: PolygonMesh, ...`. Use `import` only to add methods. There is no Reexport. Each submodule exports its own API, and `EulerDG.jl` repeats the export lists by hand, grouped.
- **Files and docstrings.** Every file starts with a `# =====` banner and a prose header. Docstrings show the signature as `f(args) -> ret`.
- **Errors.** Errors are `ArgumentError`, thrown through a `@noinline _mesh_error(msg) = throw(ArgumentError(msg))` helper.
- **Types.** Ids are `Int32`. Geometry is parametric in `T`. Builders and checks run in `Float64`. Phase 0 adopts the plan's runtime default: CPU and FP64, with the layout kept GPU-friendly.
- **Dependencies.** Add a dependency in the phase that first uses it. `[compat]` pins the version in use (for example `StaticArrays = "1.9.22"`), with `julia = "1.11"`. `Manifest.toml` is committed. Sibling packages resolve through the FlightSciences registry, not by path.
- **Outputs.** Examples write under `out/`, which is git-ignored. They run as `julia --project=examples examples/x.jl`, and `examples/Project.toml` reaches the package through `[sources]`.

Record what you confirmed or changed in `dev-notes/phase0.md`.

## Core design decisions

**1. A shared layer plus eight submodules, ordered from upstream to downstream.**

| Unit | Kind | Owns | May use | Filled in |
| --- | --- | --- | --- | --- |
| `common/` | top-level layer | Mesh contract, polygon geometry, mesh summary and VTK | — | Phase 0 |
| `Geometry` | submodule | Bodies as SDFs: polygonal circle, NACA 4-digit; grid-shift offsets | common, SDFLibrary | Phase 2 |
| `CutMesh` | submodule | Uncut and cut builders on a uniform `CartesianGrid`, preprocessor report | common, Geometry, CutCellMethods, CartesianMeshes | Phases 1–2 |
| `Quadrature` | submodule | Segment Gauss, triangle rules, ear clipping | common | Phase 1 |
| `Basis` | submodule | Per-element orthonormal P1 basis | common, Quadrature | Phase 1 |
| `Equations` | submodule | Euler state, numerical fluxes, boundary states | — | Phases 1, 3, 4 |
| `Semidiscretization` | submodule | Element and face solver data, `rhs!` | common, Quadrature, Basis, Equations | Phase 1 |
| `PseudoTime` | submodule | SSP-RK3, local steps, CFL ramp, monitors | common, Semidiscretization | Phases 1, 3, 4 |
| `Post` | submodule | Forces, Cp, entropy, field VTK, run log | common, Basis, Equations, Semidiscretization | Phases 1, 4, 5 |

Include them in this order. The top-level driver functions come after them. They are the only code that connects a preprocessor to the solver.

**Layering rule.** `common/` and every submodule from `Quadrature` on must not refer to `Geometry`, `CutMesh`, SDFLibrary, CutCellMethods, CartesianMeshes, `CartesianIndex` or `CartesianGrid`. A test scans those source directories and fails on any such reference. Match the module references (`.Geometry`, `.CutMesh`) and the five other names as whole words, so that ordinary prose is not flagged. This makes the plan's rule that nothing downstream ever sees a Cartesian index checkable.

In Phase 0 the eight submodules are stubs. Each is one file with a docstring listing its planned API and an empty body. Do not add their dependencies yet.

**2. One vertex pool; each polygon edge is exactly one face.** An element is a counter-clockwise list of ids into a single pool of points. Edge `k` of an element runs from vertex `k` to vertex `k+1` and is exactly one face. Two elements share a face exactly when one traverses `(a, b)` and the other `(b, a)`. A split point is one pool entry, so both sides see the same bits by construction. A hanging node from a later quadtree is just one more collinear vertex on the coarse polygon.

**3. Builders supply polygons; the constructor derives the faces.** Every builder supplies three things: a vertex pool, the element polygons, and a name for each boundary edge. This covers the hand-built fixtures now, the uncut grid in Phase 1 and the cut adapter in Phase 2. The constructor then:

- matches edges into interior faces,
- glues periodic pairs,
- computes all geometry,
- orders the faces,
- checks the result.

No builder writes a face by hand, so no builder can produce faces that disagree with its polygons.

**4. One geometry kernel, with face vectors taken from coordinate differences.** Every derived quantity comes from the four functions under Geometry functions. Two details keep the closure property at round-off, even for slivers far from the origin.

- **Face vectors.** The face vector is `S = (y_b − y_a, x_a − x_b)`, computed directly from the endpoint coordinates and never as length times normal. Around a closed polygon the sum of these vectors telescopes. When the coordinates lie within a factor of two of each other, every partial sum is exact (Sterbenz) and the closure is exactly zero. Otherwise it is within a few ulps of the element size. The solver passes `S` straight into its flux calls, because Trixi-style fluxes take an unnormalized `normal_direction` and scale by its length.
- **Area and centroid.** These are computed about the polygon's first vertex, not about the origin. On a fine airfoil grid (h = 1e-3), a cut cell at x ≈ 1 lies about 1000 cells from the origin. In absolute coordinates each shoelace term is then about 1e6 cell areas, so the cancellation error is about ε·1e6 ≈ 2e-10 of a cell area, more than the area of a 1e-11 sliver. A quick check on this machine confirmed it: at that distance the absolute-coordinate shoelace returned zero for the area of a 1.75e-11 sliver. About the first vertex, the error is near ε of a cell area, and the same check matched a `BigFloat` reference exactly.

**5. Boundaries are named, not given a fixed kind.** The plan's face kind (interior, wall, farfield) becomes `boundary::Int32`: 0 on an interior face, otherwise an index into `boundary_names`.

- A cut mesh has the boundaries `:wall` and `:farfield`.
- An uncut test box names its sides `:x_neg, :x_pos, :y_neg, :y_pos`, as Trixi does. A test can then put different conditions on different sides.
- The solver maps names to boundary-condition objects through a NamedTuple, which is also Trixi's convention.
- Faces are stored interior first, then each boundary in `boundary_names` order. The boundary pass is then one type-stable loop per name.

**6. Periodicity is an interior face with a shift.** A periodic face is an interior face whose `shift` is nonzero. A point `x` on the left element's copy of the face is the point `x + shift` on the right element's copy, and the solver evaluates the right element's basis there. Builders request periodicity by naming a pair of sides to glue, and the constructor pairs their faces.

**7. The mesh does not carry the background grid.** Each element records two indices:

- `parent::Int32`, a linear index into whatever background the builder used, or 0 if there is none;
- `fragment::Int32`, which says which of the parent's fluid fragments it is.

The volume fraction is computed at construction from the element area and a parent area that the builder supplies. Nothing else about the grid enters the contract, so a later quadtree only changes what `parent` indexes.

**8. Default constructors are raw.** As in CartesianMeshes, the default `PolygonMesh` constructor stores its fields and checks nothing, so that a later Adapt conversion can call it with device arrays. All checking lives in the builder constructor and in `assert_mesh`. Every array field is a type parameter.

**9. Solver data stays out of the contract.** Quadrature points, basis coefficients and per-point normals are built from the mesh by `Semidiscretization` in Phase 1. Curved faces, later, change that solver-side face data, not the contract's topology.

## Types

Implement these. Field names may change to fit package style.

```julia
struct MeshElement{T}
    area::T
    centroid::SVector{2,T}
    lo::SVector{2,T}            # bounding box
    hi::SVector{2,T}
    volume_fraction::T          # area / parent area; exactly 1 for a full cell
    parent::Int32               # background cell, linear index; 0 if none
    fragment::Int32             # 1, 2, ... among the fluid fragments of `parent`
end

struct MeshFace{T}
    nodes::NTuple{2,Int32}      # (a, b): the left element runs a -> b
    left::Int32
    right::Int32                # 0 on a boundary face
    boundary::Int32             # 0 on an interior face, else an index into boundary_names
    vector::SVector{2,T}        # (y_b - y_a, x_a - x_b): length times unit normal, out of `left`
    length::T
    normal::SVector{2,T}        # vector / length
    shift::SVector{2,T}         # x on the left copy is x + shift on the right copy; zero unless periodic
end

struct PolygonMesh{T,P<:AbstractVector{SVector{2,T}},I<:AbstractVector{Int32},E,F}
    points::P                   # the one vertex pool; every point is used
    elem_ptr::I                 # CSR offsets, length nelements + 1
    elem_nodes::I               # each element's pool ids, counter-clockwise, first not repeated
    elem_faces::I               # elem_faces[k]: the face on edge elem_nodes[k] -> next node
    elements::E                 # StructVector{MeshElement{T}}
    faces::F                    # StructVector{MeshFace{T}}
    boundary_ptr::I             # boundary k is faces boundary_ptr[k]:boundary_ptr[k+1]-1
    boundary_names::Vector{Symbol}
end
```

Interior faces are `1:boundary_ptr[1]-1`, and `boundary_ptr` has length `nboundaries + 1`.

Minimum accessors:

- Counts: `nelements(m)`, `nfaces(m)` and `npoints(m)`.
- Element views: `element_nodes(m, e)` and `element_faces(m, e)`, both views into the CSR arrays.
- Face ranges: `interior_faces(m)` and `boundary_faces(m, name::Symbol)`, both `UnitRange`s.
- Orientation: `face_sign(m, e, f)` returns +1 where `e` is the left element of `f` and −1 where it is the right.

## Building a mesh

```julia
PolygonMesh(points, polygons, boundary_edges;
            periodic = (),       # e.g. ((:x_pos, :x_neg) => SVector(-2.0, 0.0),)
            parent = zeros(Int32, n), fragment = ones(Int32, n),
            parent_area = nothing) -> PolygonMesh{T}
```

The inputs:

- `points` is an `AbstractVector{SVector{2,T}}`.
- `polygons` holds one counter-clockwise id list per element.
- `boundary_edges` maps each boundary edge, as the directed pair `(a, b)` its element traverses, to a boundary name.
- A periodic entry `(:p, :q) => shift` glues side `:p`, which becomes the left side, to side `:q`, which becomes the right.
- `parent_area[e]` is the area of element `e`'s background cell. The builder computes it from the same node coordinates as the polygon, as `(x1 - x0) * (y1 - y0)`. With that rule a full rectangle's shoelace area equals its parent area bitwise, and its volume fraction is exactly 1. With no `parent_area`, every volume fraction is 1.
- On a `CartesianGrid`, take every vertex from `get_node`. Do not take the parent area from `get_elem_area`, which returns `prod(d)`, and do not take corners from `get_elem_bounds`, whose upper corner is `lo + d`. Neither agrees with the nodes bitwise. On a 37 × 23 grid over [−0.7, 1.3] × [−0.3, 0.9], `get_elem_area` differed from the node-based area in all 851 cells, and `lo + d` differed from the next node in 794. Built from `get_node`, every full cell had a volume fraction of exactly 1.

Steps, in this order. Each failure throws and names the element, its local edge `k` and the pool ids involved.

1. **Check each polygon.** It has at least 3 ids, all in range, with no id repeated. Its area is positive. No two non-adjacent edges intersect, and adjacent edges meet only at their shared vertex. Collinear vertices are legal.
2. **Match edges.** Directed edge `(a, b)` of one element pairs with `(b, a)` of another to form an interior face. The left element is the one with the lower id, and the face's nodes follow its direction. Each edge must fall into exactly one of three cases: matched once, named in `boundary_edges`, or glued in step 3. An edge that is matched twice, matched and also named, or neither matched nor named is an error.
3. **Glue periodic pairs.** Every face named `:p` must have exactly one face named `:q` whose endpoints equal its own reversed endpoints plus `shift`, within 1e-10 of the face length. The pair becomes one interior face, and the glued names are removed from `boundary_names`.
4. **Compute geometry.** Everything is computed with the geometry functions. Unused pool points are dropped, and the remaining ids are renumbered in pool order.
5. **Order faces.** Interior faces come first, in creation order, then each boundary in `boundary_names` order. By default `boundary_names` is the order in which names are first seen in `boundary_edges`. This step also fills `elem_faces` and `boundary_ptr`.
6. **Check.** Run `assert_mesh`.

Use one `Dict` keyed by directed edge for the matching, so building is linear in the number of edges.

## Geometry functions

```julia
polygon_area(points, ids) -> T                  # about points[ids[1]]; positive when counter-clockwise
polygon_centroid(points, ids) -> SVector{2,T}   # about points[ids[1]], then shifted back
polygon_bounds(points, ids) -> (lo, hi)
face_vector(pa, pb) -> SVector{2,T}             # (pb[2] - pa[2], pa[1] - pb[1])
```

Write `q_k = p_k − p_1`, with indices taken cyclically. Then:

- area: `A = ½ Σ q_k × q_{k+1}`;
- centroid: `c = p_1 + Σ (q_k + q_{k+1}) (q_k × q_{k+1}) / (6A)`.

`ids` may be any integer vector, including a view into `elem_nodes`. The functions are pure and allocation-free, and pass `@inferred`. The Phase 1 quadrature must take its points from the same pool.

## Mesh invariants

Follow EBM's `check_invariants`/`assert_invariants` pair:

- `check_mesh(m)` returns a NamedTuple with a violation count per invariant, the first offender of each, and `ok`.
- `assert_mesh(m) -> m` throws, listing the nonzero counts and their first offenders.

The invariants:

1. **CSR.** `elem_ptr` starts at 1, increases by at least 3 per element, and ends at `length(elem_nodes) + 1`. `elem_faces` has the same length as `elem_nodes`. All ids are in range, and every pool point is used.
2. **Polygons.** Each polygon has no repeated id, positive area, and is simple, as in build step 1.
3. **Edges agree with faces.** Take edge `(a, b)` of element `e`, with `f = elem_faces[k]`. Exactly one of these holds:
   - `e` is the left element of `f`, and `nodes[f] == (a, b)`;
   - `e` is the right element of `f`, `shift` is zero, and `nodes[f] == (b, a)`;
   - `f` is periodic, and the endpoints of `e`'s edge equal the face's endpoints, reversed and moved by `shift`, within the build tolerance.
4. **Each face is used exactly once per side.** A face appears once in its left element's list and, if it is an interior face, once in its right element's list. It appears in no other element, and left ≠ right. A boundary face has `right == 0` and `boundary ∈ 1:nboundaries`. An interior face has `right > 0` and `boundary == 0`.
5. **Face order.** Interior faces come first, and boundary `k` is exactly `boundary_ptr[k]:boundary_ptr[k+1]-1`. Names are unique.
6. **Geometry is derived.** Recompute every element's area, centroid, bounds and volume fraction, and every face's vector, length and normal, with the geometry functions. The results must be bitwise equal (`isequal`) to the stored values. This catches a builder that computed geometry its own way, and a mesh edited after it was built.
7. **Closure.** For each element, `|Σ_k s_k S_{f_k}| ≤ 1e-13 × perimeter`, where `s_k = face_sign(m, e, f_k)`. The 1e-13 matches CutCellMethods' `closure_tol`.
8. **Element data.** Each element has `0 < volume_fraction ≤ 1`, `parent ≥ 0` and `fragment ≥ 1`. Each pair `(parent, fragment)` with `parent > 0` occurs only once.

## Summary and VTK

`mesh_summary(m) -> NamedTuple` and `mesh_report(io, m) -> NamedTuple` follow EBM's summary/report pairs. The report prints with `@printf` and returns the same NamedTuple. Contents:

- Counts of points, elements, interior faces and periodic faces, and the number of faces per boundary name.
- Elements by type:
  - full: volume fraction exactly 1, and the only element of its parent;
  - cut: volume fraction below 1;
  - split: the parent has two or more elements.
- The minimum volume fraction and its element, plus a histogram of volume fractions by decade, from below 1e-12 up to [0.1, 1), with full cells counted separately.
- The largest relative closure error and its element.
- The total area, the length of each boundary, and `|Σ S|` over all boundary faces.
- The largest vertex count, and the number of nonconvex elements; a trailing-edge wedge shows up here.

Phase 2 adds lost-feature and truncated-tip lists to the cut preprocessor's own summary, not to this one.

`write_mesh_vtk(prefix, m)` writes two files with WriteVTK's PolyData, as EBM's `write_polydata` does:

- `prefix_elements.vtp` has one polygon per element, with cell data for area, volume fraction, parent, fragment, closure error and vertex count.
- `prefix_faces.vtp` has one line per face, with cell data for the boundary index (0 on interior faces), left, right, length, normal and a periodic flag.

This writer belongs in Phase 0 rather than in `Post`, because the first real meshes appear at the start of Phase 1. The closure-error field is also the quickest way to spot a bad cut in Phase 2.

## Fixtures

Put the fixtures in `test/fixtures.jl`. Each one returns a mesh built through the constructor from coordinates written out by hand.

1. **`two_triangles()`.** The unit square split along a diagonal, with four named sides: 2 elements, 1 interior face and 4 boundary faces.
2. **`hanging_node()`.** A pentagon covering [0,1]×[0,2] with an extra vertex at (1, 1), beside the two unit squares [1,2]×[0,1] and [1,2]×[1,2]. The collinear vertex is legal and splits the shared side into two faces. Totals: 3 interior faces and 7 boundary faces.
3. **`periodic_2x2()`.** Four unit squares on [0,2]², glued in x and in y: 8 interior faces, 4 of them periodic with shift ±2, and no boundary faces.
4. **`tilted_wall(; y0=0.25, y1=0.85, offset=(0, 0))`.** A 2×2 background on [0,2]² cut by a straight wall from (0, y0) to (2, y1), with solid below it.
   - The crossing at x = 1 is one pool vertex, used by both bottom fragments.
   - The result is two cut elements and two full ones: 4 interior faces and 8 boundary faces, 2 of them `:wall`.
   - `offset` moves everything.
   - The sliver case is `y0 = 1 − 2δ`, `y1 = 1 − δ` with δ = 1e-11 and `offset = (1000, 1000)`, so the cells sit 1000 cells from the origin as in decision 4. The two volume fractions are then near 1.75δ and 1.25δ.
5. **`split_cell()`.** The single background cell [0,1]², crossed by a solid band 0.4 < y < 0.6. This gives two elements of volume fraction 0.4, with parent 1 and fragments 1 and 2. Each element has one `:wall` face, and no face joins the two.

Each of the following invalid inputs must throw an `ArgumentError` whose message names the element or the face:

- a clockwise polygon;
- a polygon with a repeated id, or with only two ids;
- a self-intersecting bow-tie;
- `two_triangles` with one triangle removed and the diagonal left unnamed;
- three elements on one edge;
- a duplicated pool point: the same coordinates under a new id, used by one side of a shared edge. This is the case the shared-split-point rule exists to prevent.
- an edge that is both matched and named as a boundary;
- a periodic pair whose sides do not match under the shift;
- `assert_mesh` called on a valid mesh after one of its pool points has moved, leaving stale geometry.

## The Trixi.jl decision

Load times measured on this machine with Julia 1.12.6, warm, in scratch environments:

| Environment | `using` time | Packages in manifest |
| --- | --- | --- |
| Trixi 0.17.13 alone | 5.2 s | 228 |
| Geometry stack: SDFLibrary, CutCellMethods, CartesianMeshes, StaticArrays, StructArrays, WriteVTK | 4.2 s | — |
| Geometry stack, then Trixi | 2.5 s more | 306 |

Precompiling Trixi takes 31 s on its own, and it brings MPI, HDF5, P4est and T8code binaries.

What EulerDG would use from Trixi:

- `CompressibleEulerEquations2D`, `prim2cons` and `cons2prim`;
- the physical `flux`;
- `FluxLaxFriedrichs` (Rusanov), `flux_hllc` and `FluxHLL`;
- `boundary_condition_slip_wall`.

All of these accept an unnormalized `normal_direction`, which matches decision 4; this was checked on 0.17.13. Trixi has no characteristic farfield condition for Euler, so Phase 4 writes one either way.

**Decision: copy the functions; do not depend on Trixi.** The functions listed are a few hundred lines. Depending on Trixi would add 2.5 s to every load and every test run. It would also bring a 228-package tree, with frequent breaking releases, into shared environments such as `fvm`; your local clone is at 0.13 while the registry is at 0.17.13. EBM.jl wrote its own fluxes too.

How to copy:

- Put the copies in `src/Equations/`. The file header carries Trixi's MIT copyright notice and the source version.
- Keep Trixi's names, its argument order `(u_ll, u_rr, normal_direction, equations)` and its equation type. Switching to the dependency later is then a change of `using` lines.
- Cross-check the copies against Trixi in a script under `dev/` that has its own environment, so that `Pkg.test` never loads Trixi.

In Phase 0, record the decision in the plan's Open decisions table and in `dev-notes/phase0.md`. The copying happens in Phase 1, when the first flux is needed.

## Test harness

Copy EBM's tiers:

- `const TIER = isempty(ARGS) ? get(ENV, "EULERDG_TEST_TIER", "commit") : first(ARGS)`, accepting `commit` or `scheduled` and erroring on anything else.
- One `@testset` per file, with shared fixtures in `test/fixtures.jl`.
- Phase gates live in `test/scheduled/test_phaseN_gate.jl` from Phase 1 on, and run only in the scheduled tier. The Phase 0 gate is the commit tier itself.
- Baselines, and `EULERDG_UPDATE_BASELINES=1`, arrive with the first baseline in Phase 4.

Phase 0 test files:

- `test_geometry.jl`
- `test_mesh.jl`
- `test_summary.jl`
- `test_vtk.jl`
- `test_layering.jl`

## Tests and acceptance criteria

The tests must cover the following:

1. **Geometry functions.** Test on a unit square, a triangle and a nonconvex L-shape, and on the same shapes scaled to h = 1e-3 and moved to (1, 1). Each function must allocate nothing and pass `@inferred`.
2. **Fixtures.**
   - Element, face and boundary-face counts are as listed above.
   - Areas and volume fractions are exact on the axis-aligned fixtures.
   - Closure is exactly zero on the axis-aligned fixtures, and at most 1e-13 × perimeter on `tilted_wall`, including the sliver case.
3. **Sliver precision.** At offset (1000, 1000), the sliver areas match a `BigFloat` shoelace of the same `Float64` points to a relative 1e-12. Also test that a plain `Float64` shoelace in absolute coordinates misses them by more than 50%; the quick check above measured 100%. This test is the reason for decision 4.
4. **Periodicity.** The shift is ±2 on the four glued faces of `periodic_2x2` and zero elsewhere. On every periodic face, the right element's edge endpoints equal the left's, reversed, plus the shift.
5. **Invalid inputs.** Every case listed under Fixtures throws, with the id in the message. Check this with `@test_throws` and a check on the message text.
6. **Summary.**
   - `split_cell` reports 2 split elements with one parent.
   - `tilted_wall` reports 2 cut elements and 2 full ones.
   - The sliver case reports the right minimum volume fraction and its element.
7. **VTK.** Both files are written for every fixture. Open them in ParaView once by hand.
8. **Layering.** The scan passes on `src/`, and it catches a planted `CartesianIndex` in a test string.

Acceptance criteria:

- The commit tier passes on Julia 1.12.6.
- The load time of `using EulerDG` is recorded in `dev-notes/phase0.md`. The only new dependencies are StaticArrays, StructArrays and WriteVTK, plus standard libraries.
- Every exported name has a docstring in EBM's style.
- The four settled decisions are entered in the plan document:
  - its Open decisions table;
  - the background-mesh row of its scope table;
  - its Phase 0 checklist, from which the run-configuration item moves to Phase 4;
  - its Phase 1 grid item, which drops "stretched".
- Every conflict with this brief or with the plan is listed in `dev-notes/phase0.md`.

## Deferred to later phases

Do not implement any of these now, and do not design anything that would block them.

- The uncut builder on a `CartesianGrid` (Phase 1), and the cut adapter (Phase 2).
- The run-configuration format (Phase 4). A draft was considered on Oct 8 and is a starting point: TOML files that mirror `@kwdef` structs one-to-one, strict loading, a version key, and a `config.toml` written into every run's output directory.
- Stretched background grids. See item 1 of Findings for later phases.
- Solver data for elements and faces: quadrature points and weights, basis coefficients and per-point normals (Phase 1).
- `Float32` conversion and Adapt support for the mesh. Array fields are already type parameters and the default constructor is raw, so this later costs one `adapt_structure`.
- Curved faces and a two-segment trailing-edge wedge. The contract already allows nonconvex polygons and any number of wall faces per element.

## Findings for later phases

Reading the sibling packages turned up the following. None of it changes Phase 0, but each item changes the plan for a later phase.

1. **Grids stay uniform for now.** `CartesianGrid` is uniform, and `PolylineClippingCutCell` accepts only a `CartesianGrid`. The Oct 8 decision keeps that restriction, which departs from the plan's stretched tensor-product background. It will matter by Phase 5, because a farfield at 50–100 chords needs stretching. When stretching comes back, there are two routes:
   - An EulerDG-owned tensor-product grid with a uniform core box around the body. The core is cut by the existing code, every cell outside it is full, and the core nodes must equal `CartesianGrid`'s node formula bitwise.
   - A rectilinear extension of CutCellMethods. Its host-side lattice already stores grid lines as coordinate vectors (`PolylineLattice.xs/ys`), so the extension may be small.
2. **Crossings come from exact polyline intersection, not from a root-find on the SDF.** The cutter takes only an `SDFMesh`, a closed counter-clockwise polyline, and finds crossings with exact predicates. The circle is therefore a polygon. Its segment length must shrink with h; the cutter's brief suggests Δs ≤ h/4. Otherwise the Phase 4 convergence study stalls at the polygon's own error.
3. **Lost features cannot occur with this cutter.** Exact boundary walking reproduces trailing edges thinner than a cell and reports each fragment of a split cell. The plan's lost-feature check targets classification by vertex sign, so keep it only as a cross-check.
4. **A fragment's wall can be several segments.** The wall is the part of the polyline inside the cell, so polyline vertices inside a cell give several wall edges. The plan assumes one straight segment per fragment. The contract allows both; Phase 2 decides whether to keep the pieces or collapse them to one chord.
5. **Face adjacency must come from the cutter's arcs.** EBM's `cut` keeps an outline per fragment but drops the adjacency between fragments. The adapter needs `CutCellMethods.arcs`, the fluid intervals on each grid edge with the region that owns each side. Its endpoints are shared bitwise, so both sides of every partial edge can use the same pool vertices.
6. **Safeguard rung 4 already exists.** The cutter drops slivers below `drop_area · h²` (default 1e-14) and closes their edges on both sides. Make `drop_area` a keyword of the cut builder; the plan says about 1e-12.
7. **The body needs clear cells around it.** The cutter throws unless at least one cell separates the body from every side of the grid.
