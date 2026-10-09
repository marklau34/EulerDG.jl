# =====================================
# The nested builder
#
# Option 3 of the far-boundary study (Oct 9): the body is cut on a uniform finest box, exactly as by
# `cut_mesh`, and the box is wrapped in coarser and coarser rings of uncut cells out to the far
# field. The levels are a CartesianMeshes `NestedGrids` chain: each level doubles the cell size of
# the one inside it, and each ring is at least `buffer` of its own cells thick. The cells stay square
# (or keep the finest grid's aspect ratio), so each element's local step grows with it and the
# explicit solver's iteration count stays close to the finest box's; a stretched tensor-product grid
# would leave long, thin cells beside the box whose steps are set by their short side.
#
# - **Hanging nodes.** A coarse cell beside finer cells takes their shared node as a vertex on its
#   side, so every polygon edge is one face and the contract's builder pairs them as usual.
# - **One formula for every vertex.** Every vertex, on every level, is a node of the finest level,
#   and its coordinate comes from the finest patch's own `get_node` with that node's index -- the
#   formula `cut_mesh` uses for the patch -- so a vertex shared by two levels is bitwise one point.
# - **The body stays inside.** It must lie in the finest box with full fluid cells all round the
#   box's edge, so no cut reaches a coarse-fine interface.

"""
    nested_cut_mesh(grid, body; farfield, buffer=8, origin=grid's lower corner, drop_area=1e-12)
        -> (; mesh, report, hierarchy)

Cut `body` on the uniform 2D `grid`, as [`cut_mesh`](@ref) does, and wrap the grid's box in coarser
levels out to `farfield`, a box `(lo, hi)`:

- each level doubles the cell size of the one inside it and surrounds it with a ring at least
  `buffer` of its own cells thick, snapped outward to whole cells of the level outside it;
- every level's cells are aligned to `origin`: it is a node of every level. Put it on a symmetry
  line of the body, with `grid` and `farfield` symmetric about it too, and the whole mesh is
  symmetric;
- levels are added until a ring reaches `farfield`; that level, the coarsest, fills the domain:
  `farfield` and that ring, snapped outward to its cells;
- a coarse cell is a rectangle with a hanging node on each side that meets finer cells;
- `parent` numbers the grid's cells as `cut_mesh` does, and the coarse cells after them;
- the boundaries are `:wall`, then the domain sides `:x_neg`, `:x_pos`, `:y_neg` and `:y_pos`.

`grid` needs an even number of cells along each axis, and its lower corner must lie an even number
of cells from `origin`, so that it is made of whole cells of the next level; `body` must lie inside
it with full fluid cells all round its edge. `report` is `cut_mesh`'s, over the whole domain, plus `domain`, its
corners, and `levels`: for each level, coarsest first, its `level`, cell size `h` and number of
`elements`. `hierarchy` is the CartesianMeshes `CartesianGridHierarchy` of the levels.
"""
function nested_cut_mesh(grid::CartesianGrid{2,Float64}, body::Body; farfield,
                         buffer::Integer=8, origin=grid.x0, drop_area::Real=1e-12)
    all(iseven, grid.n) ||
        throw(ArgumentError("the grid has $(grid.n) cells; the nested builder needs an even " *
                            "number along each axis, so that it is made of whole coarser cells"))
    buffer >= 1 || throw(ArgumentError("buffer must be at least 1, not $buffer"))
    O, h, n = PV(grid.x0), PV(grid.d), SVector{2,Int}(grid.n)
    flo, fhi = PV(farfield[1]), PV(farfield[2])
    (all(flo .<= O) && all(fhi .>= O .+ n .* h)) ||
        throw(ArgumentError("farfield $((Tuple(flo), Tuple(fhi))) must contain the grid's box " *
                            "$((Tuple(O), Tuple(O .+ n .* h)))"))
    # The grid's corner, in cells from the origin: an even whole number along each axis.
    o = PV(origin)
    k = (O - o) ./ h
    k0 = round.(Int, k)
    (all(abs.(k .- k0) .<= 1e-8) && all(iseven, k0)) ||
        throw(ArgumentError("the grid's lower corner $(Tuple(O)) lies $(Tuple(k)) cells from " *
                            "the origin $(Tuple(o)); it must be an even whole number along each axis"))
    hierarchy = _nested_hierarchy(o, h, k0, n, flo, fhi, Int(buffer))

    L = nlevels(hierarchy)
    pf = only(patchids(hierarchy, L))
    size(elementbox(hierarchy, pf)) == grid.n ||
        error("the finest level's patch is $(size(elementbox(hierarchy, pf))) cells, not $(grid.n)")
    patch = subgrid(hierarchy, pf)
    c = _cut_polygons(patch, body, Float64(drop_area))
    _check_inside(patch, body, c.info)
    filter!(kv -> kv.second == :wall, c.names)     # the patch's sides now face coarse cells

    # The finest level's node with global index G, through the patch's own grid.
    F = first(elementbox(hierarchy, pf))
    node(G) = PV(get_node(patch, CartesianIndex(G[1] - F[1] + 1, G[2] - F[2] + 1)))
    nf = SVector{2,Int}(size(CartesianMeshes.grid(hierarchy, L)))
    side(a, b) = a[1] == b[1] == 1 ? :x_neg : a[1] == b[1] == nf[1] + 1 ? :x_pos :
                 a[2] == b[2] == 1 ? :y_neg : a[2] == b[2] == nf[2] + 1 ? :y_pos : nothing

    counts = zeros(Int, L)
    counts[L] = length(c.polygons)
    parent = length(CartesianIndices(patch))
    outward(k) = k == 1 ? (0, -1) : k == 2 ? (1, 0) : k == 3 ? (0, 1) : (-1, 0)
    for l in 1:(L - 1)
        r = 2^(L - l)
        for I in elementbox(hierarchy, only(patchids(hierarchy, l)))
            is_covered(hierarchy, l, I) && continue
            a = SVector((I[1] - 1) * r + 1, (I[2] - 1) * r + 1)
            corners = (a, a + SVector(r, 0), a + SVector(r, r), a + SVector(0, r))
            G = SVector{2,Int}[]
            for k in 1:4
                push!(G, corners[k])
                # A side facing a finer level's cells is two of their faces.
                is_covered(hierarchy, l, I + CartesianIndex(outward(k))) &&
                    push!(G, (corners[k] + corners[mod1(k + 1, 4)]) .÷ 2)
            end
            ids = Int32[c.id!(node(g)) for g in G]
            for k in eachindex(G)
                name = side(G[k], G[mod1(k + 1, length(G))])
                name === nothing || (c.names[(ids[k], ids[mod1(k + 1, length(G))])] = name)
            end
            lo, hi = node(corners[1]), node(corners[3])
            parent += 1
            push!(c.polygons, ids); push!(c.parents, parent); push!(c.fragments, 1)
            push!(c.areas, (hi[1] - lo[1]) * (hi[2] - lo[2]))
            counts[l] += 1
        end
    end

    mesh = PolygonMesh(c.points, c.polygons, c.names; parent=c.parents, fragment=c.fragments,
                       parent_area=c.areas, boundary_names=CUT_BOUNDARIES)
    domain = (node(SVector(1, 1)), node(nf .+ 1))
    report = _cut_report(patch, body, c.info, c.cache, mesh, c.closed, Float64(drop_area); domain)
    levels = [(; level=l, h=PV(CartesianMeshes.grid(hierarchy, l).d), elements=counts[l])
              for l in 1:L]
    return (; mesh, report=(; report..., domain, levels), hierarchy)
end

# The levels as a `NestedGrids` chain. Boxes are worked out in finest cells from the origin `O`, as
# integers, so the rings and their snapping are exact and every level's cells are aligned to `O`;
# only the base grid's corner is rounded, and `NestedGrids` snaps the physical boxes back onto its
# cells. The finest box is cells `k0` to `k0 + n`.
function _nested_hierarchy(O, h, k0, n, flo, fhi, buffer)
    ilo = floor.(Int, (flo - O) ./ h)
    ihi = ceil.(Int, (fhi - O) ./ h)
    boxes = [(k0, k0 .+ n)]                            # finest first
    s = 1
    while true
        s *= 2
        glo, ghi = boxes[end][1] .- buffer * s, boxes[end][2] .+ buffer * s
        if all(glo .<= ilo) && all(ghi .>= ihi)
            # This level reaches the far field: it is the base, the far field and the ring
            # snapped outward to its cells.
            dlo, dhi = fld.(min.(glo, ilo), s) .* s, cld.(max.(ghi, ihi), s) .* s
            base = CartesianGrid{2,Float64}(O .+ dlo .* h, Tuple((dhi .- dlo) .÷ s), s .* h)
            physical = [(O .+ b[1] .* h, O .+ b[2] .* h) for b in reverse(boxes)]
            spec = NestedGrids(base, physical; ratio=2, buffer, minsize=1)
            return CartesianGridHierarchy(spec; verbose=false)
        end
        # A ring of this level's cells, snapped outward to whole cells of the next level out.
        push!(boxes, (fld.(glo, 2s) .* 2s, cld.(ghi, 2s) .* 2s))
    end
end

# The body must lie inside the cut grid, with full fluid cells all round its edge.
function _check_inside(grid, body, info)
    cells = CartesianIndices(grid)
    n1, n2 = grid.n
    for ci in cells
        (ci[1] in (1, n1) || ci[2] in (1, n2)) || continue
        info[ci].status == PL_FLUID ||
            throw(ArgumentError("cell $(Tuple(ci)) on the edge of the finest grid is not a full " *
                                "fluid cell: the body must lie inside it with fluid all round"))
    end
    lo, hi = PV(get_node(grid, first(cells))), PV(get_node(grid, last(cells) + CartesianIndex(1, 1)))
    for (k, v) in enumerate(body.points)
        (all(lo .< v) && all(v .< hi)) ||
            throw(ArgumentError("body vertex $k at $(Tuple(v)) lies outside the finest grid"))
    end
    return nothing
end
