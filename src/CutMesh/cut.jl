# =====================================
# The cut builder
#
# CutCellMethods' exact polyline clipping cuts the grid against the body. Its region loops are the
# polygons; this file turns them into a `PolygonMesh`. Three things make that work:
#
# - **Canonical points.** The cutter stores each crossing once per grid edge, as an offset along
#   the edge, and walks each cell in local coordinates from its lower corner. A loop point is that
#   corner plus the local offset, so the two cells sharing an edge compute the same coordinate along
#   the edge, but the edge line's own coordinate (`lo + h`) can differ from the node value in the
#   last bit. Snapping every loop coordinate within round-off of the cell's node lines onto them
#   makes shared points bitwise equal, so they become one pool vertex.
# - **Naming.** A loop edge that does not run along a side of its cell is a piece of the body: a
#   wall. A loop edge along a side pairs with the neighbour's, except where a dropped sliver's
#   interval was closed (the cutter closes it on both sides, moving the wall to the fluid
#   neighbour) or on the domain boundary. Everything else must pair, and the builder throws if it
#   does not -- a gap is never quietly turned into a wall.
# - **Lost features.** Exact clipping reproduces a trailing edge thinner than a cell, but a cell the
#   cutter could not walk, a body loop wholly inside one cell, or a body vertex in an uncut fluid
#   cell all stop the build, and so does fluid area missing more than the dropped slivers explain.

using CutCellMethods: PL_CUT, PL_FLUID, PL_INVALID, PL_SOLID, PL_SPLIT

const PV = SVector{2,Float64}

# A coordinate within round-off of a cell's node line, snapped onto it.
@inline function _snap(v, a, b)
    tol = 4 * eps(max(abs(a), abs(b)))
    return abs(v - a) <= tol ? a : abs(v - b) <= tol ? b : v
end

# -0.0 and 0.0 must be one pool vertex.
@inline _key(p::PV) = PV(p[1] + 0.0, p[2] + 0.0)

"""
    cut_mesh(grid, body; drop_area=1e-12) -> (; mesh, report)

Cut the uniform 2D `CartesianGrid` against `body` with CutCellMethods' exact polyline clipping, and
build the `PolygonMesh` of the fluid:

- every fluid cell is one element, and every fluid fragment of a cut cell another, numbered in
  cell order and then fragment order, with `parent` the cell's linear index and `fragment` the
  cutter's region number;
- crossings shared by two cells are one pool vertex, so the faces between fragments pair exactly;
- the boundaries are `:wall`, then the domain sides `:x_neg`, `:x_pos`, `:y_neg` and `:y_pos`;
- a fluid fragment smaller than `drop_area` times h², h the smaller grid spacing, is dropped and
  its interval closed, so the wall moves onto its neighbour (safeguard rung 4 of the plan).

Throws, naming the cell, if the cutter could not walk a cell, a body loop lies wholly inside one
cell, a body vertex lies in a cell left uncut, or the fluid area misses more than the dropped
slivers explain. `report` is described in [`cut_report`](@ref).
"""
function cut_mesh(grid::CartesianGrid{2,Float64}, body::Body; drop_area::Real=1e-12)
    cache = allocate_cache(grid, PolylineClippingCutCell(; drop_area=Float64(drop_area)))
    update_cache!(cache, body.sdf, grid)
    info = cache.info
    cells = CartesianIndices(grid)
    for ci in cells
        info[ci].status == PL_INVALID &&
            throw(ArgumentError("the cutter could not walk cell $(Tuple(ci)), flags " *
                                "$(info[ci].flags): too many crossings, or a degenerate cut; " *
                                "refine or shift the body"))
    end
    loops = CutCellMethods.region_loops(cache, grid)
    for l in loops
        l.hole && throw(ArgumentError("a body loop lies wholly inside cell $(Tuple(l.cell)); " *
                                      "the grid is too coarse to resolve it"))
    end
    _check_vertices(grid, body, info)

    node(ci) = PV(get_node(grid, ci))
    points, pool = PV[], Dict{PV,Int32}()
    id!(p) = get!(() -> (push!(points, p); Int32(length(points))), pool, _key(p))

    # Each closed interval, by its canonical ends.
    closed = Set{NTuple{2,Int32}}()
    for ci in cells, a in CutCellMethods.arcs(cache, ci)
        (a.closed && a.s1 > a.s0) || continue
        lo, hi = node(ci), node(ci + CartesianIndex(1, 1))
        p0, p1 = _arc_ends(lo, hi, Int(a.dir), a.s0, a.s1)
        push!(closed, minmax(id!(p0), id!(p1)))
    end

    byc = Dict{CartesianIndex{2},Vector{eltype(loops)}}()
    for l in loops
        push!(get!(byc, l.cell, eltype(loops)[]), l)
    end

    polygons, parents, fragments, areas = Vector{Int32}[], Int32[], Int32[], Float64[]
    names = Dict{NTuple{2,Int32},Symbol}()
    x0, x1 = node(first(cells)), node(last(cells) + CartesianIndex(1, 1))
    lin = LinearIndices(grid)
    for ci in cells
        lo, hi = node(ci), node(ci + CartesianIndex(1, 1))
        area = (hi[1] - lo[1]) * (hi[2] - lo[2])
        st = info[ci].status
        if haskey(byc, ci)
            for l in sort(byc[ci]; by=l -> l.region)
                ids = _dedupe!([id!(PV(_snap(q[1], lo[1], hi[1]), _snap(q[2], lo[2], hi[2])))
                                for q in l.points])
                problem = _polygon_problem(points, ids)
                problem === nothing ||
                    throw(ArgumentError("cell $(Tuple(ci)), fragment $(l.region): $problem"))
                _name_edges!(names, points, ids, lo, hi, x0, x1, closed; cut=true)
                push!(polygons, ids); push!(parents, lin[ci]); push!(fragments, l.region)
                push!(areas, area)
            end
        elseif st == PL_FLUID || (st == PL_CUT && info[ci].slot == 0)
            # A full cell; with status CUT, a neighbour's sliver closed one of its edges.
            ids = Int32[id!(node(ci)), id!(node(ci + CartesianIndex(1, 0))),
                        id!(node(ci + CartesianIndex(1, 1))), id!(node(ci + CartesianIndex(0, 1)))]
            _name_edges!(names, points, ids, lo, hi, x0, x1, closed; cut=false)
            push!(polygons, ids); push!(parents, lin[ci]); push!(fragments, 1); push!(areas, area)
        elseif st != PL_SOLID
            throw(ArgumentError("cell $(Tuple(ci)) has status $st but no fluid loop"))
        end
    end
    mesh = PolygonMesh(points, polygons, names; parent=parents, fragment=fragments,
                       parent_area=areas, boundary_names=[:wall, :x_neg, :x_pos, :y_neg, :y_pos])
    report = _cut_report(grid, body, info, cache, mesh, closed, Float64(drop_area))
    return (; mesh, report)
end

# The ends of the stretch [s0, s1] of face `dir` of the cell with corners `lo`, `hi`, canonical.
function _arc_ends(lo, hi, dir, s0, s1)
    P(s) = dir == 1 ? PV(lo[1], _snap(lo[2] + s, lo[2], hi[2])) :
           dir == 2 ? PV(hi[1], _snap(lo[2] + s, lo[2], hi[2])) :
           dir == 3 ? PV(_snap(lo[1] + s, lo[1], hi[1]), lo[2]) :
                      PV(_snap(lo[1] + s, lo[1], hi[1]), hi[2])
    return P(s0), P(s1)
end

# Drop consecutive repeats, the last against the first too: a body vertex on a grid line is also a
# crossing there.
function _dedupe!(ids)
    out = Int32[]
    for id in ids
        (isempty(out) || out[end] != id) && push!(out, id)
    end
    length(out) > 1 && out[end] == out[1] && pop!(out)
    return out
end

# Name the boundary edges of one polygon of the cell with corners `lo`, `hi`: closed intervals
# and, in a cut fragment, edges off the cell's sides are wall; edges on the domain boundary are its
# sides. Edges along a side inside the domain are left to pair with the neighbour.
function _name_edges!(names, points, ids, lo, hi, x0, x1, closed; cut::Bool)
    n = length(ids)
    for k in 1:n
        a, b = ids[k], ids[mod1(k + 1, n)]
        pa, pb = points[a], points[b]
        side = (pa[1] == pb[1] && (pa[1] == lo[1] || pa[1] == hi[1])) ||
               (pa[2] == pb[2] && (pa[2] == lo[2] || pa[2] == hi[2]))
        name = if minmax(a, b) in closed || (cut && !side)
            :wall
        elseif pa[1] == pb[1] == x0[1]
            :x_neg
        elseif pa[1] == pb[1] == x1[1]
            :x_pos
        elseif pa[2] == pb[2] == x0[2]
            :y_neg
        elseif pa[2] == pb[2] == x1[2]
            :y_pos
        else
            nothing
        end
        name === nothing || (names[(a, b)] = name)
    end
    return nothing
end

# Every body vertex strictly inside a cell must be in a cell the cutter cut (or one all sliver,
# dropped to solid); in an uncut fluid cell it would be a lost feature.
function _check_vertices(grid, body, info)
    for (k, v) in enumerate(body.points)
        s = (v - grid.x0) ./ grid.d
        i, j = floor(Int, s[1]) + 1, floor(Int, s[2]) + 1
        (1 <= i <= grid.n[1] && 1 <= j <= grid.n[2]) || continue
        f = s - PV(i - 1, j - 1)
        min(f[1], f[2], 1 - f[1], 1 - f[2]) > 1e-9 || continue
        info[i, j].status == PL_FLUID &&
            throw(ArgumentError("body vertex $k at $(Tuple(v)) lies in cell ($i, $j), which the " *
                                "cutter left uncut and fluid: a lost feature"))
    end
    return nothing
end

# =====================================
# The preprocessor report

# The signed area of a closed polygon, about its first point.
function _loop_area(pts)
    p1 = pts[1]
    return sum(_cross(pts[k] - p1, pts[k + 1] - p1) for k in 2:(length(pts) - 1)) / 2
end
@inline _cross(a, b) = a[1] * b[2] - a[2] * b[1]

function _cut_report(grid, body, info, cache, mesh, closed, drop_area)
    cells = CartesianIndices(grid)
    status = [info[ci].status for ci in cells]
    s = mesh_summary(mesh)
    lo, hi = PV(get_node(grid, first(cells))), PV(get_node(grid, last(cells) + CartesianIndex(1, 1)))
    domain_area = (hi[1] - lo[1]) * (hi[2] - lo[2])
    body_area = _loop_area(body.points)
    ncut = count(st -> st == PL_CUT || st == PL_SPLIT, status)
    defect = domain_area - body_area - s.area
    allowance = ncut * drop_area * minimum(grid.d)^2 + 1e-10 * domain_area
    abs(defect) <= allowance ||
        throw(ArgumentError("the fluid area misses $defect of the body's complement, more than " *
                            "the dropped slivers explain ($allowance): a lost feature"))
    count_by_parent = Dict{Int32,Int}()
    for p in mesh.elements.parent
        count_by_parent[p] = get(count_by_parent, p, 0) + 1
    end
    split = sort([cells[p] for (p, n) in count_by_parent if n > 1]; by=ci -> LinearIndices(grid)[ci])
    e = s.min_volume_fraction_element
    return (; cells=(; fluid=count(==(PL_FLUID), status), cut=count(==(PL_CUT), status),
                     split=count(==(PL_SPLIT), status), solid=count(==(PL_SOLID), status),
                     snapped=count(ci -> info[ci].flags & CutCellMethods.PL_FLAG_SNAPPED != 0,
                                   cells)),
            elements=s.nelements, full=s.nfull, fragments=s.ncut + s.nsplit, split_cells=split,
            min_volume_fraction=s.min_volume_fraction,
            min_volume_fraction_cell=cells[mesh.elements.parent[e]],
            fluid_area=s.area, body_area, domain_area, area_defect=defect,
            wall_length=s.boundaries.wall.length, body_perimeter=polygon_perimeter(body.points),
            closed_intervals=length(closed), nonconvex=s.nnonconvex, max_closure=s.max_closure,
            drop_area)
end

"""
    cut_report([io=stdout,] cut) -> report

Print the report of a [`cut_mesh`](@ref) result and return it. The report is a NamedTuple:

- `cells` -- background cells by the cutter's status: `fluid`, `cut` (one fragment), `split` (two
  or more), `solid`, and `snapped`, those a dropped sliver's interval was closed onto;
- `elements`, `full`, `fragments`, and `split_cells`, the cells with two or more fragments;
- `min_volume_fraction` and `min_volume_fraction_cell`;
- `fluid_area`, `body_area` (the polygon's), `domain_area` and `area_defect`, the domain minus body
  minus fluid: the dropped slivers' area, which `cut_mesh` checks;
- `wall_length` against `body_perimeter`, `closed_intervals`, `nonconvex`, `max_closure` and
  `drop_area`.

Every lost-feature check passed, or `cut_mesh` would have thrown.
"""
function cut_report(io::IO, cut::NamedTuple)
    r = cut.report
    c = r.cells
    @printf(io, "Cut mesh: %d elements (%d full, %d fragments) on a background of %d cells\n",
            r.elements, r.full, r.fragments, c.fluid + c.cut + c.split + c.solid)
    @printf(io, "  cells: %d fluid, %d cut, %d split, %d solid; %d snapped onto by a dropped sliver\n",
            c.fluid, c.cut, c.split, c.solid, c.snapped)
    @printf(io, "  smallest volume fraction %.3e, in cell %s\n", r.min_volume_fraction,
            Tuple(r.min_volume_fraction_cell))
    @printf(io, "  fluid area %.12e; domain %.12e, body %.12e; defect %.2e (dropped slivers)\n",
            r.fluid_area, r.domain_area, r.body_area, r.area_defect)
    @printf(io, "  wall length %.12e, body perimeter %.12e; %d closed intervals\n", r.wall_length,
            r.body_perimeter, r.closed_intervals)
    @printf(io, "  %d split cells, %d nonconvex elements; largest closure error %.2e\n",
            length(r.split_cells), r.nonconvex, r.max_closure)
    println(io, "  lost-feature checks passed")
    return r
end

cut_report(cut::NamedTuple) = cut_report(stdout, cut)
