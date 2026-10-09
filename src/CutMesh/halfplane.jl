# =====================================
# The half-plane builder
#
# A uniform grid cut by one straight line, the fluid on one side: the domain the plan's Phase 3
# tests need -- uniform flow along a wall tilted off the grid axes, and a manufactured solution on
# a domain cut by a tilted wall. The polygon cutter cannot make it, because its body must stay
# clear of the grid's edge; here the wall runs right across the box.
#
# Each cell is clipped against the line (Sutherland-Hodgman). A crossing is computed once per grid
# edge from the edge's own coordinate, so the two cells sharing the edge get it bitwise equal, and
# a crossing within round-off of a node is snapped onto it. Fragments below `drop_area` are
# dropped like the cutter's slivers, and the faces their neighbours shared with them become wall.

"""
    halfplane_mesh(grid; point, normal, drop_area=1e-12) -> PolygonMesh

The fluid part of a uniform 2D `CartesianGrid` on the side of the line through `point` that its
`normal` points to, as a `PolygonMesh`. The boundaries are `:wall` (the line, and any face a
dropped fragment closed), then the domain sides `:x_neg`, `:x_pos`, `:y_neg` and `:y_pos`.
Elements follow the cells' linear order, with `parent` the cell's linear index; a fragment below
`drop_area` times h² (h the smaller spacing) is dropped.
"""
function halfplane_mesh(grid::CartesianGrid{2,Float64}; point, normal, drop_area::Real=1e-12)
    p = PV(point)
    n = PV(normal) / hypot(normal...)
    abs(n[1]) > 0 && abs(n[2]) > 0 ||
        throw(ArgumentError("the line must be tilted off the grid axes; its normal is $(Tuple(n))"))
    n1, n2 = grid.n
    node(i, j) = PV(get_node(grid, CartesianIndex(i, j)))
    phi = [_dot(node(i, j) - p, n) for i in 1:(n1 + 1), j in 1:(n2 + 1)]
    inside(i, j) = phi[i, j] >= 0
    # The crossing on the vertical edge from node (i, j) up, and on the horizontal one to the right.
    function xing(i, j, vertical)
        a, b = node(i, j), vertical ? node(i, j + 1) : node(i + 1, j)
        if vertical
            y = p[2] - (a[1] - p[1]) * n[1] / n[2]
            return PV(a[1], _snap(clamp(y, a[2], b[2]), a[2], b[2]))
        else
            x = p[1] - (a[2] - p[2]) * n[2] / n[1]
            return PV(_snap(clamp(x, a[1], b[1]), a[1], b[1]), a[2])
        end
    end

    points, pool = PV[], Dict{PV,Int32}()
    onwall = Bool[]
    function id!(q, wall)
        k = get(pool, _key(q), Int32(0))
        if k == 0
            push!(points, q)
            push!(onwall, wall)
            k = pool[_key(q)] = Int32(length(points))
        end
        onwall[k] |= wall
        return k
    end

    h2 = minimum(grid.d)^2
    lin = LinearIndices(grid)
    polygons, parents, areas = Vector{Int32}[], Int32[], Float64[]
    dropped = Set{NTuple{2,Int32}}()
    for j in 1:n2, i in 1:n1
        # The corners counter-clockwise, and the edge leaving each: (i, j) right, (i+1, j) up,
        # (i+1, j+1) left, (i, j+1) down.
        corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
        edges = ((i, j, false), (i + 1, j, true), (i, j + 1, false), (i, j, true))
        ids = Int32[]
        for k in 1:4
            (ci, cj), (ni, nj) = corners[k], corners[mod1(k + 1, 4)]
            inside(ci, cj) && push!(ids, id!(node(ci, cj), phi[ci, cj] == 0))
            inside(ci, cj) != inside(ni, nj) && push!(ids, id!(xing(edges[k]...), true))
        end
        ids = _dedupe!(ids)
        length(ids) >= 3 || continue
        lo, hi = node(i, j), node(i + 1, j + 1)
        if polygon_area(points, ids) < drop_area * h2
            for k in eachindex(ids)
                push!(dropped, (ids[k], ids[mod1(k + 1, length(ids))]))
            end
            continue
        end
        push!(polygons, ids); push!(parents, lin[i, j])
        push!(areas, (hi[1] - lo[1]) * (hi[2] - lo[2]))
    end

    x0, x1 = node(1, 1), node(n1 + 1, n2 + 1)
    names = Dict{NTuple{2,Int32},Symbol}()
    for ids in polygons, k in eachindex(ids)
        a, b = ids[k], ids[mod1(k + 1, length(ids))]
        pa, pb = points[a], points[b]
        name = if (b, a) in dropped || (onwall[a] && onwall[b])
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
    return PolygonMesh(points, polygons, names; parent=parents, fragment=ones(Int32, length(parents)),
                       parent_area=areas, boundary_names=[:wall, :x_neg, :x_pos, :y_neg, :y_pos])
end


@inline _dot(a, b) = a[1] * b[1] + a[2] * b[2]
