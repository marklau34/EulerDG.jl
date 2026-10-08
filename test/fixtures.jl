# =====================================
# Hand-built meshes
#
# Each fixture builds a `PolygonMesh` through the builder constructor, from coordinates written out
# by hand, so the contract is exercised before any real builder exists. The comments list the pool
# ids; every polygon runs counter-clockwise. Boxes name their sides `:x_neg, :x_pos, :y_neg, :y_pos`.

const PV = SVector{2,Float64}

"""The unit square split along the diagonal from (0, 0) to (1, 1)."""
function two_triangles(::Type{T}=Float64) where {T}
    # 4 3
    # 1 2
    points = SVector{2,T}[(0, 0), (1, 0), (1, 1), (0, 1)]
    sides = Dict((1, 2) => :y_neg, (2, 3) => :x_pos, (3, 4) => :y_pos, (4, 1) => :x_neg)
    return PolygonMesh(points, [[1, 2, 3], [1, 3, 4]], sides)
end

"""
A pentagon on [0,1] x [0,2], with a hanging vertex at (1, 1), beside the unit squares
[1,2] x [0,1] and [1,2] x [1,2].
"""
function hanging_node()
    # 6 7 8
    #   4 5
    # 1 2 3
    points = PV[(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    polygons = [[1, 2, 4, 7, 6], [2, 3, 5, 4], [4, 5, 8, 7]]
    sides = Dict((1, 2) => :y_neg, (2, 3) => :y_neg, (3, 5) => :x_pos, (5, 8) => :x_pos,
                 (8, 7) => :y_pos, (7, 6) => :y_pos, (6, 1) => :x_neg)
    return PolygonMesh(points, polygons, sides)
end

"""Four unit squares on [0,2]^2, glued periodically in x and in y."""
function periodic_2x2(; shift_x=PV(-2, 0), shift_y=PV(0, -2))
    # 7 8 9
    # 4 5 6
    # 1 2 3
    points = PV[(i, j) for j in 0:2 for i in 0:2]
    polygons = [[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]]
    sides = Dict((1, 2) => :y_neg, (2, 3) => :y_neg, (3, 6) => :x_pos, (6, 9) => :x_pos,
                 (9, 8) => :y_pos, (8, 7) => :y_pos, (7, 4) => :x_neg, (4, 1) => :x_neg)
    return PolygonMesh(points, polygons, sides;
                       periodic=((:x_pos, :x_neg) => shift_x, (:y_pos, :y_neg) => shift_y))
end

"""
A 2 x 2 background on [0,2]^2, cut by a straight wall from (0, y0) to (2, y1) with solid below
it, all moved by `offset`. The wall stays in the bottom row (`0 < y0, y1 < 1`), so the two bottom
cells are cut and the two top cells full. The crossing at x = 1 is one pool vertex, used by both
bottom fragments. Parent areas come from the background's node coordinates.
"""
function tilted_wall(; y0=0.25, y1=0.85, offset=(0.0, 0.0))
    # 7 8 9
    # 4 5 6
    # 1 2 3   (1, 2, 3: the wall's crossings at x = 0, 1, 2)
    ox, oy = offset
    gx, gy = ox .+ (0.0, 1.0, 2.0), oy .+ (0.0, 1.0, 2.0)
    ym = y0 + (y1 - y0) / 2
    points = PV[(gx[1], oy + y0), (gx[2], oy + ym), (gx[3], oy + y1),
                (gx[1], gy[2]), (gx[2], gy[2]), (gx[3], gy[2]),
                (gx[1], gy[3]), (gx[2], gy[3]), (gx[3], gy[3])]
    polygons = [[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]]
    sides = Dict((1, 2) => :wall, (2, 3) => :wall, (4, 1) => :x_neg, (7, 4) => :x_neg,
                 (3, 6) => :x_pos, (6, 9) => :x_pos, (9, 8) => :y_pos, (8, 7) => :y_pos)
    cell_area(i, j) = (gx[i + 1] - gx[i]) * (gy[j + 1] - gy[j])
    return PolygonMesh(points, polygons, sides; parent=[1, 2, 3, 4], fragment=[1, 1, 1, 1],
                       parent_area=[cell_area(1, 1), cell_area(2, 1), cell_area(1, 2),
                                    cell_area(2, 2)])
end

"""
One background cell [0,1]^2 crossed by a solid band 0.4 < y < 0.6: two fragments of one parent,
each with one wall face and no face between them.
"""
function split_cell()
    # 8       7
    # 5       6   (the band's upper edge)
    # 4       3   (its lower edge)
    # 1       2
    points = PV[(0, 0), (1, 0), (1, 0.4), (0, 0.4), (0, 0.6), (1, 0.6), (1, 1), (0, 1)]
    sides = Dict((1, 2) => :y_neg, (2, 3) => :x_pos, (3, 4) => :wall, (4, 1) => :x_neg,
                 (5, 6) => :wall, (6, 7) => :x_pos, (7, 8) => :y_pos, (8, 5) => :x_neg)
    return PolygonMesh(points, [[1, 2, 3, 4], [5, 6, 7, 8]], sides; parent=[1, 1],
                       fragment=[1, 2], parent_area=[1.0, 1.0])
end

"""All the valid fixtures, by name."""
fixtures() = (; two_triangles=two_triangles(), hanging_node=hanging_node(),
              periodic_2x2=periodic_2x2(), tilted_wall=tilted_wall(), split_cell=split_cell())

# =====================================
# Helpers

# A plain shoelace sum in absolute coordinates, in the points' own precision.
function naive_area(pts)
    n = length(pts)
    return sum(pts[k][1] * pts[mod1(k + 1, n)][2] - pts[mod1(k + 1, n)][1] * pts[k][2]
               for k in 1:n) / 2
end

# Whether `f()` throws an `ArgumentError` whose message contains `needle`.
function throws_with(f, needle)
    try
        f()
    catch err
        err isa ArgumentError || rethrow()
        occursin(needle, err.msg) && return true
        @warn "the error message does not contain the expected text" needle err.msg
        return false
    end
    return false
end
