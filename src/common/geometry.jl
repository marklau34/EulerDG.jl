# =====================================
# Polygon geometry: the one kernel every derived quantity comes from
#
# Every area, centroid, bounding box and face vector of a mesh is computed here, from the shared
# vertex pool -- by the builder, and again by `check_mesh`, so the two agree bitwise. Two details
# keep cut-cell slivers accurate:
#
# - Areas and centroids are taken about the polygon's first vertex, not the origin. A cut cell 1000
#   cells from the origin makes each absolute-coordinate shoelace term about 1e6 cell areas, and
#   their cancellation error is larger than a 1e-11 sliver; about the first vertex it is near
#   machine epsilon of a cell area.
# - A face vector is a difference of endpoint coordinates, never length times normal. Around a
#   closed polygon the face vectors telescope, so they sum to zero at round-off -- exactly zero
#   when the coordinates lie within a factor of two of each other (Sterbenz).

@inline _cross(a, b) = a[1] * b[2] - a[2] * b[1]
@inline _dot(a, b) = a[1] * b[1] + a[2] * b[2]

"""
    polygon_area(points, ids) -> T

The signed area of the polygon through `points[ids]`: positive when its vertices run
counter-clockwise. It is computed about the first vertex, so slivers far from the origin keep
their precision. `ids` may be any integer vector, such as a view into a mesh's `elem_nodes`, and
must hold at least two ids.
"""
@inline function polygon_area(points::AbstractVector{SVector{2,T}},
                              ids::AbstractVector{<:Integer}) where {T}
    p1 = points[ids[1]]
    q = points[ids[2]] - p1
    a2 = zero(T)
    for k in 3:length(ids)
        r = points[ids[k]] - p1
        a2 += _cross(q, r)
        q = r
    end
    return a2 / 2
end

"""
    polygon_centroid(points, ids) -> SVector{2,T}

The centroid of the polygon through `points[ids]`, computed about its first vertex and then
shifted back. NaN for a polygon of zero area.
"""
@inline function polygon_centroid(points::AbstractVector{SVector{2,T}},
                                  ids::AbstractVector{<:Integer}) where {T}
    p1 = points[ids[1]]
    q = points[ids[2]] - p1
    a2 = zero(T)
    m = zero(SVector{2,T})
    for k in 3:length(ids)
        r = points[ids[k]] - p1
        c = _cross(q, r)
        a2 += c
        m += c * (q + r)
        q = r
    end
    return p1 + m / (3 * a2)
end

"""
    polygon_bounds(points, ids) -> (lo, hi)

The bounding box of `points[ids]`, as its lower and upper corners.
"""
@inline function polygon_bounds(points::AbstractVector{SVector{2,T}},
                                ids::AbstractVector{<:Integer}) where {T}
    lo = hi = points[ids[1]]
    for k in 2:length(ids)
        p = points[ids[k]]
        lo = min.(lo, p)
        hi = max.(hi, p)
    end
    return lo, hi
end

"""
    face_vector(pa, pb) -> SVector{2,T}

The face vector of the edge from `pa` to `pb`: `(pb[2] - pa[2], pa[1] - pb[1])`, the edge's length
times its unit normal, pointing out of a polygon that runs `pa -> pb` counter-clockwise.
"""
@inline face_vector(pa::SVector{2,T}, pb::SVector{2,T}) where {T} =
    SVector(pb[2] - pa[2], pa[1] - pb[1])

# =====================================
# Polygon validity

@inline _orient(p, q, r) = sign(_cross(q - p, r - p))

# Whether `q` lies in the bounding box of `p` and `r`; with `q` collinear, whether it is on the
# segment.
@inline _between(p, q, r) =
    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]) && min(p[2], r[2]) <= q[2] <= max(p[2], r[2])

# Whether the closed segments p1-q1 and p2-q2 share a point.
@inline function _segments_meet(p1, q1, p2, q2)
    o1, o2 = _orient(p1, q1, p2), _orient(p1, q1, q2)
    o3, o4 = _orient(p2, q2, p1), _orient(p2, q2, q1)
    o1 * o2 < 0 && o3 * o4 < 0 && return true
    return (o1 == 0 && _between(p1, p2, q1)) || (o2 == 0 && _between(p1, q2, q1)) ||
           (o3 == 0 && _between(p2, p1, q2)) || (o4 == 0 && _between(p2, q1, q2))
end

# Why `points[ids]` is not a valid element polygon, or `nothing`. A valid polygon has at least
# three vertices, no repeated id, positive (counter-clockwise) area, no zero-length edge, no edge
# folding back onto the next, and no two non-adjacent edges meeting. Collinear vertices are legal:
# a hanging node is one. The orientation tests run in floating point; a sliver thin enough to fool
# them lies far below any volume fraction a cut keeps.
function _polygon_problem(points::AbstractVector{SVector{2,T}},
                          ids::AbstractVector{<:Integer}) where {T}
    n = length(ids)
    n >= 3 || return "it has $n vertices; a polygon needs at least 3"
    for i in 1:n, j in (i + 1):n
        ids[i] == ids[j] && return "pool id $(ids[i]) appears twice (vertices $i and $j)"
    end
    area = polygon_area(points, ids)
    area > 0 || return "its signed area is $area; its vertices must run counter-clockwise"
    for k in 1:n
        a, b, c = points[ids[k]], points[ids[mod1(k + 1, n)]], points[ids[mod1(k + 2, n)]]
        a == b && return "edge $k ($(ids[k]) -> $(ids[mod1(k + 1, n)])) has zero length"
        u, v = b - a, c - b
        if _cross(u, v) == 0 && _dot(u, v) < 0
            return "edges $k and $(mod1(k + 1, n)) fold back on each other at pool id " *
                   "$(ids[mod1(k + 1, n)])"
        end
    end
    for i in 1:n, j in (i + 2):n
        i == 1 && j == n && continue
        if _segments_meet(points[ids[i]], points[ids[mod1(i + 1, n)]],
                          points[ids[j]], points[ids[mod1(j + 1, n)]])
            return "edges $i and $j cross or touch"
        end
    end
    return nothing
end
