# =====================================
# Rules on element polygons
#
# An axis-aligned rectangle -- every full cell -- gets the tensor Gauss rule on its box. Any other
# polygon is cut into triangles by ear clipping, which also handles nonconvex polygons (a
# trailing-edge wedge) and collinear vertices (a hanging node), and each triangle gets a symmetric
# rule. Points are placed relative to a vertex, so their rounding stays at the scale of the
# element even far from the origin.

@inline _cross(a, b) = a[1] * b[2] - a[2] * b[1]

# Whether `p` lies in the closed counter-clockwise triangle (a, b, c).
@inline _in_triangle(p, a, b, c) =
    _cross(b - a, p - a) >= 0 && _cross(c - b, p - b) >= 0 && _cross(a - c, p - c) >= 0

"""
    triangulate(points, ids) -> Vector{NTuple{3,Int32}}

Cut the simple, counter-clockwise polygon through `points[ids]` into counter-clockwise triangles of
pool ids, by ear clipping. An ear is a strictly convex vertex whose triangle holds no other vertex,
not even on its edges, so collinear vertices never produce a degenerate triangle.
"""
function triangulate(points::AbstractVector{SVector{2,T}}, ids::AbstractVector{<:Integer}) where {T}
    rest = Int32[id for id in ids]
    tris = NTuple{3,Int32}[]
    while length(rest) > 3
        m = length(rest)
        ear = 0
        for k in 1:m
            a, b, c = rest[mod1(k - 1, m)], rest[k], rest[mod1(k + 1, m)]
            pa, pb, pc = points[a], points[b], points[c]
            _cross(pb - pa, pc - pb) > 0 || continue
            if !any(v -> v != a && v != b && v != c && _in_triangle(points[v], pa, pb, pc), rest)
                ear = k
                break
            end
        end
        ear == 0 && throw(ArgumentError("the polygon through pool ids $(collect(ids)) has no ear " *
                                        "left at $(rest); it is not simple and counter-clockwise"))
        push!(tris, (rest[mod1(ear - 1, m)], rest[ear], rest[mod1(ear + 1, m)]))
        deleteat!(rest, ear)
    end
    push!(tris, (rest[1], rest[2], rest[3]))
    return tris
end

# Whether the polygon is an axis-aligned rectangle: four edges, alternately horizontal and vertical.
function _is_box(points, ids)
    length(ids) == 4 || return false
    flat = ntuple(4) do k
        e = points[ids[mod1(k + 1, 4)]] - points[ids[k]]
        e[2] == 0 && e[1] != 0 ? 1 : e[1] == 0 && e[2] != 0 ? 2 : 0
    end
    return flat == (1, 2, 1, 2) || flat == (2, 1, 2, 1)
end

"""
    polygon_rule(points, ids; degree=2, origin=0) -> (x::Vector{SVector{2,T}}, w::Vector{T})

A quadrature rule on the polygon through `points[ids]`, exact for polynomials up to `degree`:

- an axis-aligned rectangle gets the tensor Gauss rule on its box: 2 x 2 points to degree 3, 3 x 3
  to degree 5;
- any other polygon is triangulated with [`triangulate`](@ref), and each triangle gets
  [`TRIANGLE3`](@ref) (3 points) to degree 2, or [`TRIANGLE6`](@ref) (6 points) to degree 4.

The points are returned relative to `origin`. With the element's box centre as the origin they
keep their full precision on a sliver far from the origin; in absolute coordinates they would be
rounded to the absolute coordinate's last bit, a large fraction of a thin sliver. The weights sum
to the polygon's area, to round-off.
"""
function polygon_rule(points::AbstractVector{SVector{2,T}}, ids::AbstractVector{<:Integer};
                      degree::Integer=2, origin::SVector{2,T}=zero(SVector{2,T})) where {T}
    x, w = SVector{2,T}[], T[]
    if _is_box(points, ids)
        degree <= 5 || throw(ArgumentError("rectangle rules reach degree 5, not $degree"))
        g = degree <= 3 ? GAUSS2 : GAUSS3
        lo = hi = points[ids[1]]
        for id in ids
            lo, hi = min.(lo, points[id]), max.(hi, points[id])
        end
        d = hi - lo
        area = d[1] * d[2]
        o = lo - origin
        for (sj, wj) in zip(g.nodes, g.weights), (si, wi) in zip(g.nodes, g.weights)
            push!(x, o + SVector(T(si) * d[1], T(sj) * d[2]))
            push!(w, T(wi * wj) * area)
        end
    else
        degree <= 4 || throw(ArgumentError("triangle rules reach degree 4, not $degree"))
        rule = degree <= 2 ? TRIANGLE3 : TRIANGLE6
        for (a, b, c) in triangulate(points, ids)
            pa = points[a]
            u, v = points[b] - pa, points[c] - pa
            area = _cross(u, v) / 2
            o = pa - origin
            for (λ, wq) in zip(rule.points, rule.weights)
                push!(x, o + (T(λ[2]) * u + T(λ[3]) * v))
                push!(w, T(wq) * area)
            end
        end
    end
    return x, w
end
