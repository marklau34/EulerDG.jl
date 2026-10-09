# =====================================
# Geometry: the bodies
#
# Upstream of the mesh contract. CutCellMethods' exact polyline clipping takes only an `SDFMesh` --
# a closed, counter-clockwise polyline -- so every body is a polygon. Its segments must shrink with
# the cell size for grid convergence to hold: the cutter's brief suggests at most h/4, and the point
# generators here take the cell size and a number of segments per cell to that end.

"""
    EulerDG.Geometry

The bodies the preprocessor cuts, as closed counter-clockwise polygons.

- [`Body`](@ref) and [`polygon_body`](@ref): a polygon as the SDFLibrary `SDFMesh` the cutter takes.
- [`circle_points`](@ref) and [`naca4_points`](@ref), the latter with a sharp trailing edge, both
  with segments that shrink with the cell size.
- [`grid_shifts`](@ref), sub-cell offsets that cover the cell evenly, and [`sliver_shift`](@ref),
  an offset that puts a grid node just outside the body to make a tiny fluid fragment -- the inputs
  of the Phase 4 grid-shift sweep.

Upstream of the mesh contract: only `CutMesh` and the top-level driver may use it.
"""
module Geometry

using MeshLibrary: Line, Mesh, MeshElementSet, Point
using SDFLibrary: SDFMesh, get_sdf
using StaticArrays: SVector

export Body, polygon_body, circle_points, naca4_points, polygon_perimeter, grid_shifts
export sliver_shift

const PV = SVector{2,Float64}

"""
    Body

A body the cutter can take: its closed counter-clockwise polygon `points` (the first not
repeated), the `SDFMesh` built from them, and a `name`.
"""
struct Body{S}
    points::Vector{PV}
    sdf::S
    name::String
end

Base.show(io::IO, b::Body) = print(io, "Body(\"", b.name, "\", ", length(b.points), " points)")

"""
    polygon_body(points; name="body") -> Body

A body bounded by the closed polygon through `points`, which must run counter-clockwise (solid on
the left of each edge) without repeating the first point.
"""
function polygon_body(points::AbstractVector; name::AbstractString="body")
    pts = PV[PV(p) for p in points]
    n = length(pts)
    n >= 3 || throw(ArgumentError("a body needs at least 3 points, not $n"))
    for k in 1:n
        pts[k] == pts[mod1(k + 1, n)] &&
            throw(ArgumentError("points $k and $(mod1(k + 1, n)) of the body coincide"))
    end
    area = sum(pts[k][1] * pts[mod1(k + 1, n)][2] - pts[mod1(k + 1, n)][1] * pts[k][2]
               for k in 1:n) / 2
    area > 0 || throw(ArgumentError("the body's points must run counter-clockwise; their " *
                                    "signed area is $area"))
    mesh = Mesh([Point(p) for p in pts], [Line(Int32(k), Int32(mod1(k + 1, n))) for k in 1:n])
    push!(mesh.elemset, MeshElementSet(String(name), collect(1:n)))
    return Body(pts, SDFMesh(mesh), String(name))
end

"""
    polygon_perimeter(points) -> Float64

The perimeter of the closed polygon through `points`.
"""
polygon_perimeter(points) =
    sum(hypot((points[mod1(k + 1, length(points))] - points[k])...) for k in eachindex(points))

# The number of segments for a curve of length `L`: `n` if given, else enough for `per_cell`
# segments per cell of size `h`.
function _segments(L, n, h, per_cell)
    n === nothing || return n
    h === nothing && throw(ArgumentError("give either the number of segments n or the cell size h"))
    return ceil(Int, per_cell * L / h)
end

"""
    circle_points(; center=(0, 0), radius, n=nothing, h=nothing, per_cell=4) -> Vector{SVector{2}}

The regular `n`-gon inscribed in the circle, counter-clockwise from angle 0. Give either `n`, or
the cell size `h`, for `per_cell` segments per cell along the circumference.

The lower half mirrors the upper half bitwise, so a circle centred on a grid line is exactly
symmetric across it, and so is the cut mesh, to round-off: a symmetric flow then has zero lift to
round-off.
"""
function circle_points(; center=(0.0, 0.0), radius, n=nothing, h=nothing, per_cell=4)
    m = _segments(2π * radius, n, h, per_cell)
    c = PV(center)
    q = [PV(cospi(2k / m), sinpi(2k / m)) for k in 0:(m - 1)]
    for k in (m ÷ 2 + 1):(m - 1)
        q[k + 1] = PV(q[m - k + 1][1], -q[m - k + 1][2])
    end
    return [c + radius * p for p in q]
end

"""
    naca4_points(digits="0012"; n=nothing, h=nothing, per_cell=4, chord=1, le=(0, 0),
                 alpha_deg=0) -> Vector{SVector{2}}

A NACA 4-digit section with a sharp trailing edge (the thickness coefficient -0.1036 closes it),
as a counter-clockwise polygon from the trailing edge along the upper surface, round the leading
edge and back along the lower surface.

Each surface has `n` segments with half-cosine spacing, `x = 1 - cos(π j / 2n)`: clustered at the
leading edge, where the curvature is, and widest at the trailing edge, where the section is
nearly straight. Clustering there too would pile tiny wall faces into the trailing-edge cell. Give
`n`, or the cell size `h` for segments no longer than `h / per_cell` anywhere. The leading edge is
at `le`, and the section is turned nose-up by `alpha_deg` about it.
"""
function naca4_points(digits::AbstractString="0012"; n=nothing, h=nothing, per_cell=4,
                      chord=1.0, le=(0.0, 0.0), alpha_deg=0.0)
    (length(digits) == 4 && all(isdigit, digits)) ||
        throw(ArgumentError("a NACA 4-digit section needs 4 digits, not \"$digits\""))
    m = parse(Int, digits[1]) / 100
    p = parse(Int, digits[2]) / 10
    t = parse(Int, digits[3:4]) / 100
    (m == 0) == (p == 0) || throw(ArgumentError("NACA $digits has camber without its position"))
    yt(x) = 5t * (0.2969sqrt(x) - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4)
    yc(x) = m == 0 ? 0.0 : x < p ? m / p^2 * (2p * x - x^2) : m / (1 - p)^2 * ((1 - 2p) + 2p * x - x^2)
    dyc(x) = m == 0 ? 0.0 : x < p ? 2m / p^2 * (p - x) : 2m / (1 - p)^2 * (p - x)
    surface(x, s) = (θ = atan(dyc(x)); PV(x - s * yt(x) * sin(θ), yc(x) + s * yt(x) * cos(θ)))
    α = deg2rad(alpha_deg)
    R(q) = PV(cos(α) * q[1] + sin(α) * q[2], -sin(α) * q[1] + cos(α) * q[2])
    function section(k)
        xs = [1 - cospi(j / 2k) for j in 0:k]
        upper = [surface(x, 1) for x in reverse(xs)]
        lower = [surface(x, -1) for x in xs[2:(end - 1)]]
        # The trailing edge is one point: yt(1) is zero to round-off.
        upper[1] = PV(1, yc(1.0))
        return [PV(le) + chord * R(q) for q in vcat(upper, lower)]
    end
    # The widest spacing in x is π/(2k) chords, at the trailing edge. The surface slopes there, so
    # the segments are a little longer: grow k until they fit.
    k = _segments(π * chord / 2, n, h, per_cell)
    pts = section(k)
    while n === nothing && (w = _widest(pts)) > h / per_cell
        k = ceil(Int, k * w * per_cell / h)
        pts = section(k)
    end
    return pts
end

_widest(pts) = maximum(hypot((pts[mod1(j + 1, length(pts))] - pts[j])...) for j in eachindex(pts))

"""
    grid_shifts(n) -> Vector{SVector{2,Float64}}

`n` sub-cell offsets, in cells, spread evenly over [0, 1)^2: (0, 0) first, then the R2
low-discrepancy sequence (Roberts, 2018). Multiply by the grid spacing and add to a body's points
to move it relative to the grid.
"""
function grid_shifts(n::Integer)
    φ = 1.32471795724474602596         # the plastic number, x^3 = x + 1
    α = PV(1 / φ, 1 / φ^2)
    return [k == 0 ? PV(0, 0) : mod.(PV(0.5, 0.5) + k * α, 1.0) for k in 0:(n - 1)]
end

"""
    sliver_shift(body, nodes; gap) -> SVector{2,Float64}

An offset that moves `body` so that one of the grid `nodes` lies a distance `gap` outside it, to
make a tiny fluid fragment in the cell beside that node. The node is the one outside the body that
is nearest to it, among those whose outward normal is at least 20 degrees off both axes, so the
fragment is a corner of a cell the body otherwise covers. `gap` is a length, typically a small
multiple of the cell size.
"""
function sliver_shift(body::Body, nodes; gap::Real)
    best, d0, n0 = nothing, Inf, PV(0, 0)
    for x in nodes
        d, n = get_sdf(body.sdf, PV(x))
        (0 < d < d0 && all(isfinite, n) && min(abs(n[1]), abs(n[2])) > sind(20)) || continue
        best, d0, n0 = x, d, PV(n)
    end
    best === nothing && throw(ArgumentError("no grid node lies outside the body at a usable angle"))
    return (d0 - gap) * n0
end

end # module Geometry
