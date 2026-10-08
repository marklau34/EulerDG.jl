# =====================================
# The mesh contract
#
# After preprocessing, the solver sees only a `PolygonMesh`: polygonal elements whose vertices run
# counter-clockwise through one shared pool of points, and straight faces between them. Full cells
# and cut fragments are the same kind of element, so one code path serves both, and nothing of the
# background grid survives except each element's `parent` index.
#
# Each polygon edge is exactly one face. Two elements share a face exactly when one runs (a, b) and
# the other (b, a), so a split point shared by two fragments is one pool entry, bitwise the same on
# both sides; a hanging node is one more collinear vertex. Faces are stored interior first, then
# one contiguous range per named boundary.
#
# The default constructors are raw -- they store their fields and check nothing -- so a later Adapt
# conversion can call them with device arrays. Builders call `PolygonMesh(points, polygons,
# boundary_edges; ...)` (build.jl), which derives the faces and runs `assert_mesh`.

@noinline _mesh_error(msg) = throw(ArgumentError(msg))

"""
    MeshElement{T}

The geometry of one element: a simple polygon whose vertices run counter-clockwise.

- `area`, `centroid` -- of the polygon.
- `lo`, `hi` -- its bounding box, on which the element's basis is built.
- `volume_fraction` -- `area / parent_area`; exactly 1 for a full cell.
- `parent_area` -- the area of the background cell the element came from; the element's own area
  when there is no background.
- `parent` -- that cell's linear index in the background grid; 0 when there is none.
- `fragment` -- which of the parent's fluid fragments this is: 1, 2, ...

`area`, `centroid`, the bounds and `volume_fraction` are derived from the vertex pool by the
polygon functions, and [`check_mesh`](@ref) verifies them bitwise; the builder supplies the rest.
"""
struct MeshElement{T}
    area::T
    centroid::SVector{2,T}
    lo::SVector{2,T}
    hi::SVector{2,T}
    volume_fraction::T
    parent_area::T
    parent::Int32
    fragment::Int32
end

"""
    MeshFace{T}

One straight face between two pool vertices.

- `nodes` -- `(a, b)`: the left element runs `a -> b` counter-clockwise.
- `left`, `right` -- the elements on either side; `right == 0` on a boundary face.
- `boundary` -- 0 on an interior face; otherwise an index into the mesh's `boundary_names`.
- `vector` -- [`face_vector`](@ref)`(points[a], points[b])`: the face's length times its unit
  normal, pointing out of `left`. Fluxes take it as their unnormalized normal direction, and the
  closure property is stated in terms of it.
- `length`, `normal` -- its length, and `vector / length`.
- `shift` -- zero unless the face is periodic. A point `x` on the left element's copy of the face
  is the point `x + shift` on the right element's copy.
"""
struct MeshFace{T}
    nodes::NTuple{2,Int32}
    left::Int32
    right::Int32
    boundary::Int32
    vector::SVector{2,T}
    length::T
    normal::SVector{2,T}
    shift::SVector{2,T}
end

"""
    PolygonMesh{T,P,I,E,F}

The mesh contract between the preprocessor and the solver.

- `points` -- the one vertex pool; every point is used by some element.
- `elem_ptr`, `elem_nodes` -- CSR storage of the polygons: element `e`'s pool ids, counter-clockwise
  with the first not repeated, are `elem_nodes[elem_ptr[e]:elem_ptr[e+1]-1]`.
- `elem_faces` -- in the same layout, the face on the edge from each vertex to the next.
- `elements` -- a `StructVector{MeshElement{T}}`.
- `faces` -- a `StructVector{MeshFace{T}}`: interior faces first, then each boundary in turn.
- `boundary_ptr` -- boundary `k` is faces `boundary_ptr[k]:boundary_ptr[k+1]-1`, and the interior
  faces are `1:boundary_ptr[1]-1`.
- `boundary_names` -- one `Symbol` per boundary, such as `:wall`, `:farfield` or `:x_neg`.

Build a mesh with `PolygonMesh(points, polygons, boundary_edges; ...)`. The positional
constructor over all eight fields is raw: it checks nothing.
"""
struct PolygonMesh{T,P<:AbstractVector{SVector{2,T}},I<:AbstractVector{Int32},E,F}
    points::P
    elem_ptr::I
    elem_nodes::I
    elem_faces::I
    elements::E
    faces::F
    boundary_ptr::I
    boundary_names::Vector{Symbol}
end

# =====================================
# Accessors

"""
    nelements(m) -> Int

The number of elements of `m`.
"""
nelements(m::PolygonMesh) = length(m.elem_ptr) - 1

"""
    nfaces(m) -> Int

The number of faces of `m`, interior and boundary.
"""
nfaces(m::PolygonMesh) = length(m.faces)

"""
    npoints(m) -> Int

The number of points in the vertex pool of `m`.
"""
npoints(m::PolygonMesh) = length(m.points)

"""
    nboundaries(m) -> Int

The number of named boundaries of `m`.
"""
nboundaries(m::PolygonMesh) = length(m.boundary_names)

"""
    element_nodes(m, e) -> view

The pool ids of element `e`, counter-clockwise.
"""
element_nodes(m::PolygonMesh, e::Integer) =
    view(m.elem_nodes, Int(m.elem_ptr[e]):(Int(m.elem_ptr[e + 1]) - 1))

"""
    element_faces(m, e) -> view

The faces of element `e`: entry `k` is the face on the edge from its vertex `k` to the next.
"""
element_faces(m::PolygonMesh, e::Integer) =
    view(m.elem_faces, Int(m.elem_ptr[e]):(Int(m.elem_ptr[e + 1]) - 1))

"""
    interior_faces(m) -> UnitRange{Int}

The ids of the interior faces of `m`, periodic ones included.
"""
interior_faces(m::PolygonMesh) = 1:(Int(m.boundary_ptr[1]) - 1)

"""
    boundary_faces(m, k::Integer) -> UnitRange{Int}
    boundary_faces(m, name::Symbol) -> UnitRange{Int}

The ids of the faces on boundary `k`, or on the boundary called `name`.
"""
boundary_faces(m::PolygonMesh, k::Integer) =
    Int(m.boundary_ptr[k]):(Int(m.boundary_ptr[k + 1]) - 1)
boundary_faces(m::PolygonMesh, name::Symbol) = boundary_faces(m, boundary_index(m, name))

"""
    boundary_index(m, name) -> Int

The index of the boundary called `name` in `m.boundary_names`. Throws if there is none.
"""
function boundary_index(m::PolygonMesh, name::Symbol)
    k = findfirst(==(name), m.boundary_names)
    k === nothing && _mesh_error("the mesh has no boundary :$name; its boundaries are " *
                                 "$(m.boundary_names)")
    return k
end

"""
    face_sign(m, e, f) -> Int

`+1` when element `e` is the left element of face `f`, `-1` when it is the right one, and `0` when
it is neither. `face_sign(m, e, f) * m.faces.vector[f]` points out of `e`.
"""
@inline function face_sign(m::PolygonMesh, e::Integer, f::Integer)
    m.faces.left[f] == e && return 1
    m.faces.right[f] == e && return -1
    return 0
end

function Base.show(io::IO, m::PolygonMesh{T}) where {T}
    print(io, "PolygonMesh{", T, "}(", nelements(m), " elements, ", nfaces(m), " faces)")
end

function Base.show(io::IO, ::MIME"text/plain", m::PolygonMesh{T}) where {T}
    print(io, "PolygonMesh{", T, "} with ", nelements(m), " elements, ", nfaces(m), " faces and ",
          npoints(m), " points")
    print(io, "\n  interior: ", _faces_str(length(interior_faces(m))))
    for (k, name) in enumerate(m.boundary_names)
        print(io, "\n  ", name, ": ", _faces_str(length(boundary_faces(m, k))))
    end
end

_faces_str(n) = n == 1 ? "1 face" : "$n faces"

# =====================================
# Derived geometry
#
# The builder stores what these return and `check_mesh` compares against them, so both use exactly
# the same arithmetic.

# Tolerances, as functions of the float type: 450 eps (1e-13 in Float64, the cutter's own closure
# tolerance) for an element's closure relative to its perimeter; sqrt(eps)/100 (1.5e-10) of a
# face's length for matching periodic copies; and 16 eps above 1 for a volume fraction.
_closure_tol(::Type{T}) where {T} = 450 * eps(T)
_periodic_tol(::Type{T}) where {T} = sqrt(eps(T)) / 100
_vf_max(::Type{T}) where {T} = 1 + 16 * eps(T)

@inline function _element_geometry(points::AbstractVector{SVector{2,T}}, ids, parent_area::T,
                                   parent::Int32, fragment::Int32) where {T}
    area = polygon_area(points, ids)
    lo, hi = polygon_bounds(points, ids)
    return MeshElement{T}(area, polygon_centroid(points, ids), lo, hi, area / parent_area,
                          parent_area, parent, fragment)
end

@inline function _face_geometry(points::AbstractVector{SVector{2,T}}, a, b) where {T}
    S = face_vector(points[a], points[b])
    L = hypot(S[1], S[2])
    return S, L, S / L
end

@inline function _mesh_face(points::AbstractVector{SVector{2,T}}, a, b, left, right, boundary,
                            shift::SVector{2,T}) where {T}
    S, L, n = _face_geometry(points, a, b)
    return MeshFace{T}((Int32(a), Int32(b)), Int32(left), Int32(right), Int32(boundary), S, L, n,
                       shift)
end

@inline _near(x, y, h) = abs(x[1] - y[1]) <= h && abs(x[2] - y[2]) <= h

# The closure error of element `e`: the length of the sum of its outward face vectors, relative to
# its perimeter.
function _element_closure(m::PolygonMesh{T}, e::Integer) where {T}
    s = zero(SVector{2,T})
    perimeter = zero(T)
    for f in element_faces(m, e)
        s += face_sign(m, e, f) * m.faces.vector[f]
        perimeter += m.faces.length[f]
    end
    return hypot(s[1], s[2]) / perimeter
end
