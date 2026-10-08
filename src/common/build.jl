# =====================================
# Building a mesh from polygons
#
# Every builder -- the hand-made test fixtures, the uncut grid (Phase 1), the cut adapter (Phase 2)
# -- hands over three things: a vertex pool, one counter-clockwise polygon per element, and a name
# for every edge on the boundary. The constructor derives everything else. It pairs each directed
# edge (a, b) with its reverse (b, a) to make the interior faces, glues periodic sides, computes the
# geometry with the polygon functions, orders the faces and checks the result. No builder writes a
# face by hand, so no builder can produce faces that disagree with its polygons.

"""
    PolygonMesh(points, polygons, boundary_edges; periodic=(), boundary_names=nothing,
                parent=nothing, fragment=nothing, parent_area=nothing) -> PolygonMesh{T}

Build a mesh, and check it with [`assert_mesh`](@ref).

- `points::AbstractVector{SVector{2,T}}` -- the vertex pool. Points no polygon uses are dropped,
  and the rest are renumbered in pool order.
- `polygons` -- one vector of pool ids per element, counter-clockwise, with the first not
  repeated. Collinear vertices are legal.
- `boundary_edges` -- a dictionary from each boundary edge, as the directed pair `(a, b)` its
  element runs, to the name of its boundary.
- `periodic` -- pairs `(:p, :q) => shift`. Each face on side `:p` is glued to the face on side `:q`
  whose endpoints are its own, reversed, plus `shift` (to a tolerance of 1.5e-10 of its length in
  `Float64`). The element on `:p` becomes the left one, and glued names leave `boundary_names`.
- `boundary_names` -- the order of the boundaries. By default it is the order in which the names
  first appear, walking each element's edges in element order. An explicit list must name every
  boundary that has faces, and may name boundaries that have none.
- `parent`, `fragment` -- per element: the background cell's linear index (default 0, none) and
  which of its fluid fragments the element is (default 1).
- `parent_area` -- per element: the background cell's area. Compute it from the same node
  coordinates as the polygons, as `(x1 - x0) * (y1 - y0)`, and a full cell's volume fraction is
  exactly 1. By default it is the element's own area, so every volume fraction is 1.

An interior face takes the lower-numbered of its two elements as its left element. Every failure
throws an `ArgumentError` that names what is wrong: the element, its edge and the pool ids
involved, or the boundary side.
"""
function PolygonMesh(points::AbstractVector{SVector{2,T}},
                     polygons::AbstractVector{<:AbstractVector{<:Integer}},
                     boundary_edges::AbstractDict; periodic=(), boundary_names=nothing,
                     parent=nothing, fragment=nothing, parent_area=nothing) where {T<:AbstractFloat}
    ne, np = length(polygons), length(points)
    ne > 0 || _mesh_error("a mesh needs at least one element")
    parents = _per_element(parent, ne, 0, Int32, "parent")
    fragments = _per_element(fragment, ne, 1, Int32, "fragment")
    areas = parent_area === nothing ? nothing : _per_element(parent_area, ne, 0, T, "parent_area")

    # 1. The polygons in CSR form, each checked on its own.
    elem_ptr = Vector{Int32}(undef, ne + 1)
    elem_ptr[1] = 1
    for e in 1:ne
        elem_ptr[e + 1] = elem_ptr[e] + length(polygons[e])
    end
    elem_nodes = Vector{Int32}(undef, elem_ptr[end] - 1)
    for e in 1:ne, (k, id) in enumerate(polygons[e])
        1 <= id <= np || _mesh_error("element $e: vertex $k is pool id $id, outside the pool 1:$np")
        elem_nodes[elem_ptr[e] + k - 1] = id
    end
    for e in 1:ne
        problem = _polygon_problem(points, _csr_view(elem_ptr, elem_nodes, e))
        problem === nothing || _mesh_error("element $e: $problem")
    end

    # 2. Pair each directed edge (a, b) with its reverse (b, a). Edge `j` is the one from vertex `j`
    # of `elem_nodes` to the next vertex of the same element.
    nedge = length(elem_nodes)
    edge_elem = Vector{Int32}(undef, nedge)
    owner = Dict{NTuple{2,Int32},Int}()
    sizehint!(owner, nedge)
    for e in 1:ne, j in elem_ptr[e]:(elem_ptr[e + 1] - 1)
        edge_elem[j] = e
        ab = _edge(elem_ptr, elem_nodes, e, j)
        j0 = get(owner, ab, 0)
        j0 == 0 || _mesh_error("elements $(edge_elem[j0]) and $e both run edge $(ab[1]) -> " *
                               "$(ab[2]); an edge borders at most two elements, run in opposite " *
                               "directions")
        owner[ab] = j
    end
    named = Dict{NTuple{2,Int32},Symbol}()
    for (key, name) in boundary_edges
        ab = (Int32(key[1]), Int32(key[2]))
        haskey(owner, ab) || _mesh_error("boundary edge $(key[1]) -> $(key[2]) (:$name) is not an " *
                                         "edge of any element, in that direction")
        named[ab] = Symbol(name)
    end

    # Faces in creation order. `fedge` is each face's edge on its left element, and `face_of[j]` the
    # face on edge `j`.
    face_of = zeros(Int32, nedge)
    fedge = Int[]
    fright = Int32[]
    fname = Union{Nothing,Symbol}[]
    fshift = SVector{2,T}[]
    seen = Symbol[]
    for e in 1:ne, j in elem_ptr[e]:(elem_ptr[e + 1] - 1)
        face_of[j] == 0 || continue
        a, b = _edge(elem_ptr, elem_nodes, e, j)
        k = j - elem_ptr[e] + 1
        twin = get(owner, (b, a), 0)
        name = get(named, (a, b), nothing)
        if twin != 0
            for (x, y) in ((a, b), (b, a))
                other = get(named, (x, y), nothing)
                other === nothing ||
                    _mesh_error("element $e, edge $k ($a -> $b) borders element " *
                                "$(edge_elem[twin]), but edge $x -> $y is also named :$other")
            end
            push!(fedge, j); push!(fright, edge_elem[twin]); push!(fname, nothing)
            push!(fshift, zero(SVector{2,T}))
            face_of[j] = face_of[twin] = length(fedge)
        elseif name !== nothing
            push!(fedge, j); push!(fright, 0); push!(fname, name); push!(fshift, zero(SVector{2,T}))
            face_of[j] = length(fedge)
            name in seen || push!(seen, name)
        else
            _mesh_error("element $e, edge $k ($a -> $b) has no neighbour and no boundary name")
        end
    end

    # 3. Glue periodic sides: each face on `:p` takes the matching face on `:q` as its right side.
    dead = falses(length(fedge))
    glued = Symbol[]
    tol = _periodic_tol(T)
    for pair in periodic
        (p, q), shift = pair.first, SVector{2,T}(pair.second)
        p == q && _mesh_error("periodic pair (:$p, :$q) glues a side to itself")
        for side in (p, q)
            side in glued && _mesh_error("periodic side :$side is glued twice")
        end
        P, Q = findall(==(p), fname), findall(==(q), fname)
        isempty(P) && _mesh_error("periodic side :$p has no faces")
        isempty(Q) && _mesh_error("periodic side :$q has no faces")
        length(P) == length(Q) || _mesh_error("periodic sides :$p and :$q have $(length(P)) and " *
                                              "$(length(Q)) faces")
        taken = falses(length(Q))
        for fp in P
            j = fedge[fp]
            e = edge_elem[j]
            a, b = _edge(elem_ptr, elem_nodes, e, j)
            xa, xb = points[a] + shift, points[b] + shift
            h = tol * hypot(face_vector(points[a], points[b])...)
            iq = findfirst(eachindex(Q)) do i
                jq = fedge[Q[i]]
                c, d = _edge(elem_ptr, elem_nodes, edge_elem[jq], jq)
                !taken[i] && _near(points[c], xb, h) && _near(points[d], xa, h)
            end
            iq === nothing &&
                _mesh_error("element $e, edge $(j - elem_ptr[e] + 1) ($a -> $b) on periodic side " *
                            ":$p has no partner on :$q under the shift $(Tuple(shift))")
            taken[iq] = true
            jq = fedge[Q[iq]]
            edge_elem[jq] == e &&
                _mesh_error("element $e, edge $(j - elem_ptr[e] + 1) ($a -> $b) is glued to its " *
                            "own element; a periodic direction needs at least two elements")
            fright[fp], fname[fp], fshift[fp] = edge_elem[jq], nothing, shift
            face_of[jq] = fp
            dead[Q[iq]] = true
        end
        push!(glued, p, q)
    end
    filter!(n -> !(n in glued), seen)

    # 4. The order of the boundaries.
    names = if boundary_names === nothing
        seen
    else
        given = Symbol[Symbol(n) for n in boundary_names]
        allunique(given) || _mesh_error("boundary_names repeats a name: $given")
        for n in given
            n in glued && _mesh_error("boundary :$n is glued by `periodic`, so it cannot be " *
                                      "listed in boundary_names")
        end
        for n in seen
            n in given || _mesh_error("boundary :$n has faces but is missing from boundary_names " *
                                      "$given")
        end
        given
    end

    # 5. Order the faces: interior first, then each boundary in turn.
    order = [f for f in eachindex(fedge) if !dead[f] && fname[f] === nothing]
    boundary_ptr = Int32[]
    for n in names
        push!(boundary_ptr, length(order) + 1)
        append!(order, f for f in eachindex(fedge) if !dead[f] && fname[f] === n)
    end
    push!(boundary_ptr, length(order) + 1)
    newface = zeros(Int32, length(fedge))
    newface[order] = 1:length(order)
    elem_faces = Int32[newface[face_of[j]] for j in 1:nedge]

    # 6. Drop the points no polygon uses, and renumber the rest in pool order.
    used = falses(np)
    used[elem_nodes] .= true
    newpoint = cumsum(used)
    pool = collect(points[used])
    nodes = Int32[newpoint[id] for id in elem_nodes]

    # 7. The geometry, from the polygon functions only.
    elements = StructVector([begin
                                 ids = _csr_view(elem_ptr, nodes, e)
                                 pa = areas === nothing ? polygon_area(pool, ids) : areas[e]
                                 _element_geometry(pool, ids, pa, parents[e], fragments[e])
                             end for e in 1:ne])
    faces = StructVector([begin
                              j = fedge[f]
                              a, b = _edge(elem_ptr, nodes, edge_elem[j], j)
                              k = fname[f] === nothing ? 0 : findfirst(==(fname[f]), names)
                              _mesh_face(pool, a, b, edge_elem[j], fright[f], k, fshift[f])
                          end for f in order])
    mesh = PolygonMesh(pool, elem_ptr, nodes, elem_faces, elements, faces, boundary_ptr, names)
    return assert_mesh(mesh)
end

@inline _csr_view(ptr, vals, e) = view(vals, Int(ptr[e]):(Int(ptr[e + 1]) - 1))

# The directed edge `j` of element `e`: from vertex `j` of `nodes` to the element's next vertex.
@inline function _edge(ptr, nodes, e, j)
    next = j == ptr[e + 1] - 1 ? ptr[e] : j + 1
    return (nodes[j], nodes[next])
end

function _per_element(x, ne, default, ::Type{S}, name) where {S}
    x === nothing && return fill(S(default), ne)
    length(x) == ne || _mesh_error("$name has $(length(x)) entries for $ne elements")
    return S[S(v) for v in x]
end
