# =====================================
# The mesh invariants
#
# `check_mesh` counts the violations of each invariant and keeps the first offender of each, and
# `assert_mesh` throws if any count is nonzero. The builder runs `assert_mesh` on every mesh it
# makes, so these checks guard every builder from Phase 1 on; the Phase 2 gate runs them on every
# cut. The invariants:
#
# 1. csr -- the CSR arrays and every id are in range, and every pool point is used.
# 2. polygons -- each polygon is simple and counter-clockwise, with no repeated id.
# 3. edges -- each polygon edge agrees with its face: as its left element, as its right element in
#    reverse, or across a periodic shift.
# 4. faces -- each face is used once by its left element and, if interior, once by its right one.
# 5. order -- interior faces come first, then each boundary's faces as `boundary_ptr` says.
# 6. geometry -- every stored area, centroid, bound, volume fraction, face vector, length and normal
#    is bitwise what the polygon functions give.
# 7. closure -- each element's outward face vectors sum to zero, relative to its perimeter.
# 8. elements -- volume fractions in (0, 1], positive parent areas, parents >= 0, fragments >= 1,
#    and each (parent, fragment) pair used once.

"""
    MESH_INVARIANTS

The names of the invariants [`check_mesh`](@ref) counts, in the order it checks them.
"""
const MESH_INVARIANTS = (:csr, :polygons, :edges, :faces, :order, :geometry, :closure, :elements)

"""
    check_mesh(m) -> NamedTuple

Check the mesh invariants. The result holds:

- one violation count per invariant in [`MESH_INVARIANTS`](@ref): `csr`, `polygons`, `edges`,
  `faces`, `order`, `geometry`, `closure` and `elements`;
- `first` -- a NamedTuple with the first offender of each invariant, or `""`;
- `ok` -- whether every count is zero;
- `max_closure`, `max_closure_element` -- the largest closure error relative to the element's
  perimeter, and its element.

When the CSR arrays or an id are out of range, the other checks are skipped.
"""
function check_mesh(m::PolygonMesh{T}) where {T}
    nbad = Dict{Symbol,Int}(k => 0 for k in MESH_INVARIANTS)
    first = Dict{Symbol,String}(k => "" for k in MESH_INVARIANTS)
    function flag(msg, k)
        nbad[k] += 1
        nbad[k] == 1 && (first[k] = msg())
        return nothing
    end
    result(maxc, maxe) =
        merge((; ok=all(iszero, values(nbad))),
              NamedTuple{MESH_INVARIANTS}(Tuple(nbad[k] for k in MESH_INVARIANTS)),
              (; first=NamedTuple{MESH_INVARIANTS}(Tuple(first[k] for k in MESH_INVARIANTS)),
               max_closure=maxc, max_closure_element=maxe))

    ne, nf, np, nb = nelements(m), nfaces(m), npoints(m), nboundaries(m)
    ptr, nodes, efaces, F = m.elem_ptr, m.elem_nodes, m.elem_faces, m.faces

    # 1. CSR arrays and ids.
    if isempty(ptr) || ptr[1] != 1 || ptr[end] != length(nodes) + 1
        flag(:csr) do
            "elem_ptr must run from 1 to $(length(nodes) + 1), the element nodes plus one"
        end
    end
    for e in 1:ne
        n = ptr[e + 1] - ptr[e]
        n >= 3 || flag(() -> "element $e has $n vertices; a polygon needs at least 3", :csr)
    end
    length(efaces) == length(nodes) ||
        flag(() -> "elem_faces has $(length(efaces)) entries for $(length(nodes)) nodes", :csr)
    length(m.elements) == ne ||
        flag(() -> "there are $(length(m.elements)) element records for $ne elements", :csr)
    for (j, id) in enumerate(nodes)
        1 <= id <= np || flag(() -> "elem_nodes[$j] is pool id $id, outside 1:$np", :csr)
    end
    for (j, f) in enumerate(efaces)
        1 <= f <= nf || flag(() -> "elem_faces[$j] is face $f, outside 1:$nf", :csr)
    end
    for f in 1:nf
        a, b = F.nodes[f]
        (1 <= a <= np && 1 <= b <= np) || flag(() -> "face $f runs $a -> $b, outside 1:$np", :csr)
        1 <= F.left[f] <= ne || flag(() -> "face $f has left element $(F.left[f])", :csr)
        0 <= F.right[f] <= ne || flag(() -> "face $f has right element $(F.right[f])", :csr)
        0 <= F.boundary[f] <= nb || flag(() -> "face $f has boundary $(F.boundary[f])", :csr)
    end
    bp = m.boundary_ptr
    if length(bp) != nb + 1 || !(1 <= bp[1]) || bp[end] != nf + 1 || !issorted(bp)
        flag(() -> "boundary_ptr $bp does not split $nf faces into $nb boundaries", :csr)
    end
    if nbad[:csr] == 0
        used = falses(np)
        used[nodes] .= true
        i = findfirst(!, used)
        i === nothing || flag(() -> "pool point $i is not used by any element", :csr)
    end
    nbad[:csr] > 0 && return result(zero(T), 0)

    # 2. Polygons.
    for e in 1:ne
        problem = _polygon_problem(m.points, element_nodes(m, e))
        problem === nothing || flag(() -> "element $e: $problem", :polygons)
    end

    # 3. Edges agree with faces.
    nleft, nright = zeros(Int, nf), zeros(Int, nf)
    tol = _periodic_tol(T)
    for e in 1:ne
        ids, fs = element_nodes(m, e), element_faces(m, e)
        n = length(ids)
        for k in 1:n
            a, b, f = ids[k], ids[mod1(k + 1, n)], fs[k]
            if F.left[f] == e && F.nodes[f] == (a, b)
                nleft[f] += 1
            elseif F.right[f] == e && _runs_reverse(m, f, a, b, tol)
                nright[f] += 1
            else
                flag(:edges) do
                    "element $e, edge $k ($a -> $b) does not match its face $f " *
                    "($(F.nodes[f][1]) -> $(F.nodes[f][2]), left $(F.left[f]), right $(F.right[f]))"
                end
            end
        end
    end

    # 4. Each face used once per side.
    for f in 1:nf
        l, r, bd = F.left[f], F.right[f], F.boundary[f]
        nleft[f] == 1 || flag(() -> "face $f is used $(nleft[f]) times by its left element $l", :faces)
        if r == 0
            bd > 0 || flag(() -> "face $f has no right element and no boundary", :faces)
            iszero(F.shift[f]) || flag(() -> "boundary face $f has a periodic shift", :faces)
        else
            bd == 0 || flag(() -> "face $f has right element $r but is on boundary $bd", :faces)
            l != r || flag(() -> "face $f has element $l on both sides", :faces)
            nright[f] == 1 ||
                flag(() -> "face $f is used $(nright[f]) times by its right element $r", :faces)
        end
    end

    # 5. Face order.
    for f in 1:(bp[1] - 1)
        F.boundary[f] == 0 ||
            flag(() -> "face $f is in the interior range but on boundary $(F.boundary[f])", :order)
    end
    for k in 1:nb, f in bp[k]:(bp[k + 1] - 1)
        F.boundary[f] == k || flag(:order) do
            "face $f is in the range of boundary $k (:$(m.boundary_names[k])) but on boundary " *
            "$(F.boundary[f])"
        end
    end
    allunique(m.boundary_names) ||
        flag(() -> "boundary_names repeats a name: $(m.boundary_names)", :order)

    # 6. Derived geometry, bitwise.
    for e in 1:ne
        el = m.elements[e]
        ref = _element_geometry(m.points, element_nodes(m, e), el.parent_area, el.parent,
                                el.fragment)
        if !isequal(el, ref)
            field = fieldnames(typeof(el))[findfirst(i -> getfield(el, i) !== getfield(ref, i),
                                                     1:fieldcount(typeof(el)))]
            flag(:geometry) do
                "element $e: its $field is $(getfield(el, field)), but its polygon gives " *
                "$(getfield(ref, field))"
            end
        end
    end
    for f in 1:nf
        a, b = F.nodes[f]
        S, L, n = _face_geometry(m.points, a, b)
        (F.vector[f] === S && F.length[f] === L && F.normal[f] === n) || flag(:geometry) do
            "face $f ($a -> $b): its stored vector, length or normal differs from its endpoints'"
        end
    end

    # 7. Closure.
    tolc = _closure_tol(T)
    maxc, maxe = zero(T), 0
    for e in 1:ne
        c = _element_closure(m, e)
        (maxe == 0 || c > maxc) && ((maxc, maxe) = (c, e))
        c <= tolc || flag(:closure) do
            "element $e: its face vectors sum to $c of its perimeter, above $tolc"
        end
    end

    # 8. Element data.
    vmax = _vf_max(T)
    owner = Dict{NTuple{2,Int32},Int}()
    for e in 1:ne
        el = m.elements[e]
        0 < el.volume_fraction <= vmax ||
            flag(() -> "element $e has volume fraction $(el.volume_fraction)", :elements)
        el.parent_area > 0 ||
            flag(() -> "element $e has parent area $(el.parent_area)", :elements)
        el.parent >= 0 || flag(() -> "element $e has parent $(el.parent)", :elements)
        el.fragment >= 1 || flag(() -> "element $e has fragment $(el.fragment)", :elements)
        if el.parent > 0
            e0 = get!(owner, (el.parent, el.fragment), e)
            e0 == e || flag(:elements) do
                "elements $e0 and $e are both fragment $(el.fragment) of parent $(el.parent)"
            end
        end
    end
    return result(maxc, maxe)
end

# Whether face `f`, seen from its right element, runs `a -> b`: its own nodes reversed, or for a
# periodic face its nodes reversed and moved by the shift.
function _runs_reverse(m::PolygonMesh, f, a, b, tol)
    c, d = m.faces.nodes[f]
    s = m.faces.shift[f]
    iszero(s) && return a == d && b == c
    h = tol * m.faces.length[f]
    return _near(m.points[a], m.points[d] + s, h) && _near(m.points[b], m.points[c] + s, h)
end

"""
    assert_mesh(m) -> m

Return `m` if it satisfies every invariant of [`check_mesh`](@ref). Otherwise throw an
`ArgumentError` that lists each broken invariant with its count and first offender.
"""
function assert_mesh(m::PolygonMesh)
    c = check_mesh(m)
    c.ok && return m
    broken = [k for k in MESH_INVARIANTS if c[k] > 0]
    lines = ["the mesh breaks $(length(broken)) of its $(length(MESH_INVARIANTS)) invariants:"]
    for k in broken
        push!(lines, "  $k: $(c[k]) violation$(c[k] == 1 ? "" : "s"); the first: $(c.first[k])")
    end
    _mesh_error(join(lines, "\n"))
end
