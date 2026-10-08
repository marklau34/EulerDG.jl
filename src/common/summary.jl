# =====================================
# The mesh summary
#
# What a mesh contains, measured: counts of elements by type, the volume-fraction distribution,
# the worst closure error, and each boundary's faces and length. `mesh_summary` returns the numbers
# and `mesh_report` prints them; the cut preprocessor's own summary (Phase 2) adds its lost-feature
# and truncated-tip lists on top.

"""
    VF_BINS

The labels of the volume-fraction histogram of [`mesh_summary`](@ref): one decade per bin from
below 1e-12 up to [0.1, 1), then volume fraction 1.
"""
const VF_BINS = ("< 1e-12", "[1e-12, 1e-11)", "[1e-11, 1e-10)", "[1e-10, 1e-9)", "[1e-9, 1e-8)",
                 "[1e-8, 1e-7)", "[1e-7, 1e-6)", "[1e-6, 1e-5)", "[1e-5, 1e-4)", "[1e-4, 1e-3)",
                 "[1e-3, 0.01)", "[0.01, 0.1)", "[0.1, 1)", "1")

# The histogram bin of volume fraction `v`; a NaN, possible only in an unchecked mesh, goes to the
# lowest bin.
_vf_bin(v) = v >= 1 ? 14 : !(v >= 1e-12) ? 1 : clamp(floor(Int, log10(v)) + 14, 2, 13)

"""
    mesh_summary(m) -> NamedTuple

Measure a mesh:

- `npoints`, `nelements`, `nfaces`, `ninterior`, `nperiodic` -- counts.
- `nfull`, `ncut`, `nsplit` -- elements by type. An element is split when its parent has two or
  more elements, full when it is the only one and its volume fraction is 1, and cut otherwise.
- `nnonconvex`, `max_vertices` -- elements with a reflex vertex, and the largest vertex count.
- `min_volume_fraction`, `min_volume_fraction_element`.
- `vf_histogram` -- element counts per [`VF_BINS`](@ref) bin.
- `max_closure`, `max_closure_element` -- the largest closure error relative to the element's
  perimeter.
- `area` -- the total area of the elements.
- `boundaries` -- a NamedTuple with each boundary's `faces` count and `length`.
- `boundary_closure` -- the length of the sum of all boundary face vectors: zero, at round-off,
  for a closed domain.
"""
function mesh_summary(m::PolygonMesh{T}) where {T}
    ne = nelements(m)
    E = m.elements
    siblings = Dict{Int32,Int}()
    for p in E.parent
        p > 0 && (siblings[p] = get(siblings, p, 0) + 1)
    end
    nfull = ncut = nsplit = nnonconvex = max_vertices = 0
    hist = zeros(Int, length(VF_BINS))
    minvf, minvf_e = T(Inf), 0
    maxc, maxc_e = zero(T), 0
    for e in 1:ne
        vf, p = E.volume_fraction[e], E.parent[e]
        if p > 0 && siblings[p] > 1
            nsplit += 1
        elseif vf >= 1
            nfull += 1
        else
            ncut += 1
        end
        hist[_vf_bin(vf)] += 1
        vf < minvf && ((minvf, minvf_e) = (vf, e))
        ids = element_nodes(m, e)
        n = length(ids)
        max_vertices = max(max_vertices, n)
        reflex = any(k -> _cross(m.points[ids[mod1(k + 1, n)]] - m.points[ids[k]],
                                 m.points[ids[mod1(k + 2, n)]] - m.points[ids[mod1(k + 1, n)]]) < 0,
                     1:n)
        nnonconvex += reflex
        c = _element_closure(m, e)
        (maxc_e == 0 || c > maxc) && ((maxc, maxc_e) = (c, e))
    end
    S = zero(SVector{2,T})
    for k in 1:nboundaries(m), f in boundary_faces(m, k)
        S += m.faces.vector[f]
    end
    boundaries = NamedTuple{Tuple(m.boundary_names)}(Tuple(
        (; faces=length(boundary_faces(m, k)),
         length=sum(m.faces.length[f] for f in boundary_faces(m, k); init=zero(T)))
        for k in 1:nboundaries(m)))
    return (; npoints=npoints(m), nelements=ne, nfaces=nfaces(m),
            ninterior=length(interior_faces(m)),
            nperiodic=count(f -> !iszero(m.faces.shift[f]), interior_faces(m)),
            nfull, ncut, nsplit, nnonconvex, max_vertices,
            min_volume_fraction=minvf, min_volume_fraction_element=minvf_e,
            vf_histogram=hist, max_closure=maxc, max_closure_element=maxc_e,
            area=sum(E.area), boundaries, boundary_closure=hypot(S[1], S[2]))
end

"""
    mesh_report([io=stdout,] m) -> NamedTuple

Print [`mesh_summary`](@ref)`(m)` and return it.
"""
function mesh_report(io::IO, m::PolygonMesh{T}) where {T}
    s = mesh_summary(m)
    @printf(io, "PolygonMesh{%s}: %d elements, %d faces, %d points\n", T, s.nelements, s.nfaces,
            s.npoints)
    @printf(io, "  elements: %d full, %d cut, %d split; %d nonconvex; at most %d vertices\n",
            s.nfull, s.ncut, s.nsplit, s.nnonconvex, s.max_vertices)
    @printf(io, "  area %.12e; largest closure error %.3e (element %d)\n", s.area, s.max_closure,
            s.max_closure_element)
    @printf(io, "  faces: %d interior, %d of them periodic\n", s.ninterior, s.nperiodic)
    for (name, b) in pairs(s.boundaries)
        @printf(io, "    %-12s %8d faces, length %.12e\n", name, b.faces, b.length)
    end
    @printf(io, "  boundary closure %.3e\n", s.boundary_closure)
    @printf(io, "  volume fraction: min %.6e (element %d)\n", s.min_volume_fraction,
            s.min_volume_fraction_element)
    for (label, n) in zip(VF_BINS, s.vf_histogram)
        n > 0 && @printf(io, "    %-16s %8d\n", label, n)
    end
    return s
end

mesh_report(m::PolygonMesh) = mesh_report(stdout, m)
