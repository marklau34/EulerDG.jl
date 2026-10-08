# =====================================
# Mesh VTK output
#
# Two PolyData files per mesh: the elements as polygons and the faces as lines, each with the
# contract's per-cell data. The closure-error field is the quickest way to find a bad element.

"""
    write_mesh_vtk(prefix, m) -> Vector{String}

Write `m` for ParaView as two PolyData files, and return their names:

- `prefix_elements.vtp` -- one polygon per element, with cell data `area`, `volume_fraction`,
  `parent`, `fragment`, `closure` (relative to the perimeter) and `vertices` (the vertex count).
- `prefix_faces.vtp` -- one line per face, drawn at its left copy, with cell data `boundary`
  (0 on interior faces), `left`, `right`, `length`, `normal` and `periodic`.
"""
function write_mesh_vtk(prefix::AbstractString, m::PolygonMesh)
    pts = zeros(Float64, 3, npoints(m))
    for (i, p) in enumerate(m.points)
        pts[1, i], pts[2, i] = p[1], p[2]
    end
    ne, nf = nelements(m), nfaces(m)
    files = String[]

    polys = [MeshCell(PolyData.Polys(), collect(Int, element_nodes(m, e))) for e in 1:ne]
    vtk = vtk_grid(prefix * "_elements", pts, polys)
    vtk["area", VTKCellData()] = collect(Float64, m.elements.area)
    vtk["volume_fraction", VTKCellData()] = collect(Float64, m.elements.volume_fraction)
    vtk["parent", VTKCellData()] = collect(m.elements.parent)
    vtk["fragment", VTKCellData()] = collect(m.elements.fragment)
    vtk["closure", VTKCellData()] = [Float64(_element_closure(m, e)) for e in 1:ne]
    vtk["vertices", VTKCellData()] = Int32[length(element_nodes(m, e)) for e in 1:ne]
    append!(files, vtk_save(vtk))

    lines = [MeshCell(PolyData.Lines(), [Int(a), Int(b)]) for (a, b) in m.faces.nodes]
    vtk = vtk_grid(prefix * "_faces", pts, lines)
    vtk["boundary", VTKCellData()] = collect(m.faces.boundary)
    vtk["left", VTKCellData()] = collect(m.faces.left)
    vtk["right", VTKCellData()] = collect(m.faces.right)
    vtk["length", VTKCellData()] = collect(Float64, m.faces.length)
    normals = zeros(Float64, 3, nf)
    for (f, n) in enumerate(m.faces.normal)
        normals[1, f], normals[2, f] = n[1], n[2]
    end
    vtk["normal", VTKCellData()] = normals
    vtk["periodic", VTKCellData()] = UInt8[!iszero(s) for s in m.faces.shift]
    append!(files, vtk_save(vtk))
    return files
end
