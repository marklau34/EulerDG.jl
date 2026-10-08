"""
    EulerDG

Steady, inviscid, subsonic 2D Euler flow around static signed-distance geometry: a P1 modal
discontinuous Galerkin discretization on a Cartesian cut-cell mesh, converged by local pseudo-time
stepping.

After preprocessing, the solver sees only a [`PolygonMesh`](@ref): polygonal elements and straight
faces, with full cells and cut fragments on one code path. The mesh contract is the common layer,
included directly in this module. The pipeline stages are submodules, upstream of the contract or
downstream of it:

- upstream: [`Geometry`](@ref EulerDG.Geometry), the bodies, and [`CutMesh`](@ref EulerDG.CutMesh),
  the builders;
- downstream: [`Quadrature`](@ref EulerDG.Quadrature), [`Basis`](@ref EulerDG.Basis),
  [`Equations`](@ref EulerDG.Equations), [`Semidiscretization`](@ref EulerDG.Semidiscretization),
  [`PseudoTime`](@ref EulerDG.PseudoTime) and [`Post`](@ref EulerDG.Post).

Nothing downstream ever sees a Cartesian index; a test scans the sources to keep it that way. The
development plan is in `Cut-Cell DG Euler Solver — Architecture & Development Plan.md`, with one
implementation brief per phase in `briefs/`.
"""
module EulerDG

using Printf: @printf
using StaticArrays: SVector
using StructArrays: StructVector
using WriteVTK: MeshCell, PolyData, VTKCellData, vtk_grid, vtk_save

include("common/include.jl")

# Upstream of the mesh contract
include("Geometry/Geometry.jl")
include("CutMesh/CutMesh.jl")

# Downstream of it
include("Quadrature/Quadrature.jl")
include("Basis/Basis.jl")
include("Equations/Equations.jl")
include("Semidiscretization/Semidiscretization.jl")
include("PseudoTime/PseudoTime.jl")
include("Post/Post.jl")

# Common layer: the mesh contract
export MeshElement, MeshFace, PolygonMesh
export nelements, nfaces, npoints, nboundaries, element_nodes, element_faces
export interior_faces, boundary_faces, boundary_index, face_sign
export polygon_area, polygon_centroid, polygon_bounds, face_vector
export MESH_INVARIANTS, check_mesh, assert_mesh
export VF_BINS, mesh_summary, mesh_report, write_mesh_vtk

# The submodules export nothing yet. They fill in from Phase 1 on and are reached as
# `EulerDG.Basis` and so on; their names are not exported, because `Geometry` would collide with
# Meshes.jl's.

end # module EulerDG
