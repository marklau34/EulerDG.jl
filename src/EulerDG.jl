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
development plan is in `Cut-Cell DG Euler Solver — Architecture & Development Plan.md`, with
implementation briefs in `briefs/` and notes on each completed phase in `dev-notes/`.
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

using .Geometry
using .CutMesh
using .Quadrature
using .Basis
using .Equations
using .Semidiscretization
using .PseudoTime
using .Post

# Common layer: the mesh contract
export MeshElement, MeshFace, PolygonMesh
export nelements, nfaces, npoints, nboundaries, element_nodes, element_faces
export interior_faces, boundary_faces, boundary_index, face_sign
export polygon_area, polygon_centroid, polygon_bounds, face_vector
export MESH_INVARIANTS, check_mesh, assert_mesh
export VF_BINS, mesh_summary, mesh_report, write_mesh_vtk

# Submodules and their API. The submodule names themselves are not exported, because `Geometry`
# would collide with Meshes.jl's; reach them as `EulerDG.Basis` and so on.
export Body, polygon_body, circle_points, naca4_points, polygon_perimeter, grid_shifts
export sliver_shift
export uncut_mesh, cut_mesh, cut_report, halfplane_mesh
export GAUSS2, GAUSS3, TRIANGLE3, TRIANGLE6, triangulate, polygon_rule
export ElementBasis, element_basis, basis_values, basis_gradients, mass_matrix
export AbstractEquations, AbstractCompressibleEulerEquations, CompressibleEulerEquations2D
export nvariables, prim2cons, cons2prim, density, velocity, pressure
export flux, flux_central, FluxPlusDissipation, DissipationLocalLaxFriedrichs, FluxLaxFriedrichs
export flux_lax_friedrichs, flux_hllc, max_abs_speed, max_abs_speed_naive, max_abs_speeds
export rotate_to_x, rotate_from_x
export BoundaryConditionDirichlet, boundary_condition_slip_wall, boundary_condition_mirror_wall
export BoundaryConditionFarfield
export freestream, IsentropicVortex
export DGSemidiscretization, allocate_state, project, evaluate, cell_average, rhs!, max_dt
export local_dt!, assert_positive, face_trace
export SSPRK3Cache, ssprk3_step!, integrate!, residual_norms, solve_steady!
export l2_error, entropy_error, wall_forces, surface_cp, RunLog, write_solution_vtk

end # module EulerDG
