# =====================================
# Quadrature: integration rules on segments and polygons
#
# Downstream of the mesh contract. Every rule takes its points from the mesh's vertex pool, so the
# volume and face integrals see exactly the geometry the contract checked.

"""
    EulerDG.Quadrature

Integration rules on element polygons and faces.

- [`GAUSS2`](@ref), [`GAUSS3`](@ref): Gauss-Legendre on [0, 1]. `GAUSS2` is the face rule.
- [`TRIANGLE3`](@ref), [`TRIANGLE6`](@ref): symmetric triangle rules of degrees 2 and 4.
- [`triangulate`](@ref): ear clipping of element polygons, nonconvex ones included.
- [`polygon_rule`](@ref): the element rule. Full cells get the tensor Gauss rule; other polygons
  are triangulated, with the degree-4 rule as the over-integration switch.
"""
module Quadrature

using StaticArrays: SVector

include("rules.jl")
include("polygon.jl")

export GAUSS2, GAUSS3, TRIANGLE3, TRIANGLE6, triangulate, polygon_rule

end # module Quadrature
