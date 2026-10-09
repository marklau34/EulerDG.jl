# =====================================
# The mirror-state slip wall
#
# The plan's default wall: the outer state is the inner one with its normal momentum reversed, and
# the boundary flux is the surface flux between the two, so the wall sees the same Riemann solver
# as every interior face. Trixi's `boundary_condition_slip_wall` (trixi.jl) is the alternative the
# plan keeps for comparison: a pressure-only flux, with the pressure from the exact 1D Riemann
# problem against the wall.

"""
    boundary_condition_mirror_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                   equations::CompressibleEulerEquations2D)

A slip wall by mirror state: the outer state is `u_inner` with its normal momentum reversed --
density and energy unchanged, since the kinetic energy is -- and the boundary flux is
`surface_flux_function` between the two. For flow tangent to the wall the outer state is the
inner one bitwise, and every consistent flux returns the wall pressure's `(0, p n, 0)`.
"""
@inline function boundary_condition_mirror_wall(u_inner, normal_direction::AbstractVector, x, t,
                                                surface_flux_function,
                                                equations::CompressibleEulerEquations2D)
    n = normal_direction / norm(normal_direction)
    mn = u_inner[2] * n[1] + u_inner[3] * n[2]
    u_outer = SVector(u_inner[1], u_inner[2] - 2 * mn * n[1], u_inner[3] - 2 * mn * n[2],
                      u_inner[4])
    return surface_flux_function(u_inner, u_outer, normal_direction, equations)
end
