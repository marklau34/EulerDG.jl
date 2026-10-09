# =====================================
# The characteristic farfield
#
# A Riemann-invariant farfield, which Trixi does not have for the Euler equations. Along the face's
# outward normal, a subsonic boundary carries one invariant in from the freestream and one out from
# the interior; the entropy and the tangential velocity come from upwind, the freestream at inflow
# and the interior at outflow. The boundary state they make is the outer state of the surface flux,
# as for every other boundary, so a disturbance reaching the boundary leaves with little reflection
# and the freestream itself passes through unchanged.

"""
    BoundaryConditionFarfield(u_infinity)

A characteristic farfield for the freestream `u_infinity` (conservative variables), called like
the other boundary conditions as `bc(u_inner, normal_direction, x, t, surface_flux, equations)`.

With `n` the unit outward normal, `v_n` the normal velocity and `c` the sound speed, the Riemann
invariants `R⁺ = v_n + 2c/(γ-1)` from the interior and `R⁻ = v_n - 2c/(γ-1)` from the freestream
give the boundary's normal velocity and sound speed. The entropy `p/ρ^γ` and the tangential
velocity come from the freestream where the flow enters and from the interior where it leaves.
Supersonic inflow takes the freestream, and supersonic outflow the interior state. The boundary
flux is `surface_flux` between the interior and boundary states.
"""
struct BoundaryConditionFarfield{V}
    u_infinity::V
end

@inline function (bc::BoundaryConditionFarfield)(u_inner, normal_direction::AbstractVector, x, t,
                                                surface_flux_function,
                                                equations::CompressibleEulerEquations2D)
    γ = equations.gamma
    n = normal_direction / norm(normal_direction)
    ρi, v1i, v2i, p_i = cons2prim(u_inner, equations)
    ρo, v1o, v2o, p_o = cons2prim(bc.u_infinity, equations)
    vni = v1i * n[1] + v2i * n[2]
    vno = v1o * n[1] + v2o * n[2]
    ci = sqrt(γ * p_i / ρi)
    co = sqrt(γ * p_o / ρo)
    u_boundary = if vni <= -ci
        bc.u_infinity
    elseif vni >= ci
        u_inner
    else
        rp = vni + 2 * ci * equations.inv_gamma_minus_one
        rm = vno - 2 * co * equations.inv_gamma_minus_one
        vnb = (rp + rm) / 2
        cb = (rp - rm) / (4 * equations.inv_gamma_minus_one)
        # Upwind: the freestream's entropy and tangential velocity at inflow, the interior's at
        # outflow.
        s, vt1, vt2 = vnb < 0 ? (p_o / ρo^γ, v1o - vno * n[1], v2o - vno * n[2]) :
                      (p_i / ρi^γ, v1i - vni * n[1], v2i - vni * n[2])
        ρb = (cb^2 / (γ * s))^equations.inv_gamma_minus_one
        prim2cons(SVector(ρb, vt1 + vnb * n[1], vt2 + vnb * n[2], ρb * cb^2 / γ), equations)
    end
    return surface_flux_function(u_inner, u_boundary, normal_direction, equations)
end
