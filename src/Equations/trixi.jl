# =====================================
# Copied from Trixi.jl 0.17.13
#
# The 2D compressible Euler equations and the fluxes EulerDG uses, copied from Trixi.jl rather
# than taken as a dependency (decided Oct 8, 2026). Names, argument order and the equation type are
# Trixi's, so switching to the dependency is a change of `using` lines. `dev/check_trixi.jl`
# cross-checks these copies against Trixi itself.
#
# Sources: src/equations/equations.jl, compressible_euler_2d.jl and numerical_fluxes.jl. Changes:
#
# - The `@muladd` blocks are dropped, so there are no fused multiply-adds. Results are the same on
#   every machine, and differ from Trixi's by round-off only.
# - `@unpack` is replaced by field access, and `norm` comes from LinearAlgebra directly.
# - Only the methods EulerDG uses are kept.
#
# The original carries this notice:
#
# MIT License
#
# Copyright (c) 2020-present The Trixi.jl Authors (see AUTHORS.md)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =====================================
# equations.jl

abstract type AbstractEquations{NDIMS, NVARS} end

abstract type AbstractCompressibleEulerEquations{NDIMS, NVARS} <:
              AbstractEquations{NDIMS, NVARS} end

# Retrieve number of variables from equation instance
@inline nvariables(::AbstractEquations{NDIMS, NVARS}) where {NDIMS, NVARS} = NVARS

@inline Base.ndims(::AbstractEquations{NDIMS}) where {NDIMS} = NDIMS

"""
    BoundaryConditionDirichlet(boundary_value_function)

Create a Dirichlet boundary condition that uses the function `boundary_value_function`
to specify the values at the boundary.
This can be used to create a boundary condition that specifies exact boundary values
by passing the exact solution of the equation.
The passed boundary value function will be called with the same arguments as an initial condition function is called, i.e., as
```julia
boundary_value_function(x, t, equations)
```
where `x` specifies the coordinates, `t` is the current time, and `equation` is the corresponding system of equations.
"""
struct BoundaryConditionDirichlet{B}
    boundary_value_function::B
end

# Dirichlet-type boundary condition for use with UnstructuredMesh2D
# Note: For unstructured we lose the concept of an "absolute direction"
@inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner,
                                                                  normal_direction::AbstractVector,
                                                                  x, t,
                                                                  surface_flux_function,
                                                                  equations)
    # get the external value of the solution
    u_boundary = boundary_condition.boundary_value_function(x, t, equations)

    # Calculate boundary flux
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux
end

# =====================================
# compressible_euler_2d.jl

@doc raw"""
    CompressibleEulerEquations2D(gamma)

The compressible Euler equations
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho \\ \rho v_1 \\ \rho v_2 \\ \rho e_{\text{total}}
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
 \rho v_1 \\ \rho v_1^2 + p \\ \rho v_1 v_2 \\ (\rho e_{\text{total}} +p) v_1
\end{pmatrix}
+
\frac{\partial}{\partial y}
\begin{pmatrix}
\rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + p \\ (\rho e_{\text{total}} +p) v_2
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ 0 \\ 0
\end{pmatrix}
```
for an ideal gas with ratio of specific heats `gamma`
in two space dimensions.
Here, ``\rho`` is the density, ``v_1``, ``v_2`` the velocities, ``e_{\text{total}}`` the specific total energy, and
```math
p = (\gamma - 1) \left( \rho e_{\text{total}} - \frac{1}{2} \rho (v_1^2+v_2^2) \right)
```
the pressure.
"""
struct CompressibleEulerEquations2D{RealT <: Real} <:
       AbstractCompressibleEulerEquations{2, 4}
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

    function CompressibleEulerEquations2D(gamma)
        γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
        return new{typeof(γ)}(γ, inv_gamma_minus_one)
    end
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e_total = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e_total - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2))
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = (rho_e_total + p) * v1
    else
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p
        f4 = (rho_e_total + p) * v2
    end
    return SVector(f1, f2, f3, f4)
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector,
                      equations::CompressibleEulerEquations2D)
    rho_e_total = last(u)
    rho, v1, v2, p = cons2prim(u, equations)

    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    rho_v_normal = rho * v_normal
    f1 = rho_v_normal
    f2 = rho_v_normal * v1 + p * normal_direction[1]
    f3 = rho_v_normal * v2 + p * normal_direction[2]
    f4 = (rho_e_total + p) * v_normal
    return SVector(f1, f2, f3, f4)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquations2D)
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    # Calculate normal velocities and sound speed
    # left
    v_ll = (v1_ll * normal_direction[1]
            +
            v2_ll * normal_direction[2])
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    # right
    v_rr = (v1_rr * normal_direction[1]
            +
            v2_rr * normal_direction[2])
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end

# Less "cautious", i.e., less overestimating `λ_max` compared to `max_abs_speed_naive`
@inline function max_abs_speed(u_ll, u_rr, normal_direction::AbstractVector,
                               equations::CompressibleEulerEquations2D)
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    # Calculate normal velocities and sound speeds
    # left
    v_ll = (v1_ll * normal_direction[1]
            +
            v2_ll * normal_direction[2])
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    # right
    v_rr = (v1_rr * normal_direction[1]
            +
            v2_rr * normal_direction[2])
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    norm_ = norm(normal_direction)
    return max(abs(v_ll) + c_ll * norm_,
               abs(v_rr) + c_rr * norm_)
end

@inline function max_abs_speeds(u, equations::CompressibleEulerEquations2D)
    rho, v1, v2, p = cons2prim(u, equations)
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c
end

"""
    cons2prim(u, equations::CompressibleEulerEquations2D)

Convert conservative variables `u = (rho, rho*v1, rho*v2, rho*e_total)` to
primitive variables `(rho, v1, v2, p)`.
"""
@inline function cons2prim(u, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e_total = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e_total - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2))

    return SVector(rho, v1, v2, p)
end

"""
    prim2cons(prim, equations::CompressibleEulerEquations2D)

Convert primitive variables `prim = (rho, v1, v2, p)` to
conservative variables `(rho, rho*v1, rho*v2, rho*e_total)`.
"""
@inline function prim2cons(prim, equations::CompressibleEulerEquations2D)
    rho, v1, v2, p = prim
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_e_total = p * equations.inv_gamma_minus_one +
                  0.5f0 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e_total)
end

@inline function density(u, equations::CompressibleEulerEquations2D)
    rho = u[1]
    return rho
end

@inline function velocity(u, equations::CompressibleEulerEquations2D)
    rho = u[1]
    v1 = u[2] / rho
    v2 = u[3] / rho
    return SVector(v1, v2)
end

@inline function pressure(u, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e_total = u
    p = (equations.gamma - 1) * (rho_e_total - 0.5f0 * (rho_v1^2 + rho_v2^2) / rho)
    return p
end

# =====================================
# numerical_fluxes.jl

"""
    flux_central(u_ll, u_rr, orientation_or_normal_direction, equations::AbstractEquations)

The classical central numerical flux `f((u_ll) + f(u_rr)) / 2`. When this flux is
used as volume flux, the discretization is equivalent to the classical weak form
DG method (except floating point errors).
"""
@inline function flux_central(u_ll, u_rr, orientation_or_normal_direction,
                              equations::AbstractEquations)
    # Calculate regular 1D fluxes
    f_ll = flux(u_ll, orientation_or_normal_direction, equations)
    f_rr = flux(u_rr, orientation_or_normal_direction, equations)

    # Average regular fluxes
    return 0.5f0 * (f_ll + f_rr)
end

"""
    FluxPlusDissipation(numerical_flux, dissipation)

Combine a `numerical_flux` with a `dissipation` operator to create a new numerical flux.
"""
struct FluxPlusDissipation{NumericalFlux, Dissipation}
    numerical_flux::NumericalFlux
    dissipation::Dissipation
end

@inline function (numflux::FluxPlusDissipation)(u_ll, u_rr,
                                                orientation_or_normal_direction,
                                                equations)
    numerical_flux, dissipation = numflux.numerical_flux, numflux.dissipation

    return (numerical_flux(u_ll, u_rr, orientation_or_normal_direction, equations)
            +
            dissipation(u_ll, u_rr, orientation_or_normal_direction, equations))
end

function Base.show(io::IO, f::FluxPlusDissipation)
    print(io, "FluxPlusDissipation(", f.numerical_flux, ", ", f.dissipation, ")")
    return nothing
end

"""
    DissipationLocalLaxFriedrichs(max_abs_speed=max_abs_speed)

Create a local Lax-Friedrichs dissipation operator where the maximum absolute wave speed
is estimated as
`max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)`,
defaulting to [`max_abs_speed`](@ref).
"""
struct DissipationLocalLaxFriedrichs{MaxAbsSpeed}
    max_abs_speed::MaxAbsSpeed
end

DissipationLocalLaxFriedrichs() = DissipationLocalLaxFriedrichs(max_abs_speed)

@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations)
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)
    return -0.5f0 * λ * (u_rr - u_ll)
end

function Base.show(io::IO, d::DissipationLocalLaxFriedrichs)
    print(io, "DissipationLocalLaxFriedrichs(", d.max_abs_speed, ")")
    return nothing
end

const FluxLaxFriedrichs{MaxAbsSpeed} = FluxPlusDissipation{typeof(flux_central),
                                                           DissipationLocalLaxFriedrichs{MaxAbsSpeed}}
"""
    FluxLaxFriedrichs(max_abs_speed=max_abs_speed)

Local Lax-Friedrichs (Rusanov) flux with maximum wave speed estimate provided by
`max_abs_speed`, cf. [`DissipationLocalLaxFriedrichs`](@ref) and
[`max_abs_speed_naive`](@ref).
"""
function FluxLaxFriedrichs(max_abs_speed = max_abs_speed)
    return FluxPlusDissipation(flux_central,
                               DissipationLocalLaxFriedrichs(max_abs_speed))
end

function Base.show(io::IO, f::FluxLaxFriedrichs)
    print(io, "FluxLaxFriedrichs(", f.dissipation.max_abs_speed, ")")
    return nothing
end

"""
    flux_lax_friedrichs

See [`FluxLaxFriedrichs`](@ref).
"""
const flux_lax_friedrichs = FluxLaxFriedrichs()

# =====================================
# compressible_euler_2d.jl, continued: rotations, HLLC and the slip wall (copied in Phase 3)

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, equations::CompressibleEulerEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D rotation matrix with normal and tangent directions of the form
    # [ 1    0    0   0;
    #   0   n_1  n_2  0;
    #   0   t_1  t_2  0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    return SVector(u[1],
                   c * u[2] + s * u[3],
                   -s * u[2] + c * u[3],
                   u[4])
end

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector,
                               equations::CompressibleEulerEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D back-rotation matrix with normal and tangent directions of the form
    # [ 1    0    0   0;
    #   0   n_1  t_1  0;
    #   0   n_2  t_2  0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    return SVector(u[1],
                   c * u[2] - s * u[3],
                   s * u[2] + c * u[3],
                   u[4])
end

"""
    flux_hllc(u_ll, u_rr, orientation_or_normal_direction, equations::CompressibleEulerEquations2D)

Computes the HLLC flux (HLL with Contact) for compressible Euler equations developed by E.F. Toro
[Lecture slides](http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
Signal speeds: [DOI: 10.1137/S1064827593260140](https://doi.org/10.1137/S1064827593260140)
"""
function flux_hllc(u_ll, u_rr, normal_direction::AbstractVector,
                   equations::CompressibleEulerEquations2D)
    # Calculate primitive variables and speed of sound
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    norm_ = norm(normal_direction)
    norm_sq = norm_ * norm_
    inv_norm_sq = inv(norm_sq)

    c_ll = sqrt(equations.gamma * p_ll / rho_ll) * norm_
    c_rr = sqrt(equations.gamma * p_rr / rho_rr) * norm_

    # Obtain left and right fluxes
    f_ll = flux(u_ll, normal_direction, equations)
    f_rr = flux(u_rr, normal_direction, equations)

    # Compute Roe averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr

    v1_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr) / sum_sqrt_rho
    v2_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr) / sum_sqrt_rho
    vel_roe = v1_roe * normal_direction[1] + v2_roe * normal_direction[2]
    vel_roe_mag = v1_roe^2 + v2_roe^2

    e_ll = u_ll[4] / rho_ll
    e_rr = u_rr[4] / rho_rr

    H_ll = (u_ll[4] + p_ll) / rho_ll
    H_rr = (u_rr[4] + p_rr) / rho_rr

    H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
    c_roe = sqrt((equations.gamma - 1) * (H_roe - 0.5f0 * vel_roe_mag)) * norm_

    Ssl = min(v_dot_n_ll - c_ll, vel_roe - c_roe)
    Ssr = max(v_dot_n_rr + c_rr, vel_roe + c_roe)
    sMu_L = Ssl - v_dot_n_ll
    sMu_R = Ssr - v_dot_n_rr

    if Ssl >= 0
        f1 = f_ll[1]
        f2 = f_ll[2]
        f3 = f_ll[3]
        f4 = f_ll[4]
    elseif Ssr <= 0
        f1 = f_rr[1]
        f2 = f_rr[2]
        f3 = f_rr[3]
        f4 = f_rr[4]
    else
        SStar = (rho_ll * v_dot_n_ll * sMu_L - rho_rr * v_dot_n_rr * sMu_R +
                 (p_rr - p_ll) * norm_sq) / (rho_ll * sMu_L - rho_rr * sMu_R)
        if Ssl <= 0 <= SStar
            densStar = rho_ll * sMu_L / (Ssl - SStar)
            enerStar = e_ll +
                       (SStar - v_dot_n_ll) *
                       (SStar * inv_norm_sq + p_ll / (rho_ll * sMu_L))
            UStar1 = densStar
            UStar2 = densStar *
                     (v1_ll + (SStar - v_dot_n_ll) * normal_direction[1] * inv_norm_sq)
            UStar3 = densStar *
                     (v2_ll + (SStar - v_dot_n_ll) * normal_direction[2] * inv_norm_sq)
            UStar4 = densStar * enerStar
            f1 = f_ll[1] + Ssl * (UStar1 - u_ll[1])
            f2 = f_ll[2] + Ssl * (UStar2 - u_ll[2])
            f3 = f_ll[3] + Ssl * (UStar3 - u_ll[3])
            f4 = f_ll[4] + Ssl * (UStar4 - u_ll[4])
        else
            densStar = rho_rr * sMu_R / (Ssr - SStar)
            enerStar = e_rr +
                       (SStar - v_dot_n_rr) *
                       (SStar * inv_norm_sq + p_rr / (rho_rr * sMu_R))
            UStar1 = densStar
            UStar2 = densStar *
                     (v1_rr + (SStar - v_dot_n_rr) * normal_direction[1] * inv_norm_sq)
            UStar3 = densStar *
                     (v2_rr + (SStar - v_dot_n_rr) * normal_direction[2] * inv_norm_sq)
            UStar4 = densStar * enerStar
            f1 = f_rr[1] + Ssr * (UStar1 - u_rr[1])
            f2 = f_rr[2] + Ssr * (UStar2 - u_rr[2])
            f3 = f_rr[3] + Ssr * (UStar3 - u_rr[3])
            f4 = f_rr[4] + Ssr * (UStar4 - u_rr[4])
        end
    end
    return SVector(f1, f2, f3, f4)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::CompressibleEulerEquations2D)

Determine the boundary numerical surface flux for a slip wall condition.
Imposes a zero normal velocity at the wall.
Density is taken from the internal solution state and pressure is computed as an
exact solution of a 1D Riemann problem. Further details about this boundary state
are available in the paper:
- J. J. W. van der Vegt and H. van der Ven (2002)
  Slip flow boundary conditions in discontinuous Galerkin discretizations of
  the Euler equations of gas dynamics
  [PDF](https://reports.nlr.nl/bitstream/handle/10921/692/TP-2002-300.pdf?sequence=1)

Details about the 1D pressure Riemann solution can be found in Section 6.3.3 of the book
- Eleuterio F. Toro (2009)
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
  3rd edition
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

The implementation is modified to ensure non-negativity of `p_star`. This modification
preserves entropy stability based on the analysis in the paper:
- F. J. Hindenlang, G. J. Gassner, D. A. Kopriva (2020)
  Stability of wall boundary condition procedures for discontinuous Galerkin spectral element approximations of the compressible Euler equations
  [DOI: 10.1007/978-3-030-39647-3_1](https://doi.org/10.1007/978-3-030-39647-3_1)
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquations2D)
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # rotate the internal solution state
    u_local = rotate_to_x(u_inner, normal, equations)

    # compute the primitive variables
    rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

    # Get the solution of the pressure Riemann problem
    # See Section 6.3.3 of
    # Eleuterio F. Toro (2009)
    # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
    # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
    if v_normal <= 0
        sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
        p_scaling_base = (1 + 0.5f0 * (equations.gamma - 1) * v_normal / sound_speed)
        if p_scaling_base >= 0
            p_star = p_local *
                     p_scaling_base^(2 * equations.gamma *
                                     equations.inv_gamma_minus_one)
        else # avoid taking powers if p_scaling_base < 0
            p_star = zero(p_local)
        end
    else # v_normal > 0
        A = 2 / ((equations.gamma + 1) * rho_local)
        B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
        p_star = p_local +
                 0.5f0 * v_normal / A *
                 (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
    end

    # For the slip wall we directly set the flux as the normal velocity is zero
    return SVector(0,
                   p_star * normal[1],
                   p_star * normal[2],
                   0) * norm_
end
