# =====================================
# Reference states: the freestream and the isentropic vortex
#
# Both are called like Trixi initial conditions, as `f(x, t, equations)`, so they serve as initial
# conditions, as Dirichlet boundary values and as exact solutions.

"""
    freestream(equations; mach, alpha_deg=0) -> SVector{4}

The uniform freestream in EulerDG's scaling, as conservative variables: density 1 and sound
speed 1 (pressure `1/gamma`), moving at Mach `mach` with angle of attack `alpha_deg`.
"""
function freestream(equations::CompressibleEulerEquations2D{T}; mach, alpha_deg=0) where {T}
    α = deg2rad(T(alpha_deg))
    return prim2cons(SVector{4,T}(1, mach * cos(α), mach * sin(α), 1 / equations.gamma),
                     equations)
end

"""
    IsentropicVortex(; center=(0, 0), velocity=(1, 1), strength=5, domain=((-5, -5), (5, 5)))

The isentropic vortex of Shu (1998), advected with `velocity` through the periodic `domain`.
Far from its core the density, pressure and temperature are 1; around the core they are
perturbed by

    δv = strength/(2π) exp((1 - r²)/2) (-ȳ, x̄),    δT = -(γ - 1) strength² / (8γπ²) exp(1 - r²)

with the entropy p/ρ^γ = 1 everywhere. `vortex(x, t, equations)` gives the exact solution at time
`t`, measured from the nearest periodic image of the advected centre. Its Gaussian tail does not
vanish at the domain edge, so on the default domain the periodic problem is exact only up to the
velocity perturbation there, about 2e-5.
"""
struct IsentropicVortex{T}
    center::SVector{2,T}
    velocity::SVector{2,T}
    strength::T
    lo::SVector{2,T}
    hi::SVector{2,T}
end

function IsentropicVortex(; center=(0.0, 0.0), velocity=(1.0, 1.0), strength=5.0,
                          domain=((-5.0, -5.0), (5.0, 5.0)))
    return IsentropicVortex(SVector{2,Float64}(center), SVector{2,Float64}(velocity),
                            Float64(strength), SVector{2,Float64}(domain[1]),
                            SVector{2,Float64}(domain[2]))
end

function (v::IsentropicVortex)(x, t, equations::CompressibleEulerEquations2D)
    L = v.hi - v.lo
    d = mod.(x - v.center - v.velocity * t + L / 2, L) - L / 2
    r2 = d[1]^2 + d[2]^2
    γ = equations.gamma
    e = exp((1 - r2) / 2)
    δv1 = -v.strength / (2π) * e * d[2]
    δv2 = v.strength / (2π) * e * d[1]
    T = 1 - (γ - 1) * v.strength^2 / (8 * γ * π^2) * e^2
    ρ = T^(1 / (γ - 1))
    return prim2cons(SVector(ρ, v.velocity[1] + δv1, v.velocity[2] + δv2, ρ * T), equations)
end
