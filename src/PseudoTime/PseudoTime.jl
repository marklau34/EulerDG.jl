# =====================================
# PseudoTime: time marching
#
# Downstream of the mesh contract. Unsteady runs march in physical time with one global step.
# Steady runs march in pseudo-time with local steps: each element advances with its own stable
# step, which acts as a diagonal preconditioner, so a sliver's tiny step never limits its
# neighbours. The steps are frozen at the start of each iteration and kept across the Runge-Kutta
# stages, which integrates dU/dτ = D M⁻¹ (-R(U)) with D = diag(Δt_K) and dτ = 1. The steady
# solution does not depend on the pseudo-time path.

"""
    EulerDG.PseudoTime

Time marching.

- Phase 1: [`ssprk3_step!`](@ref), the three-stage strong-stability-preserving Runge-Kutta
  method of Shu and Osher, and [`integrate!`](@ref), which marches to an end time with a global
  step and a positivity check after every step.
- Phase 3: local steps -- `ssprk3_step!` with one step per element, frozen across the stages --
  and [`solve_steady!`](@ref), which marches to a steady state from a CFL ramp, monitoring the
  cell-average density residual with [`residual_norms`](@ref).
- Phase 4: the run log of residual and force histories.
"""
module PseudoTime

using ..EulerDG: nelements
using ..Semidiscretization: DGSemidiscretization, assert_positive, local_dt!, max_dt, rhs!

export SSPRK3Cache, ssprk3_step!, integrate!, residual_norms, solve_steady!

"""
    SSPRK3Cache(u)

Work arrays for [`ssprk3_step!`](@ref), shaped like the state `u`.
"""
struct SSPRK3Cache{A}
    u1::A
    u2::A
    du::A
end

SSPRK3Cache(u::AbstractArray) = SSPRK3Cache(similar(u), similar(u), similar(u))

"""
    ssprk3_step!(u, semi, t, dt, cache) -> u

Advance `u` by one SSP-RK3 step:

    u1 = u + dt L(u),   u2 = 3/4 u + 1/4 (u1 + dt L(u1)),   u = 1/3 u + 2/3 (u2 + dt L(u2))

`dt` is either a number, the global step from `t` to `t + dt`, or a vector of one step per element,
frozen across the stages: a pseudo-time step towards a steady state, with `t` held fixed.
"""
function ssprk3_step!(u, semi::DGSemidiscretization, t, dt::Real, cache::SSPRK3Cache)
    u1, u2, du = cache.u1, cache.u2, cache.du
    rhs!(du, u, semi, t)
    @. u1 = u + dt * du
    rhs!(du, u1, semi, t + dt)
    @. u2 = 3 / 4 * u + 1 / 4 * (u1 + dt * du)
    rhs!(du, u2, semi, t + dt / 2)
    @. u = 1 / 3 * u + 2 / 3 * (u2 + dt * du)
    return u
end

function ssprk3_step!(u, semi::DGSemidiscretization, t, dt::AbstractVector, cache::SSPRK3Cache)
    rhs!(cache.du, u, semi, t)
    return _local_step!(u, semi, t, dt, cache)
end

# The local-step SSP-RK3 step, with `cache.du` already holding L(u).
function _local_step!(u, semi, t, dt, cache)
    u1, u2, du = cache.u1, cache.u2, cache.du
    _stage!(u1, 0, u, 1, u, dt, du)
    rhs!(du, u1, semi, t)
    _stage!(u2, 3 / 4, u, 1 / 4, u1, dt, du)
    rhs!(du, u2, semi, t)
    _stage!(u, 1 / 3, u, 2 / 3, u2, dt, du)
    return u
end

# out = a x + b (y + dt_e du), element by element. `out` may be `x`.
function _stage!(out, a, x, b, y, dt, du)
    @inbounds for e in axes(out, 3)
        h = dt[e]
        for i in axes(out, 2), k in axes(out, 1)
            out[k, i, e] = a * x[k, i, e] + b * (y[k, i, e] + h * du[k, i, e])
        end
    end
    return out
end

"""
    integrate!(u, semi, (t0, t1); cfl=2, maxsteps=10^7, callback=nothing) -> (; t, steps)

March `u` from `t0` to exactly `t1` with SSP-RK3. The step is [`max_dt`](@ref) at `cfl`,
recomputed every step and shortened to land on `t1`. After each step the state is checked for
positivity, and `callback(u, t, step)` is called if given. Throws if `maxsteps` is not enough.

The default `cfl` of 2 is calibrated on full cells. On the isentropic vortex over one period, the
limit is about 4 on square cells and about 3 at aspect ratio 2.
"""
function integrate!(u, semi::DGSemidiscretization, tspan; cfl=2, maxsteps::Integer=10^7,
                    callback=nothing)
    t, t1 = tspan
    cache = SSPRK3Cache(u)
    steps = 0
    while t < t1
        steps < maxsteps || error("integrate! reached $maxsteps steps at t = $t, before $t1")
        dt = max_dt(u, semi; cfl)
        last = t + dt >= t1
        last && (dt = t1 - t)
        ssprk3_step!(u, semi, t, dt, cache)
        t = last ? t1 : t + dt
        steps += 1
        assert_positive(u, semi)
        callback === nothing || callback(u, t, steps)
    end
    return (; t, steps)
end

"""
    residual_norms(du, semi) -> (; l2, max, element)

The plan's convergence monitors, from the time derivative `du`: the cell-average density
residual of each element, `|K| du`, integrated over the element so it is the net mass flux
imbalance and independent of the element's size; its L2 norm over the elements; and its largest
value, with the element holding it. A stuck small cell shows up in `max` first.
"""
function residual_norms(du, semi::DGSemidiscretization{T}) where {T}
    sq, big, at = zero(T), zero(T), 0
    @inbounds for e in eachindex(semi.mass)
        r = abs(semi.mass[e] * du[1, 1, e])
        sq += r^2
        (at == 0 || r > big) && ((big, at) = (r, e))
    end
    return (; l2=sqrt(sq), max=big, element=at)
end

"""
    solve_steady!(u, semi; cfl=2, cfl_start=0.1, ramp=200, tol=1e-10, maxiter=100_000,
                  t=0, callback=nothing) -> (; converged, iterations, residual, history)

March `u` to a steady state in pseudo-time with local steps and SSP-RK3. The CFL number ramps
linearly from `cfl_start` to `cfl` over the first `ramp` iterations. Each element's step is
frozen at the start of an iteration and kept across the stages. The state is checked for
positivity after every iteration.

The run stops when the L2 norm of the cell-average density residual ([`residual_norms`](@ref))
has fallen by `tol` from its first value, or after `maxiter` iterations. `history` holds that
relative residual at the start of each iteration, and `callback(u, iteration, norms)` is called
with the norms each iteration if given. A zero first residual, as for a state that is already
steady to the last bit, counts as converged at once.
"""
function solve_steady!(u, semi::DGSemidiscretization{T}; cfl=2, cfl_start=0.1, ramp::Integer=200,
                       tol=1e-10, maxiter::Integer=100_000, t=zero(T),
                       callback=nothing) where {T}
    cache = SSPRK3Cache(u)
    dt = zeros(T, nelements(semi.mesh))
    history = T[]
    r0 = zero(T)
    for it in 1:maxiter
        rhs!(cache.du, u, semi, t)
        norms = residual_norms(cache.du, semi)
        it == 1 && (r0 = norms.l2)
        rel = r0 > 0 ? norms.l2 / r0 : zero(T)
        push!(history, rel)
        callback === nothing || callback(u, it, norms)
        isfinite(rel) || error("solve_steady!: the residual is $rel at iteration $it")
        rel <= tol && return (; converged=true, iterations=it - 1, residual=rel, history)
        c = cfl_start + (cfl - cfl_start) * min(1, (it - 1) / max(ramp, 1))
        local_dt!(dt, u, semi; cfl=c)
        _local_step!(u, semi, t, dt, cache)
        assert_positive(u, semi)
    end
    return (; converged=false, iterations=maxiter, residual=last(history), history)
end

end # module PseudoTime
