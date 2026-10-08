# =====================================
# PseudoTime: time marching
#
# Downstream of the mesh contract. Phase 1 marches in physical time with one global step, for the
# unsteady vortex test. Phase 3 adds local steps, each element marching with its own stable step,
# which acts as a diagonal preconditioner, so a sliver's tiny step never limits its neighbours.

"""
    EulerDG.PseudoTime

Time marching.

- Phase 1: [`ssprk3_step!`](@ref), the three-stage strong-stability-preserving Runge-Kutta
  method of Shu and Osher, and [`integrate!`](@ref), which marches to an end time with a global
  step and a positivity check after every step.
- Phase 3: local time steps, frozen at the start of each iteration and kept across the stages,
  folded into the right-hand side; a CFL ramp.
- Phase 4: convergence monitors -- the L2 norm of the cell-average density residual, and the
  maximum residual with its element.
"""
module PseudoTime

using ..Semidiscretization: DGSemidiscretization, assert_positive, max_dt, rhs!

export SSPRK3Cache, ssprk3_step!, integrate!

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

Advance `u` from `t` to `t + dt` with SSP-RK3:

    u1 = u + dt L(u),   u2 = 3/4 u + 1/4 (u1 + dt L(u1)),   u = 1/3 u + 2/3 (u2 + dt L(u2))
"""
function ssprk3_step!(u, semi::DGSemidiscretization, t, dt, cache::SSPRK3Cache)
    u1, u2, du = cache.u1, cache.u2, cache.du
    rhs!(du, u, semi, t)
    @. u1 = u + dt * du
    rhs!(du, u1, semi, t + dt)
    @. u2 = 3 / 4 * u + 1 / 4 * (u1 + dt * du)
    rhs!(du, u2, semi, t + dt / 2)
    @. u = 1 / 3 * u + 2 / 3 * (u2 + dt * du)
    return u
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

end # module PseudoTime
