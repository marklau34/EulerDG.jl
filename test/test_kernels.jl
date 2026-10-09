# =====================================
# The threaded kernels
#
# 1. Every kernel body is allocation-free: run serially over all its items, with and without
#    source terms, on uncut, cut and half-plane meshes.
# 2. A launch allocates only KernelAbstractions' own bookkeeping, the same on a small mesh as on a
#    large one.
# 3. The results do not depend on how the work is split: launches split as finely as the threads
#    allow agree bitwise with serial ones. This needs more than one thread (`julia -t`) to test
#    anything; with one thread both runs are serial.

const SD = EulerDG.Semidiscretization
const PT = EulerDG.PseudoTime

# Every kernel body, run serially over all its items.
function kernel_bodies!(du, u, semi, t, dt, cache)
    for f in 1:nfaces(semi.mesh)
        semi.fhat[f] = SD._face_flux(u, semi, t, f)
        semi.fspeed[f] = SD._face_speed(u, semi, f)
    end
    for e in 1:nelements(semi.mesh)
        SD._element_rhs!(du, u, semi, t, e)
        SD._element_rhs!(du, u, semi, t, e, Val(true))
        SD._element_speed!(dt, semi, 2.0, e)
        semi.positivity[e] = SD._first_nonpositive(u, semi, e)
        PT._stage_element!(cache.u1, 3 / 4, u, 1 / 4, u, dt, du, e)
        PT._stage_element!(cache.u2, 3 / 4, u, 1 / 4, u, 0.1, du, e)
    end
    return nothing
end

# The circle at Mach 0.3 on an n x n grid over [-3, 3]^2, with HLLC, the mirror wall and the
# farfield, from the freestream with its gradient modes perturbed so that every term is exercised.
function kernel_circle(n; kwargs...)
    eq = CompressibleEulerEquations2D(1.4)
    uinf = freestream(eq; mach=0.3)
    g = CartesianGrid(Float64; mincorner=(-3.0, -3.0), maxcorner=(3.0, 3.0), n=(n, n))
    m = cut_mesh(g, polygon_body(circle_points(; radius=0.5, h=minimum(g.d)); name="circle")).mesh
    bcs = wall_conditions(m, boundary_condition_mirror_wall, BoundaryConditionFarfield(uinf))
    semi = DGSemidiscretization(m, eq; surface_flux=flux_hllc, volume_degree=4,
                                boundary_conditions=bcs, kwargs...)
    return semi, perturbed(project((x, t, eq) -> uinf, semi))
end

# The wall MMS on an n x n half-plane mesh: source terms and Dirichlet sides.
function kernel_mms(n; kwargs...)
    eq = CompressibleEulerEquations2D(1.4)
    exact = WallMMS((0.0, 0.3), deg2rad(17))
    g = CartesianGrid(Float64; mincorner=(0.0, 0.0), maxcorner=(1.0, 1.0), n=(n, n))
    m = halfplane_mesh(g; point=exact.p0, normal=exact.n)
    bcs = wall_conditions(m, boundary_condition_mirror_wall, BoundaryConditionDirichlet(exact))
    semi = DGSemidiscretization(m, eq; surface_flux=flux_hllc, source_terms=MMSSource(exact),
                                boundary_conditions=bcs, kwargs...)
    return semi, perturbed(project(exact, semi))
end

# The isentropic vortex on a periodic n x n grid, with Rusanov.
function kernel_vortex(n; kwargs...)
    g = CartesianGrid(Float64; mincorner=(-5, -5), maxcorner=(5, 5), n=(n, n))
    semi = DGSemidiscretization(uncut_mesh(g; periodic=(true, true)),
                                CompressibleEulerEquations2D(1.4); kwargs...)
    return semi, perturbed(project(IsentropicVortex(), semi))
end

function perturbed(u)
    ne = size(u, 3)
    u[:, 2:3, :] .+= 0.01 .* sin.(reshape(1:(8 * ne), 4, 2, ne))
    return u
end

# The bytes each solver call allocates.
function launch_allocs(semi, u)
    du, dt, cache = similar(u), zeros(size(u, 3)), SSPRK3Cache(u)
    local_dt!(dt, u, semi; cfl=0.1)
    v = copy(u)
    return (; rhs=allocs(rhs!, du, u, semi, 0.0),
            local_dt=allocs((dt, u, semi) -> local_dt!(dt, u, semi; cfl=0.1), dt, u, semi),
            max_dt=allocs((u, semi) -> max_dt(u, semi; cfl=0.1), u, semi),
            positive=allocs(assert_positive, u, semi),
            local_step=allocs(ssprk3_step!, v, semi, 0.0, dt, cache),
            global_step=allocs(ssprk3_step!, v, semi, 0.0, 1e-4, cache),
            norms=allocs(residual_norms, du, semi))
end

# Everything the solver computes from `u`: a right-hand side, the steps, ten local-step and ten
# global-step iterations, and the positivity error on a spoiled state.
function kernel_results(semi, u)
    du = rhs!(similar(u), u, semi, 0.0)
    dt = local_dt!(zeros(size(u, 3)), u, semi; cfl=0.5)
    out = (; du, dt, max_dt=max_dt(u, semi; cfl=0.5), norms=residual_norms(du, semi))
    v, cache = copy(u), SSPRK3Cache(u)
    for _ in 1:10
        local_dt!(dt, v, semi; cfl=0.5)
        ssprk3_step!(v, semi, 0.0, dt, cache)
    end
    w = copy(u)
    for k in 1:10
        ssprk3_step!(w, semi, 0.01k, 0.5max_dt(w, semi; cfl=1), cache)
    end
    bad = copy(u)
    bad[1, 1, 3size(u, 3) ÷ 4] = -1
    bad[4, 1, size(u, 3) ÷ 2] = 0
    error = try
        assert_positive(bad, semi)
        nothing
    catch err
        sprint(showerror, err)
    end
    return (; out..., local_steps=v, global_steps=w, error)
end

@testset "kernels" begin
    cases = (; circle=kernel_circle, mms=kernel_mms, vortex=kernel_vortex)

    @testset "allocation-free bodies on the $name" for (name, case) in pairs(cases)
        semi, u = case(12)
        du, dt, cache = similar(u), zeros(size(u, 3)), SSPRK3Cache(u)
        @test allocs(kernel_bodies!, du, u, semi, 0.0, dt, cache) == 0
    end

    # Julia caches the boxes of integers below 512, so a launch's bookkeeping is constant only from
    # 512 items up: both meshes have more elements than that.
    @testset "launch allocations on the $name do not grow with the mesh" for (name, case) in pairs(cases)
        semi, u = case(32; grainsize=typemax(Int))
        @test nelements(semi.mesh) > 512
        small = launch_allocs(semi, u)
        large = launch_allocs(case(64; grainsize=typemax(Int))...)
        @test small == large
        @test small.norms == 0
    end

    @testset "the same results however the work is split, on the $name" for (name, case) in pairs(cases)
        serial = kernel_results(case(24; grainsize=typemax(Int))...)
        split = kernel_results(case(24; grainsize=1)...)
        @test serial.error !== nothing && occursin("DomainError", serial.error)
        for k in keys(serial)
            @test isequal(serial[k], split[k])
        end
    end

    @testset "the steady solver's fused pass on the $name" for (name, case) in pairs(cases)
        # One pass gives bitwise what rhs!, local_dt! and assert_positive give separately.
        semi, u = case(12)
        du, du2 = similar(u), similar(u)
        dt, dt2 = zeros(size(u, 3)), zeros(size(u, 3))
        rhs!(du, u, semi, 0.0)
        local_dt!(dt, u, semi; cfl=1.5)
        SD._rhs_steady!(du2, u, semi, 0.0)
        SD._steps!(dt2, semi; cfl=1.5)
        @test du2 == du && dt2 == dt
        @test all(iszero, semi.positivity) && SD._check_positivity(u, semi) === nothing
        # The flags come from the element pass; the face pass of a negative state may fail first,
        # at a square root, which the next testset covers.
        bad = copy(u)
        bad[4, 1, size(u, 3) ÷ 2] = 0                     # no energy: negative pressure
        SD._launch!(SD._element_rhs_check_kernel!, semi, size(u, 3), du2, bad, semi, 0.0)
        flagged = try SD._check_positivity(bad, semi); "" catch err; sprint(showerror, err) end
        direct = try assert_positive(bad, semi); "" catch err; sprint(showerror, err) end
        @test !isempty(flagged) && flagged == direct
    end

    @testset "a negative state in the steady solver is named" begin
        # Its fluxes may fail first, at a square root; the error still names the element.
        semi, u = kernel_circle(12)
        u[4, 1, 100] = 0
        err = try solve_steady!(u, semi; maxiter=5); nothing catch e; e end
        @test err isa DomainError && occursin("element 100", sprint(showerror, err))
    end

    @test throws_with(() -> kernel_vortex(4; grainsize=0), "grainsize must be at least 1")
end
