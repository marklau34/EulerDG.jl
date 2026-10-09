# =====================================
# Exact flows for the tests
#
# A manufactured solution on a domain cut by a straight wall, with zero normal velocity on the
# wall so the slip walls are consistent with it, and its source term by complex step -- exact to
# round-off for an analytic solution, with no automatic-differentiation dependency.

"""
    WallMMS(p0, t, n)

A smooth steady state on the side of the wall line through `p0` (tangent `t`, unit normal `n`
into the fluid). In the wall's frame, with `d` the distance from the wall and `s` the position
along it:

    ρ = 1 + 0.1 sin(2π(s + 0.1)) cos(2πd),   u_t = 0.4 + 0.1 cos(2πs) sin(2πd),
    u_n = 0.3 d sin(2πs),                    p = 1/γ + 0.05 cos(2π(s - d))

The normal velocity vanishes on the wall, and the flow is subsonic everywhere.
"""
struct WallMMS
    p0::PV
    t::PV
    n::PV
end

WallMMS(p0, θ) = WallMMS(PV(p0), PV(cos(θ), sin(θ)), PV(-sin(θ), cos(θ)))

function (w::WallMMS)(x, t, eq)
    d = (x[1] - w.p0[1]) * w.n[1] + (x[2] - w.p0[2]) * w.n[2]
    s = (x[1] - w.p0[1]) * w.t[1] + (x[2] - w.p0[2]) * w.t[2]
    ρ = 1 + 0.1 * sin(2π * (s + 0.1)) * cos(2π * d)
    ut = 0.4 + 0.1 * cos(2π * s) * sin(2π * d)
    un = 0.3 * d * sin(2π * s)
    p = 1 / eq.gamma + 0.05 * cos(2π * (s - d))
    return prim2cons(SVector(ρ, ut * w.t[1] + un * w.n[1], ut * w.t[2] + un * w.n[2], p), eq)
end

"""
    MMSSource(exact)

The source term that makes `exact(x, t, equations)` a steady state, `∇ · F(exact)`, by complex
step, as a Trixi-style `source_terms(u, x, t, equations)`.
"""
struct MMSSource{E}
    exact::E
end

function (s::MMSSource)(u, x, t, eq)
    hc = 1e-30
    fx = imag.(flux(s.exact(SVector(complex(x[1], hc), complex(x[2])), t, eq), 1, eq)) / hc
    fy = imag.(flux(s.exact(SVector(complex(x[1]), complex(x[2], hc)), t, eq), 2, eq)) / hc
    return SVector{4,Float64}(fx + fy)
end

# Boundary conditions for a half-plane mesh: `wall` on the wall, `other` on the box's sides.
function wall_conditions(m, wall, other)
    return NamedTuple{Tuple(m.boundary_names)}(Tuple(n == :wall ? wall : other
                                                     for n in m.boundary_names))
end

# The steady manufactured solution on an n x n half-plane mesh over the unit square: the mesh, the
# discretization and the converged state.
function solve_mms(nc; θ=deg2rad(17), p0=(0.0, 0.3), wall=boundary_condition_mirror_wall,
                   surface_flux=flux_hllc, tol=1e-9, start=:exact)
    eq = CompressibleEulerEquations2D(1.4)
    exact = WallMMS(p0, θ)
    g = CartesianGrid(Float64; mincorner=(0.0, 0.0), maxcorner=(1.0, 1.0), n=(nc, nc))
    m = halfplane_mesh(g; point=exact.p0, normal=exact.n)
    semi = DGSemidiscretization(m, eq; surface_flux, source_terms=MMSSource(exact),
                                boundary_conditions=wall_conditions(m, wall,
                                                                    BoundaryConditionDirichlet(exact)))
    u = start === :exact ? project(exact, semi) :
        project((x, t, eq) -> prim2cons(SVector(1.0, 0.4exact.t[1], 0.4exact.t[2], 1 / 1.4), eq), semi)
    r = solve_steady!(u, semi; tol, maxiter=200_000)
    return (; mesh=m, semi, u, run=r, error=l2_error(semi, u, exact).l2[1])
end

# Uniform flow along the wall of a half-plane mesh, marched with local steps: the largest
# deviation from it, on elements of volume fraction at least 1e-6 and on any element.
function wall_aligned_deviation(m, θ; wall, surface_flux, iterations=300)
    eq = CompressibleEulerEquations2D(1.4)
    uinf = prim2cons(SVector(1.0, 0.5cos(θ), 0.5sin(θ), 1 / 1.4), eq)
    semi = DGSemidiscretization(m, eq; surface_flux,
                                boundary_conditions=wall_conditions(m, wall,
                                                                    BoundaryConditionDirichlet((x, t, eq) -> uinf)))
    u = allocate_state(semi)
    u[:, 1, :] .= uinf
    dt = zeros(nelements(m))
    cache = SSPRK3Cache(u)
    for _ in 1:iterations
        local_dt!(dt, u, semi; cfl=2)
        ssprk3_step!(u, semi, 0.0, dt, cache)
    end
    dev = [max(maximum(abs, u[:, 1, e] .- uinf), maximum(abs, u[:, 2:3, e])) for e in 1:nelements(m)]
    # Round-off on a face's direction grows as the face shrinks: bound each element's deviation by
    # round-off times h over its shortest face.
    h = maximum(sqrt, m.elements.parent_area)
    scaled = maximum(dev[e] / max(1, h / minimum(m.faces.length[f] for f in element_faces(m, e)))
                     for e in 1:nelements(m))
    big = m.elements.volume_fraction .>= 1e-6
    return (; regular=maximum(dev[big]), scaled, all=maximum(dev))
end

# The inviscid circle of diameter 1 at Mach 0.3, centred on a grid line of a uniform grid over
# [-L, L]^2 with `cpd` cells per diameter, moved by `shift` (in cells): HLLC, the mirror wall and
# the characteristic farfield, marched to a steady state from the freestream.
function solve_circle(cpd; L=3.0, shift=(0.0, 0.0), gap=nothing, volume_degree=4, tol=1e-10)
    eq = CompressibleEulerEquations2D(1.4)
    uinf = freestream(eq; mach=0.3)
    g = CartesianGrid(Float64; mincorner=(-L, -L), maxcorner=(L, L), n=(round(Int, 2L * cpd), round(Int, 2L * cpd)))
    pts = circle_points(; radius=0.5, h=minimum(g.d))
    s = gap === nothing ? SVector(shift) .* g.d :
        sliver_shift(polygon_body(pts), grid_nodes(g); gap=gap * minimum(g.d))
    c = cut_mesh(g, polygon_body(pts .+ Ref(s); name="circle"))
    m = c.mesh
    semi = DGSemidiscretization(m, eq; surface_flux=flux_hllc, volume_degree,
                                boundary_conditions=wall_conditions(m, boundary_condition_mirror_wall,
                                                                    BoundaryConditionFarfield(uinf)))
    u = project((x, t, eq) -> uinf, semi)
    t = @elapsed r = solve_steady!(u, semi; tol, maxiter=400_000)
    return (; mesh=m, semi, u, uinf, run=r, seconds=t, center=s, minvf=c.report.min_volume_fraction,
            forces=wall_forces(semi, u, uinf; length=1, point=s), cp=surface_cp(semi, u, uinf),
            entropy=entropy_error(semi, u, uinf))
end

# The largest difference in Cp between wall Gauss points mirrored across the circle's horizontal
# axis, over the points whose mirror image is also a Gauss point.
function cp_mirror_asymmetry(c)
    x, cp = c.cp.x, c.cp.cp
    yc = c.center[2]
    index = Dict(round.((p[1], p[2] - yc); digits=10) => k for (k, p) in enumerate(x))
    worst = 0.0
    for (k, p) in enumerate(x)
        j = get(index, round.((p[1], yc - p[2]); digits=10), 0)
        j == 0 || (worst = max(worst, abs(cp[k] - cp[j])))
    end
    return worst
end

# The mean entropy error, in cell averages, over the wake: one to three diameters behind the
# circle's centre, within half a diameter of its axis.
function wake_entropy(c)
    eq = c.semi.equations
    s∞ = pressure(c.uinf, eq) / c.uinf[1]^eq.gamma
    E = [e for e in 1:nelements(c.mesh)
         if 1 < c.mesh.elements.centroid[e][1] - c.center[1] < 3 &&
            abs(c.mesh.elements.centroid[e][2] - c.center[2]) < 0.5]
    return sum(abs(begin
                       ρ, _, _, p = cons2prim(cell_average(c.u, e), eq)
                       (p / ρ^eq.gamma - s∞) / s∞
                   end) for e in E) / length(E)
end
