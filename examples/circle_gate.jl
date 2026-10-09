# =====================================
# The Phase 4 circle gate at full resolution
#
# Run with `julia --project=examples -t auto examples/circle_gate.jl [sets...]`, with sets from A,
# B, C (default: all three, then the summary) or `summary` alone. Each run is saved to
# out/circle_gate/ as it finishes, with its state, metrics, run log and, on unshifted grids, a VTK
# file; a rerun skips the runs already saved, so the study can be stopped and resumed.
#
# The circle of diameter 1 at Mach 0.3, with HLLC, the mirror wall, the characteristic farfield and
# the degree-4 volume rules, marched with local steps from the freestream. The far boundary is the
# square [-L, L]^2, so it lies 2L radii from the centre.
#
# - A. Domain study at 8 cells per diameter, the far boundary at 6, 8, 12 and 24 radii: the
#   stagnation Cp against L, extrapolated to an unbounded domain as Cp∞ + c/L^2.
# - B. Order study at 6 radii, at 8, 16, 32 and 64 cells per diameter: drag, entropy, wake and
#   asymmetry against h.
# - C. Grid-shift sweep at 8, 16 and 32 cells per diameter: four sub-cell shifts and three sliver
#   shifts (a grid node 1e-3, 1e-6 and 1e-8 h outside the circle).
#
# The gate, as decided on Oct 9:
#
# 1. Every run converges 10 orders.
# 2. On the symmetric placement, lift and the top-bottom Cp asymmetry are zero to round-off (1e-6,
#    allowing for the triangulation of mirrored fragments). On shifted grids both fall with h, and
#    |cl| is below 1e-3 on the finest sweep grid.
# 3. The fore-aft Cp asymmetry falls with h.
# 4. The stagnation Cp, extrapolated to an unbounded domain, is within 0.5% of the isentropic value.
# 5. Spurious drag and the entropy error's L2 norm fall at close to second order.
# 6. The wake entropy falls at close to second order and is below 1e-4 on the finest grid.

using EulerDG
using CartesianMeshes: CartesianMeshes, CartesianGrid
using Printf: @printf, @sprintf
using Serialization: deserialize, serialize
using StaticArrays: SMatrix, SVector

const OUT = mkpath(normpath(joinpath(@__DIR__, "..", "out", "circle_gate")))
const EQ = CompressibleEulerEquations2D(1.4)
const UINF = freestream(EQ; mach=0.3)
const CP0 = (2 / (1.4 * 0.3^2)) * ((1 + 0.2 * 0.3^2)^3.5 - 1)
const PV = SVector{2,Float64}

# =====================================
# The runs

const SHIFTS = [(; tag="shift$k", shift=Tuple(s), gap=nothing) for (k, s) in enumerate(grid_shifts(5)[2:end])]
const GAPS = [(; tag=@sprintf("gap%.0e", g), shift=(0.0, 0.0), gap=g) for g in (1e-3, 1e-6, 1e-8)]
const SYMMETRIC = (; tag="sym", shift=(0.0, 0.0), gap=nothing)

run_key(cpd, L, p) = "d$(cpd)_L$(L)_$(p.tag)"

function runs(set)
    set == "A" && return [(8, L, SYMMETRIC) for L in (3, 4, 6, 12)]
    set == "B" && return [(cpd, 3, SYMMETRIC) for cpd in (8, 16, 32, 64)]
    set == "C" && return [(cpd, 3, p) for cpd in (8, 16, 32) for p in vcat(SHIFTS, GAPS)]
    error("unknown set $set; use A, B, C or summary")
end

# The mesh and discretization, and the circle's centre.
function setup(cpd, L, p)
    n = round(Int, 2L * cpd)
    g = CartesianGrid(Float64; mincorner=(-L, -L), maxcorner=(L, L), n=(n, n))
    pts = circle_points(; radius=0.5, h=minimum(g.d))
    c = p.gap === nothing ? PV(p.shift) .* PV(g.d) :
        sliver_shift(polygon_body(pts), [PV(CartesianMeshes.get_node(g, I)) for I in CartesianIndices(g.n .+ 1)];
                     gap=p.gap * minimum(g.d))
    cut = cut_mesh(g, polygon_body(pts .+ Ref(c); name="circle"))
    m = cut.mesh
    bcs = NamedTuple{Tuple(m.boundary_names)}(Tuple(b == :wall ? boundary_condition_mirror_wall :
                                                    BoundaryConditionFarfield(UINF)
                                                    for b in m.boundary_names))
    semi = DGSemidiscretization(m, EQ; surface_flux=flux_hllc, boundary_conditions=bcs,
                                volume_degree=4)
    return (; semi, center=c, minvf=cut.report.min_volume_fraction)
end

# The run log, and a line on stdout every `every` iterations.
struct Progress{R}
    log::R
    key::String
    every::Int
    t0::Float64
end

function (p::Progress)(u, it, norms)
    p.log(u, it, norms)
    if it % p.every == 0
        @printf("    %s: iteration %d, residual %.2e, %.0f s\n", p.key, it,
                norms.l2 / p.log.r0, time() - p.t0)
        flush(stdout)
    end
    return nothing
end

function run_case(cpd, L, p)
    key = run_key(cpd, L, p)
    file = joinpath(OUT, key * ".jls")
    isfile(file) && return deserialize(file).record
    s = setup(cpd, L, p)
    semi = s.semi
    @printf("  %s: %d elements, smallest volume fraction %.1e\n", key, nelements(semi.mesh), s.minvf)
    flush(stdout)
    u = project((x, t, eq) -> UINF, semi)
    log = RunLog(joinpath(OUT, key * "_run.csv"), semi, UINF; every=100, point=Tuple(s.center))
    seconds = @elapsed r = solve_steady!(u, semi; tol=1e-10, maxiter=400_000,
                                         callback=Progress(log, key, 10_000, time()))
    close(log)
    record = (; key, cpd, L, radii=2L, tag=p.tag, elements=nelements(semi.mesh), minvf=s.minvf,
              converged=r.converged, iterations=r.iterations, residual=r.residual, seconds,
              metrics(semi, u, s.center)...)
    serialize(file, (; record, u))
    p.tag == "sym" && write_solution_vtk(joinpath(OUT, key), semi, u; freestream=UINF)
    @printf("  %s: %s after %d iterations, %.0f s; cl %+.2e, cd %.3e, stagnation Cp %.4f\n", key,
            r.converged ? "converged" : "NOT CONVERGED", r.iterations, seconds, record.cl, record.cd,
            record.cp_front)
    flush(stdout)
    return record
end

# =====================================
# Metrics

# The point where the ray from `c` at angle θ meets the wall, and the wall face it lies on.
function wall_point(semi, c, θ)
    m, d = semi.mesh, PV(cos(θ), sin(θ))
    for f in boundary_faces(m, :wall)
        a, b = m.points[m.faces.nodes[f][1]], m.points[m.faces.nodes[f][2]]
        e = b - a
        det = d[2] * e[1] - d[1] * e[2]
        abs(det) < eps() && continue
        w = a - c
        s = (w[2] * e[1] - w[1] * e[2]) / det          # c + s d = a + t e
        t = (d[1] * w[2] - d[2] * w[1]) / det
        s > 0 && -1e-12 <= t <= 1 + 1e-12 && return (a + clamp(t, 0, 1) * e, f)
    end
    error("no wall face at angle $θ")
end

# Cp on the wall in the direction θ from the centre, from the interior trace.
function wall_cp(semi, u, c, θ)
    x, f = wall_point(semi, c, θ)
    v = velocity(UINF, EQ)
    U = evaluate(semi, u, semi.mesh.faces.left[f], x)
    return (pressure(U, EQ) - pressure(UINF, EQ)) / (UINF[1] * (v' * v) / 2)
end

# The largest Cp difference between each wall Gauss point and its mirror image on the wall, across
# the horizontal (top-bottom) or vertical (fore-aft) axis through the centre.
function cp_asymmetry(semi, u, c, cp, axis)
    worst = 0.0
    for (x, v) in zip(cp.x, cp.cp)
        r = x - c
        θ = axis === :horizontal ? atan(-r[2], r[1]) : atan(r[2], -r[1])
        worst = max(worst, abs(v - wall_cp(semi, u, c, θ)))
    end
    return worst
end

# The mean entropy error, in cell averages, one to three diameters behind the centre and within
# half a diameter of its axis.
function wake_entropy(semi, u, c)
    s∞ = pressure(UINF, EQ) / UINF[1]^EQ.gamma
    total, count = 0.0, 0
    for e in 1:nelements(semi.mesh)
        x = semi.mesh.elements.centroid[e] - c
        (1 < x[1] < 3 && abs(x[2]) < 0.5) || continue
        ρ, _, _, p = cons2prim(cell_average(u, e), EQ)
        total += abs((p / ρ^EQ.gamma - s∞) / s∞)
        count += 1
    end
    return total / count
end

function metrics(semi, u, c)
    f = wall_forces(semi, u, UINF; length=1, point=Tuple(c))
    cp = surface_cp(semi, u, UINF)
    s = entropy_error(semi, u, UINF)
    return (; cl=f.cl, cd=f.cd, cp_front=wall_cp(semi, u, c, π), cp_rear=wall_cp(semi, u, c, 0.0),
            cp_max=maximum(cp.cp), asym_tb=cp_asymmetry(semi, u, c, cp, :horizontal),
            asym_fa=cp_asymmetry(semi, u, c, cp, :vertical), entropy_l2=s.l2, entropy_max=s.max,
            wake=wake_entropy(semi, u, c))
end

# =====================================
# The summary

order(a, b, ratio=2) = log(a / b) / log(ratio)

function summary()
    recs = [deserialize(joinpath(OUT, f)).record for f in readdir(OUT) if endswith(f, ".jls")]
    isempty(recs) && return println("No runs saved yet.")
    find(cpd, L, tag) = (i = findfirst(r -> r.cpd == cpd && r.L == L && r.tag == tag, recs);
                         i === nothing ? nothing : recs[i])

    open(joinpath(OUT, "runs.csv"), "w") do io
        names = keys(first(recs))
        println(io, join(names, ","))
        for r in sort(recs; by=r -> (r.cpd, r.L, r.tag))
            println(io, join((v isa AbstractFloat ? @sprintf("%.10e", v) : string(v) for v in r), ","))
        end
    end

    println("\nEvery run (", length(recs), "):")
    @printf("  %-18s %8s %10s %8s %7s %6s %10s %9s %8s %8s %8s %9s %9s\n", "run", "elements",
            "min vf", "iters", "conv", "s", "cl", "cd", "Cp front", "asym tb", "asym fa",
            "entropy", "wake")
    for r in sort(recs; by=r -> (r.cpd, r.L, r.tag))
        @printf("  %-18s %8d %10.1e %8d %7s %6.0f %+10.2e %9.3e %8.4f %8.1e %8.1e %9.3e %9.3e\n",
                r.key, r.elements, r.minvf, r.iterations, r.converged, r.seconds, r.cl, r.cd,
                r.cp_front, r.asym_tb, r.asym_fa, r.entropy_l2, r.wake)
    end

    A = filter(!isnothing, [find(8, L, "sym") for L in (3, 4, 6, 12)])
    B = filter(!isnothing, [find(cpd, 3, "sym") for cpd in (8, 16, 32, 64)])
    if length(A) == 4
        c6, c12, c24 = A[1].cp_front, A[3].cp_front, A[4].cp_front
        p = order(c6 - c12, c12 - c24)
        extrap = c24 - (c12 - c24) / (2^p - 1)
        extrap2 = c24 - (c12 - c24) / 3                    # assuming exactly L^-2
        println("\nA. Stagnation Cp against the far boundary, 8 cells per diameter:")
        for r in A
            @printf("  %2d radii: %.5f (excess %.5f)\n", r.radii, r.cp_front, r.cp_front - CP0)
        end
        @printf("  observed order in 1/L: %.2f; extrapolated %.5f (with order 2: %.5f) against %.5f: %+.2f%%\n",
                p, extrap, extrap2, CP0, 100 * (extrap - CP0) / CP0)
        # The grid's own share, from the order study at 6 radii: the change from 8 cells per
        # diameter to the finest grid there, added to the domain-extrapolated value.
        if length(B) >= 2
            fine = last(B)
            shift = fine.cp_front - first(B).cp_front
            @printf("  with the grid's share (%+.5f, 8 to %d cells/D): %.5f, %+.2f%% (gate: within 0.5%%)\n",
                    shift, fine.cpd, extrap + shift, 100 * (extrap + shift - CP0) / CP0)
        end
    end

    if length(B) >= 2
        println("\nB. Order study at 6 radii:")
        for (k, r) in enumerate(B)
            @printf("  %2d cells/D: cl %+.1e, asym tb %.1e, asym fa %.2e, cd %.3e, entropy %.3e, wake %.3e, Cp front %.4f",
                    r.cpd, r.cl, r.asym_tb, r.asym_fa, r.cd, r.entropy_l2, r.wake, r.cp_front)
            if k > 1
                q = B[k - 1]
                @printf("; orders: cd %.2f, entropy %.2f, wake %.2f, asym fa %.2f", order(q.cd, r.cd),
                        order(q.entropy_l2, r.entropy_l2), order(q.wake, r.wake),
                        order(q.asym_fa, r.asym_fa))
            end
            println()
        end
    end

    println("\nC. Grid-shift sweep at 6 radii:")
    for cpd in (8, 16, 32)
        C = filter(!isnothing, [find(cpd, 3, p.tag) for p in vcat(SHIFTS, GAPS)])
        isempty(C) && continue
        @printf("  %2d cells/D (%d runs): largest |cl| %.2e, asym tb %.2e, asym fa %.2e; cd %.3e to %.3e; all converged: %s\n",
                cpd, length(C), maximum(r -> abs(r.cl), C), maximum(r -> r.asym_tb, C),
                maximum(r -> r.asym_fa, C), minimum(r -> r.cd, C), maximum(r -> r.cd, C),
                all(r -> r.converged, C))
    end
end

# =====================================

sets = isempty(ARGS) ? ["A", "B", "C"] : uppercase.(ARGS)
@printf("%d threads; output in %s\n", Threads.nthreads(), OUT)
for set in filter(!=("SUMMARY"), sets)
    println("Set ", set, ":")
    # Within a set, the 64-cell grid last: it is the longest run by far.
    for (cpd, L, p) in sort(runs(set); by=r -> r[1] == 64)
        run_case(cpd, L, p)
    end
end
summary()
