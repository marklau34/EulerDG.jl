# =====================================
# The Phase 5 airfoil study: NACA 0012 at Mach 0.5 and 1.25 degrees, against Vassberg and Jameson
#
# Run with `julia --project=examples -t auto examples/naca0012_study.jl [cells...]`, the cells per
# chord of the grids to run (default 32, 64 and 128), or `summary` alone. Each run is saved to
# out/naca0012_study/ as it finishes -- its state, metrics, run log, trailing-edge report, Cp along
# the chord and a VTK file -- and a rerun skips the grids already saved.
#
# - Geometry: Vassberg and Jameson's NACA 0012 (J. Aircraft 47(4), 2010), the standard thickness
#   with its chord extended to the sharp trailing edge at x = 1.00893, turned nose-up 1.25 degrees
#   about its leading edge; the freestream runs along x. The reference chord is 1 and the moment is
#   about the quarter chord.
# - Mesh: a finest box half a chord round the airfoil, with the trailing edge 0.3 h into its cell
#   and on the middle line of its row on every grid, wrapped in the nested far field out to 100
#   chords or more.
# - Solver: HLLC, the mirror-state slip wall, the characteristic farfield, the degree-4 rules on
#   cut cells, local steps at the default CFL.
# - Reference: their Table 2, FLO82 extrapolated to the continuum: cl 0.180346, cd 5e-8 and
#   cm -0.002269.

using EulerDG
using CartesianMeshes: CartesianGrid
using Printf: @printf, @sprintf
using Serialization: deserialize, serialize
using StaticArrays: SVector

const OUT = mkpath(normpath(joinpath(@__DIR__, "..", "out", "naca0012_study")))
const MACH, ALPHA = 0.5, 1.25
const REF = (; cl=0.180346, cd=5e-8, cm=-0.002269)
const EQ = CompressibleEulerEquations2D(1.4)
const UINF = freestream(EQ; mach=MACH)
const PV = SVector{2,Float64}

# =====================================
# The grid and the discretization

function setup(cpc)
    h = 1 / cpc
    points = naca4_points("0012"; h, alpha_deg=ALPHA, trailing_edge=:extended)
    te = points[1]
    le = points[argmax([sum(abs2, p - te) for p in points])]
    chordline = (te - le) / hypot((te - le)...)
    # The finest box: half a chord ahead of the leading edge, behind the trailing edge, above and
    # below, with the trailing edge 0.3 h into its cell and on the middle line of its row, and an
    # even number of cells each way.
    mx, my = round(Int, (te[1] - le[1] + 0.5) / h), round(Int, 0.5 / h)
    x0, y0 = te[1] - 0.3h - mx * h, te[2] - 0.5h - my * h
    nx, ny = mx + round(Int, 0.5 / h), 2my + 2
    nx += isodd(nx)
    grid = CartesianGrid(Float64; mincorner=(x0, y0), maxcorner=(x0 + nx * h, y0 + ny * h),
                         n=(nx, ny))
    cut = nested_cut_mesh(grid, polygon_body(points; name="naca0012");
                          farfield=((-100.0, -100.0), (100.0, 100.0)))
    m = cut.mesh
    far = BoundaryConditionFarfield(UINF)
    bcs = NamedTuple{Tuple(m.boundary_names)}(Tuple(b == :wall ? boundary_condition_mirror_wall : far
                                                    for b in m.boundary_names))
    semi = DGSemidiscretization(m, EQ; surface_flux=flux_hllc, boundary_conditions=bcs,
                                volume_degree=4)
    return (; semi, cut, grid, points, te, le, chordline, quarter=le + 0.25 * chordline,
            tecell=CartesianIndex(mx + 1, my + 1))
end

# =====================================
# The trailing-edge report

function te_report(io, cpc, s)
    m, r = s.semi.mesh, s.cut.report
    wall = boundary_index(m, :wall)
    lin = LinearIndices(s.grid.n)
    # The angle between the upper and lower surfaces at the edge.
    up, lo = s.points[2] - s.te, s.points[end] - s.te
    wedge = acosd(clamp(up' * lo / (hypot(up...) * hypot(lo...)), -1, 1))
    @printf(io, "Trailing-edge report, %d cells per chord (h = %.5f)\n", cpc, 1 / cpc)
    @printf(io, "  edge at (%.10f, %.10f); a mesh vertex: %s; wedge angle %.3f degrees\n",
            s.te..., any(==(s.te), m.points), wedge)
    @printf(io, "  %d split cells; smallest volume fraction %.3e%s\n", length(r.split_cells),
            r.min_volume_fraction,
            r.min_volume_fraction_cell === nothing ? "" : " in cell $(Tuple(r.min_volume_fraction_cell))")
    @printf(io, "  wall length %.12f against the polygon's %.12f\n", r.wall_length, r.body_perimeter)
    println(io, "  The 3 x 3 cells round the edge's cell $(Tuple(s.tecell)), top row first:")
    for dj in 1:-1:-1, di in -1:1
        ci = s.tecell + CartesianIndex(di, dj)
        els = findall(==(lin[ci]), m.elements.parent)
        isempty(els) && (@printf(io, "    cell %-12s solid\n", string(Tuple(ci))); continue)
        for e in els
            nwall = count(f -> m.faces.boundary[f] == wall, element_faces(m, e))
            @printf(io, "    cell %-12s fragment %d: volume fraction %.4f, %2d vertices, %d wall faces\n",
                    string(Tuple(ci)), m.elements.fragment[e], m.elements.volume_fraction[e],
                    length(element_nodes(m, e)), nwall)
        end
    end
    return nothing
end

# =====================================
# A run

struct Progress{R}
    log::R
    cpc::Int
    t0::Float64
end

function (p::Progress)(u, it, norms)
    p.log(u, it, norms)
    if it % 20_000 == 0
        @printf("    %d cells/chord: iteration %d, residual %.2e, %.0f s\n", p.cpc, it,
                norms.l2 / p.log.r0, time() - p.t0)
        flush(stdout)
    end
    return nothing
end

function run_case(cpc)
    key = "c$(cpc)"
    file = joinpath(OUT, key * ".jls")
    isfile(file) && return deserialize(file).record
    s = setup(cpc)
    semi, m = s.semi, s.semi.mesh
    @printf("  %d cells/chord: %d elements, %d levels, smallest volume fraction %.1e\n", cpc,
            nelements(m), length(s.cut.report.levels), s.cut.report.min_volume_fraction)
    open(io -> te_report(io, cpc, s), joinpath(OUT, key * "_te.txt"), "w")
    te_report(stdout, cpc, s)
    flush(stdout)

    u = project((x, t, eq) -> UINF, semi)
    log = RunLog(joinpath(OUT, key * "_run.csv"), semi, UINF; every=100, point=Tuple(s.quarter))
    seconds = @elapsed r = solve_steady!(u, semi; tol=1e-10, maxiter=1_000_000,
                                         callback=Progress(log, cpc, time()))
    close(log)

    f = wall_forces(semi, u, UINF; length=1, point=Tuple(s.quarter))
    cp = surface_cp(semi, u, UINF)
    se = entropy_error(semi, u, UINF)
    record = (; key, cpc, elements=nelements(m), minvf=s.cut.report.min_volume_fraction,
              converged=r.converged, iterations=r.iterations, residual=r.residual, seconds,
              cl=f.cl, cd=f.cd, cm=f.cm, cm_nose_up=true, cp_max=maximum(cp.cp),
              cp_min=minimum(cp.cp),
              entropy_l2=se.l2, entropy_max=se.max)
    serialize(file, (; record, u))
    open(joinpath(OUT, key * "_cp.csv"), "w") do io
        println(io, "x,y,x_over_c,side,cp")
        for (p, v) in zip(cp.x, cp.cp)
            d = p - s.le
            side = s.chordline[1] * d[2] - s.chordline[2] * d[1] >= 0 ? "upper" : "lower"
            @printf(io, "%.12e,%.12e,%.12e,%s,%.12e\n", p[1], p[2], d' * s.chordline, side, v)
        end
    end
    write_solution_vtk(joinpath(OUT, key * "_solution"), semi, u; freestream=UINF)
    @printf("  %d cells/chord: %s after %d iterations, %.0f s; cl %.5f (%+.2f%%), cd %.2e, cm %+.5f\n",
            cpc, r.converged ? "converged" : "NOT CONVERGED", r.iterations, seconds, f.cl,
            100 * (f.cl - REF.cl) / REF.cl, f.cd, f.cm)
    flush(stdout)
    return record
end

# =====================================
# The summary

# The pitching moment, nose-up positive. Runs saved before Oct 10 stored it counter-clockwise
# positive, which with the flow along x is nose-down: their sign is flipped here.
pitch(r) = haskey(r, :cm_nose_up) ? r.cm : -r.cm

function summary()
    recs = sort([deserialize(joinpath(OUT, f)).record for f in readdir(OUT) if endswith(f, ".jls")];
                by=r -> r.cpc)
    isempty(recs) && return println("No runs saved yet.")
    @printf("\nNACA 0012, M %.2f, %.2f degrees; Vassberg and Jameson: cl %.6f, cd %.0e, cm %+.6f\n",
            MACH, ALPHA, REF.cl, REF.cd, REF.cm)
    @printf("  %6s %9s %9s %6s %9s %9s %10s %10s %10s %10s\n", "cells", "elements", "iters", "conv",
            "seconds", "cl", "cl error", "cd", "cm", "entropy")
    for (k, r) in enumerate(recs)
        @printf("  %6d %9d %9d %6s %9.0f %9.5f %+9.2f%% %10.2e %+10.5f %10.3e", r.cpc, r.elements,
                r.iterations, r.converged, r.seconds, r.cl, 100 * (r.cl - REF.cl) / REF.cl, r.cd,
                pitch(r), r.entropy_l2)
        if k > 1 && recs[k - 1].converged && r.converged
            q = recs[k - 1]
            ratio = r.cpc / q.cpc
            ord(a, b) = log(abs(a) / abs(b)) / log(ratio)
            @printf("; orders: cl error %.2f, cm error %.2f, |cd| %.2f, entropy %.2f",
                    ord(q.cl - REF.cl, r.cl - REF.cl), ord(pitch(q) - REF.cm, pitch(r) - REF.cm),
                    ord(q.cd, r.cd), ord(q.entropy_l2, r.entropy_l2))
        end
        println()
    end
end

# =====================================

cells = [parse(Int, a) for a in ARGS if all(isdigit, a)]
only_summary = "summary" in lowercase.(ARGS)
@printf("%d threads; output in %s\n", Threads.nthreads(), OUT)
if !only_summary
    for cpc in (isempty(cells) ? [32, 64, 128] : cells)
        run_case(cpc)
    end
end
summary()
