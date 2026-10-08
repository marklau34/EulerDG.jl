@testset "post" begin
    eq = CompressibleEulerEquations2D(1.4)
    g = CartesianGrid(Float64; mincorner=(-5, -5), maxcorner=(5, 5), n=(8, 8))
    semi = DGSemidiscretization(uncut_mesh(g; periodic=(true, true)), eq)

    # The projection of a linear field has no error; the vortex's does, falling with h.
    lin(x, t, eq) = SVector(1 + 0.1x[1], 0.2 - 0.05x[2], 0.3, 2 + 0.01 * (x[1] + x[2]))
    err = l2_error(semi, project(lin, semi), lin)
    @test maximum(err.l2) <= 1e-13 && maximum(err.linf) <= 1e-13
    vortex = IsentropicVortex()
    coarse = l2_error(semi, project(vortex, semi), vortex).l2[1]
    g2 = CartesianGrid(Float64; mincorner=(-5, -5), maxcorner=(5, 5), n=(16, 16))
    semi2 = DGSemidiscretization(uncut_mesh(g2; periodic=(true, true)), eq)
    fine = l2_error(semi2, project(vortex, semi2), vortex).l2[1]
    @test 0 < fine < coarse / 3

    dir = mktempdir()
    files = write_solution_vtk(joinpath(dir, "vortex"), semi2, project(vortex, semi2))
    @test length(files) == 1 && isfile(files[1])
    text = read(files[1], String)
    @test occursin("NumberOfPolys=\"256\"", text)
    @test all(occursin("Name=\"$n\"", text) for n in ("density", "velocity", "pressure", "mach",
                                                       "entropy"))
end
