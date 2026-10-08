@testset "vtk" begin
    dir = mktempdir()
    for (name, m) in pairs(fixtures())
        files = write_mesh_vtk(joinpath(dir, string(name)), m)
        @test length(files) == 2 && all(isfile, files) && all(f -> filesize(f) > 0, files)
        elements, faces = read(files[1], String), read(files[2], String)
        @test occursin("NumberOfPolys=\"$(nelements(m))\"", elements)
        @test occursin("NumberOfLines=\"$(nfaces(m))\"", faces)
        @test occursin("Name=\"volume_fraction\"", elements) && occursin("Name=\"closure\"", elements)
        @test occursin("Name=\"boundary\"", faces) && occursin("Name=\"periodic\"", faces)
    end
end
