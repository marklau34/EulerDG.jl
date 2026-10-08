# The plan's rule that nothing downstream of the mesh contract ever sees a Cartesian index, made
# checkable: no file in the contract layer or in a solver-side submodule may refer to the
# preprocessor's modules, to its libraries, or to their grid types. Module references are matched
# with their dot, and the other names as whole words, so prose such as "the geometry of a cut
# mesh" passes.

const DOWNSTREAM = ("common", "Quadrature", "Basis", "Equations", "Semidiscretization",
                    "PseudoTime", "Post")
const FORBIDDEN = (r"\.Geometry\b", r"\.CutMesh\b", r"\bSDFLibrary\b", r"\bCutCellMethods\b",
                   r"\bCartesianMeshes\b", r"\bCartesianIndex\b", r"\bCartesianGrid\b")

# Each (line, match) in `text` that breaks the layering rule.
layering_violations(text::AbstractString) =
    [(i, m.match) for (i, line) in enumerate(eachline(IOBuffer(text))) for re in FORBIDDEN
     for m in eachmatch(re, line)]

@testset "layering" begin
    src = joinpath(pkgdir(EulerDG), "src")
    for dir in DOWNSTREAM
        @test isdir(joinpath(src, dir))
        for (root, _, files) in walkdir(joinpath(src, dir)), file in files
            endswith(file, ".jl") || continue
            path = joinpath(root, file)
            bad = layering_violations(read(path, String))
            isempty(bad) || @info "layering violation" path bad
            @test isempty(bad)
        end
    end
    @test !isempty(layering_violations("idx = CartesianIndex(1, 2)"))
    @test !isempty(layering_violations("using ..CutMesh: build"))
    @test !isempty(layering_violations("import SDFLibrary"))
    @test isempty(layering_violations("the geometry of a cut mesh, by Cartesian index"))
end
