# =====================================
# The uncut builder
#
# One element per cell of a uniform CartesianGrid, numbered like the grid's linear indices. Every
# vertex comes from `get_node`, and every parent area from differences of those node coordinates:
# `get_elem_area` (`prod(d)`) and the upper corner of `get_elem_bounds` (`lo + d`) do not agree with
# the nodes bitwise, which would leave full cells a volume fraction an ulp away from 1.

"""
    uncut_mesh(grid; periodic=(false, false)) -> PolygonMesh

The `PolygonMesh` of an uncut, uniform 2D `CartesianGrid`: one rectangle per cell, numbered like
`LinearIndices(grid)`, with `parent` that index and volume fraction exactly 1.

The sides are the boundaries `:x_neg`, `:x_pos`, `:y_neg` and `:y_pos`, in that order. A periodic
direction glues its two sides instead; it needs at least two cells.
"""
function uncut_mesh(grid::CartesianGrid{2,T}; periodic::NTuple{2,Bool}=(false, false)) where {T}
    n1, n2 = grid.n
    (n1 >= 1 && n2 >= 1) || throw(ArgumentError("the grid has $(grid.n) cells"))
    id(i, j) = i + (n1 + 1) * (j - 1)
    points = [SVector{2,T}(get_node(grid, CartesianIndex(i, j))) for j in 1:(n2 + 1)
              for i in 1:(n1 + 1)]
    x(i) = points[id(i, 1)][1]
    y(j) = points[id(1, j)][2]
    polygons = [Int32[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)] for j in 1:n2
                for i in 1:n1]
    parent_area = [(x(i + 1) - x(i)) * (y(j + 1) - y(j)) for j in 1:n2 for i in 1:n1]

    sides = Dict{NTuple{2,Int},Symbol}()
    for i in 1:n1
        sides[(id(i, 1), id(i + 1, 1))] = :y_neg
        sides[(id(i + 1, n2 + 1), id(i, n2 + 1))] = :y_pos
    end
    for j in 1:n2
        sides[(id(n1 + 1, j), id(n1 + 1, j + 1))] = :x_pos
        sides[(id(1, j + 1), id(1, j))] = :x_neg
    end
    glue = Pair{NTuple{2,Symbol},SVector{2,T}}[]
    periodic[1] && push!(glue, (:x_pos, :x_neg) => SVector(x(1) - x(n1 + 1), zero(T)))
    periodic[2] && push!(glue, (:y_pos, :y_neg) => SVector(zero(T), y(1) - y(n2 + 1)))
    names = [name for (name, p) in ((:x_neg, periodic[1]), (:x_pos, periodic[1]),
                                    (:y_neg, periodic[2]), (:y_pos, periodic[2])) if !p]
    return PolygonMesh(points, polygons, sides; periodic=glue, boundary_names=names,
                       parent=1:(n1 * n2), parent_area)
end
