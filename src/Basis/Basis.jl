# =====================================
# Basis: the per-element modal basis
#
# Downstream of the mesh contract. Each element gets its own P1 basis. It starts from 1, ξ and η,
# the coordinates scaled to [-1, 1] on the element's bounding box, so a sliver's basis is built on
# the sliver's own box. Modified Gram-Schmidt against the element's own quadrature then makes the
# modes orthogonal, normalized so the mass matrix is |K| times the identity. Mode 1 is the
# constant 1, so its coefficient is the cell average.

"""
    EulerDG.Basis

The per-element P1 modal basis.

- [`ElementBasis`](@ref) and [`element_basis`](@ref), which builds it on an element's bounding box
  against the element's quadrature rule.
- [`basis_values`](@ref) and [`basis_gradients`](@ref).
- [`mass_matrix`](@ref), which is |K| times the identity to round-off.
"""
module Basis

using StaticArrays: SMatrix, SVector

export ElementBasis, element_basis, basis_values, basis_gradients, mass_matrix

"""
    ElementBasis{T}

The three P1 modes of one element. With `ξ = (x - center) .* scale`, mode `i` is
`coeffs[i, 1] + coeffs[i, 2] ξ[1] + coeffs[i, 3] ξ[2]`. Mode 1 is the constant 1, and the mass
matrix is `mass` times the identity, `mass` being the element's area under its rule.
"""
struct ElementBasis{T}
    center::SVector{2,T}
    scale::SVector{2,T}
    coeffs::SMatrix{3,3,T,9}
    mass::T
end

@inline function _monomials(center, scale, x)
    ξ = (x - center) .* scale
    return SVector(one(ξ[1]), ξ[1], ξ[2])
end

"""
    basis_values(b, x) -> SVector{3,T}
    basis_values(b, x, origin) -> SVector{3,T}

The three modes of `b` at the point `x`, or at the point `origin + x`. With `origin` the basis's
own `center`, `x` keeps its full precision on a sliver far from the origin.
"""
@inline basis_values(b::ElementBasis, x) = b.coeffs * _monomials(b.center, b.scale, x)
@inline basis_values(b::ElementBasis, x, origin) =
    b.coeffs * _monomials(b.center - origin, b.scale, x)

"""
    basis_gradients(b) -> SMatrix{2,3,T}

The gradients of the three modes, one per column. They are constant on the element, and the
first is zero.
"""
@inline function basis_gradients(b::ElementBasis)
    c, s = b.coeffs, b.scale
    return SMatrix{2,3}(c[1, 2] * s[1], c[1, 3] * s[2], c[2, 2] * s[1], c[2, 3] * s[2],
                        c[3, 2] * s[1], c[3, 3] * s[2])
end

"""
    element_basis(lo, hi, x, w; origin=0) -> ElementBasis

Build the basis of the element with bounding box `(lo, hi)` and quadrature rule `(x, w)`, its
points relative to `origin`; the rule must integrate products of linear functions exactly. The
basis's `center` is `(lo + hi) / 2`; pass that as `origin`, with the rule from `polygon_rule` at
the same origin, to keep a sliver's precision. Throws if a mode is degenerate under the rule -- its
part orthogonal to the earlier modes has a squared norm below 64 eps of the area -- as on a sliver
too thin to resolve a gradient.
"""
function element_basis(lo::SVector{2,T}, hi::SVector{2,T}, x::AbstractVector{SVector{2,T}},
                       w::AbstractVector{T}; origin::SVector{2,T}=zero(SVector{2,T})) where {T}
    center = (lo + hi) / 2
    scale = 2 ./ (hi - lo)
    G = zero(SMatrix{3,3,T,9})
    for (xq, wq) in zip(x, w)
        m = _monomials(center - origin, scale, xq)
        G += wq * (m * m')
    end
    area = G[1, 1]
    inner(a, b) = a' * G * b
    v1 = SVector{3,T}(1, 0, 0)
    v2 = SVector{3,T}(0, 1, 0)
    v2 -= (inner(v2, v1) / area) * v1
    n2 = inner(v2, v2)
    n2 > 64 * eps(T) * area ||
        throw(ArgumentError("the element's x mode is degenerate under its rule"))
    v3 = SVector{3,T}(0, 0, 1)
    v3 -= (inner(v3, v1) / area) * v1
    v3 -= (inner(v3, v2) / n2) * v2
    n3 = inner(v3, v3)
    n3 > 64 * eps(T) * area ||
        throw(ArgumentError("the element's y mode is degenerate under its rule: its x and y " *
                            "modes are nearly dependent"))
    v2 *= sqrt(area / n2)
    v3 *= sqrt(area / n3)
    coeffs = SMatrix{3,3,T}(v1[1], v2[1], v3[1], v1[2], v2[2], v3[2], v1[3], v2[3], v3[3])
    return ElementBasis{T}(center, scale, coeffs, area)
end

"""
    mass_matrix(b, x, w; origin=0) -> SMatrix{3,3,T}

The mass matrix of `b` under the rule `(x, w)`, its points relative to `origin`. For any rule
exact on products of linear functions, it is `b.mass` times the identity, to round-off.
"""
function mass_matrix(b::ElementBasis{T}, x, w; origin::SVector{2,T}=zero(SVector{2,T})) where {T}
    M = zero(SMatrix{3,3,T,9})
    for (xq, wq) in zip(x, w)
        φ = basis_values(b, xq, origin)
        M += wq * (φ * φ')
    end
    return M
end

end # module Basis
