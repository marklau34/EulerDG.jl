# =====================================
# Reference rules
#
# Gauss-Legendre on [0, 1], and symmetric rules on the triangle in barycentric coordinates. The
# weights of every rule sum to 1; a rule on a segment, triangle or rectangle scales them by its
# length or area.

"""
    GAUSS2, GAUSS3

The 2- and 3-point Gauss-Legendre rules on [0, 1], exact to degrees 3 and 5, as named tuples of
`nodes` and `weights`. `GAUSS2` is the face rule, and the 2 x 2 tensor rule on full cells.
"""
const GAUSS2 = (nodes=(1 / 2 - sqrt(3) / 6, 1 / 2 + sqrt(3) / 6), weights=(1 / 2, 1 / 2))
const GAUSS3 = (nodes=(1 / 2 - sqrt(15) / 10, 1 / 2, 1 / 2 + sqrt(15) / 10),
                weights=(5 / 18, 8 / 18, 5 / 18))

"""
    TRIANGLE3, TRIANGLE6

Symmetric rules on the triangle, exact to degrees 2 and 4 (Strang and Fix; Dunavant), as named
tuples of barycentric `points` and `weights`.
"""
const TRIANGLE3 = (points=((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)),
                   weights=(1 / 3, 1 / 3, 1 / 3))

const _T6_A1 = 0.44594849091596488631832925388305
const _T6_W1 = 0.22338158967801146569500700843312
const _T6_A2 = 0.091576213509770743459571463402202
const _T6_W2 = 0.10995174365532186763832632490021

const TRIANGLE6 = (points=((1 - 2 * _T6_A1, _T6_A1, _T6_A1), (_T6_A1, 1 - 2 * _T6_A1, _T6_A1),
                           (_T6_A1, _T6_A1, 1 - 2 * _T6_A1), (1 - 2 * _T6_A2, _T6_A2, _T6_A2),
                           (_T6_A2, 1 - 2 * _T6_A2, _T6_A2), (_T6_A2, _T6_A2, 1 - 2 * _T6_A2)),
                   weights=(_T6_W1, _T6_W1, _T6_W1, _T6_W2, _T6_W2, _T6_W2))
