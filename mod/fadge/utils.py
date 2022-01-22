# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
#
# This file is part of fadge.
#
# Fadge is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fadge is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with fadge.  If not, see <http://www.gnu.org/licenses/>.


from jax       import numpy as np
from jax.numpy import dot
from math      import isclose


def quadratic(A, b, C):
    bb = b * b
    AC = A * C
    dd = np.select([~np.isclose(bb, AC)], [bb - AC], 0)
    bs = np.heaviside(b, 1)
    D  = - (b + bs * np.sqrt(dd))
    x1 = D / A
    x2 = C / D
    return np.minimum(x1, x2), np.maximum(x1, x2)


def Nullify(metric, p=1):

    assert p > 0

    def nullify(x, v): # closure on `p`
        g = metric(x)
        A = v[:p] @ g[:p,:p] @ v[:p]
        b = v[p:] @ g[p:,:p] @ v[:p]
        C = v[p:] @ g[p:,p:] @ v[p:]

        d1, d2 = quadratic(A, b, C)
        S      = np.select([d1 > 0, d2 > 0], [d1, d2], np.nan)

        return np.concatenate([v[:p], v[p:] / S])

    return nullify
