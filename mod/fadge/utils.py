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


def quadratic(A, b, C, **kwargs): # b == B/2

    if A == 0: # linear case
        if b == 0:
            if C == 0:
                return ()
            else:
                raise ValueError(f'A={A}, B/2=b={b}, C={C} implies the contradiction {C}=0')
        else:
            return (-0.5 * C / b,)

    else: # quadratic ase
        bb = b * b
        AC = A * C
        if np.isclose(bb, AC, **kwargs): # double root
            x = - b / A
            return (x, x)
        elif bb > AC: # two roots
            if b == 0:
                x = np.sqrt(-C / A)
                return (-x, x)
            else:
                temp = -(b + np.sign(b) * np.sqrt(bb - AC))
                return tuple(sorted((temp / A, C / temp)))
        else: # no root
            return ()


def Nullify(metric, p=1):

    assert p > 0

    def nullify(x, v): # closure on `p`
        g = metric(x)
        A = v[:p] @ g[:p,:p] @ v[:p]
        b = v[p:] @ g[p:,:p] @ v[:p]
        C = v[p:] @ g[p:,p:] @ v[p:]
        return np.concatenate((v[:p], v[p:] / max(*quadratic(A, b, C))))

    return nullify
