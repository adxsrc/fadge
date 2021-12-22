# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
#
# This file is part of PRay.
#
# PRay is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PRay is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PRay.  If not, see <http://www.gnu.org/licenses/>.


from xaj import odeint

from jax              import numpy as np
from jax.numpy        import dot
from jax.numpy.linalg import inv

from jax import jacfwd
from jax import jit


def JA(metric):
    """Jacobian-Affine Formulation"""

    dmetric = jacfwd(metric) # "render" the Jacobian of the metric function

    @jit
    def rhs(state):
        x  = state[:4]
        v  = state[4:]

        g  =  metric(x)
        dg = dmetric(x)

        ig = inv(g)

        a  = (-       dot(ig, dot(dot(dg, v), v))
              + 0.5 * dot(ig, dot(v, dot(v, dg))))

        return np.concatenate([v, a])

    return rhs


class Geode:

    def __init__(self, metric, l, s, L=None):
        assert s.ndim > 0 and 8 in s.shape

        rhs = JA(metric)

        if s.ndim > 1:
            from jax.experimental.maps import xmap
            i   = len(s.shape)-1 - s.shape[::-1].index(8)
            m   = {j:j for j in range(s.ndim) if j != i}
            rhs = xmap(rhs, in_axes=m, out_axes=m)

        self.geode = odeint(lambda x, y: rhs(y), l, s, 1 if L is None else abs(L))
        if L is not None:
            self.geode.extend(L)

    def solve(self, L):
        self.geode.extend(L)

    @property
    def lambdas(self):
        return self.geode.xs

    @property
    def states(self):
        return self.geode.ys

    def __call__(self, *args, **kwargs):
        return self.geode(*args, **kwargs)
