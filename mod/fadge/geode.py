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
        x  = state[0]
        v  = state[1]

        g  =  metric(x)
        dg = dmetric(x)

        a  = inv(g) @ (- (dg @ v) @ v + 0.5 * v @ (v @ dg))
        return np.array([v, a], dtype=state.dtype)

    return rhs


class Geode:

    def __init__(self, metric, l, s, L=None, eqax=None, **kwargs):
        assert s.ndim >= 2

        rhs = lambda l, s: JA(metric)(s)
        if eqax is None:
            for offset in range(s.ndim-2+1):
                if s.shape[offset:offset+2] == (2,4):
                    eqax = [offset, offset+1]
                    break
            else:
                raise ValueError(f'cannot deduce `eqax` from state array with shape {s.shape}')

        kwargs['eqax'] = eqax
        if 'h' not in kwargs:
            kwargs['h'] = 1 if L is None else abs(L)
        self.geode = odeint(rhs, l, s, **kwargs)

        if L is not None:
            self.geode.extend(L)

    def extend(self, L, **kwargs):
        self.geode.extend(L, **kwargs)

    @property
    def lambdas(self):
        return self.geode.xs

    @property
    def states(self):
        return self.geode.ys

    def __call__(self, *args, **kwargs):
        return self.geode(*args, **kwargs)
