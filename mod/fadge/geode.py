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


from jax              import numpy as np
from jax.numpy        import dot
from jax.numpy.linalg import inv

from jax import jacfwd
from jax import jit


def Geode(metric):

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
