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


def Nullify(metric, p=1):

    assert p > 0

    def nullify(x, v): # closure on `p`
        g = metric(x)
        A = v[:p] @ g[:p,:p] @ v[:p]
        B = v[p:] @ g[p:,:p] @ v[:p]
        C = v[p:] @ g[p:,p:] @ v[p:]
        D = - 2 * B / C if A == 0 else - A / (B + np.sqrt(B * B - A * C)) # normalization
        return np.concatenate((v[:p], v[p:] * D))

    return nullify
