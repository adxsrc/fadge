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


from fadge.metric import Minkowski, KerrSchild
from fadge.utils  import Nullify

from jax import numpy as np


def test_Minkowski():

    metric  = Minkowski()
    nullify = Nullify(metric)

    x  = np.ones(4)
    v  = np.ones(4)
    vn = nullify(x, v)

    print(vn)

    assert abs(vn @ metric(x) @ vn) <= 1e-7


def test_KerrSchild():

    for a in np.linspace(-1, 1, 21):
        metric  = KerrSchild(a)
        nullify = Nullify(metric)

        x  = np.ones(4) * 2
        v  = np.ones(4)
        vn = nullify(x, v)
        vv = vn @ metric(x) @ vn

        print(a, vn, vv)

        assert abs(vv) <= 1e-7


def test_pseudoRiemannian():

    def metric(x, ndim=6, p=2, **kwargs):
        return np.diag(np.array([-1.0] * p + [1.0] * (ndim-p), **kwargs))

    nullify = Nullify(metric, p=2)

    x  = np.ones(6)
    v  = np.ones(6)
    vn = nullify(x, v)
    vv = vn @ metric(x) @ vn

    print(vn, vv)

    assert abs(vv) <= 1e-7
