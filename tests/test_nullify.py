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

        x  = np.ones(4)
        v  = np.ones(4)
        vn = nullify(x, v)

        print(a, vn)

        assert abs(vn @ metric(x) @ vn) <= 1e-7
