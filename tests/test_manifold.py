# Copyright (C) 2022 Chi-kwan Chan
# Copyright (C) 2022 Steward Observatory
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


from fadge import Manifold, DiscreteManifold
from fadge.manifold import *


def test_manifold():

    M = Manifold()
    print(M)
    assert M.ndim == 2

    D = DiscreteManifold(3)
    print(D)
    assert D.ndim == 3

    S2 = Sphere()
    print(S2)
    assert S2.ndim == 2

    S3 = Sphere(3)
    print(S3)
    assert S3.ndim == 3
