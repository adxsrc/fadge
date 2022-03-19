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


from jax import numpy as np

from .shadow import PHI, Q


def cam(rij, ab):

    ci, si = np.cos(rij[1]), np.sin(rij[1])
    cj, sj = np.cos(rij[2]), np.sin(rij[2])

    R0 = rij[0] * si - ab[1] * ci # cylindrical radius
    z  = rij[0] * ci + ab[1] * si
    y  = R0     * sj + ab[0] * cj
    x  = R0     * cj - ab[0] * sj

    return np.array([
        [0, x, y, z],
        [1, si * cj, si * sj, ci],
    ], dtype=ab.dtype)


def sphorbit(aspin, r0):

    def thetadot(a, r):
        return np.sqrt(Q(a, r))

    def phidot(a, r):
        return (2*r*a + r*(r-2) * PHI(a, r)) / (r*r + a*a - 2*r)

    R = np.sqrt(r0*r0 + aspin*aspin)
    return np.array([
        [0, R, 0, 0],
        [1, 0, R * phidot(aspin, r0), -r0 * thetadot(aspin, r0)],
    ])
