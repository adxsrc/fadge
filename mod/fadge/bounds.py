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


from math  import sqrt
from cmath import cos, acos

from jax   import numpy as np


def _cos23acos(x):
    y = cos(2 * acos(x) / 3)
    if y.imag != 0:
        return np.nan
    else:
        return y.real

def cos23acos(X):
    try:
        return np.array([_cos23acos(x) for x in np.array(X)])
    except TypeError:
        return _cos23acos(X)


Z1 = lambda a: 1 + np.cbrt(1 - a*a) * (np.cbrt(1 + a) + np.cbrt(1 - a))
Z2 = lambda a: np.sqrt(3 * a * a + Z1(a) * Z1(a))

reh1 = lambda a: 1 - np.sqrt(1 - a * a) # inner event horizon
reh2 = lambda a: 1 + np.sqrt(1 - a * a) # outer event horizon
rph1 = lambda a: 2 * (1 + cos23acos(-abs(a))) # inner bound of photon orbits
rph2 = lambda a: 2 * (1 + cos23acos( abs(a))) # outer bound of photon orbits
rmb1 = lambda a: 2 - abs(a) + 2 * np.sqrt(1 - abs(a)) # inner marginally bounded orbits
rmb2 = lambda a: 2 + abs(a) + 2 * np.sqrt(1 + abs(a)) # outer marginally bounded orbits
rms1 = lambda a: 3 + Z2(a) - np.sqrt((3 - Z1(a)) * (3 + Z1(a) + 2 * Z2(a))) # inner stable circular orbits
rms2 = lambda a: 3 + Z2(a) + np.sqrt((3 - Z1(a)) * (3 + Z1(a) + 2 * Z2(a))) # outer stable circular orbits
