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

from .bounds import *


def PHI(a, r):
    "Teo (2003) equation 11b"
    if a == 0 and r == 3:
        return 0 # == - (9/2) * (r-3)/a
    elif a == 1:
        return - (r*r - 2*r - 1)
    else:
        return - (r*r*r - 3*r*r + a*a*r + a*a) / (a * (r-1))

def Q(a, r):
    "Teo (2003) equation 11b"
    if a == 0 and r == 3:
        return 27 # == - ((9/2) * (r-3)/a)**2 + 27
    if a == 1:
        return - r*r*r * (r-4)
    else:
        return - r*r*r * (r*r*r - 6*r*r + 9*r - 4*a*a) / (a*a * (r-1)*(r-1))


def shadow(aspin=1.1, inc=np.pi/2):
    if abs(aspin) <= 1:
        r1 = rph1(aspin)
    else:
        r1 = (aspin*aspin - 1)**(1/3) + 1

    rs = np.linspace(r1+1e-6, rph2(aspin)-1e-6, num=100)
    if abs(aspin) > 1:
        rs = np.concatenate([np.array([np.nan]), rs])

    a = np.array([PHI(aspin, r) / np.sin(inc) for r in rs])
    b = np.array([np.sqrt(Q(aspin, r) + (aspin * np.cos(inc))**2 - (PHI(aspin, r) / np.tan(inc))**2) for r in rs])

    return (
        np.concatenate([-a, -a[::-1], -a[:1]]),
        np.concatenate([ b, -b[::-1],  b[:1]]),
    )
