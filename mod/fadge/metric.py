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


from jax import numpy as np, jit


def Cartesian(ndim=3, **kwargs):

    assert ndim > 0

    g = np.identity(ndim, **kwargs) # render constant metric

    def metric(x): # closure on `g`
        return g

    return metric


def Minkowski(ndim=4, **kwargs):

    assert ndim > 1

    g = np.diag(np.array([-1.0] + [1.0] * (ndim-1), **kwargs)) # render constant metric

    def metric(x): # closure on `g`
        return g

    return metric


def KerrSchild(aspin=0.0, ndim=4, **kwargs):

    assert ndim == 4

    eta = Minkowski(ndim)(None)
    aa  = aspin * aspin

    @jit
    def metric(x): # closure on `eta`, `aspin`, and `aa`
        zz = x[3] * x[3]
        kk = 0.5 * (x[1] * x[1] + x[2] * x[2] + zz - aa)
        rr = np.sqrt(kk * kk + aa * zz) + kk
        r  = np.sqrt(rr)
        f  = (2.0 * rr * r) / (rr * rr + aa * zz)
        l  = np.array([
            1.0,
            (r * x[1] + aspin * x[2]) / (rr + aa),
            (r * x[2] - aspin * x[1]) / (rr + aa),
            x[3] / r,
        ])
        return eta + f * l[:,np.newaxis] * l[np.newaxis,:]

    return metric
