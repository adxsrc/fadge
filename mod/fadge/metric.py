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


from jax import numpy as np


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

    assert aspin == 0.0 # we are implementing the non-spinning case here
    assert ndim  == 4

    eta = Minkowski(ndim)(np.arange(4))

    def metric(x): # closure on `eta`
        r = np.sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3])
        f = 2.0/r
        l = np.array([1.0, x[1]/r, x[2]/r, x[3]/r])
        return eta + f * l[:,np.newaxis] * l[np.newaxis,:]

    return metric
