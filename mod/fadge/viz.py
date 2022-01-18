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


def KSHorizons(aspin, n=None):

    if n is None:
        n = 46 if aspin <= 1 else 721

    theta = np.linspace(0, np.pi, n)
    ct    = np.cos(theta)
    st    = np.sin(theta)

    aa    = aspin * aspin
    rhp   = 1 + np.sqrt(1 - aa)
    rhm   = 1 - np.sqrt(1 - aa)
    rep   = 1 + np.sqrt(1 - aa * ct * ct)
    rem   = 1 - np.sqrt(1 - aa * ct * ct)

    def Rz(r): # work for both scalar and array `r`
        R  = st * np.sqrt(r * r + aa)
        z  = ct * r
        return R[np.isfinite(R)], z[np.isfinite(z)]

    def plot2(ax, R, z, *args, **kwargs):
        ax.plot(R,  z, *args, **kwargs)
        ax.plot(-R, z, *args, **kwargs)

    def horizons(ax=None, color=None, eh=True, es=True, **kwargs):
        if ax is None:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1,1,**kwargs)
            ax.set_aspect('equal')

        p = ax.plot([-aspin, aspin], [0,0], ':', color=color)
        c = p[0].get_color()
        ax.scatter([-aspin, aspin], [0,0], color=c, label=f'a={aspin}')

        Rep, zep = Rz(rep)
        Rem, zem = Rz(rem)
        if abs(aspin) > 1:
            Re = np.concatenate([Rep, Rem[::-1], Rep[:1]])
            ze = np.concatenate([zep, zem[::-1], zep[:1]])
            if es:
                plot2(ax, Re, ze, color=c, linewidth=1)
        else:
            if es:
                plot2(ax, Rep, zep, color=c, linewidth=1)
                plot2(ax, Rem, zem, color=c, linewidth=1)
            if eh:
                plot2(ax, *Rz(rhp), color=c)
                plot2(ax, *Rz(rhm), color=c)

    return horizons
