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
try:
    from collections import Iterable
except:
    Iterable = (tuple, list)


def KSHorizons(aspin, n=None):

    if n is None:
        n = 46 if aspin <= 1 else 721

    phi   = np.linspace(-np.pi, np.pi, 2*(n-1)+1)
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
        ax.plot( R, z, *args, **kwargs)
        ax.plot(-R, z, *args, **kwargs)

    def xy(r):
        R = max(np.sqrt(r * r + aa), 1e-4)
        return R * np.sin(phi), R * np.cos(phi)

    def plotedgeon(ax, color=None, eh=True, es=True, r=[]):
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

        for r0 in r:
            plot2(ax, *Rz(r0), '--', color=c, linewidth=1)

    def plotfaceon(ax, color=None, eh=True, es=True, r=[]):
        p = ax.plot(*xy(0.0), color=color, linewidth=4, label=f'a={aspin}')
        c = p[0].get_color()

        if abs(aspin) > 1:
            if es:
                ax.plot(*xy(2), color=c, linewidth=1)
        else:
            if es:
                ax.plot(*xy(2), color=c, linewidth=1)
            if eh:
                ax.plot(*xy(rhp), color=c)
                ax.plot(*xy(rhm), color=c)

        for r0 in r:
            ax.plot(*xy(r0), '--', color=c, linewidth=1)

    def horizons(ax=None, color=None, faceon=False, eh=True, es=True, r=None, **kwargs):
        if ax is None:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1,1,**kwargs)
            ax.set_aspect('equal')

        if r is None:
            r = []
        elif not isinstance(r, Iterable):
            r = [r]

        if faceon:
            plotfaceon(ax, color, eh, es, r)
        else:
            plotedgeon(ax, color, eh, es, r)

    return horizons
