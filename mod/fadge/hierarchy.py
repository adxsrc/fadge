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


class Polyfold:
    """Polyfold

    Polyfolds are spaces that may have varying dimensions.  They are
    generalization of the more familiar Manifolds.

    Without implementing anything concrete for Polyfold, it is served
    as a base class for Manifolds in `fadge.

    """
    def __init__(self, discrete=False, typename=None):
        self.discrete = discrete
        self.typename = typename

    def __repr__(self):
        d = 'discrete'    if self.discrete else ''
        n = self.typename if self.typename else self.__class__.__name__.lower()
        return f'{self.ndim}-{d}{n}'


class Manifold(Polyfold):
    """Manifold

    A manifold M is a topological space that resembles Euclidean space
    R^n around each point p in M.  n is called the dimension of M.

    Args:
        ndim: dimension of the manifold.

    Returns:
        An object representing the manifold.

    """
    def __init__(self, ndim=2, **kwargs):
        super().__init__(**kwargs)

        self.ndim    = ndim
        self.patches = []


class Patch(Manifold):
    """Coordinate Patch

    A simply-connected submanifold where coordinate charts can be
    defined.

    """
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)

        self.parent = M
        M.patches.append(self)
