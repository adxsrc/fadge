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


from abc import ABC, abstractmethod
from jax.tree_util import register_pytree_node_class


class Basespace(ABC):
    """Basespace

    An abstract base class (ABC) for developing topological space
    classes that have mathematical counterparts.  As an ABC, it cannot
    be instantiated---one must first subclasses it to a concrete class
    such as Topospace below.  In addition, Basespace is intent to be
    used only within this file.  Subclassing outside this file should
    always based on concrete classes below.

    """
    __slots__ = ('data', 'meta')

    @abstractmethod
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    # Methods that make `Basespace` works as JAX pytree
    def isleaf(self):
        return not isinstance(self.data, (tuple, list))

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        # All subclasses of `Basespace` will be automatically
        # registered as JAX pytrees
        register_pytree_node_class(cls)

    def tree_flatten(self):
        # Note that tree_flattern() returns (data, meta), which is
        # different from
        return (self.data, self.meta)

    @classmethod
    def tree_unflatten(cls, meta, data):
        # Note that tree_unflattern accept (meta, data)
        return cls(*data, **meta)


class Topospace(Basespace):
    """Topospace

    Topological spaces are the most general mathematical spaces that
    allow for the definition of limits, continuity, and connectedness.

    """
    def __init__(self, discrete=False, spacename=None):
        self.spacename = spacename
        self.discrete  = discrete

    def __repr__(self):
        n = self.spacename if self.spacename else super().__repr__()
        d = 'Discrete'     if self.discrete  else ''
        return f'{d}{n}'


class Polyfold(Topospace):
    """Polyfold

    Polyfolds are spaces that may have varying dimensions.  They are
    generalization of the more familiar Manifolds.

    Without implementing anything concrete for Polyfold, it is served
    as a base class for Manifolds in `fadge`.

    """


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

    def __repr__(self):
        return f'{self.ndim}-'+super().__repr__()


class Patch(Manifold):
    """Coordinate Patch

    A simply-connected submanifold where coordinate charts can be
    defined.

    """
    def __init__(self, M, **kwargs):
        super().__init__(**kwargs)

        self.parent = M
        M.patches.append(self)
