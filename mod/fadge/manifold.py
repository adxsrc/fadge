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


from .hierarchy import Manifold


#==============================================================================
# Base classes

class DiscreteManifold(Manifold):
    """DiscreteManifold

    A discrete manifold used in discrete differential geometry.

    """
    def __repr__(self):
        return f'{self.ndim}-discretemanifold'


#==============================================================================
# Concrete classes

class Sphere(Manifold):

    def __repr__(self):
        return f'S{self.ndim}'
