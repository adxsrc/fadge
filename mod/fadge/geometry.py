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


class Chart(Patch):
    """Coordinate Chart

    A coordiante chart is a map:

        phi: U -> V

    where U is an open set in a manifold M, V is an open set in R^n
    and n is the dimension of the manifold.

    Given that

    """
    def __init__(self, phi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi = phi

    def __call__(self, p):
        assert p in self
        return self.phi(p)


class Atlas:
    """Atlas

    An atlas for ...

    """
    def __init__(self, *args, **kwargs):
        self.charts = args
