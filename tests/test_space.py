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


from fadge import Basespace
from jax.tree_util import tree_flatten, tree_unflatten


class Dummyspace(Basespace):
    def __init__(self, *args):
        self.meta = None
        self.data = args[0] if len(args) == 1 else args


def test_space():

    x = Dummyspace(1)
    y = Dummyspace(x, x)
    z = Dummyspace(x, y)

    f, t = tree_flatten(z, is_leaf=lambda n: n.isleaf())

    print(f)
    print(t)

    assert f == [x, x, x]
