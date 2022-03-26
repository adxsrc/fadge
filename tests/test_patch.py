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


from fadge import Manifold, Patch


def test_patch():

    M = Manifold(3)
    print(M)
    assert M.ndim == 3

    P1 = Patch(M)
    print(P1)
    assert P1.parent == M

    P2 = Patch(M)
    print(P2)
    assert P2.parent == M

    print(M.patches)
    assert M.patches == [P1, P2]
