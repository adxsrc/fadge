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


from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import h5py
import click

from fadge import GRRT


#==============================================================================
@click.group()
def fadge():
    pass


#==============================================================================
@fadge.command()
@click.argument("aspin",       type=float)
@click.argument("inclination", type=float)
def grrt(aspin, inclination):
    print( "Fadge: general relativistic ray tracing")
    print(f"    aspin       = {aspin:.2f}")
    print(f"    inclination = {inclination:g}")

    ns = GRRT(
        aspin,
        eps=1e-3, atol=1e-3, rtol=0,
        names={'ind':'lambda'},
        dtype=np.float64,
        steps=None, dense=None,
    )

    ns.set_cam(1e4, inclination, 0)
    ns.set_image(16, 256)

    l, f = ns.geode()

    with h5py.File(f'img_a{aspin:.2f}_i{inclination:g}.h5', 'w') as h:
        h.create_dataset('l', data=np.array([l[0],l[-1]]))
        h.create_dataset('f', data=np.array([f[0],f[-1]]))


#==============================================================================
# In case run as a script
if __name__ == "__main__":
    fadge()
