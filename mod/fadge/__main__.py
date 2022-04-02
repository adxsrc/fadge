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

@click.option('--eps',         default=3e-2,    help="Stop integration `eps` outside the event horizon")
@click.option('--atol',        default=1e-3,    help="Absolute error tolerance in numerical integration")
@click.option('--setup',       default='image', help='Initial condition setup; can be "image" or "axis"')
@click.option('--alpha0',      default=0.0,     help='Initial condition offset along alpha direction')
@click.option('--beta0',       default=0.0,     help='Initial condition offset along beta  direction')
@click.option('--full/--ends', default=False,   help="Output full geodesics or only end points")

@click.argument('aspin',       type=float)
@click.argument('inclination', type=float)

def grrt(aspin, eps, setup, alpha0, beta0, full, atol, inclination):
    print( "Fadge: general relativistic ray tracing")
    print(f"    aspin       = {aspin:.2f}")
    print(f"    inclination = {inclination:g}")

    ns = GRRT(
        aspin,
        eps=eps, atol=atol, rtol=0,
        names={'ind':'lambda'},
        dtype=np.float64,
        steps=full, dense=False,
    )

    ns.set_cam(1e4, inclination, 0)
    if setup == 'image':
        ns.set_image(16, 128, alpha0=alpha0, beta0=beta0)
        out = f'image_a{aspin:.2f}_i{inclination:g}.h5'
    elif setup == 'axis':
        if beta0 != 0.0:
            print('WARNING: set_axis() does not support setting `beta0`; ignore')
        ns.set_axis(16, 1024, alpha0=alpha0)
        out = f'axis_a{aspin:.2f}_i{inclination:g}.h5'
    else:
        raise ValueError(f'Unknown setup "{setup}".')

    l, f = ns.geode()

    with h5py.File(out, 'w') as h:
        h.create_dataset('l', data=np.array(l))
        h.create_dataset('f', data=np.array(f))


#==============================================================================
# In case run as a script
if __name__ == "__main__":
    fadge()
