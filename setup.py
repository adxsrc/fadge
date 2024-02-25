#!/usr/bin/env python3
#
# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
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


from setuptools import setup, find_packages


setup(
    name='fadge',
    version='0.1.7',
    url='https://github.com/adxsrc/fadge',
    author='Chi-kwan Chan',
    author_email='chanc@arizona.edu',
    description='Fast Automatic Differential GEometry',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages('mod'),
    package_dir={'': 'mod'},
    entry_points={
        'console_scripts': [
            'fadge = fadge.__main__:fadge',
        ],
    },
    python_requires='>=3.7,<3.12',
    install_requires=[
        'xaj>=0.1.8,<0.2',
        'click>=7.1.2',
        'h5py>=3',
    ],
)
