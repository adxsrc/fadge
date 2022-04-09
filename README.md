[![Lint and test fadge](https://github.com/adxsrc/fadge/actions/workflows/python-test.yml/badge.svg)](https://github.com/adxsrc/fadge/actions/workflows/python-test.yml)
[![Publish fadge to PyPI](https://github.com/adxsrc/fadge/actions/workflows/python-publish.yml/badge.svg)](https://github.com/adxsrc/fadge/actions/workflows/python-publish.yml)


# `Fadge`

Fast Automatic Differential GEometry (`Fadge`) uses python's dynamic
nature and [Google JAX](https://github.com/google/jax)'s automatic
differentiation to create a flexible and high performance differential
geometry package.

![Kerr-Schild Horizons](horizons.png)


## Background

Differential geometry is a field of mathematics that has important
applications in physics.
For example, four-dimensional Lorentzian manifolds are models of
spacetime;
symplectic manifolds are generalization of the phase space of close
Hamiltonian systems.

Because of the wide range of applications, there are strong interests
in using differential geometry to solve particular physics and
engineering problems, resulting algorithms and software packages that
are application driven.
Examples include tensor caculus packages, numerical relativity codes,
and symplectic integrators.
However, there are not too many generic differential geometry software
that are flexible enough to extend their original application domain.

Fast Automatic Differential GEometry (`Fadge`) uses python's dynamic
nature and [Google JAX](https://github.com/google/jax)'s automatic
differentiation to create a flexible and high performance differential
geometry package.
It supports the more abstract notions of topological spaces and
manifolds *without* coordinate systems, and allows for creating more
complex manifolds by surgery.
Then, it supports definiting multiple coordinate systems, deploying
different geometric structure, and tensor calculus on these manifolds.

Target applications include cosmology with non-trivial global
topology, properties of almost extremely spinning black holes,
accurate geodesics integration by switching between coordinate
systems.


## Design

`Fadge` supports the notions of topological spaces and manifolds
*without* coordinate systems.
It is possible to first create these more "abstract manifolds", and
then assign geometric structure to them.
Because of this, the class of a `fadge` space should be updated
according to what attribute is given.
Conceptually, this is similar to (`runtime_checkable`) `Protocol` or
"interface" in the `go` programming language.
