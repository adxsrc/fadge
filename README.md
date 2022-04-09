[![Lint and test fadge](https://github.com/adxsrc/fadge/actions/workflows/python-test.yml/badge.svg)](https://github.com/adxsrc/fadge/actions/workflows/python-test.yml)
[![Publish fadge to PyPI](https://github.com/adxsrc/fadge/actions/workflows/python-publish.yml/badge.svg)](https://github.com/adxsrc/fadge/actions/workflows/python-publish.yml)


# `Fadge`

Fast Automatic Differential GEometry (`Fadge`) is a flexible and high
performance differential geometry package that use from python's
dynamic features and [Google JAX](https://github.com/google/jax)'s
automatic differentiation.

![Kerr-Schild Horizons](horizons.png)


## Background

Differential geometry is an import field of mathematics that has many
applications in physics and engineering.
For example, four-dimensional Lorentzian manifolds are models of
spacetime;
symplectic manifolds are generalization of the phase space of close
Hamiltonian systems.

Because of the wide range of applications, there are strong interests
in using differential geometry to solve particular physics and
engineering problems, resulting algorithms and software packages that
are application-driven.
Examples include tensor calculus packages, numerical relativity codes,
and symplectic integrators.
However, there are not too many generic differential geometry software
that are flexible enough to extend their original application domains.

Fast Automatic Differential GEometry (`Fadge`) is a flexible and high
performance differential geometry package that use from python's
dynamic features and [Google JAX](https://github.com/google/jax)'s
automatic differentiation.
It supports the more abstract notions of topological spaces and
manifolds *without* coordinate systems, and allows for creating more
complex manifolds by surgery.
Then, it supports defining multiple coordinate systems, deploying
different geometric structure, and tensor calculus on these manifolds.
`Fadge`'s target applications include cosmology with non-trivial
global topology, properties of almost extremely spinning black holes,
accurate numerical integration by switching between coordinate
systems, etc.


## Design

`Fadge` supports the notions of topological spaces and manifolds
*without* coordinate systems.
It is possible to first create these more generic manifolds, and then
assign geometric structure to them.
Because of this, the class and available methods of a `fadge` space
should be updated dynamically according to what attributes are given.
This is similar to (runtime checkable) python `Protocol` or
"interface" in the `go` programming language, but more extreme in the
sense that it udpates `__class__` itself.
