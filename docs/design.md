# Design

There are multiple difficulties when designing a generic differential
geometry package in python.

First, while differential geometry are done manifolds, the concept of
manifolds itself does not require coordinate systems or even
differential structures.
In order to support the abstract notions of topological spaces and
manifolds *without* coordinate systems, we need to design `fadge` so
that it is possible to first create these more generic manifolds, and
then assign more specific geometric structure to them.
Because of this, the class and available methods of a `fadge` space
should be updated dynamically according to what attributes are given.
This is similar to (runtime checkable) python `Protocol` or
"interface" in the `go` programming language, but more extreme in the
sense that it udpates `__class__` itself.

Second, topological spaces are not necessary connected.
While it is natural to use list (or tuple) to keep track of the
connected subspaces of disconnected space, it is ugly to manually
implement mapped methods of the disconnected space.
In order to promote code reuse, we dynamically create classes and use
JAX's [`pytrees`](https://jax.readthedocs.io/en/latest/pytrees.html)
mechanism to propagate methods.
