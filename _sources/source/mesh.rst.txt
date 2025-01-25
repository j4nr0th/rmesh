.. currentmodule:: rmsh

.. _mesh:

Mesh
====

:class:`Mesh2D` is the class which provides most of functionality. It is not intended
to be instantiated manually, but rather created by calling :func:`create_elliptical_mesh`.
It provides very low-level access to the mesh geometry, as it is intended to be as fast
as possible to use.

.. autoclass:: Mesh2D
    :members:
    :inherited-members:
