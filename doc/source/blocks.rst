.. currentmodule:: rmsh

.. _blocks:

Defining a Mesh
===============

Mesh Blocks
-----------

A mesh in :mod:`rmsh` consists of several blocks, which are described by :class:`MeshBlock`.
These define the boundaries of the blocks and optionally a label, by which the block specific
information can be obtained.

.. autoclass:: MeshBlock


Block Boundaries
----------------

As mentioned before, a boundary can be defined either as a curve using :class:`BoundaryCurve`
or it can be "soft" and connected to some other block using :class:`BoundaryBlock`. Regardless
of which boundary type is chosen, the boundary they connect to is indicated by a value from
a :class:`BoundaryId` enum.

.. autoclass:: BoundaryId

.. autoclass:: BoundaryCurve

.. autoclass:: BoundaryBlock
