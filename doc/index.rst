
``rmsh`` documentation
======================

This is the documentation covering the :mod:`rmsh` library. The aim purpose of the
library is to generate elliptical meshes in 2D, as well as return information
related to mesh connectivity. The user-facing side is written entirely in Python
while the computation-heavy aspects are handled in the C-extension part. As such,
the aim of the module is to allow ease of use one would expect from Python, while
providing performance of C.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/blocks
   source/mesh
