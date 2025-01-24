"""Module which deals with 2D mesh generation."""

# Geometrical primitives
from rmsh._geometry import BoundaryBlock as BoundaryBlock
from rmsh._geometry import BoundaryCurve as BoundaryCurve
from rmsh._geometry import BoundaryId as BoundaryId
from rmsh._geometry import Line as Line
from rmsh._geometry import Surface as Surface

# The mesh object
from rmsh._mesh2d import Mesh2D as Mesh2D

# Mesh blocks and solver
from rmsh._meshblocks import MeshBlock as MeshBlock
from rmsh._meshblocks import SolverConfig as SolverConfig
from rmsh._meshblocks import create_elliptical_mesh as create_elliptical_mesh
