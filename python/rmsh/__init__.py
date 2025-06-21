"""Module which deals with 2D mesh generation."""

# Geometrical primitives
from rmsh.geometry import BoundaryBlock as BoundaryBlock
from rmsh.geometry import BoundaryCurve as BoundaryCurve
from rmsh.geometry import BoundaryId as BoundaryId
from rmsh.geometry import BoundaryRef as BoundaryRef
from rmsh.geometry import Line as Line
from rmsh.geometry import MeshBlock as MeshBlock
from rmsh.geometry import Surface as Surface

# The mesh object
from rmsh.mesh2d import Mesh2D as Mesh2D
from rmsh.meshblocks import SolverConfig as SolverConfig
from rmsh.meshblocks import create_elliptical_mesh as create_elliptical_mesh
