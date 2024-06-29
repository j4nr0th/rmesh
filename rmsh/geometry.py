from dataclasses import dataclass
import numpy as np
from enum import Enum, unique


@dataclass(frozen=True)
class Line:
    """
    Defines a line in terms of its topology. Each line should have a positive orientation.
    Values of *p1* and *p2* are indices of the nodes in the mesh.
    """
    p1: int = -1
    p2: int = -1


@dataclass(frozen=True)
class Surface:
    """
    Mesh surface defined in terms of its topology. Each surface should have a CCW orientation.
    Values of *l1*, *l2*, *l3*, and *l4* are indices into the mesh line array, with 1 being the first element.
    Negative values indicate reverse orientation.
    """
    l1: int = 0
    l2: int = 0
    l3: int = 0
    l4: int = 0


@unique
class BoundaryId(Enum):
    BoundaryNorth = 1
    BoundaryEast = 2
    BoundaryWest = 3
    BoundarySouth = 4


@dataclass(frozen=True)
class BoundaryBlock:
    """
    Defines a connection to a mesh block with the label *target*.
    """
    target: str
    target_id: BoundaryId
    n: int = 0


@dataclass(frozen=True)
class BoundaryCurve:
    """
    Defines values of points along the boundary of a mesh block.
    """
    x: np.ndarray
    y: np.ndarray
