from dataclasses import dataclass
import numpy as np
from enum import Enum, unique


@dataclass(frozen=True)
class Line:
    """ Defines a line in terms of its topology. Each line should have a positive orientation.
    Values of `p1` and `p2` are indices of the nodes in the mesh.

    Attributes
    ----------
    p1 : int
        Index of the first point.
    p2 : int
        Index of the second point.
    """
    p1: int = -1
    p2: int = -1


@dataclass(frozen=True)
class Surface:
    """Mesh surface defined in terms of its topology. Each surface should have a CCW orientation. Values of `l1`, `l2`,
    `l3`, and `l4` are indices into the mesh line array, with 1 being the first element. Negative values indicate
    reverse orientation.

    Attributes
    ----------
    l1 : int
        Index of the first line.
    l2 : int
        Index of the second line.
    l3 : int
        Index of the third line.
    l4 : int
        Index of the fourth line.
    """
    l1: int = 0
    l2: int = 0
    l3: int = 0
    l4: int = 0


@unique
class BoundaryId(Enum):
    """Enum used to identify sides of the mesh in terms of topology, independent to its geometry.
    Valid values are:
        - BoundaryNorth
        - BoundaryEast
        - BoundaryWest
        - BoundarySouth
    """
    BoundaryNorth = 1
    BoundaryEast = 2
    BoundaryWest = 3
    BoundarySouth = 4


@dataclass(frozen=True)
class BoundaryBlock:
    """Defines a connection to a mesh block with the label `target`.

    Parameters
    ----------
    target : str
        Label of the block which is designated as the target.
    target_id : BoundaryId
        ID of the boundary of `target` to which this boundary will connect to.
    n : int = 0
        The number of points along this boundary, whill may be left as 0 if the value can be inferred from other
        boundaries this one has to match to.
    """
    target: str
    target_id: BoundaryId
    n: int = 0


@dataclass(frozen=True)
class BoundaryCurve:
    """Defines values of points along the boundary of a mesh block.

    Parameters
    ----------
    x : ndarray[float64]
        Values of X coordinate along the boundary.
    y : ndarray[float64]
        Values of Y coordinate along the boundary.
    """
    x: np.ndarray
    y: np.ndarray
