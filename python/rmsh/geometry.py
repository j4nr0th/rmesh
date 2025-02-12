"""Geometrical primitive objects."""

from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np
import numpy.typing as npt

INVALID_POINT_IDX = -1
INVALID_LINE_IDX = 0
INVALID_SURFACE_IDX = 0


@dataclass(frozen=True)
class Line:
    """Defines a line in terms of its topology.

    Each line should have a positive orientation. Values of `p1` and `p2` are indices of
    the nodes in the mesh.

    Parameters
    ----------
    p1 : int
        Index of the first point.
    p2 : int
        Index of the second point.
    """

    p1: int = INVALID_POINT_IDX
    p2: int = INVALID_POINT_IDX


@dataclass(frozen=True)
class Surface:
    """Mesh surface defined in terms of its topology.

    Each surface should have a CCW orientation. Values of `l1`, `l2`,
    `l3`, and `l4` are indices into the mesh line array, with 1 being the first element.
    Negative values indicate reverse orientation.

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

    l1: int = INVALID_LINE_IDX
    l2: int = INVALID_LINE_IDX
    l3: int = INVALID_LINE_IDX
    l4: int = INVALID_LINE_IDX


@unique
class BoundaryId(IntEnum):
    """Enum used to identify sides of the mesh in terms of topology.

    Valid values are:
        - BoundaryNorth
        - BoundaryEast
        - BoundaryWest
        - BoundarySouth
    """

    BoundarySouth = 1
    BoundaryEast = 2
    BoundaryNorth = 3
    BoundaryWest = 4


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
        The number of points along this boundary, may be left as 0 if the value can
        be inferred from other boundaries this one has to match to.
    """

    target: str
    target_id: BoundaryId
    n: int = 0


@dataclass(frozen=True, init=False)
class BoundaryCurve:
    """Defines values of points along the boundary of a mesh block.

    Parameters
    ----------
    x : ndarray[float64]
        Values of X coordinate along the boundary.
    y : ndarray[float64]
        Values of Y coordinate along the boundary.
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    n: int

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike) -> None:
        xv = np.array(x, np.float64)
        yv = np.array(y, np.float64)
        if xv.ndim != 1 or xv.size != yv.size or xv.size == 0:
            raise ValueError(
                "Both x and y must be 1d arrays of the same (non-zero) length."
            )
        object.__setattr__(self, "x", xv)
        object.__setattr__(self, "y", yv)
        object.__setattr__(self, "n", int(xv.size))


_SolverCfgTuple = tuple[bool, float, int, int, int]
_BoundaryInfoTuple = (
    tuple[int, int, int, npt.NDArray[np.float64], npt.NDArray[np.float64]]
    | tuple[int, int, int, int, int]
)
_BlockInfoTuple = tuple[
    str, _BoundaryInfoTuple, _BoundaryInfoTuple, _BoundaryInfoTuple, _BoundaryInfoTuple
]
