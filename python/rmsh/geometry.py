"""Geometrical primitive objects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.special import comb

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

    BoundaryBottom = 1
    BoundaryRight = 2
    BoundaryTop = 3
    BoundaryLeft = 4

    @property
    def opposite_boundary(self) -> BoundaryId:
        """Return boundary opposite the specified one."""
        return BoundaryId(((self.value + 1) & 3) + 1)

    @property
    def next(self) -> BoundaryId:
        """Return the boundary which is the next in orientation."""
        return BoundaryId((self.value & 3) + 1)

    @property
    def prev(self) -> BoundaryId:
        """Return the boundary which is the previous in orientation."""
        return BoundaryId(((self.value - 2) & 3) + 1)


@dataclass(frozen=True)
class BoundaryRef:
    """Class which acts as the reference to a boundary."""

    block: MeshBlock
    boundary: BoundaryId


@dataclass(frozen=True)
class Boundary:
    """Base class for boundaries.

    Parameters
    ----------
    n : int
        Number of points on the boundary.
    """

    n: int


@dataclass(frozen=True)
class BoundaryBlock(Boundary):
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

    ref: BoundaryRef


@dataclass(frozen=True)
class BoundaryCurve(Boundary):
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

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike) -> None:
        xv = np.array(x, np.float64)
        yv = np.array(y, np.float64)
        if xv.ndim != 1 or yv.ndim != 1 or xv.size != yv.size or xv.size == 0:
            raise ValueError(
                "Both x and y must be 1d arrays of the same (non-zero) length."
            )
        object.__setattr__(self, "x", xv)
        object.__setattr__(self, "y", yv)
        super().__init__(xv.size)

    @classmethod
    def from_samples(
        cls,
        n: int,
        func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
        distribution: Callable[[npt.NDArray[np.float64]], npt.ArrayLike] | None = None,
    ) -> Self:
        """Create boundary based on samples of a callable.

        Parameters
        ----------
        n : int
            Number of samples to take.

        func : (array) -> array_like
            Coordinates as function of fraction of distance along the curve.

        distribution : (array_like) -> array, optional
            Function to transform the uniform samples into desired distribution.

        Returns
        -------
        BoundaryCurve
            Boundary which follows the specified function.
        """
        if n < 2:
            raise ValueError(
                f"Boundary must have at least 2 points, but {n=} were given."
            )

        sample_fractions = np.linspace(0, 1, n, dtype=np.float64)
        if distribution is not None:
            sample_fractions = np.asarray(
                distribution(sample_fractions), dtype=np.float64, copy=None
            )

        # Don't copy if not necessary
        coords = np.asarray(func(sample_fractions), np.float64, copy=None)
        if coords.shape != (n, 2):
            raise ValueError(
                f"Function given did not map {n} input samples into n 2d vectors"
                f" (shape {(n, 2)}), instead the resulting array had the shape"
                f" {coords.shape}."
            )

        return cls(coords[:, 0], coords[:, 1])

    @classmethod
    def from_line(
        cls,
        n: int,
        begin: npt.ArrayLike,
        end: npt.ArrayLike,
        distribution: Callable[[npt.NDArray[np.float64]], npt.ArrayLike] | None = None,
    ) -> Self:
        """Create boundary based on samples of a callable.

        Parameters
        ----------
        n : int
            Number of samples to take.

        begin : array_like
            Coordinates of the start function.

        end : array_like
            Coordinates of the end function.

        distribution : (array_like) -> array, optional
            Function to transform the uniform samples into desired distribution.

        Returns
        -------
        BoundaryCurve
            Boundary which follows the specified function.
        """
        p0 = np.asarray(begin, np.float64, copy=None).flatten()
        p1 = np.asarray(end, np.float64, copy=None).flatten()
        if p0.shape != (2,) or p1.shape != (2,):
            raise ValueError("Both begin and end must be arrays with 2 elements.")
        return cls.from_samples(
            n,
            lambda s: p0[None, :] * (1 - s[:, None]) + p1[None, :] * s[:, None],
            distribution,
        )

    @classmethod
    def from_knots(
        cls,
        n: int,
        *knots: tuple[float, float] | npt.ArrayLike,
        distribution: Callable[[npt.NDArray[np.float64]], npt.ArrayLike] | None = None,
    ) -> Self:
        """Create the boundary using Bernstein polynomials with specified knots.

        Parameters
        ----------
        n : int
            Number of samples to take.

        *knots : (float, float) or array_like
            Knots to use for Bernstein polynomials.

        distribution : (array_like) -> array, optional
            Function to transform the uniform samples into desired distribution.

        Returns
        -------
        BoundaryCurve
            Boundary which follows the specified function.
        """
        if len(knots) < 2:
            raise ValueError("At least two knots are needed")

        weights = np.array(knots, np.float64)
        if weights.ndim != 2 or weights.shape[1] != 2:
            raise ValueError("Each knot should contain two elements.")

        pos = np.linspace(0, 1, n, dtype=np.float64)
        if distribution is not None:
            pos = np.asarray(distribution(pos), np.float64, copy=None)

        nk = weights.shape[0]
        k = np.arange(nk)
        bern = comb(nk - 1, k) * pos[:, None] ** k * (1 - pos[:, None]) ** (nk - 1 - k)
        coords = bern @ weights
        return cls(coords[:, 0], coords[:, 1])


_SolverCfgTuple = tuple[bool, float, int, int, int]
_BoundaryInfoTuple = (
    tuple[int, int, int, npt.NDArray[np.float64], npt.NDArray[np.float64]]
    | tuple[int, int, int, int, int]
)
_BlockInfoTuple = tuple[
    str, _BoundaryInfoTuple, _BoundaryInfoTuple, _BoundaryInfoTuple, _BoundaryInfoTuple
]


@dataclass
class MeshBlock:
    """Basic building block of the mesh, which describes a structured mesh block.

    Boundaries of the block can be either prescribed curves, or instead a
    "soft" boundary, where the only information specified is that it should connect to
    another block. Such "soft" boundaries allow for a potentially smoother transition
    between different blocks.

    When blocks share a boundary, the only requirement is that the boundary that is
    shared between them has such a number of nodes that the opposite boundary matches it.

    Parameters
    ----------
    label : str
            Label by which this block will be referred to in the mesh, as well as by other
            blocks which have a block
            boundary to this block.
    boundaries : Mapping of BoundaryId to BoundaryCurve or BoundaryBlock
        Dictionary containing the boundaries with their respective IDs. All four might be
        specified, or only a few.
    """

    label: str | None = None
    bottom: Boundary | None = None
    right: Boundary | None = None
    top: Boundary | None = None
    left: Boundary | None = None

    def has_all_boundaries(self) -> bool:
        """Check if a mesh block has all four boundaries specified as non-None values.

        Returns
        -------
        bool
            True if the mesh block has all needed boundaries and False if that is not the
            case.
        """
        return (
            (self.top is not None)
            and (self.bottom is not None)
            and (self.left is not None)
            and (self.right is not None)
        )

    def bbnd_top(self, n: int = 0) -> BoundaryBlock:
        """Boundary to this block's boundary."""
        return BoundaryBlock(n, BoundaryRef(self, BoundaryId.BoundaryTop))

    def bbnd_bottom(self, n: int = 0) -> BoundaryBlock:
        """Boundary to this block's boundary."""
        return BoundaryBlock(n, BoundaryRef(self, BoundaryId.BoundaryBottom))

    def bbnd_left(self, n: int = 0) -> BoundaryBlock:
        """Boundary to this block's boundary."""
        return BoundaryBlock(n, BoundaryRef(self, BoundaryId.BoundaryLeft))

    def bbnd_right(self, n: int = 0) -> BoundaryBlock:
        """Boundary to this block's boundary."""
        return BoundaryBlock(n, BoundaryRef(self, BoundaryId.BoundaryRight))

    def get_boundary_by_id(self, bnd_id: BoundaryId, /) -> Boundary | None:
        """Return the boundary with the correct id."""
        if bnd_id == BoundaryId.BoundaryBottom:
            return self.bottom
        if bnd_id == BoundaryId.BoundaryTop:
            return self.top
        if bnd_id == BoundaryId.BoundaryLeft:
            return self.left
        if bnd_id == BoundaryId.BoundaryRight:
            return self.right

        raise ValueError(f"Id {bnd_id} is not a valid value.")

    def get_boundary_by_id_existing(self, bnd_id: BoundaryId, /) -> Boundary:
        """Get a boundary for the corresponding ID and make sure it's not None."""
        bnd = self.get_boundary_by_id(bnd_id)
        if bnd is None:
            raise ValueError(f"Block does not have a defined boundary for id {bnd_id}.")
        return bnd

    def set_boundary_by_id(self, bnd_id: BoundaryId, val: Boundary | None, /) -> None:
        """Return the boundary with the correct id."""
        if bnd_id == BoundaryId.BoundaryBottom:
            self.bottom = val
        elif bnd_id == BoundaryId.BoundaryTop:
            self.top = val
        elif bnd_id == BoundaryId.BoundaryLeft:
            self.left = val
        elif bnd_id == BoundaryId.BoundaryRight:
            self.right = val
        else:
            raise ValueError(f"Id {bnd_id} is not a valid value.")
