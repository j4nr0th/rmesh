"""Stub detailing the C-extension types used internally."""

from __future__ import annotations

from typing import Self

import numpy as np
import numpy.typing as npt

from rmsh.geometry import _BlockInfoTuple, _SolverCfgTuple

class _Mesh2D:
    """Internal mesh interface."""

    def block_lines(self, idx: int, /) -> npt.NDArray[np.int32]:
        """Return indices of all lines within a block.

        Indices start at 1 and a negative value indicates a reversed orientation of
        the line.

        Parameters
        ----------
        block_id : int
            The index of the block for which the line indices should be returned.

        Returns
        -------
        ndarray[int32]
            Array with indices of all lines within the mesh block specified by
            ``block_id``.
        """
        ...

    def block_boundary_lines(
        self, block_id: int, boundary_id: int, /
    ) -> npt.NDArray[np.int32]:
        """Return indices of all lines on a boundary of a block.

        Indices start at 1 and a negative value indicates a reversed orientation of the
        line.

        Parameters
        ----------
        block_id : int
            The index of the block for which the line indices should be returned.
        boundary : BoundaryId
            The ID of a boundary from which the line indices should be returned.

        Returns
        -------
        ndarray[int32]
            Array with indices of all lines on a boundary of a block ``block_id``.
        """
        ...

    def block_boundary_points(
        self, block_id: int, boundary_id: int, /
    ) -> npt.NDArray[np.int32]:
        """Return indices of all nodes on a boundary of a block.

        Parameters
        ----------
        block_id : int
            The index of the block for which the point indices should be returned.
        boundary : BoundaryId
            The ID of a boundary from which the point indices should be returned.

        Returns
        -------
        ndarray[int32]
            Array with indices of all points on a boundary of a block ``block_id``.
        """
        ...

    def block_boundary_surfaces(
        self, block_id: int, boundary_id: int, /
    ) -> npt.NDArray[np.int32]:
        """Return indices of all surfaces on a boundary of a block.

        Indices start at 1 and a negative value indicates a reversed orientation
        of the surface, though for this function this is not needed.

        Parameters
        ----------
        block_id : str
            The label of the block for which the surfaces indices should be returned.
        boundary : BoundaryId
            The ID of a boundary from which the surfaces indices should be returned.

        Returns
        -------
        ndarray[int32]
            Array with indices of all surfaces on a boundary of a block ``block_id``.
        """
        ...

    def surface_element(self, surface_id: int, order: int, /) -> npt.NDArray[np.int32]:
        """Return indices of surfaces, which form a square element of width (2*order+1).

        This is intended to be used for computing cell-based interpolations.

        Parameters
        ----------
        surf : int
            The one-based index of the surface which should be the center of the element.
        order : int
            Size of the element in each direction away from the center (zero means only
            the element, one means 3 x 3, etc.)

        Returns
        -------
        ndarray[int32]
            Array with indices of all surfaces in the element. Note that since one-based
            indexing is used, a zero indicates a missing surface caused by a numerical
            boundary. Negative indices mean a negative orientation.
        """
        ...

    def surface_element_points(
        self, surface_id: int, order: int, /
    ) -> npt.NDArray[np.int32]:
        """Return indices of points, which form a square element of width (2*order+1).

        This is intended to be used for computing nodal-based interpolations for surface
        elements.

        Parameters
        ----------
        surf : int
            The one-based index of the surface which should be the center of the element.
        order : int
            Size of the element in each direction away from the center (zero means only
            the element, one means 3 x 3, etc.)

        Returns
        -------
        ndarray[int32]
            Array with indices of all indices in the element. Note that since one-based
            indexing is used, a value of -1 indicates a missing point caused by a
            numerical boundary.
        """
        ...

    @property
    def pos_x(self) -> npt.NDArray[np.float64]:
        """X coordinates of nodes."""
        ...

    @property
    def pos_y(self) -> npt.NDArray[np.float64]:
        """Y coordinates of nodes."""
        ...

    @property
    def z(self) -> npt.NDArray[np.float64]:
        """Z coordinates of nodes."""
        ...

    @property
    def lines(self) -> npt.NDArray[np.int32]:
        """Indices of nodes for each line.

        Has a shape ``(N, 2)``, where ``N`` is the number of lines in the mesh.
        """
        ...

    @property
    def surfaces(self) -> npt.NDArray[np.int32]:
        """Indices of lines for each surface.

        Has a shape ``(N, 4)``, where `N` is the number of surfaces in the mesh.
        Indices start at 1 instead of 0 and a negative value of the index means
        that a line should be in opposite orientation to how it is in the `lines`
        array to maintain a consistent surface orientation.
        """
        ...

    @classmethod
    def _create_elliptical_mesh(
        cls, arg1: list[_BlockInfoTuple], arg2: bool, arg3: _SolverCfgTuple, /
    ) -> tuple[Self, float, float]:
        """Create an elliptical mesh.

        This method takes in *heavily* pre-processed input. This is for the sake of
        making the parsing in C as simple as possible.

        Parameters
        ----------
        arg1 : list of _BlockInfoTuple
            List of tuples which contain information about mesh blocks.
        arg2 : bool
            Verbosity setting.
        arg3 : _SolverCfgTuple
            Tuple containing pre-processed solver config values.

        Returns
        -------
        Self
            The newly created mesh object.
        float
            Residual of the x-equation.
        float
            Residual of the y-equation.
        """
        ...
