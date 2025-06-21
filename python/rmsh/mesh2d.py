"""Implementation of the public Mesh2D type, which is a thin wrapper for the C type."""

from typing import Self

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from rmsh._rmsh import _Mesh2D
from rmsh.geometry import BoundaryId, _BlockInfoTuple, _SolverCfgTuple


class Mesh2D(_Mesh2D):
    """Class which contains the mesh information.

    It should not be created from its constructor, since it interfaces
    with the internal C extension.
    """

    _block_name_map: dict[str, int]

    def block_label_lines(self, block_id: str) -> npt.NDArray[np.int32]:
        """Return indices of all lines within a block.

        Indices start at 1 and a negative value indicates a reversed orientation of
        the line.

        Parameters
        ----------
        block_id : str
            The label of the block for which the line indices should be returned.

        Returns
        -------
        ndarray[int32]
            Array with indices of all lines within the mesh block specified by
            ``block_id``.
        """
        return super().block_lines(self._block_name_map[block_id])

    @property
    def block_names(self) -> list[str]:
        """Label strings of each block within the mesh.

        These are the only ones which are valid when referring to a block in the mesh.
        """
        return list(self._block_name_map.keys())

    def block_label_boundary_lines(
        self, block_id: str, boundary: BoundaryId
    ) -> npt.NDArray[np.int32]:
        """Return indices of all lines on a boundary of a block.

        Indices start at 1 and a negative value indicates a reversed orientation of the
        line.

        Parameters
        ----------
        block_id : str
            The label of the block for which the line indices should be returned.
        boundary : BoundaryId
            The ID of a boundary from which the line indices should be returned.

        Returns
        -------
        ndarray[int32]
            Array with indices of all lines on a boundary of a block ``block_id``.
        """
        return super().block_boundary_lines(
            self._block_name_map[block_id], boundary.value
        )

    def block_label_boundary_points(
        self, block_id: str, boundary: BoundaryId
    ) -> npt.NDArray[np.int32]:
        """Return indices of all nodes on a boundary of a block.

        Parameters
        ----------
        block_id : str
            The label of the block for which the point indices should be returned.
        boundary : BoundaryId
            The ID of a boundary from which the point indices should be returned.

        Returns
        -------
        ndarray[int32]
            Array with indices of all points on a boundary of a block ``block_id``.
        """
        return super().block_boundary_points(
            self._block_name_map[block_id], boundary.value
        )

    @classmethod
    def _create_elliptical_mesh_labeled(
        cls,
        arg1: list[_BlockInfoTuple],
        arg2: bool,
        arg3: _SolverCfgTuple,
        block_name_map: dict[str, int],
        /,
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
        self, rx, ry = super()._create_elliptical_mesh(arg1, arg2, arg3)
        self._block_name_map = block_name_map
        return (self, rx, ry)

    def plot(self, axes: Axes) -> tuple[tuple[float, float], tuple[float, float]]:
        """Create a plot to the given axes."""
        x = self.pos_x
        y = self.pos_y
        line_indices = self.lines
        xb = x[line_indices[:, 0]]
        xe = x[line_indices[:, 1]]
        yb = y[line_indices[:, 0]]
        ye = y[line_indices[:, 1]]

        rb = np.stack((xb, yb), axis=1)
        re = np.stack((xe, ye), axis=1)
        c = LineCollection(np.stack((rb, re), axis=1))
        axes.add_collection(c)

        return ((x.min(), x.max()), (y.min(), y.max()))
