"""Stub detailing the C-extension types used internally."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rmsh.geometry import _BlockInfoTuple, _SolverCfgTuple

class _Mesh:
    """Internal mesh interface."""

    def blines(self, idx: int, /) -> npt.NDArray[np.int32]:
        """Retrieve indices of lines which are in a block with the given index."""
        ...

    def boundary_lines(self, block_id: int, boundary_id: int, /) -> npt.NDArray[np.int32]:
        """Retrieve line indices for a specified boundary of a block."""
        ...

    def boundary_pts(self, block_id: int, boundary_id: int, /) -> npt.NDArray[np.int32]:
        """Retrieve point indices for a specified boundary of a block."""
        ...

    def boundary_surf(self, block_id: int, boundary_id: int, /) -> npt.NDArray[np.int32]:
        """Retrieve surface indices for a specified boundary of a block."""
        ...

    def surface_element(self, surface_id: int, order: int, /) -> npt.NDArray[np.int32]:
        """Return indices of surfaces for a surface element."""
        ...

    def surface_element_points(
        self, surface_id: int, order: int, /
    ) -> npt.NDArray[np.int32]:
        """Return indices of indices for a surface element."""
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
    def line_indices(self) -> npt.NDArray[np.int32]:
        """Line indices."""
        ...

    @property
    def surface_indices(self) -> npt.NDArray[np.int32]:
        """Surface indices."""
        ...

    @classmethod
    def create_elliptical_mesh(
        cls, arg1: list[_BlockInfoTuple], arg2: bool, arg3: _SolverCfgTuple, /
    ) -> tuple[_Mesh, float, float]:
        """Create an elliptical mesh."""
        ...
