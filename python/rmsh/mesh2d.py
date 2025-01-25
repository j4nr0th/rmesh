"""Implementation of the public Mesh2D type, which is a thin wrapper for the C type."""

import numpy as np
import numpy.typing as npt

from rmsh._rmsh import _Mesh
from rmsh.geometry import BoundaryId


# TODO: this could just be a sub-class
class Mesh2D:
    """Class which contains the mesh information.

    It should not be created from its constructor, since it interfaces
    with the internal C extension.
    """

    _internal: _Mesh
    _block_name_map: dict[str, int]

    def __init__(self, data: _Mesh, name_map: dict[str, int], /):
        self._internal = data
        self._block_name_map = name_map

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """The X coordinates of the mesh nodes."""
        return self._internal.pos_x

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """The Y coordinates of the mesh nodes."""
        return self._internal.pos_y

    @property
    def lines(self) -> npt.NDArray[np.int32]:
        """Indices of nodes for each line.

        Has a shape ``(N, 2)``, where ``N`` is the number of lines in the mesh.
        """
        lidx = self._internal.line_indices
        return np.reshape(lidx, (-1, 2))

    @property
    def surfaces(self) -> npt.NDArray[np.int32]:
        """Indices of lines for each surface.

        Has a shape ``(N, 4)``, where `N` is the number of surfaces in the mesh.
        Indices start at 1 instead of 0 and a negative value of the index means
        that a line should be in opposite orientation to how it is in the `lines`
        array to maintain a consistent surface orientation.
        """
        sidx = self._internal.surface_indices
        return np.reshape(sidx, (-1, 4))

    def block_lines(self, block_id: str) -> npt.NDArray[np.int32]:
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
            Array with indices of all lines within the mesh block specified by `block_id`.
        """
        a = self._internal.blines(self._block_name_map[block_id])
        return a

    @property
    def block_names(self) -> list[str]:
        """Label strings of each block within the mesh.

        These are the only ones which are valid when referring to a block in the mesh.
        """
        return list(self._block_name_map.keys())

    def block_boundary_lines(
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
            Array with indices of all lines on a boundary of a block `block_id`.
        """
        a = self._internal.boundary_lines(self._block_name_map[block_id], boundary.value)
        return a

    def block_boundary_points(
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
            Array with indices of all points on a boundary of a block `block_id`.
        """
        a = self._internal.boundary_pts(self._block_name_map[block_id], boundary.value)
        return a

    def block_boundary_surfaces(
        self, block_id: str, boundary: BoundaryId
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
            Array with indices of all surfaces on a boundary of a block `block_id`.
        """
        a = self._internal.boundary_surf(self._block_name_map[block_id], boundary.value)
        return a

    def surface_element(self, surf: int, order: int) -> npt.NDArray[np.int32]:
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
        return self._internal.surface_element(surf, order)

    def surface_element_points(self, surf: int, order: int) -> npt.NDArray[np.int32]:
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
        return self._internal.surface_element_points(surf, order)
