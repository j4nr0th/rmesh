import numpy as np

from ._geometry import BoundaryId


class Mesh2D:
    """Class which contains the mesh information.

    It should not be created from its constructor, since it interfaces
    with the internal C implementation.

    Attributes
    ----------
    x : ndarray[float64]
        The X coordinates of the mesh nodes.
    y : ndarray[float64]
        The Y coordinates of the mesh nodes.
    lines : ndarray[int32]
        An array containing indices of nodes for each line, consequently has a shape `(N, 2)`, where `N` is the number
        of lines in the mesh.
    surfaces : ndarray[int32]
        An array containing indices of lines for each surface, consequently has a shape `(N, 4)`, where `N` is the
        number of surfaces in the mesh. Indices start at 1 instead of 0 and a negative value of the index means that
        a line should be in opposite orientation to how it is in the `lines` array to maintain a consistent surface
        orientation.
    block_names : list[str]
        A list containing label strings of each block within the mesh. These are the only ones which are valid when
        referring to a block in the mesh.
    """
    _internal = None
    _block_name_map = None

    def __init__(self, data, name_map):
        self._internal = data
        self._block_name_map = name_map

    @property
    def x(self) -> np.ndarray:
        return self._internal.x

    @property
    def y(self) -> np.ndarray:
        return self._internal.y

    @property
    def lines(self) -> np.ndarray:
        lidx = self._internal.l
        return np.reshape(lidx, (-1, 2))

    @property
    def surfaces(self) -> np.ndarray:
        sidx = self._internal.s
        return np.reshape(sidx, (-1, 4))

    def block_lines(self, block_id: str) -> np.ndarray:
        """Returns an array with indices of all lines within a block. Indices start at 1 and a negative value indicates
        a reversed orientation of the line.

        Parameters
        ----------
        block_id : str
            The label of the block for which the line indices should be returned.

        Returns
        ----------
        ndarray[int32]
            Array with indices of all lines within the mesh block specified by `block_id`.
        """
        a = self._internal.blines(self._block_name_map[block_id])
        return a

    @property
    def block_names(self) -> list[str]:
        return [s for s in self._block_name_map.keys()]

    def block_boundary_lines(self, block_id: str, boundary: BoundaryId) -> np.ndarray:
        """Returns an array with indices of all lines on a boundary of a block. Indices start at 1 and a negative value
        indicates a reversed orientation of the line.

        Parameters
        ----------
        block_id : str
            The label of the block for which the line indices should be returned.
        boundary : BoundaryId
            The ID of a boundary from which the line indices should be returned.

        Returns
        ----------
        ndarray[int32]
            Array with indices of all lines on a boundary of a block `block_id`.
        """
        a = self._internal.boundary_lines(self._block_name_map[block_id], boundary.value)
        return a

    def block_boundary_points(self, block_id: str, boundary: BoundaryId) -> np.ndarray:
        """Returns an array with indices of all nodes on a boundary of a block.

        Parameters
        ----------
        block_id : str
            The label of the block for which the point indices should be returned.
        boundary : BoundaryId
            The ID of a boundary from which the point indices should be returned.

        Returns
        ----------
        ndarray[int32]
            Array with indices of all points on a boundary of a block `block_id`.
        """
        a = self._internal.boundary_pts(self._block_name_map[block_id], boundary.value)
        return a

    def block_boundary_surfaces(self, block_id: str, boundary: BoundaryId) -> np.ndarray:
        """Returns an array with indices of all surfaces on a boundary of a block. Indices start at 1 and a negative value
        indicates a reversed orientation of the surface, though for this function this is not needed.

        Parameters
        ----------
        block_id : str
            The label of the block for which the surfaces indices should be returned.
        boundary : BoundaryId
            The ID of a boundary from which the surfaces indices should be returned.

        Returns
        ----------
        ndarray[int32]
            Array with indices of all surfaces on a boundary of a block `block_id`.
        """
        a = self._internal.boundary_surf(self._block_name_map[block_id], boundary.value)
        return a

    def surface_element(self, surf: int|np.ndarray, order: int) -> np.ndarray:
        """Returns the indices of surfaces, which form a square element of width (2 * order + 1) elements. This is
        intended to be used for computing cell-based interpolations.

        Parameters
        ----------
        surf : int
            The one-based index of the surface which should be the center of the element.
        order : int
            Size of the element in each direction away from the center (zero means only the element, one means 3 x 3,
             etc.)

        Returns
        ----------
        ndarray[int32]
            Array with indices of all surfaces in the element. Note that since one-based indexing is used, a zero
            indicates a missing surface caused by a numerical boundary. Negative indices mean a negative orientation.
        """
        return self._internal.surface_element(surf, order)

    def surface_element_points(self, surf: int|np.ndarray, order: int) -> np.ndarray:
        """Returns the indices of points, which form a square element of width (2 * order + 1) surfaces. This is
        intended to be used for computing nodal-based interpolations for surface elements.

        Parameters
        ----------
        surf : int
            The one-based index of the surface which should be the center of the element.
        order : int
            Size of the element in each direction away from the center (zero means only the element, one means 3 x 3,
             etc.)

        Returns
        ----------
        ndarray[int32]
            Array with indices of all indices in the element. Note that since one-based indexing is used, a value of -1
            indicates a missing point caused by a numerical boundary.
        """
        return self._internal.surface_element_points(surf, order)
