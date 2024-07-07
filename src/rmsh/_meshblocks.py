from ._geometry import BoundaryBlock, BoundaryCurve, BoundaryId
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from ._rmsh import create_elliptical_mesh as _cem


@dataclass(frozen=False)
class MeshBlock:
    """Basic building block of the mesh, which describes a structured mesh block.

    Parameters
    ----------
    label : str
            Label by which this block will be referred to in the mesh, as well as by other blocks which have a block
            boundary to this block.
    boundaries : dict[BoundaryId, BoundaryCurve|BoundaryBlock]
        Dictionary containing the boundaries with their respective IDs. All four might be
        specified, or only a few.

    Attributes
    ----------
    label : str
            Label by which this block will be referred to in the mesh, as well as by other blocks which have a block
            boundary to this block.
    boundaries : dict[BoundaryId, BoundaryCurve|BoundaryBlock|None]
            Dictionary of boundaries, which has entries for the boundaries of the block. For a block to be complete,
            all four boundaries must be specified.
    """
    label: str
    boundaries: dict[BoundaryId, BoundaryCurve|BoundaryBlock|None]

    def __init__(self, label: str, boundaries: dict[BoundaryId, BoundaryCurve|BoundaryBlock]):
        self.label = label
        self.boundaries = dict()
        for k in boundaries:
            match k:
                case (BoundaryId.BoundaryNorth | BoundaryId.BoundarySouth
                      | BoundaryId.BoundaryEast | BoundaryId.BoundaryWest):
                    self.boundaries[k] = boundaries[k]
                case _:
                    raise RuntimeError(f"Boundary has a key of an invalid type ({type(k)}, should be {BoundaryId})")

    def set_boundary(self, bid: BoundaryId, b: BoundaryCurve | BoundaryBlock | None = None) -> None:
        """Sets a boundary of a block to a specified curve, block, or None. If a boundary that is being set is already
        set to a curve or a block and the new value is not None, a warning will be reported by the function.

        Parameters
        ----------
        bid : BoundaryId
              ID of the boundary to be set
        b : BoundaryCurve | BoundaryBlock | None = None
             New value of the boundary specified by `bid`.
        """
        prev = None
        match bid:
            case (BoundaryId.BoundaryNorth | BoundaryId.BoundarySouth
                  | BoundaryId.BoundaryEast | BoundaryId.BoundaryWest):
                prev = self.boundaries[bid]
                self.boundaries[bid] = b
            case _:
                raise RuntimeError("Invalid value of boundary id was specified")
        if prev is not None and b is not None:
            raise RuntimeWarning(f"Boundary with id {bid.name} for block {self.label} was set, but was not None "
                                 f"previously (was {type(prev)} instead)")

    def has_all_boundaries(self) -> bool:
        """Method, which just checks if a mesh block has all four boundaries specified as non-None values.

        Returns
        ----------
        bool
            True if the mesh block has all needed boundaries and False if that is not the case.
        """
        return ((BoundaryId.BoundaryNorth in self.boundaries and self.boundaries[BoundaryId.BoundaryNorth] is not None)
                and (BoundaryId.BoundarySouth in self.boundaries and self.boundaries[BoundaryId.BoundarySouth] is not None)
                and (BoundaryId.BoundaryEast in self.boundaries and self.boundaries[BoundaryId.BoundaryEast] is not None)
                and (BoundaryId.BoundaryWest in self.boundaries and self.boundaries[BoundaryId.BoundaryWest] is not None))


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


def _find_boundary_size(bnd: BoundaryBlock, blcks: dict[str, tuple[int, MeshBlock]]):
    if type(bnd) is BoundaryCurve:
        bnd: BoundaryCurve
        return len(bnd.x)
    checked = [bnd]
    i = 0
    while True:
        if type(bnd) == BoundaryCurve:
            return len(bnd.x)
        elif bnd.n != 0:
            return bnd.n
        _, target = blcks[bnd.target]
        other_bnd = target.boundaries[bnd.target_id]
    
        if type(other_bnd) is BoundaryBlock:
            match bnd.target_id:
                case BoundaryId.BoundaryNorth:
                    new_bnd = target.boundaries[BoundaryId.BoundarySouth]
                case BoundaryId.BoundarySouth:
                    new_bnd = target.boundaries[BoundaryId.BoundaryNorth]
                case BoundaryId.BoundaryWest:
                    new_bnd = target.boundaries[BoundaryId.BoundaryEast]
                case BoundaryId.BoundaryEast:
                    new_bnd = target.boundaries[BoundaryId.BoundaryWest]
                case _:
                    raise RuntimeError("Invalid boundary type for the block boundary encountered")
        else:
            return len(other_bnd.x)
        bnd = new_bnd
        if new_bnd in checked:
            break
        else:
            checked.append(new_bnd)
        i += 1
    raise RuntimeError(f"Circular reference for block boundaries without specifying their size")
    

def _curves_have_common_point(c1: BoundaryCurve, c2: BoundaryCurve) -> bool:
    p11 = (c1.x[0], c1.y[0])
    p12 = (c1.x[-1], c1.y[-1])
    p21 = (c2.x[0], c2.y[0])
    p22 = (c2.x[-1], c2.y[-1])
    return np.allclose(p11, p21) or np.allclose(p11, p22) or np.allclose(p12, p21) or np.allclose(p12, p22)


@dataclass
class SolverConfig:
    """
    Used to configure the solver used to solve the PDE for the positions of mesh nodes. Direct solver uses banded matrix
    representation to compute an LU decomposition for a matrix with a fixed bandwidth, which works well for small
    systems, where bandwidth is still small. After a specific mesh size, the solver will be changed to an iterative
    solver (Stabilized Bi-Conjugate Gradient Preconditioned with Incomplete Lower-Upper Decomposition - PILUBICG-STAB),
    which offers fast performance at a cost of loss of optimality guarantee. As such, it is necessary to use restarts
    after the Krylov subspace begins to degenerate.

    Parameters
    ----------
    force_direct : bool = False
        Force the direct solver to be used regardless of the problem size. This may allow for solution of some cases
        where the mesh is ill-defined or very close to ill-defined. Not recommended for other cases, since it is very
        slow.
    tolerance : float = 1e-6
        Tolerance used by the iterative solver to check for convergence. The solver will consider the iterative solution
        to be converged when `tolerance * norm(y) >= norm(r)`, where `norm(y)` is the L2 norm of the RHS of the equation
        being solved and `norm(r)` is the L2 norm of the residual. A lower value means a solution is more converged, but
        a value which is too low might be unable to give a solution.
    smoother_rounds : int = 0
        After each round of using an iterative solver, a Jacobi smoother can be used in order to `smooth` the
        intermediate iterative solution for up to `smoother_rounds`. Since the system matrix being solved is not
        strictly diagonally dominant, there is no guarantee on error reduction, but there is at least a guarantee on the
        error not being increased by this.
    max_iterations : int = 128
        How many iterations the iterative solver will perform each round at most, before restarting. Smaller values
        might cause the convergence to be very slow due discarding the entire Krylov subspace for that round. Very large
        values for `max_iterations` might also delay convergence, since Krylov subspace might degenerate and residual
        could start to grow again.
        If a solver detects that the Krylov subspace has degenerated, a round of iterative solver might end sooner.
    max_rounds : int = 8
        How many round of iterative solver to run at most. This means that the total number of rounds that PILUBICG-STAB
        could run is `max_rounds * max_iterations`, while the maximum number of smoother rounds is `max_rounds *
        smoother_rounds`
    """
    force_direct: bool = False
    tolerance: float = 1e-6
    smoother_rounds: int = 0
    max_iterations: int = 128
    max_rounds: int = 8


def create_elliptical_mesh(blocks: Sequence[MeshBlock], *, verbose: bool = False, allow_insane: bool = False,
                           solver_cfg: SolverConfig = SolverConfig()) -> tuple[Mesh2D, float, float]:
    """Creates a mesh from a list of mesh blocks by solving the Laplace equation for the coordinates of mesh nodes.

    Parameters
    ----------
    blocks : Sequence[MeshBlock]
        A sequence of blocks which constitute the mesh.
    verbose : bool = False
        When set to True the solver and function will print extra information to stdout related to solver progress. When
        False, only warnings and exceptions will be produced.
    allow_insane : bool = False
        Disable boundary connectivity checks. By default, these are enabled, causing an error to be raised if
        neighboring curve boundaries of the same block don't share a single point.
    solver_cfg : SolverConfig
        Contains additional options to configure the solver used to compute the mesh coordinates.

    Returns
    ----------
    Mesh2D
        Mesh generated from the provided blocks.
    float
        The L2 norm of the residual of the system solved for the X coordinate.
    float
        The L2 norm of the residual of the system solved for the Y coordinate.
    """
    bdict = dict() # Holds indices to which the blocks map to
    if verbose: print("Checking all blocks")
    for i, b in enumerate(blocks):
        if b.label in bdict:
            raise RuntimeError(f"Multiple blocks with the same label \"{b.label}\"")
        #   Check if boundaries are correctly set up
        if (not b.has_all_boundaries()) or (len(b.boundaries) != 4):
            raise RuntimeError(f"Block {b.label} does not have all boundaries defined (current: {b.boundaries})")
        #   Finally set the label
        bdict[b.label] = (i, b)
        if verbose: print(f"Block \"{b.label}\" was good")
    if verbose: print("Finished checking all blocks")

    if verbose: print("Checking all boundaries")
    #   Make sure that the boundaries are correctly set up
    for i, b in enumerate(blocks):
        bnd_lens = dict()
        for bid in b.boundaries:
            bnd = b.boundaries[bid]
            #   If boundary is BoundaryBlock, it should be sorted
            nbnd = 0
            if type(bnd) is BoundaryBlock:
                bnd: BoundaryBlock
                iother, other = bdict[bnd.target]
                bother = other.boundaries[bnd.target_id]
                if iother > i and type(bother) is BoundaryCurve:
                    #   Boundaries must be swapped
                    bother: BoundaryCurve
                    #   Flip the x and y arrays, since they should be in reverse order
                    b.boundaries[bid] = BoundaryCurve(np.flip(bother.x), np.flip(bother.y))
                    other.boundaries[bnd.target_id] = BoundaryBlock(b.label, bid)
                    nbnd = len(b.boundaries[bid].x)
                elif bnd.n != 0:
                    nbnd = bnd.n
                else:
                    nbnd = _find_boundary_size(bnd, bdict)
            #   Check that the corners of curve match up correctly if check is enabled
            elif not allow_insane:
                bnd: BoundaryCurve
                nbnd = len(bnd.x)
                bleft = None
                bright = None
                match bid:
                    case BoundaryId.BoundaryNorth:
                        bleft = BoundaryId.BoundaryEast
                        bright = BoundaryId.BoundaryWest
                    case BoundaryId.BoundaryWest:
                        bleft = BoundaryId.BoundaryNorth
                        bright = BoundaryId.BoundarySouth
                    case BoundaryId.BoundarySouth:
                        bleft = BoundaryId.BoundaryWest
                        bright = BoundaryId.BoundaryEast
                    case BoundaryId.BoundaryEast:
                        bleft = BoundaryId.BoundarySouth
                        bright = BoundaryId.BoundaryNorth
                bndleft = b.boundaries[bleft]
                bndright = b.boundaries[bright]
                if type(bndleft) is BoundaryCurve:
                    bndleft: BoundaryCurve
                    if not _curves_have_common_point(bnd, bndleft):
                        raise RuntimeWarning(f"Block {b.label} has curves as boundaries {bid.name} and {bleft.name}, but"
                                           f" they have no common points. To allow such meshes to be counted as valid, "
                                           f"call this function with \"allow_insane=True\"")
                if type(bndright) is BoundaryCurve:
                    bndright: BoundaryCurve
                    if not _curves_have_common_point(bnd, bndright):
                        raise RuntimeWarning(f"Block {b.label} has curves as boundaries {bid.name} and {bright.name}, but"
                                           f" they have no common points. To allow such meshes to be counted as valid, "
                                           f"call this function with \"allow_insane=True\"")
            bnd_lens[bid] = nbnd
            b.n_boundaries = bnd_lens

    for i, b in enumerate(blocks):
        nnorth = b.n_boundaries[BoundaryId.BoundaryNorth]
        nsouth = b.n_boundaries[BoundaryId.BoundarySouth]
        neast = b.n_boundaries[BoundaryId.BoundaryEast]
        nwest = b.n_boundaries[BoundaryId.BoundaryWest]
        if nnorth != nsouth:
            raise RuntimeError(f"Block {b.label} has {nnorth} points on the north boundary, but {nsouth} points on the"
                               f" south boundary")
        if neast != nwest:
            raise RuntimeError(f"Block {b.label} has {neast} points on the east boundary, but {nwest} points on the"
                               f" west boundary")

    #   Convert the input blocks into the form which is demanded by the C part of the code
    if verbose: print("Converting inputs to for usable by the C code")
    inputs = []
    for i, b in enumerate(blocks):
        boundaries = dict()
        for bid in b.boundaries:
            bnd = b.boundaries[bid]
            v = None
            if type(bnd) is BoundaryCurve:
                bnd: BoundaryCurve
                v = (0, bid.value, b.n_boundaries[bid], np.array(bnd.x, dtype=np.float64), np.array(bnd.y, dtype=np.float64))
            elif type(bnd) is BoundaryBlock:
                bnd: BoundaryBlock
                iother, other = bdict[bnd.target]
                v = (1, bid.value, b.n_boundaries[bid], iother, bnd.target_id.value)
            if v is None:
                raise RuntimeError(f"Boundary {bid.name} of block \"{b.label}\" was of invalid type f{type(bnd)}")
            boundaries[bid] = v
        bv = (b.label, boundaries[BoundaryId.BoundaryNorth], boundaries[BoundaryId.BoundarySouth],
              boundaries[BoundaryId.BoundaryEast], boundaries[BoundaryId.BoundaryWest])
        inputs.append(bv)
    extra = (solver_cfg.force_direct, solver_cfg.tolerance, solver_cfg.smoother_rounds, solver_cfg.max_iterations,
             solver_cfg.max_rounds)
    data, rx, ry = _cem(inputs, verbose, extra)
    name_map = dict()
    for k in bdict:
        name_map[k] = bdict[k][0]
    return Mesh2D(data, name_map), rx, ry
