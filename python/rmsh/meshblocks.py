"""Mesh block implementation and the function that pre-processes inputs for C code."""

#   Internal imports
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#   External imports
import numpy as np

from rmsh.geometry import (
    BoundaryBlock,
    BoundaryCurve,
    BoundaryId,
    _BlockInfoTuple,
    _BoundaryInfoTuple,
)
from rmsh.mesh2d import Mesh2D


@dataclass(frozen=True)
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

    Attributes
    ----------
    label : str
            Label by which this block will be referred to in the mesh, as well as by other
            blocks which have a block
            boundary to this block.
    boundaries : dict[BoundaryId, BoundaryCurve|BoundaryBlock|None]
            Dictionary of boundaries, which has entries for the boundaries of the block.
            For a block to be complete, all four boundaries must be specified.
    """

    label: str
    boundaries: dict[BoundaryId, BoundaryCurve | BoundaryBlock | None]

    def __init__(
        self, label: str, boundaries: Mapping[BoundaryId, BoundaryCurve | BoundaryBlock]
    ):
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "boundaries", dict())
        for k in boundaries:
            match k:
                case (
                    BoundaryId.BoundaryNorth
                    | BoundaryId.BoundarySouth
                    | BoundaryId.BoundaryEast
                    | BoundaryId.BoundaryWest
                ):
                    self.boundaries[k] = boundaries[k]
                case _:
                    raise RuntimeError(
                        f"Boundary has a key of an invalid type ({type(k)}, should be"
                        f" {BoundaryId})"
                    )

    def set_boundary(
        self, bid: BoundaryId, b: BoundaryCurve | BoundaryBlock | None = None
    ) -> None:
        """Set a boundary of a block to a specified curve, block, or None.

        If a boundary that is being set is already set to a curve or a block and the new
        value is not None, a warning will be reported by the function.

        Parameters
        ----------
        bid : BoundaryId
              ID of the boundary to be set
        b : BoundaryCurve | BoundaryBlock | None = None
             New value of the boundary specified by `bid`.
        """
        prev = None
        match bid:
            case (
                BoundaryId.BoundaryNorth
                | BoundaryId.BoundarySouth
                | BoundaryId.BoundaryEast
                | BoundaryId.BoundaryWest
            ):
                prev = self.boundaries[bid]
                self.boundaries[bid] = b
            case _:
                raise RuntimeError("Invalid value of boundary id was specified")
        if prev is not None and b is not None:
            raise RuntimeWarning(
                f"Boundary with id {bid.name} for block {self.label} was set, but was "
                f"not None previously (was {type(prev)} instead)"
            )

    def has_all_boundaries(self) -> bool:
        """Check if a mesh block has all four boundaries specified as non-None values.

        Returns
        -------
        bool
            True if the mesh block has all needed boundaries and False if that is not the
            case.
        """
        return (
            (
                BoundaryId.BoundaryNorth in self.boundaries
                and self.boundaries[BoundaryId.BoundaryNorth] is not None
            )
            and (
                BoundaryId.BoundarySouth in self.boundaries
                and self.boundaries[BoundaryId.BoundarySouth] is not None
            )
            and (
                BoundaryId.BoundaryEast in self.boundaries
                and self.boundaries[BoundaryId.BoundaryEast] is not None
            )
            and (
                BoundaryId.BoundaryWest in self.boundaries
                and self.boundaries[BoundaryId.BoundaryWest] is not None
            )
        )


def _find_boundary_size(bnd: BoundaryBlock, blcks: dict[str, tuple[int, MeshBlock]]):
    # if type(bnd) is BoundaryCurve:
    #     bnd: BoundaryCurve
    #     return len(bnd.x)
    checked: list[BoundaryBlock | BoundaryCurve] = [bnd]
    i = 0
    boundary: BoundaryBlock | BoundaryCurve = bnd
    while True:
        if type(boundary) is BoundaryCurve:
            return len(boundary.x)
        assert type(boundary) is BoundaryBlock

        if boundary.n != 0:
            return boundary.n

        _, target = blcks[boundary.target]
        other_bnd = target.boundaries[boundary.target_id]
        assert other_bnd is not None
        if type(other_bnd) is BoundaryCurve:
            return len(other_bnd.x)

        # if type(other_bnd) is BoundaryBlock:
        match boundary.target_id:
            case BoundaryId.BoundaryNorth:
                new_bnd = target.boundaries[BoundaryId.BoundarySouth]
            case BoundaryId.BoundarySouth:
                new_bnd = target.boundaries[BoundaryId.BoundaryNorth]
            case BoundaryId.BoundaryWest:
                new_bnd = target.boundaries[BoundaryId.BoundaryEast]
            case BoundaryId.BoundaryEast:
                new_bnd = target.boundaries[BoundaryId.BoundaryWest]
            case _:
                raise RuntimeError(
                    "Invalid boundary type for the block boundary encountered"
                )
        assert new_bnd is not None
        boundary = new_bnd
        if new_bnd in checked:
            break
        else:
            checked.append(new_bnd)
        i += 1
    raise RuntimeError(
        "Circular reference for block boundaries without specifying their size"
    )


def _curves_have_common_point(c1: BoundaryCurve, c2: BoundaryCurve) -> bool:
    p11 = (c1.x[0], c1.y[0])
    p12 = (c1.x[-1], c1.y[-1])
    p21 = (c2.x[0], c2.y[0])
    p22 = (c2.x[-1], c2.y[-1])
    return (
        np.allclose(p11, p21)
        or np.allclose(p11, p22)
        or np.allclose(p12, p21)
        or np.allclose(p12, p22)
    )


@dataclass
class SolverConfig:
    """Used to configure the solver used to solve the PDE for the positions of mesh nodes.

    Direct solver uses banded matrix
    representation to compute an LU decomposition for a matrix with a fixed bandwidth,
    which works well for small systems, where bandwidth is still small. After a specific
    mesh size, the solver will be changed to an iterative solver (Stabilized Bi-Conjugate
    Gradient Preconditioned with Incomplete Lower-Upper Decomposition - PILUBICG-STAB),
    which offers fast performance at a cost of loss of optimality guarantee. As such, it
    is necessary to use restarts after the Krylov subspace begins to degenerate.

    Parameters
    ----------
    force_direct : bool, default: False
        Force the direct solver to be used regardless of the problem size. This may allow
        for solution of some cases where the mesh is ill-defined or very close to
        ill-defined. Not recommended for other cases, since it is very slow.
    tolerance : float, default: 1e-6
        Tolerance used by the iterative solver to check for convergence. The solver will
        consider the iterative solution to be converged when
        ``tolerance * norm(y) >= norm(r)``, where ``norm(y)`` is the L2 norm of the RHS of
        the equation being solved and ``norm(r)`` is the L2 norm of the residual. A lower
        value means a solution is more converged, but a value which is too low might be
        unable to give a solution.
    smoother_rounds : int, default: 0
        After each round of using an iterative solver, a Jacobi smoother can be used in
        order to "smooth" the intermediate iterative solution for up to
        ``smoother_rounds``. Since the system matrix being solved is not strictly
        diagonally dominant, there is no guarantee on error reduction, but there is at
        least a guarantee on the error not being increased by this.
    max_iterations : int, default: 128
        How many iterations the iterative solver will perform each round at most, before
        restarting. Smaller values might cause the convergence to be very slow due
        discarding the entire Krylov subspace for that round. Very large values for
        ``max_iterations`` might also delay convergence, since Krylov subspace might
        degenerate and residual could start to grow again.
        If a solver detects that the Krylov subspace has degenerated, a round of iterative
        solver might end sooner.
    max_rounds : int, default: 8
        How many round of iterative solver to run at most. This means that the total
        number of rounds that PILUBICG-STAB could run is ``max_rounds * max_iterations``,
        while the maximum number of smoother rounds is ``max_rounds * smoother_rounds``.
    """

    force_direct: bool = False
    tolerance: float = 1e-6
    smoother_rounds: int = 0
    max_iterations: int = 128
    max_rounds: int = 8


def create_elliptical_mesh(
    blocks: Sequence[MeshBlock],
    verbose: bool = False,
    allow_insane: bool = False,
    solver_cfg: SolverConfig | None = None,
) -> tuple[Mesh2D, float, float]:
    """Create a mesh from mesh blocks by solving the Laplace equation for mesh nodes.

    Parameters
    ----------
    blocks : Sequence of MeshBlock
        A sequence of blocks which constitute the mesh.
    verbose : bool, default: False
        When set to True the solver and function will print extra information to stdout
        related to solver progress. When False, only warnings and exceptions will be
        produced.
    allow_insane : bool, default: False
        Disable boundary connectivity checks. By default, these are enabled, causing an
        error to be raised if neighboring curve boundaries of the same block don't share
        a single point.
    solver_cfg : SolverConfig, optional
        Contains additional options to configure the solver used to compute the mesh
        coordinates.

    Returns
    -------
    Mesh2D
        Mesh generated from the provided blocks.
    float
        The L2 norm of the residual of the system solved for the X coordinate.
    float
        The L2 norm of the residual of the system solved for the Y coordinate.
    """
    if solver_cfg is None:
        solver_cfg = SolverConfig(
            force_direct=False,
            tolerance=1e-6,
            smoother_rounds=0,
            max_iterations=128,
            max_rounds=8,
        )
    bdict: dict[str, tuple[int, MeshBlock]] = (
        dict()
    )  # Holds indices to which the blocks map to
    if verbose:
        print("Checking all blocks")
    for i, b in enumerate(blocks):
        if b.label in bdict:
            raise RuntimeError(f'Multiple blocks with the same label "{b.label}"')
        #   Check if boundaries are correctly set up
        if (not b.has_all_boundaries()) or (len(b.boundaries) != 4):
            raise RuntimeError(
                f"Block {b.label} does not have all boundaries defined"
                f" (current: {b.boundaries})"
            )
        #   Finally set the label
        bdict[b.label] = (i, b)
        if verbose:
            print(f'Block "{b.label}" was good')
    if verbose:
        print("Finished checking all blocks")

    if verbose:
        print("Checking all boundaries")
    #   Make sure that the boundaries are correctly set up
    n_bnds: list[dict[BoundaryId, int]] = []
    for i, b in enumerate(blocks):
        bnd_lens: dict[BoundaryId, int] = dict()
        for bid in b.boundaries:
            bnd = b.boundaries[bid]
            #   If boundary is BoundaryBlock, it should be sorted
            nbnd = 0
            if type(bnd) is BoundaryBlock:
                iother, other = bdict[bnd.target]
                bother = other.boundaries[bnd.target_id]
                if iother > i and type(bother) is BoundaryCurve:
                    #   Boundaries must be swapped
                    #   Flip the x and y arrays, since they should be in reverse order
                    flipped = BoundaryCurve(np.flip(bother.x), np.flip(bother.y))

                    b.boundaries[bid] = flipped
                    other.boundaries[bnd.target_id] = BoundaryBlock(b.label, bid)
                    nbnd = len(flipped.x)
                elif bnd.n != 0:
                    nbnd = bnd.n
                else:
                    nbnd = _find_boundary_size(bnd, bdict)
            #   Check that the corners of curve match up correctly if check is enabled
            elif not allow_insane:
                assert type(bnd) is BoundaryCurve
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
                    if not _curves_have_common_point(bnd, bndleft):
                        raise RuntimeWarning(
                            f"Block {b.label} has curves as boundaries {bid.name} and"
                            f" {bleft.name}, but they have no common points. To allow"
                            " such meshes to be counted as valid, "
                            'call this function with "allow_insane=True"'
                        )
                if type(bndright) is BoundaryCurve:
                    if not _curves_have_common_point(bnd, bndright):
                        raise RuntimeWarning(
                            f"Block {b.label} has curves as boundaries {bid.name} and"
                            f" {bright.name}, but they have no common points. To allow"
                            " such meshes to be counted as valid, "
                            'call this function with "allow_insane=True"'
                        )
            bnd_lens[bid] = nbnd
        n_bnds.append(bnd_lens)

    for i, (b, nb) in enumerate(zip(blocks, n_bnds)):
        n_north = nb[BoundaryId.BoundaryNorth]
        n_south = nb[BoundaryId.BoundarySouth]
        n_east = nb[BoundaryId.BoundaryEast]
        n_west = nb[BoundaryId.BoundaryWest]
        if n_north != n_south:
            raise RuntimeError(
                f"Block {b.label} has {n_north} points on the north boundary, but"
                f" {n_south} points on the south boundary"
            )
        if n_east != n_west:
            raise RuntimeError(
                f"Block {b.label} has {n_east} points on the east boundary, but {n_west}"
                " points on the west boundary"
            )

    #   Convert the input blocks into the form which is demanded by the C part of the code
    if verbose:
        print("Converting inputs to for usable by the C code")
    inputs: list[_BlockInfoTuple] = []
    for i, (b, nb) in enumerate(zip(blocks, n_bnds)):
        boundaries: dict[BoundaryId, _BoundaryInfoTuple] = dict()
        for bid in b.boundaries:
            bnd = b.boundaries[bid]
            if type(bnd) is BoundaryCurve:
                boundaries[bid] = (
                    0,
                    bid.value,
                    nb[bid],
                    np.array(bnd.x, dtype=np.float64),
                    np.array(bnd.y, dtype=np.float64),
                )
            elif type(bnd) is BoundaryBlock:
                iother, other = bdict[bnd.target]
                boundaries[bid] = (1, bid.value, nb[bid], iother, bnd.target_id.value)
            else:
                raise RuntimeError(
                    f'Boundary {bid.name} of block "{b.label}" was of invalid type'
                    f" {type(bnd)}"
                )
        bv = (
            b.label,
            boundaries[BoundaryId.BoundaryNorth],
            boundaries[BoundaryId.BoundarySouth],
            boundaries[BoundaryId.BoundaryEast],
            boundaries[BoundaryId.BoundaryWest],
        )
        inputs.append(bv)
    extra = (
        solver_cfg.force_direct,
        solver_cfg.tolerance,
        solver_cfg.smoother_rounds,
        solver_cfg.max_iterations,
        solver_cfg.max_rounds,
    )
    name_map: dict[str, int] = dict()
    for k in bdict:
        name_map[k] = bdict[k][0]

    mesh, rx, ry = Mesh2D._create_elliptical_mesh_labeled(
        inputs, verbose, extra, name_map
    )
    return mesh, rx, ry
