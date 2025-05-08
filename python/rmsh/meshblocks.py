"""Mesh block implementation and the function that pre-processes inputs for C code."""

from __future__ import annotations

#   Internal imports
from collections.abc import Sequence
from dataclasses import dataclass

#   External imports
import numpy as np

from rmsh.geometry import (
    Boundary,
    BoundaryBlock,
    BoundaryCurve,
    BoundaryId,
    BoundaryRef,
    MeshBlock,
    _BlockInfoTuple,
    _BoundaryInfoTuple,
)
from rmsh.mesh2d import Mesh2D


def _find_boundary_size(bnd: BoundaryBlock):
    if bnd.n != 0:
        return bnd.n

    checked: list[Boundary] = [bnd]
    i = 0
    boundary: BoundaryBlock = bnd
    while True:
        target = boundary.ref.block
        other_bnd = target.get_boundary_by_id_existing(boundary.ref.boundary)

        if other_bnd.n != 0:
            return other_bnd.n

        new_bnd = target.get_boundary_by_id_existing(
            boundary.ref.boundary.opposite_boundary
        )
        if new_bnd.n != 0:
            return new_bnd.n

        assert type(new_bnd) is BoundaryBlock

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

    if verbose:
        print("Checking all blocks")

    block_dict: dict[str, int] = dict()

    for i, b in enumerate(blocks):
        label = b.label
        if label is not None and label in block_dict:
            raise RuntimeError(f'Multiple blocks with the same label "{label}"')

        #   Check if boundaries are correctly set up
        if not b.has_all_boundaries():
            raise RuntimeError(
                f"Block {i} ({label=}) does not have all boundaries defined."
            )
        #   Finally set the label

        if label:
            block_dict[label] = i

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
        for bid in BoundaryId:
            #   If boundary is BoundaryBlock, it should be sorted
            bnd_lens[bid] = ensure_boundary_well_posed(allow_insane, i, b, bid)
        n_bnds.append(bnd_lens)

    for i, (b, nb) in enumerate(zip(blocks, n_bnds)):
        n_north = nb[BoundaryId.BoundaryTop]
        n_south = nb[BoundaryId.BoundaryBottom]
        n_east = nb[BoundaryId.BoundaryRight]
        n_west = nb[BoundaryId.BoundaryLeft]
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
        for bid in BoundaryId:
            bnd = b.get_boundary_by_id_existing(bid)
            if type(bnd) is BoundaryCurve:
                boundaries[bid] = (
                    0,
                    bid.value,
                    nb[bid],
                    np.array(bnd.x, dtype=np.float64),
                    np.array(bnd.y, dtype=np.float64),
                )
            elif type(bnd) is BoundaryBlock:
                # iother, _ = bnd.ref.boundary,
                boundaries[bid] = (
                    1,
                    bid.value,
                    nb[bid],
                    blocks.index(bnd.ref.block),
                    bnd.ref.boundary.value,
                )
            else:
                raise RuntimeError(
                    f'Boundary {bid.name} of block "{b.label}" was of invalid type'
                    f" {type(bnd)}"
                )
        label = b.label if b.label is not None else "_Unnamed_block_" + str(id(b))
        bv = (
            label,
            boundaries[BoundaryId.BoundaryTop],
            boundaries[BoundaryId.BoundaryBottom],
            boundaries[BoundaryId.BoundaryRight],
            boundaries[BoundaryId.BoundaryLeft],
        )
        inputs.append(bv)
    extra = (
        solver_cfg.force_direct,
        solver_cfg.tolerance,
        solver_cfg.smoother_rounds,
        solver_cfg.max_iterations,
        solver_cfg.max_rounds,
    )

    mesh, rx, ry = Mesh2D._create_elliptical_mesh_labeled(
        inputs, verbose, extra, block_dict
    )
    return mesh, rx, ry


def ensure_boundary_well_posed(
    allow_insane: bool,
    i: int,
    b: MeshBlock,
    bid: BoundaryId,
) -> int:
    """Make sure the boundary has well defined size and potentially correct ordering."""
    nbnd = 0
    bnd = b.get_boundary_by_id_existing(bid)
    if type(bnd) is BoundaryBlock:
        iother, other = bnd.ref.boundary, bnd.ref.block
        bother = other.get_boundary_by_id_existing(bnd.ref.boundary)
        if iother > i and type(bother) is BoundaryCurve:
            #   Boundaries must be swapped
            #   Flip the x and y arrays, since they should be in reverse order
            flipped = BoundaryCurve(np.flip(bother.x), np.flip(bother.y))

            b.set_boundary_by_id(bid, flipped)
            other.set_boundary_by_id(
                bnd.ref.boundary, BoundaryBlock(0, BoundaryRef(b, bid))
            )

            nbnd = flipped.n

        elif bnd.n != 0:
            nbnd = bnd.n
        else:
            nbnd = _find_boundary_size(bnd)
            #   Check that the corners of curve match up correctly if check is enabled
    elif not allow_insane:
        assert type(bnd) is BoundaryCurve
        nbnd = bnd.n
        bleft = None
        bright = None
        bleft = bid.prev
        bright = bid.next
        bndleft = b.get_boundary_by_id_existing(bleft)
        bndright = b.get_boundary_by_id_existing(bright)
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

    return nbnd
