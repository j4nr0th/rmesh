from .geometry import BoundaryBlock, BoundaryCurve, BoundaryId
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from rmsh._rmsh import create_elliptical_mesh as _cem


@dataclass(frozen=False)
class MeshBlock:
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
        prev = None
        match bid:
            case (BoundaryId.BoundaryNorth | BoundaryId.BoundarySouth
                  | BoundaryId.BoundaryEast | BoundaryId.BoundaryWest):
                prev = self.boundaries[bid]
                self.boundaries[bid] = b
            case _:
                raise RuntimeError("Invalid value of boundary id was specified")
        if prev is not None:
            raise RuntimeWarning(f"Boundary with id {bid.name} for block {self.label} was set, but was not None "
                                 f"previously (was {type(prev)} instead)")

    def has_all_boundaries(self) -> bool:
        return ((BoundaryId.BoundaryNorth in self.boundaries and self.boundaries[BoundaryId.BoundaryNorth] is not None)
                and (BoundaryId.BoundarySouth in self.boundaries and self.boundaries[BoundaryId.BoundarySouth] is not None)
                and (BoundaryId.BoundaryEast in self.boundaries and self.boundaries[BoundaryId.BoundaryEast] is not None)
                and (BoundaryId.BoundaryWest in self.boundaries and self.boundaries[BoundaryId.BoundaryWest] is not None))


def connect_mesh_blocks(b1: MeshBlock, id1: BoundaryId, b2: MeshBlock, id2: BoundaryId) -> None:
    b1.set_boundary(id1, BoundaryBlock(b2.label, id2))
    b2.set_boundary(id2, BoundaryBlock(b1.label, id1))


class Mesh2D:
    _internal = None

    def __init__(self, data):
        self._internal = data

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


def _find_boundary_size(bnd: BoundaryBlock, blcks: dict[str, tuple[int, MeshBlock]], first: BoundaryBlock|None) -> int:
    _, target = blcks[bnd.target]
    other_bnd = target.boundaries[bnd.target_id]
    if (first is bnd):
        raise RuntimeError("Circular reference for block boundaries without specifying their size")
    if type(other_bnd) is BoundaryBlock:
        match bnd.target_id:
            case BoundaryId.BoundaryNorth:
                other_bnd = target.boundaries[BoundaryId.BoundarySouth]
            case BoundaryId.BoundarySouth:
                other_bnd = target.boundaries[BoundaryId.BoundaryNorth]
            case BoundaryId.BoundaryWest:
                other_bnd = target.boundaries[BoundaryId.BoundaryEast]
            case BoundaryId.BoundaryEast:
                other_bnd = target.boundaries[BoundaryId.BoundaryWest]
            case _:
                raise RuntimeError("Invalid boundary type for the block boundary encountered")
    if type(other_bnd) is BoundaryCurve:
        return len(other_bnd.x)
    elif other_bnd.n != 0:
        return other_bnd.n
    return _find_boundary_size(other_bnd, blcks, first if first is not None else bnd)


def _curves_have_common_point(c1: BoundaryCurve, c2: BoundaryCurve) -> bool:
    p11 = (c1.x[0], c1.y[0])
    p12 = (c1.x[-1], c1.y[-1])
    p21 = (c2.x[0], c2.y[0])
    p22 = (c2.x[-1], c2.y[-1])
    return np.allclose(p11, p21) or np.allclose(p11, p22) or np.allclose(p12, p21) or np.allclose(p12, p22)


@dataclass
class SolverConfig:
    force_direct: bool = False
    tolerance: float = 1e-6
    smoother_rounds: int = 16
    max_iterations: int = 128
    max_rounds: int = 8


def create_elliptical_mesh(blocks: Sequence[MeshBlock], *, verbose: bool = False, allow_insane: bool = False,
                           solver_cfg: SolverConfig = SolverConfig()) -> tuple[Mesh2D, float, float]:
    #   Holds indices to which the blocks map to
    bdict = dict()
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
                    nbnd = _find_boundary_size(bnd, bdict, None)
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

    return Mesh2D(data), rx, ry
