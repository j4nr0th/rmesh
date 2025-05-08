"""Create a "circular" mesh."""

# import time

import numpy as np
import numpy.typing as npt

# from matplotlib import collections as col
# from matplotlib import pyplot as plt
from rmsh import (
    BoundaryCurve,
    SolverConfig,
    create_elliptical_mesh,
)
from rmsh.geometry import MeshBlock


def test_circular():
    """Create the circular mesh to check if it works."""
    N1 = 80
    N2 = 70
    NR = 70

    bc = MeshBlock("center")
    bb = MeshBlock("bottom")
    br = MeshBlock("right")
    bt = MeshBlock("top")
    bl = MeshBlock("left")

    def bnd_function(theta: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Define the boundary with."""
        return np.stack((np.cos(theta), np.sin(theta)), axis=1, dtype=np.float64)

    br.right = BoundaryCurve.from_samples(bnd_function, N1, lambda x: (x + 1) * np.pi / 2)
    br.left = bc.bbnd_right()
    br.top = bt.bbnd_right(NR)
    br.bottom = bb.bbnd_right()

    bl.right = bc.bbnd_left()
    bl.left = BoundaryCurve.from_samples(bnd_function, N1, lambda x: (x + 3) * np.pi / 2)
    bl.top = bt.bbnd_left()
    bl.bottom = bb.bbnd_left()

    bt.right = br.bbnd_top()
    bt.left = bl.bbnd_top()
    bt.top = BoundaryCurve.from_samples(bnd_function, N2, lambda x: (x + 2) * np.pi / 2)
    bt.bottom = bc.bbnd_top()

    bb.right = br.bbnd_bottom()
    bb.left = bl.bbnd_bottom()
    bb.top = bc.bbnd_bottom()
    bb.bottom = BoundaryCurve.from_samples(
        bnd_function, N2, lambda x: (x + 0) * np.pi / 2
    )

    bc.right = br.bbnd_left()
    bc.top = bt.bbnd_bottom()
    bc.left = bl.bbnd_right()
    bc.bottom = bb.bbnd_top()

    # t0 = time.time_ns()
    cfg = SolverConfig(tolerance=1e-5, max_iterations=512, smoother_rounds=0)
    m, rx, ry = create_elliptical_mesh(
        [bc, br, bt, bl, bb], verbose=False, solver_cfg=cfg
    )
    # del m
    assert rx <= cfg.tolerance and ry <= cfg.tolerance
    # t1 = time.time_ns()

    # print("Time taken for the mesh generation:", (t1 - t0) / 1e9, "seconds")
    # print("Residuals:", rx, ry)

    # x = m.pos_x
    # y = m.pos_y
    # ln = m.lines
    # # lnvals = []
    # xb = x[ln[:, 0]]
    # yb = y[ln[:, 0]]
    # xe = x[ln[:, 1]]
    # ye = y[ln[:, 1]]

    # rb = np.stack((xb, yb), axis=1)
    # re = np.stack((xe, ye), axis=1)
    # # for i in range(ln.shape[0]):
    # #     lnvals.append((rb[i, :], re[i, :]))#[(xb[i], yb[i]), (xe[i], ye[i])])
    # lnvals = np.stack((rb, re), axis=1)

    # plt.gca().add_collection(
    #     col.LineCollection(lnvals, linestyle="dashed", color="black")
    # )
    # # plt.scatter(x, y)
    # #     plt.plot((x[ln[i, 0]], x[ln[i, 1]]), (y[ln[i, 0]], y[ln[i, 1]]),
    # # color="black", linestyle="dashed")
    # plt.xlim(-1, +1)
    # plt.ylim(-1, +1)
    # plt.gca().set_aspect("equal")
    # plt.show()
