"""Test that the extension works on a square mesh of a single block."""

from typing import Callable

import numpy as np

# from matplotlib import collections as col
# from matplotlib import pyplot as plt
from rmsh import (
    BoundaryCurve,
    SolverConfig,
    create_elliptical_mesh,
)
from rmsh.geometry import MeshBlock


def test_single_square():
    """Test on a single square block."""
    X0 = 0.2
    X1 = 1.1
    X2 = 0.76
    X3 = -0.1
    Y0 = -0.2
    Y1 = 0.1
    Y2 = 0.98
    Y3 = 1.1

    # X0 = 0; X1 = 1; X2 = 1; X3 = 0
    # Y0 = 0; Y1 = 0; Y2 = 1; Y3 = 1

    def interpolate(v1: float, v2: float, fn: Callable, n: int) -> np.ndarray:
        """Interpolate based on a power law."""
        s = fn(np.linspace(0, 1, n))
        return np.array(v1 * (1 - s) + v2 * s)

    NBND = 30

    b1 = MeshBlock(
        "first",
        # bottom=BoundaryCurve.from_line((X0, Y0), (X1, Y1), NBND, lambda t: t**2),
        bottom=BoundaryCurve.from_knots(
            NBND, (X0, Y0), (0.5, -1), (X1, Y1), distribution=lambda t: t**2
        ),
        right=BoundaryCurve.from_line((X1, Y1), (X2, Y2), NBND, lambda t: t**2),
        top=BoundaryCurve.from_line((X2, Y2), (X3, Y3), NBND, lambda t: t**2),
        left=BoundaryCurve.from_line((X3, Y3), (X0, Y0), NBND, lambda t: t**2),
    )

    cfg = SolverConfig(tolerance=1e-5)
    m, ry, rx = create_elliptical_mesh([b1], verbose=False, solver_cfg=cfg)
    assert ry <= cfg.tolerance and rx <= cfg.tolerance

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
    # plt.scatter(x, y)
    # #     plt.plot((x[ln[i, 0]], x[ln[i, 1]]), (y[ln[i, 0]], y[ln[i, 1]]),
    # #       color="black", linestyle="dashed")

    # plt.gca().set_aspect("equal")
    # plt.show()
