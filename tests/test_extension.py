"""Test that the extension works on a square mesh of a single block."""

from typing import Callable

import numpy as np

# from matplotlib import collections as col
# from matplotlib import pyplot as plt
from rmsh import (
    BoundaryCurve,
    BoundaryId,
    MeshBlock,
    SolverConfig,
    create_elliptical_mesh,
)


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

    c0 = BoundaryCurve(
        x=interpolate(X0, X1, lambda t: t**2, NBND),
        y=interpolate(Y0, Y1, lambda t: t**2, NBND),
    )
    c1 = BoundaryCurve(
        x=interpolate(X1, X2, lambda t: t**2, NBND),
        y=interpolate(Y1, Y2, lambda t: t**2, NBND),
    )
    c2 = BoundaryCurve(
        x=interpolate(X2, X3, lambda t: t**2, NBND),
        y=interpolate(Y2, Y3, lambda t: t**2, NBND),
    )
    c3 = BoundaryCurve(
        x=interpolate(X3, X0, lambda t: t**2, NBND),
        y=interpolate(Y3, Y0, lambda t: t**2, NBND),
    )

    b1 = MeshBlock(
        "first",
        {
            BoundaryId.BoundarySouth: c0,
            BoundaryId.BoundaryEast: c1,
            BoundaryId.BoundaryNorth: c2,
            BoundaryId.BoundaryWest: c3,
        },
    )

    cfg = SolverConfig(tolerance=1e-5)
    m, ry, rx = create_elliptical_mesh([b1], verbose=False, solver_cfg=cfg)
    assert ry <= cfg.tolerance and rx <= cfg.tolerance

    # x = m.x
    # y = m.y
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
