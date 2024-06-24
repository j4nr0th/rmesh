from rmsh import meshblocks as mb
from rmsh import geometry as ge
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as col


if __name__ == "__main__":
    X0 = 0.2; X1 = 1.1; X2 = 0.76; X3 = -0.1
    Y0 = -0.2; Y1 = 0.1; Y2 = 0.98; Y3 = 1.1

    # X0 = 0; X1 = 1; X2 = 1; X3 = 0
    # Y0 = 0; Y1 = 0; Y2 = 1; Y3 = 1


    def interpolate(v1: float, v2: float, fn: Callable, n: int) -> np.ndarray:
        s = fn(np.linspace(0, 1, n))
        return np.array(v1 * (1 - s) + v2 * s)

    NBND = 400

    c0 = ge.BoundaryCurve(x=interpolate(X0, X1, lambda t: t**2, NBND), y=interpolate(Y0, Y1, lambda t: t**2, NBND))
    c1 = ge.BoundaryCurve(x=interpolate(X1, X2, lambda t: t**2, NBND), y=interpolate(Y1, Y2, lambda t: t**2, NBND))
    c2 = ge.BoundaryCurve(x=interpolate(X2, X3, lambda t: t**2, NBND), y=interpolate(Y2, Y3, lambda t: t**2, NBND))
    c3 = ge.BoundaryCurve(x=interpolate(X3, X0, lambda t: t**2, NBND), y=interpolate(Y3, Y0, lambda t: t**2, NBND))

    b1 = mb.MeshBlock("first",
                      {
                          ge.BoundaryId.BoundarySouth: c0,
                          ge.BoundaryId.BoundaryEast: c1,
                          ge.BoundaryId.BoundaryNorth: c2,
                          ge.BoundaryId.BoundaryWest: c3
                      })

    m = mb.create_elliptical_mesh([b1], verbose=True)

    x = m.x
    y = m.y
    ln = m.lines
    # lnvals = []
    xb = x[ln[:, 0]]
    yb = y[ln[:, 0]]
    xe = x[ln[:, 1]]
    ye = y[ln[:, 1]]

    rb = np.stack((xb, yb), axis=1)
    re = np.stack((xe, ye), axis=1)
    # for i in range(ln.shape[0]):
    #     lnvals.append((rb[i, :], re[i, :]))#[(xb[i], yb[i]), (xe[i], ye[i])])
    lnvals = np.stack((rb, re), axis=1)

    plt.gca().add_collection(col.LineCollection(lnvals, linestyle="dashed", color="black"))
    plt.scatter(x, y)
    #     plt.plot((x[ln[i, 0]], x[ln[i, 1]]), (y[ln[i, 0]], y[ln[i, 1]]), color="black", linestyle="dashed")

    plt.gca().set_aspect("equal")
    plt.show()

