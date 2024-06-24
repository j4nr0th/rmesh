import time

from rmsh import meshblocks as mb
from rmsh import geometry as ge
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as col



if __name__ == "__main__":
    N1 = 80; N2 = 70; NR = 60
    angle1 = np.linspace(0, np.pi/2, N1)
    angle2 = np.linspace(np.pi/2, np.pi, N2)
    angle3 = np.linspace(np.pi, 3*np.pi/2, N1)
    angle4 = np.linspace(3*np.pi/2, 2*np.pi, N2)

    bnd1 = mb.BoundaryCurve(np.cos(angle1), np.sin(angle1))
    bnd2 = mb.BoundaryCurve(np.cos(angle2), np.sin(angle2))
    bnd3 = mb.BoundaryCurve(np.cos(angle3), np.sin(angle3))
    bnd4 = mb.BoundaryCurve(np.cos(angle4), np.sin(angle4))

    b1 = mb.MeshBlock("right",
                      {
                          ge.BoundaryId.BoundaryEast: bnd1,
                          ge.BoundaryId.BoundaryNorth: ge.BoundaryBlock("top", ge.BoundaryId.BoundaryEast, NR),
                          ge.BoundaryId.BoundarySouth: ge.BoundaryBlock("btm", ge.BoundaryId.BoundaryEast, NR),
                          ge.BoundaryId.BoundaryWest: ge.BoundaryBlock("center", ge.BoundaryId.BoundaryEast)
                               })

    b3 = mb.MeshBlock("left",
                      {
                          ge.BoundaryId.BoundaryWest: bnd3,
                          ge.BoundaryId.BoundaryNorth: ge.BoundaryBlock("top", ge.BoundaryId.BoundaryWest),
                          ge.BoundaryId.BoundarySouth: ge.BoundaryBlock("btm", ge.BoundaryId.BoundaryWest),
                          ge.BoundaryId.BoundaryEast: ge.BoundaryBlock("center", ge.BoundaryId.BoundaryWest)
                      })

    b2 = mb.MeshBlock("top",
                      {
                          ge.BoundaryId.BoundaryWest: ge.BoundaryBlock("left", ge.BoundaryId.BoundaryNorth),
                          ge.BoundaryId.BoundaryNorth: bnd2,
                          ge.BoundaryId.BoundarySouth: ge.BoundaryBlock("center", ge.BoundaryId.BoundaryNorth),
                          ge.BoundaryId.BoundaryEast: ge.BoundaryBlock("right", ge.BoundaryId.BoundaryNorth)
                      })

    b4 = mb.MeshBlock("btm",
                      {
                          ge.BoundaryId.BoundaryWest: ge.BoundaryBlock("left", ge.BoundaryId.BoundarySouth),
                          ge.BoundaryId.BoundaryNorth: ge.BoundaryBlock("center", ge.BoundaryId.BoundarySouth),
                          ge.BoundaryId.BoundarySouth: bnd4,
                          ge.BoundaryId.BoundaryEast: ge.BoundaryBlock("right", ge.BoundaryId.BoundarySouth)
                      })

    b0 = mb.MeshBlock("center",
                      {
                          ge.BoundaryId.BoundaryWest: ge.BoundaryBlock("left", ge.BoundaryId.BoundaryEast),
                          ge.BoundaryId.BoundaryEast: ge.BoundaryBlock("right", ge.BoundaryId.BoundaryWest),
                          ge.BoundaryId.BoundaryNorth: ge.BoundaryBlock("top", ge.BoundaryId.BoundarySouth),
                          ge.BoundaryId.BoundarySouth: ge.BoundaryBlock("btm", ge.BoundaryId.BoundaryNorth),
                      })
    t0 = time.time_ns()
    m = mb.create_elliptical_mesh([b0, b1, b2, b3, b4], verbose=True)
    t1 = time.time_ns()

    print("Time taken for the mesh generation:", (t1 - t0)/1e9, "seconds")

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

