import time

from rmsh import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as col



if __name__ == "__main__":
    N1 = 80; N2 = 70; NR = 70
    angle1 = np.linspace(0, np.pi/2, N1)
    angle2 = np.linspace(np.pi/2, np.pi, N2)
    angle3 = np.linspace(np.pi, 3*np.pi/2, N1)
    angle4 = np.linspace(3*np.pi/2, 2*np.pi, N2)

    bnd1 = BoundaryCurve(np.cos(angle1), np.sin(angle1))
    bnd2 = BoundaryCurve(np.cos(angle2), np.sin(angle2))
    bnd3 = BoundaryCurve(np.cos(angle3), np.sin(angle3))
    bnd4 = BoundaryCurve(np.cos(angle4), np.sin(angle4))

    b1 = MeshBlock("right",
                      {
                          BoundaryId.BoundaryEast: bnd1,
                          BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundaryEast, NR),
                          BoundaryId.BoundarySouth: BoundaryBlock("btm", BoundaryId.BoundaryEast, NR),
                          BoundaryId.BoundaryWest: BoundaryBlock("center", BoundaryId.BoundaryEast)
                               })

    b3 = MeshBlock("left",
                      {
                          BoundaryId.BoundaryWest: bnd3,
                          BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundaryWest),
                          BoundaryId.BoundarySouth: BoundaryBlock("btm", BoundaryId.BoundaryWest),
                          BoundaryId.BoundaryEast: BoundaryBlock("center", BoundaryId.BoundaryWest)
                      })

    b2 = MeshBlock("top",
                      {
                          BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundaryNorth),
                          BoundaryId.BoundaryNorth: bnd2,
                          BoundaryId.BoundarySouth: BoundaryBlock("center", BoundaryId.BoundaryNorth),
                          BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundaryNorth)
                      })

    b4 = MeshBlock("btm",
                      {
                          BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundarySouth),
                          BoundaryId.BoundaryNorth: BoundaryBlock("center", BoundaryId.BoundarySouth),
                          BoundaryId.BoundarySouth: bnd4,
                          BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundarySouth)
                      })

    b0 = MeshBlock("center",
                      {
                          BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundaryEast),
                          BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundaryWest),
                          BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundarySouth),
                          BoundaryId.BoundarySouth: BoundaryBlock("btm", BoundaryId.BoundaryNorth),
                      })
    t0 = time.time_ns()
    m, rx, ry = create_elliptical_mesh([b0, b1, b2, b3, b4], verbose=True, solver_cfg=SolverConfig(
                                                                                                 tolerance=1e-5,
                                                                                                 max_iterations=512,
                                                                                                 smoother_rounds=0))
    t1 = time.time_ns()

    print("Time taken for the mesh generation:", (t1 - t0)/1e9, "seconds")
    print("Residuals:", rx, ry)

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
    # plt.scatter(x, y)
    #     plt.plot((x[ln[i, 0]], x[ln[i, 1]]), (y[ln[i, 0]], y[ln[i, 1]]), color="black", linestyle="dashed")
    plt.xlim(-1, +1)
    plt.ylim(-1, +1)
    plt.gca().set_aspect("equal")
    plt.show()

