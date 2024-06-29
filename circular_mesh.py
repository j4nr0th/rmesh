import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from rmsh.base import BoundaryCurve, MeshBlock, Mesh2D, create_elliptical_mesh, BoundaryId, BoundaryBlock


def one_block_only(n1: int, n2: int) -> Mesh2D:
    angle_l = np.linspace(+0*np.pi/2, +1*np.pi/2, n1)
    angle_b = np.linspace(+1*np.pi/2, +2*np.pi/2, n2)
    angle_r = np.linspace(+2*np.pi/2, +3*np.pi/2, n1)
    angle_t = np.linspace(+3*np.pi/2, +4*np.pi/2, n2)

    cl = BoundaryCurve(np.cos(angle_l), np.sin(angle_l))
    cr = BoundaryCurve(np.cos(angle_r), np.sin(angle_r))
    ct = BoundaryCurve(np.cos(angle_t), np.sin(angle_t))
    cb = BoundaryCurve(np.cos(angle_b), np.sin(angle_b))

    block = MeshBlock("only one", {
        BoundaryId.BoundaryWest: cl,
        BoundaryId.BoundaryEast: cr,
        BoundaryId.BoundaryNorth: ct,
        BoundaryId.BoundarySouth: cb,
    })

    m, _, _ = create_elliptical_mesh([block], verbose=True)
    return m


def self_closed_mesh(n1: int, n2: int) -> Mesh2D:
    angle_b = np.linspace(-np.pi, 0, n2)
    angle_t = np.linspace(0, +np.pi, n2)

    ct = BoundaryCurve(np.cos(angle_t), np.sin(angle_t))
    cb = BoundaryCurve(np.cos(angle_b), np.sin(angle_b))

    bl = BoundaryBlock("only one", BoundaryId.BoundaryEast, n1)
    br = BoundaryBlock("only one", BoundaryId.BoundaryWest, n1)

    block = MeshBlock("only one", {
        BoundaryId.BoundaryWest: bl,
        BoundaryId.BoundaryEast: br,
        BoundaryId.BoundaryNorth: ct,
        BoundaryId.BoundarySouth: cb,
    })

    m, _, _ = create_elliptical_mesh([block], verbose=True)
    return m




def plot_mesh(m: Mesh2D) -> None:
    x = m.x
    y = m.y
    line_indices = m.lines
    xb = x[line_indices[:, 0]]
    xe = x[line_indices[:, 1]]
    yb = y[line_indices[:, 0]]
    ye = y[line_indices[:, 1]]

    rb = np.stack((xb, yb), axis=1)
    re = np.stack((xe, ye), axis=1)
    c = LineCollection(np.stack((rb, re), axis=1))
    plt.scatter(x, y, s=8, color="red")
    # for idx in range(line_indices.shape[0]):
    #     plt.plot((xb[idx], xe[idx]), (yb[idx], ye[idx]))
    plt.gca().add_collection(c)
    plt.gca().set_aspect("equal")
    plt.xlim(-1.1, +1.1)
    plt.ylim(-1.1, +1.1)
    plt.show()


if __name__ == "__main__":
    m = one_block_only(50, 50)
    plot_mesh(m)
    m = self_closed_mesh(5, 5)
    plot_mesh(m)
