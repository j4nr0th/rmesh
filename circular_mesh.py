import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from rmsh.base import BoundaryCurve, MeshBlock, Mesh2D, create_elliptical_mesh, BoundaryId


def one_block_only(n1: int, n2: int) -> Mesh2D:
    angle_l = np.linspace(-1*np.pi/2, +1*np.pi/2, n1)
    angle_r = np.linspace(+5*np.pi/2, +3*np.pi/2, n1)
    angle_t = np.linspace(+1*np.pi/2, +3*np.pi/2, n2)
    angle_b = np.linspace(-3*np.pi/2, -1*np.pi/2, n2)

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


def plot_mesh(m: Mesh2D) -> None:
    x = m.x
    y = m.y
    line_indices = m.lines
    xb = x[line_indices[:, 0]]
    xe = x[line_indices[:, 1]]
    yb = x[line_indices[:, 0]]
    ye = x[line_indices[:, 1]]

    rb = np.stack((xb, yb), axis=1)
    re = np.stack((xe, ye), axis=1)

    c = LineCollection(np.stack((rb, re), axis=1))
    plt.gca().add_collection(c)
    plt.gca().set_aspect("equal")
    plt.xlim(-1, +1)
    plt.ylim(-1, +1)
    plt.show()


if __name__ == "__main__":
    input()
    m = one_block_only(3, 3)
    plot_mesh(m)
