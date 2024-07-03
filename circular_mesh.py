import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from rmsh.base import BoundaryCurve, MeshBlock, Mesh2D, create_elliptical_mesh, BoundaryId, BoundaryBlock, SolverConfig
from scipy import sparse as sp


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

    m, _, _ = create_elliptical_mesh([block], verbose=False)
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

    m, _, _ = create_elliptical_mesh([block], verbose=False)
    return m


def four_wierdly_connected_ones(n1: int, n2: int) -> Mesh2D:
    angle_l = np.linspace(+0*np.pi/2, +1*np.pi/2, n1)
    angle_b = np.linspace(+1*np.pi/2, +2*np.pi/2, n2)
    angle_r = np.linspace(+2*np.pi/2, +3*np.pi/2, n1)
    angle_t = np.linspace(+3*np.pi/2, +4*np.pi/2, n2)

    cl = BoundaryCurve(np.cos(angle_l), np.sin(angle_l))
    cr = BoundaryCurve(np.cos(angle_r), np.sin(angle_r))
    ct = BoundaryCurve(np.cos(angle_t), np.sin(angle_t))
    cb = BoundaryCurve(np.cos(angle_b), np.sin(angle_b))

    blockwest = MeshBlock("left", {
        BoundaryId.BoundaryWest: cl,
        BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundaryWest, n1),
        BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundaryWest, n2),
        BoundaryId.BoundarySouth: BoundaryBlock("bottom", BoundaryId.BoundaryWest, n2),
    })
    blockeast = MeshBlock("right", {
        BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundaryEast, n1),
        BoundaryId.BoundaryEast: cr,
        BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundaryEast, n2),
        BoundaryId.BoundarySouth: BoundaryBlock("bottom", BoundaryId.BoundaryEast, n2),
    })
    blocknorth = MeshBlock("top", {
        BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundaryNorth, n1),
        BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundaryNorth, n1),
        BoundaryId.BoundaryNorth: ct,
        BoundaryId.BoundarySouth: BoundaryBlock("bottom", BoundaryId.BoundaryNorth, n2),
    })
    blocksouth = MeshBlock("bottom", {
        BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundarySouth, n1),
        BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundarySouth, n1),
        BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundarySouth, n2),
        BoundaryId.BoundarySouth: cb,
    })

    m, _, _ = create_elliptical_mesh([blockwest, blockeast, blocknorth, blocksouth], verbose=False,
                                     solver_cfg=SolverConfig(smoother_rounds=0, max_iterations=64))
    return m


def as_god_intended(n1: int, n2: int, n3: int) -> Mesh2D:
    angle_l = np.linspace(+0*np.pi/2, +1*np.pi/2, n1)
    angle_b = np.linspace(+1*np.pi/2, +2*np.pi/2, n2)
    angle_r = np.linspace(+2*np.pi/2, +3*np.pi/2, n1)
    angle_t = np.linspace(+3*np.pi/2, +4*np.pi/2, n2)

    cl = BoundaryCurve(np.cos(angle_l), np.sin(angle_l))
    cr = BoundaryCurve(np.cos(angle_r), np.sin(angle_r))
    ct = BoundaryCurve(np.cos(angle_t), np.sin(angle_t))
    cb = BoundaryCurve(np.cos(angle_b), np.sin(angle_b))

    blockwest = MeshBlock("left", {
        BoundaryId.BoundaryWest: cl,
        BoundaryId.BoundaryEast: BoundaryBlock("center", BoundaryId.BoundaryWest, n1),
        BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundaryWest, n3),
        BoundaryId.BoundarySouth: BoundaryBlock("bottom", BoundaryId.BoundaryWest, n3),
    })
    blockeast = MeshBlock("right", {
        BoundaryId.BoundaryWest: BoundaryBlock("center", BoundaryId.BoundaryEast, n1),
        BoundaryId.BoundaryEast: cr,
        BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundaryEast, n3),
        BoundaryId.BoundarySouth: BoundaryBlock("bottom", BoundaryId.BoundaryEast, n3),
    })
    blocknorth = MeshBlock("top", {
        BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundaryNorth, n3),
        BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundaryNorth, n3),
        BoundaryId.BoundaryNorth: ct,
        BoundaryId.BoundarySouth: BoundaryBlock("center", BoundaryId.BoundaryNorth, n2),
    })
    blocksouth = MeshBlock("bottom", {
        BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundarySouth, n3),
        BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundarySouth, n3),
        BoundaryId.BoundaryNorth: BoundaryBlock("center", BoundaryId.BoundarySouth, n2),
        BoundaryId.BoundarySouth: cb,
    })
    blockmiddle = MeshBlock("center", {
        BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundaryEast, n1),
        BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundaryWest, n1),
        BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundarySouth, n2),
        BoundaryId.BoundarySouth: BoundaryBlock("bottom", BoundaryId.BoundaryNorth, n2),
    })

    m, _, _ = create_elliptical_mesh([blockwest, blockeast, blocknorth, blocksouth, blockmiddle], verbose=False,
                                     solver_cfg=SolverConfig(smoother_rounds=0, max_iterations=128, tolerance=5e-6))
    return m


def ungodly(n1: int, n2: int) -> Mesh2D:
    blockmiddle = MeshBlock("center", {
        BoundaryId.BoundaryWest: BoundaryBlock("center", BoundaryId.BoundaryEast, n1),
        BoundaryId.BoundaryEast: BoundaryBlock("center", BoundaryId.BoundaryWest, n1),
        BoundaryId.BoundaryNorth: BoundaryBlock("center", BoundaryId.BoundarySouth, n2),
        BoundaryId.BoundarySouth: BoundaryBlock("center", BoundaryId.BoundaryNorth, n2),
    })

    m, _, _ = create_elliptical_mesh([blockmiddle], verbose=False,
        solver_cfg=SolverConfig(smoother_rounds=0, max_iterations=64))
    return m


def four_weirder(n: int) -> Mesh2D:
    angle_l = np.linspace(+0*np.pi/2, +1*np.pi/2, n)
    angle_b = np.linspace(+1*np.pi/2, +2*np.pi/2, n)
    angle_r = np.linspace(+2*np.pi/2, +3*np.pi/2, n)
    angle_t = np.linspace(+3*np.pi/2, +4*np.pi/2, n)

    cl = BoundaryCurve(np.cos(angle_l), np.sin(angle_l))
    cr = BoundaryCurve(np.cos(angle_r), np.sin(angle_r))
    ct = BoundaryCurve(np.cos(angle_t), np.sin(angle_t))
    cb = BoundaryCurve(np.cos(angle_b), np.sin(angle_b))

    blockwest = MeshBlock("left", {
        BoundaryId.BoundaryWest: cl,
        BoundaryId.BoundaryEast: BoundaryBlock("bottom", BoundaryId.BoundaryWest),
        BoundaryId.BoundaryNorth: BoundaryBlock("top", BoundaryId.BoundarySouth),
        BoundaryId.BoundarySouth: BoundaryBlock("right", BoundaryId.BoundaryNorth),
    })
    blockeast = MeshBlock("right", {
        BoundaryId.BoundaryWest: BoundaryBlock("top", BoundaryId.BoundaryEast),
        BoundaryId.BoundaryEast: cr,
        BoundaryId.BoundaryNorth: BoundaryBlock("left", BoundaryId.BoundarySouth),
        BoundaryId.BoundarySouth: BoundaryBlock("bottom", BoundaryId.BoundaryNorth),
    })
    blocknorth = MeshBlock("top", {
        BoundaryId.BoundaryWest: BoundaryBlock("bottom", BoundaryId.BoundaryEast),
        BoundaryId.BoundaryEast: BoundaryBlock("right", BoundaryId.BoundaryWest),
        BoundaryId.BoundaryNorth: ct,
        BoundaryId.BoundarySouth: BoundaryBlock("left", BoundaryId.BoundaryNorth),
    })
    blocksouth = MeshBlock("bottom", {
        BoundaryId.BoundaryWest: BoundaryBlock("left", BoundaryId.BoundaryEast),
        BoundaryId.BoundaryEast: BoundaryBlock("top", BoundaryId.BoundaryWest),
        BoundaryId.BoundaryNorth: BoundaryBlock("right", BoundaryId.BoundarySouth),
        BoundaryId.BoundarySouth: cb,
    })

    m, _, _ = create_elliptical_mesh([blockwest, blockeast, blocknorth, blocksouth], verbose=False,
                                     solver_cfg=SolverConfig(smoother_rounds=0, max_iterations=64))
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
    m = one_block_only(100, 100)
    plot_mesh(m)
    m = self_closed_mesh(100, 100)
    plot_mesh(m)
    m = four_wierdly_connected_ones(50, 50)
    plot_mesh(m)
    m = as_god_intended(5, 5, 5)
    plot_mesh(m)
    lines = m.lines
    x = m.x
    # p1 = lines[:, 0]
    # p2 = lines[:, 1]
    rows = np.arange(lines.shape[0])
    rows = np.stack((rows, rows), axis=1).flatten()
    cols = lines.flatten()
    data = np.stack((np.ones(lines.shape[0]),-np.ones(lines.shape[0])), axis=1).flatten()
    e10 = sp.csr_array((data, (rows, cols)), shape=(lines.shape[0], x.shape[0]))
    plt.spy(e10)
    plt.show()

    m = ungodly(10, 10)
    plot_mesh(m)
    m = four_weirder(10)
    plot_mesh(m)
