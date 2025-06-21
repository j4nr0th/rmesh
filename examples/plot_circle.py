"""
Meshing a Circle
================

.. currentmodule:: rmsh

This example shows how a simple circular mesh can be created using ``rmsh``.
It is also intended to show what incorrect usage may lead to in hopes of
being useful when trying to identify errors.
"""  # noqa: D205, D400

# %%
#
# The Setup
# ---------
#
# First the common setup for plotting has to be made. For this case, it will be done using
# the :mod:`matplotlib` module, using :class:`matplotlib.collections.LineCollection` to
# plot cell boundaries.
#
from time import perf_counter

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from rmsh import (
    BoundaryCurve,
    Mesh2D,
    SolverConfig,
    create_elliptical_mesh,
)
from rmsh.geometry import MeshBlock


def plot_mesh(m: Mesh2D) -> None:
    """Show the mesh using matplotlib."""
    x = m.pos_x
    y = m.pos_y
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


# %%
#
# Simplest
# --------
#
# First is the most basic approach. In this case, only a single block is used,
# with its boundaries being the circumference of the circle that we want to mesh.
#


def circle_function(theta: npt.ArrayLike, r: float = 1.0) -> npt.NDArray[np.float64]:
    """Return point(s) on a circle of given radius for specified angle(s)."""
    return np.astype(
        r * np.stack((np.cos(theta), np.sin(theta)), axis=1, dtype=np.float64),
        np.float64,
        copy=False,
    )


def one_block_only(n1: int, n2: int) -> Mesh2D:
    """Mesh a circle with a single block."""
    cl = BoundaryCurve.from_samples(circle_function, n1, lambda t: (t + 2.5) * np.pi / 2)
    cr = BoundaryCurve.from_samples(circle_function, n1, lambda t: (t + 0.5) * np.pi / 2)
    ct = BoundaryCurve.from_samples(circle_function, n2, lambda t: (t + 1.5) * np.pi / 2)
    cb = BoundaryCurve.from_samples(circle_function, n2, lambda t: (t + 3.5) * np.pi / 2)

    block = MeshBlock("only one", left=cl, right=cr, top=ct, bottom=cb)
    t0 = perf_counter()
    m, _, _ = create_elliptical_mesh([block])
    t1 = perf_counter()
    print(f"Meshed in {t1 - t0:g} seconds.")
    return m


# Now let's show it!
# %%
plot_mesh(one_block_only(2, 2))
# %%
plot_mesh(one_block_only(10, 10))
# %%
plot_mesh(one_block_only(50, 50))

# %%
#
# Almost the Worst
# ----------------
#
# Next is one worst possible things you can make while meshing: a pair of
# boundaries, which at no point have a connection with a curve. In this
# case, the left and right boundaries only need to connect to one another
# and have no requirement on their position. Top and bottom boundaries
# each cover half of a circle.
#
# The resulting mesh is very overlapping and very silly looking. If you
# obtain such a mesh from your meshing, consider checking how you've defined
# the boundaries defined by :class:`BoundaryBlock` objects.


def self_closed_mesh(n1: int, n2: int) -> Mesh2D:
    """Mesh a circle by self connecting two opposing boundaries."""
    ct = BoundaryCurve.from_samples(circle_function, n2, lambda t: (t + 0) * np.pi)
    cb = BoundaryCurve.from_samples(circle_function, n2, lambda t: (t + 1) * np.pi)
    block = MeshBlock("only one", top=ct, bottom=cb)
    block.left = block.bbnd_right(n1)
    block.right = block.bbnd_left(n1)

    t0 = perf_counter()
    m, _, _ = create_elliptical_mesh([block])
    t1 = perf_counter()
    print(f"Meshed in {t1 - t0:g} seconds.")
    return m


# Now let's show it!
# %%
plot_mesh(self_closed_mesh(2, 2))
# %%
plot_mesh(self_closed_mesh(10, 10))
# %%
plot_mesh(self_closed_mesh(50, 50))

# %%
#
# Doing it Wrong
# --------------
#
# A way to make the initial circular mesh better would be to mesh with five blocks,
# rather than five. However if they are connected incorrectly, the result will be
# significantly worse.


def four_wierdly_connected_ones(n1: int, n2: int) -> Mesh2D:
    """Mesh in a weird way, where four blocks are used, but weirdly connected."""
    cl = BoundaryCurve.from_samples(circle_function, n1, lambda t: (t + 2.5) * np.pi / 2)
    cr = BoundaryCurve.from_samples(circle_function, n1, lambda t: (t + 0.5) * np.pi / 2)
    ct = BoundaryCurve.from_samples(circle_function, n2, lambda t: (t + 1.5) * np.pi / 2)
    cb = BoundaryCurve.from_samples(circle_function, n2, lambda t: (t + 3.5) * np.pi / 2)

    block_left = MeshBlock("left", left=cl)
    block_right = MeshBlock("right", right=cr)
    block_bottom = MeshBlock("bottom", bottom=cb)
    block_top = MeshBlock("top", top=ct)

    block_left.right = block_right.bbnd_left(n1)
    block_left.top = block_top.bbnd_left(n2)
    block_left.bottom = block_bottom.bbnd_left(n2)

    block_right.left = block_left.bbnd_right(n1)
    block_right.top = block_top.bbnd_right(n2)
    block_right.bottom = block_bottom.bbnd_right(n2)

    block_top.left = block_left.bbnd_top(n1)
    block_top.right = block_right.bbnd_top(n1)
    block_top.bottom = block_bottom.bbnd_top(n2)

    block_bottom.left = block_left.bbnd_bottom(n1)
    block_bottom.right = block_right.bbnd_bottom(n1)
    block_bottom.top = block_top.bbnd_bottom(n2)

    t0 = perf_counter()
    m, _, _ = create_elliptical_mesh(
        [block_left, block_right, block_top, block_bottom],
        verbose=False,
        solver_cfg=SolverConfig(smoother_rounds=0, max_iterations=64),
    )
    t1 = perf_counter()
    print(f"Meshed in {t1 - t0:g} seconds.")
    return m


# Now let's show it!
# %%
plot_mesh(four_wierdly_connected_ones(2, 2))
# %%
plot_mesh(four_wierdly_connected_ones(10, 10))
# %%
plot_mesh(four_wierdly_connected_ones(50, 50))

# %%
#
# Doing it Right
# --------------
#
# Probably the best way to mesh the circle is using five blocks, so that a quarter
# of the boundary is take care of by one block each, then they're all connected
# together using a central block with no numerical boundaries.


def as_god_intended(n1: int, n2: int, n3: int) -> Mesh2D:
    """Mesh the circle the way God intended."""
    cl = BoundaryCurve.from_samples(circle_function, n1, lambda t: (t + 2.5) * np.pi / 2)
    cr = BoundaryCurve.from_samples(circle_function, n1, lambda t: (t + 0.5) * np.pi / 2)
    ct = BoundaryCurve.from_samples(circle_function, n2, lambda t: (t + 1.5) * np.pi / 2)
    cb = BoundaryCurve.from_samples(circle_function, n2, lambda t: (t + 3.5) * np.pi / 2)

    block_left = MeshBlock("left", left=cl)
    block_right = MeshBlock("right", right=cr)
    block_bottom = MeshBlock("bottom", bottom=cb)
    block_top = MeshBlock("top", top=ct)
    block_center = MeshBlock(
        "center",
        left=block_left.bbnd_right(),
        right=block_right.bbnd_left(),
        top=block_top.bbnd_bottom(),
        bottom=block_bottom.bbnd_top(),
    )

    block_left.right = block_center.bbnd_left(n1)
    block_left.top = block_top.bbnd_left(n3)
    block_left.bottom = block_bottom.bbnd_left(n3)

    block_right.left = block_center.bbnd_right(n1)
    block_right.top = block_top.bbnd_right(n3)
    block_right.bottom = block_bottom.bbnd_right(n3)

    block_top.left = block_left.bbnd_top(n3)
    block_top.right = block_right.bbnd_top(n3)
    block_top.bottom = block_center.bbnd_top(n2)

    block_bottom.left = block_left.bbnd_bottom(n3)
    block_bottom.right = block_right.bbnd_bottom(n3)
    block_bottom.top = block_center.bbnd_bottom(n2)

    t0 = perf_counter()
    m, _, _ = create_elliptical_mesh(
        [block_left, block_right, block_bottom, block_top, block_center],
        verbose=False,
        solver_cfg=SolverConfig(
            smoother_rounds=0, max_iterations=64, tolerance=5e-6, force_direct=False
        ),
    )
    t1 = perf_counter()
    print(f"Meshed in {t1 - t0:g} seconds.")
    return m


# %%
plot_mesh(as_god_intended(2, 2, 2))
# %%
plot_mesh(as_god_intended(5, 5, 5))
# %%
plot_mesh(as_god_intended(25, 25, 25))

# %%
#
# No Boundaries
# -------------
#
# If you decide to define a mesh which has no numerical boundaries, it is technically
# possible, however, the solver will just place all points in the origin, though the
# topology will still remain correct.


def ungodly(n1: int, n2: int) -> Mesh2D:
    """Mesh with no hard boundaries."""
    block_center = MeshBlock("center")
    block_center.left = block_center.bbnd_right(n1)
    block_center.right = block_center.bbnd_left(n1)
    block_center.top = block_center.bbnd_bottom(n2)
    block_center.bottom = block_center.bbnd_top(n2)

    t0 = perf_counter()
    m, _, _ = create_elliptical_mesh(
        [block_center],
        verbose=False,
        solver_cfg=SolverConfig(smoother_rounds=0, max_iterations=64),
    )
    t1 = perf_counter()
    print(f"Meshed in {t1 - t0:g} seconds.")
    return m


# %%
plot_mesh(ungodly(2, 2))
# %%
plot_mesh(ungodly(5, 5))
# %%
plot_mesh(ungodly(25, 25))

# %%
#
# Mixing Up Boundaries
# --------------------
#
# Once, again, mixing up boundaries will inevitably result in a self-overlapping mesh.
# This example here does so with four different blocks.


def four_weirder(n: int) -> Mesh2D:
    """Outer boundary's correct, but the rest is not."""
    cl = BoundaryCurve.from_samples(circle_function, n, lambda t: (t + 2.5) * np.pi / 2)
    cr = BoundaryCurve.from_samples(circle_function, n, lambda t: (t + 0.5) * np.pi / 2)
    ct = BoundaryCurve.from_samples(circle_function, n, lambda t: (t + 1.5) * np.pi / 2)
    cb = BoundaryCurve.from_samples(circle_function, n, lambda t: (t + 3.5) * np.pi / 2)

    block_left = MeshBlock("left", left=cl)
    block_right = MeshBlock("right", right=cr)
    block_top = MeshBlock("top", top=ct)
    block_bottom = MeshBlock("bottom", bottom=cb)

    block_left.right = block_bottom.bbnd_left()
    block_left.top = block_top.bbnd_bottom()
    block_left.bottom = block_right.bbnd_top()

    block_right.left = block_top.bbnd_right()
    block_right.top = block_left.bbnd_bottom()
    block_right.bottom = block_bottom.bbnd_top()

    block_top.left = block_bottom.bbnd_right()
    block_top.right = block_right.bbnd_left()
    block_top.bottom = block_left.bbnd_top()

    block_bottom.left = block_left.bbnd_right()
    block_bottom.right = block_top.bbnd_left()
    block_bottom.top = block_right.bbnd_bottom()

    t0 = perf_counter()
    m, _, _ = create_elliptical_mesh(
        [block_left, block_right, block_top, block_bottom],
        verbose=False,
        solver_cfg=SolverConfig(smoother_rounds=0, max_iterations=64),
    )
    t1 = perf_counter()
    print(f"Meshed in {t1 - t0:g} seconds.")
    return m


# %%
plot_mesh(four_weirder(2))
# %%
plot_mesh(four_weirder(10))
# %%
plot_mesh(four_weirder(50))
