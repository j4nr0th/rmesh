import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as col
from rmsh import geometry
from rmsh import meshblocks
from scipy.interpolate import make_interp_spline
from scipy.interpolate import BSpline, NdBSpline
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Line:
    p1: int
    p2: int


@dataclass
class Spline:
    p1: int
    p2: int
    p3: int
    p4: int


def discretize(pts: list[Point], o, n: int, prog: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n) ** prog
    to = type(o)
    if to == Line:
        o: Line
        p1 = pts[o.p1]
        p2 = pts[o.p2]
        return t * p2.x + (1 - t) * p1.x, t * p2.y + (1 - t) * p1.y
    if to == Spline:
        o: Spline
        p1 = pts[o.p1]
        p2 = pts[o.p2]
        p3 = pts[o.p3]
        p4 = pts[o.p4]
        bsx = BSpline([0, 1/3, 2/3, 1], (p1.x, p2.x, p3.x, p4.x), k=3)
        bsy = BSpline([0, 1/3, 2/3, 1], (p1.y, p2.y, p3.y, p4.y), k=3)
        return bsx(t), bsy(t)
    raise RuntimeError(f"Invalid type of the object: {to}")


def make_split_curves(x: np.ndarray, y: np.ndarray, lengths: list[int], indices: list[int]) -> list[np.ndarray]:
    nodes = []
    # s = CubicSpline(x, y)
    n = x.shape[0]
    s = make_interp_spline(np.linspace(0, 1, n), np.stack((x, y), axis=1))
    for i, idx1 in enumerate(indices[:-1]):
        idx2 = indices[i + 1]
        t = np.linspace(idx1, idx2, lengths[i])/n
        sp = s(t)
        nodes.append(sp)
    return nodes


if __name__ == "__main__":
    data_af = np.loadtxt("airfoil_13.dat")
    data_fl = np.loadtxt("flap_13.dat")

    naf = data_af.shape[0]
    xaf = data_af[:, 0]
    yaf = data_af[:, 1]

    nfl = data_fl.shape[0]
    xfl = data_fl[:, 0]
    yfl = data_fl[:, 1]

    pt_list = []
    ln_list = []
    for i in range(naf):
        pt_list.append(Point(xaf[i], yaf[i]))
    for i in range(nfl):
        pt_list.append(Point(xfl[i], yfl[i]))


    # Far field parameters
    D_FAR_FIELD = 10
    THETA_FAR_FIELD = 10
    D_FAR_FRONT = 5     # Controls the distance of the front section
    THETA_FRONT = 30    # Controls the angle of the front section

    # PARAMETERS WING C MESH
    P_WING_TOP = 259
    P_WING_TOP_X = pt_list[P_WING_TOP].x
    P_WING_TOP_Y = pt_list[P_WING_TOP].y

    P_WING_BOTTOM = 199
    P_WING_BOTTOM_X = pt_list[P_WING_BOTTOM].x
    P_WING_BOTTOM_Y = pt_list[P_WING_BOTTOM].y

    P_WING_FLAP = 35
    P_WING_FLAP_X = pt_list[P_WING_FLAP].x
    P_WING_FLAP_Y = pt_list[P_WING_FLAP].y

    P_WING_MIDDLE_BOTTOM = 136
    P_WING_MIDDLE_BOTTOM_X = pt_list[P_WING_MIDDLE_BOTTOM].x
    P_WING_MIDDLE_BOTTOM_Y = pt_list[P_WING_MIDDLE_BOTTOM].y

    P_WING_TE = 0
    P_WING_TE_X = pt_list[P_WING_TE].x
    P_WING_TE_Y = pt_list[P_WING_TE].y

    P_WING_LE = 228
    P_WING_LE_X = pt_list[P_WING_LE].x
    P_WING_LE_Y = pt_list[P_WING_LE].y

    # PARAMETERS FLAP
    P_FLAP_TOP = 518
    P_FLAP_TOP_X = pt_list[P_FLAP_TOP].x
    P_FLAP_TOP_Y = pt_list[P_FLAP_TOP].y

    P_FLAP_BOTTOM = 506
    P_FLAP_BOTTOM_X = pt_list[P_FLAP_BOTTOM].x
    P_FLAP_BOTTOM_Y = pt_list[P_FLAP_BOTTOM].y

    P_FLAP_TE = 427
    P_FLAP_TE_X = pt_list[P_FLAP_TE].x
    P_FLAP_TE_Y = pt_list[P_FLAP_TE].y

    P_FLAP_WING_TOP = 558
    P_FLAP_WING_TOP_X = pt_list[P_FLAP_WING_TOP].x
    P_FLAP_WING_TOP_Y = pt_list[P_FLAP_WING_TOP].y

    P_FLAP_LE = 509
    P_FLAP_LE_X = pt_list[P_FLAP_LE].x
    P_FLAP_LE_Y = pt_list[P_FLAP_LE].y

    R_FLAP = 0.02   # Width of the flap mesh on the bottom.

    SPLINE_THETA_FLAP = 15
    SPLINE_R_FLAP = 0.02
    SPLINE_THETA_FLAP_BOTTOM = 7
    SPLINE_R_FLAP_BOTTOM = 0.15
    SPLINE_THETA_FLAP_BOTTOM_TE = 21

    # Corner points flap c mesh
    split_fl = [P_FLAP_TE, P_FLAP_TOP, P_FLAP_BOTTOM, P_FLAP_WING_TOP]#[0, 0.1*naf, 0.45*naf, 0.65*naf, naf-1]
    split_fl = np.sort(np.array(split_fl, dtype=int))

    pt_list.insert(657, Point(P_FLAP_LE_X - R_FLAP, (P_WING_FLAP_Y + P_FLAP_TOP_Y)/2))
    pt_list.insert(658, Point(P_FLAP_LE_X - R_FLAP, P_FLAP_LE_Y - R_FLAP))
    pt_list.insert(659, Point(P_FLAP_TE_X + 0.005, P_FLAP_TE_Y - R_FLAP))
    pt_list.insert(660, Point(P_FLAP_TE_X, (P_WING_TE_Y + P_FLAP_WING_TOP_Y)/2 + (P_FLAP_TE_X - (P_WING_TE_X + P_FLAP_WING_TOP_X)/2)*np.tan(-THETA_FAR_FIELD*np.pi/180) - 0.01))
    pt_list.insert(661, Point((P_WING_TE_X + P_FLAP_WING_TOP_X)/2, (P_WING_TE_Y + P_FLAP_WING_TOP_Y)/2)) # Middle between te wing and flap.

    #  Spline control points
    pt_list.insert(669, Point(P_FLAP_LE_X -R_FLAP + SPLINE_R_FLAP*np.cos(SPLINE_THETA_FLAP*np.pi/180), (P_WING_FLAP_Y + P_FLAP_TOP_Y)/2 + SPLINE_R_FLAP*np.sin(SPLINE_THETA_FLAP*np.pi/180)))
    pt_list.insert(670, Point((P_WING_TE_X + P_FLAP_WING_TOP_X)/2 - SPLINE_R_FLAP*np.cos(THETA_FAR_FIELD*np.pi/180), (P_WING_TE_Y + P_FLAP_WING_TOP_Y)/2 - SPLINE_R_FLAP*np.sin(-THETA_FAR_FIELD*np.pi/180)))
    pt_list.insert(671, Point(P_FLAP_LE_X - R_FLAP + SPLINE_R_FLAP_BOTTOM*np.cos(SPLINE_THETA_FLAP_BOTTOM*np.pi/180), P_FLAP_LE_Y - R_FLAP - SPLINE_R_FLAP_BOTTOM*np.sin(SPLINE_THETA_FLAP_BOTTOM*np.pi/180)))
    pt_list.insert(672, Point(P_FLAP_TE_X - SPLINE_R_FLAP_BOTTOM*np.cos(SPLINE_THETA_FLAP_BOTTOM_TE*np.pi/180), P_FLAP_TE_Y - R_FLAP + SPLINE_R_FLAP_BOTTOM*np.sin(SPLINE_THETA_FLAP_BOTTOM_TE*np.pi/180)))

    #  Far field points flap c mesh
    pt_list.insert(662, Point(D_FAR_FIELD, -0.1047 + (D_FAR_FIELD - P_FLAP_TE_X)*np.tan(-THETA_FAR_FIELD*np.pi/180 / 2)))
    pt_list.insert(663, Point(D_FAR_FIELD, 0))
    pt_list.insert(664, Point(D_FAR_FIELD, -0.1323 + (D_FAR_FIELD - P_FLAP_TE_X)*np.tan(-THETA_FAR_FIELD*np.pi/180)))

    #  Edges flap to corner point
    Line_10 = Line(656, P_FLAP_TOP)
    Line_11 = Line(657, P_FLAP_BOTTOM)
    Line_12 = Line(660, P_FLAP_WING_TOP)
    Line_13 = Line(658, P_FLAP_TE)
    Line_14 = Line(659, P_FLAP_TE)

    split_af = [P_WING_TOP, P_WING_BOTTOM, P_WING_MIDDLE_BOTTOM, P_WING_TE, P_WING_FLAP] #[0, 0.35*nfl, 0.45*nfl, 0.6*nfl, nfl-1]
    split_af = np.sort(np.array(split_af, dtype=int))

    nnodes = len(pt_list)
    xn = np.zeros(nnodes)
    yn = np.zeros(nnodes)
    l = 0
    for i, p in enumerate(pt_list):
        if p is not None:
            xn[i] = p.x
            yn[i] = p.y
            l += 1
    plt.scatter(xn[:l], yn[:l])
    plt.gca().set_aspect("equal")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()


