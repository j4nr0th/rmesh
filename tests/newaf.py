import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
import matplotlib.collections as col
from scipy.interpolate import (splprep, splev)
from bezier import curve
from os import path

from rmsh import *



@dataclass(frozen=True)
class Point:
    x: float
    y: float


class Line:
    points: list[Point]
    def __init__(self, *pts):
        self.points = np.array(pts, dtype=int).flatten()

    def split(self, *indices):
        new_lines = []
        indices = np.sort(np.unique(np.concatenate(((self.points[0], self.points[-1]), indices))))
        for i in range(indices.shape[0] - 1):
            i1 = self.points.tolist().index(int(indices[i]))
            i2 = self.points.tolist().index(int(indices[i+1]))
            new_lines.append(Line(self.points[i1:i2+1]))
        return new_lines


class BSpline:
    points: list[Point]
    def __init__(self, *pts):
        self.points = np.array(pts, dtype=int).flatten()


# @dataclass(frozen=True)
# class Circle:
#     p1: int
#     p2: int
#     p3: int



def discretize(pts: dict[int, Point], o, n: int, prog: float) -> tuple[np.ndarray, np.ndarray]:
    prog = 1
    t = np.linspace(0, 1, n) ** prog
    to = type(o)
    if to == Line:
        o: Line
        xvals = np.array([pts[p].x for p in o.points]).flatten()
        yvals = np.array([pts[p].y for p in o.points]).flatten()
        tck, u = splprep(np.array([xvals, yvals]), s=0 if to == Line else 1, k=3 if len(xvals) > 4 else len(xvals)-1)
        return splev(u[0] * (1-t) + u[-1] * t, tck)

    if to == BSpline:
        o: BSpline
        nx = np.array([pts[p].x for p in o.points]).flatten()
        ny = np.array([pts[p].y for p in o.points]).flatten()
        nodes = np.stack((nx, ny), axis=0)
        c = curve.Curve.from_nodes([nx, ny])
        return c.evaluate_multi(t)
    
    # if to == Circle:
    #     o: Circle
    #     p1: Point = o.p1
    #     p2: Point = o.p2
    #     p3: Point = o.p3

    raise RuntimeError(f"Invalid type of the object: {to}")


def transfinite_curve(pts: dict[int, Point], o: Line|BSpline, n: int, prog: float) -> BoundaryCurve:
    posx, posy = discretize(pts, o, n, prog)
    return BoundaryCurve(posx, posy)


def make_bblock_from_tc(id: int, tc_dict: dict[int, BoundaryCurve], north: int, west: int, south: int, east: int) -> MeshBlock:
    b1 = tc_dict[abs(north)]
    b2 = tc_dict[abs(west)]
    b3 = tc_dict[abs(south)]
    b4 = tc_dict[abs(east)]
    if north < 0:
        b1 = BoundaryCurve(np.flip(b1.x), np.flip(b1.y))
    if west < 0:
        b2 = BoundaryCurve(np.flip(b2.x), np.flip(b2.y))
    if south < 0:
        b3 = BoundaryCurve(np.flip(b3.x), np.flip(b3.y))
    if east < 0:
        b4 = BoundaryCurve(np.flip(b4.x), np.flip(b4.y))

    return MeshBlock(f"{id}",
                        {
                            BoundaryId.BoundaryNorth: b1,
                            BoundaryId.BoundaryWest: b2,
                            BoundaryId.BoundarySouth: b3,
                            BoundaryId.BoundaryEast: b4,
                        })


pt_dict = dict()
ln_dict = dict()
tc_dict = dict()
b_dict = dict()
lcntr = 0
pcntr = 0

loc = path.split(__file__)[0]

af_data = np.loadtxt(path.join(loc, "airfoil_13.dat"))
fl_data = np.loadtxt(path.join(loc, "flap_13.dat"))

for i in range(af_data.shape[0]):
    pt_dict[pcntr+1] = Point(af_data[i, 0], af_data[i, 1])
    pcntr += 1

ln_dict[lcntr+1] = Line(np.arange(1, 1+af_data.shape[0]))
lcntr += 1

for i in range(fl_data.shape[0]):
    pt_dict[pcntr+1] = Point(fl_data[i, 0], fl_data[i, 1])
    pcntr += 1

ln_dict[lcntr+1] = Line(np.arange(1+af_data.shape[0], len(pt_dict)+1))
lcntr += 1

# ------------------------------------------------------------
#
#            CURVE AND POINT DEFINITION
#
# ------------------------------------------------------------
#  Far field parameters
D_FAR_FIELD = 10
THETA_FAR_FIELD = 10
D_FAR_FRONT = 5    # Controls the distance of the front section
THETA_FRONT = 30   # Controls the angle of the front section

# PARAMETERS WING C MESH
P_WING_TOP = 260
P_WING_TOP_COORDS = pt_dict[P_WING_TOP]
P_WING_TOP_X = P_WING_TOP_COORDS.x#1.9614610000000001E-002
P_WING_TOP_Y = P_WING_TOP_COORDS.y#4.4528220000000000E-002

P_WING_BOTTOM = 200
P_WING_BOTTOM_COORDS = pt_dict[P_WING_BOTTOM]
P_WING_BOTTOM_X = P_WING_BOTTOM_COORDS.x#4.1685250000000000E-002
P_WING_BOTTOM_Y = P_WING_BOTTOM_COORDS.y#-4.5499329999999998E-002

P_WING_FLAP = 36
P_WING_FLAP_COORDS = pt_dict[P_WING_FLAP]
P_WING_FLAP_X = P_WING_FLAP_COORDS.x#0.86350629999999995
P_WING_FLAP_Y = P_WING_FLAP_COORDS.y#1.6218969999999999E-002

P_WING_MIDDLE_BOTTOM = 137
P_WING_MIDDLE_BOTTOM_COORDS = pt_dict[P_WING_MIDDLE_BOTTOM]
P_WING_MIDDLE_BOTTOM_X = P_WING_MIDDLE_BOTTOM_COORDS.x#0.33873009999999998
P_WING_MIDDLE_BOTTOM_Y = P_WING_MIDDLE_BOTTOM_COORDS.y#-7.7077880000000001E-002

P_WING_TE = 1
P_WING_TE_COORDS = pt_dict[P_WING_TE]
P_WING_TE_X = P_WING_TE_COORDS.x#0.94440000000000000
P_WING_TE_Y = P_WING_TE_COORDS.y#1.4500000000000000E-002

P_WING_LE = 229
P_WING_LE_COORDS = pt_dict[P_WING_LE]
P_WING_LE_X = P_WING_LE_COORDS.x#-3.6730000000000002E-005
P_WING_LE_Y = P_WING_LE_COORDS.y#-1.3892200000000000E-003

R_WING = 0.02
R_OUTER_WING = 0.1
THETA_WING = 40
SPLINE_R_WING = 0.07
SPLINE_R_WING_BOTTOM = 0.01
SPLINE_R_WING_TOP = 0.01
SPLINE_R_WING_BOTTOM_OUTER = 0.075
SPLINE_R_WING_TOP_OUTER = 0.075
SPLINE_R_WING_TE = 0.45
SPLINE_THETA_WING_TE = 15

# PARAMETERS FLAP
P_FLAP_TOP = 519
P_FLAP_TOP_COORDS = pt_dict[P_FLAP_TOP]
P_FLAP_TOP_X = P_FLAP_TOP_COORDS.x#0.8973106
P_FLAP_TOP_Y = P_FLAP_TOP_COORDS.y#-0.01836688

P_FLAP_BOTTOM = 507
P_FLAP_BOTTOM_COORDS = pt_dict[P_FLAP_BOTTOM]
P_FLAP_BOTTOM_X = P_FLAP_BOTTOM_COORDS.x#0.8925225
P_FLAP_BOTTOM_Y = P_FLAP_BOTTOM_COORDS.y#-0.03110314

P_FLAP_TE = 428
P_FLAP_TE_COORDS = pt_dict[P_FLAP_TE]
P_FLAP_TE_X = P_FLAP_TE_COORDS.x#1.2025000000000000
P_FLAP_TE_Y = P_FLAP_TE_COORDS.y#-0.10470000000000000

P_FLAP_WING_TOP = 559
P_FLAP_WING_TOP_COORDS = pt_dict[P_FLAP_WING_TOP]
P_FLAP_WING_TOP_X = P_FLAP_WING_TOP_COORDS.x#0.94449620000000001
P_FLAP_WING_TOP_Y = P_FLAP_WING_TOP_COORDS.y#-1.1973540000000000E-002

P_FLAP_LE = 510
P_FLAP_LE_COORDS = pt_dict[P_FLAP_LE]
P_FLAP_LE_X = P_FLAP_LE_COORDS.x#0.89028439999999998
P_FLAP_LE_Y = P_FLAP_LE_COORDS.y#-2.7303760000000000E-002

R_FLAP = 0.02  #Width of the flap mesh on the bottom.

SPLINE_THETA_FLAP = 15
SPLINE_R_FLAP = 0.02
SPLINE_THETA_FLAP_BOTTOM = 7
SPLINE_R_FLAP_BOTTOM = 0.15
SPLINE_THETA_FLAP_BOTTOM_TE = 21

#  Split the flap
l2:Line = ln_dict[2]
new_curves2 = l2.split(P_FLAP_TE, P_FLAP_TOP, P_FLAP_BOTTOM, P_FLAP_WING_TOP)
new_idx = max(ln_dict.keys())
for c in new_curves2:
    ln_dict[new_idx+1] = c
    new_idx += 1
l3 = ln_dict[3]
l4 = ln_dict[4]
l5 = ln_dict[5]
l6 = ln_dict[6]

ln_dict[3] = l4
ln_dict[4] = l5
ln_dict[5] = l6
ln_dict[6] = l3

#  Corner points flap c mesh
pt_dict[657] = Point (P_FLAP_LE_X - R_FLAP, (P_WING_FLAP_Y + P_FLAP_TOP_Y)/2 )
pt_dict[658] = Point (P_FLAP_LE_X - R_FLAP, P_FLAP_LE_Y - R_FLAP)
pt_dict[659] = Point (P_FLAP_TE_X + 0.005, P_FLAP_TE_Y - R_FLAP)
pt_dict[660] = Point (P_FLAP_TE_X, (P_WING_TE_Y + P_FLAP_WING_TOP_Y)/2 + (P_FLAP_TE_X - (P_WING_TE_X + P_FLAP_WING_TOP_X)/2)*np.tan(-THETA_FAR_FIELD*np.pi/180) - 0.01)# + (P_FLAP_TE_X - (P_WING_TE_X + P_FLAP_WING_TOP_X)/2)*np.tan(-THETA_FAR_FIELD)))
pt_dict[661] = Point ((P_WING_TE_X + P_FLAP_WING_TOP_X)/2, (P_WING_TE_Y + P_FLAP_WING_TOP_Y)/2) #Middle between te wing and flap

#  Spline control points
pt_dict[669] = Point (P_FLAP_LE_X -R_FLAP + SPLINE_R_FLAP*np.cos(SPLINE_THETA_FLAP*np.pi/180), (P_WING_FLAP_Y + P_FLAP_TOP_Y)/2 + SPLINE_R_FLAP*np.sin(SPLINE_THETA_FLAP*np.pi/180))
pt_dict[670] = Point ((P_WING_TE_X + P_FLAP_WING_TOP_X)/2 - SPLINE_R_FLAP*np.cos(THETA_FAR_FIELD*np.pi/180), (P_WING_TE_Y + P_FLAP_WING_TOP_Y)/2 - SPLINE_R_FLAP*np.sin(-THETA_FAR_FIELD*np.pi/180))
pt_dict[671] = Point (P_FLAP_LE_X - R_FLAP + SPLINE_R_FLAP_BOTTOM*np.cos(SPLINE_THETA_FLAP_BOTTOM*np.pi/180), P_FLAP_LE_Y - R_FLAP - SPLINE_R_FLAP_BOTTOM*np.sin(SPLINE_THETA_FLAP_BOTTOM*np.pi/180))
pt_dict[672] = Point (P_FLAP_TE_X - SPLINE_R_FLAP_BOTTOM*np.cos(SPLINE_THETA_FLAP_BOTTOM_TE*np.pi/180), P_FLAP_TE_Y - R_FLAP + SPLINE_R_FLAP_BOTTOM*np.sin(SPLINE_THETA_FLAP_BOTTOM_TE*np.pi/180))


#  Far field points flap c mesh
pt_dict[662] = Point (D_FAR_FIELD, -0.1047 + (D_FAR_FIELD - P_FLAP_TE_X)*np.tan(-THETA_FAR_FIELD*np.pi/180 / 2))
pt_dict[663] = Point (D_FAR_FIELD, 0)# 0.002 + (D_FAR_FIELD - P_FLAP_TE_X)*np.tan(THETA_FAR_FIELD*np.pi/180)))
pt_dict[664] = Point (D_FAR_FIELD, -0.1323 + (D_FAR_FIELD - P_FLAP_TE_X)*np.tan(-THETA_FAR_FIELD*np.pi/180))

#  Edges flap to corner point
ln_dict[10] = Line (657, P_FLAP_TOP)
ln_dict[11] = Line (658, P_FLAP_BOTTOM)
ln_dict[12] = Line (661, P_FLAP_WING_TOP)
ln_dict[13] = Line (659, P_FLAP_TE)
ln_dict[14] = Line (660, P_FLAP_TE)

#  Edges wing to corner point
ln_dict[34] = Line (657, P_WING_FLAP)
ln_dict[35] = Line (661, P_WING_TE)


#  Edges c mesh
ln_dict[17] = Line (657, 658)
ln_dict[26] = Line (661, 660)
ln_dict[24] = BSpline (657, 669, 670, 661)
ln_dict[25] = BSpline (658, 671, 672, 659)

ln_dict[19] = Line (P_FLAP_TE, 662)
ln_dict[20] = Line (659, 664)
ln_dict[21] = Line (660, 663)
ln_dict[22] = Line (662, 663)
ln_dict[23] = Line (662, 664)



#  Split the wing
l1:Line = ln_dict[1]
new_curves1 = l1.split(P_WING_TOP, P_WING_BOTTOM, P_WING_MIDDLE_BOTTOM, P_WING_TE, P_WING_FLAP)
new_idx = max(ln_dict.keys())
new_idx0 = new_idx
for c in new_curves1:
    ln_dict[new_idx+1] = c
    new_idx += 1

l36 = ln_dict[36]
l37 = ln_dict[37]
l38 = ln_dict[38]
l39 = ln_dict[39]
l40 = ln_dict[40]

ln_dict[40] = l36
ln_dict[36] = l37
ln_dict[37] = l38
ln_dict[38] = l39
ln_dict[39] = l40

# for i in np.arange(new_idx0+1, new_idx+1):
#     x, y = discretize(pt_dict, ln_dict[i], 50, 1)
#     plt.plot(x, y, label=f"{i}")
# plt.gca().set_aspect("equal")
# plt.legend()
# plt.show()
# exit()

#  Corner points wing c mesh
pt_dict[666] = Point (P_WING_BOTTOM_X - R_WING*np.cos(THETA_WING*np.pi/180), P_WING_BOTTOM_Y - R_WING*np.sin(THETA_WING*np.pi/180))
pt_dict[667] = Point (P_WING_TOP_X - R_WING*np.cos(THETA_WING*np.pi/180), P_WING_TOP_Y + R_WING*np.sin(THETA_WING*np.pi/180))
pt_dict[668] = Point (P_WING_TE_X, P_WING_TE_Y + R_WING)

# Spline control points
pt_dict[673] = Point (P_FLAP_LE_X -R_FLAP - SPLINE_R_FLAP*np.cos(SPLINE_THETA_FLAP*np.pi/180), (P_WING_FLAP_Y + P_FLAP_TOP_Y)/2 - SPLINE_R_FLAP*np.sin(SPLINE_THETA_FLAP*np.pi/180))
pt_dict[674] = Point (P_WING_BOTTOM_X - R_WING*np.cos(THETA_WING*np.pi/180) + SPLINE_R_WING_BOTTOM*np.sin(THETA_WING*np.pi/180), P_WING_BOTTOM_Y - R_WING*np.sin(THETA_WING*np.pi/180) - SPLINE_R_WING_BOTTOM*np.cos(THETA_WING*np.pi/180))

pt_dict[679] = Point (P_WING_TE_X - SPLINE_R_WING_TE*np.cos(SPLINE_THETA_WING_TE*np.pi/180), P_WING_TE_Y + R_WING + SPLINE_R_WING_TE*np.sin(SPLINE_THETA_WING_TE*np.pi/180))
pt_dict[680] = Point (P_WING_TOP_X - R_WING*np.cos(THETA_WING*np.pi/180) + SPLINE_R_WING*np.sin(THETA_WING*np.pi/180), P_WING_TOP_Y + R_WING*np.sin(THETA_WING*np.pi/180) + SPLINE_R_WING*np.cos(THETA_WING*np.pi/180))

pt_dict[681] = Point (P_WING_MIDDLE_BOTTOM_X, P_WING_MIDDLE_BOTTOM_Y - R_WING)
pt_dict[682] = Point (P_WING_MIDDLE_BOTTOM_X + 0.28, P_WING_MIDDLE_BOTTOM_Y - R_WING)
pt_dict[683] = Point (P_WING_MIDDLE_BOTTOM_X - 0.2, P_WING_MIDDLE_BOTTOM_Y - R_WING)
pt_dict[684] = Point (P_WING_BOTTOM_X - R_WING*np.cos(THETA_WING*np.pi/180) - SPLINE_R_WING_BOTTOM*np.sin(THETA_WING*np.pi/180), P_WING_BOTTOM_Y - R_WING*np.sin(THETA_WING*np.pi/180) + SPLINE_R_WING_BOTTOM*np.cos(THETA_WING*np.pi/180))
pt_dict[685] = Point (P_WING_TOP_X - R_WING*np.cos(THETA_WING*np.pi/180) - SPLINE_R_WING_TOP*np.sin(THETA_WING*np.pi/180), P_WING_TOP_Y + R_WING*np.sin(THETA_WING*np.pi/180) - SPLINE_R_WING_TOP*np.cos(THETA_WING*np.pi/180))

# Far field points wing mesh
pt_dict[675] = Point (D_FAR_FIELD, P_WING_TE_Y + (D_FAR_FIELD - P_FLAP_TE_X)*np.tan(THETA_FAR_FIELD*np.pi/180 / 2))
pt_dict[676] = Point (P_FLAP_TE_X, P_WING_TE_Y + (P_FLAP_TE_X - P_WING_TE_X)*np.tan(THETA_FAR_FIELD*np.pi/180 * 0.25))
pt_dict[677] = Point (P_FLAP_TE_X, P_WING_TE_Y + R_WING + (P_FLAP_TE_X - P_WING_TE_X)*np.tan(THETA_FAR_FIELD*np.pi/180))
pt_dict[678] = Point (D_FAR_FIELD, P_WING_TE_Y + R_WING + (D_FAR_FIELD - P_FLAP_TE_X)*np.tan(THETA_FAR_FIELD*np.pi/180))

# Edges wing to corner point
ln_dict[31] = Line (P_WING_BOTTOM, 666)
ln_dict[32] = Line (P_WING_TOP, 667)
ln_dict[44] = Line (660, 676)
ln_dict[45] = Line (P_WING_TE, 668)
ln_dict[48] = Line (676, 677)
ln_dict[49] = Line (675, 678)
ln_dict[50] = Line (675, 663)

# Edges c mesh
ln_dict[33] = BSpline (681, 682, 673, 657)
ln_dict[52] = BSpline (666, 674, 683, 681)
ln_dict[51] = BSpline (667, 680, 679, 668)
ln_dict[42] = Line (P_WING_TE, 676)
ln_dict[43] = Line (676, 675)
ln_dict[46] = Line (668, 677)
ln_dict[47] = Line (677, 678)

#  Point that controls the curvature of the front secod the inner C mesh and the spline it is the part of
pt_dict[686] = Point (- (P_WING_TOP_X + P_WING_BOTTOM_X) - R_WING, -(P_WING_TOP_Y + P_WING_BOTTOM_Y) / 2)
ln_dict[53] = BSpline (666, 684, 686, 685, 667)


#  Corner points wing outer c mesh
pt_dict[687] = Point (P_WING_BOTTOM_X - R_OUTER_WING*np.cos(THETA_WING*np.pi/180), P_WING_BOTTOM_Y - R_OUTER_WING*np.sin(THETA_WING*np.pi/180))
pt_dict[688] = Point (P_WING_TOP_X - R_OUTER_WING*np.cos(THETA_WING*np.pi/180), P_WING_TOP_Y + R_OUTER_WING*np.sin(THETA_WING*np.pi/180))
pt_dict[689] = Point (P_WING_TE_X, P_WING_TE_Y + R_OUTER_WING)
pt_dict[690] = Point (P_WING_MIDDLE_BOTTOM_X, P_WING_MIDDLE_BOTTOM_Y - R_OUTER_WING)
pt_dict[691] = Point (- (P_WING_TOP_X + P_WING_BOTTOM_X) - R_OUTER_WING, -(P_WING_TOP_Y + P_WING_BOTTOM_Y) / 2)
ln_dict[54] = Line (666, 687)
ln_dict[55] = Line (667, 688)
#  Bottom part of the outer wing C mesh that joins to the front of the flap
pt_dict[692] = Point (P_FLAP_LE_X - R_FLAP - 0.1, P_FLAP_LE_Y - R_FLAP - 0.1)
pt_dict[693] = Point (P_WING_BOTTOM_X - R_OUTER_WING*np.cos(THETA_WING*np.pi/180) + SPLINE_R_WING_BOTTOM_OUTER*np.sin(THETA_WING*np.pi/180), P_WING_BOTTOM_Y - R_OUTER_WING*np.sin(THETA_WING*np.pi/180) - SPLINE_R_WING_BOTTOM_OUTER*np.cos(THETA_WING*np.pi/180))
pt_dict[694] = Point (P_WING_MIDDLE_BOTTOM_X - (R_OUTER_WING), P_WING_MIDDLE_BOTTOM_Y - R_OUTER_WING)
pt_dict[695] = Point (P_WING_MIDDLE_BOTTOM_X + (R_OUTER_WING), P_WING_MIDDLE_BOTTOM_Y - R_OUTER_WING)
ln_dict[56] = BSpline (687, 693, 694, 690)
ln_dict[57] = BSpline (690, 695, 692, 658)



#  Front part of the curved outer wing c mesh
pt_dict[696] = Point (P_WING_BOTTOM_X - R_OUTER_WING*np.cos(THETA_WING*np.pi/180) - SPLINE_R_WING_BOTTOM_OUTER*np.sin(THETA_WING*np.pi/180), P_WING_BOTTOM_Y - R_OUTER_WING*np.sin(THETA_WING*np.pi/180) + SPLINE_R_WING_BOTTOM_OUTER*np.cos(THETA_WING*np.pi/180))
pt_dict[697] = Point (P_WING_TOP_X - R_OUTER_WING*np.cos(THETA_WING*np.pi/180) - SPLINE_R_WING_TOP_OUTER*np.sin(THETA_WING*np.pi/180), P_WING_TOP_Y + R_OUTER_WING*np.sin(THETA_WING*np.pi/180) - SPLINE_R_WING_TOP_OUTER*np.cos(THETA_WING*np.pi/180))
ln_dict[58] = BSpline (687, 696, 691, 697, 688)

#  Top part of the outer wing c mesh
pt_dict[698] = Point (P_WING_TE_X - SPLINE_R_WING_TE*np.cos(SPLINE_THETA_WING_TE*np.pi/180), P_WING_TE_Y + R_OUTER_WING + SPLINE_R_WING_TE*np.sin(SPLINE_THETA_WING_TE*np.pi/180))
pt_dict[699] = Point (P_WING_TOP_X - R_OUTER_WING*np.cos(THETA_WING*np.pi/180) + SPLINE_R_WING*np.sin(THETA_WING*np.pi/180), P_WING_TOP_Y + R_OUTER_WING*np.sin(THETA_WING*np.pi/180) + SPLINE_R_WING*np.cos(THETA_WING*np.pi/180))
ln_dict[59] = BSpline (688, 699, 698, 689)

#  Top part of the outer c mesh that goes to the far wake
pt_dict[700] = Point (D_FAR_FIELD, P_WING_TE_Y + R_OUTER_WING + (D_FAR_FIELD - P_FLAP_TE_X)*np.tan((THETA_FAR_FIELD * 1.2)*np.pi/180))



#  Points and lines to form and connect to the outer wake
Y_FAR_BOTTOM = P_WING_BOTTOM_Y - (D_FAR_FRONT - P_WING_BOTTOM_X)*np.tan(THETA_WING*np.pi/180)
Y_FAR_TOP = P_WING_TOP_Y + (D_FAR_FRONT - P_WING_TOP_X) * np.tan(THETA_WING*np.pi/180)
pt_dict[701] = Point (-D_FAR_FRONT, Y_FAR_BOTTOM)
pt_dict[702] = Point (-D_FAR_FRONT, Y_FAR_TOP)
pt_dict[703] = Point (D_FAR_FIELD, Y_FAR_BOTTOM)
pt_dict[704] = Point (D_FAR_FIELD, Y_FAR_TOP)
pt_dict[705] = Point (P_FLAP_LE_X - R_FLAP, Y_FAR_BOTTOM)
pt_dict[706] = Point (P_FLAP_TE_X + 1, Y_FAR_BOTTOM)
pt_dict[707] = Point (P_WING_TE_X-0.3, Y_FAR_TOP)
pt_dict[708] = Point (P_FLAP_TE_X, P_WING_TE_Y + R_OUTER_WING + (P_FLAP_TE_X -  P_WING_TE_X)*np.tan((THETA_FAR_FIELD/2)*np.pi/180))
pt_dict[709] = Point (P_FLAP_TE_X + 0.2, Y_FAR_TOP)
pt_dict[711] = Point (P_WING_MIDDLE_BOTTOM_X - 0.75, Y_FAR_BOTTOM)
ln_dict[62] = Line (687, 701)
ln_dict[63] = Line (688, 702)
ln_dict[64] = Line (703, 664)
ln_dict[66] = Line (704, 700)
ln_dict[68] = Line (705, 658)
ln_dict[69] = Line (705, 706)
ln_dict[70] = Line (706, 659)
ln_dict[71] = Line (706, 703)
ln_dict[72] = Line (702, 707)
ln_dict[73] = Line (707, 689)
ln_dict[75] = Line (704, 700)
ln_dict[76] = Line (668, 689)
ln_dict[77] = Line (689, 708)
ln_dict[78] = Line (677, 708)
ln_dict[65] = Line (708, 700)
ln_dict[79] = Line (707, 709)
ln_dict[80] = Line (709, 704)
ln_dict[74] = Line (708, 709)
ln_dict[82] = Line (700, 678)
ln_dict[83] = Line (137, 681)
ln_dict[84] = Line (681, 690)
ln_dict[85] = Line (690, 711)
ln_dict[86] = Line (711, 705)
ln_dict[87] = Line (711, 701)

#  Center of the front arc and the arc itself
pt_dict[710] = Point (-D_FAR_FRONT - (Y_FAR_TOP - Y_FAR_BOTTOM) / (2 * np.tan((90 - THETA_FRONT) * np.pi / 180)), (Y_FAR_TOP + Y_FAR_BOTTOM) / 2)
ln_dict[81] = BSpline (701, 710, 702)


P_FLAP_MF = 570
P_FLAP_MF_COORDS = pt_dict[P_FLAP_MF]
P_FLAP_MF_X = P_FLAP_MF_COORDS.x
P_FLAP_MF_Y = P_FLAP_MF_COORDS.y
pt_dict[712] = Point (P_FLAP_MF_X, P_FLAP_MF_Y + 0.01)

P_FLAP_BF = 610
P_FLAP_BF_COORDS = pt_dict[P_FLAP_BF]
P_FLAP_BF_X = P_FLAP_BF_COORDS.x
P_FLAP_BF_Y = P_FLAP_BF_COORDS.y
pt_dict[713] = Point (P_FLAP_BF_X, P_FLAP_BF_Y + 0.03)
ln_dict[26] = BSpline (661, 712, 713, 660)


N_OUTER = 32
N_MIDDLE = 16  #  Also goes to the front of the flap
N_INNER = 80
N_FLAP = 8     #  Around the flap
N_FRONT = 16
N_WAKE = 128
N_FLAP_TOP = 96
N_FLAP_BOTTOM = 24
N_WING_TOP = 48
N_WING_BOTTOM_FRONT = 48
N_WING_BOTTOM_BACK = 64
N_TRANSITION = 5

#  Discratization parameters


tc_dict[66] = transfinite_curve(pt_dict, ln_dict[66], N_OUTER, 0.97)
tc_dict[74] = transfinite_curve(pt_dict, ln_dict[74], N_OUTER, 1.115)
tc_dict[73] = transfinite_curve(pt_dict, ln_dict[73], N_OUTER, 0.925)
tc_dict[63] = transfinite_curve(pt_dict, ln_dict[63], N_OUTER, 1.1)
tc_dict[62] = transfinite_curve(pt_dict, ln_dict[62], N_OUTER, 1.1)
tc_dict[68] = transfinite_curve(pt_dict, ln_dict[68], N_OUTER, 0.9)
tc_dict[70] = transfinite_curve(pt_dict, ln_dict[70], N_OUTER, 0.875)
tc_dict[64] = transfinite_curve(pt_dict, ln_dict[64], N_OUTER, 0.975)
tc_dict[85] = transfinite_curve(pt_dict, ln_dict[85], N_OUTER, 1.1)


tc_dict[81] = transfinite_curve(pt_dict, ln_dict[81], N_FRONT, 1)
tc_dict[58] = transfinite_curve(pt_dict, ln_dict[58], N_FRONT, 1)
tc_dict[53] = transfinite_curve(pt_dict, ln_dict[53], N_FRONT, 3)
tc_dict[38] = transfinite_curve(pt_dict, ln_dict[38], N_FRONT, 4)


tc_dict[80] = transfinite_curve(pt_dict, ln_dict[80], N_WAKE, 1.03)
tc_dict[65] = transfinite_curve(pt_dict, ln_dict[65], N_WAKE, 1.07)
tc_dict[47] = transfinite_curve(pt_dict, ln_dict[47], N_WAKE, 1.07)
tc_dict[43] = transfinite_curve(pt_dict, ln_dict[43], N_WAKE, 1.07)
tc_dict[21] = transfinite_curve(pt_dict, ln_dict[21], N_WAKE, 1.06)
tc_dict[19] = transfinite_curve(pt_dict, ln_dict[19], N_WAKE, 1.06)
tc_dict[20] = transfinite_curve(pt_dict, ln_dict[20], N_WAKE, 1.06)
tc_dict[71] = transfinite_curve(pt_dict, ln_dict[71], N_WAKE, 1.03)


tc_dict[79] = transfinite_curve(pt_dict, ln_dict[79], N_FLAP_TOP, 1.0)
tc_dict[77] = transfinite_curve(pt_dict, ln_dict[77], N_FLAP_TOP, 1)
tc_dict[46] = transfinite_curve(pt_dict, ln_dict[46], N_FLAP_TOP, 1)
tc_dict[42] = transfinite_curve(pt_dict, ln_dict[42], N_FLAP_TOP, 1)
tc_dict[26] = transfinite_curve(pt_dict, ln_dict[26], N_FLAP_TOP, 0.995)
tc_dict[5] = transfinite_curve(pt_dict, ln_dict[5], N_FLAP_TOP, 1)


tc_dict[6] = transfinite_curve(pt_dict, ln_dict[6], N_FLAP_BOTTOM, 0.925)
tc_dict[25] = transfinite_curve(pt_dict, ln_dict[25], N_FLAP_BOTTOM, 1.045)
tc_dict[69] = transfinite_curve(pt_dict, ln_dict[69], N_FLAP_BOTTOM, 1)


tc_dict[87] = transfinite_curve(pt_dict, ln_dict[87], N_WING_BOTTOM_FRONT, 1.05)
tc_dict[56] = transfinite_curve(pt_dict, ln_dict[56], N_WING_BOTTOM_FRONT, 1.00)
tc_dict[52] = transfinite_curve(pt_dict, ln_dict[52], N_WING_BOTTOM_FRONT, 1.0)
tc_dict[37] = transfinite_curve(pt_dict, ln_dict[37], N_WING_BOTTOM_FRONT, 1)


tc_dict[36] = transfinite_curve(pt_dict, ln_dict[36], N_WING_BOTTOM_BACK, 1.02)
tc_dict[33] = transfinite_curve(pt_dict, ln_dict[33], N_WING_BOTTOM_BACK, 0.97)
tc_dict[57] = transfinite_curve(pt_dict, ln_dict[57], N_WING_BOTTOM_BACK, 0.97)
tc_dict[86] = transfinite_curve(pt_dict, ln_dict[86], N_WING_BOTTOM_BACK, 1)


tc_dict[39] = transfinite_curve(pt_dict, ln_dict[39], N_WING_TOP, 1.01)
tc_dict[51] = transfinite_curve(pt_dict, ln_dict[51], N_WING_TOP, 1.0075)
tc_dict[59] = transfinite_curve(pt_dict, ln_dict[59], N_WING_TOP, 1.0055)
tc_dict[72] = transfinite_curve(pt_dict, ln_dict[72], N_WING_TOP, 0.98)


tc_dict[82] = transfinite_curve(pt_dict, ln_dict[82], N_MIDDLE, 0.99)
tc_dict[78] = transfinite_curve(pt_dict, ln_dict[78], N_MIDDLE, 1.0)
tc_dict[76] = transfinite_curve(pt_dict, ln_dict[76], N_MIDDLE, 1.07)
tc_dict[55] = transfinite_curve(pt_dict, ln_dict[55], N_MIDDLE, 1.07)
tc_dict[54] = transfinite_curve(pt_dict, ln_dict[54], N_MIDDLE, 1.07)
tc_dict[17] = transfinite_curve(pt_dict, ln_dict[17], N_MIDDLE, 1.07)
tc_dict[3] = transfinite_curve(pt_dict, ln_dict[3], N_MIDDLE, 1)
tc_dict[84] = transfinite_curve(pt_dict, ln_dict[84], N_MIDDLE, 1.07)


tc_dict[32] = transfinite_curve(pt_dict, ln_dict[32], N_INNER, 1.075)
tc_dict[31] = transfinite_curve(pt_dict, ln_dict[31], N_INNER, 1.075)
tc_dict[45] = transfinite_curve(pt_dict, ln_dict[45], N_INNER, 1.075)
tc_dict[34] = transfinite_curve(pt_dict, ln_dict[34], N_INNER, 0.98)
tc_dict[35] = transfinite_curve(pt_dict, ln_dict[35], N_INNER, 0.98)
tc_dict[44] = transfinite_curve(pt_dict, ln_dict[44], N_INNER, 0.97)
tc_dict[48] = transfinite_curve(pt_dict, ln_dict[48], N_INNER, 1.025)
tc_dict[49] = transfinite_curve(pt_dict, ln_dict[49], N_INNER, 1)
tc_dict[50] = transfinite_curve(pt_dict, ln_dict[50], N_INNER, 1)
tc_dict[83] = transfinite_curve(pt_dict, ln_dict[83], N_INNER, 1.075)


tc_dict[23] = transfinite_curve(pt_dict, ln_dict[23], N_FLAP, 1)
tc_dict[22] = transfinite_curve(pt_dict, ln_dict[22], N_FLAP, 1)
tc_dict[14] = transfinite_curve(pt_dict, ln_dict[14], N_FLAP, 0.925)
tc_dict[13] = transfinite_curve(pt_dict, ln_dict[13], N_FLAP, 0.95)
tc_dict[10] = transfinite_curve(pt_dict, ln_dict[10], N_FLAP, 0.975)
tc_dict[11] = transfinite_curve(pt_dict, ln_dict[11], N_FLAP, 0.95)
tc_dict[12] = transfinite_curve(pt_dict, ln_dict[12], N_FLAP, 0.95)


tc_dict[40] = transfinite_curve(pt_dict, ln_dict[40], N_TRANSITION, 0.825)
tc_dict[24] = transfinite_curve(pt_dict, ln_dict[24], N_TRANSITION, 1.25)
tc_dict[4] = transfinite_curve(pt_dict, ln_dict[4], N_TRANSITION, 1.45)



b_dict[1] = make_bblock_from_tc(1, tc_dict, 63, -81, -62, 58)
b_dict[2] = make_bblock_from_tc(2, tc_dict, -56, 62, -87,-85)


b_dict[2].set_boundary(BoundaryId.BoundaryWest, None)
b_dict[2].set_boundary(BoundaryId.BoundaryWest, BoundaryBlock("1", BoundaryId.BoundarySouth))
b_dict[1].set_boundary(BoundaryId.BoundarySouth, None)
b_dict[1].set_boundary(BoundaryId.BoundarySouth, BoundaryBlock("2", BoundaryId.BoundaryWest))


b_dict[3] = make_bblock_from_tc(3, tc_dict,85, 86, 68, -57)

b_dict[2].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[2].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("3", BoundaryId.BoundaryNorth))
b_dict[3].set_boundary(BoundaryId.BoundaryNorth, None)
b_dict[3].set_boundary(BoundaryId.BoundaryNorth, BoundaryBlock("2", BoundaryId.BoundaryEast))


b_dict[4] = make_bblock_from_tc(4, tc_dict,-68, 69, 70, -25)


b_dict[3].set_boundary(BoundaryId.BoundarySouth, None)
b_dict[3].set_boundary(BoundaryId.BoundarySouth, BoundaryBlock("4", BoundaryId.BoundaryNorth))
b_dict[4].set_boundary(BoundaryId.BoundaryNorth, None)
b_dict[4].set_boundary(BoundaryId.BoundaryNorth, BoundaryBlock("3", BoundaryId.BoundarySouth))


b_dict[5] = make_bblock_from_tc(5, tc_dict,71, 64, -20, -70)


b_dict[4].set_boundary(BoundaryId.BoundarySouth, None)
b_dict[4].set_boundary(BoundaryId.BoundarySouth, BoundaryBlock("5", BoundaryId.BoundaryEast))
b_dict[5].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[5].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("4", BoundaryId.BoundarySouth))


b_dict[6] = make_bblock_from_tc(6, tc_dict, -23, -19,  -13, +20)


b_dict[5].set_boundary(BoundaryId.BoundarySouth, None)
b_dict[5].set_boundary(BoundaryId.BoundarySouth, BoundaryBlock("6", BoundaryId.BoundaryEast))
b_dict[6].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[6].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("5", BoundaryId.BoundarySouth))


b_dict[7] = make_bblock_from_tc(7, tc_dict, 14, 19, 22, -21)


b_dict[6].set_boundary(BoundaryId.BoundaryWest, None)
b_dict[6].set_boundary(BoundaryId.BoundaryWest, BoundaryBlock("7", BoundaryId.BoundaryWest))
b_dict[7].set_boundary(BoundaryId.BoundaryWest, None)
b_dict[7].set_boundary(BoundaryId.BoundaryWest, BoundaryBlock("6", BoundaryId.BoundaryWest))


b_dict[8] = make_bblock_from_tc(8, tc_dict,-44, -21, -50, -43)


b_dict[7].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[7].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("8", BoundaryId.BoundaryWest))
b_dict[8].set_boundary(BoundaryId.BoundaryWest, None)
b_dict[8].set_boundary(BoundaryId.BoundaryWest, BoundaryBlock("7", BoundaryId.BoundaryEast))


b_dict[9] = make_bblock_from_tc(9, tc_dict,49, -47, -48, -43)


b_dict[8].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[8].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("9", BoundaryId.BoundaryEast))
b_dict[9].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[9].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("8", BoundaryId.BoundaryEast))


b_dict[10] = make_bblock_from_tc(10, tc_dict,-78, 47, -82, -65)


b_dict[9].set_boundary(BoundaryId.BoundaryWest, None)
b_dict[9].set_boundary(BoundaryId.BoundaryWest, BoundaryBlock("10", BoundaryId.BoundaryWest))
b_dict[10].set_boundary(BoundaryId.BoundaryWest, None)
b_dict[10].set_boundary(BoundaryId.BoundaryWest, BoundaryBlock("9", BoundaryId.BoundaryWest))


b_dict[11] = make_bblock_from_tc(11, tc_dict, -66, -80, -74, 65)


b_dict[10].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[10].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("11", BoundaryId.BoundaryEast))
b_dict[11].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[11].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("10", BoundaryId.BoundaryEast))


b_dict[12] = make_bblock_from_tc(12, tc_dict,77, -74, -79, 73)


b_dict[11].set_boundary(BoundaryId.BoundarySouth, None)
b_dict[11].set_boundary(BoundaryId.BoundarySouth, BoundaryBlock("12", BoundaryId.BoundaryWest))
b_dict[12].set_boundary(BoundaryId.BoundaryWest, None)
b_dict[12].set_boundary(BoundaryId.BoundaryWest, BoundaryBlock("11", BoundaryId.BoundarySouth))


b_dict[13] = make_bblock_from_tc(13, tc_dict,-72, -63, +59, -73)


b_dict[12].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[12].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("13", BoundaryId.BoundaryEast))
b_dict[13].set_boundary(BoundaryId.BoundaryEast, None)
b_dict[13].set_boundary(BoundaryId.BoundaryEast, BoundaryBlock("12", BoundaryId.BoundaryEast))


b_dict[13].set_boundary(BoundaryId.BoundaryWest, None)
b_dict[13].set_boundary(BoundaryId.BoundaryWest, BoundaryBlock("1", BoundaryId.BoundaryNorth))
b_dict[1].set_boundary(BoundaryId.BoundaryNorth, None)
b_dict[1].set_boundary(BoundaryId.BoundaryNorth, BoundaryBlock("13", BoundaryId.BoundaryWest))


b_dict[14] = make_bblock_from_tc(14, tc_dict,-59, -55, 51, 76)
b_dict[15] = make_bblock_from_tc(15, tc_dict,-77, -76, 46, 78)
b_dict[16] = make_bblock_from_tc(16, tc_dict,55, -58, -54, 53)
b_dict[17] = make_bblock_from_tc(17, tc_dict,56, -84, -52, 54)
b_dict[18] = make_bblock_from_tc(18, tc_dict,57, -17, -33, 84)
b_dict[19] = make_bblock_from_tc(19, tc_dict,25, 13, 6, -11)

b_dict[20] = make_bblock_from_tc(20, tc_dict,-14, -26, 12, 5)
# b_dict[20] = make_bblock_from_tc(20, tc_dict,14, -5, -12, 26)

b_dict[21] = make_bblock_from_tc(21, tc_dict,-24, 10, 4, -12)
# b_dict[21] = make_bblock_from_tc(21, tc_dict,24, 12, -4, -10)
b_dict[22] = make_bblock_from_tc(22, tc_dict,-42,-35, 26, 44)

b_dict[23] = make_bblock_from_tc(23, tc_dict,40, -34, 24, 35)
# b_dict[23] = make_bblock_from_tc(23, tc_dict,-40, -35, -24, 34)
b_dict[24] = make_bblock_from_tc(24, tc_dict,33, 34, 36, 83)

b_dict[25] = make_bblock_from_tc(25, tc_dict,-83, 37, 31, 52)
b_dict[26] = make_bblock_from_tc(26, tc_dict,38, 32, -53, -31)
b_dict[27] = make_bblock_from_tc(27, tc_dict,-51, -32, 39, 45)

b_dict[28] = make_bblock_from_tc(28, tc_dict,-46, -45, 42, 48)
b_dict[29] = make_bblock_from_tc(29, tc_dict,-10, 17, 11, 3)
# b_dict[29] = make_bblock_from_tc(29, tc_dict,10, -3, -11, -17)


def have_common_points(bnd1: BoundaryCurve, bnd2: BoundaryCurve) -> bool:
    p11 = (bnd1.x[0], bnd1.y[0])
    p12 = (bnd1.x[-1], bnd1.y[-1])
    p21 = (bnd2.x[0], bnd2.y[0])
    p22 = (bnd2.x[-1], bnd2.y[-1])
    return (np.allclose(p11, p21) and np.allclose(p12, p22)) or (np.allclose(p11, p22) and np.allclose(p12, p21))


skip = [19, 20, 21, 23, 24, 25, 26, 27, 28, 29]

for bname1 in b_dict:
    b1 = b_dict[bname1]
    for bid1 in b1.boundaries:
        bnd1 = b1.boundaries[bid1]
        if type(bnd1) == BoundaryCurve:
            bnd1: BoundaryCurve
            for bname2 in b_dict:
                if bname1 == bname2:
                    continue
                b2 = b_dict[bname2]
                for bid2 in b2.boundaries:
                    bnd2 = b2.boundaries[bid2]
                    if type(bnd2) == BoundaryCurve:
                        bnd2: BoundaryCurve
                        if have_common_points(bnd1, bnd2):
                            if bname1 not in skip:
                                b1.boundaries[bid1] = BoundaryBlock(f"{bname2}", bid2)
                            if bname2 not in skip:
                                b2.boundaries[bid2] = BoundaryBlock(f"{bname1}", bid1)

m, rx, ry = create_elliptical_mesh([b for b in b_dict.values()], verbose=True)

x = m.x
y = m.y
ln = m.lines
surf = m.surfaces
ncols = len(m.block_names)
cmap = plt.colormaps.get_cmap("jet")
#show_list = [str(j) for j in np.arange(22, 28) + 1]
xn = np.zeros(len(pt_dict))
yn = np.zeros(len(pt_dict))

for i, p in enumerate(pt_dict):
    pt = pt_dict[p]
    xn[i] = pt.x
    yn[i] = pt.y

plt.gca().set_aspect("equal")
plt.scatter(xn, yn)
i = 0

for bname1 in m.block_names:
    neighbors = []
    lines1 = np.abs(m.block_lines(bname1))
    for bname2 in m.block_names:
        if bname1 == bname2:
            continue
        lines2 = np.abs(m.block_lines(bname2))
        l12 = np.unique(np.concatenate((lines1, lines2)))
        if l12.shape[0] != lines1.shape[0] + lines2.shape[0]:
            neighbors.append(bname2)
    print(f"Block {bname1} has {len(neighbors)} neighbors: {neighbors}")

for bname in m.block_names:
    line_indices = np.abs(m.block_lines(bname)) - 1
    block_lines = ln[line_indices]
    xb = x[block_lines[:, 0]]
    yb = y[block_lines[:, 0]]
    xe = x[block_lines[:, 1]]
    ye = y[block_lines[:, 1]]
    rb = np.stack((xb, yb), axis=1)
    re = np.stack((xe, ye), axis=1)
    lnvals = np.stack((rb, re), axis=1)
    plt.gca().add_collection(col.LineCollection(lnvals, color=cmap((i/ncols)), label=bname))
    i += 1

bname = m.block_names[0]

north_node_indices = m.block_boundary_points(bname, BoundaryId.BoundaryNorth)
nnorth = len(north_node_indices)
south_node_indices = m.block_boundary_points(bname, BoundaryId.BoundarySouth)
nsouth = len(south_node_indices)
east_node_indices = m.block_boundary_points(bname, BoundaryId.BoundaryEast)
neast = len(east_node_indices)
west_node_indices = m.block_boundary_points(bname, BoundaryId.BoundaryWest)
nwest = len(west_node_indices)
cmap = plt.colormaps.get_cmap("magma")

for i, idx in enumerate(north_node_indices):
    plt.scatter(x[idx], y[idx], color=cmap(i/(nnorth-1)))
for i, idx in enumerate(south_node_indices):
    plt.scatter(x[idx], y[idx], color=cmap(i/(nsouth-1)))
for i, idx in enumerate(east_node_indices):
    plt.scatter(x[idx], y[idx], color=cmap(i/(neast-1)))
for i, idx in enumerate(west_node_indices):
    plt.scatter(x[idx], y[idx], color=cmap(i/(nwest-1)))


bname = m.block_names[0]

surf_indices = m.block_boundary_surfaces(bname, BoundaryId.BoundaryWest) - 1
nsurf = len(surf_indices)
vertices = []
for i, idx in enumerate(surf_indices):
    line_indices = abs(surf[idx]) - 1
    lines = ln[line_indices, :]
    point_indices = lines[:, 0] * (surf[idx] > 0) + lines[:, 1] * (surf[idx] < 0)
    xvals = x[point_indices]
    yvals = y[point_indices]
    pts = np.stack((xvals, yvals), axis=1)
    vertices.append(pts)
vertices = np.array(vertices)
plt.gca().add_collection(col.PolyCollection(vertices, color="green"))

print(m.surface_element(10, 0))
print(m.surface_element(10, 1))
print(m.surface_element(10, 2))
print(m.surface_element(10, 3))

plt.legend()

plt.show()
