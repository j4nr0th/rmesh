import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    nodal_data = np.loadtxt("cmake-build-debug/bigger_pts.dat")
    line_data = np.loadtxt("cmake-build-debug/bigger_lns.dat")
    plt.gca().set_aspect("equal")
    x = nodal_data[:, 0]
    y = nodal_data[:, 1]
    n1 = np.array(line_data[:, 0], dtype=int)
    n2 = np.array(line_data[:, 1], dtype=int)
    x1 = x[n1]
    x2 = x[n2]
    y1 = y[n1]
    y2 = y[n2]
    for i in range(len(n1)):
        plt.plot((x1[i], x2[i]), (y1[i], y2[i]), color="black", linestyle="dashed")
    plt.scatter(x, y, marker="o", color="red")
    plt.show()
