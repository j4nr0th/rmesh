import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    data1 = np.loadtxt("cmake-build-debug/out.dat")
    data2 = np.loadtxt("cmake-build-debug/out1.dat")

    plt.figure()
    plt.gca().set_aspect("equal")
    plt.scatter(data1[:, 0], data1[:, 1])
    plt.show()


    plt.figure()
    plt.gca().set_aspect("equal")
    plt.scatter(data2[:50, 0], data2[:50, 1], marker="x")
    plt.scatter(data2[50:, 0], data2[50:, 1], marker="o")
    plt.show()

    data3 = np.loadtxt("cmake-build-debug/bigger.dat")
    plt.gca().set_aspect("equal")
    plt.scatter(data3[:, 0], data3[:, 1], marker="x")
    plt.show()
