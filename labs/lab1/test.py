import numpy as np
import math


def driver():
    x = np.matrix([[1, 2], [3, 4], [5, 6]])
    y = np.matrix([[1, 3], [2, 4]])
    print("x: \n", x, "\nshape: ", x.shape, "\n\n")
    print("y: \n", y, "\nshape: ", y.shape, "\n\n")

    new = np.zeros((x.shape[0], y.shape[1]))

    vector_length = x.shape[1]
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            new[i, j] = dotProductMatrix(x[i, :], y[:, j].transpose(), vector_length)

    print("new: \n", new, "\nshape: ", new.shape, "\n\n")
    return


def dotProductMatrix(x, y, n):
    dp = 0.0
    print("here", x, y)
    for j in range(n):
        dp = dp + x[0, j] * y[0, j]

    return dp


driver()
