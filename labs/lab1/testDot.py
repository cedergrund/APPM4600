import numpy as np
import numpy.linalg as la
import math


def driver():
    # original code

    n = 100
    x = np.linspace(0, np.pi, n)

    # this is a function handle.  You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.
    f = lambda x: x**2 + 4 * x + 2 * np.exp(x)
    g = lambda x: 6 * x**3 + 2 * np.sin(x)

    y = f(x)
    w = g(x)

    # evaluate the dot product of y and w
    dp = dotProduct(y, w, n)

    # print the output
    print("\nthe dot product is : ", dp)

    # 4.2.1 added code - orthogonal vectors
    new_y = [0, 1, 0]
    new_w = [1, 0, 1]
    new_dp = dotProduct(new_y, new_w, 3)
    print("\nthe orthogonal dot product is : ", new_dp)

    # 4.2.2 & 4.2.3 matrix multiplication

    matrix_x = np.matrix([[1, 2], [3, 4]])
    matrix_y = np.matrix([[1, 3], [2, 4]])

    print("my method gives: \n", matrixMult(matrix_x, matrix_y))
    print("built-in method gives: \n", np.matmul(matrix_x, matrix_y))

    return


def dotProduct(x, y, n):
    dp = 0.0
    for j in range(n):
        dp = dp + x[j] * y[j]

    return dp


# dot product method using in matrix multiplication
def dotProduct4Matrix(x, y, n):
    dp = 0.0
    for j in range(n):
        dp = dp + x[0, j] * y[0, j]

    return dp


# basic driver function for matrix multiplication
def matrixMult(x, y):
    if x.shape[1] != y.shape[0]:
        print("Cannot matrix multiply with matrices of these shapes. Try again")
        return np.zeros((1, 1))

    new = np.zeros((x.shape[0], y.shape[1]))
    vector_length = x.shape[1]

    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            new[i, j] = dotProduct4Matrix(x[i, :], y[:, j].transpose(), vector_length)
    return new


driver()
