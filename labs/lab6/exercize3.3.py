import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm


def driver():
    x0 = [1, 0]
    tol = 10 ** (-10)
    h = [1e-1, 1e-3, 1e-5, 1e-7, 1e-11, 1e-13]

    print("now running newtons for various h:\n")
    for i in h:
        print("h = ", i)
        print(finite_Newton(x0, tol, 100, i))
        print()

    print("\n")
    print("now running normal newtons method:\n")
    print(Newton(x0, tol, 100))
    print("\n")

    return


def evalF(x):
    F = np.zeros(2)
    F[0] = 4 * x[0] ** 2 + x[1] ** 2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0] - x[1])
    return F


def finite_evalJ(x, h):
    f1 = lambda x: 4 * x[0] ** 2 + x[1] ** 2 - 4
    f2 = lambda x: x[0] + x[1] - np.sin(x[0] - x[1])
    F = [f1, f2]
    [j11, j12] = forwardDifference(F[0], h, x)
    [j21, j22] = forwardDifference(F[1], h, x)
    J = np.array([[j11, j12], [j21, j22]])

    return J


def evalJ(x):
    J = np.array(
        [[8 * x[0], 2 * x[1]], [1 - np.cos(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]]
    )

    return J


def forwardDifference(f, h, s):
    f_s = f(s)
    approx_x = f([s[0] + h, s[1]]) - f_s
    approx_y = f([s[0], s[1] + h]) - f_s

    return [approx_x / h, approx_y / h]


def finite_Newton(x0, tol, Nmax, h):
    """inputs: x0 = initial guess, tol = tolerance, Nmax = max its, h for finite evaluation of J"""
    """ Outputs: xstar= approx root, ier = error message, its = num its"""
    for its in range(Nmax):
        J = finite_evalJ(x0, h)
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            print(
                "converged after",
                its,
                "iterations.",
            )
            return [xstar, ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier, its]


def Newton(x0, tol, Nmax):
    """inputs: x0 = initial guess, tol = tolerance, Nmax = max its"""
    """ Outputs: xstar= approx root, ier = error message, its = num its"""
    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            print(
                "converged after",
                its,
                "iterations.",
            )
            return [xstar, ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier, its]


print("\n")
driver()
print("\n")
