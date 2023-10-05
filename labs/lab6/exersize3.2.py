import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm


def driver():
    x0 = [1, 0]
    tol = 10 ** (-10)

    print("now running lazy newtons:")
    print(LazyNewton(x0, tol, 100))
    print("\n")

    print("now running slacker newtons:")
    print(slackerNewton(x0, tol, 100))

    return


def evalF(x):
    F = np.zeros(2)
    F[0] = 4 * x[0] ** 2 + x[1] ** 2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0] - x[1])
    return F


def evalJ(x):
    J = np.array(
        [[8 * x[0], 2 * x[1]], [1 - np.cos(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]]
    )

    return J


def LazyNewton(x0, tol, Nmax):
    """Lazy Newton = use only the inverse of the Jacobian for initial guess"""
    """ inputs: x0 = initial guess, tol = tolerance, Nmax = max its"""
    """ Outputs: xstar= approx root, ier = error message, its = num its"""
    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            print("converged after", its, "iterations")
            return [xstar, ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier, its]


def slackerNewton(x0, tol, Nmax):
    """Slacker Newton = update Jacobian every 3 iteration"""
    """ inputs: x0 = initial guess, tol = tolerance Nmax = max its"""
    """ Outputs: xstar= approx root, ier = error message, its = num its"""

    count = 0

    for its in range(Nmax):
        if its % 3 == 0:
            J = evalJ(x0)
            Jinv = inv(J)
            count += 1

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        dif = norm(x1 - x0)
        if dif < tol:
            xstar = x1
            ier = 0
            print(
                "converged after",
                its,
                "iterations with",
                count,
                "jacobian calculations.",
            )
            return [xstar, ier, its, count]

        x0 = x1

    xstar = x1
    ier = 1
    print(
        "unable to converge after",
        Nmax,
        "iterations with ",
        count,
        "jacobian calculations.",
    )
    return [xstar, ier, its, count]


print("\n")
driver()
print("\n")
