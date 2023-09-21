# import libraries
import numpy as np


def driver():
    # functions
    f = lambda x: np.sqrt(10 / (x + 4))
    # fixed point is p = 1.3652300134140976...

    Nmax = 100
    tol = 1e-10

    # run fixed point
    x0 = 1.5
    print("")
    [xstar, ier, iteration] = fixedpt(f, x0, tol, Nmax)
    print("the approximate fixed point is:", xstar)
    print("f1(xstar):", f(xstar))
    print("Error message reads:", ier)
    print(
        "approximations of fixed points at all",
        len(iteration),
        "iterations:\n",
        iteration,
    )
    print("\n")

    # if fixed point iteration did not work, don't test for aitken's method
    if ier == 1:
        return

    # printing order of convergence for fixed point method
    print(
        "running order of convergence check for fixed point method: \n",
        orderOfConvergence(xstar, iteration, alpha=1),
        "\n",
    )

    steffenson_seq = steffensonsMethodSequence(x0, f, tol, Nmax)
    # printing aikten's method sequence
    print(
        "running steffensons's method implementation: \n",
        steffenson_seq,
        "\n",
    )

    # printing new order of convergence
    print(
        "running order of convergence check for steffensons's method: \n",
        orderOfConvergence(xstar, steffenson_seq, alpha=1),
        "\n",
    )

    return


# define routines
def fixedpt(f, x0, tol, Nmax):
    """x0 = initial guess"""
    """ Nmax = max number of iterations"""
    """ tol = stopping tolerance"""
    iterations = np.zeros(Nmax)
    iterations[0] = x0

    count = 0
    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        iterations[count] = x1
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, iterations[:count]]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, iterations]


def steffensonsMethodSequence(x0, g, tolerance, Nmax):
    seq = np.zeros(Nmax)
    seq[0] = x0

    for i in range(Nmax - 1):
        a = x0
        b = g(x0)
        c = g(b)

        x1 = a - ((b - a) ** 2) / (c - 2 * b + a)
        seq[i + 1] = x1
        if np.abs(x1 - x0) < tolerance:
            return seq[: i + 2]
        x0 = x1

    return seq


def orderOfConvergence(fixed_point, iterations, alpha):
    seq = np.zeros(len(iterations) - 1)

    for n, i in enumerate(iterations):
        if n == len(iterations) - 1:
            continue

        seq[n] = (
            np.abs(iterations[n + 1] - fixed_point) / (np.abs(i - fixed_point)) ** alpha
        )

    return seq


driver()
