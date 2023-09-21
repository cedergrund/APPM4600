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

    aitken_seq = aitkensMethodSequence(iteration, tol, Nmax)
    # printing aikten's method sequence
    print(
        "running aitken's method implementation: \n",
        aitken_seq,
        "\n",
    )

    # printing new order of convergence
    print(
        "running order of convergence check for aitken's method: \n",
        orderOfConvergence(xstar, aitken_seq, alpha=1),
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


def aitkensMethodSequence(input_sequence, tolerance, Nmax):
    seq = np.zeros(len(input_sequence) - 2)

    for n, i in enumerate(input_sequence):
        if n > Nmax or n >= len(seq):
            return seq

        seq[n] = i - (
            ((input_sequence[n + 1] - i) ** 2)
            / (input_sequence[n + 2] - 2 * input_sequence[n + 1] + i)
        )

        if n != 0:
            if np.abs(seq[n] - seq[n - 1]) < tolerance:
                return seq[: n + 1]

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
