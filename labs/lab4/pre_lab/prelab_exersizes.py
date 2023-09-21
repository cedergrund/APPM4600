# import libraries
import numpy as np

# technique to check order of convergence:
#  run definition 2.1 algorithm to develop sequence using approximation vector generated with new fixed point subroutine
#  if sequence trends along value less than 1 for a certain alpha value, it converges with that alpha value as rate of convergence

# 2.2.2a
# takes 12 iteration for fixed point iteration to converge

# 2.2.2b
# order of convergence is linear with asymptotic error constant ~0.127


def driver():
    # functions
    f = lambda x: (10 / (x + 4)) ** (1 / 2)
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

    # if fixed point iteration did not work, don't test for rate of convergence
    if ier == 1:
        return

    # printing rate of convergence sequences
    print(
        "running rate of convergence with alpha = 1: \n",
        orderOfConvergence(xstar, iteration, alpha=1),
        "\n",
    )
    print(
        "running rate of convergence with alpha = 2: \n",
        orderOfConvergence(xstar, iteration, alpha=2),
        "\n",
    )

    return


# define routines
def fixedpt(f, x0, tol, Nmax):
    """x0 = initial guess"""
    """ Nmax = max number of iterations"""
    """ tol = stopping tolerance"""
    iterations = np.zeros(Nmax)

    count = 0
    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        iterations[count - 1] = x1
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, iterations[:count]]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, iterations]


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
