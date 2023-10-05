import numpy as np


def driver():
    # constants
    n = 3  # dimensions
    x0, y0, z0 = 1, 1, 1  # initial points
    init = [x0, y0, z0]
    epsilon = 5e-16  # tolerance

    # function definitions
    f = lambda x: x[0] ** 2 + 4 * x[1] ** 2 + 4 * x[2] ** 2 - 16
    fx = lambda x: 2 * x[0]
    fy = lambda x: 8 * x[1]
    fz = lambda x: 8 * x[2]
    partials = [fx, fy, fz]

    print("(3b)")
    print("iteration with initial guess at x0=y0=z0=1:\ntolerance @", epsilon, "\n")
    [astar1, iter1] = nd_iteration(f, partials, init, epsilon, n, max_iter=100)
    print("the approximate root is:")
    print("  x=", astar1[1], "\n  y=", astar1[1], "\n  z=", astar1[2])
    print("f(root) =", f(astar1))

    print("\niterations:")
    print(iter1)
    print()

    alpha = 1
    print("\nrunning order of convergence for alpha =", alpha)
    print(nd_orderConvergence(astar1, iter1, alpha, dimensions=n))
    alpha = 2
    print("\nrunning order of convergence for alpha =", alpha)
    print(nd_orderConvergence(astar1, iter1, alpha, dimensions=n))
    alpha = 3
    print("\nrunning order of convergence for alpha =", alpha)
    print(nd_orderConvergence(astar1, iter1, alpha, dimensions=n))
    print("\n")

    return


def nd_iteration(f, partials, x0, epsilon, dimensions, max_iter):
    xn = np.zeros((max_iter, dimensions))
    xn[0] = x0

    for n in range(0, max_iter - 1):
        fxn = f(xn[n])

        if np.linalg.norm(fxn) < epsilon:
            print("Found solution after", n, "iterations.\n")
            return xn[n], xn[: n + 1]

        d_bot = 0
        for i in range(dimensions):
            d_bot += partials[i](xn[n]) ** 2

        if d_bot < np.FPE_UNDERFLOW:
            print(xn)
            print("divide by 0 error")
            return None

        d = fxn / d_bot
        for i in range(dimensions):
            xn[n + 1][i] = xn[n][i] - partials[i](xn[n]) * d

    print(xn)
    print("Exceeded maximum iterations. No solution found.")
    return None


def nd_orderConvergence(astar, iterations, alpha, dimensions):
    seq = np.zeros((len(iterations) - 1, dimensions))

    for n, i in enumerate(iterations):
        if n >= len(iterations) - 1:
            break

        seq[n] = np.linalg.norm(iterations[n + 1] - astar) / (
            (np.linalg.norm(i - astar)) ** alpha
        )

    return seq[:-1]


print("\n")
driver()
print("\n")
