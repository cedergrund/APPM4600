import numpy as np


def driver():
    # function definitions for first
    f = lambda x: (1 / np.sqrt(2)) * np.sqrt(1 + (x[0] + x[1]) ** 2) - (2 / 3)
    g = lambda x: (1 / np.sqrt(2)) * np.sqrt(1 + (x[0] - x[1]) ** 2) - (2 / 3)
    f_combined = lambda x: [f(x), g(x)]

    # constants
    tol = 1e-03
    x0, y0 = -1000, 0

    # part a

    print("fixed pt. iteration with initial guess at x0=y0=1:\ntolerance @", tol, "\n")
    [astar1, iter1] = nd_fixed(f_combined, [x0, y0], tol, dimensions=2, max_iter=30000)
    print("the approximate root is", astar1)
    print("f(root) =", [f(astar1), g(astar1)])
    # print("\niterations:")
    # print(iter1)

    return


def nd_fixed(f, x0, epsilon, dimensions, max_iter):
    xn = np.zeros((max_iter, dimensions))
    xn[0] = x0

    for n in range(0, max_iter - 1):
        fxn = f(xn[n])
        if np.abs(np.linalg.norm(fxn) - np.linalg.norm(xn[n])) < epsilon:
            xstar = fxn
            xn[n + 1] = fxn
            print("Found solution after", n, "iterations.")
            return [xstar, xn[: n + 2]]
        xn[n + 1] = fxn
    print("Exceeded maximum iterations. No solution found.")
    return None


print("\n")
driver()
print("\n")
