import numpy as np


def driver():
    # function definitions for first
    f = lambda x, y: 3 * x**2 - y**2
    g = lambda x, y: 3 * x * y**2 - x**3 - 1

    # constants
    J_constant = np.matrix([[1 / 6, 1 / 18], [0, 1 / 6]])
    tol = 5e-16
    x0, y0 = 1, 1

    # part a
    print("(1a)")
    print("iteration with initial guess at x0=y0=1:\ntolerance @", tol, "\n")
    [astar1, iter1] = iteration(f, g, J_constant, x0, y0, tol, max_iter=100)
    print("the approximate root is", astar1)
    print("f(root) =", [f(astar1[0], astar1[1]), g(astar1[0], astar1[1])])
    # print("\niterations:")
    # print(iter1)
    alpha = 1
    print("\nrunning order of convergence for alpha =", alpha)
    print(nd_orderConvergence(astar1, iter1, alpha, dimensions=2))
    alpha = 1.6
    print("\ntesting super-linear convergence")
    print("running order of convergence for alpha =", alpha)
    print(nd_orderConvergence(astar1, iter1, alpha, dimensions=2))
    print("\n")

    # part c

    # function definitions for second
    f = lambda x: 3 * x[0] ** 2 - x[1] ** 2
    g = lambda x: 3 * x[0] * x[1] ** 2 - x[0] ** 3 - 1
    f_combined = lambda x: [f(x), g(x)]

    dfdx = lambda x: 6 * x[0]
    dfdy = lambda x: -2 * x[1]
    dgdx = lambda x: -3 * (x[0] ** 2 - x[1] ** 2)
    dgdy = lambda x: 6 * x[0] * x[1]
    J = lambda x: np.matrix([[dfdx(x), dfdy(x)], [dgdx(x), dgdy(x)]])

    print("\nnewton's method with initial guess at x0=y0=1:\ntolerance @", tol, "\n")
    [astar2, iter2] = nd_newton(
        f_combined, J, [x0, y0], tol, dimensions=2, max_iter=100
    )
    print("the approximate root is", astar2)
    print("f(root) =", f_combined(astar2))
    # print("\niterations:")
    # print(iter2)
    alpha = 1
    print("\nrunning order of convergence for alpha =", alpha)
    print(nd_orderConvergence(astar2, iter2, alpha, dimensions=2))
    alpha = 2
    print("\nrunning order of convergence for alpha =", alpha)
    print(nd_orderConvergence(astar2, iter2, alpha, dimensions=2))
    alpha = 3
    print("\nrunning order of convergence for alpha =", alpha)
    print(nd_orderConvergence(astar2, iter2, alpha, dimensions=2))

    return


def iteration(f, g, J_constant, x0, y0, epsilon, max_iter):
    xn = np.zeros((max_iter, 2))
    xn[0] = [x0, y0]
    for n in range(0, max_iter - 1):
        fxn = [f(xn[n][0], xn[n][1]), g(xn[n][0], xn[n][1])]
        if np.linalg.norm(fxn) < epsilon:
            print("Found solution after", n, "iterations.\n")
            return xn[n], xn[: n + 1]
        subtract_vector = np.matmul(J_constant, fxn)

        xn[n + 1] = xn[n] - subtract_vector

    print(xn)
    print("Exceeded maximum iterations. No solution found.")
    return None


def nd_newton(f, J, x0, epsilon, dimensions, max_iter):
    xn = np.zeros((max_iter, dimensions))
    xn[0] = x0
    for n in range(0, max_iter - 1):
        fxn = f(xn[n])
        if np.linalg.norm(fxn) < epsilon:
            print("Found solution after", n, "iterations.")
            return xn[n], xn[: n + 1]
        J_found = J(xn[n])
        inv_J = np.linalg.inv(J_found)
        xn[n + 1] = xn[n] - np.matmul(inv_J, fxn)
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
