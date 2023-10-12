import numpy as np
import math


def driver():
    # function definitions for all methods
    f1 = lambda x: x[0] + np.cos(x[0] * x[1] * x[2]) - 1
    f2 = lambda x: (1 - x[0]) ** (1 / 4) + x[1] + 0.05 * (x[2] ** 2) - 0.15 * x[2] - 1
    f3 = lambda x: -(x[0] ** 2) - 0.1 * (x[1] ** 2) + x[2] - 1
    F = lambda x: [f1(x), f2(x), f3(x)]

    # Jacobian creation for Newton's
    f1x = lambda x: 1 - x[1] * x[2] * np.sin(x[0] * x[1] * x[2])
    f1y = lambda x: -x[0] * x[2] * np.sin(x[0] * x[1] * x[2])
    f1z = lambda x: -x[0] * x[1] * np.sin(x[0] * x[1] * x[2])
    f2x = lambda x: -1 / (4 * (1 - x[0]) ** (3 / 4))
    f2y = lambda x: 1
    f2z = lambda x: (2 * x[2] - 3) / 20
    f3x = lambda x: -2 * x[0]
    f3y = lambda x: 1 / 100 - x[1] / 5
    f3z = lambda x: 1
    J = lambda x: np.matrix(
        [[f1x(x), f1y(x), f1z(x)], [f2x(x), f2y(x), f2z(x)], [f3x(x), f3y(x), f3z(x)]]
    )

    # constants
    dimensions = 3
    tol = 1e-06
    x0 = [-1, -1, 1]

    # newtons method
    print("NEWTONS METHOD:")
    print("\nnewton's method with initial guess at", x0, "\ntolerance @", tol, "\n")
    [astar1, iter1] = nd_newton(F, J, x0, tol, dimensions, max_iter=100)
    print("the approximate root is", astar1)
    print("F(root) =", F(astar1))
    print("\n\n")

    # steepest descent
    print("STEEPEST DESCENT:")
    print(
        "steepest descent method with initial guess at", x0, "\ntolerance @", tol, "\n"
    )
    [astar2, g2, ier2] = SteepestDescent(x0, tol, Nmax=100)
    print("the approximate root is", astar2)
    print("F(root) =", F(astar2))
    print("\n\n")

    # steepest descent into Newtons
    tol1 = 5e-2
    print("HYBRID APPROACH:")
    print(
        "steepest descent method with initial guess at", x0, "\ntolerance @", tol1, "\n"
    )
    [astar3, g3, ier3] = SteepestDescent(x0, tol1, Nmax=100)
    print(
        "\nnow finishing iteration through newtons.\ninitial guess @",
        astar3,
        "\ntolerance @",
        tol,
    )
    print()
    [astar3, iter1] = nd_newton(F, J, astar3, tol, dimensions, max_iter=100)
    print("the approximate root is", astar3)
    print("F(root) =", F(astar3))

    return


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


###########################################################
# functions:
def evalF(x):
    F = np.zeros(3)
    F[0] = x[0] + math.cos(x[0] * x[1] * x[2]) - 1.0
    F[1] = (1.0 - x[0]) ** (0.25) + x[1] + 0.05 * x[2] ** 2 - 0.15 * x[2] - 1
    F[2] = -x[0] ** 2 - 0.1 * x[1] ** 2 + 0.01 * x[1] + x[2] - 1
    return F


def evalJ(x):
    J = np.array(
        [
            [
                1.0 + x[1] * x[2] * math.sin(x[0] * x[1] * x[2]),
                x[0] * x[2] * math.sin(x[0] * x[1] * x[2]),
                x[1] * x[0] * math.sin(x[0] * x[1] * x[2]),
            ],
            [-0.25 * (1 - x[0]) ** (-0.75), 1, 0.1 * x[2] - 0.15],
            [-2 * x[0], -0.2 * x[1] + 0.01, 1],
        ]
    )
    return J


def evalg(x):
    F = evalF(x)
    g = F[0] ** 2 + F[1] ** 2 + F[2] ** 2
    return g


def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code
def SteepestDescent(x, tol, Nmax):
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = np.linalg.norm(z)
        if z0 == 0:
            print("zero gradient")
        z = z / z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3 * z
        g3 = evalg(dif_vec)
        while g3 >= g1:
            alpha3 = alpha3 / 2
            dif_vec = x - alpha3 * z
            g3 = evalg(dif_vec)

        if alpha3 < tol:
            print("no likely improvement")
            ier = 0
            return [x, g1, ier]
        alpha2 = alpha3 / 2
        dif_vec = x - alpha2 * z
        g2 = evalg(dif_vec)
        h1 = (g2 - g1) / alpha2
        h2 = (g3 - g2) / (alpha3 - alpha2)
        h3 = (h2 - h1) / alpha3
        alpha0 = 0.5 * (alpha2 - h1 / h3)
        dif_vec = x - alpha0 * z
        g0 = evalg(dif_vec)
        if g0 <= g3:
            alpha = alpha0
            gval = g0
        else:
            alpha = alpha3
            gval = g3
        x = x - alpha * z
        if abs(gval - g1) < tol:
            ier = 0
            print("Found solution after", its, "iterations.")
            return [x, gval, ier]
    print("max iterations exceeded")
    ier = 1
    return [x, g1, ier]


print("\n")
driver()
print("\n")
