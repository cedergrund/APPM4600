import numpy as np
import math


def driver():
    # function definitions
    f = lambda x: x[0] ** 2 + x[1] ** 2 - 4
    g = lambda x: np.exp(x[0]) + x[1] - 1
    F = lambda x: np.array([f(x), g(x)])

    # Jacobian creation for Newton's
    fx = lambda x: 2 * x[0]
    fy = lambda x: 2 * x[1]
    gx = lambda x: np.exp(x[0])
    gy = lambda x: 1
    J = lambda x: np.matrix([[fx(x), fy(x)], [gx(x), gy(x)]])

    # constants
    dimensions = 2
    tol = 1e-12
    x0_1 = [1, 1]
    x0_2 = [1, -1]
    x0_3 = [0, 0]

    # newtons method
    print("NEWTONS METHOD:")

    for i in [x0_1, x0_2, x0_3]:
        print("\nnewton's method with initial guess at", i, "\ntolerance @", tol, "\n")
        [astar, iter] = nd_newton(F, J, i, tol, dimensions, max_iter=100)
        if len(iter) == 0:
            print("")
            continue
        print("the approximate root is", astar)
        print("F(root) =", F(astar))
        print("")
    print("\n")

    # QUASI NETWTONS

    # lazy newton method
    print("LAZY NEWTON'S METHOD:")

    for i in [x0_1, x0_2, x0_3]:
        print(
            "\nlazy newton's method with initial guess at",
            i,
            "\ntolerance @",
            tol,
            "\n",
        )
        [astar, ier, iter] = LazyNewton(i, tol, Nmax=100)
        if ier == 1:
            print("Solution not found.")
            print()
            continue
        print("the approximate root is", astar)
        print("F(root) =", F(astar))
        print()
    print("\n\n")

    # brodens method
    print("BROYDEN'S METHOD:")

    for i in [x0_1, x0_2, x0_3]:
        print("\nbroyden's method with initial guess at", i, "\ntolerance @", tol, "\n")
        [astar, msg] = broyden(F, J, i, niter=100, ftol=tol)
        if len(astar) == 0:
            print("Solution not found,", msg)
            print()
            continue
        print("the approximate root is", astar)
        print("F(root) =", F(astar))
        print()

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
        if np.linalg.det(J_found) == 0:
            print(
                "Jacobian at iteration",
                n,
                "is\n",
                J_found,
                "\nwhich is singular. unable to converge.",
            )
            return xn[n], []
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


###############################
### lazy code
def evalF(x):
    F = np.zeros(2)
    F[0] = x[0] ** 2 + x[1] ** 2 - 4
    F[1] = np.exp(x[0]) + x[1] - 1
    return F


def evalJ(x):
    J = np.array([[2 * x[0], 2 * x[1]], [np.exp(x[0]), 1]])

    return J


def LazyNewton(x0, tol, Nmax):
    """Lazy Newton = use only the inverse of the Jacobian for initial guess"""
    """ inputs: x0 = initial guess, tol = tolerance, Nmax = max its"""
    """ Outputs: xstar= approx root, ier = error message, its = num its"""
    J = evalJ(x0)
    if np.linalg.det(J) == 0:
        print("Jacobian matrix at initial point is singular.")
        return [1, 1, 1]

    Jinv = np.linalg.inv(J)
    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        if np.linalg.norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            print("converged after", its, "iterations")
            return [xstar, ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    print("Could not converge after", Nmax, "iterations.")
    return [xstar, ier, its]


###############################
### broyden's code
def broyden(fun, jaco, x, niter=50, ftol=1e-12, verbose=False):
    msg = "Maximum number of iterations reached."
    J = jaco(x)
    for cont in range(niter):
        if np.linalg.det(J) == 0:
            x = []
            msg = "Jacobian is singular at iteration " + str(cont)
            break
        if verbose:
            print("n: {}, x: {}".format(cont, x))
        f_old = fun(x)
        dx = -np.linalg.solve(J, f_old)
        x = x + dx
        f = fun(x)
        df = f - f_old
        J = J + np.outer(df - np.dot(J, dx), dx) / np.dot(dx, dx)
        if np.linalg.norm(f) < ftol:
            msg = "Root found with desired accuracy."
            print("converged after", cont, "iterations")
            break
    return x, msg


print("\n")
driver()
print("\n")
