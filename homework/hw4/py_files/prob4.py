import numpy as np


def driver():
    # function declarations
    f = (
        lambda x: np.exp(3 * x)
        - 27 * x**6
        + 27 * x**4 * np.exp(x)
        - 9 * x**2 * np.exp(2 * x)
    )
    Df = (
        lambda x: 3 * np.exp(3 * x)
        + (-18 * x**2 - 18 * x) * np.exp(2 * x)
        + (27 * x**4 + 108 * x**3) * np.exp(x)
        - 162 * x**5
    )
    DDf = (
        lambda x: 9 * np.exp(3 * x)
        + (-36 * x**2 - 72 * x - 18) * np.exp(2 * x)
        + (27 * x**4 + 216 * x**3 + 324 * x**2) * np.exp(x)
        - 810 * x**4
    )

    tol = 1e-13
    x0 = 3
    m = 3

    print("(i): \nnewton's method with initial guess at x=3\n")
    [astar1, iter1] = newton(f, Df, x0, tol, max_iter=1000)
    print("the approximate root is", astar1)
    print("f(root) =", f(astar1))
    print("order of convergence evaluated with alpha = 1:")
    print(orderOfConvergence(astar1, iter1, 1))
    print("\n")

    print("(ii):\nmodified newton's method from class x0=3:\n")
    mu = lambda x: f(x) / Df(x)
    Dmu = lambda x: (Df(x) * Df(x) - f(x) * DDf(x)) / (Df(x) ** 2)
    [astar2, iter2] = newton(mu, Dmu, x0, tol, max_iter=1000)
    print("the approximate root is", astar2)
    print("f(root) =", f(astar2))
    print("order of convergence evaluated with alpha = 2:")
    print(orderOfConvergence(astar2, iter2, 2))
    print("\n")

    print("(iii):\nmodified newton's (fixed point) method from (2)\nx0=3 and m=3:\n")
    g = lambda x: x - m * f(x) / Df(x)
    [astar3, _, iter3] = fixedpt(g, x0, tol, Nmax=1000)
    print("Found solution after", len(iter3), "iterations.")
    print("the approximate root is", astar3)
    print("f(root) =", f(astar3))
    print("order of convergence evaluated with alpha = 2:")
    print(orderOfConvergence(astar3, iter3[:-1], 2))
    print("\n")

    return


2


def newton(f, Df, x0, epsilon, max_iter):
    """Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    """
    xn = np.zeros(max_iter)
    xn[0] = x0
    for n in range(0, max_iter - 1):
        fxn = f(xn[n])
        if np.abs(fxn) < epsilon:
            print("Found solution after", n, "iterations.")
            return xn[n], xn[: n + 1]
        Dfxn = Df(xn[n])
        if Dfxn == 0:
            print("Zero derivative. No solution found.")
            return None
        xn[n + 1] = xn[n] - fxn / Dfxn
    print(xn)
    print("Exceeded maximum iterations. No solution found.")
    return None


def m_newton(f, Df, x0, m, epsilon, max_iter):
    """Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    """
    xn = np.zeros(max_iter)
    xn[0] = x0
    for n in range(0, max_iter):
        fxn = f(xn[n])
        if np.abs(fxn) < epsilon:
            print("Found solution after", n, "iterations.")
            return xn[n], xn[: n + 1]
        Dfxn = Df(xn[n])
        if Dfxn == 0:
            print("Zero derivative. No solution found.")
            return None
        xn[n + 1] = xn[n] - m * fxn / Dfxn
    print("Exceeded maximum iterations. No solution found.")
    return None


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
            return [xstar, ier, iterations[: count + 1]]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, iterations]


def orderOfConvergence(fixed_point, iterations, alpha):
    seq = np.zeros(len(iterations) - 1)

    for n, i in enumerate(iterations):
        if n == len(iterations) - 1:
            continue

        seq[n] = np.abs(iterations[n + 1] - fixed_point) / (
            (np.abs(i - fixed_point)) ** alpha
        )

    return seq


print("\n")
driver()
print("\n")
