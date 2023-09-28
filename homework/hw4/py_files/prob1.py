import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def driver():
    # define constants
    T_i, T_s, alpha, t = 20, -15, 0.138 * 10 ** (-6), 5184000
    tol = 1e-13

    # define function and first derivative
    f = lambda x: (T_i - T_s) * sp.special.erf(x / (2 * np.sqrt(alpha * t))) + T_s
    Df = (
        lambda x: (T_i - T_s)
        * (1 / (np.sqrt(np.pi * alpha * t)))
        * np.exp(-((x) ** 2) / (4 * alpha * t))
    )

    # part (a)
    # plot over x-axis from 0 to x-bar (1 as f(1)>0)
    X = np.arange(0, 1, 0.0001)
    fig, ax1 = plt.subplots()
    ax1.plot(X, f(X))
    ax1.set_title(
        "$f(x)=(T_i-T_s) \cdot erf \left( \\frac{x}{2\sqrt{ \\alpha \cdot t}} \\right)+T_s$",
        fontsize=16,
    )
    ax1.set_ylabel("$f(x)$", fontsize=14)
    ax1.set_xlabel("x", fontsize=14)
    plt.grid(True)
    plt.show()

    # part (b)
    # defining endpoints
    a, b = 0, 1
    print("(b):\n")
    print("bisection method:")
    [astar, _] = bisection(f, a, b, tol)
    print("the approximate root is", astar)
    print("f(root) =", f(astar))
    print("\n")

    # part (c)
    # defining inital guesses
    x0_1, x0_2 = 0.01, 1
    print("(c):\n")
    print("newton's method:\n")
    print("initial guess at x=0.01 meters")
    astar = newton(f, Df, x0_1, tol, 1000)
    print("the approximate root is", astar)
    print("f(root) =", f(astar))
    print("\ninitial guess at x=1 meters")
    astar = newton(f, Df, x0_2, tol, 1000)
    print("the approximate root is", astar)
    print("f(root) =", f(astar))
    print("\n")

    return


# define routines
def bisection(f, a, b, tol):
    #    Inputs:
    #     f,a,b       - function and endpoints of initial interval
    #      tol  - bisection stops when interval length < tol

    #    Returns:
    #      astar - approximation of root
    #      ier   - error message
    #            - ier = 1 => Failed
    #            - ier = 0 == success

    #     first verify there is a root we can find in the interval

    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        print("f(", a, ") = ", fa)
        print("f(", b, ") = ", fb)
        ier = 1
        astar = a
        return [astar, ier]

    #   verify end points are not a root
    if fa == 0:
        astar = a
        ier = 0
        return [astar, ier]

    if fb == 0:
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5 * (a + b)
    while abs(d - a) > tol:
        fd = f(d)
        if fd == 0:
            astar = d
            ier = 0
            return [astar, ier]
        if fa * fd < 0:
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count = count + 1
        print("iteration: ", count, " | curr_root = ", d)
    #      print('abs(d-a) = ', abs(d-a))

    print("Number of iterations: ", count, "\n")
    astar = d
    ier = 0
    return [astar, ier]


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
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print("Found solution after", n, "iterations.")
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print("Zero derivative. No solution found.")
            return None
        xn = xn - fxn / Dfxn
    print("Exceeded maximum iterations. No solution found.")
    return None


print("\n")
driver()
