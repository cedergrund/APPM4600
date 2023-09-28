import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def driver():
    # function declerations
    f = lambda x: x**6 - x - 1
    Df = lambda x: 6 * x**5 - 1

    tol = 1e-3
    print("Note: using tolerance 10^-3\n# of iterations too large otherwise\n\n")

    x0 = 2
    print("newton's method with initial guess at x=2:\n")
    [astar1, iter1] = newton(f, Df, x0, tol, max_iter=1000)
    print("the approximate root is", astar1)
    print("f(root) =", f(astar1))
    # print("\nerror at each iteration:")
    # print(iter - astar)
    print("\n")

    a, b = 2, 1
    print("secant method with x0=2 and x1=1:\n")
    [astar2, iter2] = secant(f, a, b, tol, max_iter=1000)
    print("root found after", len(iter2), "iterations.")
    print("the approximate root is", astar2)
    print("f(root) =", f(astar2))
    # print("\nerror at each iteration:")
    # print(iter - astar)
    # print("\n")

    # prob a
    print("\n\ntable with errors:")
    iter1 = np.pad(
        iter1, (0, len(iter2) - len(iter1)), mode="constant", constant_values=iter1[-1]
    )

    df = create_table(np.abs(iter1 - astar1), np.abs(iter2 - astar2))
    print(df)

    # prob b

    k_term1 = np.zeros(len(iter1) - 1)
    k1_term1 = np.zeros(len(iter1) - 1)

    k_term2 = np.zeros(len(iter2) - 1)
    k1_term2 = np.zeros(len(iter2) - 1)

    for n in range(len(iter1) - 1):
        k_term1[n] = np.abs(iter1[n] - astar1)
        k1_term1[n] = np.abs(iter1[n + 1] - astar1)

    for n in range(len(iter2) - 1):
        k_term2[n] = np.abs(iter2[n] - astar2)
        k1_term2[n] = np.abs(iter2[n + 1] - astar2)

    fig, ax1 = plt.subplots()
    ax1.plot(k_term1, k1_term1)
    ax1.set_title(
        "Newton's Method on log-log axes",
        fontsize=16,
    )
    ax1.set_ylabel("$|x_{k+1}- \\alpha|$", fontsize=14)
    ax1.set_xlabel("$|x_{k}-\\alpha|$", fontsize=14)
    plt.yscale("log")
    plt.xscale("log")

    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(k_term2, k1_term2)
    ax1.set_title(
        "Secant Method on log-log axes",
        fontsize=16,
    )
    ax1.set_ylabel("$|x_{k+1}- \\alpha|$", fontsize=14)
    ax1.set_xlabel("$|x_{k}-\\alpha|$", fontsize=14)
    plt.yscale("log")
    plt.xscale("log")

    plt.show()

    return


# Function to create the tables
def create_table(x_delta, y_delta):
    data = {
        "Error with newtons method:": x_delta,
        "Error with secant method:": y_delta,
    }

    df = pd.DataFrame(data)
    return df


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
    for n in range(0, max_iter):
        fxn = f(xn[n])
        if np.abs(fxn) < epsilon:
            print("Found solution after", n, "iterations.")
            return xn[n], xn[: n + 1]
        Dfxn = Df(xn[n])
        if Dfxn == 0:
            print("Zero derivative. No solution found.")
            return None
        xn[n + 1] = xn[n] - fxn / Dfxn
    print("Exceeded maximum iterations. No solution found.")
    return None


def secant(f, a, b, epsilon, max_iter):
    """Approximate solution of f(x)=0 on interval [a,b] by the secant method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    m_N : number
        The x intercept of the secant line on the the Nth interval
            m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
        The initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0
        for some intercept m_n then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iterations, the secant method fails and return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> secant(f,1,2,5)
    1.6180257510729614
    """
    if f(a) * f(b) >= 0:
        print("Secant method fails.")
        return None
    a_n = a
    b_n = b

    m_n = np.zeros(max_iter)

    for n in range(1, max_iter + 1):
        m_n[n - 1] = a_n - f(a_n) * (b_n - a_n) / (f(b_n) - f(a_n))
        f_m_n = f(m_n[n - 1])

        if np.abs(f_m_n) < epsilon:
            return m_n[n - 1], m_n[:n]
        elif f(a_n) * f_m_n < 0:
            a_n = a_n
            b_n = m_n[n - 1]
        elif f(b_n) * f_m_n < 0:
            a_n = m_n[n - 1]
            b_n = b_n
        else:
            print("Secant method fails.")
            return None
    return a_n - f(a_n) * (b_n - a_n) / (f(b_n) - f(a_n)), m_n


print("\n")
driver()
print("\n")
