import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv


def driver():
    f = lambda x: math.exp(x)
    a = 0
    b = 10

    """ create points you want to evaluate at"""
    Neval = 100
    xeval = np.linspace(a, b, Neval)

    """ number of intervals"""
    Nint = 10

    """evaluate the linear spline"""
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)

    """ evaluate f at the evaluation points"""
    fex = np.zeros(Neval)
    for j in range(Neval):
        fex[j] = f(xeval[j])

    plt.figure()
    plt.plot(xeval, fex, "k", label="$f(x)$")
    plt.plot(xeval, yeval, "r--", label="$Linear Spline$")
    plt.legend()
    plt.show()

    err = abs(yeval - fex)
    plt.figure()
    plt.plot(xeval, err, "ro--", label="error")
    plt.show()


def eval_lin_spline(xeval, Neval, a, b, f, Nint):
    """create the intervals for piecewise approximations"""
    xint = np.linspace(a, b, Nint + 1)

    """create vector to store the evaluation of the linear splines"""
    yeval = np.zeros(Neval)

    for jint in range(Nint):
        """find indices of xeval in interval (xint(jint),xint(jint+1))"""
        """let ind denote the indices in the intervals"""
        """let n denote the length of ind"""
        ind = subintervalFind(xint, xeval, interval=jint)
        n = len(ind)

        """temporarily store your info for creating a line in the interval of 
         interest"""
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint + 1]
        fb1 = f(b1)

        for kk in range(n):
            """use your line evaluator to evaluate the lines at each of the points
            in the interval"""
            line = lineGenerator(a1, fa1, b1, fb1)
            """yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)"""
            yeval[ind[kk]] = line(xeval[ind[kk]])
    return yeval


def subintervalFind(xint, xeval, interval):
    if len(xint) <= (interval + 1):
        print("error: inputted interval not valid")
        return []
    a, b = xint[interval], xint[interval + 1]
    ind = np.where((xeval >= a) & (xeval <= b))
    return ind[0]


def lineGenerator(x0, fx0, x1, fx1):
    if x0 == x1:
        print("error: x0=x1. line generator requires two separate points x0 and x1")
    m = (fx1 - fx0) / (x1 - x0)
    f = lambda x: fx0 + m * (x - x0)
    return f


driver()
# if __name__ == "__main__":
#     # run the drivers only if this is called from the command line
#     driver()
