import numpy as np


def driver():
    # testing methods
    xeval = np.linspace(0, 10, 1000)
    xint = np.linspace(0, 10, 11)

    ind1 = subintervalFind(xint, xeval, 1)
    ind2 = subintervalFind(xint, xeval, 2)
    print(ind1[-1], ind2[0])

    y = lineGenerator(0, -10, 1, 10)
    print(y(0.5))
    return


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


print("\n")
driver()
print("\n")
