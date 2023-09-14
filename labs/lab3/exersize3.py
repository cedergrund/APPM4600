# import libraries
import numpy as np

# imported fixed-point method from provided example


def driver():
    # function definitions a-d
    fa = lambda x: x * (1 + ((7 - x**5) / (x**2))) ** 3
    fb = lambda x: x - ((x**5 - 7) / (x**2))
    fc = lambda x: x - ((x**5 - 7) / (5 * x**4))
    fd = lambda x: x - ((x**5 - 7) / (12))

    Nmax = 100
    tol = 1e-10
    x0 = 1

    # running method
    print("(a):")
    [xstar, ier] = fixedpt(fa, x0, tol, Nmax)
    print("the approximate fixed point is:", xstar)
    print("fa(xstar):", fa(xstar))
    print("Error message reads:", ier)
    print("\n")

    print("(b):")
    [xstar, ier] = fixedpt(fb, x0, tol, Nmax)
    print("the approximate fixed point is:", xstar)
    print("fb(xstar):", fb(xstar))
    print("Error message reads:", ier)
    print("\n")

    print("(c):")
    [xstar, ier] = fixedpt(fc, x0, tol, Nmax)
    print("the approximate fixed point is:", xstar)
    print("fc(xstar):", fc(xstar))
    print("Error message reads:", ier)
    print("\n")

    print("(d):")
    [xstar, ier] = fixedpt(fd, x0, tol, Nmax)
    print("the approximate fixed point is:", xstar)
    print("fd(xstar):", fd(xstar))
    print("Error message reads:", ier)
    print("\n")
    return


# define routines
def fixedpt(f, x0, tol, Nmax):
    """x0 = initial guess"""
    """ Nmax = max number of iterations"""
    """ tol = stopping tolerance"""

    count = 0
    while count < Nmax:
        count = count + 1
        x1 = f(x0)
        if abs(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]


driver()
