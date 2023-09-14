import numpy as np

# imported bisection method from provided example


def driver():
    # function decleration
    f = lambda x: x**2 * (x - 1)

    # endpoints for parts a,b,c
    a_a = 0.5
    b_a = 2
    a_b = -1
    b_b = 0.5
    a_c = -1
    b_c = 2

    # tolerance
    tol = 1e-7

    print("(a):")
    [astar, ier] = bisection(f, a_a, b_a, tol)
    print("the approximate root is", astar)
    print("the error message reads:", ier)
    print("f(astar) =", f(astar))
    print("\n")

    print("(b):")
    [astar, ier] = bisection(f, a_b, b_b, tol)
    print("the approximate root is", astar)
    print("the error message reads:", ier)
    print("f(astar) =", f(astar))
    print("\n")

    print("(c):")
    [astar, ier] = bisection(f, a_c, b_c, tol)
    print("the approximate root is", astar)
    print("the error message reads:", ier)
    print("f(astar) =", f(astar))
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
    #      print('abs(d-a) = ', abs(d-a))

    astar = d
    ier = 0
    return [astar, ier]


print("")
driver()
