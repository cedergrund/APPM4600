import numpy as np

# imported bisection method from provided example


def driver():
    # function declerations for parts a,b,c
    f_a = lambda x: (x - 1) * (x - 3) * (x - 5)
    f_b = lambda x: (x - 1) ** 2 * (x - 3)
    f_c = lambda x: np.sin(x)

    # endpoints for parts a,b,c
    a_a = 0
    b_a = 2.4
    a_b = 0
    b_b = 2
    a_c_1 = 0
    b_c_1 = 0.1
    a_c_2 = 0.5
    b_c_2 = 3 * np.pi / 4

    # tolerance
    tol = 1e-5

    print("(a):")
    [astar, ier] = bisection(f_a, a_a, b_a, tol)
    print("the approximate root is", astar)
    print("the error message reads:", ier)
    print("f(astar) =", f_a(astar))
    print("\n")

    print("(b):")
    [astar, ier] = bisection(f_b, a_b, b_b, tol)
    print("the approximate root is", astar)
    print("the error message reads:", ier)
    print("f(astar) =", f_b(astar))
    print("\n")

    print("(c) on interval (0,0.5):")
    [astar, ier] = bisection(f_b, a_c_1, b_c_1, tol)
    print("the approximate root is", astar)
    print("the error message reads:", ier)
    print("f(astar) =", f_c(astar))
    print("\n")

    print("(c) on interval (0.5,(3/4)*pi):")
    [astar, ier] = bisection(f_b, a_c_2, b_c_2, tol)
    print("the approximate root is", astar)
    print("the error message reads:", ier)
    print("f(astar) =", f_c(astar))
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
