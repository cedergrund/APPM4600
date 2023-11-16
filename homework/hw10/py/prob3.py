import numpy as np
from scipy.integrate import quad


def driver():
    # define function and interval
    f = lambda x: 1 / (1 + x**2)
    a, b = -5, 5

    # define n for composite num. integration
    nt = 11106
    nc = 394

    # calculation and output
    ct = compositeTrapezoid(f, a, b, nt)
    cs = compositeSimpsons(f, a, b, nc)
    f_10m6 = quad(f, a, b, full_output=1)
    f_10m4 = quad(f, a, b, full_output=1, epsabs=1e-4)
    print("Composite Trapezoidal Rule (n={}): ".format(nt), ct)
    print("Composite Simpson's Rule (n={}): ".format(nc), cs)
    print("Default quad Evaluation (n={}):".format(f_10m6[2]["neval"]), f_10m6[0])
    print(
        "Quad Evaluation 1e-4 tolerance (n={}):".format(f_10m4[2]["neval"]), f_10m4[0]
    )
    print()
    print("Error from default evalation of quad():")
    print("-> Trapezoidal Rule: ", abs(f_10m6[0] - ct))
    print("-> Simpson's Rule: ", abs(f_10m6[0] - cs))

    return


def compositeTrapezoid(f, a, b, n):
    # define h and other sums
    h = (b - a) / n
    sum = f(a) + f(b)

    # perform trapezoid
    for i in range(1, n):
        x = a + i * h
        sum += 2 * f(x)

    # return as given by composite formula
    return sum * h / 2


def compositeSimpsons(f, a, b, n):
    if n % 2 == 1:  # ensure n is even
        n += 1
    # define h and other sums
    h = (b - a) / n
    sum = f(a) + f(b)

    # perform composite simpsons summation
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            sum += 2 * f(x)
        else:
            sum += 4 * f(x)

    # return as given by composite formula
    return sum * h / 3


if __name__ == "__main__":
    print("\n")
    driver()
    print("\n")
