import numpy as np


def driver():
    print("hello")
    return


""" composite trapezoidal method """


def compositeTrapezoid(f, a, b, N):
    # define h and other sums
    h = (b - a) / N
    sum = f(a) + f(b)

    # perform trapezoid
    for i in range(1, N):
        x = a + i * h
        sum += 2 * f(x)

    # return as given by composite formula
    return sum * h / 2


def compositeSimpsons(f, a, b, N):
    assert N % 2 == 0  # ensure n is even

    # define h and other sums
    h = (b - a) / N
    sum = f(a) + f(b)

    # perform composite simpsons summation
    for i in range(1, N):
        x = a + i * h
        if i % 2 == 0:
            sum += 2 * f(x)
        else:
            sum += 4 * f(x)

    # return as given by composite formula
    return sum * h / 3


if __name__ == "__main__":
    driver()
