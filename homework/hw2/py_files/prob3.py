import math


def driver():
    x = 9.999999995000000e-10

    # y = math.exp(x)

    # return y - 1

    # return x * 10e-16
    return x + 1 / 2 * x**2


print(driver())
