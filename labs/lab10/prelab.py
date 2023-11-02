import numpy as np


def driver():
    print(eval_legendre(10, -1))
    return


def eval_legendre(n, x):
    assert n >= 0, "Invalid n as input"

    p = np.zeros(n + 1)
    if n >= 0:
        p[0] = 1
    if n >= 1:
        p[1] = x

    for i in range(2, n + 1):
        p[i] = (1 / (i)) * ((2 * (i - 1) + 1) * x * p[i - 1] - (i - 1) * p[i - 2])

    return p


print("\n")
driver()
print("\n")
