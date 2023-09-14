import numpy as np


def driver():
    # define vectors
    t = np.arange(0, np.pi, np.pi / 30)
    y = np.cos(t)

    # compute sum
    S = 0
    for k in range(len(t)):
        S += t[k] * y[k]

    # print sum and stop
    print("the sum is:", S)
    return


driver()
