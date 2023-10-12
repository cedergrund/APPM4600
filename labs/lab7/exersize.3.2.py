import numpy as np
import matplotlib.pyplot as plt


def driver(n=3):
    f = lambda x: 1 / (1 + (10 * x) ** 2)

    N = n
    """ interval"""
    a = -1
    b = 1

    """ create equispaced interpolation nodes"""
    xint = np.linspace(a, b, N + 1)
    xint = np.zeros(N)
    for j in range(1, N + 1):
        xint[j - 1] = np.cos(((2 * j - 1) * np.pi) / (2 * N))

    """ create interpolation data"""
    yint = f(xint)

    """ create points for evaluating the Lagrange interpolating polynomial"""
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    yeval_l = np.zeros(Neval + 1)

    """ evaluate lagrange poly """
    for kk in range(1, Neval + 1):
        yeval_l[kk] = eval_lagrange(xeval[kk], xint, yint, N)

    """ create vector with exact values"""
    fex = f(xeval)
    plt.figure()
    plt.title("plot. N=" + str(N))
    plt.plot(xeval, fex, "ro-", label="actual")
    plt.plot(xeval, yeval_l, "b--")
    plt.legend()

    plt.figure()
    err_l = abs(yeval_l - fex)
    plt.title("error. N=" + str(N))
    plt.semilogy(xeval, err_l, "b--", label="lagrange")
    plt.legend()
    plt.show()

    return


# imported methods
###########################################
def eval_lagrange(xeval, xint, yint, N):
    lj = np.ones(N)

    for count in range(N):
        for jj in range(N):
            if jj != count:
                if xint[count] - xint[jj] == 0:
                    print(count, jj)
                lj[count] = lj[count] * (xeval - xint[jj]) / (xint[count] - xint[jj])

    yeval = 0.0

    for jj in range(N):
        yeval = yeval + yint[jj] * lj[jj]

    return yeval


print("\n")
driver(3)
driver(10)
driver(18)
# for i in range(17, 20):
#     driver(i)
print("\n")
