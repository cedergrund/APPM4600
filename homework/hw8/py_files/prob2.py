import numpy as np
import matplotlib.pyplot as plt


def driver(N=3, a=-5, b=5):
    n = N
    # function
    f = lambda x: 1 / (1 + x**2)
    df = lambda x: (-2 * x) / (1 + x**2) ** 2

    # create interpolating nodes - Chebychev nodes
    # x = np.zeros(N)
    # for j in range(1, N + 1):
    #     x[j - 1] = 5 * np.cos(((2 * j - 1) * np.pi) / (2 * N))
    # N = n - 1
    x = np.zeros(N + 1)
    for j in range(0, N + 1):
        x[j] = 5 * np.cos(j / n * np.pi)
    x = np.sort(x)

    # evaluate f at interpolating nodes
    y = f(x)
    dy = df(x)

    # create interval of points for plotting
    Neval = 1000
    xeval = np.linspace(a, b, Neval)
    yeval_l = np.zeros(Neval)
    yeval_h = np.zeros(Neval)
    yeval_nc = evalCubicSpline(x, y, xeval, Neval, N)
    yeval_cc = evalCubicSpline(x, y, xeval, Neval, N, clamped=[df(a), df(b)])

    # evaluate lagrange poly
    for kk in range(Neval):
        yeval_l[kk] = eval_lagrange(xeval[kk], x, y, N)
        yeval_h[kk] = eval_hermite(xeval[kk], x, y, dy, N)

    # f(x) and P(x) evaluations
    fex = f(xeval)

    plt.figure()
    if N == 2:
        plt.title(
            str(n)
            + "nd Interpolating Polynomials on $f(x)= \\frac{1}{1+x^2}$, ["
            + str(a)
            + ","
            + str(b)
            + "]"
        )
    elif N == 3:
        plt.title(
            str(n)
            + "rd Interpolating Polynomials on $f(x)= \\frac{1}{1+x^2}$, ["
            + str(a)
            + ","
            + str(b)
            + "]"
        )
    else:
        plt.title(
            str(n)
            + "th Interpolating Polynomials on $f(x)= \\frac{1}{1+x^2}$, x$\in$["
            + str(a)
            + ","
            + str(b)
            + "]"
        )

    plt.plot(x, y, "ro", label="Interpolating points")
    plt.plot(xeval, yeval_l, "b--", label="Lagrange")
    plt.plot(xeval, yeval_h, "m--", label="Hermite")
    plt.plot(xeval, yeval_nc, "g--", label="Natural Cubic")
    plt.plot(xeval, yeval_cc, "c--", label="Clamped Cubic")
    plt.plot(xeval, fex, "k", label="$f(x)$")
    # plt.xlim([a, b])
    plt.legend()
    plt.show

    plt.figure()
    err_l = abs(yeval_l - fex)
    err_h = abs(yeval_h - fex)
    err_nc = abs(yeval_nc - fex)
    err_cc = abs(yeval_cc - fex)
    plt.title("error. N=" + str(n))
    plt.semilogy(xeval, err_l, "b--", label="Lagrange")
    plt.semilogy(xeval, err_h, "m--", label="Hermite")
    plt.semilogy(xeval, err_nc, "g--", label="Natural Cubic")
    plt.semilogy(xeval, err_cc, "c--", label="Clamped Cubic")
    plt.legend()
    plt.show()

    return


# lagrange
def eval_lagrange(xeval, xint, yint, N):
    lj = np.ones(N + 1)

    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lj[count] = lj[count] * (xeval - xint[jj]) / (xint[count] - xint[jj])

    yeval = 0.0

    for jj in range(N + 1):
        yeval = yeval + yint[jj] * lj[jj]

    return yeval


# hermite
def eval_hermite(xeval, xint, yint, ypint, N):
    # Evaluate all Lagrange polynomials
    lj = np.ones(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lj[count] = lj[count] * (xeval - xint[jj]) / (xint[count] - xint[jj])

    # Construct the l_j'(x_j)
    lpj = np.zeros(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lpj[count] = lpj[count] + 1.0 / (xint[count] - xint[jj])

    yeval = 0.0

    for jj in range(N + 1):
        Qj = (1.0 - 2.0 * (xeval - xint[jj]) * lpj[jj]) * lj[jj] ** 2
        Rj = (xeval - xint[jj]) * lj[jj] ** 2
        yeval = yeval + yint[jj] * Qj + ypint[jj] * Rj

    return yeval


# cubic spline
def evalCubicSpline(xint, yint, xeval, Neval, Nint, clamped=[0, 0]):
    #  create vector to store the evaluation of the linear splines
    yeval = np.zeros(Neval)
    M = identifyM(xint, yint, Nint, clamped)

    for jint in range(Nint):
        # find indices of xeval in interval (xint(jint),xint(jint+1))
        # let ind denote the indices in the intervals
        # let n denote the length of ind
        ind = subintervalFind(xint, xeval, interval=jint)

        n = len(ind)

        # create cubic polynomial for subinterval
        a1 = xint[jint]
        fa1 = yint[jint]
        b1 = xint[jint + 1]
        fb1 = yint[jint + 1]
        cubic = cubicPolyEvaluator(M[jint], M[jint + 1], a1, b1, fa1, fb1)

        for kk in range(n):
            # using cubic polynomial to evaluate plot at each point in xeval
            yeval[ind[kk]] = cubic(xeval[ind[kk]])
    return yeval


def identifyM(x_int, y_int, N, clamped):
    # create right hand side + h vector
    y = np.zeros(N + 1)
    h = np.zeros(N)

    if clamped != [0, 0]:
        h0 = x_int[1] - x_int[0]
        y[0] = -clamped[0] + (y_int[1] - y_int[0]) / h0
        hnm1 = x_int[N] - x_int[N - 1]
        y[N] = -clamped[1] + (y_int[N] - y_int[N - 1]) / hnm1

    for i in range(1, N):
        hi = x_int[i] - x_int[i - 1]
        hip = x_int[i + 1] - x_int[i]
        y[i] = (y_int[i + 1] - y_int[i]) / hip - (y_int[i] - y_int[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    # create matrix
    A = np.zeros((N + 1, N + 1))
    if clamped != [0, 0]:
        A[0][0] = h[0] / 3
        A[0][1] = h[0] / 6
        A[N][N] = h[N - 1] / 3
        A[N][N - 1] = h[N - 1] / 6
    else:
        A[0][0] = 1
        A[N][N] = 1

    for i in range(1, N):
        A[i][i - 1] = h[i - 1] / 6
        A[i][i] = (h[i] + h[i - 1]) / 3
        A[i][i + 1] = h[i] / 6

    M = np.matmul(np.linalg.inv(A), y)
    return M


def cubicPolyEvaluator(Mi, Mi1, xi, xi1, fxi, fxi1):
    hi = xi1 - xi
    C = (fxi / hi) - (Mi * hi / 6)
    D = (fxi1 / hi) - (hi * Mi1 / 6)

    f = (
        lambda x: (((xi1 - x) ** 3) * Mi + ((x - xi) ** 3) * Mi1) / (6 * hi)
        + C * (xi1 - x)
        + D * (x - xi)
    )
    return f


def subintervalFind(xint, xeval, interval):
    if len(xint) <= (interval + 1):
        print("error: inputted interval not valid")
        return []
    a, b = xint[interval], xint[interval + 1]
    ind = np.where((xeval >= a) & (xeval <= b))
    return ind[0]


print("\n")
driver(5)
driver(10)
driver(15)
driver(20)
print("\n")
