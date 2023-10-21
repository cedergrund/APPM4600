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

    """ create interpolation data"""
    yint = f(xint)

    """ create matrix Vandermonde matrix """
    A = np.zeros((N + 1, N + 1))
    for i, x_j in enumerate(xint):
        for j in range(N + 1):
            A[i][j] = x_j**j
    a_coefficients = np.matmul(np.linalg.inv(A), yint)

    """ create points for evaluating the Lagrange interpolating polynomial"""
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    y_eval_m = np.zeros(Neval + 1)
    yeval_l = np.zeros(Neval + 1)
    yeval_dd = np.zeros(Neval + 1)

    y_eval_m = evalMonomial(a_coefficients, xeval, N)

    """Initialize and populate the first columns of the
     divided difference matrix. We will pass the x vector"""
    y = np.zeros((N + 1, N + 1))

    for j in range(N + 1):
        y[j][0] = yint[j]

    y = dividedDiffTable(xint, y, N + 1)

    """evaluate the linear spline"""
    yevallin = evalCubicSpline(xeval, Neval + 1, a, b, f, N)

    """ evaluate lagrange poly """
    for kk in range(Neval + 1):
        yeval_l[kk] = eval_lagrange(xeval[kk], xint, yint, N)
        yeval_dd[kk] = evalDDpoly(xeval[kk], xint, y, N)

    """ create vector with exact values"""
    fex = f(xeval)

    plt.figure()
    plt.title("plot. N=" + str(N))
    plt.plot(xeval, fex, "r-", label="actual")
    plt.plot(xint, yint, "ro", label="Interpolating nodes")
    plt.plot(xeval, y_eval_m, "k--")
    plt.plot(xeval, yeval_l, "b--")
    plt.plot(xeval, yeval_dd, "c:")
    plt.plot(xeval, yevallin, "m")
    plt.legend()

    plt.figure()
    err_m = abs(y_eval_m - fex)
    err_l = abs(yeval_l - fex)
    err_dd = abs(yeval_dd - fex)
    err_lin = abs(yevallin - fex)
    plt.title("error. N=" + str(N))
    plt.semilogy(xeval, err_m, "k--", label="monomial")
    plt.semilogy(xeval, err_l, "b--", label="lagrange")
    plt.semilogy(xeval, err_dd, "c:", label="Newton DD")
    plt.semilogy(xeval, err_lin, "m", label="cubic spline")
    plt.legend()
    plt.show()
    return


def evalCubicSpline(xeval, Neval, a, b, f, Nint):
    """create the intervals for piecewise approximations"""
    xint = np.linspace(a, b, Nint + 1)

    """create vector to store the evaluation of the linear splines"""
    yeval = np.zeros(Neval)
    M = identifyM(xint, f(xint), Nint)
    for jint in range(Nint):
        """find indices of xeval in interval (xint(jint),xint(jint+1))"""
        """let ind denote the indices in the intervals"""
        """let n denote the length of ind"""
        ind = subintervalFind(xint, xeval, interval=jint)
        n = len(ind)

        """temporarily store your info for creating a line in the interval of 
         interest"""
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint + 1]
        fb1 = f(b1)
        line = cubicPolyEvaluator(M[jint], M[jint + 1], a1, b1, fa1, fb1)

        for kk in range(n):
            """use your line evaluator to evaluate the lines at each of the points
            in the interval"""

            """yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)"""
            yeval[ind[kk]] = line(xeval[ind[kk]])
    return yeval


def identifyM(x_int, y_int, N):
    # create right hand side + h vector
    y = np.zeros(N + 1)
    h = np.zeros(N)
    for i in range(1, N):
        hi = x_int[i] - x_int[i - 1]
        hip = x_int[i + 1] - x_int[i]
        y[i] = (y_int[i + 1] - y_int[i]) / hip - (y_int[i] - y_int[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    # create matrix
    A = np.zeros((N + 1, N + 1))
    A[0][0] = 1
    for i in range(1, N):
        A[i][i - 1] = h[i - 1] / 6
        A[i][i] = (h[i] + h[i - 1]) / 3
        A[i][i + 1] = h[i] / 6
    A[N][N] = 1

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


# monomial
def evalMonomial(a, x_eval, N):
    y_eval = np.zeros(len(x_eval))

    for i, x in enumerate(x_eval):
        sum = 0
        for j in range(N + 1):
            sum += a[j] * x**j
        y_eval[i] = sum
    return y_eval


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


""" create divided difference matrix"""


def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = (y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j])
    return y


def evalDDpoly(xval, xint, y, N):
    """evaluate the polynomial terms"""
    ptmp = np.zeros(N + 1)

    ptmp[0] = 1.0
    for j in range(N):
        ptmp[j + 1] = ptmp[j] * (xval - xint[j])

    """evaluate the divided difference polynomial"""
    yeval = 0.0
    for j in range(N + 1):
        yeval = yeval + y[0][j] * ptmp[j]

    return yeval


print("\n")
driver(9)
# driver(10)
# driver(18)
# for i in range(2, 18):
#     driver(i)
print("\n")
