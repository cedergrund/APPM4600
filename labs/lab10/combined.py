import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad


def driver(n=2):
    #  function you want to approximate
    f = lambda x: 1 / (1 + x**2)
    f = lambda x: np.exp(x)

    # Interval of interest
    a = -1
    b = 1
    # weight function
    w1 = lambda x: 1.0
    w2 = lambda x: 1 / np.sqrt(1 - x**2)

    # order of approximation
    n = n

    #  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a, b, N + 1)
    lpval = np.zeros(N + 1)
    cpval = np.zeros(N + 1)

    for kk in range(N + 1):
        lpval[kk] = eval_legendre_expansion(f, a, b, w1, n, xeval[kk])
        cpval[kk] = eval_chebychev_expansion(f, a, b, w2, n, xeval[kk])

    """ create vector with exact values"""
    fex = np.zeros(N + 1)
    for kk in range(N + 1):
        fex[kk] = f(xeval[kk])

    plt.figure()
    plt.plot(xeval, fex, "k-", label="f(x)")
    plt.plot(xeval, lpval, "r--", label="legendre")
    plt.plot(xeval, cpval, "b--", label="chebychev")
    plt.suptitle("Legendre Expansion approximation of $f(x)= \exp{x}$ on $[-1,1]$")
    plt.title("order {}".format(n))
    plt.legend()
    plt.show

    plt.figure()
    lerr = abs(lpval - fex)
    cerr = abs(cpval - fex)
    plt.semilogy(xeval, lerr, "r--", label="legendre")
    plt.semilogy(xeval, cerr, "b--", label="chebychev")
    plt.title("Error")
    plt.legend()
    plt.show()


def eval_legendre_expansion(f, a, b, w, n, x):
    #   This subroutine evaluates the Legendre expansion

    #  Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab
    p = eval_legendre(n, x)
    # initialize the sum to 0
    pval = 0.0
    for j in range(0, n + 1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_legendre(j, x)[-1]
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x) ** 2 * w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac, err = quad(phi_j_sq, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: f(x) * phi_j(x) * w(x) / norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj, err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval + aj * p[j]

    return pval


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


def eval_chebychev_expansion(f, a, b, w, n, x):
    #   This subroutine evaluates the Legendre expansion

    #  Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab
    p = eval_chebychev(n, x)
    # initialize the sum to 0
    pval = 0.0
    for j in range(0, n + 1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_chebychev(j, x)[-1]
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x) ** 2 * w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac, err = quad(phi_j_sq, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: f(x) * phi_j(x) * w(x) / norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj, err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval + aj * p[j]

    return pval


def eval_chebychev(n, x):
    assert n >= 0, "Invalid n as input"

    p = np.zeros(n + 1)
    if n >= 0:
        p[0] = 1
    if n >= 1:
        p[1] = x

    for i in range(2, n + 1):
        p[i] = 2 * x * p[i - 1] - p[i - 2]

    return p


if __name__ == "__main__":
    # run the drivers only if this is called from the command line
    driver(2)
