import numpy as np
import matplotlib.pyplot as plt


def driver(n=3):
    # function
    f = lambda x: 1 / (1 + (10 * x) ** 2)

    # constant definitions
    N = n
    h = 2 / (N - 1)

    # create interpolating nodes
    x = np.zeros(N)
    for i in range(1, N + 1):
        x[i - 1] = -1 + (i - 1) * h

    # evaluate f at interpolating nodes
    y = f(x)

    # create matrix Vandermonde matrix
    V = np.zeros((N, N))
    for i, x_j in enumerate(x):
        for j in range(N):
            V[i][j] = x_j**j

    # solve for a_coefficients
    a_coefficients = np.matmul(np.linalg.inv(V), y)

    # create interval of points for plotting
    Neval = 1001
    a, b = -1, 1
    xeval = np.linspace(a, b, Neval)

    # f(x) and P(x) evaluations
    fex = f(xeval)
    yeval = evalMonomial(a_coefficients, xeval, N)

    plt.figure()
    if N == 2:
        plt.title(
            str(N)
            + "nd Lagrange Interpolating Polynomial on $f(x)= \\frac{1}{1+(10x)^2}$, x$\in$["
            + str(a)
            + ","
            + str(b)
            + "]"
        )
    elif N == 3:
        plt.title(
            str(N)
            + "rd Lagrange Interpolating Polynomial on $f(x)= \\frac{1}{1+(10x)^2}$, x$\in$["
            + str(a)
            + ","
            + str(b)
            + "]"
        )
    else:
        plt.title(
            str(N)
            + "th Lagrange Interpolating Polynomial on $f(x)= \\frac{1}{1+(10x)^2}$, x$\in$["
            + str(a)
            + ","
            + str(b)
            + "]"
        )
    plt.plot(x, y, "ro", label="Interpolating points")
    plt.plot(xeval, fex, "k", label="$f(x)$")
    plt.plot(xeval, yeval, "r--", label="P(x)")
    plt.legend()
    plt.show()

    return


def evalMonomial(a, x_eval, N):
    y_eval = np.zeros(len(x_eval))

    for i, x in enumerate(x_eval):
        sum = 0
        for j in range(N):
            sum += a[j] * x**j
        y_eval[i] = sum
    return y_eval


print("\n")
driver(3)
driver(6)
driver(7)
driver(10)
driver(11)
driver(19)
# for i in range(2, 20):
#     driver(i)
print("\n")
