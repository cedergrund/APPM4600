import numpy as np
import matplotlib.pyplot as plt


def driver(n=3, a=-1, b=1, M=True, high=False):
    # function
    f = lambda x: 1 / (1 + (10 * x) ** 2)

    # constant definitions
    N = n
    h = 2 / (N - 1)

    # create interpolating nodes
    x = np.zeros(N)
    for j in range(1, N + 1):
        x[j - 1] = np.cos(((2 * j - 1) * np.pi) / (2 * N))

    # evaluate f at interpolating nodes
    y = f(x)

    # create matrix Vandermonde matrix
    V = np.zeros((N, N))
    for i, x_j in enumerate(x):
        for j in range(N):
            V[i][j] = x_j**j

    # solve for a_coefficients
    a_coefficients = np.matmul(np.linalg.inv(V), y)

    # barycentric Lagrange interpolation formulas
    phi = lambda point: phiCalculation(N, point, x)
    w = wCalculation(N, x)

    # create interval of points for plotting
    Neval = 1001
    a, b = -1, 1
    xeval = np.linspace(a, b, Neval)

    # f(x) and P(x) evaluations
    fex = f(xeval)
    yeval1 = evalBarycentricLagrange1(phi, f, w, xeval, x, N)
    yevalm = evalMonomial(a_coefficients, xeval, N)

    plt.figure()
    if N == 2:
        plt.title(
            str(N)
            + "nd Lagrange Interpolating Polynomial on $f(x)= \\frac{1}{1+(10x)^2}$, ["
            + str(a)
            + ","
            + str(b)
            + "]"
        )
    elif N == 3:
        plt.title(
            str(N)
            + "rd Lagrange Interpolating Polynomial on $f(x)= \\frac{1}{1+(10x)^2}$, ["
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
    if high:
        plt.plot(x, y, "ro", label="Interpolating points")
        if M:
            plt.plot(xeval, yevalm, "b--", label="P(x) - Monomial")
            plt.plot(xeval, yeval1, "r.--", label="P(x) - Barycentric")
        else:
            plt.plot(xeval, yeval1, "r.--", label="P(x) - Barycentric")

    else:
        plt.plot(x, y, "ro", label="Interpolating points")
        plt.plot(xeval, yevalm, "b", label="P(x) - Monomial")
        plt.plot(xeval, yeval1, "r--", label="P(x) - Barycentric")
    plt.plot(xeval, fex, "k", label="$f(x)$")
    plt.xlim([a, b])
    plt.legend()
    plt.show()

    return


def phiCalculation(N, point, x):
    product = 1
    for x_i in x:
        product *= point - x_i

    return product


def wCalculation(N, x):
    w = np.ones(N)

    for j, x_j in enumerate(x):
        for i, x_i in enumerate(x):
            if i != j:
                w[j] *= (x_j - x_i) ** -1
    return w


def evalBarycentricLagrange1(phi, f, w, x_eval, nodes, N):
    y_eval = np.zeros(len(x_eval))

    for i, x in enumerate(x_eval):
        sum = 0

        for j in range(1, N + 1):
            if x == nodes[j - 1]:
                sum = f(x)

                break

            base = w[j - 1] / (x - nodes[j - 1])
            sum += base * f(nodes[j - 1])

        if not phi(x):
            y_eval[i] = sum
            continue
        y_eval[i] = phi(x) * sum

    return y_eval


def evalMonomial(a, x_eval, N):
    y_eval = np.zeros(len(x_eval))

    for i, x in enumerate(x_eval):
        sum = 0
        for j in range(N):
            sum += a[j] * x**j
        y_eval[i] = sum
    return y_eval


print("\n")
# for i in range(2, 20):
#     driver(i)
driver(19)
driver(81, high=True, M=False)
driver(81, high=True)
print("\n")
