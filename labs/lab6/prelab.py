import numpy as np


def driver():
    # function definition
    f = lambda x: np.cos(x)

    # constants
    s = np.pi / 2
    actual_derivative = -1

    # h iterations
    h = 0.01 * 2 ** (-1 * np.arange(0, 10, dtype=float))

    # running forward difference method
    print("FORWARD DIFFERENCE")
    fd_iterations = forwardDifference(f, h, s)
    print("iterations:")
    print(fd_iterations)
    alpha = 1
    print("\nrunning order of convergence with alpha =", alpha)
    print(orderOfConvergence(actual_derivative, fd_iterations, alpha))
    # alpha = 2
    # print("\nrunning order of convergence with alpha =", alpha)
    # print(orderOfConvergence(actual_derivative, fd_iterations, alpha))
    print("\n")

    # running centered difference method
    print("CENTERED DIFFERENCE")
    cd_iterations = centeredDifference(f, h, s)
    print("iterations:")
    print(cd_iterations)
    alpha = 1
    print("\nrunning order of convergence with alpha =", alpha)
    print(orderOfConvergence(actual_derivative, cd_iterations, alpha))
    # alpha = 2
    # print("\nrunning order of convergence with alpha =", alpha)
    # print(orderOfConvergence(actual_derivative, cd_iterations, alpha))
    print("\n")

    print("Difference in techniques / h value:")
    print(np.abs(fd_iterations - cd_iterations))
    return


def forwardDifference(f, h, s):
    f_s = f(s)
    approximations = np.zeros(len(h))

    for i, n in enumerate(h):
        approximations[i] = f(s + n) - f_s

    return approximations / h


def centeredDifference(f, h, s):
    approximations = np.zeros(len(h))

    for i, n in enumerate(h):
        approximations[i] = f(s + n) - f(s - n)

    return approximations / (2 * h)


def orderOfConvergence(actual, iterations, alpha):
    seq = np.zeros(len(iterations) - 1)

    for n, i in enumerate(iterations):
        if n == len(iterations) - 1:
            continue

        seq[n] = np.abs(iterations[n + 1] - actual) / (np.abs(i - actual)) ** alpha

    return seq


print("\n")
driver()
print("\n")
