import numpy as np
import matplotlib.pyplot as plt


def driver():
    # function definition
    f = lambda x: x - 4 * np.sin(2 * x) - 3

    # plot over x-axis that includes all roots
    X = np.arange(-np.pi, 2 * np.pi, 0.001)
    fig, ax1 = plt.subplots()
    ax1.plot(X, f(X))
    ax1.set_title(
        "$f(x)=x-4*sin(2x)-3$",
        fontsize=16,
    )
    ax1.set_ylabel("$f(X)$", fontsize=14)
    ax1.set_xlabel("x", fontsize=14)
    plt.grid(True)
    plt.show()

    return


driver()
