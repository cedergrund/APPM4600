import numpy as np
import matplotlib.pyplot as plt


def driver():
    # constants
    R = 1.2
    delta_r = 0.1
    f = 15
    p = 0

    # create parametric curves
    x = lambda x: R * (1 + delta_r * np.sin(f * x + p)) * np.cos(x)
    y = lambda x: R * (1 + delta_r * np.sin(f * x + p)) * np.sin(x)

    # map parametric functions over theta domain
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_vals = x(theta)
    y_vals = y(theta)

    # plot figure and stop
    ax = plt.figure().add_subplot()
    ax.plot(x_vals, y_vals, label="$0\leq \\theta \leq 2\pi$")
    ax.set_title("Wavy Circles", fontsize=16)
    ax.set_xlabel("x($\\theta$)", fontsize=14)
    ax.set_ylabel("y($\\theta$)", fontsize=14)
    ax.legend()
    plt.show()
    return


driver()
