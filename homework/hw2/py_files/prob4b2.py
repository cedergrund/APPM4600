import numpy as np
import matplotlib.pyplot as plt


def driver():
    # create parametric curves
    x = (
        lambda theta, R, delta_r, f, p: R
        * (1 + delta_r * np.sin(f * theta + p))
        * np.cos(theta)
    )
    y = (
        lambda theta, R, delta_r, f, p: R
        * (1 + delta_r * np.sin(f * theta + p))
        * np.sin(theta)
    )

    # plot 10 parametric functions using for loop
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_vals = np.zeros((10, 1000))
    y_vals = np.zeros((10, 1000))
    ax = plt.figure().add_subplot()

    for i in range(10):
        curve_num = i + 1
        R = curve_num
        delta_r = 0.05
        f = 2 + curve_num
        p = np.random.uniform(0, 2)

        x_vals[i] = x(theta, R, delta_r, f, p)
        y_vals[i] = y(theta, R, delta_r, f, p)

        ax.plot(x_vals[i], y_vals[i], label="curve " + str(curve_num))

    # finish plotting and stop
    ax.set_title("Wavy Circles pt.2", fontsize=16)
    ax.set_xlabel("x($\\theta$)", fontsize=14)
    ax.set_ylabel("y($\\theta$)", fontsize=14)
    ax.legend()
    plt.show()
    return


driver()
