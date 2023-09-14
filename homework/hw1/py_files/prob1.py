import numpy as np
import matplotlib.pyplot as plt


def driver():
    # defining variable for x-axis and the 2 P(x) functions
    x = np.arange(1.920, 2.080, 0.001)
    p1 = (
        lambda x: x**9
        - 18 * x**8
        + 144 * x**7
        - 672 * x**6
        + 2016 * x**5
        - 4032 * x**4
        + 5376 * x**3
        - 4608 * x**2
        + 2304 * x
        - 512
    )
    p2 = lambda x: (x - 2) ** 9

    # plotting for part a
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(x, p1(x))
    ax1.set_title(
        "$P(x)=x^9-18x^8+144x^7-672x^6+2016x^5-4032x^4+5376x^3-4608x^2+2304x-51$",
        fontsize=16,
    )
    ax1.set_ylabel("$P_1(X)$", fontsize=14)

    # plotting for part b
    ax2.plot(x, p2(x))
    ax2.set_title(
        "$P(x)=(x-2)^9$",
        fontsize=16,
    )
    ax2.set_xlabel("$X$", fontsize=14)
    ax2.set_ylabel("$P_2(X)$", fontsize=14)

    # show figure
    plt.show()

    return


driver()
