import numpy as np
import matplotlib.pyplot as plt


def driver():
    # define functions to plot error
    f_real = lambda x: np.sin(x)
    t6 = lambda x: x - (x**3) / 6 + (x**5) / 120

    p33 = lambda x: (x - 7 / 60 * x**3) / (1 + 1 / 20 * x**2)
    p24 = lambda x: (x) / (1 + 1 / 6 * x**2 + 7 / 360 * x**4)
    p42 = lambda x: (x - 7 / 60 * x**3) / (1 + 1 / 20 * x**2)

    # Interval of interest
    a = 0
    b = 5

    #  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a, b, N + 1)
    fex = f_real(xeval)
    ft6 = t6(xeval)
    fp33 = p33(xeval)
    fp24 = p24(xeval)
    fp42 = p42(xeval)

    plt.figure()
    plt.plot(xeval, fex, "k-", label="f(x)")
    plt.plot(xeval, ft6, "r--", label="6th order Maclaurin")
    plt.plot(xeval, fp33, "b-.", label="Pade (a)")
    plt.plot(xeval, fp24, "g--", label="Pade (b)")
    plt.plot(xeval, fp42, "m:", label="Pade (c)")
    plt.title("Maclaurin vs Pade approximation of $f(x)= \sin{x}$ on $[0,5]$")
    plt.legend()
    plt.show

    plt.figure()
    ft6err = abs(ft6 - fex)
    fp33err = abs(fp33 - fex)
    fp24err = abs(fp24 - fex)
    fp42err = abs(fp42 - fex)
    plt.semilogy(xeval, ft6err, "r--", label="6th order Maclaurin")
    plt.semilogy(xeval, fp33err, "b-.", label="Pade (a)")
    plt.semilogy(xeval, fp24err, "g--", label="Pade (b)")
    plt.semilogy(xeval, fp42err, "m:", label="Pade (c)")
    plt.title("Error")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    driver()
