import numpy as np
import matplotlib.pyplot as plt


def driver():
    sigma = np.zeros(17)
    for i, j in enumerate(range(-16, 1)):
        sigma[i] = 10**j

    f_pi = lambda x: np.cos(np.pi + sigma[x]) - np.cos(np.pi)
    f_10e6 = lambda x: np.cos(10e6 + sigma[x]) - np.cos(10e6)
    g_pi = lambda x: -2 * np.sin(np.pi + sigma[x] / 2) * np.sin(sigma[x] / 2)
    g_10e6 = lambda x: -2 * np.sin(10e6 + sigma[x] / 2) * np.sin(sigma[x] / 2)

    indices = [i for i in range(0, 17)]

    colors = plt.cm.Oranges(np.linspace(22, 3, 12))

    output = f_pi(indices)
    print(output)
    fig, ax = plt.subplot()
    the_table = plt.table(
        cellText=output,
        rowLabels=sigma,
        rowColours=colors,
        colLabels=["$X=\pi$"],
        loc="bottom",
    )
    plt.show()
    # # plotting for part a
    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.tight_layout(pad=4)
    # ax1.plot(sigma, f_pi(indices))
    # ax1.plot(sigma, g_pi(indices))
    # ax1.set_title(
    #     "$X=\pi$",
    #     fontsize=16,
    # )
    # ax1.set_xscale("log")
    # # ax1.set_ylabel("", fontsize=14)

    # # plotting for part b
    # ax2.plot(sigma, f_10e6(indices))
    # ax2.plot(sigma, g_10e6(indices))
    # ax2.set_title(
    #     "$X=10e6$",
    #     fontsize=16,
    # )
    # ax2.set_xscale("log")

    # # ax2.set_xlabel("$\sigma$", fontsize=14)
    # # ax2.set_ylabel("$P_2(X)$", fontsize=14)

    # # show figure
    # plt.show()

    return


driver()
