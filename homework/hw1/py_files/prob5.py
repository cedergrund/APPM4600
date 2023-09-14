import pandas as pd
import numpy as np


# Function to create the tables
def create_table(delta, x_delta, y_delta, n1, n2):
    data = {
        "δ ": delta,
        "Given expression (part b)": x_delta,
        "My expression (part b)": y_delta,
        "Taylor w ξ=x (part c)": n1,
        "Taylor w ξ=x+δ (part c)": n2,
    }

    df = pd.DataFrame(data)
    return df


# building delta range
delta = np.zeros(17)
indices = [i for i in range(0, 17)]
for i, j in enumerate(range(-16, 1)):
    delta[i] = 10**j


# creating functions
f_pi = lambda x: np.cos(np.pi + delta[x]) - np.cos(np.pi)
f_10e6 = lambda x: np.cos(10e6 + delta[x]) - np.cos(10e6)
g_pi = lambda x: -2 * np.sin(np.pi + delta[x] / 2) * np.sin(delta[x] / 2)
g_10e6 = lambda x: -2 * np.sin(10e6 + delta[x] / 2) * np.sin(delta[x] / 2)
h_pi_lower = lambda x: -(
    delta[x] * np.sin(np.pi) + 1 / 2 * delta[x] ** 2 * np.cos(np.pi)
)
h_pi_upper = lambda x: -(
    delta[x] * np.sin(np.pi) + 1 / 2 * delta[x] ** 2 * np.cos(np.pi + delta[x])
)
h_10e6_lower = lambda x: -(
    delta[x] * np.sin(10e6) + 1 / 2 * delta[x] ** 2 * np.cos(10e6)
)
h_10e6_upper = lambda x: -(
    delta[x] * np.sin(10e6) + 1 / 2 * delta[x] ** 2 * np.cos(10e6 + delta[x])
)

# data
d_f_pi = f_pi(indices)
d_g_pi = g_pi(indices)
d_f_10e6 = f_10e6(indices)
d_g_10e6 = g_10e6(indices)
d_h_pi_lower = h_pi_lower(indices)
d_h_pi_upper = h_pi_upper(indices)
d_h_10e6_lower = h_10e6_lower(indices)
d_h_10e6_upper = h_10e6_upper(indices)

# Create the table
table1 = create_table(delta, d_f_pi, d_g_pi, d_h_pi_lower, d_h_pi_upper)
table2 = create_table(delta, d_f_10e6, d_g_10e6, d_h_10e6_lower, d_h_10e6_upper)

# Display the table
print("\n\n")
print("     X = Pi \n", table1)
print("\n\n")
print("     X = 10e6 \n", table2)
print("\n\n")
