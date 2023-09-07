import pandas as pd
import numpy as np


# Function to create the table
def create_table1(sigma, x_data, y_data):
    data = {
        "Sigma": sigma,
        "Original expression at X=Pi": x_data,
        "My expression at X=Pi": y_data,
    }

    df = pd.DataFrame(data)
    return df


def create_table2(sigma, x_sigma, y_sigma):
    data = {
        "Sigma": sigma,
        "Original expression at X=10^6": x_sigma,
        "My expression at X=10^6": y_sigma,
    }

    df = pd.DataFrame(data)
    return df


sigma = np.zeros(17)
for i, j in enumerate(range(-16, 1)):
    sigma[i] = 10**j

f_pi = lambda x: np.cos(np.pi + sigma[x]) - np.cos(np.pi)
f_10e6 = lambda x: np.cos(10e6 + sigma[x]) - np.cos(10e6)
g_pi = lambda x: -2 * np.sin(np.pi + sigma[x] / 2) * np.sin(sigma[x] / 2)
g_10e6 = lambda x: -2 * np.sin(10e6 + sigma[x] / 2) * np.sin(sigma[x] / 2)

indices = [i for i in range(0, 17)]
# Example data
d_f_pi = f_pi(indices)
d_g_pi = g_pi(indices)
d_f_10e6 = f_10e6(indices)
d_g_10e6 = g_10e6(indices)

# Create the table
table1 = create_table1(sigma, d_f_pi, d_g_pi)
table2 = create_table2(sigma, d_f_10e6, d_g_10e6)


# Display the table
print(table1)
print("\n\n")
print(table2)
