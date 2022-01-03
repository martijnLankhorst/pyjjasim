import time

import numpy as np
import scipy.linalg

import josephson_circuit as jja
import static_problem as sp

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def get_width(f, I, N):
    Ip = 0.5 * N
    idx = np.flatnonzero((I[:-1] < Ip) & (I[1:] > Ip))
    f1, f2, I1, I2 = f[idx], f[idx + 1], I[idx], I[idx + 1]
    return ((I2 - Ip) * f1 + (Ip - I1) * f2) / (I2 - I1)

if __name__ == "__main__":
    beta_L_list = np.array([0.1, 0.2, 0.5, 1, 1.5, 2])
    # N_list = np.array([2, 3, 4, 5, 7, 10, 15, 25])
    N = 10
    widths = np.zeros(len(beta_L_list))
    for i, beta_L in enumerate(beta_L_list):
        array = jja.SquareArray(N, N)
        array.set_inductance_factors(beta_L)
        f, I, _, _ = sp.StaticProblem(array, current_sources=array.horizontal_junctions()).compute_stable_region(angles=np.linspace(0, 0.5 * np.pi, 16))
        widths[i] = get_width(f, I, N)
        print(widths[i])

    plt.plot(beta_L_list, widths)
    plt.plot(beta_L_list, (1 + 0.5 * beta_L_list) * 0.5/9 * 0.98)
    plt.show()

