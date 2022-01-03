import numpy as np

from multiprocessing import Pool

from josephson_circuit import *
from static_problem import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


"""
EXAMPLE 11: Multithreading

"""

def func(L):
    N = 10
    sq_array = SquareArray(N, N)
    sq_array.set_inductance_factors(L)
    prob = StaticProblem(sq_array, current_sources=sq_array.horizontal_junctions())
    f, I, _, _ = prob.compute_stable_region(angles=np.linspace(0, 2*np.pi, 61))
    return f, I

if __name__ == '__main__':
    L = [0, 0.5, 1, 2]
    with Pool() as pool:
        out = pool.map(func, np.arange(4))

    for i in range(4):
        plt.plot(out[i][0], out[i][1], label=f"beta_L={L[i]}")
    plt.xlabel("frustration")
    plt.ylabel("maximal array current")
    plt.legend()
    plt.show()
