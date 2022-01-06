
from pyJJAsim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

"""
Dynamic example 6: Annealing

Finds low energy states by doing a time evolution with gradually decreasing temperature.

"""

if __name__ == "__main__":

    # define array
    N = 20
    sq_array = SquareArray(N, N)

    # define problem
    f = 0.1
    time_step = 2.0
    interval_steps = 20
    interval_count = 400
    vortex_mobility = 0.001
    problem_count = 50
    start_T = 1.0
    T_factor = 1.03
    problem = AnnealingProblem(sq_array, time_step=time_step, interval_steps=interval_steps,
                               interval_count=interval_count, vortex_mobility=vortex_mobility,
                               frustration=f, problem_count=problem_count,
                               start_T=start_T, T_factor=T_factor)
    # do time simulation
    out = problem.compute()
    vortex_configurations, energies, status, configurations, T = out

    plt.subplots()
    plt.hist(energies)
    plt.xlabel("mean energy per junction")
    plt.ylabel("histogram count")

    # plot temperature evolution
    plt.subplots()
    plt.plot(time_step * interval_steps * np.arange(interval_count), T)
    plt.xlabel("time")
    plt.ylabel("annealing temperature")

    lowest_state = np.argmin(energies)
    configurations[lowest_state].plot()
    plt.title(f"lowest found state at E={np.round(energies[lowest_state], 5)}")
    plt.show()
