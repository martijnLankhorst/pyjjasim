
from pyjjasim import *

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
    time_step = 1.0
    interval_steps = 20
    interval_count = 300
    vortex_mobility = 0.001
    problem_count = 40
    start_T = 1.0
    T_factor = 1.03
    problem = AnnealingProblem(sq_array, time_step=time_step, interval_steps=interval_steps,
                               interval_count=interval_count, vortex_mobility=vortex_mobility,
                               frustration=f, problem_count=problem_count,
                               start_T=start_T, T_factor=T_factor)
    # do time simulation
    status, configurations, T = problem.compute()
    energies = np.array([np.mean(c.get_energy()) for c in configurations], dtype=np.double)

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
    fig, ax = configurations[lowest_state].plot()
    fig.suptitle(f"lowest found state at E={np.round(energies[lowest_state], 5)}")
    plt.show()
