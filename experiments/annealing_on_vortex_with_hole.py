
from pyJJAsim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

"""
Annealing on array with hole
"""

def make_hole_array(N, L):
    full_array = SquareArray(N, N)
    x, y = full_array.get_node_coordinates()
    hole_face_ids = (x > (N-1-L)/2) & (x < (N-1+L)/2) & \
                    (y > (N-1-L)/2) & (y < (N-1+L)/2)
    hole_array = full_array.remove_nodes(hole_face_ids.flatten())
    return hole_array

if __name__ == "__main__":

    # define array
    N = 20
    L = 6
    hole_array = make_hole_array(N, L)

    # define problem
    f = 0.03 * hole_array.get_face_areas()
    time_step = 2.0
    interval_steps = 20
    interval_count = 2000
    vortex_mobility = 0.001
    problem_count = 250
    start_T = 1.0
    T_factor = 1.03
    problem = AnnealingProblem(hole_array, time_step=time_step, interval_steps=interval_steps,
                               interval_count=interval_count, vortex_mobility=vortex_mobility,
                               frustration=f, problem_count=problem_count,
                               start_T=start_T, T_factor=T_factor)
    # do time simulation
    out = problem.compute()
    vortex_configurations, energies, status, configurations, T = out

    plt.subplots()
    plt.hist(energies, bins=100)
    plt.xlabel("mean energy per junction")
    plt.ylabel("histogram count")

    # plot temperature evolution
    plt.subplots()
    plt.plot(time_step * interval_steps * np.arange(interval_count), T)
    plt.xlabel("time")
    plt.ylabel("annealing temperature")

    lowest_state = np.argmin(energies)
    configurations[lowest_state].plot(vortex_diameter=0.8, figsize=[8, 8])
    plt.title(f"lowest found state at E={np.round(energies[lowest_state], 5)}")

    mask = energies > (energies[lowest_state] + 1E-7)
    second_lowest = np.flatnonzero(mask)[np.argmin(energies[mask])]
    configurations[second_lowest].plot(vortex_diameter=0.8, figsize=[8, 8])
    plt.title(f"second lowest state at E={np.round(energies[second_lowest], 5)}")

    plt.show()
