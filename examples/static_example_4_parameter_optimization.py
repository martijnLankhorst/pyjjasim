import numpy as np

from pyjjasim import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


"""
EXAMPLE 4: Parameter optimization

Compute maximal frustration bounds and maximal current 
"""

if __name__ == "__main__":
    # define arrays
    array = SquareArray(12, 12)


    # compute frustration bounds for zero vortex state
    prob = StaticProblem(array, current_sources=0)
    (smallest_f, largest_f), (s_config, l_config), _ = prob.compute_frustration_bounds()
    print(f"smallest and largest frustration for which the zero-vortex state exists: {smallest_f}, {largest_f}")
    s_config.plot(title=f"minimal frustration in zero vortex state (f={smallest_f})")

    # compute maximal current
    prob = StaticProblem(array, frustration=0, current_sources=array.vertical_junctions())
    I_factor, net_I, max_I_config, _ = prob.compute_maximal_current()
    np.set_printoptions(linewidth=10000000)
    print(f"largest current factor {I_factor} (corresponding to net current of  {net_I}) at which the zero-vortex state exists at zero frustration")
    max_I_config.plot(title=f"maximal current in zero vortex state (net_I={net_I})")

    # compute extermum in Is-f space using compute_stable_region()
    prob = StaticProblem(array, frustration=0, current_sources=array.vertical_junctions())
    f, net_I, _, _ = prob.compute_stable_region()
    plt.subplots()
    plt.plot(f, net_I)
    plt.xlabel("frustration")
    plt.ylabel("net I")
    plt.title("Region in f-I parameter space where the zero-vortex state is stable")

    # compute direction dependent maximal current using compute_maximal_parameter()
    prob = StaticProblem(array)
    angles = np.linspace(0, 2*np.pi, 33)
    Ih, Iv = array.horizontal_junctions(), array.vertical_junctions()
    I_factor = np.zeros(len(angles))
    f_func = lambda x: 0
    for i, angle in enumerate(angles):
        Is_func = lambda x: x * (np.cos(angle) * Ih + np.sin(angle) * Iv)
        I_factor[i], _, _, _ = prob.compute_maximal_parameter(Is_func, f_func)
    plt.subplots()
    plt.plot(np.cos(angles) * I_factor, np.sin(angles) * I_factor, marker="o")
    plt.xlabel("horizontal current")
    plt.ylabel("vertical current")
    plt.title("angle dependent maximal current for zero-vortex state")
    plt.show()
