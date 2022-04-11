import time

from pyjjasim import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

"""
EXAMPLE 4: Parameter optimization

Compute maximal external_flux bounds and maximal current 
"""

if __name__ == "__main__":
    # define arrays
    array = SquareArray(12, 12)
    # Ih, Iv = 1.0 * array.horizontal_junctions(), 1.0 * array.vertical_junctions()

    # compute external_flux bounds for zero vortex state
    prob = StaticProblem(array, external_flux=1, current_sources=0)
    (smallest_f, largest_f), (s_config, l_config), _ = prob.compute_external_flux_bounds()
    print(f"smallest and largest external_flux for which the zero-vortex state exists: {smallest_f}, {largest_f}")
    s_config.plot(title=f"minimal external_flux in zero vortex state (f={np.round(smallest_f, 4)})")

    # compute maximal current
    prob = StaticProblem(array, external_flux=0, current_sources=array.current_base(angle=0))
    I_factor, max_I_config, info = prob.compute_maximal_current()
    np.set_printoptions(linewidth=10000000)
    print(f"largest current factor {I_factor} at which the zero-vortex state exists at zero external_flux")
    max_I_config.plot(title=f"maximal current in zero vortex state (I_factor={I_factor:.3f})")

    # compute extermum in Is-f space using compute_stable_region()
    prob = StaticProblem(array, external_flux=1, current_sources=array.current_base(angle=np.pi/2))
    f_factors, I_factors, _, _ = prob.compute_stable_region()
    plt.subplots()
    plt.plot(f_factors, I_factors)
    plt.xlabel("external_flux factors")
    plt.ylabel("current source factors")
    plt.title("Region in f-I parameter space where the zero-vortex state is stable")

    # compute direction dependent maximal current using compute_maximal_parameter()
    prob = StaticProblem(array)
    angles = np.linspace(0, 2*np.pi, 33)
    I_factor = np.zeros(len(angles))
    tic = time.perf_counter()
    for i, angle in enumerate(angles):
        prob_func = lambda x: prob.new_problem(current_sources=x * array.current_base(angle=angle))
        I_factor[i], _, _, _ = compute_maximal_parameter(prob_func)
    print(time.perf_counter() - tic)
    plt.subplots()
    plt.plot(np.cos(angles) * I_factor, np.sin(angles) * I_factor, marker="o")
    plt.xlabel("horizontal current")
    plt.ylabel("vertical current")
    plt.title("angle dependent maximal current for zero-vortex state")
    plt.show()
