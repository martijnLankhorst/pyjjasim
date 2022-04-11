import numpy as np

from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

if __name__ == "__main__":

    array = SQUID()
    array.plot()

    Is = array.horizontal_junctions()
    angles = np.linspace(0, np.pi, 31)
    prob = StaticProblem(array, current_sources=Is, external_flux=1)

    plt.subplots()
    f, I, _, _ = prob.compute_stable_region(angles=angles)
    plt.plot(f, I, color=[0, 0, 1], label="n=0, beta_L=0")

    array.set_inductance(1)
    prob = StaticProblem(array, current_sources=Is, external_flux=1)
    f, I, _, _ = prob.compute_stable_region(angles=angles)
    plt.plot(f, I, color=[0, 0, 0], label="n=0, beta_L=1")

    plt.xlabel("external_flux")
    plt.ylabel("maximal current")
    plt.legend()
    plt.title("SQUID maximal current")
    plt.show()
