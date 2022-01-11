
from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

"""
Static Example 0: Single vortex

Compute a single vortex at the centre of a small square array.

"""

if __name__ == "__main__":

    N = 10
    sq_array = SquareArray(N, N)
    dt = 0.05
    Nt = 1000
    f = 0.1
    Is = 0.9 * sq_array.horizontal_junctions()
    T = 0.01
    problem = TimeEvolutionProblem(sq_array, time_step=dt,
                                   time_step_count=Nt, current_sources=Is,
                                   frustration=f, temperature=T)
    config = problem.compute()
    config.animate(junction_quantity="theta", arrow_scale=None)
    plt.show()
