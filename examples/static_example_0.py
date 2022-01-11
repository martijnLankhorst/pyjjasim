
from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

"""
Static Example 0: Single vortex

Compute a single vortex at the centre of a small square array.

"""

if __name__ == "__main__":

    N = 4
    sq_array = SquareArray(N, N)
    f = 0.01
    n = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    problem = StaticProblem(sq_array, frustration=f, vortex_configuration=n)
    config, _, _ = problem.compute()
    config.plot()
    plt.show()




