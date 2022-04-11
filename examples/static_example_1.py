
from pyjjasim import *

import matplotlib.pyplot as plt

"""
Static Example 0: Single vortex

Compute a single vortex at the centre of a small square array.

"""

if __name__ == "__main__":

    N = 4
    sq_array = SquareArray(N, N)
    f = 0.1
    n = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    problem = StaticProblem(sq_array, external_flux=f, vortex_configuration=n)
    config, _, info = problem.compute()
    print(info)
    handles = config.plot()
    plt.show()







