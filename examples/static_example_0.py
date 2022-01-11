
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

    np.set_printoptions(linewidth=10000)
    A = sq_array.get_cycle_matrix()
    M = sq_array.get_cut_matrix().astype(int)
    Mg = sq_array.graph.cut_space_matrix()
    Ag = sq_array.graph.face_cycle_matrix()
    # print(M.todense())
    print(A.todense())
    # print(Mg.todense())
    print(Ag.todense())
    print("orthog", M @ A.T)

    sq_array.plot(show_junction_ids=True)
    plt.show()
    f = 0.01
    n = [0, 0, 0, 0, 1, 0, 0, 0, 0]

    problem = StaticProblem(sq_array, frustration=f, vortex_configuration=n)

    config, _, _ = problem.compute()

    config.plot()
    plt.show()




