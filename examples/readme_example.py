
from pyjjasim import *

import matplotlib.pyplot as plt

"""
Static Example 0: Single vortex

Compute a single vortex at the centre of a small square array.

"""

if __name__ == "__main__":

    sq_array = SquareArray(10, 10)
    problem = StaticProblem(sq_array, external_flux=0.1)
    config, status, info = problem.compute()
    config.plot(node_quantity="phi")

    n = np.zeros(sq_array.face_count())
    n[sq_array.locate_faces(x=[2.5,6.5], y=[2.5,6.5])] = 1
    config, status, info = problem.new_problem(vortex_configuration=n).compute()
    config.plot(node_quantity="phi")
    plt.show()




