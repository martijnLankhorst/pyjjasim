
from pyjjasim import *

import matplotlib.pyplot as plt


"""
Static Example 1: Single vortex

Compute a single vortex at the centre of a square array.

"""

if __name__ == "__main__":

    print("STATIC EXAMPLE1: Single vortex")

    # define arrays
    N = 10
    sq_array = SquareArray(N, N)

    # define problem parameters
    f = 0.01                                                          # frustration
    n = np.zeros(sq_array.face_count(), dtype=int)                    # target vortex configuration
    centre_face_idx = sq_array.locate_faces((N - 1) / 2, (N - 1) / 2) # locating face idx at coordinate x=(N-1)/2, y=(N-1)/2
    n[centre_face_idx] = 1

    # define static problems
    problem = StaticProblem(sq_array, frustration=f, vortex_configuration=n)

    # approximations:
    approximation = problem.approximate()


    print(f"approx error: {approximation.get_error_kirchhoff_rules()} and "
          f"{approximation.get_error_winding_rules()}")

    approximation.plot(node_quantity="phi", title="approximation")

    config1, status1, info1 = problem.compute(initial_guess=approximation)

    print(f"config1 error: {config1.get_error_kirchhoff_rules()} and "
          f"{config1.get_error_winding_rules()}")
    print(info1)

    config1.plot(node_quantity="phi",
                 title="exact solution with approximation as initial guess")

    plt.show()

    # In practice one does not need to manually specify initial conditions.
    # By default an approximation is used as initial condition. One would just do:
    # problem = StaticProblem(sq_array, frustration=f, vortex_configuration=n)
    # config, status, info = problem.compute()





