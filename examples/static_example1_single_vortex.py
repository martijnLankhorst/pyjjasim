
from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

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
    approx_arctan = problem.approximate(algorithm=0)
    approx_london = problem.approximate(algorithm=1)

    print(f"arctan approx error: {approx_arctan.get_error_kirchhoff_rules()} and {approx_arctan.get_error_winding_rules()}")
    print(f"london approx error: {approx_london.get_error_kirchhoff_rules()} and {approx_london.get_error_winding_rules()}")

    approx_arctan.plot(show_node_quantity=True, node_quantity_clim=[-np.pi, np.pi], title="arctan approximation")
    approx_london.plot(show_node_quantity=True, node_quantity_clim=[-np.pi, np.pi], title="london approximation")

    config1, status1, info1 = problem.compute(initial_guess=approx_arctan)
    config2, status2, info2 = problem.compute(initial_guess=approx_london)

    print(f"config1 error: {config1.get_error_kirchhoff_rules()} and {config1.get_error_winding_rules()}")
    print(f"config2 error: {config2.get_error_kirchhoff_rules()} and {config2.get_error_winding_rules()}")


    print(info1)
    print(info2)
    config1.plot(show_node_quantity=True, node_quantity_clim=[-np.pi, np.pi], title="exact solution with actan initial guess")
    config2.plot(show_node_quantity=True, node_quantity_clim=[-np.pi, np.pi], title="exact solution with london initial guess")

    def principle_value(x):
        return x - 2 * np.pi * np.round(x / (2 * np.pi))

    theta_difference = config1.get_theta() - config2.get_theta()
    print("||conf1.th - conf2.th||:", scipy.linalg.norm(theta_difference))
    print("||pv(conf1.th - conf2.th)||:", scipy.linalg.norm(principle_value(theta_difference)))

    plt.show()

    # conclusion:
    # - both solutions (i.e. both theta's) are the same modulus 2*pi.
    # - the 2*pi multiples mean both solutions live in a different phase-zone

    # In practice one does not need to manually specify initial conditions.
    # By default the london approximation is used as initial condition. One would just do:
    # problem = StaticProblem(sq_array, frustration=f, vortex_configuration=n)
    # config, status, info = problem.compute()





