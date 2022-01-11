from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

"""
EXAMPLE 2: Energy barrier

Compute energy barrier for a vortex to move to a neighbouring site.
This is done by computing a "junction-vortex" state where the vortex is centered on
the junction in between the two sites (such a state is unstable). The energy
barrier is then the difference in the energy of both states.

The junction-vortex is achieved using the problem method .approximate_placed_vortices(x, y, n).
This results in an approximation that can be used as an initial guess.

"""

if __name__ == "__main__":

    N = 11

    # Square array no screening
    array1 = SquareArray(N, N - 1)

    # frustration factor
    f = 0.01

    # define physical problems
    prob = StaticProblem(array1, frustration=f)

    # compute initial guess for vortex centered at junction (coordinates x=(N-2)/2, y=(N-1)/2)
    init_config_junc_vortex = prob.approximate_placed_vortices(1, (N - 1) / 2, (N-2) / 2)
    init_config_face_vortex = prob.approximate_placed_vortices(1, (N) / 2, (N-2) / 2)

    # plot solutions
    init_config_junc_vortex.plot(title="arctan approximation of vortex placed at junction")
    init_config_face_vortex.plot(title="arctan approximation of vortex placed at face")


    # find solutions and check if they are stable and satisfy the equations.
    config_junc_vortex, status_jv, info_jv = prob.compute(initial_guess=init_config_junc_vortex)
    config_face_vortex, status_fv, info_fv = prob.compute(initial_guess=init_config_face_vortex)

    print("Junction vortex solution: ")
    config_junc_vortex.report()
    print("face vortex solution: ")
    config_face_vortex.report()

    # plot solutions
    E_junc_no_scr = np.sum(config_junc_vortex.get_Etot())
    E_face_no_scr = np.sum(config_face_vortex.get_Etot())
    config_junc_vortex.plot(title="exact solution of vortex at junction, E=" + str(E_junc_no_scr))
    config_face_vortex.plot(title="exact solution of vortex at face, E=" + str(E_face_no_scr))

    print("energy barrier: ", str(E_junc_no_scr - E_face_no_scr))

    # Note that for the junction-vortex the location is not displayed properly.

    plt.show()

