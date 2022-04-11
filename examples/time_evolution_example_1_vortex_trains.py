from pyjjasim import *

import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use("TkAgg")

"""
TIME EVOLUTION EXAMPLE 1: Vortex trains

"""

if __name__ == "__main__":

    # define circuit
    N = 20
    sq_array = SquareArray(N, N)

    # define problem
    # Note: Is, f, T broadcast compatible with shape (face/junc_count,problem_count,time_step_count)
    f = 0.1
    problem_count = 1
    Is = 0.9 * sq_array.current_base(angle=0)[:, None, None] * np.ones((1, problem_count, 1))
    dt = 0.05
    Nt = 50000
    T = 0.05 * np.ones((sq_array.junction_count(), problem_count, 1))
    np.set_printoptions(linewidth=1000000)
    problem = TimeEvolutionProblem(sq_array, time_step=dt, time_step_count=Nt,
                                   current_sources=Is, frustration=f, temperature=T)

    # do transient analysis and visualize
    out = problem.compute()
    out.animate(junction_quantity="supercurrent")
    plt.show()

