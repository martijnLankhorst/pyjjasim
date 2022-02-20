import time

from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

"""
TIME EVOLUTION EXAMPLE 1: Vortex trains

"""

if __name__ == "__main__":

    N = 50

    sq_array = SquareArray(N, N)
    # sq_array.set_inductance_factors(1)
    # x = np.ones(sq_array._Nj())
    # x[np.random.rand(sq_array._Nj()) < 0.3] = 0
    # x = np.random.rand(sq_array._Nj())
    # print(len(x))
    # sq_array.set_inductance_factors(x)
    # print(sq_array._get_mixed_inductance_mask())
    # sq_array.set_capacitance_factors(1)

    f = 0.05
    W = 1
    Is = 0.6 * sq_array.current_base(angle=0)[:, None, None] * np.ones((1, W, 1))
    T = 0.01
    dt = 0.02
    Nt = 10000
    store_stepsize = 25
    ts = np.arange(0, Nt, store_stepsize)

    problem = TimeEvolutionProblem(sq_array, time_step=dt,
                                   time_step_count=Nt, current_sources=Is,
                                   frustration=f, temperature=T,
                                   store_time_steps=ts)
    tic = time.perf_counter()
    out = problem.compute()
    print(time.perf_counter() - tic)
    # out.animate(junction_quantity="Isup", face_quantity="flux", face_quantity_clim=[-1, 2],
    #             figsize=[10, 10])
    # out.animate()
    plt.show()
