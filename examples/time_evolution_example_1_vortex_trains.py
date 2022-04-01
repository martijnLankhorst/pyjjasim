import time

from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("TkAgg")

"""
TIME EVOLUTION EXAMPLE 1: Vortex trains

"""

if __name__ == "__main__":

    N = 20

    sq_array = SquareArray(N, N)
    sq_array.set_inductance_factors(10)
    # x = np.ones(sq_array._Nj())
    # x[np.random.rand(sq_array._Nj()) < 0.3] = 0
    # x = np.random.rand(sq_array._Nj())
    # print(len(x))
    # sq_array.set_inductance_factors(x)
    # print(sq_array._get_mixed_inductance_mask())
    sq_array.set_capacitance_factors(0)

    f = 0.1
    W = 1
    Is = 0.9 * sq_array.current_base(angle=0)[:, None, None] * np.ones((1, W, 1))
    dt = 0.05
    Nt = 50000
    T = 0.001 * np.ones((sq_array.junction_count(), W, 1))
    store_stepsize = 5
    ts = np.arange(0, Nt, store_stepsize)

    np.set_printoptions(linewidth=1000000)
    problem = TimeEvolutionProblem(sq_array, time_step=dt,
                                   time_step_count=Nt, current_sources=Is,
                                   frustration=f, temperature=T,
                                   store_time_steps=ts, stencil_width=3, store_voltage=True,
                                   store_theta=True, store_current=True)
    tic = time.perf_counter()
    out = problem.compute()
    print(out.theta.shape if out.theta is not None else None)
    print(out.current.shape if out.current is not None else None)
    print(out.voltage.shape if out.voltage is not None else None)
    print(time.perf_counter() - tic)
    # plt.plot(out.theta[:, 0, :].T)


    # out.animate(junction_quantity="Isup", face_quantity="flux", face_quantity_clim=[-1, 2],
    #             figsize=[10, 10])
    out.animate(junction_quantity="I", figsize=[10, 10], face_quantity="flux",
                face_quantity_clim=[-0.1, 1.1])
    plt.show()

