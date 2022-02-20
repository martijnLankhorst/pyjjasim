
from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

"""
TIME EVOLUTION EXAMPLE 2: IV curve

"""

if __name__ == "__main__":

    sq_array = SquareArray(20, 20)
    hor_junc = sq_array.current_base(angle=0)
    f = 0.05
    I = np.linspace(0, 2, 21)
    Is = hor_junc[:, None] * I
    T = 0.05
    dt = 0.05
    Nt = 10000

    ts = [Nt//3, Nt-1]
    problem = TimeEvolutionProblem(sq_array, time_step=dt, time_step_count=Nt, current_sources=Is,
                             frustration=f, temperature=T, store_time_steps=ts)

    out = problem.compute()

    th = out.get_theta()
    V = (th[:, :, 1] - th[:, :, 0]) / (dt * (ts[1]-ts[0]))
    V_mean = np.mean(V[hor_junc!=0, :], axis=0)
    plt.plot(20 * I, 20 * V_mean, label="square array")
    plt.xlabel("net array current")
    plt.ylabel("net array voltage")


    hc_array = HoneycombArray(14, 20)

    x, y = hc_array.get_node_coordinates()
    lower = x==0
    higher = np.isclose(x, np.max(x))
    Is_node = lower.astype(int) - higher.astype(int)
    Is_base = node_to_junction_current(hc_array, Is_node)

    f = 0.02
    I = np.linspace(0, 2.5, 31)
    Is = Is_base[:, None] * I
    T = 0.05
    dt = 0.05
    Nt = 6000
    ts = [Nt//3, Nt-1]

    problem = TimeEvolutionProblem(hc_array, time_step=dt, time_step_count=Nt, current_sources=Is,
                             frustration=f, temperature=T, store_time_steps=ts)
    out = problem.compute()

    phi = out.get_phi()
    U = (phi[:, :, 1] - phi[:, :, 0]) / (dt * (ts[1]-ts[0]))
    V_mean = np.mean(U[higher, :], axis=0) - np.mean(U[lower, :], axis=0)
    plt.plot(20 * I, V_mean, label="honeycomb array (armchair orientation)")
    plt.xlabel("net array current")
    plt.ylabel("net array voltage")
    plt.title("IV curve for square and honeycomb array")
    plt.legend()
    plt.show()

