
from pyjjasim import *

import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use("TkAgg")

"""
Time evolution example 4: Hysteresis

Ramp current up and down to see if the voltage is hysteretic.

Done with and without capacitance and inductance. 

"""

if __name__ == "__main__":

    # define array
    N = 20
    sq_array_no_scr_no_cap = SquareArray(N, N)

    sq_array_scr_no_cap = SquareArray(N, N)
    sq_array_scr_no_cap.set_inductance(2)

    sq_array_no_scr_cap = SquareArray(N, N)
    sq_array_no_scr_cap.set_capacitance(3)

    sq_array_scr_cap = SquareArray(N, N)
    sq_array_scr_cap.set_inductance(2)
    sq_array_scr_cap.set_capacitance(3)


    # define problem
    T = 0.05
    dt = 0.05
    Nt = 30000
    t_gap = 200
    ts = np.arange(0, 2 * Nt, t_gap)
    Imin = 0.4
    Imax = 1.6
    Ih = sq_array_no_scr_no_cap.current_base(angle=0)
    It = np.append(np.linspace(Imin, Imax, Nt), np.linspace(Imax, Imin, Nt))
    Is = Ih[:, None, None] * It
    prob_no_scr_no_cap = TimeEvolutionProblem(sq_array_no_scr_no_cap, time_step=dt, time_step_count=2*Nt, current_sources=Is, temperature=T, store_time_steps=ts)
    prob_scr_no_cap = TimeEvolutionProblem(sq_array_scr_no_cap, time_step=dt, time_step_count=2*Nt, current_sources=Is, temperature=T, store_time_steps=ts)
    prob_no_scr_cap = TimeEvolutionProblem(sq_array_no_scr_cap, time_step=dt, time_step_count=2*Nt, current_sources=Is, temperature=T, store_time_steps=ts)
    prob_scr_cap = TimeEvolutionProblem(sq_array_scr_cap, time_step=dt, time_step_count=2*Nt, current_sources=Is, temperature=T, store_time_steps=ts)

    # do time simulation
    out_no_scr_no_cap = prob_no_scr_no_cap.compute()
    out_scr_no_cap = prob_scr_no_cap.compute()
    out_no_scr_cap = prob_no_scr_cap.compute()
    out_scr_cap = prob_scr_cap.compute()

    def get_V(th):
        return np.append([0], np.mean(np.diff(th[Ih!=0, 0, :], axis=-1) / (dt * t_gap), axis=0))


    # compute array voltage
    V_no_scr_no_cap = get_V(out_no_scr_no_cap.get_theta())
    V_scr_no_cap = get_V(out_scr_no_cap.get_theta())
    V_no_scr_cap = get_V(out_no_scr_cap.get_theta())
    V_scr_cap = get_V(out_scr_cap.get_theta())

    # plot array voltage
    plt.plot(It[ts], V_no_scr_no_cap, label='betaL=0, betaC=0')
    plt.plot(It[ts], V_scr_no_cap, label='betaL=2, betaC=0')
    plt.plot(It[ts], V_no_scr_cap, label='betaL=0, betaC=2')
    plt.plot(It[ts], V_scr_cap, label='betaL=2, betaC=2')
    plt.xlabel("current")
    plt.ylabel("voltage")
    plt.title("Hysteresis of IV curve with or without screening and capacitance")
    plt.legend()
    plt.show()
