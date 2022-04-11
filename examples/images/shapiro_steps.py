import time
from multiprocessing import Pool

import scipy.interpolate

from pyjjasim import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import matplotlib.cm

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

"""
Giant shapiro steps

With a biassed AC current, the DC voltage has plateaus. This effect is called shapiro steps for single junctions,
and giant shapiro steps for arrays as the step height is proportional to the array size. 

Of course after the array voltage is normalized, the steps are not so 
giant, and the step heights will be Ifreq.
"""


def func(N, Nt, T, Ifreq, IDC, IAmp):
    sq_array = SquareArray(N, N)
    dt = 0.05
    ts = [Nt//3, Nt - 1]
    Is = lambda i: sq_array.current_base(angle=0)[:, None] * (IDC + IAmp * np.sin(Ifreq * i * dt))

    prob = TimeEvolutionProblem(sq_array, time_step=dt, time_step_count=Nt,
                                current_sources=Is, temperature=T, store_time_steps=ts)
    th = prob.compute().get_theta()[sq_array.current_base(angle=0) == 1, :, :]
    return np.mean((th[:, :, 1] - th[:, :, 0]) / (dt * (ts[1] - ts[0])), axis=0)



if __name__ == "__main__":

    fn = "shapiro_data_Ifreq_0p25_B.npy"

    # define array
    Nt = 3000
    N = 20
    T = 0.01
    Ifreq = 0.25
    NA = 101
    IAmpList = np.linspace(0, 5, NA)
    ND = 301
    IDC = np.linspace(0, 3, ND)

    # with Pool() as pool:
    #     out = pool.starmap(func, [(N, Nt, T, Ifreq, IDC, IAmp) for IAmp in IAmpList])
    # Vout = np.stack(out)
    # Vout = np.concatenate((-Vout[:, ::-1], Vout[:, 1:]), axis=1)
    # IDC = np.concatenate((-IDC[::-1], IDC[1:]), axis=0)
    # with open(fn, "wb") as ffile:
    #     np.save(ffile, IDC)
    #     np.save(ffile, IAmpList)
    #     np.save(ffile, Vout)

    with open(fn, "rb") as ffile:
        IDC = np.load(ffile)
        IAmpList = np.load(ffile)
        Vout = np.load(ffile)

    # fig, (a1, a2) = plt.subplots(nrows=1, ncols=2, figsize=[12, 6])
    # print(fig, (a1, a2))
    # a2.pcolor(IAmpList, IDC, Vout.T)
    # a2.set_xlabel("AC current")
    # a2.set_ylabel("DC current")

    X, Y = np.meshgrid(IDC, IAmpList)
    print(X.shape, Vout.shape)
    tic = time.perf_counter()
    F = scipy.interpolate.interp2d(IDC, IAmpList, Vout)
    print(time.perf_counter() - tic)
    fig2, (a12, a22) = plt.subplots(nrows=1, ncols=2, figsize=[12, 5.5])
    IDCp = np.linspace(-3, 3, 8 * ND)
    IACp = np.linspace(0, 5, 3 * NA)
    phandle = a22.pcolor(IACp, IDCp, F(IDCp,  IACp).T, cmap="inferno")
    a22.set_xlabel("amplitude")
    a22.set_ylabel("DC current")
    # fig2.colorbar()
    pcm = matplotlib.cm.get_cmap("inferno")
    a12.plot(IDC, Vout[1, :], linewidth=1.8, label="amplitude=0", color=pcm(0.15))
    a12.plot(IDC, Vout[11, :],linewidth=1.8, label="amplitude=0.5", color=pcm(0.4))
    a12.plot(IDC, Vout[21, :],linewidth=1.8, label="amplitude=1", color=pcm(0.6))
    a12.set_xlabel("DC current")
    a12.set_ylabel("Voltage")
    a12.legend()
    a12.text(1, 3.5, "Giant Shapiro steps with frequency=0.25")
    a22.text(5.1, 3.25, "voltage")
    a12.set_xlim([-3, 3])
    a12.set_ylim([-3, 3])

    # change all spines
    lw = 1.5
    for axis in ['top', 'bottom', 'left', 'right']:
        a12.spines[axis].set_linewidth(lw)
        a22.spines[axis].set_linewidth(lw)

    # increase tick width
    a12.tick_params(width=lw)
    a22.tick_params(width=lw)
    # plt.title("giant shapiro steps in square array")
    plt.colorbar(phandle)
    fig2.savefig('shapiro_steps.png', bbox_inches=fig2.get_tightbbox(fig2.canvas.get_renderer()))
    plt.show()

