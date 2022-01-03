import time

import numpy as np

from experiments.MaxIcProject.generate_ensembles import compute, load, plot_all, plot_max

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

if __name__ == "__main__":
    N_list = [3, 4, 5,]
    beta_L_list = [0]
    A = 121
    run = "C"
    run_List = ["B", "B", "C"]
    for N, run in zip(N_list, run_List):
        for beta_L in beta_L_list:
            tic = time.perf_counter()
            title = f'sqN{N}_complete_ensemble__num_angles{A}_betaL_{beta_L}_{run}.npy'
            # compute(N, A,  beta_L, filename=title, processes=8)
            print(time.perf_counter() - tic)

            f, n, f_out, I_out = load(title)
            print(f)
            # plot_all(N, f, f_out, I_out)

            f_probe = np.linspace(0, 0.5, 501)
            plot_max(N, f_probe, f_out, I_out, color=np.random.rand(3))
    plt.show()