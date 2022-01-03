import numpy as np

from Experiments.MaxIcProject.generate_ensembles import compute, load, plot_all, plot_max

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

if __name__ == "__main__":
    N_list = [4]
    beta_L_list = [0, 1, 5]
    A = 121
    run = "B"
    for N in N_list:
        for beta_L in beta_L_list:
            title = f'sqN{N}_complete_ensemble__num_angles{A}_betaL_{beta_L}_{run}.npy'
            # compute(N, A,  beta_L, filename=title)

            f, n, f_out, I_out = load(title)

            # plot_all(N, f, f_out, I_out)

            f_probe = np.linspace(0, 0.5, 501)
            plot_max(N, f_probe, f_out, I_out)
    plt.show()