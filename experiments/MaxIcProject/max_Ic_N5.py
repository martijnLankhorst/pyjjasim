import itertools
import time
from multiprocessing import Pool

import numpy as np

from experiments.MaxIcProject.generate_ensembles import load, plot_max, compute, plot_all
from josephson_circuit import SquareArray
from static_problem import StaticProblem

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def ensemble(N):
    P = (N-1) ** 2
    x = np.arange(P)
    p = sum([list(itertools.combinations(x, i)) for i in range(1 + (P//2))], [])
    L = np.array([len(x) for x in p], dtype=int)
    f = L / P
    n = np.zeros((P, len(L)), dtype=int)
    n[np.concatenate(p).astype(int), np.repeat(np.arange(len(L)), L)] = 1
    return f, n


def do(i):
    N = 5
    A = 121
    print(i)
    f, n = ensemble(N)
    array = SquareArray(N, N)
    prob = StaticProblem(array, vortex_configuration=n[:, i], current_sources=array.horizontal_junctions())
    f_out, Is_out, _, _ = prob.compute_stable_region(angles=np.linspace(0, np.pi, A))
    return f_out, Is_out

# if __name__ == "__main__":
#     N = 5
#     A = 121
#     P = (N-1) ** 2
#     f, n = ensemble(N)
#     S = len(f)
#     print(S)
#     title = 'sqN5_complete_ensemble__num_angles121_A.npy'
#
#     mode = "compute"
#
#     if mode == "compute":
#         with Pool(processes=16) as pool:
#             out = pool.map(do, np.arange(S))
#         mask = np.zeros(S, dtype=bool)
#         f_out_store = np.zeros((A, S), dtype=float)
#         I_out_store = np.zeros((A, S), dtype=float)
#         for i, (fp, o) in enumerate(zip(f, out)):
#             f_out = o[0]
#             Is_out = o[1]
#             mask[i] = f_out is not None
#             if mask[i]:
#                 f_out_store[:, i] = f_out
#                 I_out_store[:, i] = Is_out
#                 plt.plot(f_out, Is_out)
#                 if fp != 0.5:
#                     plt.plot(1 - f_out, Is_out)
#         with open(title, 'wb') as ffile:
#             np.save(ffile, f[mask])
#             np.save(ffile, n.astype(np.int8)[:, mask])
#             np.save(ffile, f_out_store[:, mask])
#             np.save(ffile, I_out_store[:, mask])
#         plt.show()
#
#     if mode == "import":
#         with open(title, 'rb') as ffile:
#             f_ens = np.load(ffile)
#             n_ens = np.load(ffile)
#             f_out = np.load(ffile)
#             I_out = np.load(ffile)
#         for (fp, f, I) in zip(f_ens, f_out.T, I_out.T):
#             plt.plot(f, I)
#             if fp != 0.5:
#                 plt.plot(1 - f, I)
#         plt.show()


if __name__ == "__main__":
    N_list = [3, 4, 5,]
    N = 5
    beta_L = 0
    beta_L_list = [0]
    A = 121
    run = "D"

    tic = time.perf_counter()
    title = f'sqN{N}_complete_ensemble__num_angles{A}_betaL_{beta_L}_{run}.npy'
    # compute(N, A,  beta_L, filename=title, processes=8)
    print(time.perf_counter() - tic)

    f, n, f_out, I_out = load(title)
    print(f)
    print(len(f))
    plot_all(N, f, f_out, I_out)

    f_probe = np.linspace(0, 0.5, 501)
    plot_max(N, f_probe, f_out, I_out, color=np.random.rand(3))
    plt.show()