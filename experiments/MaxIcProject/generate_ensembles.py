import itertools
from multiprocessing import Pool

from josephson_circuit import SquareArray
from static_problem import StaticProblem
from vortex_configuration_ensemble import generate_vortex_ensemble_with_high_current_sources
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

# if __name__ == "__main__":
#     N = 10
#     array = SquareArray(N, N)
#     Is_base = array.horizontal_junctions()
#     f_list = np.linspace(0, 0.5, 256)
#     title = 'high_I_vortex_ensemble_sqN10_A.npy'
#     fs, ns = generate_vortex_ensemble_with_high_current_sources(array, Is_base, f_list,
#                                                                 interval_count=2000, title=title)

def ensemble(N):
    P = (N-1) ** 2
    x = np.arange(P)
    p = sum([list(itertools.combinations(x, i)) for i in range(1 + (P//2))], [])
    L = np.array([len(x) for x in p], dtype=int)
    f = L / P
    n = np.zeros((P, len(L)), dtype=int)
    n[np.concatenate(p).astype(int), np.repeat(np.arange(len(L)), L)] = 1
    n = n[:, np.lexsort(n)]

    # remove mirrored
    mirrored = n.reshape(N-1, N-1, len(L))[:, ::-1, :].reshape(P, len(L))
    s = np.lexsort(np.append(n, mirrored, axis=1))
    idx = np.flatnonzero(s >= len(L))
    v = s[s >= len(L)]
    old_idx = idx - np.arange(len(L)) - 1
    paired_idx = v - len(L)
    print(old_idx)
    print(paired_idx)
    keep_ids = old_idx <= paired_idx
    return f[keep_ids], n[:, keep_ids]

def compute_single(N, A, f, n, beta_L):
    print(f"computing single at f = {f}")
    array = SquareArray(N, N)
    array.set_inductance_factors(beta_L)
    prob = StaticProblem(array, vortex_configuration=n, current_sources=array.horizontal_junctions())
    f_out, Is_out, _, _ = prob.compute_stable_region(angles=np.linspace(0, np.pi, A))
    return f_out, Is_out


def compute(N, A,  beta_L, filename=None, processes=16):
    f, n = ensemble(N)
    S = len(f)
    with Pool(processes=processes) as pool:
        out = pool.starmap(compute_single, [(N, A, fv, nv, beta_L) for (fv, nv) in zip(f, n.T)] )
    mask = np.zeros(S, dtype=bool)
    f_out_store = np.zeros((A, S), dtype=float)
    I_out_store = np.zeros((A, S), dtype=float)
    for i, (fp, o) in enumerate(zip(f, out)):
        f_out = o[0]
        Is_out = o[1]
        mask[i] = f_out is not None
        if mask[i]:
            f_out_store[:, i] = f_out
            I_out_store[:, i] = Is_out
    f = f[mask]
    n = n.astype(np.int8)[:, mask]
    f_out_store = f_out_store[:, mask]
    I_out_store = I_out_store[:, mask]
    if filename is not None:
        save(f, n, f_out_store, I_out_store, filename)
    return f, n, f_out_store, I_out_store

def save(f, n, f_out, I_out, filename):
    with open(filename, 'wb') as ffile:
        np.save(ffile, f)
        np.save(ffile, n)
        np.save(ffile, f_out)
        np.save(ffile, I_out)

def load(filename):
    with open(filename, 'rb') as ffile:
        f = np.load(ffile)
        n = np.load(ffile)
        f_out = np.load(ffile)
        I_out = np.load(ffile)
        return f, n, f_out, I_out

def extract_max_I(f_probe, f_out, I_out):
    I = []
    for f in f_probe:
        angle_nr, prob_nr = np.nonzero(((f_out[:-1, :] >= f) & (f_out[1:, :] < f)) | ((f_out[:-1, :] <= f) & (f_out[1:, :] > f)))
        f1, f2 = f_out[angle_nr, prob_nr], f_out[angle_nr+1, prob_nr]
        I1, I2 = I_out[angle_nr, prob_nr], I_out[angle_nr+1, prob_nr]
        I += [np.max(((f2 - f) * I1 + (f - f1) * I2) / (f2 - f1))]
    return np.array(I)

def plot_all(N, f, f_out, I_out):
    for (f_v, f_out_v, I_out_v) in zip(f, f_out.T, I_out.T):
        plt.plot(f_out_v, I_out_v / N)
        if f_v != 0.5:
            plt.plot(1 - f_out_v, I_out_v / N)

def plot_max(N, f_probe, f_out, I_out, color=None):
    I_max = extract_max_I(f_probe, f_out, I_out)
    if color is None:
        plt.plot(f_probe, I_max/N)
        plt.plot(1 - f_probe, I_max/N)
    else:
        plt.plot(f_probe, I_max/N, color=color)
        plt.plot(1 - f_probe, I_max/N, color=color)