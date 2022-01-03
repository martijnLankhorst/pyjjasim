


import time
import numpy as np
import scipy
import scipy.sparse.linalg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from josephson_circuit import HoneycombArray, SquareArray


array = SquareArray(50, 50)
Nn, Nj, Nf = array._Nnr(), array._Nj(), array._Nf()
array.set_inductance_factors(1)

M = array._Mr().A
A = array.get_cycle_matrix()

m = scipy.sparse.vstack([M, A ]).tocsc()

W = 1000
x = np.random.rand(Nn, W)
y = np.random.rand(Nj, W)

r = scipy.sparse.diags(np.random.rand(Nj), 0)

Mf = scipy.sparse.linalg.factorized(M @ r @ M.T)
mf = scipy.sparse.linalg.factorized(m @ m.T)

tic = time.perf_counter()
M.T @ Mf(M @ y)
print(time.perf_counter() - tic)

tic = time.perf_counter()
mf(y)
print(time.perf_counter() - tic)


tic = time.perf_counter()
fluctuations = np.random.randn(Nj, W)
fluctuations = np.random.randn(Nj, W)
fluctuations = np.random.randn(Nj, W)
print(time.perf_counter() - tic)

tic = time.perf_counter()
fluctuations = np.random.randn(Nj, W)
perm = np.random.permutation(Nj)
fluctuations = fluctuations[perm, :]
perm = np.random.permutation(Nj)
fluctuations = fluctuations[perm, :]
print(time.perf_counter() - tic)