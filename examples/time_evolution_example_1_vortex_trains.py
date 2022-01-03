import time

import numpy as np

from josephson_circuit import *
from time_evolution import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

"""
TIME EVOLUTION EXAMPLE 1: Vortex trains

"""

N = 20

sq_array = SquareArray(N, N)
x = np.ones(sq_array._Nj())
x[np.random.rand(sq_array._Nj()) < 0.3] = 0
x = np.random.rand(sq_array._Nj())
print(len(x))
sq_array.set_inductance_factors(x)
# print(sq_array._get_mixed_inductance_mask())
# sq_array.set_capacitance_factors(1)

f = 0.05
Is = 0.6 * sq_array.horizontal_junctions()
T = 0
dt = 0.02
Nt = 10000
store_stepsize = 25
ts = np.arange(0, Nt, store_stepsize)
a = np.random.rand(3,)
b = np.atleast_2d(a)
print(b.shape )
problem = TimeEvolutionProblem(sq_array, time_step=dt,
                               time_step_count=Nt, current_sources=Is,
                               frustration=f, temperature=T,
                               store_time_steps=ts)
tic = time.perf_counter()
out = problem.compute()
print(time.perf_counter() - tic)
out.animate(arrow_quantity="Isup", face_quantity="flux", show_face_quantity=True, face_quantity_clim=[-1, 2],
            figsize=[10, 10])
plt.show()
