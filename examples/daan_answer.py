import numpy as np

from josephson_circuit import *
from static_problem import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

"""
EXAMPLE 3: Vortex with inductance: Flux quantization

"""

# Square array with screening
N = 12
array = SquareArray(N, N)
Llist = [0.01, 0.1, 1, 10]
c = (N-1)/2
f = 0  # frustration
n = np.zeros(array.face_count(), dtype=int)  # target vortex configuration
centre_face_idx = array.locate_faces(c, c)  # locating face idx at coordinate x=(N-1)/2, y=(N-1)/2
n[centre_face_idx] = 1

for i, L in enumerate(Llist):
    array.set_inductance_factors(L)
    out, _, _ = StaticProblem(array, frustration=f, vortex_configuration=n).compute()
    flux = out.get_flux()
    flux_center = np.round(flux[array.locate_faces(c, c)], 4)
    flux_sum = np.round(np.sum(flux), 4)
    out.plot(show_face_quantity=True, face_quantity="flux", face_quantity_logarithmic_colors=True,
             face_quantity_clim=[1E-10, 1], title=f"beta_L={L}, flux centre {flux_center}, sum flux {flux_sum}", arrow_color=[1, 1, 1])


N = 30
array = SquareArray(N, N)
Llist = [0.01, 0.1, 1, 10]
c = (N-1)/2
f = 0  # frustration
n = np.zeros(array.face_count(), dtype=int)  # target vortex configuration
centre_face_idx = array.locate_faces(c, c)  # locating face idx at coordinate x=(N-1)/2, y=(N-1)/2
n[centre_face_idx] = 1


Llist = 10 ** np.linspace(-2, 2, 21)
Phi = np.zeros(len(Llist))
for i, L in enumerate(Llist):
    array.set_inductance_factors(L)
    out, _, _ = StaticProblem(array, vortex_configuration=n).compute()
    Phi[i] = np.sum(out.get_flux())
plt.subplots()
plt.semilogx(Llist, Phi, marker="o", label="total flux")
plt.semilogx(Llist, Llist / (1 + Llist), label="beta_L / (1 + beta_L)")
plt.xlabel("beta_L")
plt.ylabel("total flux")

plt.legend()
plt.show()
