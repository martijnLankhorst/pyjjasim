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
N = 14
array = SquareArray(N, N)

Llist = [0.001, 0.1, 1, 10]

f = 1 / (N - 1) ** 2  # frustration
n = np.zeros(array.face_count(), dtype=int)  # target vortex configuration
centre_face_idx = array.locate_faces((N - 1) / 2, (N - 1) / 2)  # locating face idx at coordinate x=(N-1)/2, y=(N-1)/2
n[centre_face_idx] = 1
for i, L in enumerate(Llist):
    array.set_inductance_factors(L)
    out, _, _ = StaticProblem(array, frustration=f, vortex_configuration=n).compute()
    out.plot(show_face_quantity=True, face_quantity="flux", face_quantity_logarithmic_colors=True,
             face_quantity_clim=[1E-3, 1], title=f"beta_L={L}", arrow_color=[1, 1, 1])


plt.show()



