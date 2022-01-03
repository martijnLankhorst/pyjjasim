from josephson_circuit import *
from static_problem import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


"""
EXAMPLE 5: Lattices

Compute maximal current in a square, honeycomb and triangular lattice.
"""

# define arrays
square_array = SquareArray(12, 12)
honeycomb_array = HoneycombArray(8, 12)
triangular_array = TriangularArray(12, 8)

# frustration factor
f = 0.01

# define physical problems
prob_sq = StaticProblem(square_array, frustration=f, current_sources=square_array.horizontal_junctions())
prob_hc = StaticProblem(honeycomb_array, frustration=f, current_sources=honeycomb_array.horizontal_junctions())
prob_tr = StaticProblem(triangular_array, frustration=f, current_sources=triangular_array.horizontal_junctions())

# compute maximal current
_, _, config_sq, _ = prob_sq.compute_maximal_current()
_, _, config_hc, _ = prob_hc.compute_maximal_current()
_, _, config_tr, _ = prob_tr.compute_maximal_current()

# plot result
config_sq.plot()
config_hc.plot()
config_tr.plot()
plt.show()