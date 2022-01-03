import numpy as np

from josephson_circuit import *
from static_problem import *

import matplotlib
import matplotlib.pyplot as plt

from static_problem import node_to_junction_current

matplotlib.use("TkAgg")

"""
Static Example 7: Randomized square array

First a square array with varying critical current is investigated.

Secondly, a square array with random holes is investigated.

"""


# EXAMPLE A: RANDOM CRITICAL CURRENT

np.random.seed(1)

# define arrays
N = 12
array = SquareArray(N, N)

Ic = 1 + 0.3 * np.random.randn(array.junction_count())
Ic[Ic < 0.1] = 0.1
plt.hist(Ic)
plt.title("histogram of junction Ic")
plt.xlabel("critical current")
plt.ylabel("count")

array.set_critical_current_factors(Ic)

# define problem parameters
problem_no_scr = StaticProblem(array, current_sources=array.horizontal_junctions())
I_factor, net_I, config, _ = problem_no_scr.compute_maximal_current()

print(f"max net current: {net_I}. For uniform Ic, it would be 12.")
config.plot(title="maximum current with random Ic")


# EXAMPLE B: Arrays with holes
remove_node_count = 20
array.set_critical_current_factors(1)
array = array.remove_nodes(np.random.permutation(array.node_count())[:remove_node_count])

# create junction-current sources such that uniform current is injected
x, y = array.get_node_coordinates()
left_nodes = x==0
right_nodes = x==(N-1)
Is_node = np.zeros(array.node_count(), dtype=float)
Is_node[left_nodes] = np.sum(right_nodes)
Is_node[right_nodes] = -np.sum(left_nodes)
# assert sum is zero (current conservation):
print(f"this sum must be zero: {np.sum(Is_node)}")
Is = node_to_junction_current(array, Is_node)

problem_no_scr = StaticProblem(array, current_sources=Is)
I_factor, net_I, config, _ = problem_no_scr.compute_maximal_current()

print(f"max net current: {net_I}. Without holes, it would be 12.")
config.plot(title="maximum current with holes")
plt.show()
