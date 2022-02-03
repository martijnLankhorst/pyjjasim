from pyjjasim import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


"""
EXAMPLE 5: Lattices

Compute maximal current in a square, honeycomb and triangular lattice.
"""

if __name__ == "__main__":
    # define arrays
    square_array = SquareArray(12, 12)
    honeycomb_array = HoneycombArray(8, 12)
    triangular_array = TriangularArray(12, 8)

    # frustration factor
    f = 0.02

    # define physical problems
    prob_sq = StaticProblem(square_array, frustration=f, current_sources=square_array.current_base(angle=0))
    prob_hc = StaticProblem(honeycomb_array, frustration=f, current_sources=honeycomb_array.current_base(angle=0))
    prob_tr = StaticProblem(triangular_array, frustration=f, current_sources=triangular_array.current_base(angle=0))

    square_array.plot()
    honeycomb_array.plot()
    triangular_array.plot()
    # plt.show()

    # compute maximal current
    _, _, config_sq, _ = prob_sq.compute_maximal_current()
    _, _, config_hc, _ = prob_hc.compute_maximal_current()
    _, _, config_tr, _ = prob_tr.compute_maximal_current()

    # plot result
    config_sq.plot()
    config_hc.plot()
    config_tr.plot()
    plt.show()