# import scipy.sparse.linalg
#
# from pyjjasim import *
# ar = HoneycombArray(4, 4)
#
# A = ar.get_cycle_matrix()
# A = A @ A.T
#
# np.set_printoptions(linewidth=100000)
#
# f = scipy.sparse.linalg.splu(A, diag_pivot_thresh=0)
# Up = (f.L @ scipy.sparse.diags(f.U.diagonal())).T
# print(np.allclose((Up - f.U).data, 0))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from pyjjasim import *

if __name__ == "__main__":

    square_array = SquareArray(3, 4)
    honeycomb_array = HoneycombArray(3, 4)
    np.set_printoptions(linewidth=100000, threshold=100000000)

    plt.show()

    triangular_array = TriangularArray(4, 3)

    print(square_array)
    print(square_array.current_base(np.pi/4))

    print(honeycomb_array)
    print(honeycomb_array.current_base(0, type="junction"))
    print(np.sign(np.array([False, True]).astype(int)))
    # honeycomb_array.current_base(0, type="node")
    CircuitPlot(triangular_array, arrow_data=triangular_array.current_base(np.pi/4, type="junction")).make()
    plt.show()
    # frustration factor
    f = 0.01

    # define physical problems
    # # prob_sq = StaticProblem(square_array, frustration=f, current_sources=square_array.horizontal_junctions())
    # # prob_hc = StaticProblem(honeycomb_array, frustration=f, current_sources=honeycomb_array.horizontal_junctions())
    # # prob_tr = StaticProblem(triangular_array, frustration=f, current_sources=triangular_array.horizontal_junctions())
    #
    # # square_array.plot()
    # # honeycomb_array.plot()
    # # triangular_array.plot()
    # # plt.show()
    # #
    # # # compute maximal current
    # # _, _, config_sq, _ = prob_sq.compute_maximal_current()
    # # _, _, config_hc, _ = prob_hc.compute_maximal_current()
    # # _, _, config_tr, _ = prob_tr.compute_maximal_current()
    # #
    # # # plot result
    # # config_sq.plot()
    # # config_hc.plot()
    # # config_tr.plot()
    # # plt.show()