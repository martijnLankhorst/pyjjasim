import time

import scipy.sparse
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pyjjasim.static_problem import lobpcg_test_negative_definite

matplotlib.use("TkAgg")
from pyjjasim import *

N = 30


# a = EmbeddedSquareGraph(N, N)
# # print(time.perf_counter() - tic)
# # tic = time.perf_counter()
# # b, map = a.extended_dual_graph()
# # print(time.perf_counter() - tic)
# # print(map)
# tic = time.perf_counter()
# fig, ax = a.plot(figsize=[20, 10], show_face_ids=True, cycles="l_cycles")
# print(time.perf_counter() - tic)
# # b.plot(ax=ax, show_face_ids=True)
#
#
#
#
# plt.show()

# N = a.junction_count()
# L1 = scipy.sparse.rand(N, N, 0.5)
# L1 = L1 + L1.T + 5 * scipy.sparse.eye(N, N)
# np.set_printoptions(linewidth=1000000)
# a.set_inductance_factors(L1)
# L2 = scipy.sparse.hstack([ scipy.sparse.rand(5, N, 0.5), 3 * scipy.sparse.eye(5)])
# L3 = scipy.sparse.rand(5, 5, 0.5)
# L3 = L3 + L3.T + 5 * scipy.sparse.eye(5)
# print(L2.shape)
# b = a.add_nodes_and_junctions([3, 3, 3], [0, 1, 2],
#                               [2, 5, 8, 9, 10],
#                               [9, 10,11,10, 11],
#                               inductance_factors=L3)
# print(np.round(b.inductance_factors.todense(), 3))
# a.plot()
# plt.show()
# N = 5
# A = scipy.sparse.rand(N, N, 1)
# A = A.tocsc()
# print(A[[True, False, False, True, True], :][:, [False, False, False, True, True]])
# print(a.graph.get_areas())


# def flipp(n1, n2):
#     flip = n1 > n2
#     nn1, nn2 =  np.where(flip, n2, n1), np.where(flip, n1, n2)
#     s = np.lexsort((nn2, nn1))
#     print("s",  s)
#     return  nn1[s], nn2[s], flip, s
#
# def flipi(N1, N2, flip, s):
#     inv_s = np.argsort(s)
#     print("inv s",  inv_s)
#     nn1, nn2 = N1[inv_s], N2[inv_s]
#     n1, n2 = np.where(flip, nn2, nn1), np.where(flip, nn1, nn2)
#     return  n1, n2
#
#
#
# n1 = np.array([2, 4, 6, 0, 1, 2])
# n2 = np.array([1, 3, 0, 4, 5, 3])
#
# print(n1)
# print(n2)
#
# N1, N2, flip, s = flipp(n1, n2)
#
# print(N1)
# print(N2)
#
#
# n1r, n2r = flipi(N1, N2, flip, s)
#
# print(n1r)
# print(n2r)

# fig, ax = plt.subplots()
# out = ax.plot([1, 2, 3], [1, 4, 9])
# o = out[0]
# o.set_xdata([3, 4, 5])
# print([method_name for method_name in dir(o)
#                   if callable(getattr(o, method_name))])
# plt.show()

# def nearest_anchor(x):
#     b = np.log(x) / np.log(10)
#     decade = np.floor(b)
#     B = b - decade
#     s = np.array([0, np.log(2) / np.log(10), np.log(5) / np.log(10), 1])
#     sub = np.argmin(np.abs(B - s))
#     if sub == 0:
#         return 10 ** decade
#     if sub == 1:
#         return 2 * 10 ** decade
#     if sub == 2:
#         return 5 * 10 ** decade
#     if sub == 3:
#         return 10 ** (decade + 1)
#
# print(nearest_anchor(14.8))
#
# a = np.zeros(3)
# print(np.atleast_2d(a).shape)

# ar = SquareArray(5, 5)
# M = ar.get_cycle_matrix()
# A = M @ M.T
# out = lobpcg_test_negative_definite(-A, preconditioner=None, accept_ratio=10, maxiter=200)
# print(out)
# print(len(out[1]))
# print(len(out[2]))


"""
EXAMPLE 4: Parameter optimization

Compute maximal frustration bounds and maximal current 
"""

if __name__ == "__main__":
    # define arrays
    array = SquareArray(12, 12)
    array.set_inductance_factors(1)
    Ih, Iv = 1.0 * array.horizontal_junctions(), 1.0 * array.vertical_junctions()

    # compute frustration bounds for zero vortex state
    prob = StaticProblem(array, current_sources=0)
    (smallest_f, largest_f), (s_config, l_config), (i1, i2) = prob.compute_frustration_bounds()
    s_config.plot(title=f"minimal frustration in zero vortex state (f={np.round(smallest_f, 4)})")
    print(i2)
    scheme=1
    # prec = stability_get_preconditioner(array, prob.get_current_phase_relation(), scheme=scheme)
    st = {"scheme": scheme, "algorithm": 0, "preconditioner": "auto"}
    prob = StaticProblem(array, current_sources=0)
    (smallest_f, largest_f), (s_config, l_config), (i1, i2) = prob.compute_frustration_bounds(stability_parameters=st)
    s_config.plot(title=f"minimal frustration in zero vortex state (f={np.round(smallest_f, 4)})")
    print(i2)


    # # compute maximal current
    # prob = StaticProblem(array, frustration=0, current_sources=Iv)
    # I_factor, net_I, max_I_config, info = prob.compute_maximal_current()
    # np.set_printoptions(linewidth=10000000)
    # print(f"largest current factor {I_factor} (corresponding to net current of  {net_I}) at which the zero-vortex state exists at zero frustration")
    # print(info)
    # max_I_config.plot(title=f"maximal current in zero vortex state (net_I={np.round(net_I, 4)})")
    #
    #
    # # compute maximal current
    # n = np.zeros(array.face_count())
    # n[array.locate_faces(5.5, 5.5)] = 1
    # prob = StaticProblem(array, frustration=0.05, current_sources=Iv, vortex_configuration=n)
    # I_factor, net_I, max_I_config, info = prob.compute_maximal_current(require_vortex_configuration_equals_target=False)
    # np.set_printoptions(linewidth=10000000)
    # print(f"largest current factor {I_factor} (corresponding to net current of  {net_I}) at which the zero-vortex state exists at zero frustration")
    # print(info)
    # max_I_config.plot(title=f"maximal current in zero vortex state (net_I={np.round(net_I, 4)})")

    #
    # # compute extermum in Is-f space using compute_stable_region()
    # prob = StaticProblem(array, frustration=0, current_sources=Iv)
    # f, net_I, _, _ = prob.compute_stable_region()
    # plt.subplots()
    # plt.plot(f, net_I)
    # plt.xlabel("frustration")
    # plt.ylabel("net I")
    # plt.title("Region in f-I parameter space where the zero-vortex state is stable")
    #
    # # compute direction dependent maximal current using compute_maximal_parameter()
    # prob = StaticProblem(array)
    # angles = np.linspace(0, 2*np.pi, 33)
    # I_factor = np.zeros(len(angles))
    # for i, angle in enumerate(angles):
    #     prob_func = lambda x: prob.new_problem(current_sources=x * (np.cos(angle) * Ih + np.sin(angle) * Iv))
    #     I_factor[i], _, _, _ = compute_maximal_parameter(prob_func)
    # plt.subplots()
    # plt.plot(np.cos(angles) * I_factor, np.sin(angles) * I_factor, marker="o")
    # plt.xlabel("horizontal current")
    # plt.ylabel("vertical current")
    # plt.title("angle dependent maximal current for zero-vortex state")
    # plt.show()
