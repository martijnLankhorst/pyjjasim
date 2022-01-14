import time

import scipy.sparse
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from pyjjasim import *

N = 30


a = EmbeddedSquareGraph(N, N)
# print(time.perf_counter() - tic)
# tic = time.perf_counter()
# b, map = a.extended_dual_graph()
# print(time.perf_counter() - tic)
# print(map)
tic = time.perf_counter()
fig, ax = a.plot(figsize=[20, 10], show_face_ids=True, cycles="l_cycles")
print(time.perf_counter() - tic)
# b.plot(ax=ax, show_face_ids=True)

plt.show()

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