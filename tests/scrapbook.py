import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
from pyjjasim import SquareArray

a = SquareArray(4, 4)
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