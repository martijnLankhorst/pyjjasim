import scipy.sparse
import numpy as np

from pyJJAsim import SquareArray

# a = SquareArray(3, 3)
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

N = 5
A = scipy.sparse.rand(N, N, 1)
A = A.tocsc()
print(A[[True, False, False, True, True], :][:, [False, False, False, True, True]])
