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
import time


import matplotlib
import numpy as np

from pyjjasim.josephson_circuit import SquarePeriodicArray
from pyjjasim.static_problem import london_approximation, static_compute

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from pyjjasim import *

if __name__ == "__main__":

    # square_array = SquareArray(3, 4)
    # honeycomb_array = HoneycombArray(3, 4)
    # np.set_printoptions(linewidth=100000, threshold=100000000)
    #
    # plt.show()
    #
    # triangular_array = TriangularArray(4, 3)
    #
    # print(square_array)
    # print(square_array.current_base(np.pi/4))
    #
    # print(honeycomb_array)
    # print(honeycomb_array.current_base(0, type="junction"))
    # print(np.sign(np.array([False, True]).astype(int)))
    # # honeycomb_array.current_base(0, type="node")
    # CircuitPlot(triangular_array, arrow_data=triangular_array.current_base(np.pi/4, type="junction")).make()
    # plt.show()
    # # frustration factor
    # f = 0.01

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

    # g = EmbeddedSquareGraph(4, 4, periodic=True)
    # g.plot(cycles="l_cycles", show_node_ids=True, show_face_ids=True, show_edge_ids=True)
    #
    # print(np.stack(g.get_edges()))
    # print(g.get_face_cycles(to_list=True))
    # print(g.get_l_cycles(to_list=True))
    # plt.show()

    # N = 100
    # g = SquareArray(N, N)
    # g.set_inductance_factors(5)
    # n = np.zeros(g.face_count())
    # n[g.locate_faces((N-1)/2, (N-1)/2)] = 1
    #
    # x = StaticProblem(g, vortex_configuration=n)
    # _, status,  info = x.compute(tol=1E-5)
    # print(status)
    # print(info)

    # A = g.get_cycle_matrix()
    # v1 = np.random.rand(A.shape[1]) + 0.5
    # v2 = np.random.rand(A.shape[1])
    # b1 = np.random.rand(A.shape[0])
    # AA1 = A @ scipy.sparse.diags(v1) @ A.T
    # AA2 = A @ scipy.sparse.diags(v2) @ A.T
    #
    # tic = time.perf_counter()
    # x1 = scipy.sparse.linalg.splu(AA1)
    # print(time.perf_counter() - tic)
    #
    # tic = time.perf_counter()
    # x1 = scipy.sparse.linalg.spsolve(AA1, b1)
    # print(time.perf_counter() - tic)
    #
    # tic = time.perf_counter()
    # import pyamg
    # print(time.perf_counter() - tic)

    # # A = pyamg.gallery.poisson((500, 500), format='csr')  # 2D Poisson problem on 500x500 grid
    # tic = time.perf_counter()
    # w = 0.5
    # ml = pyamg.ruge_stuben_solver(AA1, strength=('classical', {'theta': 0.1})) # construct the multigrid hierarchy
    # print(ml)
    # print(time.perf_counter() - tic)
    # tic = time.perf_counter()
    # x = ml.solve(b1, tol=1e-10)  # solve Ax=b to a tolerance of 1e-10
    # print(time.perf_counter() - tic)
    # print("residual: ", np.linalg.norm(b1 - AA1 * x))  # compute norm of residual vector


    # N = 3
    # g = SquareArray(N, N)
    # A = g.get_cycle_matrix() @ g.get_cycle_matrix().T
    # Af = scipy.sparse.linalg.factorized(A)
    # Nf = g.face_count()
    # b1 = np.random.rand(Nf)
    # b2 = b1[:, None]
    # b3 = np.random.rand(Nf, 3)
    # print((A @ b1).shape)
    # print((A @ b2).shape)
    # print((A @ b3).shape)
    # print((scipy.sparse.linalg.spsolve(A, b1)).shape)
    # print((scipy.sparse.linalg.spsolve(A, b2)).shape)
    # print((scipy.sparse.linalg.spsolve(A, b3)).shape)
    # print((Af(b1)).shape)
    # print((Af(b2)).shape)
    # print((Af(b3)).shape)
    #
    # print((np.random.rand(4, 4) @ b1).shape)
    # print((np.random.rand(4, 4) @ b2).shape)
    # print((np.random.rand(4, 4) @ b3).shape)
    #
    # def broadcast(x, shape):
    #     x_shape = np.array(x).shape
    #     x = x.reshape(x_shape + (1,) * (len(shape) - len(x_shape)))
    #     return np.broadcast_to(x, shape)
    #
    # print(broadcast(np.zeros((4,)), (4, 2, 5)).shape)
    # print(np.zeros((3, 1)).reshape(3, -1).shape)
    # print(np.zeros((3,)).reshape(3, -1).shape)
    # print(np.zeros((3, 3)).reshape(3, -1).shape)
    #
    # x =np.zeros((3, 1))
    # print(x.shape)
    # print(x[:1, ...].shape)
    # print(x[-1, ...].shape)
    # print(x[-1:, ...].shape)
    # print(x[2, ...].shape)
    # print(x[2:3, ...].shape)


    # N = 300
    # g = SquareArray(N, N)
    # A = g.get_cycle_matrix() @ g.get_cycle_matrix().T
    #
    # tic = time.perf_counter()
    # d = A.diagonal()
    # print(time.perf_counter() - tic)
    #
    #
    # tic = time.perf_counter()
    # Af = scipy.sparse.linalg.factorized(A)
    # print(time.perf_counter() - tic)
    #
    # tic = time.perf_counter()
    # Af(np.random.rand(A.shape[0]))
    # print(time.perf_counter() - tic)

    # N = 100
    # sq_array = SquareArray(N, N)
    # sq_array.set_inductance_factors(5)
    # f = 0
    # n = np.zeros(sq_array.face_count())
    # n[sq_array.locate_faces((N-1)/2, (N-1)/2)] = 1
    # problem = StaticProblem(sq_array, frustration=f, vortex_configuration=n, current_sources=sq_array.current_base(angle=0))
    #
    # config0, _, info0 = problem.compute(algorithm=1)
    #
    # # config0, _, info0 = problem.compute(algorithm=0)
    # # config1, _, info1 = problem.compute(algorithm=1)
    # # config2, _, info2 = problem.compute(algorithm=1, use_pyamg=True)
    # #
    # # print(info0)
    # # print(info1)
    # # print(info2)
    #
    # _, _, _, info0 = problem.compute_maximal_current(compute_parameters={"algorithm": 0, "use_pyamg": False})
    # _, _, _, info1 = problem.compute_maximal_current(compute_parameters={"algorithm": 1, "use_pyamg": False})
    # _, _, _, info2 = problem.compute_maximal_current(compute_parameters={"algorithm": 1, "use_pyamg": True})
    #
    #
    # print(info0)
    # print(info1)
    # print(info2)

    # import scipy.fftpack
    # import pyamg
    # import pyamg.util.linalg
    #
    # Nx = 50
    # Ny = 50
    # W = 4
    # g = SquareArray(Nx, Ny)
    # A = g.get_cycle_matrix()
    # b = np.random.rand((Nx - 1) * (Ny - 1))
    # S = scipy.sparse.diags(np.random.rand(A.shape[1]) - 0.1)
    # AA = A @ S @ A.T
    # ml = pyamg.ruge_stuben_solver(AA, strength=('classical', {'theta': 0.1}))

    # class DivergenceError(Exception):
    #     pass
    #
    # class PyamgCallback:
    #
    #     def __init__(self, A, b):
    #         self.A = A
    #         self.b = b
    #         self.iter = 0
    #         self.resid = np.zeros((0,))
    #         self.x = None
    #         self.has_diverged = False
    #
    #     def cb(self, x):
    #         self.x = x
    #         r = pyamg.util.linalg.residual_norm(self.A, x, self.b)
    #         self.resid = np.append(self.resid, [np.abs(r)])
    #         self.iter += 1
    #         if np.any(np.isnan(x)) or np.any(np.isinf(x)):
    #             self._diverge()
    #         if self.iter > 1:
    #             if self.resid[-1] > self.resid[-2]:
    #                 self._diverge()
    #
    #     def _diverge(self):
    #         self.has_diverged = True
    #         raise DivergenceError("diverged")
    #
    # pyamg_cb = PyamgCallback(AA, b)
    # try:
    #     mg_out = ml.solve(b, tol=1e-8, maxiter=50, callback=pyamg_cb.cb)
    # except DivergenceError:
    #     mg_out = pyamg_cb.x
    # print(pyamg_cb.has_diverged)
    # print(pyamg_cb.resid)
    # print(mg_out)
    #
    #


    # AA = A @ A.T
    # b = np.random.rand((Nx-1) * (Ny-1),)
    #
    # g.Asq_solve(b, "fast")
    # tic = time.perf_counter()
    # for i in range(100):
    #     g.Asq_solve(b, "fast")
    # print(time.perf_counter() - tic)
    #
    # g._Asq_factorized = None
    # g.Asq_solve(b, "slow")
    # tic = time.perf_counter()
    # for i in range(100):
    #     g.Asq_solve(b, "slow")
    # print(time.perf_counter() - tic)

    # Af = scipy.sparse.linalg.factorized(AA)
    #
    # tic = time.perf_counter()
    # Af(b)
    # print(time.perf_counter() - tic)
    #
    # tic = time.perf_counter()
    # x = scipy.fftpack.ifft2(scipy.fftpack.fft2(b.reshape((Ny-1, Nx-1))) * 2)
    # print(time.perf_counter() - tic)

    # print(np.stack(g.get_face_centroids()))
    # print(np.arange(6).reshape(3, 2).ravel())
    # b = np.random.rand((Nx-1) * (Ny-1))
    # print(g.Asq_solve(b))
    #
    # print(np.arange(24).reshape(4, 2, 3))
    # b = np.arange(24).reshape(4, 6).T
    # print(b)
    # s = b.T.reshape(W, Ny-1, Nx-1)
    # print(s.reshape(W, 6).T)
    np.set_printoptions(linewidth=1000, threshold=10000)
    tic = time.perf_counter()
    g1 = EmbeddedTriangularGraph(3, 4)
    M1 = g1.cut_space_matrix()
    # print(np.stack((np.arange(g1.edge_count()), g1.edge_flip, *g1.get_edges())))
    # print(np.stack(M1.nonzero()))
    #
    # print(g1.get_edge_ids([14, 15, 5], [5, 3, 14], return_direction=True))
    # a = np.array([2, 2, 3, 1], dtype=int)
    # b = np.array([3, 1, 1, 1], dtype=int)
    # d1 = np.random.rand(sum(a))
    # d2 = np.random.rand(sum(b))
    #
    # out = np.zeros(sum(a) + sum(b))
    # out[VarRowArray(a).merge(VarRowArray(b))] = np.append(d1, d2)
    # print(d1)
    # print(d2)
    # print(out)

    #
    # print(g.get_edge_ids([2, 6, 1, 12], [1, 3, 2, 2]))
    g = EmbeddedSquareGraph(3, 4)
    G = EmbeddedPeriodicSquareGraph(3, 4)

    n, e, d, l = G._get_face_cycles()

    A = G.face_cycle_matrix()
    M = G.cut_space_matrix()
    print(A.todense())
    print(M.todense())

    print("!", M @ A.T)

    print(M.shape)
    print(A.shape)
    m = scipy.sparse.vstack((M[:-1, :], A))
    print(np.linalg.det(m.todense()))
    m = scipy.sparse.vstack((M, A[:-1, :]))
    print(np.linalg.det(m.todense()))
    # g.plot(show_node_ids=True, show_edge_ids=True)
    # plt.show()


    N = 10
    a = SquarePeriodicArray(N, N)

    p = StaticProblem(a)
    n = np.zeros(a.face_count())
    n[0] = 1
    f = np.ones(a.face_count())/(N**2)
    f[-2:] = 0
    A = a.graph.face_cycle_matrix()
    Asq = A @ A.T
    th0 = - 2 * np.pi * A.T @ scipy.sparse.linalg.spsolve(Asq, f - n)
    print(th0)

    x, y = a.graph.coo()
    n1, n2 = a.graph.get_edges()
    Ih = (y[n1] == y[n2]).astype(int)
    Iv = (x[n1] == x[n2]).astype(int)

    th, stat, info = static_compute(a, th0, 0.20 * Ih, f, n, n)

    print(th)
    print(stat)
    print(info)
    th0[a.graph.g.edge_count():] = 0
    th[a.graph.g.edge_count():] = 0

    CircuitPlot(a, arrow_data=np.sin(th)).make()
    print(np.sum(Ih * np.sin(th))/np.sum(Ih))
    print(np.sum(Iv * np.sin(th))/np.sum(Iv))
    plt.show()
    # print(np.stack((*G.get_edges(), np.arange(24))))
    # print(G.get_edge_ids([2, 6, 1, 14, 14], [1, 3, 2, 11, 23]))
    # g.plot(show_node_ids=True, show_edge_ids=True)
    # plt.show()
    # g.face_count()
    # print(time.perf_counter() - tic)
    #
    # P = 100
    # p1 = np.random.randint(0, g.node_count(), P)
    # p2 = np.random.randint(0, g.node_count(), P)
    #
    # tic = time.perf_counter()
    # g.shortest_path(p1, p2, to_list=False)
    # print(time.perf_counter() - tic)
    #
    # p1 = np.array([2, 6, 10, 4, 2])
    # p2 = np.array([3, 11, 3, 8, 1])
    # P = len(p1)
    # up1, idx = np.unique(p1, return_inverse=True)
    # out, pred = scipy.sparse.csgraph.shortest_path(M, return_predecessors=True, indices=up1)
    # K = pred.shape[1]
    # pred[pred < 0] = K
    # pred = np.concatenate((pred, K * np.ones((pred.shape[0], 1), dtype=int)), axis=1)
    # print(pred)
    # # print(pred)
    # p = p2
    # out = np.zeros((5, P), dtype=int)
    # k = 0
    # while not np.all(p == K):
    #     if k >= out.shape[0]:
    #         out = np.stack((out, 0 * out), axis=0)
    #     p = pred[(idx, p) ]
    #     out[k, :] = p
    #     k += 1
    # out = out[:k, :]
    # print(out)
    # print(np.argmax(out == K, axis=0))
    # out = out.T
    #
    # print(out[out != K].ravel())
    # print(g.path([2, 6, 10, 4, 2], [3, 11, 3, 8, 1]))

    # p1 = [0, 5, 4, 2, 4, 6, 2, 3, 5, 6]
    # p2 = [4, 5, 2, 1, 2, 4, 5, 6]
    # l1 = [2, 3, 4, 1]
    # l2 = [2, 1, 3, 2]
    # print(p1)
    # print(np.delete(p1, np.cumsum(l1) - 1))
    # lengths = np.array(l1) + np.array(l2)
    # print(np.append(VarRowArray(l1).rows(), VarRowArray(l2).rows()))
    # idx = np.argsort(np.append(VarRowArray(l1).rows(), VarRowArray(l2).rows()), kind="mergesort")
    # nodes = np.append(p1, p2)[idx]
    # print(np.cumsum(lengths) - l2)
    # cross = np.zeros(len(nodes) + len(l1))
    # cross[np.cumsum(lengths + 1) - l2 - 1] = 1
    # cross[np.cumsum(lengths + 1) - 1] = -1
    # print(idx)
    # print(nodes)
    # print(cross)


    # n1b = [5, 10, 12, 3, 15, 3, 4,  5,  1,   3]
    # n2b = [2, 6,  4, 14, 6, 12, 10, 12, 10, 12]
    #
    # N1b = [3,  5, 4,  5]
    # N2b = [14, 2, 10, 12]
    # map1 = np.zeros(20, dtype=int)
    # map1[n1b] = np.arange(len(n1b))
    # map2 = np.zeros(20, dtype=int)
    # map2[n2b] = np.arange(len(n2b))
    # print(map1[n1b])
    # print(map2[n2b])
    # hash = map1[n1b] + len(N1b) * map2[n2b]
    # print(map1[N1b])
    # print(map2[N2b])
    # hash_p = map1[N1b] + len(N1b) * map2[N2b]
    # idx = np.argsort(hash)
    #
    #
    # print(hash)
    # print(hash_p)
    # print(    idx[np.searchsorted(hash[idx], hash_p)])
    # g.plot(show_node_ids=True)
    # plt.show()
import pyamg


