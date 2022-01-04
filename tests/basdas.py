import time
import numpy as np
import scipy
import scipy.sparse.linalg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from josephson_circuit import HoneycombArray, SquareArray
from josephson_circuit import Circuit
from static_problem import lobpcg_test_negative_definite, DefaultCurrentPhaseRelation, StaticProblem

array = SquareArray(10, 10)
array.set_inductance_factors(1)
Is = array.horizontal_junctions()
f = 0.01

n = np.zeros(array.face_count())
n[array.locate_faces(4.5, 4.5)] = 1

prob = StaticProblem(array, vortex_configuration=n, frustration=f, current_sources=Is)
_, _, conf, _ = prob.compute_maximal_current()


def is_stable_no_inductance(array, theta):
    """
    Determines if a configuration on an array with no inductance is stable in the sense that
    the Jacobian is negative definite, which it returns along with a StabilityInfo handle.
    Does not explicitly check if configuration is a stationairy point.
    """
    tic = time.perf_counter()
    cp = DefaultCurrentPhaseRelation()
    Nj, Nnr = array._Nj(), array._Nnr()
    Ic = array._Ic().diag(force_as_vector=True, vector_length=Nj)
    d_theta = cp.d_eval(Ic, theta)
    Mr = array._Mr().A
    MrT = Mr.T
    M = array.get_cut_matrix()
    MT = M.T
    A = array.get_cycle_matrix()
    L = array._L().matrix(Nj)
    q = cp.d_eval(Ic, theta)

    J2 = -scipy.sparse.vstack(
        [Mr @ scipy.sparse.diags(q, 0), A @ (scipy.sparse.eye(Nj) + L @ scipy.sparse.diags(q, 0))]).tocsc()

    m = scipy.sparse.vstack(
        [Mr, A @ L]).tocsc()

    J1 = -Mr @ scipy.sparse.diags(d_theta, 0) @ MrT
    preconditioner = scipy.sparse.linalg.LinearOperator((Nnr, Nnr), scipy.sparse.linalg.factorized(J1))

    AL = (A @ L @ A.T).tocoo()
    ALL = scipy.sparse.coo_matrix((AL.data, (AL.row + Nnr, AL.col + Nnr)), shape=(Nj, Nj)).tocsc()
    J3 = - (m @ scipy.sparse.diags(q, 0) @ m.T + ALL)
    select = np.diff(J3.indptr)!=0

    J3 = J3[select, :][:, select]

    print(time.perf_counter() - tic)
    x0 = np.random.rand(Nnr, 1)
    tic = time.perf_counter()
    lobpcg_out = scipy.sparse.linalg.lobpcg(J1, x0, M=preconditioner,
                                            maxiter=1000, tol=1E-10,
                                            retLambdaHistory=True,
                                            retResidualNormsHistory=True)
    print(lobpcg_out[0], time.perf_counter() - tic)

    tic = time.perf_counter()
    w, v = scipy.sparse.linalg.eigs(J1, 1, maxiter=1000, which="LR")
    print(w, time.perf_counter() - tic)


    # tic = time.perf_counter()
    # print(J2.shape)
    # w, v = scipy.sparse.linalg.eigs(J2, 1, M=m, maxiter=1000, which="LR")
    # print(w, time.perf_counter() - tic)


    tic = time.perf_counter()
    print(J2.shape)
    w, v = scipy.sparse.linalg.eigsh(J3, 1, maxiter=1000, which="LA")
    print(w, time.perf_counter() - tic)

    return lobpcg_out


is_stable_no_inductance(array, conf.get_theta())
