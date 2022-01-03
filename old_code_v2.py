def lobpcg_test_negative_definite(A, B, preconditioner=None, accept_ratio=DEF_STAB_RATIO, maxiter=DEF_STAB_MAXITER,
                                  tol=DEF_TOL):

    """
    Computes if inv(B) @ A is negative definite and returns this, along with the largest eigenvalue
    and residual at the last lobpcg step.

    Uses lobpcg iteratively with increasing maxiter and stops if its output max_eigv and residual:
     * if max_eigv + accept_ratio * residual < eps, considered negative definite
     * if max_eigv > eps, considered not negative definite
     * if total_iters > maxiter, apply first criterion
     * if residual < tol, apply first criterion

    Accepts a preconditioner for A.
    """
    eps = 2 * np.finfo(float).eps

    step_iters = 2
    total_iters = step_iters

    x0 = np.random.rand(A.shape[0], 1)
    lobpcg_out = scipy.sparse.linalg.lobpcg(A, x0, B=B, M=preconditioner, maxiter=step_iters, tol=tol,
                                            retLambdaHistory=True, retResidualNormsHistory=True)
    if len(lobpcg_out) <= 2:
        max_eigenvalue = lobpcg_out[0]
        residual = np.array([0])
    else:
        max_eigenvalue = np.stack(lobpcg_out[2]).ravel()[1:]
        residual = np.stack(lobpcg_out[3]).ravel()
    if max_eigenvalue.size == 0:
        max_eigenvalue = np.array([-np.inf])
    max_eigenvalue = np.array(max_eigenvalue)

    while (max_eigenvalue[-1] + accept_ratio * residual[-1] > eps) and total_iters < maxiter and residual[-1] > tol:
        if max_eigenvalue[-1] > eps or total_iters > maxiter:
            return False, max_eigenvalue, residual
        lobpcg_out = scipy.sparse.linalg.lobpcg(A, lobpcg_out[1], B=B, M=preconditioner, maxiter=step_iters, tol=tol,
                                                retLambdaHistory=True, retResidualNormsHistory=True)
        total_iters += step_iters
        step_iters += 1
        if len(lobpcg_out) <= 2:
            max_eigenvalue = np.append(max_eigenvalue, [lobpcg_out[0]])
            residual = np.append(residual, np.array([0]))
        else:
            new_max_eigv = np.stack(lobpcg_out[2]).ravel()[1:]
            if new_max_eigv.size == 0:
                break
            max_eigenvalue = np.append(max_eigenvalue, new_max_eigv)
            residual = np.append(residual, lobpcg_out[3])

    is_negative_definite = max_eigenvalue[-1] + accept_ratio * residual[-1] < eps
    if isinstance(is_negative_definite, np.ndarray):
        is_negative_definite = is_negative_definite[0]
    return is_negative_definite, max_eigenvalue, residual


def eigsh_test_negative_definite(A, B, maxiter=DEF_STAB_MAXITER):
    eps = 2 * np.finfo(float).eps

    w, v = scipy.sparse.linalg.eigsh(A, 1, M=B, maxiter=maxiter)
    eigv = w[0]
    return eigv < eps, eigv, 0.0

def get_max_eigenvalue(A, B, maxiter=DEF_STAB_MAXITER, is_symmetric=True):
    eps = 2 * np.finfo(float).eps
    if is_symmetric:
        w, v = scipy.sparse.linalg.eigsh(A, 1, M=B, maxiter=maxiter)
    else:
        w, v = scipy.sparse.linalg.eigh(A, 1, M=B, maxiter=maxiter)
    eigv = w[0]
    return eigv < eps, eigv, 0.0

def is_stable_no_inductance(array, theta, cp, maxiter=DEF_STAB_MAXITER, accept_ratio=DEF_STAB_RATIO, tol=DEF_TOL):
    """
    Determines if a configuration on an array with no inductance is stable in the sense that
    the Jacobian is negative definite, which it returns along with a StabilityInfo handle.
    Does not explicitly check if configuration is a stationairy point.
    """
    info = StabilityInfo(maxiter, accept_ratio)
    Nj, Nnr = array._Nj(), array._Nnr()
    Ic = array._Ic().diag(force_as_vector=True, vector_length=Nj)
    d_theta = cp.d_eval(Ic, theta)
    M = array._Mr().A
    MT = M.T
    M_sw = -M @ scipy.sparse.diags(d_theta, 0) @ MT
    preconditioner = scipy.sparse.linalg.LinearOperator((Nnr, Nnr), scipy.sparse.linalg.factorized(M_sw))
    B = M @ scipy.sparse.diags(array._R().diag(force_as_vector=True, vector_length=Nj) ** -1, 0) @ MT
    is_stable, max_eigenvalue, residual = lobpcg_test_negative_definite(M_sw, B, accept_ratio=accept_ratio,
                                                                        maxiter=maxiter, preconditioner=preconditioner,
                                                                        tol=tol)
    return is_stable, info._set(max_eigenvalue, residual)

def is_stable_inductance(array: Array, theta, cp, ALA_solver=None, maxiter=DEF_STAB_MAXITER, accept_ratio=5, tol=DEF_TOL):
    """
    Determines if a configuration on an array with inductance is stable in the sense that
    the Jacobian is negative definite, which it returns along with a StabilityInfo handle.
    Does not explicitly check if configuration is a stationairy point.
    """
    info = StabilityInfo(maxiter, accept_ratio)
    A = array.get_cycle_matrix()
    AT = A.T
    Nj = array.junction_count()
    if ALA_solver is None:
        L = array._L().matrix(Nj)
        ALA_solver = scipy.sparse.linalg.factorized(A @ L @ AT)
    d_theta = cp.d_eval(array._Ic().diag(force_as_vector=True, vector_length=array._Nj()), theta)[:, None]
    B = scipy.sparse.diags(array._R().diag(force_as_vector=True, vector_length=array._Nj()) ** -1, 0)
    def func(x):
        x = x.reshape(Nj, 1)
        return -d_theta * x - AT @ ALA_solver(A @ x)
    A_lin = scipy.sparse.linalg.LinearOperator((array._Nj(), array._Nj()), matvec=func)
    is_stable, max_eigenvalue, residual = lobpcg_test_negative_definite(A_lin, B, accept_ratio=accept_ratio, maxiter=maxiter, tol=tol)
    return is_stable, info._set(max_eigenvalue, residual)


class StabilityInfo:

    """
    Information about numerical procedure used to determine stability of a configuration.
    Use print(stability_info) to display the information.

    Methods:
    -------
    get_maxiter()                       Returns maximum number of lobpcg iterations before abort.
    get_accept_ratio()                  Returns eigenvalue_to_residual_ratio at which lobpcg iteration
                                        is aborted because residual is sufficiently below 0 and considered stable.
    get_max_eigenvalue()                Returns maximum eigenvalue found by lobpcg iteration.
    get_iter_count()                    Returns number of lobpcg iterations.
    get_residual()                      Returns residual of last step of lobpcg iteration.
    is_stable()                         Returns if configuration was found to be stable.
    get_runtime()                       Returns runtime in seconds.
    get_eigenvalue_to_residual_ratio()  -returns get_max_eigenvalue() / get_residual()
    """

    def __init__(self,  maxiter, accept_ratio):
        self.maxiter = maxiter
        self.accept_ratio = accept_ratio
        self.start_time = time.perf_counter()
        self.iter_count = 0
        self.residual = np.array(0)
        self.max_eigenvalue = np.array(0)
        self.is_stable_v =  False
        self.elapsed_time = 0

    def get_maxiter(self):
        return self.maxiter

    def get_accept_ratio(self):
        return self.accept_ratio

    def get_max_eigenvalue(self):
        return self.max_eigenvalue[-1]

    def get_iter_count(self):
        return self.iter_count

    def get_residual(self):
        return self.residual[-1]

    def is_stable(self):
        return self.is_stable_v

    def get_runtime(self):
        return self.elapsed_time

    def get_eigenvalue_to_residual_ratio(self):
        return -self.get_max_eigenvalue()/self.get_residual()

    def __str__(self):
        out = f"stability check info: (maxiter={self.get_maxiter()}, accept_ratio={self.get_accept_ratio()})\n\t"
        out += f"is stable: {self.is_stable()}\n\t"
        out += f"step count (of lobpcg method): {self.get_iter_count()}\n\t"
        out += f"maximum eigenvalue: {self.get_max_eigenvalue()}\n\t"
        out += f"lobpcg residual: {self.get_residual()}\n\t"
        out += f"max_eigv-to-residual ratio: {self.get_eigenvalue_to_residual_ratio()}\n\t"
        out += f"elapsed time: {self.get_runtime()} sec"
        return out

    def _set(self, max_eigenvalue, residual):
        self.iter_count = np.array(residual).size
        self.residual = np.array(residual)
        self.max_eigenvalue = np.array(max_eigenvalue)
        self.is_stable_v = max_eigenvalue[-1] < (2 * np.finfo(float).eps)
        self.elapsed_time = time.perf_counter() - self.start_time
        return self

        # if array._has_capacitance():
        # else:
        #     Cnext = Rv
        #     Asq_fact = scipy.sparse.linalg.factorized(A @ scipy.sparse.diags(1.0 / Cnext[:, 0], 0) @ AT)
        #     for i in range(self._Nt()):
        #         Is, T, theta_s, f = self._Is(i), self._T(i), self._theta_s(i), self._f(i)
        #
        #         rand = np.random.randn(Nj, W) if i % 3 == 0 else rand[np.random.permutation(Nj), :]
        #         fluctuations = ((2.0 * T * Rv) ** 0.5) * rand
        #
        #         theta = theta_next.copy()
        #         if array._has_inductance():
        #             y = AT @ L_sw_fact(A @ (theta + theta_s + L @ Is) + 2 * np.pi * f)
        #             theta_next = theta - (self._cp(theta) + fluctuations - Is + y) / Cnext
        #             if self.store_time_steps[i]:
        #                 out._update([theta_next if store_th else None,
        #                              (theta_next - theta) / dt if store_V else None,
        #                              Is - y if store_I else None])
        #         else:
        #             x = (self._cp(theta) + fluctuations - Is - Cnext * theta) / Cnext
        #             theta_next = -x + (AT @ Asq_fact(A @ (x - theta_s) + 2 * np.pi * f)) / Cnext
        #             if self.store_time_steps[i]:
        #                 out._update([theta_next if self.store_theta else None,
        #                              (theta_next - theta) / dt if store_V else None,
        #                              (x + theta_next) * Cnext + Is if store_I else None])
