from __future__ import annotations

import time

import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.optimize
from scipy.sparse.linalg import ArpackNoConvergence

from compute import Matrix
from josephson_circuit import Circuit

__all__ = ["CurrentPhaseRelation", "DefaultCPR",
           "StaticProblem", "StaticConfiguration"]

DEF_TOL = 1E-11

DEF_NEWTON_MAXITER = 30
DEF_STAB_MAXITER = 100

DEF_MAX_PAR_TOL = 1E-4
DEF_MAX_PAR_REDUCE_FACT = 0.42

class CurrentPhaseRelation:

    """

    Current-Phase relation Icp(Ic, theta). The default value is Icp = Ic * sin(theta).

    Input:
    ------
    func            lambda Ic, theta: ...   current-phase relation
    d_func          lambda Ic, theta: ...   derivative of current-phase relation to theta
    i_func          lambda Ic, theta: ...   integral of current-phase relation over theta (starting at 0)

    Methods:
    --------
    eval(Ic: np.ndarray, theta: np.ndarray) -> np.ndarray
    d_eval(Ic: np.ndarray, theta: np.ndarray) -> np.ndarray
    i_eval(Ic: np.ndarray, theta: np.ndarray) -> np.ndarray

    Remarks:
    --------
     - func, d_func and i_func must be numpy ufunc, so their output must be broadcast of input Ic and theta.
    """
    def __init__(self, func, d_func, i_func):
        self.func = func
        self.d_func = d_func
        self.i_func = i_func

    def eval(self, Ic, theta):
        return self.func(Ic, theta)

    def d_eval(self, Ic, theta):
        return self.d_func(Ic, theta)

    def i_eval(self, Ic, theta):
        return self.i_func(Ic, theta)

class DefaultCPR(CurrentPhaseRelation):

    """
    Default current-phase relation Icp = Ic * sin(theta).

    """
    def __init__(self):
        super().__init__(lambda Ic, th: Ic * np.sin(th),
                         lambda Ic, th: Ic * np.cos(th),
                         lambda Ic, th: Ic * (1.0 - np.cos(th)))


class NewtonIterInfo:

    """
    Information about the newton iteration used to find static configurations.
    Use print(newton_iter_info) to display the information.

    Methods:
    -------
    get_max_iter()                          Returns number of iterations after which iteration is aborted.
    get_tol()                               Returns tolerance.
    get_status()                            Returns 0: converged. residual < tolerance
                                                    1: diverged before reaching maxiter.
                                                    2: reached max_iter without converging or diverging.
    has_converged()                         Returns if iteration has converged.
    get_is_target_vortex_configuration()    Returns (nr_of_iters,) bool array if vortex configuration at iter
                                            agrees with vortex configuration specified in problem.
    found_target_solution()                 Returns True if has_converged() and final iter obeys target vortex config.
    get_number_of_iterations()              Returns number of newton iterations done.
    get_residual()                          (nr_of_iters,) array containing residual at each iteration.
    get_runtime()                           Returns runtime in seconds.
    plot_residuals()                        Plots residual vs iteration number.

    """
    def __init__(self, tol, maxiter):
        self.start_time = time.perf_counter()
        self.tol = tol
        self.maxiter = maxiter
        self.iteration = 0
        self.error = np.zeros(self.maxiter + 1, dtype=np.double)
        self.has_converged = False
        self.is_target_n = np.zeros(self.maxiter + 1, dtype=int)
        self.runtime = 0.0

    def get_max_iter(self):
        return self.maxiter

    def get_tol(self):
        return self.tol

    def get_status(self):
        return int(not self.found_target_solution()) + 2 * int(self.iteration >= self.maxiter)

    def has_converged(self):
        return self.has_converged

    def get_is_target_vortex_configuration(self):
        return self.is_target_n[:(self.get_number_of_iterations()+1)]

    def found_target_solution(self):
        return self.has_converged and self.is_target_n[self._get_iteration()]

    def get_number_of_iterations(self):
        return self._get_iteration()

    def get_residual(self):
        return self.error[:(self.get_number_of_iterations()+1)]

    def get_runtime(self):
        return self.runtime

    def plot_residuals(self):
        import matplotlib.pyplot as plt
        n = self.get_is_target_vortex_configuration().astype(bool)
        y = self.get_residual()
        x = np.arange(len(y))
        plt.semilogy(x[n], y[n], color=[0, 0, 0], label="n is target_n", linestyle="None", marker="o")
        plt.semilogy(x[~n], y[~n], color=[1, 0, 0], label="n is not target_n", linestyle="None", marker="o")
        plt.xlabel("Newton iteration number")
        plt.ylabel("residual")
        plt.title("Evolution of residual for newton iteration.")
        plt.legend()

    def __str__(self):
        out = f"newton iteration info: (tol={ self.get_tol()}, maxiter={self.get_max_iter()})\n\t"
        out += f"status: {self.get_status()}"
        if self.get_status() == 0:
            out += f" (converged)\n\t"
        if self.get_status() == 1:
            out += f" (diverged)\n\t"
        if self.get_status() == 2:
            out += f" (indeterminate; reached max_iter without converging or diverging)\n\t"
        out += f"number of iterations: {self.get_number_of_iterations()}\n\t"
        out += f"residual: {self.get_residual()}\n\t"
        out += f"runtime (in sec): {self.get_runtime()}"
        return out

    def _set(self, error, is_target_n_v):
        self.has_converged = error < self.tol
        self.is_target_n[self.iteration] = is_target_n_v
        self.error[self.iteration] = error
        self.runtime += time.perf_counter() - self.start_time
        self.start_time = time.perf_counter()
        self.iteration += 1
        return self

    def _get_iteration(self):
        return max(0, self.iteration - 1)


class ParameterOptimizeInfo:

    """
    Information about the parameter optimization process.
    Use print(parameter_optimize_info) to display the information.

    Methods:
    -------
    get_has_stable_target_solution_at_zero()    Returns if a stable target solution is found at lambda=0
    get_lambda()                                Returns (nr_of_steps,) array with lambda at each step
    get_lambda_error()                          Returns (nr_of_steps,) array with error in lambda
    get_lambda_lower_bound()                    Returns lower bound for lambda
    get_lambda_upper_bound()                    Returns upper bound for lambda
    get_found_stable_target_solution()          Returns (nr_of_steps,) array if a stable target solution is found at step
    get_newton_converged()                      Returns (nr_of_steps,) array if newton iteration converged at step
    get_newton_target_n()                       Returns (nr_of_steps,) array if target n at step
    get_newton_steps()                          Returns (nr_of_steps,) array with nr of newton iterations at step
    get_newton_iter_all_info()                  Returns (nr_of_steps,) list containing newton_iter_infos
    get_is_stable()                             Returns (nr_of_steps,) array if a stable at step
    get_stability_steps()                       Returns (nr_of_steps,) array with stability algorithm iterations at step
    get_stable_iter_all_info()                  Returns (nr_of_steps,) list containing stable_iter_infos
    get_runtime()                               Returns runtime in seconds.
    plot_residuals()                            Plots residual vs iteration number.

    """
    def __init__(self, Is_func, f_func, lambda_tol, M):
        self.Is_func = Is_func
        self.f_func = f_func
        self.lambda_tol = lambda_tol
        self.has_stable_target_solution_at_zero = False
        self.lambda_history = np.zeros(1000, dtype=np.double)
        self.stepsize_history = np.zeros(1000, dtype=np.double)
        self.solution_history = np.zeros(1000, dtype=np.bool)
        self.newton_iter_infos = []
        self.M = M
        self._step = 0
        self._time = time.perf_counter()

    def get_has_stable_target_solution_at_zero(self):
        return self.has_stable_target_solution_at_zero

    def get_lambda(self):
        return self.lambda_history[:self._step]

    def get_lambda_error(self):
        return self._get_lambda_stepsize() / self.get_lambda()

    def get_lambda_lower_bound(self):
        if not self.get_has_stable_target_solution_at_zero():
            return np.nan
        s = self.get_lambda()[self.get_found_stable_target_solution()]
        return s[-1] if s.size > 0 else 0

    def get_lambda_upper_bound(self):
        s = self.get_lambda()[~self.get_found_stable_target_solution()]
        return s[-1] if s.size > 0 else np.inf

    def get_found_stable_target_solution(self):
        return self.solution_history[:self._step]

    def get_newton_iter_all_info(self):
        return self.newton_iter_infos

    def get_newton_converged(self):
        return np.array([info.converged() for info in self.newton_iter_infos], dtype=int)

    def get_newton_target_n(self):
        return np.array([info.get_is_target_vortex_configuration()[-1] for info in self.newton_iter_infos], dtype=int)

    def get_newton_steps(self):
        return np.array([info.get_number_of_iterations() for info in self.newton_iter_infos], dtype=int)

    def get_runtime(self):
        return self._time

    def plot_residuals(self):
        import matplotlib.pyplot as plt
        for i, n_info in enumerate(self.get_newton_iter_all_info()):
            n = n_info.get_is_target_vortex_configuration().astype(bool)
            y = n_info.get_residual()
            x = np.arange(len(y))
            plt.semilogy(x[n], y[n], color=[0, 0, 0], linestyle="None", marker="o")
            plt.semilogy(x[~n], y[~n], color=[1, 0, 0], linestyle="None", marker="o")
            plt.text(x[-1], y[-1], str(i), color=[0.3, 0.3, 0.3])
            plt.semilogy(x, y, color=[0, 0, 0])
        plt.xlabel("Newton iteration number")
        plt.ylabel("residual")
        plt.title("Newton residuals in parameter optimization.")
        plt.legend(["n is target_n", "n is not target_n"])

    def __str__(self):
        np.set_printoptions(linewidth=100000)
        out = "parameter optimize info:\n\t"
        if not self.get_has_stable_target_solution_at_zero():
            out += "optimization failed because not solution was found at lambda=0"
        else:
            def int_digit_count(x):
                return np.ceil(np.log(np.max(x)) / np.log(10)).astype(int)
            n = max(5, 3 + int_digit_count(1/self.lambda_tol), int_digit_count(self.get_newton_steps()))
            out += f"Found lambda between {self.get_lambda_lower_bound()} and {self.get_lambda_upper_bound()}"
            if self.last_step_status != 2:
                out += f" at desired tolerance (resid={self.get_lambda_error()[-1]}) \n\t"
            else:
                out += f"; stopped because newton iteration was indeterminate. Consider increasing newton_maxiter.)\n\t"
            out += f"runtime: {np.round(self.get_runtime(), 7)} sec\n\t"
            out += f"lambda: {self.get_lambda()}\n\t"
            out += f"net I:  {self._get_net_I()}\n\t"
            out += f"mean f: {self._get_mean_f()}\n\t"
            np.set_printoptions(formatter={'float': lambda x: ("{0:0." + str(n - 2) + "f}").format(x)})
            out += f"lambda / lambda[0]:    {self.get_lambda() / self.get_lambda()[0]}\n\t"
            np.set_printoptions(formatter={'bool': lambda x: ("{:>" + str(n) + "}").format(x)})
            out += f"newton converged:      {self.get_found_stable_target_solution().astype(bool)}\n\t"
            out += f"to target n:           {self.get_newton_target_n().astype(bool)}\n\t"
            np.set_printoptions(formatter={'int': lambda x: ("{:>" + str(n) + "}").format(x)})
            out += f"total newton steps:    {self.get_newton_steps()}\n\t"
        return out

    def _preset(self, has_stable_target_solution_at_zero):
        self.has_stable_target_solution_at_zero = has_stable_target_solution_at_zero
        return self

    def _set(self, lambda_value, lambda_stepsize, found_stable_target_solution, newton_iter_info):
        self.lambda_history[self._step] = lambda_value
        self.stepsize_history[self._step] = lambda_stepsize
        self.solution_history[self._step] = found_stable_target_solution
        self.newton_iter_infos += [newton_iter_info]
        self._step += 1
        return self

    def _finish(self, last_step_status):
        self.last_step_status = last_step_status
        self._time = time.perf_counter() - self._time
        return self

    def _get_lambda_stepsize(self):
        return self.stepsize_history[:self._step]

    def _get_net_I(self):
        return np.array([0.5 * np.sum(np.abs(self.M @ np.broadcast_to(self.Is_func(l), (self.M.shape[1],)))) for l in self.get_lambda()])

    def _get_mean_f(self):
        return np.array([np.mean(self.f_func(l)) for l in self.get_lambda()])


class StaticProblem:
    """
    Define a static josephson junction array problem.

    Input:
    ------                  symbol  type                      getter
    circuit                         Circuit                   get_circuit()
    current_sources         Is      (Nj,) ndarray or scalar   get_current_sources()
    frustration             f       (Nf,) ndarray or scalar   get_frustration()
    vortex_configuration    n       (Nf,) ndarray or scalar   get_vortex_configuration()
    current_phase_relation  cp      CurrentPhaseRelation      get_current_phase_relation()

    Methods:
    --------
    get_net_sourced_current         Gets the sum of all (positive) current injected at nodes to create Is.
    new_problem                     Makes copy of self with specified modifications.
    approximate                     Computes approximate solution (London approximation or arctan approximation)
    approximate_placed_vortices     Computes arctan approximation where vortices are placed at specified
                                    coordinates, rather than in face centers.
    compute                         Compute exact solution
    compute_maximal_parameter       Computes largest x where stable solution exists with f=f(x) and Is=Is(x)
    compute_frustration_bounds      Compute smallest and largest frustration with a stable solution
    compute_maximal_current         Compute largest x where Is=x * self.Is with a stable solution
    compute_stable_region           Compute boundary of stable region in f-Is space of vortex configuration n.

    get_net_sourced_current() -> scalar

    new_problem(current_sources=None, frustration=None, vortex_configuration=None,
                current_phase_relation=None) -> StaticProblem

    approximate(algorithm=1) -> StaticConfiguration

    approximate_placed_vortices(n, x_n, y_n) -> StaticConfiguration

    compute(initial_guess: StaticConfiguration, ...) -> StaticConfiguration, int, NewtonIterInfo

    compute_maximal_parameter(Is_func, f_func, initial_guess: StaticConfiguration, ...)
     -> lower_bound, upper_bound, StaticConfiguration, ParameterOptimizeInfo

    compute_frustration_bounds(initial_guess: StaticConfiguration, start_frustration, ...)
     -> (smallest_f, largest_f), (StaticConfiguration, StaticConfiguration), (ParameterOptimizeInfo, ParameterOptimizeInfo)

    compute_maximal_current(initial_guess: StaticConfiguration, ...)
     -> Is_factor, net_I, StaticConfiguration, ParameterOptimizeInfo

    compute_stable_region(num_angles=40, start_frustration, start_initial_guess, ...)
     -> [f], [net_I], [StaticConfiguration], [ParameterOptimizeInfo]

    """

    def __init__(self, circuit: Circuit, current_sources=0.0, frustration=0.0,
                 vortex_configuration=0, current_phase_relation=DefaultCPR()):
        self.circuit = circuit
        self.current_sources = np.atleast_1d(current_sources)
        self.frustration = np.atleast_1d(frustration)
        self.vortex_configuration = np.atleast_1d(vortex_configuration)
        self.current_phase_relation = current_phase_relation
        self.current_sources_norm = None
        self.Asq_factorization = None
        self.AIpLIcA_factorization = None
        self.IpLIc_factorization = None
        self.Msq_factorization = None

    def get_circuit(self) -> Circuit:
        return self.circuit

    def get_current_sources(self):
        return self.current_sources

    def get_frustration(self):
        return self.frustration

    def get_vortex_configuration(self):
        return self.vortex_configuration

    def get_current_phase_relation(self):
        return self.current_phase_relation

    def new_problem(self, current_sources=None, frustration=None,
                    vortex_configuration=None, current_phase_relation=None):
        """
        Makes copy of self with specified modifications.
        """
        return StaticProblem(self.circuit, current_sources=self.current_sources if current_sources is None else current_sources,
                             frustration=self.frustration if frustration is None else frustration,
                             vortex_configuration=self.vortex_configuration if vortex_configuration is None else vortex_configuration,
                             current_phase_relation=self.current_phase_relation if current_phase_relation is None else current_phase_relation)

    def get_phase_zone(self):
        return 0

    def get_net_sourced_current(self):
        """
        Gets the sum of all (positive) current injected at nodes to create Is.
        """
        M = self.get_circuit().get_cut_matrix()
        return 0.5 * np.sum(np.abs((M @ self._Is())), axis=0)

    def approximate(self, algorithm=1) -> StaticConfiguration:
        """
        Computes approximate solutions.  Has two algorithms; the arctan approximation or
        the London approximation.

        alg: name                  description
         0   arctan approximation  assigns phases that "wind" 2*pi around vortices in z=0 phase zone,
                                   phi(x,y) = sum_i 2 * pi * n_i * atan2(y-y_n_i,x-x_n_i)
                                   where vortices are located at centres of their respective faces.
         1   london approximation  Find theta in cycle space (theta = A.T @ ...) that obeys winding rule.
        """
        if algorithm == 0:
            theta = arctan_approximation(self.circuit, self._f(), self._nt(),
                                         Asq_solver=self._Asq_factorization(),
                                         IpLIc_solver=self._IpLIc_factorization())
        elif algorithm == 1:
            theta = london_approximation(self.circuit, self._f(), self._nt(),
                                         AIpLIcA_solver=self._AIpLIcA_factorization())
            theta = change_phase_zone(self.get_circuit(), theta, self._nt(), 0)
        else:
            raise ValueError("invalid algorithm")
        return StaticConfiguration(self, theta)

    def approximate_placed_vortices(self, n, x_n, y_n) -> StaticConfiguration:
        """
        Compute arctan approximation with manual placement of vortices.

        Input:
        ------
        n       (N,) int array      vorticity at location (x_n, y_n)
        x_n     (N,) float array    x-coordinates of vortices
        y_n     (N,) float array    y-coordinates of vortices
        """

        theta = arctan_approximation_placed_vortices(self.circuit,
            self._f(), n, x_n, y_n, Asq_solver=self._Asq_factorization(), IpLIc_solver=self._IpLIc_factorization())

        return StaticConfiguration(self, theta)


    def compute(self, initial_guess: StaticConfiguration | np.ndarray = None,
                tol=DEF_TOL, maxiter=DEF_NEWTON_MAXITER, stop_as_residual_increases=True,
                stop_if_not_target_n=False) -> tuple[StaticConfiguration, int, NewtonIterInfo]:

        """
        Compute solutions using Newton iteration.

        Input:
        ------
        initial_guess=None                  None (London approximation is used)
                                            or (Nj,) np.ndarray representing theta
                                            or StaticConfiguration
        tol=DEF_TOL                         scalar      tolerance; is solution if |residual| < tol
        maxiter=DEF_NEWTON_MAXITER          int         maximum number of newton iterations.
        stop_if_not_target_n=False          bool        iteration stops  if n(iter) != n (diverged)
        stop_as_residual_increases=True     bool        iteration stops if error(iter) > error(iter - 3)  (diverged)

        Output:
        -------
        config      StaticConfiguration     object containing solution
        status      int                     0       converged
                                            or 1    diverged if error(iter)>0.5 or above reasons
                                            or 2    max_iter reached without converging or diverging
        iter_info   NewtonIterInfo          handle containing information about newton iteration
        """
        if initial_guess is None:
            initial_guess = self.approximate(algorithm=1)

        if isinstance(initial_guess, StaticConfiguration):
            initial_guess = initial_guess._th()

        theta, status, iter_info = static_compute(self.get_circuit(), initial_guess, Is=self._Is(),
                                                  f=self._f(), n=self._nt(), z=0,
                                                  cp=self.current_phase_relation, tol=tol,
                                                  maxiter=maxiter, Asq_solver=self._Asq_factorization(),
                                                  stop_as_residual_increases=stop_as_residual_increases,
                                                  stop_if_not_target_n=stop_if_not_target_n)
        config = StaticConfiguration(self, theta)
        return config, status, iter_info

    def compute_maximal_parameter(self, Is_func, f_func,
                                  initial_guess: StaticConfiguration | np.ndarray = None,
                                  lambda_tol=DEF_MAX_PAR_TOL, estimated_upper_bound=1.0,
                                  newton_tol=DEF_TOL, newton_maxiter=DEF_NEWTON_MAXITER,
                                  newton_stop_as_residual_increases=True,
                                  newton_stop_if_not_target_n=False,
                                  require_stability=True, stable_maxiter=DEF_STAB_MAXITER):

        """
        Finds the largest value of lambda for which a problem which current_sources=Is_func(lambda)
        and frustration=f_func(lambda) has a stable stationary state.

         - Must be able to find a stable configuration at lambda=0.
         - One can manually specify an initial_guess for lambda=0.
         - returns a lower- and upperbound for lambda. Stops when the difference < lambda_tol * lower_bound
         - furthermore returns config containing the solutions at the lower_bound. Also its
           accompanied problem has f and Is of lower_bound.
         - ignores self.current_sources and self.frustration.
         - Also returns ParameterOptimizeInfo object containing information about the iteration.
         - Algorithm stops if lambda_tol is reached or when newton_iteration failed to converge or diverge.
         - Algorithm needs an estimate of the upperbound for lambda to work.

        Input:
        ------
        Is_func                                 Is_func(lambda) -> Is
        f_func                                  f_func(lambda) -> f
        initial_guess=None                      manual approximation at lambda=0
                                                or None; then London approximation is used.
        lambda_tol=DEF_MAX_PAR_TOL              stop iterating if upperbound - lowerbound < lambda_tol * lower_bound
        estimated_upper_bound=1.0               Estimate for the upperbound for lambda

        (for newton iteration at given lambda, see .compute() for documentation)
        newton_tol=DEF_TOL
        max_total_newton_steps=DEF_MAX_PAR_NEWT_STEPS
        newton_stop_as_residual_increases=True
        newton_stop_if_not_target_n=False

        (for determining stability of solution at given lambda, see StaticConfiguration.is_stable())
        require_stability=True
        stable_maxiter=DEF_STAB_MAXITER

        Output:
        -------
        lambda_lowerbound       float                   Lowerbound of lambda
        lambda_upperbound       float                   Upperbound of lambda
        config                  StaticConfiguration     Containing solutions at lambda=lambda_lowerbound
        iteration_info          ParameterOptimizeInfo   Object containing information about the iteration.
        """
        if initial_guess is None:
            initial_guess = self.new_problem(frustration=f_func(0), current_sources=Is_func(0)).approximate(algorithm=1)

        if isinstance(initial_guess, StaticConfiguration):
            initial_guess = initial_guess._th()

        out = compute_maximal_parameter(self.get_circuit(), Is_func, f_func, z=0, n=self._nt(),
                                        cp=self.get_current_phase_relation(), theta_0=initial_guess,
                                        lambda_tol=lambda_tol, lambda_initial_stepsize=estimated_upper_bound,
                                        stepsize_reduction_factor=DEF_MAX_PAR_REDUCE_FACT,
                                        newton_tol=newton_tol, newton_maxiter=newton_maxiter,
                                        newton_stop_as_residual_increases=newton_stop_as_residual_increases,
                                        newton_stop_if_not_target_n=newton_stop_if_not_target_n, require_stability=require_stability,
                                        stable_maxiter=stable_maxiter, Asq_factorized=self._Asq_factorization())
        lower_bound, upper_bound, theta, info = out

        if lower_bound is None:
            config = None
        else:
            out_problem = self.new_problem(current_sources=Is_func(lower_bound), frustration=f_func(lower_bound))
            config = StaticConfiguration(out_problem, theta)
        return lower_bound, upper_bound, config, info

    def compute_frustration_bounds(self, initial_guess: StaticConfiguration = None,
                                   start_frustration=None, lambda_tol=DEF_MAX_PAR_TOL,
                                   newton_tol=DEF_TOL,
                                   newton_maxiter=DEF_NEWTON_MAXITER,
                                   newton_stop_as_residual_increases=True,
                                   newton_stop_if_not_target_n=False, require_stability=True,
                                   stable_maxiter=DEF_STAB_MAXITER):

        """
        Computes smallest and largest uniform frustration for which a stable solution exists at the
        specified target vortex configuration and source current.

        Input:
        ------
        start_frustration=None                  frustration factor somewhere in the middle of range.
                                                or None; is estimated based on vortex configuration.
        initial_guess=None                      manual approximation at f=start_frustration
                                                or None; then London approximation is used.
        lambda_tol=DEF_MAX_PAR_TOL              see .compute_maximal_parameter()

        (for newton iteration at given lambda, see .compute() for documentation)
        newton_tol=DEF_TOL
        max_total_newton_steps=DEF_MAX_PAR_NEWT_STEPS
        newton_stop_as_residual_increases=True
        newton_stop_if_not_target_n=False

        (for determining stability of solution at given lambda, see StaticConfiguration.is_stable())
        require_stability=True
        stable_maxiter=DEF_STAB_MAXITER

        Output:
        -------
        (smallest_f, largest_f)                 resulting f range
        (smallest_f_config, largest_f_config)   StaticConfigurations at bounds of range.
        (smallest_f_info, largest_f_info)       ParameterOptimizeInfo objects containing
                                                information about the iterations.
        """
        # TODO: take into account areas.

        if start_frustration is None:
            start_frustration = np.mean(self._nt())
        frustration_initial_stepsize = 1.0
        Is_func = lambda x: self._Is()
        f_func_smallest = lambda x: start_frustration - x
        f_func_largest = lambda x: start_frustration + x
        start_problem = self.new_problem(frustration=start_frustration)
        start_config, status, info = start_problem.compute(initial_guess=initial_guess)
        if not start_config.is_stable_target_solution():
            return (None, None), (None, None), (None, None)
        out = self.compute_maximal_parameter(Is_func, f_func_smallest, initial_guess=start_config._th(),
                                             lambda_tol=lambda_tol, estimated_upper_bound=frustration_initial_stepsize,
                                             newton_tol=newton_tol,
                                             newton_maxiter=newton_maxiter,
                                             newton_stop_as_residual_increases=newton_stop_as_residual_increases,
                                             newton_stop_if_not_target_n=newton_stop_if_not_target_n,
                                             require_stability=require_stability, stable_maxiter=stable_maxiter)
        smallest_factor, _, smallest_f_config, smallest_f_info = out
        smallest_f = f_func_smallest(smallest_factor) if smallest_factor is not None else None
        out = self.compute_maximal_parameter(Is_func, f_func_largest, initial_guess=start_config._th(),
                                             lambda_tol=lambda_tol, estimated_upper_bound=frustration_initial_stepsize,
                                             newton_tol=newton_tol,
                                             newton_maxiter=newton_maxiter,
                                             newton_stop_as_residual_increases=newton_stop_as_residual_increases,
                                             newton_stop_if_not_target_n=newton_stop_if_not_target_n,
                                             require_stability=require_stability, stable_maxiter=stable_maxiter)

        largest_factor, _, largest_f_config, largest_f_info = out
        largest_f = f_func_largest(largest_factor) if largest_factor is not None else None
        return (smallest_f, largest_f), (smallest_f_config, largest_f_config), (smallest_f_info, largest_f_info)

    def compute_maximal_current(self, initial_guess: StaticConfiguration = None,
                                lambda_tol=DEF_MAX_PAR_TOL, newton_tol=DEF_TOL,
                                newton_maxiter=DEF_NEWTON_MAXITER,
                                newton_stop_as_residual_increases=True,
                                newton_stop_if_not_target_n=False,
                                require_stability=True, stable_maxiter=DEF_STAB_MAXITER):

        """
        Computes largest source current for which a stable solution exists at the specified target
        vortex configuration and frustration, where the source current is assumed to be
        max_current_factor * self.get_current_sources().

        Input:
        ------
        see .compute_maximal_parameter() for documentation.

        Output:
        -------
        max_current_factor
        net_sources_current     Net sourced current at max_current_factor.
        out_config              StaticConfiguration of state with maximal current.
        info                    ParameterOptimizeInfo objects containing information about the iterations.

        """
        M, Nj = self.get_circuit()._Mr().matrix(), self.get_circuit()._Nj()
        if np.all(self._Is() == 0):
            raise ValueError("Problem must contain nonzero current sources.")

        Is_per_node = np.abs(M @ self._Is())
        max_super_I_per_node = np.abs(M) @ self.get_circuit()._Ic().diag(True, Nj)
        current_factor_initial_stepsize = 1.0 / np.max(Is_per_node / max_super_I_per_node)
        Is_func = lambda x: x * self._Is()
        f_func = lambda x: self._f()
        out = self.compute_maximal_parameter(Is_func, f_func, initial_guess=initial_guess, lambda_tol=lambda_tol,
                                             estimated_upper_bound=current_factor_initial_stepsize,
                                             newton_maxiter=newton_maxiter, newton_tol=newton_tol,
                                             newton_stop_as_residual_increases=newton_stop_as_residual_increases,
                                             newton_stop_if_not_target_n=newton_stop_if_not_target_n,
                                             require_stability=require_stability, stable_maxiter=stable_maxiter)
        max_current_factor, upper_bound, out_config, info = out
        net_I = out_config.get_problem().get_net_sourced_current() if out_config is not None else None
        return max_current_factor, net_I, out_config, info


    def compute_stable_region(self, angles=np.linspace(0, 2*np.pi, 61), start_frustration=None,
                              start_initial_guess: StaticConfiguration | np.ndarray = None,
                              lambda_tol=DEF_MAX_PAR_TOL, newton_tol=DEF_TOL,
                              newton_maxiter=DEF_NEWTON_MAXITER,
                              newton_stop_as_residual_increases=True,
                              newton_stop_if_not_target_n=False,
                              require_stability=True, stable_maxiter=DEF_STAB_MAXITER):

        """
        Finds edge of stable region in (f, Is) space for vortex configuration n.

        Input:
        ------
        angles=np.linspace(0, 2*np.pi, 60)   Angles at which an extremum in (f, Is) space is searched for.
        (see documentation of .compute_maximal_parameter() for other input.)

        Output:
        -------
        frustration  (num_angles,) ndarray          net extermum frustration at each angle
        net_current  (num_angles,) ndarray          net extremum sourced current at each angle
        all_configs  list of StaticConfiguration    config at extreme value for each angle
        all_infos    list of ParameterOptimizeInfo  objects containing information about the iterations at each angle.

        """
        import matplotlib.pyplot as plt

        num_angles = len(angles)

        frust_bnd_prb = self.new_problem(current_sources=0)
        out = frust_bnd_prb.compute_frustration_bounds(initial_guess=start_initial_guess,
                                              start_frustration=start_frustration,
                                              lambda_tol=lambda_tol,
                                              newton_tol=newton_tol, newton_maxiter=newton_maxiter,
                                              newton_stop_as_residual_increases=newton_stop_as_residual_increases,
                                              newton_stop_if_not_target_n=newton_stop_if_not_target_n,
                                              require_stability=require_stability, stable_maxiter=stable_maxiter)

        (smallest_f, largest_f), _, _ = out
        if smallest_f is None:
            return None, None, None, None
        dome_center_f = 0.5 * (smallest_f + largest_f)
        dome_center_problem = self.new_problem(frustration=dome_center_f)
        out = dome_center_problem.compute_maximal_current(lambda_tol=lambda_tol,
                                                          newton_tol=newton_tol, newton_maxiter=newton_maxiter,
                                                          newton_stop_as_residual_increases=newton_stop_as_residual_increases,
                                                          newton_stop_if_not_target_n=newton_stop_if_not_target_n,
                                                          require_stability=require_stability, stable_maxiter=stable_maxiter)
        max_current_factor, _, _, _ = out
        if max_current_factor is None:
            return None, None, None, None
        frustration = np.zeros(num_angles, dtype=np.double)
        net_current = np.zeros(num_angles, dtype=np.double)
        all_configs, all_infos = [], []
        dome_center_problem = self.new_problem(frustration=dome_center_f, current_sources=0)
        for angle_nr in range(num_angles):
            angle = angles[angle_nr]
            Is_func = lambda x: x * self._Is() * np.sin(angle) * max_current_factor
            f_func = lambda x: dome_center_f + x * np.cos(angle) * (0.5 * (largest_f - smallest_f))
            out = dome_center_problem.compute_maximal_parameter(Is_func, f_func, lambda_tol=lambda_tol,
                                                 newton_tol=newton_tol, newton_maxiter=newton_maxiter,
                                                 newton_stop_as_residual_increases=newton_stop_as_residual_increases,
                                                 newton_stop_if_not_target_n=newton_stop_if_not_target_n,
                                                 require_stability=require_stability, stable_maxiter=stable_maxiter)
            lower_bound, upper_bound, out_config, info = out
            net_current[angle_nr] = out_config.get_problem().get_net_sourced_current() * np.sign(np.sin(angle)) if lower_bound is not None else np.nan
            frustration[angle_nr] = f_func(lower_bound) if lower_bound is not None else np.nan
            all_configs += [out_config]
            all_infos += [info]

        return frustration, net_current, all_configs, all_infos

    def __str__(self):
        return "static problem: " + \
               "\n\tcurrent sources: " + self.get_current_sources().__str__() + \
               "\n\tfrustration: " + self.get_frustration().__str__() + \
               "\n\tvortex configuration: " + self.get_vortex_configuration().__str__() + \
               "\n\tphase zone: " + self.get_phase_zone().__str__() + \
               "\n\tcurrent-phase relation: " + self.current_phase_relation.__str__()

    def _Is(self):
        return self.current_sources

    def _f(self):
        return self.frustration

    def _nt(self):
        return self.vortex_configuration

    def _cp(self, Ic: Matrix, theta):
        return self.current_phase_relation.eval(Ic.diag(), theta)

    def _dcp(self, Ic: Matrix, theta):
        return self.current_phase_relation.d_eval(Ic.diag(), theta)

    def _icp(self, Ic: Matrix, theta):
        return self.current_phase_relation.i_eval(Ic.diag(), theta)

    def _Is_norm(self):
        if self.current_sources_norm is None:
            self.current_sources_norm = scipy.linalg.norm(np.broadcast_to(self.current_sources, (self.circuit._Nj(),)))
        return self.current_sources_norm

    def _Asq_factorization(self):
        if self.Asq_factorization is None:
            A = self.get_circuit().get_cycle_matrix()
            self.Asq_factorization = scipy.sparse.linalg.factorized(A @ A.T)
        return self.Asq_factorization

    def _AIpLIcA_factorization(self):
        if self.AIpLIcA_factorization is None:
            Nj, A = self.get_circuit()._Nj(), self.get_circuit().get_cycle_matrix()
            L, Ic = self.get_circuit()._L().matrix(Nj), self.get_circuit()._Ic().matrix(Nj)
            self.AIpLIcA_factorization = scipy.sparse.linalg.factorized(A @ (scipy.sparse.eye(Nj) + L @ Ic) @ A.T)
        return self.AIpLIcA_factorization

    def _IpLIc_factorization(self):
        if self.IpLIc_factorization is None:
            Nj = self.get_circuit()._Nj()
            L, Ic = self.get_circuit()._L().matrix(Nj), self.get_circuit()._Ic().matrix(Nj)
            self.IpLIc_factorization = scipy.sparse.linalg.factorized(scipy.sparse.eye(Nj) + L @ Ic)
        return self.IpLIc_factorization

    def _Msq_factorization(self):
        if self.Msq_factorization is None:
            M = self.get_circuit()._Mr().A
            self.Msq_factorization = scipy.sparse.linalg.factorized(M @ M.T)
        return self.Msq_factorization

class StaticConfiguration:
    """
    Approximation or solution to static problem.

    It is defined by a StaticProblem and theta. Here theta must be a
    numpy array of shape (Nj,).

    Methods:
    --------
    get_problem()                    Returns problem
    get_circuit()                      Returns circuit (stored in problem)
    get_phi()                        Returns (Nn,) array containing phases at each node
    get_theta()                      Returns (Nj,) array containing gauge invariant phase difference at each junction
    get_n()                          Returns (Nf,) int array containing vorticity at each face
    get_I()                          Returns (Nj,) array containing current through each junction
    get_J()                          Returns (Nf,) array containing path current around each face
    get_flux()                       Returns (Nf,) array containing magnetic flux at each face
    get_EM()                         Returns (Nj,) array containing magnetic energy at each junction
    get_EJ()                         Returns (Nj,) array containing Josephson energy at each junction
    get_Etot()                       Returns get_EM() + get_EJ()
    satisfies_kirchhoff_rules()      Returns if configuration satisfies kirchhoff's rules
    satisfies_winding_rules()        Returns if configuration satisfies the winding rules
    satisfies_target_vortices()      Returns if vortex configuration equals that of problem
    is_stable(maxiter, accept_ratio) Returns if configuration is dynamically stable.
    is_solution(tol)                 Returns if configuration is a solution meaning it must satisfy both kirchhoff
                                     and winding rules
    is_target_solution(tol)          is_solution() and satisfies_target_vortices()
    is_stable_target_solution()      is_solution() and satisfies_target_vortices() and is_stable()
    get_error_kirchhoff_rules()      Returns normalized residual of kirchhoff's rules (normalized so cannot exceed 1)
    get_error_winding_rules()        Returns normalized residual of the winding rules (normalized so cannot exceed 1)
    get_error()                      Returns get_error_kirchhoff_rules(), get_error_winding_rules()
    plot(...) -> plot_handles        Creates plot of configuration.
    """

    def __init__(self, problem: StaticProblem, theta: np.ndarray):
        self.problem = problem
        self.theta = np.array(theta)
        if not self.theta.shape == (self.problem.get_circuit()._Nj(),):
            raise ValueError("theta must be of shape (Nj,)")

    def get_circuit(self) -> Circuit:
        return self.problem.get_circuit()

    def get_problem(self) -> StaticProblem:
        return self.problem

    def get_phi(self) -> np.ndarray:
        # by default the last node (node with highest index number) is grounded.
        M, Msq_solver = self.get_circuit()._Mr().matrix(), self.get_problem()._Msq_factorization()
        return np.append(Msq_solver(M @ self._th()), [0])

    def get_theta(self) -> np.ndarray:
        return self.theta

    def get_n(self) -> np.ndarray:
        A, tpr = self.get_circuit().get_cycle_matrix(), 1.0 / (2.0 * np.pi)
        return - (A @ np.round(self._th() / (2.0 * np.pi))).astype(int)

    def get_I(self) -> np.ndarray:
        return self.problem._cp(self.get_circuit()._Ic(), self._th())

    def get_J(self) -> np.ndarray:
        A, Asq_solver = self.get_circuit().get_cycle_matrix(), self.get_problem()._Asq_factorization()
        return Asq_solver(A @ self.get_I())

    def get_flux(self) -> np.ndarray:
        A, Nj = self.get_circuit().get_cycle_matrix(), self.get_circuit()._Nj()
        return self.problem.frustration + A @ self.get_circuit()._L().matrix(Nj) @ self.get_I() / (2 * np.pi)

    def get_EM(self) -> np.ndarray:
        Nj = self.get_circuit()._Nj()
        return 0.5 * self.get_circuit()._L().matrix(Nj) @ (self.get_I() ** 2)

    def get_EJ(self) -> np.ndarray:
        return self.problem._icp(self.get_circuit()._Ic(), self._th())

    def get_Etot(self) -> np.ndarray:
        return self.get_EJ() + self.get_EM()

    def satisfies_kirchhoff_rules(self, tol=DEF_TOL):
        return self.get_error_kirchhoff_rules() < tol

    def satisfies_winding_rules(self, tol=DEF_TOL):
        return self.get_error_winding_rules() < tol

    def satisfies_target_vortices(self):
        return np.all(self.get_n() == self.problem.get_vortex_configuration())

    def is_stable(self, maxiter=DEF_STAB_MAXITER) -> bool:
        """
        Determines if a configuration is dynamically stable by determining if the Jacobian matrix
        of the time-evolution at the stationairy point is negative definite.

        This is done using the scipy implementation of the lobpcg method.
         - This iterative method converges to the largest eigenvalue (from below).
         - It is prematurely cut-off if the eigenvalue l at iteration i:
           * l(i) > 0 (hopeless)
           * l(i) + accept_ratio * tolerance(i) < 0 ("judgement call" lambda will never exceed 0)

        The last can be rephrased as that it stops when the ratio -l(i)/tolerance(i) exceeds accept_tol
        and is "accepted".

        Inputs:
        -------
        maxiter=DEF_STAB_MAXITER        maximum number of iterations to determine if solutions are stable
        accept_ratio=DEF_STAB_RATIO     residual-to-max-eigenvalue ratio above which is decided max-eigv
                                        must be negative.
        Outputs:
        --------
        is_stable               True if configuration is dynamically stable
        stable_iter_info        StableIterInfo object containing numerical info about the lobpcg iteration
        """
        cp = self.get_problem().get_current_phase_relation()
        out_is_stable = is_stable(self.get_circuit(), self._th(), cp, maxiter=maxiter)
        return out_is_stable

    def is_solution(self, tol=DEF_TOL):
        return self.satisfies_kirchhoff_rules(tol) & self.satisfies_winding_rules(tol)

    def is_target_solution(self, tol=DEF_TOL):
        return self.is_solution(tol=tol) & self.satisfies_target_vortices()

    def is_stable_target_solution(self, tol=DEF_TOL, stable_maxiter=DEF_STAB_MAXITER):
        return self.is_target_solution(tol=tol) & self.is_stable(maxiter=stable_maxiter)

    def get_error_kirchhoff_rules(self) -> np.ndarray:
        return get_kirchhoff_error(self.get_circuit(), self.get_I(), self.get_problem()._Is(),
                                   precomputed_Is_norm=self.problem._Is_norm())

    def get_error_winding_rules(self) -> np.ndarray:
        circuit, problem = self.get_circuit(), self.get_problem()
        f, Asq_factorized = problem._f(),  problem._Asq_factorization()
        L = circuit._L().matrix(self.get_circuit()._Nj())
        return get_winding_error(circuit, self._th() + L @ self.get_I(), get_g(circuit, f, 0, Asq_solver=Asq_factorized))

    def get_error(self):
        return self.get_error_kirchhoff_rules(), self.get_error_winding_rules()

    def plot(self, show_vortices=True, vortex_diameter=0.25, vortex_color=(0, 0, 0),
             anti_vortex_color=(0.8, 0.1, 0.2), vortex_alpha=1, show_grid=True, grid_width=1,
             grid_color=(0.3, 0.5, 0.9), grid_alpha=0.5, show_colorbar=True, show_arrows=True,
             arrow_quantity="I", arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
             arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1, arrow_color=(0.2, 0.4, 0.7),
             arrow_alpha=1, show_nodes=True, node_diameter=0.2,
             node_face_color=(1, 1, 1), node_edge_color=(0, 0, 0), node_alpha=1, show_node_quantity=False,
             node_quantity="phase", node_quantity_cmap=None, node_quantity_clim=(0, 1), node_quantity_alpha=1,
             node_quantity_logarithmic_colors=False, show_face_quantity=False, face_quantity="n",
             face_quantity_cmap=None, face_quantity_clim=(0, 1), face_quantity_alpha=1,
             face_quantity_logarithmic_colors=False,
             figsize=None, title="", **kwargs):
        """
        See CircuitPlot for documentation.
        """
        from pyJJAsim.circuit_visualize import CircuitPlot

        return CircuitPlot(self, show_vortices=show_vortices, vortex_diameter=vortex_diameter,
                         vortex_color=vortex_color, anti_vortex_color=anti_vortex_color,
                         vortex_alpha=vortex_alpha, show_grid=show_grid, grid_width=grid_width,
                         grid_color=grid_color, grid_alpha=grid_alpha, show_colorbar=show_colorbar,
                         show_arrows=show_arrows,
                         arrow_quantity=arrow_quantity, arrow_width=arrow_width, arrow_scale=arrow_scale,
                         arrow_headwidth=arrow_headwidth, arrow_headlength=arrow_headlength,
                         arrow_headaxislength=arrow_headaxislength, arrow_minshaft=arrow_minshaft,
                         arrow_minlength=arrow_minlength, arrow_color=arrow_color,
                         arrow_alpha=arrow_alpha,
                         show_nodes=show_nodes, node_diameter=node_diameter,
                         node_face_color=node_face_color, node_edge_color=node_edge_color,
                         node_alpha=node_alpha, show_node_quantity=show_node_quantity,
                         node_quantity=node_quantity, node_quantity_cmap=node_quantity_cmap,
                         node_quantity_clim=node_quantity_clim, node_quantity_alpha=node_quantity_alpha,
                         node_quantity_logarithmic_colors=node_quantity_logarithmic_colors,
                         show_face_quantity=show_face_quantity, face_quantity=face_quantity,
                         face_quantity_cmap=face_quantity_cmap, face_quantity_clim=face_quantity_clim,
                         face_quantity_alpha=face_quantity_alpha,
                         face_quantity_logarithmic_colors=face_quantity_logarithmic_colors,
                         figsize=figsize, title=title, **kwargs).make()

    def report(self):
        print("Kirchhoff rules error:    ", self.get_error_kirchhoff_rules())
        print("Path rules error:         ", self.get_error_winding_rules())
        print("is stable:                ", self.is_stable())
        print("is target vortex solution:", self.satisfies_target_vortices())

    def _th(self):
        return self.theta


"""
UTILITY ALGORITHMS
"""

def get_kirchhoff_error(circuit: Circuit, I, Is, precomputed_Is_norm=None):
    # Residual of kirchhoffs current law: M @ (I - Is) = 0. Normalized; so between 0 and 1.
    if precomputed_Is_norm is None:
        precomputed_Is_norm = scipy.linalg.norm(Is)
    b = circuit.get_cut_matrix() @ (I - Is)
    M_norm = circuit.get_M_norm()
    normalizer = M_norm * (precomputed_Is_norm + scipy.linalg.norm(I))
    return np.finfo(float).eps if np.abs(normalizer) < 1E-20 else scipy.linalg.norm(b) / normalizer

def get_winding_error(circuit: Circuit, th_p, g):
    # Residual of winding rule: A @ (thp - g) = 0. Normalized; so between 0 and 1. (where thp = th + L @ I)
    A = circuit.get_cycle_matrix()
    A_norm = circuit.get_A_norm()
    normalizer = A_norm * (scipy.linalg.norm(th_p) + scipy.linalg.norm(g))
    return np.finfo(float).eps if np.abs(normalizer) < 1E-20 else scipy.linalg.norm(A @ (th_p - g)) / normalizer

def principle_value(theta):
    return theta - 2 * np.pi * np.round(theta / (2 * np.pi))

def get_g(circuit: Circuit, f=0, z=0, Asq_solver=None):
    # g vector obeying A @ g = 2 * pi * (z - f)
    A, Nf = circuit.get_cycle_matrix(), circuit._Nf()
    if Asq_solver is None:
        Asq_solver = scipy.sparse.linalg.factorized(A @ A.T)
    return 2 * np.pi * A.T @ Asq_solver(np.broadcast_to(z - f, (Nf,)))

def change_phase_zone(circuit: Circuit, theta, z_old, z_new):
    # adds multiples of 2*pi to theta such that it obeys A @ (th_new + L @ I) = 2 * pi * (z_new - f)
    # (assuming it already satisfied A @ (th_old + L @ I) = 2 * pi * (z_old- f))
    return theta + circuit._A_solve(np.broadcast_to(z_new - z_old, (circuit._Nf(),)).copy()) * 2.0 * np.pi

def node_to_junction_current(circuit: Circuit, node_current):
    Mr = circuit._Mr().A
    return -Mr.T @ scipy.sparse.linalg.spsolve(Mr @ Mr.T, node_current[:-1])

"""
PARAMETER MAXIMIZATION ALGORITHMS
"""

def compute_maximal_parameter(circuit: Circuit, Is_func, f_func, z, n, cp=DefaultCPR(),
                              theta_0=None, lambda_tol=DEF_MAX_PAR_TOL, lambda_initial_stepsize=1.0,
                              stepsize_reduction_factor=DEF_MAX_PAR_REDUCE_FACT, newton_tol=DEF_TOL,
                              newton_maxiter=DEF_NEWTON_MAXITER, newton_stop_as_residual_increases=True,
                              newton_stop_if_not_target_n=False, require_stability=True,
                              stable_maxiter=DEF_STAB_MAXITER, Asq_factorized=None):

    """
    Core algorithm for parameter optimization.

    Stand-alone method. The wrappers StaticProblem and StaticConfiguration are more convenient.

    Finds the largest value of lambda for which a problem which current_sources=Is_func(lambda)
    and frustration=f_func(lambda) has a stable stationary state.

     - Must be able to find a stable configuration at lambda=0.
     - One can manually specify an initial_guess for lambda=0.
     - returns a lower- and upperbound for lambda. Stops when the difference < lambda_tol * lower_bound
     - furthermore returns config containing the solutions at the lower_bound. Also its
       accompanied problem has f and Is of lower_bound.
     - ignores self.current_sources and self.frustration.
     - Also returns ParameterOptimizeInfo object containing information about the iteration.
     - Algorithm stops if lambda_tol is reached or when newton_iteration failed to converge or diverge.

    Input:
    ------
    circuit                                   Circuit object
    Is_func                                 Is_func(lambda) -> Is
    f_func                                  f_func(lambda) -> f
    z                                       (Nf,) array containing phase zone
    n                                       (Nf,) array containing vortex configuration
    cp=DefaultCPR()        current-phase relation
    theta_0=None                            manual approximation at lambda=0
                                            or None; then London approximation is used.
    lambda_tol=DEF_MAX_PAR_TOL              stop iterating if upperbound - lowerbound < lambda_tol * lower_bound
    lambda_initial_stepsize=1.0             Estimate for the upperbound for lambda used as initial stepsize

    (for newton iteration at given lambda, see .compute() for documentation)
    newton_tol=DEF_TOL
    max_total_newton_steps=DEF_MAX_PAR_NEWT_STEPS
    newton_stop_as_residual_increases=True
    newton_stop_if_not_target_n=False

    (for determining stability of solution at given lambda, see StaticConfiguration.is_stable())
    require_stability=True,
    stable_maxiter=DEF_STAB_MAXITER
    Asq_factorized                      Solver for A @ A.T @ x == b. If None, set equal to
                                        scipy.sparse.linalg.factorized(A @ A.T)
    Output:
    -------
    lambda_lowerbound       float                   Lowerbound of lambda
    lambda_upperbound       float                   Upperbound of lambda
    theta0                  ndarray                 solution at lambda=lambda_lowerbound
    iteration_info          ParameterOptimizeInfo   Object containing information about the iteration.
    """

    # prepare info handle
    info = ParameterOptimizeInfo(Is_func, f_func, lambda_tol, circuit.get_cut_matrix())

    # prepare matrices
    if Asq_factorized is None:
        A = circuit.get_cycle_matrix()
        Asq_factorized = scipy.sparse.linalg.factorized(A @ A.T)

    # determine solution at lambda=0
    Is, f = Is_func(0), f_func(0)
    if theta_0 is None:
        theta_0 = london_approximation(circuit, f, n)
        theta_0 = change_phase_zone(circuit, theta_0, n, z)
    out = static_compute(circuit, theta_0, Is, f, n, z, cp, tol=newton_tol,
                         maxiter=newton_maxiter, Asq_solver=Asq_factorized,
                         stop_as_residual_increases=newton_stop_as_residual_increases,
                         stop_if_not_target_n=newton_stop_if_not_target_n)
    theta, status, newton_iter_info = out[0], out[1], out[2]
    is_solution = newton_iter_info.found_target_solution()

    if is_solution and require_stability:
        is_solution &= is_stable(circuit, theta, cp, maxiter=stable_maxiter)

    info._preset(is_solution)

    # return if no solution at lambda=0
    if not is_solution:
        return None, None, None, info

    # prepare iteration to find maximum lambda
    found_upper_bound = False
    lambda_stepsize = lambda_initial_stepsize
    lambda_val = lambda_stepsize
    theta0 = theta

    # start iteration to find maximum lambda
    iter_nr = 0
    while True:

        # determine solution at current lambda
        Is, f = Is_func(lambda_val), f_func(lambda_val)
        out = static_compute(circuit, theta_0, Is, f, n, z, cp, tol=newton_tol,
                             maxiter=newton_maxiter, Asq_solver=Asq_factorized,
                             stop_as_residual_increases=newton_stop_as_residual_increases,
                             stop_if_not_target_n=newton_stop_if_not_target_n)
        theta, status, newton_iter_info = out[0], out[1],  out[2]
        is_solution = newton_iter_info.found_target_solution()
        if status == 2:
            break

        if require_stability and is_solution:
            is_solution &= is_stable(circuit, theta, cp, maxiter=stable_maxiter)

        # update information on current iteration in info handle
        info._set(lambda_val, lambda_stepsize, is_solution, newton_iter_info)

        # determine new lambda value to try (and corresponding initial condition)
        if is_solution:
            lambda_val += lambda_stepsize
            theta0 = theta
        else:
            lambda_val -= lambda_stepsize * (1 - stepsize_reduction_factor)
            lambda_stepsize*=stepsize_reduction_factor
            found_upper_bound = True
        if (lambda_stepsize / lambda_val) < lambda_tol:
            break
        iter_nr += 1

    # determine lower- and upperbound on lambda
    info._finish(status)
    lower_bound = lambda_val - lambda_stepsize
    upper_bound = lambda_val if found_upper_bound else np.inf

    return lower_bound, upper_bound, theta0, info


"""
APPROXIMATE STATE FINDING ALGORITHMS
"""

def london_approximation(circuit: Circuit, f, n, AIpLIcA_solver=None):
    A, Nf = circuit.get_cycle_matrix(),  circuit._Nf()
    if AIpLIcA_solver is None:
        Nj = circuit._Nj()
        L, Ic = circuit._L().matrix(Nj), circuit._Ic().matrix(Nj)
        AIpLIcA_solver = scipy.sparse.linalg.factorized(A @ (scipy.sparse.eye(Nj) + L @ Ic) @ A.T)
    return 2 * np.pi * A.T @ AIpLIcA_solver(np.broadcast_to(n - f, (Nf,)))

def arctan_approximation(circuit: Circuit, f, n, Asq_solver=None, IpLIc_solver=None):
    centr_x, centr_y = circuit.get_face_centroids()
    return arctan_approximation_placed_vortices(circuit, f, n[n != 0], centr_x[n != 0], centr_y[n != 0],
                                                Asq_solver=Asq_solver, IpLIc_solver=IpLIc_solver)

def arctan_approximation_placed_vortices(circuit: Circuit, f, n, x_n, y_n, Asq_solver=None, IpLIc_solver=None):
    n = np.atleast_1d(n)
    x_n = np.atleast_1d(x_n)
    y_n = np.atleast_1d(y_n)
    MT = circuit.get_cut_matrix().T
    if IpLIc_solver is None:
        Nj = circuit._Nj()
        L, Ic = circuit._L().matrix(Nj), circuit._Ic().matrix(Nj)
        IpLIc_solver = scipy.sparse.linalg.factorized(scipy.sparse.eye(Nj) + L @ Ic)
    x, y = circuit.get_node_coordinates()
    MTphi = MT @ np.sum(np.arctan2(y - y_n[:, None], x - x_n[:, None]) * n[:, None], axis=0)
    out = principle_value(MTphi) + get_g(circuit, f=f, z=0, Asq_solver=Asq_solver)
    return IpLIc_solver(out) + 2 * np.pi * np.round(MTphi / (2 * np.pi))


"""
STATIONAIRY STATE FINDING ALGORITHMS
"""


def static_compute(circuit: Circuit, theta0=0, Is=0, f=0, n=0, z=0,
                   cp=DefaultCPR(), tol=DEF_TOL,
                   maxiter=DEF_NEWTON_MAXITER, Asq_solver=None,
                   stop_as_residual_increases=True, stop_if_not_target_n=False):
    """
    Core algorithm computing stationary state of a Josephson Junction Circuit using Newtons method.

    Stand-alone method. The wrappers StaticProblem and StaticConfiguration are more convenient.

    Input:
    ------
    circuit                               Circuit                   josephson junction circuit
    theta0                              (Nj,) ndarray           initial guess
    Is=0                                (Nj,) ndarray           Current sources at each junction
    f=0                                 (Nf,) ndarray           Frustration in each face
    n=0                                 (Nf,) ndarray           number of vortices in each face
    z=0                                 (Nf,) ndarray           phase zone of each face
    cp=DefaultCPR()    CurrentPhaseRelation    current phase relation
    tol=DEF_TOL                         scalar                  tolerance. is solution if |residual| < tol
    max_iter=100                        scalar                  maximum number of newton iterations.
    Asq_solver=None                     func: vector->vector    Solver for A @ A.T @ x == b. If None, set equal to
                                                                scipy.sparse.linalg.factorized(A @ A.T)
    stop_as_residual_increases=True     bool                    iteration stops if error(iter) > error(iter - 3)
    stop_if_not_target_n=False          bool                    iteration stops if n != target_n

    Output:
    -------
    theta                              (Nj,) ndarray            gauge invariant phase difference of solution
    convergence_status                 int                      0 -> converged
                                                                1 -> diverged
                                                                2 -> max_iter reached without converging or diverging.
    info                               NewtonIterInfo           Information about iteration (timing, steps, residuals, etc)


    Extra information:
    ------------------
    Stops iterating if:                         convergence_status
     - residual is smaller than tol             0 (converged) if get_n(theta) == n
                                                1 (diverged)  if get_n(theta) != n
     - iteration number iter exceeds maxiter    2 (indeterminate)
     - residual exceeds 0.5                     1 (diverged)
     - if get_n(theta) != n and                 1 (diverged)
       stop_if_not_target_n==True
     - resid(iter) > resid(iter-3) and          1 (diverged)
       stop_as_residual_increases==True

    """

    # prepare newton iter info
    info = NewtonIterInfo(tol, maxiter)

    # get circuit quantities and matrices
    Nj, Nf, M, A = circuit._Nj(), circuit._Nf(), circuit.get_cut_matrix()[:-1, :], circuit.get_cycle_matrix()
    L = circuit._L().matrix(Nj)
    Ic = np.broadcast_to(circuit.get_critical_current_factors(), (Nj,))

    Is = np.ones((Nj,), dtype=np.double) * Is if np.array(Is).size == 1 else Is
    f = np.ones((Nf,), dtype=np.double) * f if np.array(f).size == 1 else f
    n = np.ones((Nf,), dtype=int) * n if np.array(n).size == 1 else n

    # iteration-0 computations
    if Asq_solver is None:
        Asq_solver = scipy.sparse.linalg.factorized(A @ A.T)
    g = get_g(circuit, f=f, z=z, Asq_solver=Asq_solver)

    theta = theta0.copy()
    I = cp.eval(Ic, theta)
    Is_norm = scipy.linalg.norm(Is)

    # iteration-0 compute errors
    error1 = get_kirchhoff_error(circuit, I, Is,  precomputed_Is_norm=Is_norm)
    error2 = get_winding_error(circuit, theta + L @ I, g)

    error = max(error1, error2)
    is_target_n = np.all(A @ (np.round(theta / (2 * np.pi))).astype(int) == z - n)
    info._set(error, is_target_n)

    # prepare newton iteration
    prev_error = np.inf
    iteration = 0


    while not (error < tol or (error > 0.5 and iteration > 5) or (stop_if_not_target_n and is_target_n) or
               (stop_as_residual_increases and error > prev_error) or iteration >= maxiter):
        # iteration computations

        q = cp.d_eval(Ic, theta)
        J = scipy.sparse.vstack([M @ scipy.sparse.diags(q, 0), A @ (scipy.sparse.eye(Nj) + L @ scipy.sparse.diags(q, 0))]).tocsc()
        J_solver = scipy.sparse.linalg.factorized(J)
        F = np.concatenate([M @ (I - Is), A @  (theta - g + L @ I)])
        theta -= J_solver(F)
        I = cp.eval(Ic, theta)

        # iteration error computations
        error1 = get_kirchhoff_error(circuit, I, Is, precomputed_Is_norm=Is_norm)
        error2 = get_winding_error(circuit, theta + L @ I, g)
        error = max(error1, error2)
        is_target_n = np.all(A @ (np.round(theta / (2 * np.pi))).astype(int) == z - n)
        info._set(error, is_target_n)
        if iteration >= 3:
            prev_error = info.error[iteration - 3]

        iteration += 1

    return theta, info.get_status(), info

"""
STABILITY ALGORITHMS
"""

def is_stable(circuit: Circuit, theta, cp, maxiter=DEF_STAB_MAXITER):
    """
    Determines if a configuration on a circuit with inductance is stable in the sense that
    the Jacobian is negative definite. Does not explicitly check if configuration is a stationairy point.
    """
    Nj, Nnr = circuit._Nj(), circuit._Nnr()
    A, M, L = circuit.get_cycle_matrix(),  circuit._Mr().A, circuit._L().matrix(Nj)
    Ic = circuit._Ic().diag(force_as_vector=True, vector_length=Nj)
    q = cp.d_eval(Ic, theta)
    AL = (A @ L @ A.T).tocoo()
    ALL = scipy.sparse.coo_matrix((AL.data, (AL.row + Nnr, AL.col + Nnr)), shape=(Nj, Nj)).tocsc()
    m = scipy.sparse.vstack([M, A @ L]).tocsc()
    J = - (m @ scipy.sparse.diags(q, 0) @ m.T + ALL)
    select = np.diff(J.indptr)!=0
    J = J[select, :][:, select]
    try:
        w, v = scipy.sparse.linalg.eigsh(J, 1, maxiter=maxiter, which="LA")
        out_is_stable = w[0] < 20 * np.finfo(float).eps
    except ArpackNoConvergence:
        print("warning: eigsh ran out of steps")
        out_is_stable = False
    return out_is_stable


