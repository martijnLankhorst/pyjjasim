
import time

import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.optimize
from scipy.sparse.linalg import ArpackNoConvergence

from pyjjasim.embedded_graph import EmbeddedGraph
from pyjjasim.josephson_circuit import Circuit

__all__ = ["CurrentPhaseRelation", "DefaultCPR", "StaticProblem",
           "StaticConfiguration", "compute_maximal_parameter",
           "node_to_junction_current", "DEF_TOL", "DEF_NEWTON_MAXITER",
           "DEF_STAB_MAXITER", "DEF_MAX_PAR_TOL", "DEF_MAX_PAR_REDUCE_FACT",
           "NewtonIterInfo", "ParameterOptimizeInfo", "stability_get_preconditioner"]


"""
Static Problem Module
"""

DEF_TOL = 1E-10

DEF_NEWTON_MAXITER = 30

DEF_STAB_MAXITER = 2000

DEF_MAX_PAR_TOL = 1E-4
DEF_MAX_PAR_REDUCE_FACT = 0.42
DEF_MAX_PAR_MAXITER = 100


class CurrentPhaseRelation:

    """

    Current-Phase relation Icp(Ic, theta). The default value is Icp = Ic * sin(theta).

    Parameters
    ----------
    func : func(Ic, theta)
        Current-phase relation.
    d_func : func(Ic, theta)
        Derivative of current-phase relation to theta.
    i_func : func(Ic, theta)
        Integral of current-phase relation over theta (starting at 0).

    Notes
    -----
     - func, d_func and i_func must be numpy ufunc, so their output must be broadcast
       of input Ic and theta.
    """
    def __init__(self, func, d_func, i_func):
        self.func = func
        self.d_func = d_func
        self.i_func = i_func

    def eval(self, Ic, theta):
        """
        Evaluate current phase relation; returns func(Ic, theta).
        """
        return self.func(Ic, theta)

    def d_eval(self, Ic, theta):
        """
        Evaluate derivative of current phase relation; returns d_func(Ic, theta).
        """
        return self.d_func(Ic, theta)

    def i_eval(self, Ic, theta):
        """
        Evaluate integral of current phase relation; returns i_func(Ic, theta).
        """
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
        """
        Returns number of iterations after which iteration is aborted.
        """
        return self.maxiter

    def get_tol(self):
        """
        Returns tolerance.
        """
        return self.tol

    def get_status(self):
        """
        Returns status of newton iteration result; returns value:
         * 0: converged. residual < tolerance
         * 1: diverged before reaching maxiter.
         * 2: reached max_iter without converging or diverging.
        """
        return int(not self.found_target_solution()) + 2 * int(self.iteration >= self.maxiter)

    def has_converged(self):
        """
        Returns if iteration has converged.
        """
        return self.has_converged

    def get_is_target_vortex_configuration(self):
        """
        Returns (nr_of_iters,) bool array if vortex configuration at iter
        agrees with vortex configuration specified in problem.
        """
        return self.is_target_n[:(self.get_number_of_iterations()+1)]

    def found_target_solution(self):
        """
        Returns True if has_converged() and final iter obeys target vortex config.
        """
        return self.has_converged and self.is_target_n[self._get_iteration()]

    def get_number_of_iterations(self):
        """
        Returns number of newton iterations done.
        """
        return self._get_iteration()

    def get_residual(self):
        """
        Returns (nr_of_iters,) array containing residual at each iteration.
        """
        return self.error[:(self.get_number_of_iterations()+1)]

    def get_runtime(self):
        """
        Returns runtime in seconds.
        """
        return self.runtime

    def plot_residuals(self):
        """
        Plots residual vs iteration number.
        """
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
    Use print(parameter_optimize_info) to display a summary of
    the information.
    """
    def __init__(self, problem_func, lambda_tol, require_stability, require_target_n, maxiter):
        self.problem_func = problem_func
        self.lambda_tol = lambda_tol
        self.require_target_n = require_target_n
        self.require_stability = require_stability
        self.maxiter = maxiter
        self.has_solution_at_zero = False
        self.lambda_history = np.zeros(self.maxiter, dtype=np.double)
        self.solutions = []
        self.stepsize_history = np.zeros(self.maxiter, dtype=np.double)
        self.solution_history = np.zeros(self.maxiter, dtype=np.bool)
        self.stable_history = np.zeros(self.maxiter, dtype=np.int)
        self.target_n_history = np.zeros(self.maxiter, dtype=np.int)
        self.newton_iter_infos = []
        self._step = 0
        self._time = time.perf_counter()
        self.last_step_status = None
        self.last_step_stable_status = None

    def get_has_solution_at_zero(self):
        """
        Returns if a stable target solution is found at lambda=0.
        """
        return self.has_solution_at_zero

    def get_lambda(self):
        """
        Returns (nr_of_steps,) array with lambda at each step.
        """
        return self.lambda_history[:self._step]

    def get_lambda_error(self):
        """
        Returns (nr_of_steps,) array with error in lambda.
        """
        return self._get_lambda_stepsize() / self.get_lambda()

    def get_lambda_lower_bound(self):
        """
        Returns lower bound for lambda.
        """
        if not self.get_has_solution_at_zero():
            return np.nan
        s = self.get_lambda()[self.get_found_solution()]
        return s[-1] if s.size > 0 else 0

    def get_lambda_upper_bound(self):
        """
        Returns upper bound for lambda.
        """
        s = self.get_lambda()[~self.get_found_solution()]
        return s[-1] if s.size > 0 else np.inf

    def get_found_solution(self):
        """
        Returns (nr_of_steps,) array if a stable target solution is found at step.
        """
        return self.solution_history[:self._step]

    def get_is_stable(self):
        """
        Returns (nr_of_steps,) array if a stable target solution is found at step.
        """
        return self.stable_history[:self._step]

    def get_is_target_vortex_configuration(self):
        """
        Returns (nr_of_steps,) array if a solution has target vortex configuration.
        """
        return self.target_n_history[:self._step]

    def get_newton_iter_all_info(self):
        """
        Returns (nr_of_steps,) list containing newton_iter_infos.
        """
        return self.newton_iter_infos

    def get_newton_steps(self):
        """
        Returns (nr_of_steps,) array with nr of newton iterations at step.
        """
        return np.array([info.get_number_of_iterations() for info in self.newton_iter_infos], dtype=int)

    def get_runtime(self):
        """
        Returns runtime in seconds.
        """
        return self._time

    def plot_residuals(self):
        """
        Plots residual vs iteration number.
        """
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

    def animate_solutions(self):
        import matplotlib.animation as anim
        fig, ax = self.solutions[0].plot()
        lambdas = self.get_lambda()[self.get_found_solution()]
        stable = self.get_is_stable()[self.get_found_solution()]
        def _animate(i):
            p_fig, p_ax = self.solutions[i].plot(fig=fig)
            p_ax.set_title(f"lambda={np.round(lambdas[i], 5)}, is stable: {stable[i]}")
            return [p_ax]
        ani = anim.FuncAnimation(fig, _animate, frames=range(len(self.solutions)),
                                 interval=1000, blit=False)
        return ani

    def __str__(self):
        np.set_printoptions(linewidth=100000)
        out = "Parameter optimize info:\n\t"
        if not self.get_has_solution_at_zero():
            out += "Optimization failed because not solution was found at lambda=0."
        else:
            def int_digit_count(x):
                return np.ceil(np.log(np.max(x)) / np.log(10)).astype(int)
            n = max(5, 3 + int_digit_count(1/self.lambda_tol), int_digit_count(self.get_newton_steps()))
            out += f"Found lambda between {self.get_lambda_lower_bound()} and {self.get_lambda_upper_bound()}.\n\t"
            if self.last_step_stable_status == 2:
                out += f"Stopped because stability could not be determined. Consider increasing stable_maxiter " \
                       f"or changing stability algorithm.)\n\t"
            elif self.last_step_status == 2:
                out += f"Stopped because newton iteration was indeterminate. Consider increasing newton_maxiter.)\n\t"
            elif self._step == self.maxiter:
                out += f"Optimization reached maxiter {self.maxiter} before reaching desired tolerance. (resid={self.get_lambda_error()[-1]})\n\t"
            else:
                out += f" at desired tolerance (resid={self.get_lambda_error()[-1]}) \n\t"
            out += f"runtime: {np.round(self.get_runtime(), 7)} sec\n\t"
            np.set_printoptions(formatter={'float': lambda x: ("{0:0." + str(n - 2) + "f}").format(x)})
            out += f"lambda:              {self.get_lambda()}\n\t"
            np.set_printoptions(formatter={'bool': lambda x: ("{:>" + str(n) + "}").format(x)})
            out += f"found solution:      {self.get_found_solution().astype(bool)}\n\t"
            if self.require_target_n:
                out += f"if so; has target n: {self.get_is_target_vortex_configuration().astype(bool)}\n\t"
            if self.require_stability:
                out += f"is so; is stable:    {self.get_is_stable().astype(bool)}\n\t"
            np.set_printoptions(formatter={'int': lambda x: ("{:>" + str(n) + "}").format(x)})
            out += f"newton step count:   {self.get_newton_steps()}\n\t"
        return out

    def _preset(self, has_solution_at_zero):
        self.has_solution_at_zero = has_solution_at_zero
        return self

    def _set(self, lambda_value, solution, lambda_stepsize, found_solution, newton_iter_info, is_target_n, is_stable=1):
        self.lambda_history[self._step] = lambda_value
        self.stepsize_history[self._step] = lambda_stepsize
        self.solution_history[self._step] = found_solution
        self.stable_history[self._step] = is_stable
        self.newton_iter_infos += [newton_iter_info]
        if solution is not None:
            self.solutions += [solution]
        if is_target_n is not None:
            self.target_n_history[self._step] = is_target_n
        self._step += 1
        return self

    def _finish(self, last_step_status, last_step_stable_status):
        self.last_step_status = last_step_status
        self.last_step_stable_status = last_step_stable_status
        self._time = time.perf_counter() - self._time
        return self

    def _get_lambda_stepsize(self):
        return self.stepsize_history[:self._step]


class StaticProblem:
    """
    Define a static josephson junction array problem.

    Parameters
    ----------
    circuit : Circuit
         Circuit on which the problem is based.
    current_sources=0.0 : (Nj,) ndarray or scalar
         Current sources at each junction in circuit (abbreviated Is). If scalar the same
         value is used for all junctions.

    frustration=0.0 : (Nf,) ndarray or scalar
         frustration, or normalized external magnetic flux, through each face in circuit
         (abbreviated f). If scalar the same value is used for all faces.
    vortex_configuration=0 : (Nf,) ndarray or scalar
         Target vorticity at each face in circuit (abbreviated n).  If scalar the same value is
         used for all faces.
    current_phase_relation=DefaultCPR() : CurrentPhaseRelation
        Current-phase relation used to do computations on problem.

     Notes
     -----
     - All physical quantities are dimensionless. See the UserManual (on github)
       for how all quantities are normalized.
     - It is assumed each junction has a current source, see user manual
       (on github) for diagram of junction. To omit the sources in particular
       junctions set the respective values to zero.
     - To use a node-based souce current (represented as an (Nn,) array Is_node
       with current in/e-jected at each node), convert it to a junction-based
       source with Is = node_to_junction_current(circuit, Is_node) and
       us Is as input for a static problem.

    """

    def __init__(self, circuit: Circuit, current_sources=0.0, frustration=0.0,
                 vortex_configuration=0, current_phase_relation=DefaultCPR()):
        self.circuit = circuit
        self.current_sources = np.atleast_1d(current_sources)
        self.frustration = np.atleast_1d(frustration)
        self.vortex_configuration = np.atleast_1d(vortex_configuration)
        self.current_phase_relation = current_phase_relation
        self.current_sources_norm = None
        # self.Asq_factorization = None
        # self.AIpLIcA_factorization = None
        # self.IpLIc_factorization = None
        # self.Msq_factorization = None

    def save(self, filename):
        """
        Store problem in .npy file. Note that the current-phase-relation is not stored!
        """
        with open(filename, "wb") as ffile:
            x, y = self.circuit.graph.coo()
            n1, n2 = self.circuit.graph.get_edges()
            np.save(ffile, x)
            np.save(ffile, y)
            np.save(ffile, n1)
            np.save(ffile, n2)
            np.save(ffile, self.circuit.critical_current_factors)
            np.save(ffile, self.circuit.resistance_factors)
            np.save(ffile, self.circuit.capacitance_factors)
            L_is_sparse = scipy.sparse.issparse(self.circuit.inductance_factors)
            np.save(ffile, L_is_sparse)
            if L_is_sparse:
                np.save(ffile, self.circuit.inductance_factors.indptr)
                np.save(ffile, self.circuit.inductance_factors.indices)
                np.save(ffile, self.circuit.inductance_factors.data)
            else:
                np.save(ffile, self.circuit.inductance_factors)
            np.save(ffile, self.current_sources)
            np.save(ffile, self.frustration)
            np.save(ffile, self.vortex_configuration)

    @staticmethod
    def load(filename):
        """
        Load problems created with the .save(filename) method. Returns StaticProblem.
        Note that the loaded problem will always have the default current-phase-relation.
        """
        with open(filename, "rb") as ffile:
            x = np.load(ffile)
            y = np.load(ffile)
            node1 = np.load(ffile)
            node2 = np.load(ffile)
            g = EmbeddedGraph(x, y, node1, node2)
            Ic = np.load(ffile)
            R = np.load(ffile)
            C = np.load(ffile)
            L_is_sparse = np.load(ffile)
            if L_is_sparse:
                indptr = np.load(ffile)
                indices = np.load(ffile)
                data = np.load(ffile)
                Nj = len(node1)
                L = scipy.sparse.csc_matrix((data, indices, indptr), shape=(Nj, Nj))
            else:
                L = np.load(ffile)
            circuit = Circuit(g, critical_current_factors=Ic, resistance_factors=R,
                              capacitance_factors=C, inductance_factors=L)
            Is = np.load(ffile)
            f = np.load(ffile)
            n = np.load(ffile)
            return StaticProblem(circuit, current_sources=Is, frustration=f, vortex_configuration=n)

    def get_circuit(self) -> Circuit:
        """
        Returns the circuit.
        """
        return self.circuit

    def get_current_sources(self):
        """
        Returns the current sources (abbreviated Is).
        """
        return self.current_sources

    def get_frustration(self):
        """
        Returns the frustration (abbreviated f).
        """
        return self.frustration

    def get_vortex_configuration(self):
        """
        Returns the vortex configuration.
        """
        return self.vortex_configuration

    def get_current_phase_relation(self):
        """
        Returns the current-phase relation.
        """
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
        """
        Returns the phase zone (In all of pyJJAsim phase_zone=0).
        """
        return 0

    def get_net_sourced_current(self):
        """
        Gets the sum of all (positive) current injected at nodes to create Is.
        """
        M = self.get_circuit().get_cut_matrix()
        return 0.5 * np.sum(np.abs((M @ self._Is())), axis=0)

    def get_node_current_sources(self):
        """
        Returns (Nn,) array of currents injected at nodes to create Is.
        """
        M = self.get_circuit().get_cut_matrix()
        return M @ self.current_sources

    def approximate(self, algorithm=1):
        """
        Computes approximate solutions.

        Parameters
        ----------
        algorithm=1:
            Algorithm used in approximation. Can have values:
             * 0: Does arctan approximation. This assigns phases that "wind" 2*pi around
               vortices in z=0 phase zone, phi(x,y) = sum_i 2 * pi * n_i *
               atan2(y-y_n_i,x-x_n_i) where vortices are located at centres of their
               respective faces.
             * 1: London approximation. Find theta in cycle space (theta = A.T @ ...)
               that obeys winding rule.
        """
        if algorithm == 0:
            theta = arctan_approximation(self.circuit, self._f(), self._nt())
        elif algorithm == 1:
            theta = london_approximation(self.circuit, self._f(), self._nt())
            theta = change_phase_zone(self.get_circuit(), theta, self._nt(), 0)
        else:
            raise ValueError("invalid algorithm")
        return StaticConfiguration(self, theta)

    def approximate_placed_vortices(self, n, x_n, y_n):
        """
        Compute arctan approximation with manual placement of vortices.

        Parameters
        ----------
        n : (N,) int array
            Vorticity at location (x_n, y_n).
        x_n, y_n : (N,) float arrays
            The x,y-coordinates of vortices.
        """
        theta = arctan_approximation_placed_vortices(self.circuit, self._f(), n, x_n, y_n)
        return StaticConfiguration(self, theta)

    def compute(self, initial_guess = None, tol=DEF_TOL, maxiter=DEF_NEWTON_MAXITER,
                stop_as_residual_increases=True, stop_if_not_target_n=False, algorithm=1,
                use_pyamg=False):
        """
        Compute solution to static_problem using Newton iteration.

        Parameters
        ----------
        initial_guess=None : (Nj,) array, StaticConfiguration or None
            Guess for initial state. If None; uses London approximation. If input                 None (London approximation is used)
            is array; it must contain values of theta to represent state.

        tol=DEF_TOL : scalar
            Tolerance; is solution if |residual| < tol.
        maxiter=DEF_NEWTON_MAXITER : int
            Maximum number of newton iterations.
        stop_if_not_target_n=False : bool
            Iteration stops  if n(iter) != n (diverged)
        stop_as_residual_increases=True : bool
            Iteration stops if error(iter) > error(iter - 3) (diverged).

        Returns
        -------
        config : StaticConfiguration
            Object containing solution.
        status : int
            * 0: converged
            * 1: diverged if error(iter)>0.5 or above reasons.
            * 2: max_iter reached without converging or diverging.
        iter_info :  NewtonIterInfo
            Handle containing information about newton iteration.
        """
        if initial_guess is None:
            initial_guess = self.approximate(algorithm=1)

        if isinstance(initial_guess, StaticConfiguration):
            initial_guess = initial_guess._th()

        initial_guess = np.array(initial_guess, dtype=np.double)

        theta, status, iter_info = static_compute(self.get_circuit(), initial_guess, Is=self._Is(),
                                                  f=self._f(), n=self._nt(), z=0,
                                                  cp=self.current_phase_relation, tol=tol,
                                                  maxiter=maxiter,
                                                  stop_as_residual_increases=stop_as_residual_increases,
                                                  stop_if_not_target_n=stop_if_not_target_n,
                                                  algorithm=algorithm, use_pyamg=use_pyamg)
        config = StaticConfiguration(self, theta)
        return config, status, iter_info


    def compute_frustration_bounds(self, initial_guess = None,
                                   start_frustration=None, lambda_tol=DEF_MAX_PAR_TOL,
                                   maxiter=DEF_MAX_PAR_MAXITER, require_stability=True,
                                   require_vortex_configuration_equals_target=True,
                                   compute_parameters=None, stability_parameters=None):

        """


        Computes smallest and largest uniform frustration for which a (stable) solution
        exists at the specified target vortex configuration and source current.

        For unlisted parameters see documentation of compute_maximal_parameter()

        Parameters
        ----------
        start_frustration=None : valid frustration input for StaticProblem  or None.
            Frustration factor somewhere in the middle of range. If None; this is
            estimated based on vortex configuration.
        initial_guess=None : valid initial_guess input for StaticProblem.compute()
            Initial guess for the algorithm to start at frustration=start_frustration.

        Returns
        -------
        (smallest_f_factor, largest_f_factor) : (float, float)
            Resulting frustration range.
        (smallest_f_config, largest_f_config) : (StaticConfiguration, StaticConfiguration)
            StaticConfigurations at bounds of range.
        (smallest_f_info, largest_f_info) : (ParameterOptimizeInfo, ParameterOptimizeInfo)
             ParameterOptimizeInfo objects containing information about the iterations.
        """

        options = {"lambda_tol": lambda_tol, "maxiter": maxiter, "compute_parameters": compute_parameters,
                   "stability_parameters": stability_parameters, "require_stability": require_stability,
                   "require_vortex_configuration_equals_target": require_vortex_configuration_equals_target}


        if start_frustration is None:
            start_frustration = np.mean(self._nt())
        frustration_initial_stepsize = 1.0
        problem_small_func = lambda x: self.new_problem(frustration=start_frustration - x)
        problem_large_func = lambda x: self.new_problem(frustration=start_frustration + x)
        out = compute_maximal_parameter(problem_small_func, initial_guess=initial_guess,
                                        estimated_upper_bound=frustration_initial_stepsize, **options)
        smallest_factor, _, smallest_f_config, smallest_f_info = out
        smallest_f = start_frustration - smallest_factor if smallest_factor is not None else None
        out = compute_maximal_parameter(problem_large_func, initial_guess=initial_guess,
                                        estimated_upper_bound=frustration_initial_stepsize, **options)

        largest_factor, _, largest_f_config, largest_f_info = out
        largest_f = start_frustration + largest_factor if largest_factor is not None else None
        return (smallest_f, largest_f), (smallest_f_config, largest_f_config), (smallest_f_info, largest_f_info)

    def compute_maximal_current(self, initial_guess=None, lambda_tol=DEF_MAX_PAR_TOL,
                                maxiter=DEF_MAX_PAR_MAXITER, require_stability=True,
                                require_vortex_configuration_equals_target=True,
                                compute_parameters=None, stability_parameters=None):

        """
        Computes largest source current for which a stable solution exists at the
        specified target vortex configuration and frustration, where the  source
        current is assumed to be max_current_factor * self.get_current_sources().

        For parameters see documentation of compute_maximal_parameter()

        Returns
        -------
        max_current_factor : float
            Maximal current factor for which a problem with max_current_factor * Is
            has a (stable) solution.
        net_sources_current : float
            Net sourced current at max_current_factor.
        out_config : StaticConfiguration
            StaticConfiguration of state with maximal current.
        info : ParameterOptimizeInfo
            ParameterOptimizeInfo objects containing information about the iterations.

        """
        M, Nj = self.get_circuit()._Mr(), self.get_circuit()._Nj()
        if np.all(self._Is() == 0):
            raise ValueError("Problem must contain nonzero current sources.")
        Is_per_node = np.abs(M @ self._Is())
        max_super_I_per_node = np.abs(M) @ self.get_circuit()._Ic()
        current_factor_initial_stepsize = 1.0 / np.max(Is_per_node / max_super_I_per_node)
        problem_func = lambda x: self.new_problem(current_sources=x * self._Is())
        out = compute_maximal_parameter(problem_func, initial_guess=initial_guess,
                                        lambda_tol=lambda_tol, maxiter=maxiter,
                                        estimated_upper_bound=current_factor_initial_stepsize,
                                        compute_parameters=compute_parameters,
                                        stability_parameters=stability_parameters,
                                        require_stability=require_stability,
                                        require_vortex_configuration_equals_target=
                                        require_vortex_configuration_equals_target)
        max_current_factor, upper_bound, out_config, info = out
        net_I = out_config.get_problem().get_net_sourced_current() if out_config is not None else None
        return max_current_factor, net_I, out_config, info

    def compute_stable_region(self, angles=np.linspace(0, 2*np.pi, 61), start_frustration=None,
                              start_initial_guess=None, lambda_tol=DEF_MAX_PAR_TOL,
                              maxiter=DEF_MAX_PAR_MAXITER, require_stability=True,
                              require_vortex_configuration_equals_target=True,
                              compute_parameters=None, stability_parameters=None):

        """
        Finds edge of stable region in (f, Is) space for vortex configuration n.

        The frustration is assumed to be uniform. Ignores self.frustration and
        works with constant * self.current_sources.

        For unlisted parameters see documentation of compute_maximal_parameter()

        Parameters
        ----------
        angles=np.linspace(0, 2*np.pi, 61) : array
            Angles at which an extremum in (f, Is) space is searched for.

        Returns
        -------
        frustration : (num_angles,) array
            Net extermum frustration at each angle.
        net_current : (num_angles,) array
            Net extremum sourced current at each angle.
        all_configs : list containing StaticConfiguration
            Configurations at extreme value for each angle.
        all_infos : list containing ParameterOptimizeInfo
            Objects containing information about the iterations at each angle.

        """
        num_angles = len(angles)
        options = {"lambda_tol": lambda_tol, "maxiter": maxiter, "compute_parameters": compute_parameters,
                   "stability_parameters": stability_parameters, "require_stability": require_stability,
                   "require_vortex_configuration_equals_target": require_vortex_configuration_equals_target}

        frust_bnd_prb = self.new_problem(current_sources=0)
        out = frust_bnd_prb.compute_frustration_bounds(initial_guess=start_initial_guess,
                                                       start_frustration=start_frustration, **options)

        (smallest_f, largest_f), _, _ = out
        if smallest_f is None:
            return None, None, None, None
        dome_center_f = 0.5 * (smallest_f + largest_f)
        dome_center_problem = self.new_problem(frustration=dome_center_f)
        out = dome_center_problem.compute_maximal_current(initial_guess=start_initial_guess, **options)
        max_current_factor, _, _, info = out
        if max_current_factor is None:
            return None, None, None, None

        frustration = np.zeros(num_angles, dtype=np.double)
        net_current = np.zeros(num_angles, dtype=np.double)
        all_configs, all_infos = [], []
        for angle_nr in range(num_angles):
            angle = angles[angle_nr]
            Is_func = lambda x: x * self._Is() * np.sin(angle) * max_current_factor
            f_func = lambda x: dome_center_f + x * np.cos(angle) * (0.5 * (largest_f - smallest_f))
            problem_func = lambda x: self.new_problem(frustration=f_func(x), current_sources=Is_func(x))
            out = compute_maximal_parameter(problem_func, initial_guess=start_initial_guess, **options)
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

    def _cp(self, Ic, theta):
        return self.current_phase_relation.eval(Ic, theta)

    def _dcp(self, Ic, theta):
        return self.current_phase_relation.d_eval(Ic, theta)

    def _icp(self, Ic, theta):
        return self.current_phase_relation.i_eval(Ic, theta)

    def _Is_norm(self):
        if self.current_sources_norm is None:
            self.current_sources_norm = scipy.linalg.norm(np.broadcast_to(self.current_sources, (self.circuit._Nj(),)))
        return self.current_sources_norm

    # def _Asq_factorization(self):
    #     if self.Asq_factorization is None:
    #         A = self.get_circuit().get_cycle_matrix()
    #         self.Asq_factorization = scipy.sparse.linalg.factorized(A @ A.T)
    #     return self.Asq_factorization
    #
    # def _AIpLIcA_factorization(self):
    #     if self.AIpLIcA_factorization is None:
    #         Nj, A = self.get_circuit()._Nj(), self.get_circuit().get_cycle_matrix()
    #         L, Ic = self.get_circuit()._L(), scipy.sparse.diags(self.get_circuit()._Ic())
    #         self.AIpLIcA_factorization = scipy.sparse.linalg.factorized(A @ (scipy.sparse.eye(Nj) + L @ Ic) @ A.T)
    #     return self.AIpLIcA_factorization
    #
    # def _IpLIc_factorization(self):
    #     if self.IpLIc_factorization is None:
    #         Nj = self.get_circuit()._Nj()
    #         L, Ic = self.get_circuit()._L(), scipy.sparse.diags(self.get_circuit()._Ic())
    #         self.IpLIc_factorization = scipy.sparse.linalg.factorized(scipy.sparse.eye(Nj) + L @ Ic)
    #     return self.IpLIc_factorization
    #
    # def _Msq_factorization(self):
    #     if self.Msq_factorization is None:
    #         M = self.get_circuit()._Mr()
    #         self.Msq_factorization = scipy.sparse.linalg.factorized(M @ M.T)
    #     return self.Msq_factorization


class StaticConfiguration:
    """
    Approximation or solution to static problem.

    It is defined by a StaticProblem and theta. Here theta must be a
    numpy array of shape (Nj,).

    Provides methods to compute all physical quantities associated with the state.
    The quantities are dimensionless, see the user manual (on github) for a list
    of definitions.

    Furthermore provides a .plot() method to visualize the quantities superimposed
    on the circuit.

    Parameters
    ----------
    problem : StaticProblem
        Static problem object for which this is an approximation or solution.
    theta : (Nj,) array
        Gauge invariant phase differences at each junction, which fully encodes
        the state.
    """

    def __init__(self, problem: StaticProblem, theta: np.ndarray):
        self.problem = problem
        self.theta = np.array(theta)
        if not self.theta.shape == (self.problem.get_circuit()._Nj(),):
            raise ValueError("theta must be of shape (Nj,)")

    def get_circuit(self) -> Circuit:
        """
        Returns circuit (stored in problem).
        """
        return self.problem.get_circuit()

    def get_problem(self) -> StaticProblem:
        """
        Returns the static problem this configuration is associated with.
        """
        return self.problem

    def get_phi(self) -> np.ndarray:
        """
        Returns (Nn,) array containing phases at each node
        """
        # by default the last node (node with highest index number) is grounded.
        M = self.get_circuit().get_cut_matrix()
        return self.get_circuit().Msq_solve(M @ self._th())

    def get_theta(self) -> np.ndarray:
        """
        Returns (Nj,) array containing gauge invariant phase difference at each junction.
        """
        return self.theta

    def get_n(self) -> np.ndarray:
        """
        Returns (Nf,) int array containing vorticity at each face.
        """
        A, tpr = self.get_circuit().get_cycle_matrix(), 1.0 / (2.0 * np.pi)
        return - (A @ np.round(self._th() / (2.0 * np.pi))).astype(int)

    def get_I(self) -> np.ndarray:
        """
        Returns (Nj,) array containing current through each junction.
        """
        return self.problem._cp(self.get_circuit()._Ic(), self._th())

    def get_J(self) -> np.ndarray:
        """
        Returns (Nf,) array containing path current around each face.
        """
        A, Asq_solver = self.get_circuit().get_cycle_matrix(), self.get_problem()._Asq_factorization()
        return Asq_solver(A @ self.get_I())

    def get_flux(self) -> np.ndarray:
        """
        Returns (Nf,) array containing magnetic flux at each face.
        """
        A = self.get_circuit().get_cycle_matrix()
        L = self.get_circuit()._L()
        return self.problem.frustration + A @ L @ self.get_I() / (2 * np.pi)

    def get_EM(self) -> np.ndarray:
        """
        Returns (Nj,) array containing magnetic energy at each junction.
        """
        return 0.5 * self.get_circuit()._L() @ (self.get_I() ** 2)

    def get_EJ(self) -> np.ndarray:
        """
        Returns (Nj,) array containing Josephson energy at each junction.
        """
        return self.problem._icp(self.get_circuit()._Ic(), self._th())

    def get_Etot(self) -> np.ndarray:
        """
        Returns get_EM() + get_EJ().
        """
        return self.get_EJ() + self.get_EM()

    def satisfies_kirchhoff_rules(self, tol=DEF_TOL):
        """
        Returns if configuration satisfies Kirchhoff's current law.

        """
        return self.get_error_kirchhoff_rules() < tol

    def satisfies_winding_rules(self, tol=DEF_TOL):
        """
        Returns if configuration satisfies the winding rules.

        """
        return self.get_error_winding_rules() < tol

    def satisfies_target_vortices(self):
        """
        Returns if vortex configuration equals that of problem.
        """
        return np.all(self.get_n() == self.problem.get_vortex_configuration())

    def is_stable(self, maxiter=DEF_STAB_MAXITER, scheme=0, algorithm=2,
                  accept_ratio=10, preconditioner=None) -> int:
        """
        Determines if a configuration is dynamically stable.

        The criterion for stability is that the Jacobian matrix of the time-evolution at the
        stationairy point is negative definite.

        Parameters
        ----------
        maxiter=DEF_STAB_MAXITER : int
                maximum number of iterations to determine if solutions are stable
        scheme=0 : int
            Scheme for what Jaccobian to compute to determine if the system is
            stable. 0 works for all cases; 1 does not work if there is mixed inductance,
            meaning only some faces have any inductance associated with them.
        algorithm=0 : int
            Algorithm used to find eigenvalues. 0 uses eigsh to find eigenvalues, 1 uses lobpcg.
        accept_ratio=10 : int (only if algorithm=1)
            Parameter used by lobpcg_test_negative_definite.
        preconditioner : {None, "auto", sparse/dense matrix or LinearOperator} (only if algorithm=1)
            Uses preconditioner which must approximate inv(J). If None, no preconditioner is used.
            if "auto", automatically computes preconditioner using stability_get_preconditioner().
            Note that this is independent of theta and can be used for multiple problems.

        Returns
        -------
        status : int
            0: stable, 1: unstable or 2: indeterminate
        """
        cp = self.get_problem().get_current_phase_relation()
        status = compute_stability(self.get_circuit(), self._th(), cp, maxiter=maxiter,
                                   scheme=scheme, algorithm=algorithm, accept_ratio=accept_ratio,
                                   preconditioner=preconditioner)
        return status

    def is_solution(self, tol=DEF_TOL):
        """
        Returns if configuration is a solution meaning it must satisfy both Kirchhoff
        current law and winding rules.
        """
        return self.satisfies_kirchhoff_rules(tol) & self.satisfies_winding_rules(tol)

    def is_target_solution(self, tol=DEF_TOL):
        """
        Returns if configuration is a solution and its vortex_configuration equals
        the one specified in problem.
        """
        return self.is_solution(tol=tol) & self.satisfies_target_vortices()

    def is_stable_target_solution(self, tol=DEF_TOL, stable_maxiter=DEF_STAB_MAXITER):
        """
        Returns if configuration is a solution, is stable and its vortex_configuration equals
        the one specified in problem.
        """
        return self.is_target_solution(tol=tol) & (self.is_stable(maxiter=stable_maxiter) == 0)

    def get_error_kirchhoff_rules(self) -> np.ndarray:
        """
        Returns normalized residual of kirchhoff's rules (normalized so cannot exceed 1).
        """
        return get_kirchhoff_error(self.get_circuit(), self.get_I(), self.get_problem()._Is(),
                                   precomputed_Is_norm=self.problem._Is_norm())

    def get_error_winding_rules(self) -> np.ndarray:
        """
        Returns normalized residual of the winding rules (normalized so cannot exceed 1).
        """
        circuit, problem = self.get_circuit(), self.get_problem()
        f, L = problem._f(), circuit._L()
        return get_winding_error(circuit, self._th() + L @ self.get_I(), get_g(circuit, f, 0))

    def get_error(self):
        """
        Returns get_error_kirchhoff_rules(), get_error_winding_rules().
        """
        return self.get_error_kirchhoff_rules(), self.get_error_winding_rules()

    def plot(self, fig=None, node_quantity=None, junction_quantity="I", face_quantity=None,
             vortex_quantity="n", show_grid=True, show_nodes=True, **kwargs):
        """
        Visualize static configuration on circuit.

        See :py:attr:`circuit_visualize.CircuitPlot` for documentation.
        """
        from pyjjasim.circuit_visualize import ConfigPlot

        self.plot_handle = ConfigPlot(self, vortex_quantity=vortex_quantity, show_grid=show_grid,
                                      junction_quantity=junction_quantity,  show_nodes=show_nodes,
                                      node_quantity=node_quantity, face_quantity=face_quantity,
                                      fig=fig, **kwargs).make()

        return self.plot_handle

    def report(self):
        print("Kirchhoff rules error:    ", self.get_error_kirchhoff_rules())
        print("Path rules error:         ", self.get_error_winding_rules())
        print("is stable:                ", self.is_stable())
        print("is target vortex solution:", self.satisfies_target_vortices())


    def save(self, filename):
        """
        Store configuration in .npy file. Note that the current-phase-relation is not stored!
        """
        with open(filename, "wb") as ffile:
            x, y = self.problem.circuit.graph.coo()
            n1, n2 = self.problem.circuit.graph.get_edges()
            np.save(ffile, x)
            np.save(ffile, y)
            np.save(ffile, n1)
            np.save(ffile, n2)
            np.save(ffile, self.problem.circuit.critical_current_factors)
            np.save(ffile, self.problem.circuit.resistance_factors)
            np.save(ffile, self.problem.circuit.capacitance_factors)
            L_is_sparse = scipy.sparse.issparse(self.problem.circuit.inductance_factors)
            np.save(ffile, L_is_sparse)
            if L_is_sparse:
                np.save(ffile, self.problem.circuit.inductance_factors.indptr)
                np.save(ffile, self.problem.circuit.inductance_factors.indices)
                np.save(ffile, self.problem.circuit.inductance_factors.data)
            else:
                np.save(ffile, self.problem.circuit.inductance_factors)
            np.save(ffile, self.problem.current_sources)
            np.save(ffile, self.problem.frustration)
            np.save(ffile, self.problem.vortex_configuration)
            np.save(ffile, self.theta)

    @staticmethod
    def load(filename):
        """
        Load configuration created with the .save(filename) method. Returns StaticConfiguration.
        Note that the loaded problem will always have the default current-phase-relation.
        """

        with open(filename, "rb") as ffile:
            x = np.load(ffile)
            y = np.load(ffile)
            node1 = np.load(ffile)
            node2 = np.load(ffile)
            g = EmbeddedGraph(x, y, node1, node2)
            Ic = np.load(ffile)
            R = np.load(ffile)
            C = np.load(ffile)
            L_is_sparse = np.load(ffile)
            if L_is_sparse:
                indptr = np.load(ffile)
                indices = np.load(ffile)
                data = np.load(ffile)
                Nj = len(node1)
                L = scipy.sparse.csc_matrix((data, indices, indptr), shape=(Nj, Nj))
            else:
                L = np.load(ffile)
            circuit = Circuit(g, critical_current_factors=Ic, resistance_factors=R,
                              capacitance_factors=C, inductance_factors=L)
            Is = np.load(ffile)
            f = np.load(ffile)
            n = np.load(ffile)
            prob = StaticProblem(circuit, current_sources=Is, frustration=f, vortex_configuration=n)
            th = np.load(ffile)
            return StaticConfiguration(problem=prob, theta=th)

    def _th(self):
        return self.theta


"""
UTILITY ALGORITHMS
"""

def get_kirchhoff_error(circuit: Circuit, I, Is, precomputed_Is_norm=None):
    """
    Residual of kirchhoffs current law: M @ (I - Is) = 0. Normalized; so between 0 and 1.
    """
    if precomputed_Is_norm is None:
        precomputed_Is_norm = scipy.linalg.norm(Is)
    b = circuit.get_cut_matrix() @ (I - Is)
    M_norm = circuit._get_M_norm()
    normalizer = M_norm * (precomputed_Is_norm + scipy.linalg.norm(I))
    return np.finfo(float).eps if np.abs(normalizer) < 1E-20 else scipy.linalg.norm(b) / normalizer

def get_winding_error(circuit: Circuit, th_p, g):
    """
    Residual of winding rule: A @ (thp - g) = 0. Normalized; so between 0 and 1.
    (where thp = th + L @ I)
    """
    A = circuit.get_cycle_matrix()
    A_norm = circuit._get_A_norm()
    normalizer = A_norm * (scipy.linalg.norm(th_p) + scipy.linalg.norm(g))
    return np.finfo(float).eps if np.abs(normalizer) < 1E-20 else scipy.linalg.norm(A @ (th_p - g)) / normalizer

def principle_value(theta):
    """
    Principle value of angle quantity defined as its value in range [-pi, pi)
    """
    return theta - 2 * np.pi * np.round(theta / (2 * np.pi))

def get_g(circuit: Circuit, f=0, z=0):
    """
    g vector obeying A @ g = 2 * pi * (z - f)
    """
    A, Nf = circuit.get_cycle_matrix(), circuit._Nf()
    return 2 * np.pi * A.T @ circuit.Asq_solve(np.broadcast_to(z - f, (Nf,)))

def change_phase_zone(circuit: Circuit, theta, z_old, z_new):
    """
    Converts solution theta in old phase zone z_old to the equivalent
    state theta_out in new phase zone z_new.

    More precisely: adds multiples of 2*pi to theta such that it obeys
    A @ (th_new + L @ I) = 2 * pi * (z_new - f)
    (assuming it already satisfied A @ (th_old + L @ I) = 2 * pi * (z_old- f))

    Parameters
    ----------
    circuit : Circuit
        Circuit.
    theta : (Nj,) array
        Theta in old phase zone.
    z_old : (Nf,) int array
        Old phase zone.
    z_new : (Nf,) int array
        New phase zone.

    Returns
    -------
    theta_new : (Nj,) array
        Theta expressed in new phase zone.
    """
    if np.all(z_new == z_old):
        return theta
    return theta + circuit._A_solve(np.broadcast_to(z_new - z_old, (circuit._Nf(),)).copy()) * 2.0 * np.pi

def node_to_junction_current(circuit: Circuit, node_current):
    """
    Conversion from node_current to junction_current.

    Parameters
    ----------
    node_current : (Nn,) array
        At each node how much current is injected or ejected  (+ if injected)

    Returns
    -------
    junction_current: (Nj,) array
        Returns a configuration of currents at each junction such that at any node
        the net injected current through all its neighbouring edges matches the specified
        node_current.
    """
    # Mr = circuit._Mr()
    # return -Mr.T @ scipy.sparse.linalg.spsolve(Mr @ Mr.T, node_current[:-1])
    return - circuit.get_cut_matrix().T @ circuit.Msq_solve(node_current)

"""
PARAMETER MAXIMIZATION ALGORITHMS
"""


def compute_maximal_parameter(problem_function, initial_guess=None, lambda_tol=DEF_MAX_PAR_TOL,
                              estimated_upper_bound=1.0, maxiter=DEF_MAX_PAR_MAXITER,
                              stepsize_reduction_factor=DEF_MAX_PAR_REDUCE_FACT, require_stability=True,
                              require_vortex_configuration_equals_target=True,
                              compute_parameters=None, stability_parameters=None):
    """
    Finds the largest value of lambda for which problem_function(lambda)
    has a stable stationary state.

     - Must be able to find a stable configuration at lambda=0.
     - One can manually specify an initial_guess for lambda=0.
     - returns a lower- and upperbound for lambda. Stops when the difference < lambda_tol * lower_bound
     - furthermore returns config containing the solutions at the lower_bound. Also its
       accompanied problem has f and Is of lower_bound.
     - Also returns ParameterOptimizeInfo object containing information about the iteration.
     - Algorithm stops if lambda_tol is reached or when newton_iteration failed to converge or diverge.
     - Algorithm needs an estimate of the upperbound for lambda to work.

    Parameters
    ----------
    problem_function : func(lambda) -> StaticProblem
        Function with argument the optimization parameter lambda returning a valid
        StaticProblem object.
    initial_guess=None : valid initial_guess input for StaticProblem.compute()
        Initial guess for problem_function(0) used as starting point for iteration.
    lambda_tol=DEF_MAX_PAR_TOL : float
        Target precision for parameter lambda. Stops iterating if
        upperbound - lowerbound < lambda_tol * lower_bound.
    estimated_upper_bound=1.0 : float
        Estimate for the upperbound for lambda.
    maxiter=DEF_MAX_PAR_MAXITER : int
        Maximum number of iterations.
    stepsize_reduction_factor=DEF_MAX_PAR_REDUCE_FACT : float
        Lambda is multiplied by this factor every time an upper_bound is found.
    require_stability=True : bool
        If True, convergence to a state that is dynamically unstable is
        considered diverged. (see StaticConfiguration.is_stable())
    require_vortex_configuration_equals_target=True : bool
        If True, A result of .compute() is only considered a solution if its
        vortex configuration matches its set "target" vortex configuration
        in the source static_problem.
    compute_parameters: dict or func(lambda) -> dict
        Keyword-argument parameters passed to problem_function(lambda).compute()
        defined as a dictionary or as a function with lambda as input that
        generates a dictionary.
    stability_parameters: dict or func(lambda) -> dict
        Keyword-argument parameters passed to config.is_stable() defined as a
        dictionary or as a function with lambda as input that generates a dictionary.

    Returns
    -------
    lambda_lowerbound : float
        Lowerbound of lambda.
    lambda_upperbound : float
        Upperbound of lambda.
    config : StaticConfiguration
        Containing solutions at lambda=lambda_lowerbound
    iteration_info : ParameterOptimizeInfo
        Object containing information about the iteration.
    """

    if compute_parameters is None:
        compute_parameters = {}
    if stability_parameters is None:
        stability_parameters = {}

    stable_status = None

    # prepare info handle
    info = ParameterOptimizeInfo(problem_function, lambda_tol, require_stability,
                                 require_vortex_configuration_equals_target, maxiter)

    # determine solution at lambda=0
    cur_problem = problem_function(0)
    if initial_guess is None:
        initial_guess = cur_problem.approximate(algorithm=1)
    if isinstance(initial_guess, StaticConfiguration):
        initial_guess = initial_guess._th()
    theta0 = initial_guess
    compute_param = compute_parameters if isinstance(compute_parameters, dict) else compute_parameters(0)
    out = cur_problem.compute(initial_guess=theta0, **compute_param)
    config, status, newton_iter_info = out[0], out[1], out[2]
    is_solution = status==0
    if require_vortex_configuration_equals_target:
        is_target_vortex_config = config.satisfies_target_vortices()
        is_solution &= is_target_vortex_config
    if is_solution and require_stability:
        stab_param = stability_parameters if isinstance(stability_parameters, dict) else stability_parameters(0)
        stable_status = config.is_stable(**stab_param)
        is_solution &= stable_status == 0
    theta = config.theta

    info._preset(is_solution)

    # return if no solution at lambda=0
    if not is_solution:
        return None, None, None, info

    # prepare iteration to find maximum lambda
    found_upper_bound = False
    lambda_stepsize = estimated_upper_bound
    lambda_val = lambda_stepsize
    theta0 = theta

    # start iteration to find maximum lambda
    iter_nr = 0
    while True:

        # determine solution at current lambda
        cur_problem = problem_function(lambda_val)
        compute_param = compute_parameters if isinstance(compute_parameters, dict) else compute_parameters(lambda_val)
        out = cur_problem.compute(initial_guess=theta0, **compute_param)
        config, status, newton_iter_info = out[0], out[1], out[2]
        has_converged = status == 0
        if require_vortex_configuration_equals_target and has_converged:
            is_target_vortex_config = config.satisfies_target_vortices()
            has_converged &= is_target_vortex_config
        else:
            is_target_vortex_config = False

        theta = config.theta

        if status == 2:
            break

        if require_stability:
            if has_converged:
                stab_param = stability_parameters if isinstance(stability_parameters, dict) else stability_parameters(lambda_val)
                stable_status = config.is_stable(**stab_param)
                is_stable = stable_status == 0
            else:
                is_stable = False
            is_solution = has_converged and is_stable
        else:
            is_stable = False
            is_solution = has_converged

        # update information on current iteration in info handle
        info._set(lambda_val, config if has_converged else None, lambda_stepsize,
                  has_converged, newton_iter_info, is_target_vortex_config, is_stable)

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
        if iter_nr >= (maxiter - 1):
            break
        if require_stability and has_converged:
            if stable_status == 2:
                break
        iter_nr += 1

    # determine lower- and upperbound on lambda
    info._finish(status, stable_status)
    lower_bound = lambda_val - lambda_stepsize
    upper_bound = lambda_val if found_upper_bound else np.inf

    if lower_bound is None:
        config = None
    else:
        out_problem = problem_function(lower_bound)
        config = StaticConfiguration(out_problem, theta0)
    return lower_bound, upper_bound, config, info


"""
APPROXIMATE STATE FINDING ALGORITHMS
"""

def london_approximation(circuit: Circuit, f, n):
    """
    Core algorithm computing london approximation.
    """
    A, Nf = circuit.get_cycle_matrix(),  circuit._Nf()
    # if AIpLIcA_solver is None:
    #     Nj = circuit._Nj()
    #     L, Ic = circuit._L(), circuit._Ic()
    #     AIpLIcA_solver = scipy.sparse.linalg.factorized(A @ (scipy.sparse.eye(Nj) + L @ Ic) @ A.T)
    return 2 * np.pi * A.T @ circuit._AIpLIcA_solve(np.broadcast_to(n - f, (Nf,)))


def arctan_approximation(circuit: Circuit, f, n):
    """
    arctan_approximation implemented in arctan_approximation_placed_vortices()
    """
    centr_x, centr_y = circuit.get_face_centroids()
    return arctan_approximation_placed_vortices(circuit, f, n[n != 0], centr_x[n != 0], centr_y[n != 0])

def arctan_approximation_placed_vortices(circuit: Circuit, f, n, x_n, y_n):
    """
    Core algorithm computing arctan approximation.
    """
    n = np.atleast_1d(n)
    x_n = np.atleast_1d(x_n)
    y_n = np.atleast_1d(y_n)
    MT = circuit.get_cut_matrix().T
    # if IpLIc_solver is None:
    #     Nj = circuit._Nj()
    #     L, Ic = circuit._L(), circuit._Ic()
    #     IpLIc_solver = scipy.sparse.linalg.factorized(scipy.sparse.eye(Nj) + L @ Ic)
    x, y = circuit.get_node_coordinates()
    MTphi = MT @ np.sum(np.arctan2(y - y_n[:, None], x - x_n[:, None]) * n[:, None], axis=0)
    out = principle_value(MTphi) + get_g(circuit, f=f, z=0)
    return circuit._IpLIc_solve(out) + 2 * np.pi * np.round(MTphi / (2 * np.pi))


"""
STATIONAIRY STATE FINDING ALGORITHMS
"""


def static_compute(circuit: Circuit, theta0, Is, f, n, z=0,
                   cp=DefaultCPR(), tol=DEF_TOL, maxiter=DEF_NEWTON_MAXITER,
                   stop_as_residual_increases=True, stop_if_not_target_n=False,
                   algorithm=1, use_pyamg=False):
    """
    Core algorithm computing stationary state of a Josephson Junction Circuit using Newtons method.

    Stand-alone method. The wrappers StaticProblem and StaticConfiguration are more convenient.

    Status
    ------
    Stops iterating if ( -> status):
     - residual is smaller than tol, target_n: 0 (converged)
     - residual smaller than tol, not target_n:  1 (diverged)
     - iteration number iter exceeds maxiter: 2 (indeterminate)
     - residual exceeds 0.5: 1 (diverged)
     - if get_n(theta) != n and stop_if_not_target_n==True: 1 (diverged)
     - resid(iter) > resid(iter-3) and stop_as_residual_increases==True : 1 (diverged)

    Parameters
    ------
    circuit: Circuit
        Josephson junction circuit
    theta0 : (Nj,) ndarray
        Initial guess
    Is : (Nj,) ndarray
        Current sources at each junction
    f : (Nf,) ndarray
        Frustration in each face
    n : (Nf,) int ndarray
        Number of vortices in each face
    z=0 : (Nf,) int ndarray or scalar
        Phase zone of each face
    cp=DefaultCPR() : CurrentPhaseRelation
        Current phase relation
    tol=DEF_TOL :  scalar
        Tolerance. is solution if |residual| < tol.
    max_iter=100 :  scalar
        Maximum number of newton iterations.
    stop_as_residual_increases=True : bool
        Iteration stops if error(iter) > error(iter - 3)
    stop_if_not_target_n=False : bool
        Iteration stops if n != target_n

    Returns
    -------
    theta : (Nj,) ndarray
        Gauge invariant phase difference of solution
    convergence_status : int
        * 0 -> converged
        * 1 -> diverged
        * 2 -> max_iter reached without converging or diverging.
    info : NewtonIterInfo
        Information about iteration (timing, steps, residuals, etc)
    """

    # prepare newton iter info
    info = NewtonIterInfo(tol, maxiter)

    # get circuit quantities and matrices
    Nj, Nf = circuit._Nj(), circuit._Nf()
    Mr, M, A = circuit._Mr(), circuit.get_cut_matrix(), circuit.get_cycle_matrix()
    L = circuit._L()
    Ic = np.broadcast_to(circuit.get_critical_current_factors(), (Nj,))

    Is = np.ones((Nj,), dtype=np.double) * Is if np.array(Is).size == 1 else Is
    f = np.ones((Nf,), dtype=np.double) * f if np.array(f).size == 1 else f
    n = np.ones((Nf,), dtype=int) * n if np.array(n).size == 1 else n

    # iteration-0 computations
    g = get_g(circuit, f=f, z=z)

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
        if algorithm == 0:
            J = scipy.sparse.vstack([Mr @ scipy.sparse.diags(q, 0), A @ (scipy.sparse.eye(Nj) + L @ scipy.sparse.diags(q, 0))])

            J_solver = scipy.sparse.linalg.factorized(J.tocsc())
            F = np.concatenate([Mr @ (I - Is), A @ (theta - g + L @ I)])
            theta -= J_solver(F)
        if algorithm == 1:
            q[np.abs(q) < 0.1 * tol] = 0.1 * tol
            S = L + scipy.sparse.diags(1/q, 0)
            b1 = M @ (Is - I)
            yb = M.T @ circuit.Msq_solve(b1)
            b2 = A @ (g - theta - L @ I)
            s = circuit.Asq_solve_sandwich(b2 - A @ S @ yb, S, use_pyamg=use_pyamg)
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                theta += 10 ** 10
            else:
                theta += (yb + A.T @ s) / q

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

def compute_stability(circuit: Circuit, theta, cp, maxiter=DEF_STAB_MAXITER,
                      scheme=0, algorithm=2, accept_ratio=10, preconditioner=None):
    """
    Core implementation to determine if a configuration on a circuit is stable in the sense that
    the Jacobian is negative definite. Does not explicitly check if configuration is a stationairy point.

    Parameters
    ----------
    circuit : Circuit
        Circuit
    theta : (Nj,) array
        Gauge invariant phase difference of static configuration of circuit.
    cp : CurrentPhaseRelation
        Current-phase relation.
    maxiter=DEF_STAB_MAXITER : int
        Maximum number of iterations done to determine stability.
    algorithm=0 : int
        Algorithm used. 0 uses eigsh to find eigenvalues, 1 uses lobpcg.
    accept_ratio :
        Parameter used by lobpcg_test_negative_definite (if algorithm=1).
    preconditioner : {sparse matrix, dense matrix, LinearOperator, None or "auto"}
        Only if algorithm is 1. Uses preconditioner which must approximate inv(J).
        If None, no preconditioner is used. if "auto", automatically computes preconditioner
        using stability_get_preconditioner(). Note that this is independent of theta and can
        be used for multiple problems.
    Returns
    -------
    status : int
        0: stable, 1: unstable or 2: indeterminate
    """
    if scheme == 0:
        J = stability_scheme_0(circuit, theta, cp)
    if scheme == 1:
        J = stability_scheme_1(circuit, theta, cp)

    if algorithm == 0:
        status, largest_eigenvalue = eigsh_test_negative_definite(J, maxiter=maxiter)
        return status
    if algorithm == 1:
        if preconditioner == "auto":
            preconditioner = stability_get_preconditioner(circuit, cp, scheme)
        out = lobpcg_test_negative_definite(J, preconditioner=preconditioner, accept_ratio=accept_ratio,
                                            maxiter=maxiter)
        status, eigenvalue_list, residual_list = out
        return status
    if algorithm == 2:
        eps = 2 * np.finfo(float).eps
        f = scipy.sparse.linalg.splu(J, diag_pivot_thresh=0)
        Up = (f.L @ scipy.sparse.diags(f.U.diagonal())).T
        if not np.allclose((Up - f.U).data, 0):
            print("warning: choleski factorization failed")
            return 2
        return int(~np.all(f.U.diagonal() < eps))
    raise ValueError("invalid algorithm. Must be 0 (eigsh) or 1 (lobpcg)")

def stability_get_preconditioner(circuit: Circuit, cp, scheme):
    """
    Compute preconditioner to determine stability

    Scheme 1 with inductance cannot be preconditioned.

    Note that this preconditioner is independent of theta, so can be reused
    for multiple problems. Generating a preconditioner is slow as it does
    a factorization.
    """
    Nj, Nnr = circuit._Nj(), circuit._Nnr()
    q = cp.d_eval(circuit._Ic(), np.zeros(Nj))
    A, M, L = circuit.get_cycle_matrix(), circuit._Mr(), circuit._L()
    if scheme == 0:
        AL = (A @ L @ A.T).tocoo()
        ALL = scipy.sparse.coo_matrix((AL.data, (AL.row + Nnr, AL.col + Nnr)), shape=(Nj, Nj)).tocsc()
        m = scipy.sparse.vstack([M, A @ L]).tocsc()
        X = - (m @ scipy.sparse.diags(q, 0) @ m.T + ALL)
        select = np.diff(X.indptr) != 0
        X = X[select, :][:, select]
    if scheme == 1:
        if circuit._has_inductance():
            raise ValueError("Scheme 1 with inductance cannot be preconditioned.")
        else:
            X = - M @ scipy.sparse.diags(q, 0) @ M.T
    X_solver = scipy.sparse.linalg.factorized(X)
    return scipy.sparse.linalg.LinearOperator(X.shape, matvec=X_solver)

def stability_scheme_0(circuit: Circuit, theta, cp):
    """
    Scheme to determine matrix for which the system is stable if it is negative definite.

    Works for mixed inductance but generally slower than scheme 1.

    Scheme 0: matrix is:
     * J = m @ X @ m.T
     * where X = -grad cp(Ic, theta) - A.T @ inv(A @ L @ A.T) @ A
     * and m = [M ; A @ L]
     * all-zero rows and columns are removed.
    """
    Nj, Nnr = circuit._Nj(), circuit._Nnr()
    A, M, L = circuit.get_cycle_matrix(),  circuit._Mr(), circuit._L()
    Ic = circuit._Ic()
    q = cp.d_eval(Ic, theta)
    AL = (A @ L @ A.T).tocoo()
    ALL = scipy.sparse.coo_matrix((AL.data, (AL.row + Nnr, AL.col + Nnr)), shape=(Nj, Nj)).tocsc()
    m = scipy.sparse.vstack([M, A @ L]).tocsc()
    J = - (m @ scipy.sparse.diags(q, 0) @ m.T + ALL)
    select = np.diff(J.indptr)!=0
    J = J[select, :][:, select]
    return J

def stability_scheme_1(circuit: Circuit, theta, cp):
    """
    Scheme to determine matrix for which the system is stable if it is negative definite.

    Generally faster than scheme 0 but does not work for mixed inductance (where only some
    faces have any inductance associated with them).

    Scheme 1: matrix is:
     * if L=0: J = -M @ grad cp(Ic, theta) @ M.T
     * if L!=0: J = -grad cp(Ic, theta) - A.T @ inv(A @ L @ A.T) @ A
    """
    if circuit._has_mixed_inductance():
        raise ValueError("Scheme 1 does not allow mixed inductance.")
    Ic = circuit._Ic()
    q = cp.d_eval(Ic, theta)
    if circuit._has_inductance():
        Nj = circuit._Nj()
        A, L = circuit.get_cycle_matrix(), circuit._L()
        ALA_solver = scipy.sparse.linalg.factorized(A @ L @ A.T)
        def func(x):
            x = x.reshape(Nj, 1)
            return -q[:, None] * x - A.T @ ALA_solver(A @ x)
        J = scipy.sparse.linalg.LinearOperator((Nj, Nj), matvec=func)
    else:
        M = circuit._Mr()
        J = -M @ scipy.sparse.diags(q, 0) @ M.T
    return J

def eigsh_test_negative_definite(A, maxiter=DEF_STAB_MAXITER):
    """
    Determines if symmetric matrix A is negative definite using eigsh.

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        Matrix to determine if positive definite
    maxiter=200 :
        Maximum number of iterations. If exceeded; result is indeterminate.

    Returns
    -------
    status : int
        0: negative definite, 1: not negative definite or 2: indeterminate
    largest_eigenvalue : float
        Largest eigenvalue determined by eigsh algorithm.
    """
    try:
        w, v = scipy.sparse.linalg.eigsh(A, 1, maxiter=maxiter, which="LA")
        largest_eigenvalue = w[0]
        is_stable = largest_eigenvalue < 20 * np.finfo(float).eps
        status = int(~is_stable)
    except ArpackNoConvergence:
        print("warning: eigsh ran out of steps. Consider increasing maxiter or"
              " using other algorithm.")
        status = 2
        largest_eigenvalue = np.nan
    return status, largest_eigenvalue

def lobpcg_test_negative_definite(A, preconditioner=None, accept_ratio=10, maxiter=DEF_STAB_MAXITER):

    """
    Determines if symmetric matrix A is negative definite using the LOBPCG method.

    Does several lobpcg runs with increasing iter_count. Algorithm stops if a run
    has outcome max_eigv and residual that obey:
     * max_eigv + accept_ratio * residual < eps -> negative definite (status=0)
     * max_eigv > eps -> not negative definite (status=1)
     * total_iters > maxiter -> indeterminate (status=2)

    The iter_count at the first run is 2 and increased by one every run, and every consecutive
    run uses the previous eigenvector as starting vector.

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        Matrix to determine if positive definite
    preconditioner=None : {dense matrix, sparse matrix, LinearOperator, None}
        Preconditioner to A, should approximate the iverse of A. None means identity matrix.
    accept_ratio=10 : int
        Considered converged if max_eigv + accept_ratio * residual < eps.
    maxiter=200 :
        Maximum number of iterations. If exceeded; result is indeterminate.

    Returns
    -------
    status : int
        0: negative definite, 1: not negative definite or 2: indeterminate
    eigenvalue_list (steps,) array
        Largest eigenvalue found at each iteration.
    residual_list (steps,) array
        Residual at each iteration.
    """
    eps = 2 * np.finfo(float).eps

    iter_count = 2
    total_iters = iter_count

    x0 = np.random.rand(A.shape[0], 1)
    lobpcg_out = scipy.sparse.linalg.lobpcg(A, x0, M=preconditioner, maxiter=iter_count, tol=eps,
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

    while (max_eigenvalue[-1] + accept_ratio * residual[-1] > eps) and total_iters < maxiter \
            and max_eigenvalue[-1] < eps:
        if total_iters > maxiter - iter_count:
            return 2, max_eigenvalue, residual
        lobpcg_out = scipy.sparse.linalg.lobpcg(A, lobpcg_out[1], M=preconditioner,
                                                maxiter=iter_count, tol=eps, retLambdaHistory=True,
                                                retResidualNormsHistory=True)
        total_iters += iter_count
        iter_count += 1
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
    status = int(~is_negative_definite)
    eigenvalue_list, residual_list = max_eigenvalue, residual
    return status, eigenvalue_list, residual_list

