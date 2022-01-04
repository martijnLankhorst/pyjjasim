from __future__ import annotations

import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.optimize

from pyJJAsim.josephson_circuit import Circuit
from pyJJAsim.static_problem import DefaultCPR
from pyJJAsim.static_problem import StaticConfiguration, StaticProblem

__all__ = ["TimeEvolutionProblem", "TimeEvolutionResult"]

DEF_TEMPERATURE_PROFILE = lambda search_T: lambda t: np.interp(t, [0, 0.25, 0.75, 1], [1, 1.1 * search_T, 0.9 * search_T, 0])


class TimeEvolutionProblem:
    """
    Define multiple time evolution problems with varying parameters in a Josephson Junction Circuit circuit.

     - All problems are computed in one call (generally much faster than computing individual problems).
     - Use self.compute() to obtain a TimeEvolutionResult object containing the resulting time evolution.

    Physical parameters:       symbol  default
     - time_step               dt      0.05
     - time_step_count         Nt      1000
     - current_phase_relation  cp      DefaultCPR()

    Problem space parameters: symbol  default   shape
     - frustration            f       0.0       array broadcast compatible to (Nf, W, Nt)
                                                or f(i) -> broadcast compatible to (Nf, W) for i in range(Nt)
     - current_sources        Is      0.0       array broadcast compatible to (Nj, W, Nt)
                                                or Is(i) -> broadcast compatible to (Nj, W) for i in range(Nt)
     - voltage_sources        Vs      0.0       array broadcast compatible to (Nj, W, Nt)
                                                or Vs(i) -> broadcast compatible to (Nj, W) for i in range(Nt)
     - temperature            T       0.0       array broadcast compatible to (Nj, W, Nt)
                                                or T(i) -> broadcast compatible to (Nj, W) for i in range(Nt)

    Where W is problem_count.

    Store parameters:     default    type                           Needed for:
     - store_time_steps   None       None (represents all points)
                                     or array in range(Nt)
                                     or mask of shape (Nt,)
     - store_theta        True       bool                           phi, EJ, ...
     - store_voltage      True       bool
     - store_current      True       bool                           Phi, ...

    The store parameters allow one to control what quantities are stored at what timesteps. Only the base quantities
    theta, voltage and current can be stored in the resulting TimeEvolutionResult; other quantities like energy
    or magnetic_flux are computed based on these. A list is given which quantities require what base quantities
    to be stored (see documentation of TimeEvolutionResult for more information).

    Initial condition parameters:   default  type
     - config_at_minus_1            None     StaticConfiguration    initial condition at timestep=-1
                                             or None                 -> set to self.get_static_problem(t=0,n=z).compute()
     - config_at_minus_2            None     StaticConfiguration    initial condition at timestep=-2
                                             or None                 -> set to self.get_static_problem(t=0,n=z).compute()

    Methods
     - compute() -> TimeEvolutionResult
     - get_static_problem(time_step=0) -> StaticProblem
     - get_problem_count()
     - get_circuit()
     - get_time_step()
     - get_time_step_count()
     - get_current_phase_relation()
     - get_phase_zone()
     - get_frustration()
     - get_current_sources()
     - get_voltage_sources()
     - get_temperature()
     - get_store_time_steps()
     - get_store_theta()
     - get_store_voltage()
     - get_store_current()


    """

    def __init__(self, circuit: Circuit, time_step=0.05, time_step_count=1000,
                 current_phase_relation=DefaultCPR(),
                 frustration=0.0, current_sources=0.0,
                 voltage_sources=0.0, temperature=0.0,
                 store_time_steps=None, store_theta=True, store_voltage=True, store_current=True,
                 config_at_minus_1: np.ndarray = None,
                 config_at_minus_2: np.ndarray = None):

        self.circuit = circuit
        self.time_step = time_step
        self.time_step_count = time_step_count
        self.current_phase_relation = current_phase_relation

        def get_prob_cnt(x):
            s = np.array(x(0) if hasattr(x, "__call__") else x).shape
            return s[1] if len(s) > 1 else 1

        self.problem_count = max(get_prob_cnt(frustration),
                                 get_prob_cnt(current_sources), get_prob_cnt(voltage_sources),
                                 get_prob_cnt(temperature))
        Nj, Nf, W, Nt = self.circuit._Nj(), self.circuit._Nf(), self.get_problem_count(), self.get_time_step_count()
        self.frustration = frustration if hasattr(frustration, "__call__") else \
            self._broadcast(np.array(frustration), (Nf, W, Nt))
        self.current_sources = current_sources if hasattr(current_sources, "__call__") else \
            self._broadcast(np.array(current_sources), (Nj, W, Nt))
        self.voltage_sources = voltage_sources if hasattr(voltage_sources, "__call__") else \
            self._broadcast(np.array(voltage_sources), (Nj, W, Nt))
        self.temperature = temperature if hasattr(temperature, "__call__") else \
            self._broadcast(np.array(temperature), (Nj, W, Nt))

        self.store_time_steps = np.ones(self._Nt(), dtype=bool)
        self.store_time_steps = self._to_time_point_mask(store_time_steps)
        self.store_theta = store_theta
        self.store_voltage = store_voltage
        self.store_current = store_current

        self.config_at_minus_1 = np.zeros((Nj, W), dtype=np.double) if config_at_minus_1 is None else config_at_minus_1
        self.config_at_minus_2 = np.zeros((Nj, W), dtype=np.double) if config_at_minus_2 is None else config_at_minus_2
        self.prepared_theta_s = np.zeros((Nj, W), dtype=np.double)

    def get_static_problem(self, vortex_configuration, problem_nr=0, time_step=0) -> StaticProblem:
        return StaticProblem(self.circuit, current_sources=self._Is(time_step)[:, problem_nr].copy(),
                             frustration=self._f(time_step)[:, problem_nr].copy(),
                             vortex_configuration=vortex_configuration,
                             current_phase_relation=self.current_phase_relation)

    def get_problem_count(self):
        return self.problem_count

    def get_circuit(self) -> Circuit:
        return self.circuit

    def get_time_step(self):
        return self.time_step

    def get_time_step_count(self):
        return self.time_step_count

    def get_current_phase_relation(self):
        return self.current_phase_relation

    def get_phase_zone(self):
        return 0

    def get_frustration(self):
        return self.frustration

    def get_current_sources(self):
        return self.current_sources

    def get_net_sourced_current(self, time_step):
        M = self.get_circuit().get_cut_matrix()
        return 0.5 * np.sum(np.abs((M @ self._Is(time_step))), axis=0)

    def get_voltage_sources(self):
        return self.voltage_sources

    def get_temperature(self):
        return self.temperature

    def get_store_time_steps(self):
        return self.store_time_steps

    def get_store_theta(self):
        return self.store_theta

    def get_store_voltage(self):
        return self.store_voltage

    def get_store_current(self):
        return self.store_current

    def get_time(self):
        return np.arange(self._Nt(), dtype=np.double) * self._dt()

    def get_time_at_stored(self):
        return self.get_time()[self.store_time_steps]

    def compute(self) -> TimeEvolutionResult:
        """
        Compute time evolution on an Josephson Circuit.

        Requires an initial configuration; step0_init. Must be StaticConfiguration or None.
        If None; it is set to self.to_static_problem().compute()

        If the circuit has capacitance, requires a second initial configuration; step1_init. If
        this is set to None, it will be assigned the value of step0_init.

        If there is no inductance and algorithm=0; the initial condition must obey A(th-g) = 0.
        If this is not obeyed, it is automatically projected to obey the constraint. Due to numerical
        rounding, it will slowly drift away from this condition. Use rounding_flux_drift_correction=True
        to apply projection every 100 timesteps.

        """
        if self.get_circuit()._has_mixed_inductance():
            return time_evolution_algo_1(self)
        else:
            return time_evolution_algo_0(self)

    def __str__(self):
        return "time evolution problem: " + \
               "\n\ttime: " + self.time_step_count.__str__() + " steps of " + self.time_step.__str__() + \
               "\n\tcurrent sources: " + self.current_sources.__str__() + \
               "\n\tvoltage sources: " + self.voltage_sources.__str__() + \
               "\n\tfrustration: " + self.frustration.__str__() + \
               "\n\ttemperature: " + self.temperature.__str__() + \
               "\n\tcurrent-phase relation: " + self.current_phase_relation.__str__()

    def _Nt(self):
        return self.time_step_count

    def _Nt_s(self):
        return np.asscalar(np.sum(self.store_time_steps))

    def _dt(self):
        return self.time_step

    def _f(self, time_step) -> np.ndarray:  # (Nf, W), read-only
        return np.broadcast_to(self.frustration(time_step), (self.circuit._Nf(), self.get_problem_count())) \
            if hasattr(self.frustration, "__call__") else self.frustration[:, :, time_step]

    def _Is(self, time_step) -> np.ndarray: # (Nj, W), read-only
        return np.broadcast_to(self.current_sources(time_step), (self.circuit._Nj(), self.get_problem_count()))\
            if hasattr(self.current_sources, "__call__") else self.current_sources[:, :, time_step]

    def _Vs(self, time_step) -> np.ndarray: # (Nj, W), read-only
        return np.broadcast_to(self.voltage_sources(time_step), (self.circuit._Nj(), self.get_problem_count()))\
            if hasattr(self.voltage_sources, "__call__") else self.voltage_sources[:, :, time_step]

    def _T(self, time_step) -> np.ndarray:  # (Nj, W), read-only
        return np.broadcast_to(self.temperature(time_step), (self.circuit._Nj(), self.get_problem_count()))\
            if hasattr(self.temperature, "__call__") else self.temperature[:, :, time_step]

    def _theta_s(self, time_step) -> np.ndarray:    # (Nj, W), read-only
        # warning: must be called exactly once at every timestep in order, starting below zero.
        if time_step < 0:
            self.prepared_theta_s = self._Vs(0) * (time_step * self._dt())
        else:
            self.prepared_theta_s = self.prepared_theta_s + self._Vs(time_step) * self._dt()
            return self.prepared_theta_s

    def _cp(self, theta) -> np.ndarray:   # (Nj, W)
        # theta -> (Nj, W)
        Ic = self.get_circuit()._Ic().diag(force_as_vector=True, vector_length=self.get_circuit()._Nj())[:, None]
        return self.current_phase_relation.eval(Ic, theta)

    def _dcp(self, theta) -> np.ndarray:  # (Nj, W)
        Ic = self.get_circuit()._Ic().diag(force_as_vector=True, vector_length=self.get_circuit()._Nj())[:, None]
        return self.current_phase_relation.d_eval(Ic, theta)

    def _icp(self, theta) -> np.ndarray:  # (Nj, W)
        Ic = self.get_circuit()._Ic().diag(force_as_vector=True, vector_length=self.get_circuit()._Nj())[:, None]
        return self.current_phase_relation.i_eval(Ic, theta)

    def _broadcast(self, x, shape):
        x_shape = np.array(x).shape
        x = x.reshape(x_shape + (1,) * (len(shape) - len(x_shape)))
        return np.broadcast_to(x, shape)

    def _to_time_point_mask(self, time_points):
        # time_points:  None (represents all points)
        #               or slice
        #               or array in range(Nt)
        #               or mask of shape (Nt,)
        if time_points is None:
            time_points = self.store_time_steps
        time_points = np.array(time_points)
        if not (time_points.dtype in (bool, np.bool)):
            try:
                x = np.zeros(self._Nt(), dtype=bool)
                x[time_points] = True
                time_points = x
            except:
                raise ValueError("Invalid store_time_steps; must be None, mask, slice or index array")
        return time_points

def time_evolution_algo_0(problem: TimeEvolutionProblem) -> TimeEvolutionResult:
    out = TimeEvolutionResult(problem)

    circuit = problem.get_circuit()
    Nj, W = circuit._Nj(), problem.get_problem_count()
    dt = problem._dt()
    if circuit._has_mixed_inductance():
        raise ValueError("Time evolution algorithm 0 does not work with mixed inductance "
                         "(some loops have no inductance while others do. Use algorithm 1.")
    store_th, store_I, store_V = problem.store_theta, problem.store_current, problem.store_voltage

    A = circuit.get_cycle_matrix()
    AT = A.T
    Rv = 1 / (dt * circuit._R().diag(force_as_vector=True, vector_length=Nj)[:, None])
    Cv = circuit._C().diag(force_as_vector=True, vector_length=Nj)[:, None] / (dt ** 2)
    Cprev, C0, Cnext = Cv, -2.0 * Cv - Rv, Cv + Rv

    if circuit._has_inductance():
        L = problem.circuit._L().matrix(Nj)
        L_sw_fact = scipy.sparse.linalg.factorized(A @ L @ AT)
    Asq_fact = scipy.sparse.linalg.factorized(A @ scipy.sparse.diags(1.0 / Cnext[:, 0], 0) @ AT)

    theta_next = problem.config_at_minus_1
    theta = problem.config_at_minus_2

    for i in range(problem._Nt()):
        Is, T, theta_s, f = problem._Is(i), problem._T(i), problem._theta_s(i), problem._f(i)

        rand = np.random.randn(Nj, W) if i % 3 == 0 else rand[np.random.permutation(Nj), :]
        fluctuations = ((2.0 * T * Rv) ** 0.5) * rand

        theta_prev = theta.copy()
        theta = theta_next.copy()
        if circuit._has_inductance():
            y = AT @ L_sw_fact(A @ (theta + theta_s + L @ Is) + 2 * np.pi * f)
            theta_next = -(problem._cp(theta) + fluctuations - Is + C0 * theta + Cprev * theta_prev + y) / Cnext
            I = Is - y if store_I else None
        else:
            x = (problem._cp(theta) + fluctuations - Is + C0 * theta + Cprev * theta_prev) / Cnext
            theta_next = -x + (AT @ Asq_fact(A @ (x - theta_s) - 2 * np.pi * f)) / Cnext
            I = (x + theta_next) * Cnext + Is if store_I else None
        V = (theta_next - theta) / dt if store_V else None
        if problem.store_time_steps[i]:
            out._update([theta_next if problem.store_theta else None, V, I])
    return out

def time_evolution_algo_1(problem: TimeEvolutionProblem) -> TimeEvolutionResult:
    """

    """

    out = TimeEvolutionResult(problem)

    circuit = problem.circuit
    Nj, Nf, W = circuit._Nj(),  circuit._Nf(), problem.get_problem_count()
    dt = problem._dt()

    store_th, store_I, store_V = problem.store_theta, problem.store_current, problem.store_voltage

    A = circuit.get_cycle_matrix()
    M = circuit._Mr().A
    Rv = 1 / (dt * circuit._R().diag(force_as_vector=True, vector_length=Nj)[:, None])
    Cv = circuit._C().diag(force_as_vector=True, vector_length=Nj)[:, None] / (dt ** 2)
    Cprev, C0, Cnext = Cv, -2.0 * Cv - Rv, Cv + Rv

    L = problem.circuit._L().matrix(Nj)
    L_mask = circuit._get_mixed_inductance_mask()
    A1 = A[~L_mask, :]
    A2 = A[L_mask, :]

    Cnext_mat = scipy.sparse.diags(Cnext[:, 0], 0)
    matrix = scipy.sparse.vstack([M @ Cnext_mat, A1 @ L @ Cnext_mat, A2]).tocsc()
    m_fact = scipy.sparse.linalg.factorized(matrix)

    theta = problem.config_at_minus_2
    theta_next = problem.config_at_minus_1

    for i in range(problem._Nt()):
        Is, T, theta_s, f = problem._Is(i), problem._T(i), problem._theta_s(i), problem._f(i)

        # optimization to only generate new gaussian noise every three timesteps.
        # On the other timesteps, the last generated noise is shuffled.
        rand = np.random.randn(Nj, W) if i % 3 == 0 else rand[np.random.permutation(Nj), :]
        fluctuations = ((2.0 * T * Rv) ** 0.5) * rand

        theta_prev = theta.copy()
        theta = theta_next.copy()
        y = problem._cp(theta) + fluctuations + C0 * theta + Cprev * theta_prev
        F = np.concatenate([M @ (y - Is), A1 @ (theta + theta_s + L @ y) + 2 * np.pi * f[~L_mask],
                            A2 @ theta_s + 2 * np.pi * f[L_mask]], axis=0)
        theta_next = -m_fact(F)

        x = Cprev * theta_prev + C0 * theta + Cnext * theta_next + problem._cp(theta) + fluctuations
        matrix = scipy.sparse.vstack([M, A @ L]).tocsc()
        print(scipy.linalg.norm(matrix @ x + np.concatenate([-M @ Is, A @ (theta + theta_s) + 2 * np.pi * f], axis=0)))

        if problem.store_time_steps[i]:
            out._update([theta_next if problem.store_theta else None,
                         (theta_next - theta) / dt if store_V else None,
                         y + Cnext * theta_next if store_I else None])
    return out


class TimeEvolutionResult:
    """
    Represents data of simulated time evolution(s) on a Josephson circuit.

    It is defined by a problem, circuit and any of the quantities theta, current and voltage.
    These must be of shape (*problem.get_shape(), Nj, Nt_stored). Here Nj is the junction count
    and Nt_stored is the number of time steps that are stored during simulation.

    One can query several properties of the circuit configurations:

     (property)                           (symbol)           (needs)    (shape)
     - phases                             phi                th         (pr_shape, Nn, Nt_stored)
     - gauge_invariant_phase_difference   theta                         (pr_shape, Nj, Nt_stored)
     - vortex_configuration               n                  th         (pr_shape, Nj, Nt_stored)
     - junction_current                   I                             (pr_shape, Nj, Nt_stored)
     - supercurrent                       Isup               th         (pr_shape, Nj, Nt_stored)
     - path_current                       J                  I          (pr_shape, Np, Nt_stored)
     - magnetic_flux (through faces)      flux               I          (pr_shape, Nn, Nt_stored)
     - junction_voltage                   V                             (pr_shape, Nj, Nt_stored)
     - node_voltage                       U                  V          (pr_shape, Nn, Nt_stored)
     - josephson_energy                   EJ                 th         (pr_shape, Nj, Nt_stored)
     - magnetic_energy                    EM                 I          (pr_shape, Nj, Nt_stored)
     - capacitor_energy                   EC                 V          (pr_shape, Nj, Nt_stored)
     - total_energy                       Etot               th,(I),(V) (pr_shape, Nj, Nt_stored)

    A property query is done with te command .get_[symbol](select_time_points=None)

    where:
        select_time_points      None (represents all stored points)
                                or array in range(Nt)
                                or mask of shape (Nt,)

    TimeEvolutionResult only store theta, current and voltage data at specified timepoints,
    other queried properties must be calculated from that. The above table shows which
    quantities need to be stored to be able to query a certain property. (parenthesis
    mean quantities may be needed).

    Thermal quantities are based only on theta, and represent "thermal averages", in the
    sense that derivatives are computed on theta over the data at the queried timepoints
    (nót the stored timepoints). This naturally smooths the quantity, which filters the
    thermal noise present at nonzero temperature.

    """

    def __init__(self, problem: TimeEvolutionProblem):
        self.problem = problem
        Nj, W, Nt_s = problem.circuit._Nj(), self.get_problem_count(), problem._Nt_s()
        self.theta = np.zeros((Nj, W, Nt_s), dtype=np.double) if problem.store_theta else None
        self.voltage = np.zeros((Nj, W, Nt_s), dtype=np.double) if problem.store_theta else None
        self.current = np.zeros((Nj, W, Nt_s), dtype=np.double) if problem.store_theta else None
        self.store_point = 0
        s = self.problem.store_time_steps.astype(int)
        self.time_point_indices = np.cumsum(s) - s

    def _update(self, data):
        th, V, I = data[0], data[1], data[2]
        if th is not None:
            self.theta[:, :, self.store_point] = th
        if V is not None:
            self.voltage[:, :, self.store_point] = V
        if I is not None:
            self.current[:, :, self.store_point] = I
        if (th is not None) or (th is not None) or (th is not None):
            self.store_point += 1

    def _th(self, time_point) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Cannot query theta; quantity is not stored during time evolution simulation.")
        return self.theta[:, :, self._time_point_index(time_point)]

    def _V(self, time_point) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Cannot query voltage; quantity is not stored during time evolution simulation.")
        return self.voltage[:, :, self._time_point_index(time_point)]

    def _I(self, time_point) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Cannot query current; quantity is not stored during time evolution simulation.")
        return self.current[:, :, self._time_point_index(time_point)]

    def _time_point_index(self, time_points):
        if time_points is None:
            time_points = self.problem.store_time_steps
        if not np.all(self.problem.store_time_steps[time_points]):
            raise ValueError("Queried a timepoint that is not stored during time evolution simulation.")
        return self.time_point_indices[time_points]

    def get_problem_count(self):
        return self.problem.get_problem_count()

    def get_circuit(self) -> Circuit:
        return self.problem.get_circuit()

    def select_static_configuration(self, prob_nr, time_step) -> StaticConfiguration:
        if self.theta is None:
            raise ValueError("Theta not stored; cannot select static configuration.")
        problem = StaticProblem(self.get_circuit(), current_sources=self.problem._Is(time_step)[:, prob_nr],
                                frustration=self.problem._f(time_step)[:, prob_nr],
                                vortex_configuration=self.get_n(time_step)[:, prob_nr],
                                current_phase_relation=self.problem.current_phase_relation)
        return StaticConfiguration(problem, self.theta[:, prob_nr, time_step])

    def get_phi(self, select_time_points=None) -> np.ndarray:
        M = self.get_circuit()._Mr().A
        Mrsq = M @ M.T
        Z = np.zeros((1, self.get_problem_count()), dtype=np.double)
        func = lambda tp: np.concatenate((scipy.sparse.linalg.spsolve(Mrsq, M @ self._th(tp)), Z), axis=0)
        return self._select(select_time_points, self.get_circuit()._Nn(), func)

    def get_theta(self, select_time_points=None) -> np.ndarray:
        return self._select(select_time_points, self.get_circuit()._Nj(), self._th)

    def get_n(self, select_time_points=None) -> np.ndarray:
        A = self.get_circuit().get_cycle_matrix()
        func = lambda tp:  -A @ np.round(self._th(tp) / (2.0 * np.pi))
        return self._select(select_time_points, self.get_circuit()._Nf(), func).astype(int)

    def get_EJ(self, select_time_points=None) -> np.ndarray:
        func = lambda tp: self.problem._icp(self._th(tp))
        return self._select(select_time_points, self.get_circuit()._Nj(), func)

    def get_I(self, select_time_points=None) -> np.ndarray:
        return self._select(select_time_points, self.get_circuit()._Nj(), self._I)

    def get_Isup(self, select_time_points=None) -> np.ndarray:
        func = lambda tp: self.problem._cp(self._th(tp))
        return self._select(select_time_points, self.get_circuit()._Nj(), func)

    def get_J(self, select_time_points=None) -> np.ndarray:
        A = self.get_circuit().get_cycle_matrix()
        func = lambda tp: scipy.sparse.linalg.spsolve(A @ A.T,  A @ self._I(tp))
        return self._select(select_time_points, self.get_circuit()._Nf(), func)

    def get_flux(self, select_time_points=None) -> np.ndarray:
        Nj, Nf = self.get_circuit()._Nj(), self.get_circuit()._Nf()
        A = self.get_circuit().get_cycle_matrix()
        func = lambda tp: A @ (self.get_circuit()._L().matrix(Nj) @ self._I(tp))
        return self._select(select_time_points, Nf, func)

    def get_EM(self, select_time_points=None) -> np.ndarray:
        Nj = self.get_circuit()._Nj()
        func = lambda tp: 0.5 * self.get_circuit()._L().matrix(Nj) @ (self._I(tp) ** 2)
        return self._select(select_time_points, Nj, func)

    def get_V(self, select_time_points=None):
        return self._select(select_time_points, self.get_circuit()._Nj(), self._V)

    def get_U(self, select_time_points=None):
        M = self.get_circuit()._Mr().A
        Mrsq = M @ M.T
        Z = np.zeros((1, self.get_problem_count()), dtype=np.double)
        func = lambda tp: np.concatenate((scipy.sparse.linalg.spsolve(Mrsq, M @ self._V(tp)), Z), axis=0)
        return self._select(select_time_points, self.get_circuit()._Nn(), func)

    def get_EC(self, select_time_points=None):
        Nj = self.get_circuit()._Nj()
        C = self.get_circuit()._C().diag(force_as_vector=True, vector_length=Nj)
        func = lambda tp: 0.5 * C[:, None] * self._V(tp) ** 2
        return self._select(select_time_points, Nj, func)

    def get_Etot(self, select_time_points=None) -> np.ndarray:
        return self.get_EJ(select_time_points) + self.get_EM(select_time_points) + \
               self.get_EC(select_time_points)

    def plot(self, time_point=0, show_vortices=True, vortex_diameter=0.25, vortex_color=(0, 0, 0),
             anti_vortex_color=(0.8, 0.1, 0.2), vortex_alpha=1, show_grid=True, grid_width=1,
             grid_color=(0.3, 0.5, 0.9), grid_alpha=0.5, show_colorbar=True, show_arrows=True,
             arrow_quantity="I", arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
             arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1, arrow_color=(0.2, 0.4, 0.7),
             arrow_alpha=1, show_nodes=True, node_diameter=0.2,
             node_face_color=(1, 1, 1), node_edge_color=(0, 0, 0), node_alpha=1, show_node_quantity=False,
             node_quantity="phase", node_quantity_cmap=None, node_quantity_clim=(0, 1), node_quantity_alpha=1,
             node_quantity_logarithmic_colors=False, show_face_quantity=False, face_quantity="n",
             face_quantity_cmap=None, face_quantity_clim=(0, 1), face_quantity_alpha=1,
             face_quantity_logarithmic_colors=False, figsize=None, title="", **kwargs):

        from circuit_visualize import CircuitPlot

        return CircuitPlot(self, time_point=time_point, show_vortices=show_vortices, vortex_diameter=vortex_diameter,
                         vortex_color=vortex_color, anti_vortex_color=anti_vortex_color,
                         vortex_alpha=vortex_alpha, show_grid=show_grid, grid_width=grid_width,
                         grid_color=grid_color, grid_alpha=grid_alpha, show_colorbar=show_colorbar,
                         show_arrows=show_arrows,
                         arrow_quantity=arrow_quantity, arrow_width=arrow_width, arrow_scale=arrow_scale,
                         arrow_headwidth=arrow_headwidth, arrow_headlength=arrow_headlength,
                         arrow_headaxislength=arrow_headaxislength, arrow_minshaft=arrow_minshaft,
                         arrow_minlength=arrow_minlength, arrow_color=arrow_color,
                         arrow_alpha=arrow_alpha, show_nodes=show_nodes, node_diameter=node_diameter,
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

    def animate(self, problem_nr=0, time_points=None, show_vortices=True,
                vortex_diameter=0.25, vortex_color=(0, 0, 0), anti_vortex_color=(0.8, 0.1, 0.2),
                vortex_alpha=1, show_grid=True, grid_width=1,
                grid_color=(0.3, 0.5, 0.9), grid_alpha=0.5, show_colorbar=True, show_arrows=True, arrow_quantity="I",
                arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
                arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1, arrow_color=(0.2, 0.4, 0.7),
                arrow_alpha=1, show_nodes=True, node_diameter=0.2,
                node_face_color=(1, 1, 1), node_edge_color=(0, 0, 0), node_alpha=1, show_node_quantity=False,
                node_quantity="phase", node_quantity_cmap=None, node_quantity_clim=(0, 1), node_quantity_alpha=1,
                node_quantity_logarithmic_colors=False,
                show_face_quantity=False, face_quantity="n", face_quantity_cmap=None,
                face_quantity_clim=(0, 1), face_quantity_alpha=1,
                face_quantity_logarithmic_colors=False, figsize=None,
                animate_interval=5, title=""):

        from pyJJAsim.circuit_visualize import CircuitMovie

        return CircuitMovie(self, problem_nr=problem_nr, time_points=time_points, show_vortices=show_vortices,
                          vortex_diameter=vortex_diameter, vortex_color=vortex_color,
                          anti_vortex_color=anti_vortex_color, vortex_alpha=vortex_alpha,
                          show_grid=show_grid, grid_width=grid_width,
                          grid_color=grid_color, grid_alpha=grid_alpha,
                          show_colorbar=show_colorbar, show_arrows=show_arrows,
                          arrow_quantity=arrow_quantity, arrow_width=arrow_width, arrow_scale=arrow_scale,
                          arrow_headwidth=arrow_headwidth, arrow_headlength=arrow_headlength,
                          arrow_headaxislength=arrow_headaxislength, arrow_minshaft=arrow_minshaft,
                          arrow_minlength=arrow_minlength, arrow_color=arrow_color,
                          arrow_alpha=arrow_alpha, show_nodes=show_nodes, node_diameter=node_diameter,
                          node_face_color=node_face_color, node_edge_color=node_edge_color,
                          node_alpha=node_alpha, show_node_quantity=show_node_quantity,
                          node_quantity=node_quantity, node_quantity_cmap=node_quantity_cmap,
                          node_quantity_clim=node_quantity_clim, node_quantity_alpha=node_quantity_alpha,
                          node_quantity_logarithmic_colors=node_quantity_logarithmic_colors,
                          show_face_quantity=show_face_quantity, face_quantity=face_quantity,
                          face_quantity_cmap=face_quantity_cmap, face_quantity_clim=face_quantity_clim,
                          face_quantity_alpha=face_quantity_alpha,
                          face_quantity_logarithmic_colors=face_quantity_logarithmic_colors,
                          figsize=figsize, animate_interval=animate_interval, title=title).show()

    def __str__(self):
        return "time evolution configuration: (" + ("th" + self.theta.shape.__str__() + ", ") * (
                    self.theta is not None) + \
               ("I" + self.current.shape.__str__() + ", ") * (self.current is not None) + \
               ("V" + self.voltage.shape.__str__()) * (self.current is not None) + ")" + \
               "\nproblem: " + self.problem.__str__() + \
               "\ncircuit: " + self.get_circuit().__str__()

    def _select(self, select_time_points, N, func):
        select_time_points = np.flatnonzero(self.problem._to_time_point_mask(select_time_points))
        W = self.get_problem_count()
        out = np.zeros((N, W, len(select_time_points)), dtype=np.double)
        for i, tp in enumerate(select_time_points):
            out[:, :, i] = func(tp)
        return out

class AnnealingProblem:

    def __init__(self, circuit: Circuit, time_step=0.5, time_step_count=10 ** 4,
                 frustration=0.0, current_sources=0, search_T=0.0,
                 problem_count=1, select_lowest_energy=False):
        self.circuit = circuit
        self.time_step = time_step
        self.time_step_count = time_step_count
        self.current_sources = current_sources
        self.frustration = frustration
        self.search_T = search_T
        self.problem_count = problem_count
        self.select_lowest_energy = select_lowest_energy

    def compute(self):
        """
        Find low energy state by using a time evolution simulation with a slowly decreasing temperature
        (called annealing). The duration and temperature profile can be controlled.
        """

        store_time_steps = np.zeros(self.time_step_count, dtype=bool)
        store_time_steps[-1] = True
        T = DEF_TEMPERATURE_PROFILE(self.search_T)(np.linspace(0, 1, self.time_step_count))
        f = self.frustration * np.ones((1, self.problem_count), dtype=np.double)
        prob = TimeEvolutionProblem(self.circuit, time_step_count=self.time_step_count, time_step=self.time_step,
                                    frustration=f, current_sources=self.current_sources, temperature=T[None, None, :],
                                    store_current=False, store_voltage=False,
                                    store_time_steps=store_time_steps)
        out = prob.compute()
        vortex_configurations = out.get_n()[..., 0]
        data = [prob.get_static_problem(vortex_configurations[:, p], problem_nr=0, time_step=0).compute()
                for p in range(self.problem_count)]
        configurations = [d[0] for d in data]
        energies = [np.mean(d[0].get_Etot()) for d in data]
        print(energies)
        status = [d[1] for d in data]
        if self.select_lowest_energy:
            lowest_index = np.argmin(energies)
            return vortex_configurations[:, lowest_index], energies[lowest_index], status[lowest_index], configurations[lowest_index]
        return vortex_configurations, energies, status, configurations


class AnnealingProblem2:

    def __init__(self, circuit: Circuit, time_step=0.5, interval_steps=10,
                 interval_count=1000, vortex_mobility=0.001,
                 frustration=0.0, current_sources=0, problem_count=1,
                 start_T=1.0, T_factor=1.03):
        self.circuit = circuit
        self.time_step = time_step
        self.interval_steps = interval_steps
        self.interval_count = interval_count
        self.vortex_mobility = vortex_mobility
        self.current_sources = current_sources
        self.frustration = frustration
        self.problem_count = problem_count
        self.T =start_T * np.ones((1, self.problem_count, 1))
        self.T_factor = T_factor

    def get_vortex_mobility(self, n):
        Nf = self.circuit.face_count()
        return np.sum(np.sum(np.abs(np.diff(n, axis=2)), axis=2), axis=0) / (Nf * self.time_step * (self.interval_count - 1))

    def temperature_adjustment(self, vortex_mobility, iteration):
        v = self.vortex_mobility
        upper = v[iteration] if (np.array(v)).size == self.interval_count else \
            v * ((self.interval_count - iteration) / self.interval_count) ** 1.5
        factor = (vortex_mobility > upper) * (1/self.T_factor) +  (vortex_mobility <= upper) * self.T_factor
        self.T *= factor[..., None]

    def compute(self):
        """
        Find low energy state by using a time evolution simulation with a slowly decreasing temperature
        (called annealing). The duration and temperature profile can be controlled.
        """

        f = self.frustration * np.ones((1, self.problem_count), dtype=np.double)
        th = np.zeros((self.circuit.junction_count(), self.problem_count))
        prob = TimeEvolutionProblem(self.circuit, time_step_count=self.interval_steps, time_step=self.time_step,
                                    frustration=f, current_sources=self.current_sources, temperature=self.T,
                                    store_current=False, store_voltage=False)
        T = np.zeros((self.interval_count, self.problem_count))
        for i in range(self.interval_count):
            prob.temperature = self.T * np.ones((1, 1, self.interval_steps))
            prob.config_at_minus_1 = th
            out = prob.compute()
            vortex_configurations = out.get_n()
            vortex_mobility = self.get_vortex_mobility(vortex_configurations)
            self.temperature_adjustment(vortex_mobility, i)
            th = out.get_theta()[..., -1]
            # stat_conf = out.select_static_configuration(prob_nr=0, time_step=self.interval_steps-1)
            # print(stat_conf.get_error())
            T[i, :] = self.T[0, :, 0]
        prob.temperature = np.zeros((1, 1, self.interval_steps))
        prob.time_step /=2
        for i in range(1 + (self.interval_count//5)):
            prob.config_at_minus_1 = th
            out = prob.compute()
        vortex_configurations = out.get_n()[:, :, -1]
        data = [prob.get_static_problem(vortex_configurations[:, p], problem_nr=0, time_step=0).compute()
                for p in range(self.problem_count)]
        configurations = [d[0] for d in data]
        energies = [np.mean(d[0].get_Etot()) for d in data]
        status = np.array([d[1] for d in data])
        return vortex_configurations, energies, status, configurations, T













