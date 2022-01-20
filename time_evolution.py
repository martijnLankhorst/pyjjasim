import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.optimize

from pyjjasim.josephson_circuit import Circuit
from pyjjasim.static_problem import DefaultCPR
from pyjjasim.static_problem import StaticConfiguration, StaticProblem

__all__ = ["TimeEvolutionProblem", "TimeEvolutionResult", "AnnealingProblem"]

"""
Time Evolution Module
"""

class TimeEvolutionProblem:
    """

    Define multiple time evolution problems with varying parameters in a Josephson circuit.

     - All problems are computed in one call (ofter faster than computing one by one).
     - The problem count is abbreviated with the symbol W.
     - Use self.compute() to obtain a TimeEvolutionResult object containing the
       resulting time evolution.
     - Be careful input arrays for Is, f, etc. have the correct shape. See numpy
       broadcasting for details https://numpy.org/doc/stable/user/basics.broadcasting.html
       Remember Nj is number of junctions, Nn is number of nodes and Nf is number of faces.

    Parameters
    ----------
    circuit : Circuit
        Josephson circuit on which the problem is based.
    time_step=0.05 :
        Time between consecutive steps (abbreviated dt).
    time_step_count=1000 :         Nt
        Total number of time steps in evolution (abbreviated Nt).
    current_phase_relation= DefaultCPR()
        Current-phase relation (abbreviated cp).
    frustration=0.0 : array broadcastable to (Nf, W, Nt).
        Normalized external flux per site, called frustration (abbreviated f)
    frustration (alternative) : f(i) -> array broadcastable to (Nf, W) for i in range(Nt)
        Alternative input type for frustration using a function.
    current_sources : array broadcastable to (Nj, W, Nt)
        Current source strength at each junction in circuit (abbreviated Is).
    current_sources (alternative) : Is(i) -> array broadcastable to (Nj, W) for i in range(Nt)
        Alternative input type for current_sources using a function.
    voltage_sources=0.0 : array broadcastable to (Nj, W, Nt)
        Voltage source value at each junction in circuit (abbreviated Vs).
    voltage_sources (alternative) : Vs(i) -> array broadcastable to (Nj, W) for i in range(Nt)
        Alternative input type for voltage_sources using a function.
    temperature=0.0 : array broadcastable to (Nj, W, Nt)
        Temperature at each junction in circuit (abbreviated T).
    temperature (alternative) : T(i) -> array broadcastable to (Nj, W) for i in range(Nt)
        Alternative input type for temperature using a function.
    store_time_steps=None : array in range(Nt), mask of shape (Nt,) or None
        Indicate at which timesteps data is stored in the output array(s).
    store_theta=True : bool
        If true, theta (the gauge invarian t phase difference) is stored during
        a time evolution (at timesteps specified with store_time_steps) and
        returned when simulation is done.
    store_voltage=True : bool
        If true, voltage is stored during a time evolution (at timesteps specified
        with store_time_steps) and returned when simulation is done.
    store_current=True : bool
        If true, voltage is stored during a time evolution (at timesteps specified
        with store_time_steps) and returned when simulation is done.
    config_at_minus_1=None : (Nj, W) array or StaticConfiguration or None
        initial condition at timestep=-1. set to self.get_static_problem(t=0,n=z).compute()
    config_at_minus_2=None : (Nj, W) array or StaticConfiguration or None
        initial condition at timestep=-2

    Notes
    -----
     - All physical quantities are dimensionless. See the UserManual (on github)
       for how all quantities are normalized.
     - It is assumed each junction has a current source and voltage source see
       user manual (on github) for diagram of junction. To omit the sources in
       particular junctions set the respective values to zero.
     - To use a node-based source current (represented as an (Nn,) array Is_node
       with current in/e-jected at each node), convert it to a junction-based
       source with Is = static_problem.node_to_junction_current(circuit, Is_node)
       and use Is as input for a time evolution.
     - The store parameters allow one to control what quantities are stored at what timesteps.
       Only the base quantities theta, voltage and current can be stored in the resulting
       TimeEvolutionResult; other quantities like energy or magnetic_flux are computed based
       on these. See documentation of `TimeEvolutionResult` to see per  quantity which
       base quantities are required.
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
        if hasattr(self.config_at_minus_1, "get_theta"):
            self.config_at_minus_1.get_theta()[:, None]
        self.config_at_minus_2 = np.zeros((Nj, W), dtype=np.double) if config_at_minus_2 is None else config_at_minus_2
        if hasattr(self.config_at_minus_2, "get_theta"):
            self.config_at_minus_2.get_theta()[:, None]
        self.prepared_theta_s = np.zeros((Nj, W), dtype=np.double)

    def get_static_problem(self, vortex_configuration, problem_nr=0, time_step=0) -> StaticProblem:
        """
        Return a static problem with properties copied from this time evolution.

        Parameters
        ----------
        vortex_configuration : (Nf,) array or None
            Vortex configuration of returned static problem.
        problem_nr=0 : int in range(W)
            Selected problem number to copy properties of.
        time_step=0 : int in range(Nt)
            Selected timestep to copy properties of.
        Returns
        -------
        static_problem : StaticProblem
            Static problem where the parameters are copied from this time evolution.
        """
        return StaticProblem(self.circuit, current_sources=self._Is(time_step)[:, problem_nr].copy(),
                             frustration=self._f(time_step)[:, problem_nr].copy(),
                             vortex_configuration=vortex_configuration,
                             current_phase_relation=self.current_phase_relation)

    def get_problem_count(self):
        """
        Return number of problems (abbreviated W).
        """
        return self.problem_count

    def get_circuit(self) -> Circuit:
        """
        Returns the circuit.
        """
        return self.circuit

    def get_time_step(self):
        """
        Return the timestep (abbreviated dt).
        """
        return self.time_step

    def get_time_step_count(self):
        """
        Return the number of timesteps (abbreviated Nt).
        """
        return self.time_step_count

    def get_current_phase_relation(self):
        """
        Returns the current-phase relation.
        """
        return self.current_phase_relation

    def get_phase_zone(self):
        """
        Returns the phase zone (In all of pyJJAsim phase_zone=0).
        """
        return 0

    def get_frustration(self):
        """
        Returns the frustration (abbreviated f).
        """
        return self.frustration

    def get_current_sources(self):
        """
        Returns the current sources (abbreviated Is).
        """
        return self.current_sources

    def get_net_sourced_current(self, time_step):
        """
        Gets the sum of all (positive) current injected at nodes to create Is.

        Parameters
        ----------
        time_step : int
            Time step at which to return net sourced current.

        Returns
        -------
        net_sourced_current : (W,) array
            Net sourced current through array for each problem at specified timestep.
        """
        M = self.get_circuit().get_cut_matrix()
        return 0.5 * np.sum(np.abs((M @ self._Is(time_step))), axis=0)

    def get_node_current_sources(self, time_step):
        """
        Returns (Nn,) array of currents injected at nodes to create Is.

        Parameters
        ----------
        time_step : int
            Time step at which to return node currents.

        Returns
        -------
        node_current : (Nn, W) array
            Node currents for each problem at specified timestep.
        """
        M = self.get_circuit().get_cut_matrix()
        return M @ self._Is(time_step)

    def get_voltage_sources(self):
        """
        Return voltage sources at each junction (abbreviated Vs)
        """
        return self.voltage_sources

    def get_temperature(self):
        """
        Return temperature at each junction (abbreviated T)
        """
        return self.temperature

    def get_store_time_steps(self):
        """
        Return at which timesteps data is stored in the output array(s).

    config_at_minus_1=None : (Nj, W) array or StaticConfiguration or None
        initial condition at timestep=-1. set to self.get_static_problem(t=0,n=z).compute()
    config_at_minus_2=None : (Nj, W) array or StaticConfiguration or None
        initial condition at timestep=-2
        """
        return self.store_time_steps

    def get_store_theta(self):
        """
        Return if theta is stored during a time evolution.
        """
        return self.store_theta

    def get_store_voltage(self):
        """
        Return if voltage is stored during a time evolution.
        """
        return self.store_voltage

    def get_store_current(self):
        """
        Return if current is stored during a time evolution.
        """
        return self.store_current

    def get_time(self):
        """
        Return (Nt,) array with time value at each step of evolution.
        """
        return np.arange(self._Nt(), dtype=np.double) * self._dt()

    def compute(self):
        """
        Compute time evolution on an Josephson Circuit.
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
        Ic = self.get_circuit()._Ic()[:, None]
        return self.current_phase_relation.eval(Ic, theta)

    def _dcp(self, theta) -> np.ndarray:  # (Nj, W)
        Ic = self.get_circuit()._Ic()[:, None]
        return self.current_phase_relation.d_eval(Ic, theta)

    def _icp(self, theta) -> np.ndarray:  # (Nj, W)
        Ic = self.get_circuit()._Ic()[:, None]
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

def time_evolution_algo_0(problem: TimeEvolutionProblem):
    """
    Algorithm 0 for time-evolution. Does not allow for mixed inductance, but if it
    does it is usually faster than algorithm 1.
    """

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
    Rv = 1 / (dt * circuit._R()[:, None])
    Cv = circuit._C()[:, None] / (dt ** 2)
    Cprev, C0, Cnext = Cv, -2.0 * Cv - Rv, Cv + Rv

    if circuit._has_inductance():
        L = problem.circuit._L()
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

def time_evolution_algo_1(problem: TimeEvolutionProblem):
    """
    Algorithm 1 for time-evolution. Allows for mixed inductance, but is
    usually slower than algorithm 0.
    """

    out = TimeEvolutionResult(problem)

    circuit = problem.circuit
    Nj, Nf, W = circuit._Nj(),  circuit._Nf(), problem.get_problem_count()
    dt = problem._dt()

    store_th, store_I, store_V = problem.store_theta, problem.store_current, problem.store_voltage

    A = circuit.get_cycle_matrix()
    M = circuit._Mr()
    Rv = 1 / (dt * circuit._R()[:, None])
    Cv = circuit._C()[:, None] / (dt ** 2)
    Cprev, C0, Cnext = Cv, -2.0 * Cv - Rv, Cv + Rv

    L = problem.circuit._L()
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


class ThetaNotStored(Exception):
    pass

class CurrentNotStored(Exception):
    pass

class VoltageNotStored(Exception):
    pass

class DataAtTimepointNotStored(Exception):
    pass

class TimeEvolutionResult:
    """
    Represents data of simulated time evolution(s) on a Josephson circuit.

    One can query several properties of the circuit configurations, like currents
    and voltages. TimeEvolutionResult only store theta, current and voltage data
    from which all quantities are computed. However, one can choose not to store
    all three to save memory. An error is raised if one attempts to compute a property
    for which the required data is not stored.
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
        self.animation = None

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
            raise ThetaNotStored("Cannot query theta; quantity is not stored during time evolution simulation.")
        return self.theta[:, :, self._time_point_index(time_point)]

    def _V(self, time_point) -> np.ndarray:
        if self.theta is None:
            raise VoltageNotStored("Cannot query voltage; quantity is not stored during time evolution simulation.")
        return self.voltage[:, :, self._time_point_index(time_point)]

    def _I(self, time_point) -> np.ndarray:
        if self.theta is None:
            raise CurrentNotStored("Cannot query current; quantity is not stored during time evolution simulation.")
        return self.current[:, :, self._time_point_index(time_point)]

    def _time_point_index(self, time_points):
        if time_points is None:
            time_points = self.problem.store_time_steps
        if not np.all(self.problem.store_time_steps[time_points]):
            raise DataAtTimepointNotStored("Queried a timepoint that is not stored during time evolution simulation.")
        return self.time_point_indices[time_points]

    def get_problem_count(self):
        """
        Return number of problems (abbreviated W).
        """
        return self.problem.get_problem_count()

    def get_circuit(self) -> Circuit:
        """
        Return Josephson circuit.
        """
        return self.problem.get_circuit()

    def select_static_configuration(self, prob_nr, time_step) -> StaticConfiguration:
        """
        Return a StaticConfiguration object with the data copied from this result for
        a given problem number at a given timestep.

        Parameters
        ----------
        prob_nr : int
            Selected problem.
        time_step : int
            Selected timestep.

        Returns
        -------
        static_conf : StaticConfiguration
            A StaticConfiguration object with the data copied from this result
        """
        if self.theta is None:
            raise ValueError("Theta not stored; cannot select static configuration.")
        problem = StaticProblem(self.get_circuit(), current_sources=self.problem._Is(time_step)[:, prob_nr],
                                frustration=self.problem._f(time_step)[:, prob_nr],
                                vortex_configuration=self.get_n(time_step)[:, prob_nr],
                                current_phase_relation=self.problem.current_phase_relation)
        return StaticConfiguration(problem, self.theta[:, prob_nr, time_step])

    def get_phi(self, select_time_points=None) -> np.ndarray:
        """
        Return node phases. Last node is grounded. Requires theta to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        phi : (Nn, W, nr_of_selected_timepoints) array
             Node phases.
        """
        M = self.get_circuit()._Mr()
        Mrsq = M @ M.T
        Z = np.zeros((1, self.get_problem_count()), dtype=np.double)
        def stretch(x):
            return x if x.ndim == 2 else x[:, None]
        solver = scipy.sparse.linalg.factorized(Mrsq)
        func = lambda tp: np.concatenate((stretch(solver(M @ self._th(tp))), Z), axis=0)
        return self._select(select_time_points, self.get_circuit()._Nn(), func)

    def get_theta(self, select_time_points=None) -> np.ndarray:
        """
        Return gauge invariant phase differences.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        theta : (Nj, W, nr_of_selected_timepoints) array
             Gauge invariant phase differences.
        """
        return self._select(select_time_points, self.get_circuit()._Nj(), self._th)

    def get_n(self, select_time_points=None) -> np.ndarray:
        """
        Return vorticity at faces. Requires theta to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        n : (Nf, W, nr_of_selected_timepoints) int array
             Vorticity.
        """
        A = self.get_circuit().get_cycle_matrix()
        func = lambda tp:  -A @ np.round(self._th(tp) / (2.0 * np.pi))
        return self._select(select_time_points, self.get_circuit()._Nf(), func).astype(int)

    def get_EJ(self, select_time_points=None) -> np.ndarray:
        """
        Return Josephson energy of junctions. Requires theta to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        EJ : (Nf, W, nr_of_selected_timepoints) array
             Josephson energy.
        """
        func = lambda tp: self.problem._icp(self._th(tp))
        return self._select(select_time_points, self.get_circuit()._Nj(), func)

    def get_I(self, select_time_points=None) -> np.ndarray:
        """
        Return current through junctions.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        I : (Nj, W, nr_of_selected_timepoints) array
             Current.
        """
        return self._select(select_time_points, self.get_circuit()._Nj(), self._I)

    def get_Isup(self, select_time_points=None) -> np.ndarray:
        """
        Return supercurrent through junctions. Requires theta to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        Isup : (Nj, W, nr_of_selected_timepoints) array
             Supercurrent.
        """
        func = lambda tp: self.problem._cp(self._th(tp))
        return self._select(select_time_points, self.get_circuit()._Nj(), func)

    def get_J(self, select_time_points=None) -> np.ndarray:
        """
        Return cycle-current around faces. Requires current to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        J : (Nf, W, nr_of_selected_timepoints) array
             Cycle-current around faces.
        """
        A = self.get_circuit().get_cycle_matrix()
        solver = scipy.sparse.linalg.factorized(A @ A.T)
        func = lambda tp: solver(A @ self._I(tp))
        return self._select(select_time_points, self.get_circuit()._Nf(), func)

    def get_flux(self, select_time_points=None) -> np.ndarray:
        """
        Return magnetic flux through faces. Requires current to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        flux : (Nf, W, nr_of_selected_timepoints) array
             Magnetic flux through faces
        """
        Nj, Nf = self.get_circuit()._Nj(), self.get_circuit()._Nf()
        A = self.get_circuit().get_cycle_matrix()
        func = lambda tp: A @ (self.get_circuit()._L() @ self._I(tp))
        return self._select(select_time_points, Nf, func)

    def get_EM(self, select_time_points=None) -> np.ndarray:
        """
        Return magnetic energy associated with wires. Requires current to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        EM : (Nf, W, nr_of_selected_timepoints) array
             Magnetic energy associated with wires.
        """
        Nj = self.get_circuit()._Nj()
        func = lambda tp: 0.5 * self.get_circuit()._L() @ (self._I(tp) ** 2)
        return self._select(select_time_points, Nj, func)

    def get_V(self, select_time_points=None):
        """
        Return voltage over junctions.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        V : (Nj, W, nr_of_selected_timepoints) array
             Voltage.
        """
        return self._select(select_time_points, self.get_circuit()._Nj(), self._V)

    def get_U(self, select_time_points=None):
        """
        Return voltage potential at nodes. Last node is groudend.Requires
        voltage to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        U : (Nn, W, nr_of_selected_timepoints) array
             Voltage potential at nodes.
        """
        M = self.get_circuit()._Mr()
        Mrsq = M @ M.T
        Z = np.zeros((1, self.get_problem_count()), dtype=np.double)
        solver = scipy.sparse.linalg.factorized(Mrsq)
        func = lambda tp: np.concatenate((solver(M @ self._V(tp)), Z), axis=0)
        return self._select(select_time_points, self.get_circuit()._Nn(), func)

    def get_EC(self, select_time_points=None):
        """
        Return energy stored in capacitors at each junction. Requires voltage
        to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        EC : (Nj, W, nr_of_selected_timepoints) array
            Energy stored in capacitors
        """
        C, Nj = self.get_circuit()._C(), self.get_circuit()._Nj()
        func = lambda tp: 0.5 * C[:, None] * self._V(tp) ** 2
        return self._select(select_time_points, Nj, func)

    def get_Etot(self, select_time_points=None) -> np.ndarray:
        """
        Return total energy associated with each junction. Requires theta,
        current and voltage to be stored.

        Parameters
        ----------
        select_time_points=None : array in range(Nt), (Nt,) mask or None
            Selected time_points at which to return data. If None, all stored
            timepoints are returned. Raises error if no data is available at
            requested timepoint.

        Returns
        -------
        Etot : (Nj, W, nr_of_selected_timepoints) array
            Total energy associated with each junction.
        """
        return self.get_EJ(select_time_points) + self.get_EM(select_time_points) + \
               self.get_EC(select_time_points)

    def plot(self, problem_nr=0, time_point=0, fig=None,
                node_quantity=None, junction_quantity="I", face_quantity=None,
                vortex_quantity="n", show_grid=True, show_nodes=True, **kwargs):
        """
        Visualize a problem at a specified timestep.

        See :py:attr:`circuit_visualize.CircuitPlot` for documentation.
        """
        return self.animate(problem_nr=problem_nr, time_points=np.array([time_point]),
                            node_quantity=node_quantity, junction_quantity=junction_quantity,
                            face_quantity=face_quantity, vortex_quantity=vortex_quantity,
                            show_grid=show_grid, show_nodes=show_nodes, fig=fig, **kwargs)

    def animate(self, problem_nr=0, time_points=None, fig=None,
                node_quantity=None, junction_quantity="I", face_quantity=None,
                vortex_quantity="n", show_grid=True, show_nodes=True, **kwargs):

        """
        Animate time evolution of a problem as a movie.

        See :py:attr:`circuit_visualize.TimeEvolutionMovie` for documentation.
        """

        from pyjjasim.circuit_visualize import TimeEvolutionMovie

        self.animation = TimeEvolutionMovie(self, problem_nr=problem_nr, time_points=time_points,
                                            vortex_quantity=vortex_quantity, show_grid=show_grid,
                                            show_nodes=show_nodes, junction_quantity=junction_quantity,
                                            node_quantity=node_quantity, face_quantity=face_quantity,
                                            fig=fig, **kwargs).make()
        return self.animation

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

    """
    Anneals a circuit by gradually lowering the temperature, with the goal for finding a stationairy
    state with reasonably low energy. The temperature profile is computed automatically based on the
    measured vortex mobility during the run. Annealing is executed with the .compute() method.

     - Does interval_count iterations of interval_steps timeseteps.
     - The first iteration is done at T=start_T
     - During each iteration it measures the average vortex mobility. If the target mobility is exceeded,
       the temperature is divided by T_factor, otherwise it is multiplied by it.
     - The target vortex mobility v_t at iter i  is v_t(i) = v * ((N - i)/N) ** 1.5, so goes from v to 0.
     - At the end it does some steps at T=0 to ensure it is settled in a stationairy state.
     - vortex mobility is defined as abs(n(i+1) - n(i))) / dt averaged over space and time.

    Parameters
    ----------
    circuit : Circuit
        Circuit used for annealing
    time_step=0.5 : float
        time step used in time evolution. Can be set quite large as the simulation does not
        need to be accurate.
    frustration=0.0 : float or (Nf,) array
        Frustration at each face. If scalar; same frustration is assumed for each face.
    current_sources=0 : float or (Nj,) array
        Current sources. Note that if this is set too high, a static configuration
        may not exist, so likely the temperature will go to zero.
    problem_count=1 : int
        Number of problems computed simultaneously. The problems are identical,
        but will likely end up in different state.
    interval_steps=10 : int
        Number of timesteps for every iteration.
    interval_count=1000 : int
        Number of iterations. After each iteration the temperature is recomputed.
    vortex_mobility=0.001 : float
        Target vortex mobility is vortex_mobility at iteration and lowered to
        zero by the last iteration. Temperature is adjusted such that this
        target mobility is achieved.
    start_T=1.0 : float
        Temperature at first iteration.
    T_factor=1.03 : float
        Factor with which temperature is multiplied or divided by every iteration.
    """
    def __init__(self, circuit: Circuit, time_step=0.5, interval_steps=10,
                 frustration=0.0, current_sources=0, problem_count=1,
                 interval_count=1000, vortex_mobility=0.001,
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
        """
        Computes vortex mobility on a set of consecutive vortex configurations.
        """
        Nf = self.circuit.face_count()
        return np.sum(np.sum(np.abs(np.diff(n, axis=2)), axis=2), axis=0) / (Nf * self.time_step * (self.interval_count - 1))

    def _temperature_adjustment(self, vortex_mobility, iteration):
        v = self.vortex_mobility
        upper = v[iteration] if (np.array(v)).size == self.interval_count else \
            v * ((self.interval_count - iteration) / self.interval_count) ** 1.5
        factor = (vortex_mobility > upper) * (1/self.T_factor) +  (vortex_mobility <= upper) * self.T_factor
        self.T *= factor[..., None]

    def compute(self):
        """
        Executes the annealing procedure.

        Returns
        -------
        status : (problem_count,) int array
            The value 0 means converged, 1 means diverged, 2 means indeterminate status.
        configurations : (problem_count,) list
            A list of StaticConfiguration objects containing the resulting configurations.
        temperature_profiles : (interval_count, problem_count) array
            For each problem the temperature profile used for the annealing.
        """

        # prepare runs
        f = np.atleast_1d(self.frustration)[:, None, None]
        th = np.zeros((self.circuit.junction_count(), self.problem_count))
        prob = TimeEvolutionProblem(self.circuit, time_step_count=self.interval_steps, time_step=self.time_step,
                                    frustration=f, current_sources=self.current_sources, temperature=self.T,
                                    store_current=False, store_voltage=False)
        temperature_profiles = np.zeros((self.interval_count, self.problem_count))

        # Do interval_count runs of interval_steps steps. After each step update temperature.
        for i in range(self.interval_count):
            prob.temperature = self.T * np.ones((1, 1, self.interval_steps))
            prob.config_at_minus_1 = th
            out = prob.compute()
            vortex_configurations = out.get_n()
            vortex_mobility = self.get_vortex_mobility(vortex_configurations)
            self._temperature_adjustment(vortex_mobility, i)
            th = out.get_theta()[..., -1]
            temperature_profiles[i, :] = self.T[0, :, 0]

        # finish with some runs at T=0 with half the timestep
        prob.temperature = np.zeros((1, 1, self.interval_steps))
        prob.time_step /=2
        for i in range(5):
            prob.config_at_minus_1 = th
            out = prob.compute()
            th = out.get_theta()[..., -1]

        # extract result
        vortex_configurations = out.get_n()[:, :, -1]
        data = [prob.get_static_problem(vortex_configurations[:, p], problem_nr=0, time_step=0).compute()
                for p in range(self.problem_count)]
        configurations = [d[0] for d in data]
        status = np.array([d[1] for d in data])
        return status, configurations, temperature_profiles













