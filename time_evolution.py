import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.optimize
from numba import jit

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
                 config_at_minus_2: np.ndarray = None,
                 config_at_minus_3: np.ndarray = None,
                 config_at_minus_4: np.ndarray = None,
                 stencil_width=3):

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

        self._f_is_timedep = TimeEvolutionProblem._is_timedep(frustration)
        self.frustration = frustration if hasattr(frustration, "__call__") else \
            np.broadcast_to(np.array(frustration), (Nf, W, Nt))
        self._Is_is_timedep = TimeEvolutionProblem._is_timedep(current_sources)
        self.current_sources = current_sources if hasattr(current_sources, "__call__") else \
            np.broadcast_to(np.array(current_sources), (Nj, W, Nt))
        self._Vs_is_timedep = TimeEvolutionProblem._is_timedep(voltage_sources)
        self.voltage_sources = voltage_sources if hasattr(voltage_sources, "__call__") else \
            np.broadcast_to(np.array(voltage_sources), (Nj, W, Nt))
        self._T_is_timedep = TimeEvolutionProblem._is_timedep(temperature)
        self.temperature = temperature if hasattr(temperature, "__call__") else \
            np.broadcast_to(np.array(temperature), (Nj, W, Nt))

        self.store_time_steps = np.ones(self._Nt(), dtype=bool)
        self.store_time_steps = self._to_time_point_mask(store_time_steps)
        self.store_theta = store_theta
        self.store_voltage = store_voltage
        self.store_current = store_current
        if not (self.store_theta or self.store_voltage or self.store_current):
            raise ValueError("No output is stored")
        if np.sum(self.store_time_steps) == 0:
            raise ValueError("No output is stored")
        self.stencil_width = stencil_width
        self.stencil = self._get_stencil(self.stencil_width)

        self.config_at_minus_1 = self._get_config(config_at_minus_1, np.zeros((Nj, W), dtype=np.double), (Nj, W))
        self.config_at_minus_2 = self._get_config(config_at_minus_2, self.config_at_minus_1, (Nj, W))
        if self.stencil_width >= 4:
            self.config_at_minus_3 = self._get_config(config_at_minus_3, self.config_at_minus_2, (Nj, W))
        if self.stencil_width >= 5:
            self.config_at_minus_4 = self._get_config(config_at_minus_4, self.config_at_minus_3, (Nj, W))

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

    def write_cir(self):
        pass


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
        return time_evolution(self)

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

    @staticmethod
    def _get_config(config_cur, config_prev, shape):
        config_cur = config_prev.copy() if config_cur is None else config_cur
        if hasattr(config_cur, "get_theta"):
            config_cur = config_cur.get_theta()
        return config_cur.reshape(shape)  # always (Nj, W) shaped

    @staticmethod
    def _is_timedep(x):
        if len(np.array(x).shape) == 0:
            return False
        return hasattr(x, "__call__") or np.array(x).shape[-1] > 1

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

    def _get_stencil(self, width : int):
        if width == 3:
            return (1.0, -1.0, 0.0), (1.0, -2.0, 1.0)
        if width == 4:
            return (1.5, -2.0, 0.5, 0.0), (2.0, -5.0, 4.0, -1.0)
        if width == 5:
            return (11.0/6, -3.0, 1.5, -1.0/3, 0.0), (35.0/12, -26.0/3, 9.5, -14.0/3, 11.0/12)
        raise ValueError(f"stencil width must be 3, 4 or 5 (equals {width})")


def _apply_derivative(x, index, stencil, dt):
    w = len(stencil)
    if w == 3:
        return (stencil[0] * x[:, :, index] + stencil[1] * x[:, :, index - 1]) / dt
    if w == 4:
        return (stencil[0] * x[:, :, index] + stencil[1] * x[:, :, index - 1] +
                stencil[2] * x[:, :, index - 2]) / dt
    if w == 5:
        return (stencil[0] * x[:, :, index] + stencil[1] * x[:, :, index - 1] +
                stencil[2] * x[:, :, index - 2] + stencil[3] * x[:, :, index - 3]) / dt

def time_evolution(problem: TimeEvolutionProblem):
    # determine what timepoints to store. Complicated by the fact that voltage needs both derivative
    # of theta and current (if inductance is present) for which extra timepoints must be stored depending
    # on the derivative stencil.
    th_store_mask = problem.store_time_steps.copy() if (problem.store_theta or problem.store_voltage) else np.zeros(problem._Nt(), dtype=bool)
    I_store_mask = problem.store_time_steps.copy() if (problem.store_current or problem.store_voltage) else np.zeros(problem._Nt(), dtype=bool)
    V_th_store_mask = th_store_mask.copy()
    V_I_store_mask = I_store_mask.copy()
    t_ids = np.flatnonzero(problem.store_time_steps)
    Nj = problem.circuit.junction_count()
    offset = problem.stencil_width - 1 # number of initial conditions used
    if problem.store_voltage:
        if len(t_ids) > 0:
            Vt_ids = (t_ids[:, None] - np.arange(offset)).ravel()
            Vt_ids = Vt_ids[(Vt_ids >= 0) & (Vt_ids < problem._Nt())]
            V_th_store_mask[Vt_ids] = True
            if problem.circuit._has_inductance():
                V_I_store_mask[Vt_ids] = True

    # th_out will be of shape (Nj, W, offset + sum(V_th_store_mask)). Note that the initial
    # conditions before time_point=0 are always stored.
    th_out, I_out = time_evolution_core(problem, V_th_store_mask, V_I_store_mask)

    if problem.store_voltage:
        ts = np.flatnonzero((problem.store_time_steps)[V_th_store_mask])
        V_out = _apply_derivative(th_out, index=ts + offset, stencil=problem.stencil[0], dt=problem._dt())
        if problem.circuit._has_inductance():
            V_ind = _apply_derivative(I_out, index=ts + offset, stencil=problem.stencil[0], dt=problem._dt())
            V_out += (problem.circuit.get_inductance_factors() @ V_ind.reshape((Nj, -1))).reshape(V_out.shape)
        th_out = np.delete(th_out, np.flatnonzero((V_th_store_mask & ~ th_store_mask)[V_th_store_mask]) + offset, axis=2)
        I_out = np.delete(I_out, np.flatnonzero((V_I_store_mask & ~ I_store_mask)[V_I_store_mask]) + offset, axis=2)

    th_out = th_out[:, :, offset:]  # remove initial conditions -> shape=(Nj, W, sum(th_store_mask))
    I_out = I_out[:, :, offset:]
    return TimeEvolutionResult(problem, th_out if problem.store_theta else None,
                               I_out if problem.store_current else None,
                               V_out if problem.store_voltage else None)


def time_evolution_core(problem: TimeEvolutionProblem, th_store_mask, I_store_mask):
    """
    Algorithm 2 for time-evolution. Best algo?
    """

    circuit = problem.get_circuit()
    Nj, W = circuit._Nj(), problem.get_problem_count()
    dt = problem._dt()

    A = circuit.get_cycle_matrix()
    AT = A.T
    Rv = 1 / (dt * circuit._R()[:, None])
    Cv = circuit._C()[:, None] / (dt ** 2)
    C1, C2 = problem.stencil

    Cprev, C0, Cnext = Cv, -2.0 * Cv - Rv, Cv + Rv
    c0 = C1[0] * Rv + C2[0] * Cv
    c1 = C1[1] * Rv + C2[1] * Cv

    theta_next = problem.config_at_minus_1.copy()

    s_width = problem.stencil_width

    th_out = np.zeros((Nj, W, np.sum(th_store_mask) + s_width - 1), dtype=np.double)
    I_out = np.zeros((Nj, W, np.sum(I_store_mask) + s_width - 1), dtype=np.double)
    th_out[:, :, s_width - 2] = theta_next
    I_out[:, :, s_width - 2] = problem._cp(theta_next)

    c2 = C1[2] * Rv + C2[2] * Cv
    theta1 = problem.config_at_minus_2.copy()
    th_out[:, :, s_width - 3] = theta1
    I_out[:, :, s_width - 3] = problem._cp(theta1)
    if s_width >= 4:
        c3 = C1[3] * Rv + C2[3] * Cv
        theta2 = problem.config_at_minus_3.copy()
        th_out[:, :, s_width - 4] = theta2
        I_out[:, :, s_width - 4] = problem._cp(theta2)
    if s_width >= 5:
        c4 = C1[4] * Rv + C2[4] * Cv
        theta3 = problem.config_at_minus_4.copy()
        th_out[:, :, s_width - 5] = theta3
        I_out[:, :, s_width - 5] = problem._cp(theta3)

    L = problem.circuit._L()
    A_mat = A @ (L + scipy.sparse.diags(1.0 / Cnext[:, 0], 0)) @ AT
    Asq_fact = scipy.sparse.linalg.factorized(A_mat)
    theta_s = np.zeros((Nj, W), dtype=np.double)

    Is, T, Vs, f = problem._Is(0), problem._T(0), problem._Vs(0), problem._f(0)
    Is_zero, T_zero, Vs_zero, f_zero = False, False, False, False
    fluctuations = 0.0
    if not problem._T_is_timedep:
        T_zero = np.allclose(T, 0)
    if not problem._Is_is_timedep:
        Is_zero = np.allclose(Is, 0)
    if not problem._Vs_is_timedep:
        Vs_zero = np.allclose(Vs, 0)
    if not problem._f_is_timedep:
        f_zero = np.allclose(f, 0)

    i_th = 0
    i_I = 0
    for i in range(problem._Nt()):
        if problem._T_is_timedep:
            T = problem._T(i)
        if problem._Is_is_timedep:
            Is = problem._Is(i)
        if problem._Vs_is_timedep:
            Vs = problem._Vs(i)
        if problem._f_is_timedep:
            f = problem._f(i)

        if not T_zero:
            if circuit.junction_count() > 500:
                rand = np.random.randn(Nj, W) if i % 3 == 0 else rand[np.random.permutation(Nj), :]
            else:
                rand = np.random.randn(Nj, W)
            fluctuations = ((2.0 * T * Rv) ** 0.5) * rand

        if s_width >= 5:
            theta4 = theta3.copy()
        if s_width >= 4:
            theta3 = theta2.copy()
        theta2 = theta1.copy()
        theta1 = theta_next.copy()

        if s_width == 3:
            X = problem._cp(2 * theta1 - theta2) + c1 * theta1 + c2 * theta2
        if s_width == 4:
            X = problem._cp(3 * theta1 - 3 * theta2 + theta3) + c1 * theta1 + c2 * theta2 + c3 * theta3
        if s_width == 5:
            X = problem._cp(4 * theta1 - 6 * theta2 + 4 * theta3 - theta4) + \
                c1 * theta1 + c2 * theta2 + c3 * theta3 + c4 * theta4

        if Is_zero:
            x = fluctuations + X
        else:
            x = fluctuations - Is + X

        if Vs_zero:
            if f_zero:
                y = AT @ Asq_fact(A @ (x / c0))
            else:
                y = AT @ Asq_fact(A @ (x / c0) - 2 * np.pi * f)
        else:
            if f_zero:
                y = AT @ Asq_fact(A @ (x / c0 - theta_s))
            else:
                y = AT @ Asq_fact(A @ (x / c0 - theta_s) - 2 * np.pi * f)
        theta_next = (y - x) / c0

        if th_store_mask[i]:
            th_out[:, :, i_th + s_width - 1] = theta_next
            i_th += 1
        if I_store_mask[i]:
            I_out[:, :, i_I + s_width - 1] = y + Is
            i_I += 1

        if not Vs_zero:
            theta_s += Vs *dt

    return th_out, I_out


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

    def __init__(self, problem: TimeEvolutionProblem, theta, current, voltage):
        self.problem = problem
        Nj, W, Nt_s = problem.circuit._Nj(), self.get_problem_count(), problem._Nt_s()
        self.theta = theta
        self.voltage = voltage
        self.current = current
        if problem.store_theta:
            if self.theta.shape != (Nj, W, Nt_s):
                raise ValueError(f"theta must have shape {(Nj, W, Nt_s)}; has shape {self.theta.shape}")
        else:
            self.theta = None
        if problem.store_current:
            if self.current.shape != (Nj, W, Nt_s):
                raise ValueError(f"current must have shape {(Nj, W, Nt_s)}; has shape {self.current.shape}")
        else:
            self.current = None
        if problem.store_voltage:
            if self.voltage.shape != (Nj, W, Nt_s):
                raise ValueError(f"voltage must have shape {(Nj, W, Nt_s)}; has shape {self.voltage.shape}")
        else:
            self.voltage = None
        s = self.problem.store_time_steps.astype(int)
        self.time_point_indices = np.cumsum(s) - s
        self.animation = None

    def _th(self, time_point) -> np.ndarray:
        if self.theta is None:
            raise ThetaNotStored("Cannot query theta; quantity is not stored during time evolution simulation.")
        return self.theta[:, :, self._time_point_index(time_point)]

    def _V(self, time_point) -> np.ndarray:
        if self.voltage is None:
            raise VoltageNotStored("Cannot query voltage; quantity is not stored during time evolution simulation.")
        return self.voltage[:, :, self._time_point_index(time_point)]

    def _I(self, time_point) -> np.ndarray:
        if self.current is None:
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
        c = self.get_circuit()
        M, Nj = c.get_cut_matrix(), c._Nj()
        func = lambda tp: c.Msq_solve(M @ self._th(tp).reshape(Nj, -1))
        try:
            return self._select(select_time_points, self.get_circuit()._Nn(), func)
        except ThetaNotStored:
            raise ThetaNotStored("Cannot compute phi; requires theta to be stored in TimeEvolutionConfig")

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
        func = lambda tp: -A @ np.round(self._th(tp) / (2.0 * np.pi))
        try:
            return self._select(select_time_points, self.get_circuit()._Nf(), func).astype(int)
        except ThetaNotStored:
            raise ThetaNotStored("Cannot compute n; requires theta to be stored in TimeEvolutionConfig")

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
        try:
            return self._select(select_time_points, self.get_circuit()._Nj(), func)
        except ThetaNotStored:
            raise ThetaNotStored("Cannot compute Josephson energy EJ; requires theta to be stored in TimeEvolutionConfig")

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
        try:
            return self._select(select_time_points, self.get_circuit()._Nj(), func)
        except ThetaNotStored:
            raise ThetaNotStored("Cannot compute supercurrent Isup; requires theta to be stored in TimeEvolutionConfig")

    def get_J(self, select_time_points=None) -> np.ndarray:
        """
        Return cycle-current J around faces. Requires current to be stored. Defined
        as I = A.T @ J + I_source.

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
        func = lambda tp: self.get_circuit().Asq_solve(A @ (self._I(tp) - self.problem._Is(tp)))
        try:
            return self._select(select_time_points, self.get_circuit()._Nf(), func)
        except CurrentNotStored:
            raise CurrentNotStored("Cannot compute cycle-current J; requires current to be stored in TimeEvolutionConfig")

    def get_flux(self, select_time_points=None) -> np.ndarray:
        """
        Return magnetic flux through faces. Requires current to be stored.
        Defined as f + (A @ L @ I) / (2 * pi).

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
        func = lambda tp: self.problem._f(tp) + A @ (self.get_circuit()._L() @ self._I(tp)) / (2 * np.pi)
        try:
            return self._select(select_time_points, Nf, func)
        except CurrentNotStored:
            raise CurrentNotStored("Cannot compute magnetic flux; requires current to be stored in TimeEvolutionConfig")

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
        try:
            is_zero = not self.get_circuit()._has_inductance()
            return self._select(select_time_points, Nj, func, is_zero=is_zero)
        except CurrentNotStored:
            raise CurrentNotStored("Cannot compute magnetic energy EM; requires current to be stored in TimeEvolutionConfig")

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
        M, Nj = self.get_circuit().get_cut_matrix(), self.get_circuit()._Nj()
        # Mrsq = M @ M.T
        # Z = np.zeros((1, self.get_problem_count()), dtype=np.double)
        # solver = scipy.sparse.linalg.factorized(Mrsq)
        # func = lambda tp: np.concatenate((solver(M @ self._V(tp)), Z), axis=0)
        func = lambda tp: self.get_circuit().Msq_solve(M @ self._V(tp).reshape(Nj, -1))
        try:
            return self._select(select_time_points, self.get_circuit()._Nn(), func)
        except VoltageNotStored:
            raise VoltageNotStored(
                "Cannot compute electric potential U; requires voltage to be stored in TimeEvolutionConfig")

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
        try:
            is_zero = not self.get_circuit()._has_capacitance()
            return self._select(select_time_points, Nj, func, is_zero=is_zero)
        except VoltageNotStored:
            raise VoltageNotStored(
                "Cannot compute capacitive energy EC; requires voltage to be stored in TimeEvolutionConfig")

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

    def plot(self, problem_nr=0, time_point=0, fig=None, node_quantity=None,
             junction_quantity="I", face_quantity=None, vortex_quantity="n",
             show_grid=True, show_nodes=True, return_plot_handle=False, **kwargs):
        """
        Visualize a problem at a specified timestep.

        See :py:attr:`circuit_visualize.CircuitPlot` for documentation.

        Attributes
        ----------
        return_plot_handle=False : bool
            If True this method returns the ConfigPlot object used to create the plot.

        Returns
        -------
        fig : matplotlib figure handle
            Returns figure handle
        ax : matplotlib axis handle
            Returns axis handle
        plot_handle : ConfigPlot (optional)
            Object used to create the plot

        """
        return self.animate(problem_nr=problem_nr, time_points=np.array([time_point]),
                            node_quantity=node_quantity, junction_quantity=junction_quantity,
                            face_quantity=face_quantity, vortex_quantity=vortex_quantity,
                            show_grid=show_grid, show_nodes=show_nodes, fig=fig,
                            return_plot_handle=return_plot_handle, **kwargs)

    def animate(self, problem_nr=0, time_points=None, fig=None,
                node_quantity=None, junction_quantity="I", face_quantity=None,
                vortex_quantity="n", show_grid=True, show_nodes=True,
                return_plot_handle=False, **kwargs):

        """
        Animate time evolution of a problem as a movie.

        See :py:attr:`circuit_visualize.TimeEvolutionMovie` for documentation.

        Attributes
        ----------
        return_plot_handle=False : bool
            If True this method returns the TimeEvolutionMovie object used to create the movie.

        Returns
        -------
        fig : matplotlib figure handle
            Returns figure handle
        ax : matplotlib axis handle
            Returns axis handle
        plot_handle : TimeEvolutionMovie (optional)
            Object used to create the movie

        """

        from pyjjasim.circuit_visualize import TimeEvolutionMovie

        self.animation = TimeEvolutionMovie(self, problem_nr=problem_nr, time_points=time_points,
                                            vortex_quantity=vortex_quantity, show_grid=show_grid,
                                            show_nodes=show_nodes, junction_quantity=junction_quantity,
                                            node_quantity=node_quantity, face_quantity=face_quantity,
                                            fig=fig, **kwargs)
        if return_plot_handle:
            return *self.animation.make(), self.animation
        return self.animation.make()

    def __str__(self):
        return "time evolution configuration: (" + ("th" + self.theta.shape.__str__() + ", ") * (
                    self.theta is not None) + \
               ("I" + self.current.shape.__str__() + ", ") * (self.current is not None) + \
               ("V" + self.voltage.shape.__str__()) * (self.current is not None) + ")" + \
               "\nproblem: " + self.problem.__str__() + \
               "\ncircuit: " + self.get_circuit().__str__()

    def _select(self, select_time_points, N, func, is_zero=False):
        select_time_points = np.flatnonzero(self.problem._to_time_point_mask(select_time_points))
        W = self.get_problem_count()
        out = np.zeros((N, W, len(select_time_points)), dtype=np.double)
        if is_zero:
            return out
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













