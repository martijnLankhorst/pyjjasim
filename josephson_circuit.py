from pyjjasim.embedded_graph import EmbeddedGraph, EmbeddedTriangularGraph, EmbeddedHoneycombGraph, EmbeddedSquareGraph


import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import scipy.fftpack

__all__ = ["Circuit", "SquareArray", "HoneycombArray", "TriangularArray", "SQUID"]

# TODO:
# - work out if area and frustration definition
# - give error if criterion all-or-nothing inductance is not met.
# - give warning if timestep too small in low-inductance-case
# - give error for mixed-inductance problems
# - only show combined error?
# - handle time evolution proper errors if required quantities are not computed.
# - time evolution config_at_minus1 list of configs
# - put references in documentation
# - fix whitepaper mistakes
# - A_solve algo use guassian elimination


# nice to haves
# - lattices
# - periodic lattices
# - 3D
# - Multigrid
# - Nonlinear resistors
# - GPU
#   * single/double precision
#   * phase build-up rounding correction


"""
Josephson Circuit Module
"""

class NoCurrentConservationError(Exception):
    pass

class Circuit:
    """
    Construct a Josephson Circuit, also called a Josephson Junction Array (JJA).

    A JJA is an electric circuit that can include Josephson junctions, passive components,
    current sources and voltage sources. The network is required to be a planar embedding
    and single component, so junctions cannot intersect.

    Defined with a graph of nodes and edges where each edge contains a junction.
    A junction is the basic 2-terminal element which contains one of each component,
    see the included user manual for the precise definition. To omit a component;
    set it's value to zero.

    Attributes
    ----------
    graph : :py:attr:`embedded_graph.EmbeddedGraph`
        EmbeddedGraph instance with Nn nodes, Nj edges and Nf faces.
    critical_current_factors=1.0 : (Nj,) array or scalar
        Critical current factors of junctions. Same value for all junctions if scalar.
    resistance_factors=1.0 : (Nj,) array or scalar
        Resistance factors of junctions. Same value for all junctions if scalar.
    capacitance_factors=0.0 : (Nj,) array or scalar
        Capacitance factors of junctions. Same value for all junctions if scalar.
    inductance_factors=0.0 : (Nj,) array or scalar
        Self-inductance factors of junctions. Same value for all junctions if scalar.
    or inductance_factors=0.0 : (Nj, Nj) matrix (dense or sparse)
        L_ij coupling between junction i and j. L_ii self inductance. Must be symmetric
        positive definite.

    Notes
    -----
    * All physical quantities are normalized in pyjjasim, see the user manual for details.
      For example the critical current of each junction in Ampere is
      :math:`\mathtt{critical\_current\_factors} * I_0`, where :math:`I_0` is the
      normalizing scalar for all current values.
    * Sources are specified at problems, not explicitly as part of the circuit.
    """

    def __init__(self, graph: EmbeddedGraph, critical_current_factors=1.0, resistance_factors=1.0,
                 capacitance_factors=0.0, inductance_factors=0.0):

        self.graph = graph
        # self.graph._assign_faces()
        self.graph._assert_planar_embedding()
        self.graph._assert_single_component()
        # n, _ = self.graph.get_l_cycles(to_list=False)
        # self.graph.permute_faces(np.argsort(n[self.graph.faces_v_array.cum_counts]))
        # self.graph._assert_planar_embedding()
        # self.graph._assert_single_component()

        self.resistance_factors = None
        self.capacitance_factors = None
        self.critical_current_factors=None
        self.inductance_factors = None
        self._has_inductance_v = False

        self.locator = None

        self.cut_matrix = self.graph.cut_space_matrix()
        self.cycle_matrix = self.graph.face_cycle_matrix()

        self.set_resistance_factors(resistance_factors)
        self.set_critical_current_factors(critical_current_factors)
        self.set_capacitance_factors(capacitance_factors)
        self.set_inductance_factors(inductance_factors)

        self.cut_matrix_reduced = None

        self._Mnorm = None
        self._Anorm = None
        self._Msq_factorized = None
        self._Asq_factorized = None
        self._AiIcpLA_factorized = None
        self._IpLIc_factorized = None
        self._ALA_factorized = None
        self._Msq_S_factorized = None
        self._Asq_S_factorized = None

    def Msq_solve(self, b):
        """
        Solves M @ M.T @ x = b (where M is self.get_cut_matrix()).

        Parameters
        ----------
        b : (Nn,) or (Nn, W) array
            Right-hand side of system

        Returns
        -------
        x : shape of b
            solution of system
        """
        if self._Msq_factorized is None:
            self._Msq_factorized = scipy.sparse.linalg.factorized(self._Mr() @ self._Mr().T)
        return np.append(self._Msq_factorized(b[:-1, ...]), np.zeros(b[-1:, ...].shape), axis=0)

    def Asq_solve(self, b):
        """
        Solves A @ A.T @ x = b (where A is self.get_cycle_matrix()).

        Parameters
        ----------
        b : (Nf,) or (Nf, W) array
            Right-hand side of system

        Returns
        -------
        x : shape of b
            solution of system
        """
        if self._Asq_factorized is None:
            self._Asq_factorized = scipy.sparse.linalg.factorized(self.cycle_matrix @ self.cycle_matrix.T)
        result = self._Asq_factorized(b)
        return result

    def Msq_solve_sandwich(self, b, S):
        """
        Solves M @ S @ M.T @ x = b (where M is self.get_cut_matrix()).

        Parameters
        ----------
        b : (Nn,) or (Nn, W) array
            Right-hand side of system.
        S : (Nj, Nj) sparse array
            Sandwich matrix.

        Returns
        -------
        x : shape of b
            Solution of system.
        """
        Sd = S.diagonal()
        if S.nnz == np.count_nonzero(Sd):
            if np.allclose(Sd[0], Sd):
                return self.Msq_solve(b) / Sd[0]
        MsqF = scipy.sparse.linalg.factorized(self._Mr() @ S @ self._Mr().T)
        return np.append(MsqF(b[:-1, ...]), np.zeros(b[-1:, ...].shape), axis=0)

    def Asq_solve_sandwich(self, b, S, use_pyamg=False):
        """
        Solves A @ S @ A.T @ x = b (where A is self.get_cycle_matrix()).

        Parameters
        ----------
        b : (Nf,) or (Nf, W) array
            Right-hand side of system.
        S : (Nj, Nj) sparse array
            Sandwich matrix.

        Returns
        -------
        x : shape of b
            Solution of system.
        """
        Sd = S.diagonal()
        if S.nnz == np.count_nonzero(Sd):
            if np.allclose(Sd[0], Sd, rtol=1E-14):
                return self.Asq_solve(b) / Sd[0]
        if use_pyamg:
            import pyamg
            import pyamg.util.linalg
            A = self.cycle_matrix @ S @ self.cycle_matrix.T
            pyamg_cb = PyamgCallback(A, b)
            ml = pyamg.ruge_stuben_solver(A, strength=('classical', {'theta': 0.2}))
            try:
                mg_out = ml.solve(b, tol=1e-8, maxiter=3, callback=pyamg_cb.cb)
            except DivergenceError:
                mg_out = pyamg_cb.x
            if not pyamg_cb.has_diverged:
                return mg_out
            else:
                return self.Asq_solve_sandwich(b, S, use_pyamg=False)
        else:
            AsqF = scipy.sparse.linalg.factorized(self.cycle_matrix @ S @ self.cycle_matrix.T)
            return AsqF(b)

    def _AiIcpLA_solve(self, b):
        if self._has_identical_critical_current():
            if np.abs(self.critical_current_factors[0]) < 1E-12:
                return np.zeros(self.face_count(), dtype=np.double)
        if self._has_only_identical_self_inductance() and self._has_identical_critical_current():
            const = 1/self.critical_current_factors[0] + self.inductance_factors.diagonal()[0]
            return self.Asq_solve(b) / const
        if self._AiIcpLA_factorized is None:
            Nj, A = self._Nj(), self.get_cycle_matrix()
            Ic = self._Ic().copy()
            Ic[np.abs(Ic) > 1E-12] = Ic[np.abs(Ic) > 1E-12] ** -1
            L, iIc = self._L(), scipy.sparse.diags(Ic)
            self._AiIcpLA_factorized = scipy.sparse.linalg.factorized(A @ (iIc + L) @ A.T)
        return self._AiIcpLA_factorized(b)

    def _ALA_solve(self, b):
        if self._has_only_identical_self_inductance():
            const = self.inductance_factors.diagonal()[0]
            return self.Asq_solve(b) / const
        if self._ALA_factorized is None:
            A, L = self.get_cycle_matrix(), self._L()
            self._ALA_factorized = scipy.sparse.linalg.factorized(A @ L @ A.T)
        return self._ALA_factorized(b)

    def _IpLIc_solve(self, b):
        if self._has_only_identical_self_inductance() and self._has_identical_critical_current():
            const = 1 + self.inductance_factors.diagonal()[0] * self.critical_current_factors[0]
            return b / const
        if self._IpLIc_factorized is None:
            L, Ic, Nj = self._L(), scipy.sparse.diags(self._Ic()), self._Nj()
            self._IpLIc_factorized = scipy.sparse.linalg.factorized(scipy.sparse.eye(Nj) + L @ Ic)
        return self._IpLIc_factorized(b)

    def get_junction_nodes(self):
        """Get indices of nodes at endpoints of all junctions.

        Notes
        -----
        For all junctions node1 < node2, even if it was defined in reverse order.

        Returns
        -------
        node1, node2 : (Nj,) arrays
            Endpoint node indices of all junctions.
        """
        return self.graph.get_edges()

    def get_juncion_coordinates(self):
        """Get coordinates of nodes at endpoints of all junctions.

        Notes
        -----
        For all junctions node1 < node2, even if it was defined in reverse order.

        Returns
        -------
        x1, y1, x2, y2 : (Nj,) arrays
            Coordinates of node1 and node2 respectively.
        """
        x, y = self.get_node_coordinates()
        n1, n2 = self.get_junction_nodes()
        return x[n1], y[n1], x[n2], y[n2]

    # noinspection PyArgumentList
    def copy(self):
        """
        Return copy of circuit.
        """
        n1, n2 = self.get_junction_nodes()
        return Circuit(EmbeddedGraph(self.graph.x, self.graph.y, n1, n2),
                       critical_current_factors=self.get_critical_current_factors(),
                       resistance_factors=self.get_resistance_factors(),
                       capacitance_factors=self.get_capacitance_factors(),
                       inductance_factors=self.get_inductance_factors())

    # noinspection PyArgumentList
    def add_nodes_and_junctions(self, x, y, node1, node2,
                                critical_current_factors=1.0, resistance_factors=1.0,
                                capacitance_factors=1.0, inductance_factors=1.0):
        """ Add nodes to array and junctions to array.

            Attributes
            ----------
            x, y : (Nn_new,) arrays
                Coordinates of added nodes.
            node1, node2 : (Nj_new,) int arrays
                Nodes at endpoints of added junctions.
            critical_current_factors : scalar or (Nj_new,) array
                Critical current factors of added junctions. Same value for
                all new junctions if scalar.
            resistance_factors : scalar or (Nj_new,) array
                Resistance factors of added junctions. Same value for
                all new junctions if scalar.
            capacitance_factors : scalar or (Nj_new,) array
                Capacitance factors of added junctions. Same value for
                all new junctions if scalar.
            inductance_factors : scalar or (Nj_new,) array
                Self-inductance factors of added junctions. Same value for
                all new junctions if scalar.
            or inductance_factors : (Nj_new, Nj_new) array
                Mutual inductance factors between new junctions.
            or inductance_factors : (Nj_new, Nj) array
                Mutual inductance factors between new junctions and all junctions.

            Returns
            -------
            new_circuit : :py:attr:`Circuit`
                New Circuit object with nodes and junctions added.
        """
        x = np.array(x, dtype=np.double).flatten()
        new_x = np.append(self.graph.x, x)
        new_y = np.append(self.graph.y, np.array(y, dtype=np.double).flatten())
        n1, n2 = self.get_junction_nodes()
        new_node1 = np.append(n1, np.array(node1, dtype=int).flatten())
        new_node2 = np.append(n2, np.array(node2, dtype=int).flatten())
        Nj, Nj_new = self.junction_count(), len(node1)
        new_Ic = np.append(self.critical_current_factors,
                           self._prepare_junction_quantity(critical_current_factors, Nj_new, x_name="Ic"))
        new_R = np.append(self.resistance_factors,
                          self._prepare_junction_quantity(resistance_factors, Nj_new, x_name="R"))
        new_C = np.append(self.capacitance_factors,
                          self._prepare_junction_quantity(capacitance_factors, Nj_new, x_name="C"))
        new_L = None
        if hasattr(inductance_factors, 'shape'):
            if inductance_factors.shape == (Nj_new, Nj+Nj_new):
                if scipy.sparse.issparse(inductance_factors):
                    inductance_factors = inductance_factors.tocsc()
                A_block = self.inductance_factors
                C_block = inductance_factors[:, :self.junction_count()]
                D_block = inductance_factors[:, self.junction_count():]
                new_L = scipy.sparse.bmat([[A_block, C_block.T], [C_block, D_block]])
        if new_L is None:
            D_block, _, _ = Circuit._prepare_inducance_matrix(inductance_factors, Nj_new)
            new_L = scipy.sparse.block_diag([self.inductance_factors, D_block])
        new_circuit = Circuit(EmbeddedGraph(new_x, new_y, new_node1, new_node2),
                       critical_current_factors=new_Ic, resistance_factors=new_R,
                       capacitance_factors=new_C, inductance_factors=new_L)
        return new_circuit

    # noinspection PyArgumentList
    def remove_nodes(self, nodes_to_remove):
        """
        Remove nodes from circuit.

        Attributes
        ----------
        nodes_to_remove : int array in range(Nn)
            Indices of nodes to remove from circuit.
        or nodes_to_remove : (Nn,) mask
            Mask selecting nodes to remove from circuit.

        Returns
        -------
        new_circuit : :py:attr:`Circuit`
            New Circuit object with removed nodes.

        Notes
        -----
        Junctions whose endpoints are removed are also removed.


        """
        nodes_to_remove = np.array(nodes_to_remove).flatten()
        if not len(nodes_to_remove) == self.node_count():
            nodes_to_remove = np.array(nodes_to_remove, dtype=int)
        if not isinstance(nodes_to_remove.dtype, (bool, np.bool)):
            try:
                node_remove_mask = np.zeros(self.node_count(), dtype=bool)
                node_remove_mask[nodes_to_remove] = True
            except:
                raise ValueError("Invalid nodes_to_remove; must be None, mask, slice or index array")
        else:
            node_remove_mask = nodes_to_remove
        new_x = self.graph.x[~node_remove_mask]
        new_y = self.graph.y[~node_remove_mask]
        n1, n2 = self.get_junction_nodes()
        junc_remove_mask, new_node_id = self._junction_remove_mask(n1, n2, node_remove_mask)
        new_node1 = new_node_id[n1][~junc_remove_mask]
        new_node2 = new_node_id[n2][~junc_remove_mask]
        new_Ic = self.critical_current_factors[~junc_remove_mask]
        new_R = self.resistance_factors[~junc_remove_mask]
        new_C = self.capacitance_factors[~junc_remove_mask]
        new_L = self.inductance_factors[~junc_remove_mask, :][:, ~junc_remove_mask]
        return Circuit(EmbeddedGraph(new_x, new_y, new_node1, new_node2),
                       critical_current_factors=new_Ic, resistance_factors=new_R,
                       capacitance_factors=new_C, inductance_factors=new_L)

    # noinspection PyArgumentList
    def remove_junctions(self, junctions_to_remove):
        """
        Remove junctions from circuit.

        Attributes
        ----------
        junctions_to_remove : int array in range(Nj)
            Indices of junctions to remove from circuit.
        or junctions_to_remove : (Nj,) mask
            Mask selecting junctions to remove from circuit.

        Returns
        -------
        new_circuit : :py:attr:`Circuit`
            New Circuit object with removed junctions.
        """
        junctions_to_remove = np.array(junctions_to_remove).flatten()
        if not len(junctions_to_remove) == self.junction_count():
            junctions_to_remove = np.array(junctions_to_remove, dtype=int)
        if not isinstance(junctions_to_remove.dtype, (bool, np.bool)):
            try:
                junction_mask = np.zeros(self.junction_count(), dtype=bool)
                junction_mask[junctions_to_remove] = True
            except:
                raise ValueError("Invalid junctions_to_remove; must be None, mask, slice or index array")
        else:
            junction_mask = junctions_to_remove
        n1, n2 = self.get_junction_nodes()
        new_node1, new_node2 = n1[~junction_mask], n2[~junction_mask]
        new_Ic = self.critical_current_factors[~junction_mask]
        new_R = self.resistance_factors[~junction_mask]
        new_C = self.capacitance_factors[~junction_mask]
        new_L = self.inductance_factors[~junction_mask, :][:, ~junction_mask]
        return Circuit(EmbeddedGraph(self.graph.x, self.graph.y, new_node1, new_node2),
                       critical_current_factors=new_Ic, resistance_factors=new_R,
                       capacitance_factors=new_C, inductance_factors=new_L)

    def get_node_coordinates(self):
        """
        Returns coordinates of nodes in circuit.

        Returns
        -------
        x, y : (Nn,) arrays
            Coordinates of nodes in circuit.
        """
        return self.graph.coo()

    def node_count(self):
        """
        Returns number of nodes in the circuit (abbreviated Nn).
        """
        return self._Nn()

    def get_critical_current_factors(self):
        """
        Returns critical current factors of each junction in the circuit.
        """
        return self.critical_current_factors

    def set_critical_current_factors(self, Ic):
        """
        Modify critical current factors of all junctions in the circuit.

        Attributes
        ----------
        Ic : (Nj,) array or scalar
            New critical current factors. Same for all junctions if scalar.
        """
        self.critical_current_factors = self._prepare_junction_quantity(Ic, self._Nj(), x_name="Ic")
        self._AIpLIcA_factorized = None
        self._IpLIc_factorized = None
        return self

    def get_resistance_factors(self):
        """
        Returns resistance factors assigned to each junction in the circuit.
        """
        return self.resistance_factors

    def set_resistance_factors(self, R):
        """
        Modify resistance factors of all junctions in the circuit.

        Attributes
        ----------
        R : (Nj,) array or scalar
            New resistance factors. Same for all junctions if scalar.
        """
        self.resistance_factors = self._prepare_junction_quantity(R, self._Nj(), x_name="R")
        if np.any(self.resistance_factors <= 0.0):
            raise ValueError("All junctions must have a positive resistor")
        return self

    def get_capacitance_factors(self):
        """
        Returns capacitance factors assigned to each junction in the circuit.
        """
        return self.capacitance_factors

    def set_capacitance_factors(self, C):
        """
        Modify capacitance factors of all junctions in the circuit.

        Attributes
        ----------
        C : (Nj,) array or scalar
            New capacitance factors. Same for all junctions if scalar.
        """
        self.capacitance_factors = self._prepare_junction_quantity(C, self._Nj(), x_name="C")
        if np.any(self.capacitance_factors < 0.0):
            raise ValueError("Capacitance cannot be negative.")
        return self

    def junction_count(self):
        """
        Returns number of junctions in the circuit (abbreviated Nj).
        """
        return self._Nj()

    def face_count(self):
        """
        Returns number of faces in the circuit (abbreviated Nf).
        """
        return self._Nf()

    def get_faces(self):
        """
        Returns a list of all faces.

        A face is defined as an array containing indices of nodes encountered when
        traversing the boundary of a face counter-clockwise.

        Returns
        -------
        faces : List
            List of faces.
        """
        # Returns a list of faces.
        # if to_list==True returns in format:  [[n11, n12, n13], [n21], [n31, n32]]
        # if to_list==False returns in format: [n11, n12, n13, n21, n31, n32], [3, 1, 2]
        return self.graph.get_face_cycles(to_list=True)[0]

    def get_face_areas(self):
        """
        Returns area of all faces in the circuit.
        """
        # Returns unsigned area of each face in array.
        # out: areas            (face_count,) positive float array
        return self.graph.get_face_areas()

    def get_face_centroids(self):
        """
        Returns coordinates of centroids of all faces in the circuit.
        """
        # Returns centroid face_x, face_y of each face in array.
        # out: face_x, face_y   (face_count,) float arrRuehliay
        return self.graph.get_face_centroids()

    def locate_faces(self, x, y):
        """

        Get faces whose centroids are closest to queried coordinate.

        Attributes
        ----------
        x, y : arrays:
            Coordinates at which one wants to locate faces.


        Returns
        -------
        face_ids : int array with same size as x in range(Nf)
            Indices of located faces.
        """
        if self.locator is None:
            faces, self.locator = self.graph.locate_faces(x, y)
        else:
            _, faces = self.locator.query(np.stack(np.broadcast_arrays(x, y), axis=-1), k=1)
        return faces

    def approximate_inductance(self, factor, junc_L=1, junc_M=0, max_dist=3):
        """
        Approximate inductance in circuit.

        Computes a matrix L as an approximation for the inductance factors and
        does self.set_inductance_factors(L).

        L is computed using a crude approximation of Neumann's formula for two wire segments.

        Attributes
        ----------
        factor : scalar float
            mu0 * a0 in units of L0.
        junc_L : scalar float
            Self-inductance prefactor.
        junc_M : scalar float
            Mutual-inductance prefactor (usually << junc_L).
        max_dist : scalar float
            Cut-off distance between junctions included in L.

        Notes
        -----
        The self and mutual inductance are respectively (in units of :math:`\mu_0 a_0`):

        .. math:: L_i = \mathtt{junc\_L} * l_i

        .. math:: M_{ij} = \mathtt{junc\_M} * l_i * l_j * cos( \gamma_{ij}) / d_{ij}

        Where :math:`l_i` is junction length in units of :math:`a_0`,
        :math:`\gamma_{ij}` is angle between junctions and
        :math:`d_{ij}` is distance between centres of junctions in units of :math:`a_0`
        and afterwards they are multiplied by the conversion factor :math:`\mathtt{factor}=\mu_0 a_0 / L_0`
        to obain the required units of :math:`L_0`.

        """

        self.inductance_factors = None
        i, j = np.arange(self._Nj(), dtype=int), np.arange(self._Nj(), dtype=int)
        data = self._junction_lengths() * junc_L
        if junc_M > 0 and max_dist > 0:
            tree = scipy.spatial.KDTree(np.stack(self._junction_centers(), axis=-1))
            pairs = tree.query_pairs(max_dist, 2, output_type='ndarray')
            i, j = np.append(i, pairs[:, 0]), np.append(j, pairs[:, 1])
            i, j = np.append(i, pairs[:, 1]), np.append(j, pairs[:, 0])
            inner = self._junction_inner(*pairs.T)
            distance = self._junction_distance(*pairs.T)
            mutual = junc_M * inner / distance
            data = np.append(data, mutual)
            data = np.append(data, mutual)
        self.set_inductance_factors(factor * scipy.sparse.coo_matrix((data, (i, j)), shape=(self._Nj(), self._Nj())).tocsr())
        return self

    def get_inductance_factors(self):
        """
        Returns the inductance factors between each pair of junctions.

        Returns
        -------
        inductance_factors : (Nj, Nj) array
            Diagonal entries are self-inductance factors, off-diagonal entries are mutual.
        """
        # return matrix whose entry (r, c) is the magnetic coupling between wire r and wire c.
        # out: (junction_count, junction_count) sparse symmetric float matrix
        return self.inductance_factors

    def set_inductance_factors(self, inductance_factors):
        """
        Modify the inductances factors of all junctions in the circuit.
        """
        self.inductance_factors, is_positive_definite, self._has_inductance_v = \
            Circuit._prepare_inducance_matrix(inductance_factors, self._Nj())
        if not is_positive_definite:
            raise ValueError("Inductance matrix not positive definite")
        self._AIpLIcA_factorized = None
        self._IpLIc_factorized = None
        self._ALA_factorized = None
        return self

    def get_cut_matrix(self):
        """Returns cut matrix.

        The cut matrix is a sparse matrix (shape (Nn, Nj), abbreviated M), which represents
        Kirchhoffs current law M @ I = 0.

        It is +1 if node is node_2 of junction and -1 otherwise.
        """
        return self.cut_matrix

    def get_cycle_matrix(self):
        """Returns cycle matrix.

        The cycle matrix is a sparse matrix (shape (Nf, Nj) abbreviated A), which represents
        Kirchhoffs voltage law A @ V = 0.

        It is +1 if traversing a face counter-clockwise passes through a junction in its direction, and -1
        otherwise.
        """
        return self.cycle_matrix

    def plot(self, show_node_ids=True, show_junction_ids=False, show_faces=True,
             show_face_ids=True, markersize=5, linewidth=1, face_shrink_factor=0.9,
             figsize=None, fig=None, ax=None, ax_position=None, title=""):
        """Visualize array.

        Can show nodes, junctions and faces; and their respective indices.

        For documentation see :py:attr:`embedded_graph.EmbeddedGraph.plot`
        """
        cr = self.graph.plot(show_cycles=show_faces, figsize=figsize, cycles="face_cycles",
                             show_node_ids=show_node_ids, show_edge_ids=show_junction_ids,
                             show_face_ids=show_face_ids, markersize=markersize,
                             linewidth=linewidth, face_shrink_factor=face_shrink_factor,
                             fig=fig, ax=ax, ax_position=ax_position, title=title)
        return cr

    def save(self, filename):
        with open(filename, "wb") as ffile:
            x, y = self.graph.coo()
            n1, n2 = self.graph.get_edges()
            np.save(ffile, x)
            np.save(ffile, y)
            np.save(ffile, n1)
            np.save(ffile, n2)
            np.save(ffile, self.critical_current_factors)
            np.save(ffile, self.resistance_factors)
            np.save(ffile, self.capacitance_factors)
            L_is_sparse = scipy.sparse.issparse(self.inductance_factors)
            np.save(ffile, L_is_sparse)
            if L_is_sparse:
                np.save(ffile, self.inductance_factors.indptr)
                np.save(ffile, self.inductance_factors.indices)
                np.save(ffile, self.inductance_factors.data)
            else:
                np.save(ffile, self.inductance_factors)

    @staticmethod
    def load(filename):
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
            return Circuit(g, critical_current_factors=Ic, resistance_factors=R,
                           capacitance_factors=C, inductance_factors=L)


    # abbreviations and aliases
    def _Nn(self):   # alias for node_count
        return self.graph.node_count()

    def _Nnr(self):
        # reduced node count; returns node_count - 1
        return self._Nn() - 1

    def _Nj(self):   # alias for get_junction_count()
        return self.graph.edge_count()

    def _Nf(self):
        return self.graph.face_count()

    def _Ic(self) -> np.ndarray:     # alias for get_critical_current_factors
        return self.critical_current_factors

    def _R(self) -> np.ndarray:      # alias for get_resistance_factors
        return self.resistance_factors

    def _C(self) -> np.ndarray:      # alias for get_resistance_factors
        return self.capacitance_factors

    def _L(self):
        return self.inductance_factors

    def _Mr(self):
        if self.cut_matrix_reduced is None:
            self.cut_matrix_reduced = self.cut_matrix[:-1, :]
        return self.cut_matrix_reduced

    @staticmethod
    def _prepare_junction_quantity(x, N, x_name="x"):
        try:
            x = np.broadcast_to(x, (N,)).copy()
        except ValueError:
            raise ValueError(x_name + " must be scalar or array of length equal to junction count")
        return x

    @staticmethod
    def _prepare_inducance_matrix(L, N):
        eps = 10 * np.finfo(float).eps
        if not hasattr(L, "ndim"):
            L = np.array(L)
        if L.ndim <= 1:
            x = Circuit._prepare_junction_quantity(L, N, "L")
            return scipy.sparse.diags(x, 0).tocsc(), np.all(x > -eps), np.any(x != 0.0)
        if L.shape == (N, N):
            if not Circuit._is_symmetric(L):
                raise ValueError("inductance matrix must be symmetric")
            if scipy.sparse.issparse(L):
                L = L.tocsc()
                from pyjjasim.static_problem import is_positive_definite_superlu
                status = is_positive_definite_superlu(L)
                if status == 2:
                    raise ValueError(
                        "Choleski factorization failed; unable to determine positive definiteness of inductance matrix")
                is_positive_definite = status == 0
                is_zero = L.nnz == 0
            else:
                (_, pd) = scipy.linalg.lapack.dpotrf(np.array(L))
                is_positive_definite = pd == 0
                is_zero = np.all(L == 0)
            return L, is_positive_definite, is_zero
        else:
            raise ValueError("L must be scalar, (Nj,) array or (Nj, Nj) matrix")

    def __str__(self):
        return "(x, y): \n" + str(np.stack(self.get_node_coordinates())) + \
               "\n(node1_id, node2_id): \n" + str(np.stack(self.get_junction_nodes()))

    def _get_A_norm(self):
        # return ||A||_2 = sqrt(max(eig(A.T @ A))). (however computes sqrt(max(eig(A @ A.T))) which seems to be the same and is quicker)
        if self._Anorm is None:
            A = self.get_cycle_matrix()
            if A.shape[0] == 1:
                self._Anorm = np.sqrt((A @ A.T).todense()[0, 0])
            else:
                self._Anorm = np.sqrt(scipy.sparse.linalg.eigsh((A @ A.T).astype(np.double), 1, maxiter=1000, which="LA")[0][0])
        return self._Anorm

    def _get_M_norm(self):
        # return ||M||_2 = sqrt(max(eig(M.T @ M))). (however computes sqrt(max(eig(M @ M.T))) which seems to be the same and is quicker)
        if self._Mnorm is None:
            M = self.get_cut_matrix()
            if M.shape[0] == 1:
                self._Mnorm = np.sqrt((M @ M.T).todense()[0, 0])
            else:
                self._Mnorm = np.sqrt(scipy.sparse.linalg.eigsh((M @ M.T).astype(np.double), 1, maxiter=1000, which="LA")[0][0])
        return self._Mnorm

    def _A_solve(self, b):
        """
        Solves the equation: A @ x = b (where A = cycle_matrix).
        If b is integral (contain only integers), the output array x will also be integral.

        input:  b (..., Nf)
        output: x (..., Nj)

        Notes:
            - The equation is underdetermined, so the solution x is not unique.

        Use cases:
            - Used for changing phase zones (theta, z) -> (theta', z').
              Here theta' = theta + 2 * pi * Z where A @ Z = z' - z. Crucially, Z must
              be integral to ensure theta keeps obeying Kirchhoff's current rule.
            - Used for projecting theta onto cycle space; theta' = theta - g so that A @ theta'= 0.
              Then A @ g = 2 * pi * (z - areas * f)
        """
        return self.graph._cycle_space_solve_for_integral_x(b)


    def _has_capacitance(self):
        # returns False if self.capacitance_factors is zero, True otherwise
        return np.any(self.capacitance_factors > 0)

    def _has_inductance(self):
        # returns False if self.inductance_factors is zero, True otherwise
        return self._has_inductance_v

    def _has_only_self_inductance(self):
        L = self.get_inductance_factors()
        Ls = L.diagonal()
        if scipy.sparse.issparse(L):
            return L.nnz == np.count_nonzero(Ls)
        else:
            return np.allclose(L - np.diag(Ls), 0)

    def _has_mixed_inductance(self):
        mask = self._get_mixed_inductance_mask()
        return np.any(mask) and not np.all(mask)

    def _get_mixed_inductance_mask(self):
        L = self._L()
        A = self.get_cycle_matrix()
        ALA = A @ L @ A.T
        return np.isclose(np.array(np.sum(np.abs(ALA), axis=1))[:, 0], 0)

    def _has_identical_critical_current(self):
        Ic = self.get_critical_current_factors()
        return np.allclose(Ic, Ic[0])

    def _has_identical_resistance(self):
        R = self.get_resistance_factors()
        return np.allclose(R, R[0])

    def _has_identical_capacitance(self):
        C = self.get_capacitance_factors()
        return np.allclose(C, C[0])

    def _has_only_identical_self_inductance(self):
        if self._has_only_self_inductance():
            L = self.get_inductance_factors().diagonal()
            return np.allclose(L, L[0])
        return False

    def _assign_cut_matrix(self):
        if self.cut_matrix_reduced is None or self.cut_matrix is None:
            cut_matrix = -self.graph.cut_space_matrix()
            self.cut_matrix = cut_matrix.asformat("csc")
            self.cut_matrix_reduced = cut_matrix[:-1, :].asformat("csc")
            return self.cut_matrix, self.cut_matrix_reduced

    def _junction_centers(self):
        x, y = self.get_node_coordinates()
        n1, n2 = self.get_junction_nodes()
        return 0.5 * (x[n1] + x[n2]),  0.5 * (y[n1] + y[n2])

    def _junction_lengths(self):
        x, y = self.get_node_coordinates()
        n1, n2 = self.get_junction_nodes()
        return np.sqrt((x[n2] - x[n1]) ** 2 + (y[n2] - y[n1]) ** 2)

    def _junction_inner(self, ids1, ids2):
        x, y = self.get_node_coordinates()
        n1, n2 = self.get_junction_nodes()
        x_n1_j1, y_n1_j1 = x[n1[ids1]], y[n1[ids1]]
        x_n2_j1, y_n2_j1 = x[n2[ids1]], y[n2[ids1]]
        x_n1_j2, y_n1_j2 = x[n1[ids2]], y[n1[ids2]]
        x_n2_j2, y_n2_j2 = x[n2[ids2]], y[n2[ids2]]
        return (x_n2_j1 - x_n1_j1) * (x_n2_j2 - x_n1_j2) + (y_n2_j1 - y_n1_j1) * (y_n2_j2 - y_n1_j2)

    def _junction_distance(self, ids1, ids2):
        x, y = self._junction_centers()
        return np.sqrt((x[ids2] - x[ids1]) ** 2 + (y[ids2] - y[ids1]) ** 2)

    @staticmethod
    def _is_symmetric(A):
        if scipy.sparse.isspmatrix(A):
            return (A - A.T).nnz == 0
        else:
            return np.all(A == A.T)

    @staticmethod
    def _junction_remove_mask(nodes1, nodes2, node_remove_mask):
        node_remove_mask = node_remove_mask.copy().astype(int)
        remove_nodes = np.flatnonzero(node_remove_mask)
        new_node_id = np.arange(node_remove_mask.size, dtype=int) - (np.cumsum(node_remove_mask) - node_remove_mask)
        junc_remove_mask = (np.isin(nodes1, remove_nodes) | np.isin(nodes2, remove_nodes))
        return junc_remove_mask, new_node_id

    @staticmethod
    def _lobpcg_matrix_norm(A, preconditioner=None, maxiter=1000, tol=1E-5):
        """
        Computes ||A||_2
        """
        x0 = np.random.rand(A.shape[0], 1)
        lobpcg_out = scipy.sparse.linalg.lobpcg(A, x0, B=None, M=preconditioner, maxiter=maxiter, tol=tol)
        return np.sqrt(lobpcg_out[0])

    # def get_junctions_at_angle(self, angle):
    #     x1, y1, x2, y2 = self.get_juncion_coordinates()
    #     return np.isclose(np.arctan2(y2 - y1, x2 - x1), angle)
    #
    # def horizontal_junctions(self):
    #     return self.get_junctions_at_angle(0) | self.get_junctions_at_angle(np.pi) | \
    #            self.get_junctions_at_angle(-np.pi)
    #
    # def vertical_junctions(self):
    #     return self.get_junctions_at_angle(np.pi/2) | self.get_junctions_at_angle(-np.pi/2)

class Lattice:

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0):
        self.count_x = count_x
        self.count_y = count_y
        self.x_scale = x_scale
        self.y_scale = y_scale

    def get_count_x(self):
        return self.count_x

    def get_count_y(self):
        return self.count_y

    def get_x_scale(self):
        return self.x_scale

    def get_y_scale(self):
        return self.y_scale

    def current_base(self, angle, type="junction"):
        """
        Returns uniform current sources at angle

        If type="junction"; returns current_sources at each junction in the correct ratio to
        produce uniform current at angle. If type="node", returns amount of current injected at
        each node in the ratio to produce a uniform current at an angle.

        If x_scale=y_scale=1, the magnitude is such that per unit coordinate a current of 1 is sourced.
        The horizontal component scales linearly with x_scale and the vertical with y_scale.

        Parameters
        ----------
        angle : float
            Angle of the uniform current (in radians)
        type : "junction" or "node"
            If "junction", returns junction-based current sources. If "node"; returns
            node-based current sources.


        Returns
        -------
        current_sources : (Nj,) array (type = "junction")
            Junction-based current sources representing uniform current at angle.
        or current_sources : (Nn,) array (type = "node")
            Node-based current sources representing uniform current at angle.
        """
        if type == "junction":
            from pyjjasim import node_to_junction_current
            return node_to_junction_current(self, self.current_base(angle, type="node"))
        if type == "node":
            return self._I_node_h() * np.cos(angle) + self._I_node_v() * np.sin(angle)

    def _I_node_h(self) -> np.ndarray:
        pass

    def _I_node_v(self) -> np.ndarray:
        pass

class SquareArray(Circuit, Lattice):

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0):
        Lattice.__init__(self, count_x, count_y, x_scale, y_scale)
        Lattice.__init__(self, count_x, count_y, x_scale, y_scale)
        Circuit.__init__(self, EmbeddedSquareGraph(count_x, count_y, x_scale, y_scale))
        self._Asq_dst_K = None

    def _I_node_h(self):
        x, y = self.get_node_coordinates()
        return (x == 0).astype(int) - np.isclose(x, (self.count_x - 1) * self.x_scale).astype(int)

    def _I_node_v(self):
        x, y = self.get_node_coordinates()
        return (y == 0).astype(int) - np.isclose(y, (self.count_y - 1) * self.y_scale).astype(int)

    def Asq_solve(self, b, algorithm=0):
        """
        Solves A @ A.T @ x = b (where A is self.get_cycle_matrix()).

        Parameters
        ----------
        b : (Nf,) or (Nf, W) array
            Right-hand side of system
        algorithm=0 : int
            0 means solve with LU factorization, 1  means using sine transform (is faster for large arrays)

        Returns
        -------
        x : shape of b
            solution of system
        """
        if algorithm == 0:
            return super().Asq_solve(b)
        if algorithm == 1:
            Nx, Ny = self.count_x, self.count_y
            squeeze = False
            if self._Asq_dst_K is None:
                k1 = np.cos(np.pi * (np.arange(Nx-1) + 1) / Nx)
                k2 = np.cos(np.pi * (np.arange(Ny-1) + 1) / Ny)
                self._Asq_dst_K = 4 - 2 * k1 - 2 * k2[:, None]
            if b.ndim == 2:
                if b.shape[1] != 1:
                    W = b.shape[1]
                    B = scipy.fftpack.dstn(b.T.reshape(W, Ny-1, Nx-1), type=1, norm="ortho", axes=[1, 2])
                    out = scipy.fftpack.idstn(B / self._Asq_dst_K, type=1, norm="ortho", axes=[1, 2])
                    return out.reshape(W, -1).T
                squeeze = True
                b = b[:, 0]
            if b.ndim == 1:
                out = scipy.fftpack.idstn(scipy.fftpack.dstn(b.reshape(Ny-1, Nx-1), type=1, norm="ortho")
                                          / self._Asq_dst_K, type=1, norm="ortho")
                return out.ravel() if not squeeze else out.ravel()[:, None]
        raise ValueError("Invalid algorithm for square array Asq_solve. Must be 0 or 1.")



class HoneycombArray(Circuit, Lattice):

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0):
        Lattice.__init__(self, count_x, count_y, x_scale, y_scale)
        Circuit.__init__(self, EmbeddedHoneycombGraph(count_x, count_y, x_scale, y_scale))

    def _I_node_h(self):
        x, y = self.get_node_coordinates()
        n = self.count_y / (self.count_y - 0.5)
        return ((x == 0).astype(int) - np.isclose(x, (3 * self.count_x - 1) * self.x_scale).astype(int)) * n

    def _I_node_v(self):
        x, y = self.get_node_coordinates()
        return ((y < 0.1 * np.sqrt(3) * self.y_scale).astype(int) -
                (y > (self.count_y - 0.6) * np.sqrt(3) * self.y_scale).astype(int)) * 1.5

class TriangularArray(Circuit, Lattice):

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0):
        Lattice.__init__(self, count_x, count_y, x_scale, y_scale)
        Circuit.__init__(self, EmbeddedTriangularGraph(count_x, count_y, x_scale, y_scale))

    def _I_node_h(self):
        x, y = self.get_node_coordinates()
        return ((x < 0.1 * self.x_scale).astype(int) -
                (x > (self.count_x - 0.6) * self.x_scale).astype(int))*np.sqrt(3)

    def _I_node_v(self):
        x, y = self.get_node_coordinates()
        return (y == 0).astype(int) - np.isclose(y, (self.count_y - 0.5) * np.sqrt(3) * self.y_scale).astype(int)


class SQUID(Circuit):

    """
    A SQUID is modeled as a square where the vertical junctions have Ic=1000
    and the horizontal Ic=1.
    """
    def __init__(self):
        x = [0, 1, 1, 0]
        y = [0, 0, 1, 1]
        node1 = [0, 1, 2, 0]
        node2 = [1, 2, 3, 3]
        graph = EmbeddedGraph(x, y, node1, node2)
        Ic = [1, 1000, 1, 1000]
        super().__init__(graph, critical_current_factors=Ic)

    def horizontal_junctions(self):
        return np.array([1, 0, -1, 0])

    def vertical_junctions(self):
        return  np.array([0, 1, 0, 1])



class DivergenceError(Exception):
    pass

class PyamgCallback:

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.iter = 0
        self.resid = np.zeros((0,))
        self.x = None
        self.has_diverged = False

    def cb(self, x):
        self.x = x
        dx = np.ravel(self.b) - self.A * np.ravel(x)
        r = np.sqrt(np.inner(dx.conj(), dx).real)
        self.resid = np.append(self.resid, [np.abs(r)])
        self.iter += 1
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            self._diverge()
        if self.iter > 1:
            if self.resid[-1] > self.resid[-2]:
                self._diverge()

    def _diverge(self):
        self.has_diverged = True
        raise DivergenceError("diverged")
