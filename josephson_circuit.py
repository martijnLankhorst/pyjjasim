from __future__ import annotations

from pyJJAsim.embedded_graph import EmbeddedGraph, EmbeddedTriangularGraph, EmbeddedHoneycombGraph, EmbeddedSquareGraph

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial

__all__ = ["Circuit", "SquareArray", "HoneycombArray", "TriangularArray", "SQUID"]

# TODO:
# work out if area and frustration definition
# give error if criterion all-or-nothing inductance is not met.
# give warning if timestep too small in low-inductance-case

# requirements
# - check/debug all physical quantities
# - dynamic compute broadcasting slicing
# - documentation
# - unit testing
# - publish


# nice to haves
# - lattices
# - periodic lattices
# - 3D
# - Multigrid
# - Nonlinear resistors
# - mixed inductor/non-inductor elements
# - GPU
#   * single/double precision
#   * phase build-up rounding correction


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

    All physical quantities are normalized in pyjjasim, see the user manual for details.
    For example the critical current of each junction in Ampere is critical_current_factors * I0,
    where I0 is the normalizing scalar for all current values.

    Attributes
    ----------
    graph: EmbeddedGraph
        EmbeddedGraph instance with Nn nodes, Nj edges and Nf faces
    critical_current_factors=1.0:
        critical current factors of junctions, must be (Nj,) array or
        scalar (same value for all junctions)
    resistance_factors=1.0:
        resistance factors of junctions, must be (Nj,) array or
        scalar (same value for all junctions)
    capacitance_factors=0.0:
        capacitance factors of junctions, must be (Nj,) array or
        scalar  (same value for all junctions)
    inductance_factors=0.0:
        L_ij coupling between junction i and j. Either (Nj, Nj) sparse matrix,
        (Nj, Nj) array, (Nj,) array (entries of diagonal matrix) or
        scalar (constant diagonal)
    matrix_format="csc" :
        Format of sparse problem matrices, "csr" or "csc"

    """

    def __init__(self, graph: EmbeddedGraph, critical_current_factors=1.0, resistance_factors=1.0,
                 capacitance_factors=0.0, inductance_factors=0.0, matrix_format="csc"):

        """
        Constructor for Circuit, which assesses if the graph and input is valid.
        """

        self.graph = graph
        self.graph.get_faces()
        n, _ = self.graph.get_face_nodes()
        self.graph.permute_faces(np.argsort(n[self.graph.faces_v_array.cum_counts]))
        self.graph.assert_planar_embedding()
        self.graph.assert_single_component()
        self.junction_unsort_perm = np.argsort(self.graph.edge_sorter)  # needed because EmbeddedGraph sorts edges.

        self.resistance_factors = None
        self.capacitance_factors = None
        self.critical_current_factors=None
        self.inductance_factors = None
        self.set_resistance_factors(resistance_factors)
        self.set_critical_current_factors(critical_current_factors)
        self.set_capacitance_factors(capacitance_factors)
        self.set_inductance_factors(inductance_factors)

        self.locator = None

        self.matrix_format = matrix_format
        self.cut_matrix = self.graph.cut_space()[:, self.junction_unsort_perm]
        self.cycle_matrix = self.graph.cycle_space(include_boundary_faces=False)[:, self.junction_unsort_perm]
        self.cut_reduced_square = None
        self.cut_matrix_reduced = None
        self.cut_square = None
        self.cut_matrix_reduced_transposed = None
        self.cut_matrix_transposed = None
        self.cycle_matrix_transposed = None
        self.cycle_square = None
        self._Mnorm = None
        self._Anorm = None
        self._has_inductance_v = False

    def get_junction_nodes(self):
        """
        Get ids of nodes at endpoints of all junctions
        """
        return self.graph.node1[self.junction_unsort_perm], self.graph.node2[self.junction_unsort_perm]

    def get_juncion_coordinates(self):
        """
        Get coordinates of nodes at endpoints of all junctions.
        Returns:
            x1  (Nj,) array with x-coordinate of node1
            y1  (Nj,) array with y-coordinate of node1
            x2  (Nj,) array with x-coordinate of node2
            y2  (Nj,) array with y-coordinate of node2
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
                       inductance_factors=self.get_inductance_factors(),
                       matrix_format=self.matrix_format)

    # noinspection PyArgumentList
    def add_nodes_and_junctions(self, x, y, node1, node2,
                                critical_current_factors=1.0, resistance_factors=1.0,
                                capacitance_factors=1.0, inductance_factors=1.0) -> Circuit:
        """ Add nodes to array and junctions to array.

            Attributes
            ==========
            x, y: arrays
                coordinates of added nodes
            node1, node2: int arrays
                nodes at endpoints of added junctions
            critical_current_factors: scalar or array
                critical current factors of added junctions. Same value for
                all new junctions if scalar.
            resistance_factors: scalar or array
                resistance factors of added junctions. Same value for
                all new junctions if scalar.
            capacitance_factors: scalar or array
                capacitance factors of added junctions. Same value for
                all new junctions if scalar.
            inductance_factors: scalar or (Nj_new,) array
                self-inductance factors of added junctions. Same value for
                all new junctions if scalar.
            or inductance_factors: (Nj_new, Nj_new) array
                Mutual inductance factors between new junctions.
            or inductance_factors: (Nj_new, Nj) array
                Mutual inductance factors between new junctions and all junctions.

            Returns
            -------
            new_circuit: Circuit
                new Circuit object with nodes and junctions added.
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
        return Circuit(EmbeddedGraph(new_x, new_y, new_node1, new_node2),
                       critical_current_factors=new_Ic, resistance_factors=new_R,
                       capacitance_factors=new_C, inductance_factors=new_L,
                       matrix_format=self.matrix_format)

    # noinspection PyArgumentList
    def remove_nodes(self, nodes_to_remove):
        """
        Remove nodes with a list/array of node ids of with a mask of shape (Nn,)
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
                       capacitance_factors=new_C, inductance_factors=new_L,
                       matrix_format=self.matrix_format)

    # noinspection PyArgumentList
    def remove_junctions(self, junctions_to_remove):
        """
        Remove junctions with a list/array of junction ids of with a mask of shape (Nj,)
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
                       capacitance_factors=new_C, inductance_factors=new_L,
                       matrix_format=self.matrix_format)

    def get_node_coordinates(self):
        """
        Returns x, y representing the coordinates of the nodes in the circuit.
        """
        return self.graph.x, self.graph.y

    def node_count(self):
        """
        Returns the number of nodes in the circuit (abbreviated Nn)
        """
        return self._Nn()

    def get_critical_current_factors(self):
        """
        Returns the critical current factors assigned to each junction in the circuit
        """
        return self.critical_current_factors

    def set_critical_current_factors(self, Ic):
        """
        Modify the critical current factors of all junctions in the circuit
        """
        self.critical_current_factors = self._prepare_junction_quantity(Ic, self._Nj(), x_name="Ic")
        return self

    def get_resistance_factors(self):
        """
        Returns the critical current factors assigned to each junction in the circuit
        """
        return self.resistance_factors

    def set_resistance_factors(self, R):
        """
        Modify the resistance factors of all junctions in the circuit
        """
        self.resistance_factors = self._prepare_junction_quantity(R, self._Nj(), x_name="R")
        if np.any(self.resistance_factors <= 0.0):
            raise ValueError("All junctions must have a positive resistor")
        return self

    def get_capacitance_factors(self):
        """
        Returns the capacitance factors assigned to each junction in the circuit
        """
        return self.capacitance_factors

    def set_capacitance_factors(self, C):
        """
        Modify the capacitance factors of all junctions in the circuit
        """
        self.capacitance_factors = self._prepare_junction_quantity(C, self._Nj(), x_name="C")
        if np.any(self.capacitance_factors < 0.0):
            raise ValueError("Capacitance cannot be negative.")
        return self

    def junction_count(self):
        """
        Returns the number of junctions in the circuit (abbreviated Nj)
        """
        return self._Nj()

    def face_count(self):
        """
        Returns the number of faces in the circuit (abbreviated Nf)
        """
        return self._Nf()

    def get_faces(self, to_list=True):
        """
        Returns a list of all faces, where a face is defined as an array
        containing ids of nodes encountered when traversing the boundary
        of a face counter-clockwise.
        """
        # Returns a list of faces.
        # if to_list==True returns in format:  [[n11, n12, n13], [n21], [n31, n32]]
        # if to_list==False returns in format: [n11, n12, n13, n21, n31, n32], [3, 1, 2]
        return self.graph.get_face_nodes(include_boundary_faces=False, to_list=to_list)

    def get_face_areas(self):
        """
        Returns area of all faces in the circuit.
        """
        # Returns unsigned area of each face in array.
        # out: areas            (face_count,) positive float array
        return self.graph.get_areas(include_boundary_faces=False)

    def get_face_centroids(self):
        """
        Returns coordinates of the centroids of all faces in the circuit.
        """
        # Returns centroid face_x, face_y of each face in array.
        # out: face_x, face_y   (face_count,) float arrRuehliay
        return self.graph.get_centroids(include_boundary_faces=False)

    def locate_faces(self, x, y):
        """

        Get faces whose centroids are closest to queried (x,y) coordinate pairs.
        in:  x, y               float arrays of shape (N,)
        out: face_ids           int array in range(face_count) of shape (N,)
        """
        if self.locator is None:
            self.locator = scipy.spatial.KDTree(np.stack(self.get_face_centroids(), axis=-1))
        _, faces = self.locator.query(np.stack(np.broadcast_arrays(x, y), axis=-1), k=1)
        return faces

    def approximate_inductance(self, factor, junc_L=1, junc_M=0, max_dist=3):
        """
        Sets self.junction_inductance_factors (which is a Nj by Nj matrix) according to the formula:

         - self:     L = junc_L * l                                             in units of mu0 * a0
         - mutual:   L = junc_M * l_1 * l_2 * cos(alpha_12) / distance_12       in units of mu0 * a0
         -> self.junction_inductance_factors = factor * L                       -> in units of L0

        as a crude approximation of Neumann's formula for two wire segments.

        Here:
         - l, l1, l2    junction lengths                                        in units of a0
         - alpha_12     angle between junction 1 and junction 2
         - distance_12  distance between centres of junctions 1 and 2           in units of a0
         - max_dist     cutoff distance for included junction pairs             in units of a0

        junction_inductance_factors must be set in units of L0, so factor = mu0*a0/L0.
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
        Returns the inductance factors, which is a matrix containing both
        self-inductances and mutual inductances between junctions.
        """
        # return matrix whose entry (r, c) is the magnetic coupling between wire r and wire c.
        # out: (junction_count, junction_count) sparse symmetric float matrix
        return self.inductance_factors

    def set_inductance_factors(self, inductance_factors):
        """
        Modify the inductances factors of all junctions in the circuit
        """
        self.inductance_factors, is_positive_definite, self._has_inductance_v = \
            Circuit._prepare_inducance_matrix(inductance_factors, self._Nj())
        if not is_positive_definite:
            raise ValueError("Inductance matrix not positive definite")
        return self

    def get_cut_matrix(self):
        """
        Returns sparse cut matrix (shape (Nn, Nj), abbreviated M). Represents
        Kirchhoffs current law M @ I = 0

        returns the "cut-matrix" (abbreviated M in this code), which is a node_count by junction_count matrix.
        Its transpose is  the incidence matrix, node1 of a junction is -1 and node2 is -1 (abbreviated MT).
        Its rows span the cut-space (which is orthogonal to the cycle space). The cut-space has rank node_count - 1.

        Important: The row number does nót correspond with node_idx, because the nodes are permuted internally.

         out: (node_count, junction_count) sparse matrix
        """
        return self.cut_matrix

    def get_cycle_matrix(self):
        """
        Returns sparse cycle matrix (shape (Nf, Nj) abbreviated A). Represents
        Kirchhoffs voltage law A @ V = 0

        returns the "cycle-matrix" (abbreviated A in this code), which is a face_count by junction_count matrix.
        It is +1 if traversing a face counter-clockwise passes through a junction in its direction, and -1
        otherwise. Its rows span the cycle-space (which is orthogonal to the cut space). The cut-space has a
        span equal to face_count = junction_count - node_count + 1.

        out: (face_count, junction_count) sparse matrix
        """
        return self.cycle_matrix

    def plot(self, show_node_ids=True, show_junction_ids=False, show_faces=True,
             show_face_ids=True, face_shrink_factor=0.9, figsize=None):
        """
        Visualize array. Can show nodes, junctions and faces; and their respective indices.
        For documentation see EmbeddedGraph.plot()
        """
        cr = self.graph.plot(show_faces=show_faces, figsize=figsize,
                             show_node_ids=show_node_ids, show_edge_ids=show_junction_ids,
                             show_face_ids=show_face_ids, face_shrink_factor=face_shrink_factor,
                             show_boundary_face=False)
        return cr

    # abbreviations and aliases
    def _Nn(self):   # alias for node_count
        return self.graph.node_count()

    def _Nnr(self):
        # reduced node count; returns node_count - 1
        return self._Nn() - 1

    def _Nj(self):   # alias for get_junction_count()
        return self.graph.edge_count()

    def _Nf(self):
        return self.graph.face_count(include_boundary_faces=False)

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
    def _prepare_inducance_matrix(A, N):
        if not hasattr(A, "ndim"):
            A = np.array(A)
        if A.ndim <= 1:
            x = Circuit._prepare_junction_quantity(A, N, "L")
            return scipy.sparse.diags(x, 0), np.all(x >= 0), np.any(x != 0.0)
        if A.shape == (N, N):
            if not Circuit._is_symmetric(A):
                raise ValueError("inductance matrix must be symmetric")
            eigv = scipy.sparse.linalg.eigsh(-A, 1, maxiter=1000, which="LA")[0][0]
            is_positive_definite = eigv < 100 * np.finfo(float).eps
            if scipy.sparse.issparse(A):
                A = A.tocsc()
                is_zero = A.nnz == 0
            else:
                is_zero = np.all(A == 0)
            return A, is_positive_definite, is_zero
        else:
            raise ValueError("L must be scalar, (Nj,) array or (Nj, Nj) matrix")

    def __str__(self):
        return "x: \n" + str(self.graph.x) +"y: \n" + str(self.graph.y) + \
               "\nnode1: \n" + str(self.graph.node1) + "\nnode2: \n" + str(self.graph.node2)

    def _get_A_norm(self):
        # return ||A||_2 = sqrt(max(eig(A.T @ A))). (however computes sqrt(max(eig(A @ A.T))) which seems to be the same and is quicker)
        if self._Anorm is None:
            A = self.get_cycle_matrix()
            self._Anorm = np.sqrt(scipy.sparse.linalg.eigsh((A @ A.T).astype(np.double), 1, maxiter=1000, which="LA")[0][0])
        return self._Anorm

    def _get_M_norm(self):
        # return ||M||_2 = sqrt(max(eig(M.T @ M))). (however computes sqrt(max(eig(M @ M.T))) which seems to be the same and is quicker)
        if self._Mnorm is None:
            M = self.get_cut_matrix()
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
        return self.graph._cycle_space_solve_for_integral_x(b)[..., self.junction_unsort_perm]


    def _has_capacitance(self):
        # returns False if self.capacitance_factors is zero, True otherwise
        np.any(self.capacitance_factors > 0)

    def _has_inductance(self):
        # returns False if self.inductance_factors is zero, True otherwise
        return self._has_inductance_v

    def _has_mixed_inductance(self):
        mask = self._get_mixed_inductance_mask()
        return np.any(mask) and not np.all(mask)

    def _get_mixed_inductance_mask(self):
        L = self._L().matrix(self._Nj())
        A = self.get_cycle_matrix()
        ALA = A @ L @ A.T
        return np.isclose(np.array(np.sum(np.abs(ALA), axis=1))[:, 0], 0)

    def _assign_cut_matrix(self):
        self.cut_square = None
        self.cut_matrix_reduced_transposed = None
        self.cut_matrix_transposed = None
        if self.cut_matrix_reduced is None or self.cut_matrix is None:
            cut_matrix = -self.graph.cut_space()
            self.cut_matrix = cut_matrix.asformat(self.matrix_format)
            self.cut_matrix_reduced = cut_matrix[:-1, :].asformat(self.matrix_format)
            return self.cut_matrix, self.cut_matrix_reduced

    def _junction_centers(self):
        x, y = self.get_node_coordinates()
        return 0.5 * (x[self.graph.node1] + x[self.graph.node2]),  0.5 * (y[self.graph.node1] + y[self.graph.node2])

    def _junction_lengths(self):
        x, y = self.get_node_coordinates()
        return np.sqrt((x[self.graph.node2] - x[self.graph.node1]) ** 2 + (y[self.graph.node2] - y[self.graph.node1]) ** 2)

    def _junction_inner(self, ids1, ids2):
        x, y = self.get_node_coordinates()
        x_n1_j1, y_n1_j1 = x[self.graph.node1[ids1]], y[self.graph.node1[ids1]]
        x_n2_j1, y_n2_j1 = x[self.graph.node2[ids1]], y[self.graph.node2[ids1]]
        x_n1_j2, y_n1_j2 = x[self.graph.node1[ids2]], y[self.graph.node1[ids2]]
        x_n2_j2, y_n2_j2 = x[self.graph.node2[ids2]], y[self.graph.node2[ids2]]
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


class SquareArray(Circuit):

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0, matrix_format="csc"):
        super().__init__(EmbeddedSquareGraph(count_x, count_y, x_scale, y_scale),
                         matrix_format=matrix_format)

    def horizontal_junctions(self):
        x1, y1, x2, y2 = self.get_juncion_coordinates()
        return y1 == y2

    def vertical_junctions(self):
        return ~self.horizontal_junctions()


class HoneycombArray(Circuit):

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0, matrix_format="csc"):
        super().__init__(EmbeddedHoneycombGraph(count_x, count_y, x_scale, y_scale),
                         matrix_format=matrix_format)

    def horizontal_junctions(self):
        x1, y1, x2, y2 = self.get_juncion_coordinates()
        return (y1 == y2).astype(int) * np.sign(x2 - x1)

    def vertical_junctions(self):
        return ~self.horizontal_junctions()


class TriangularArray(Circuit):

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0, matrix_format="csc"):
        super().__init__(EmbeddedTriangularGraph(count_x, count_y, x_scale, y_scale),
                         matrix_format=matrix_format)

    def horizontal_junctions(self):
        x1, y1, x2, y2 = self.get_juncion_coordinates()
        return y1 == y2

    def vertical_junctions(self):
        return ~self.horizontal_junctions()


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