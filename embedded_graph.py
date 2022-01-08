import numpy as np
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from pyJJAsim.variable_row_array import VarRowArray


"""
Embedded Graph Module
"""

class NotSingleComponentError(Exception):
    pass

class NotPlanarEmbeddingError(Exception):
    pass

class NonSimpleError(Exception):
    pass

class SelfLoopError(Exception):
    pass

class NodeNotExistError(Exception):
    pass

class EdgeNotExistError(Exception):
    pass


class EmbeddedGraph:

    """
    Class for embedded 2D graphs. Can be used to construct faces and check planarity.

    Requirements:
     - No self-loops
     - Simple (no multigraph)

    """

    def __init__(self, x, y, node1, node2, require_single_component=False,
                 require_planar_embedding=False, _edges_are_sorted=False):
        """
        Parameters
        ----------
        x, y: (N,) array
            coordinates of nodes of embedded graph
        node1, node2: (E,) int array in range(N)
            endpoint nodes of edges in embedded graph. Nodes are referred to by their index in
            the coordinate arrays.
        require_single_component=False:
            If True, an error is raised if the graph is not single-component
        require_planar_embedding=False:
            If True, an error is raised if the graph is not a planar embedding

        Raises
        ------
        NonSimpleError
            If graph is not simple
        SelfLoopError
            if graph contains self-loops
        NotSingleComponentError
            if graph has disconnected components and require_single_component=True
        NotPlanarEmbeddingError
            If graph in not a planar embedding and require_planar_embedding=True
        """

        self.x = np.array(x, dtype=np.double).ravel()
        self.y = np.array(y, dtype=np.double).ravel()
        if len(self.x) != len(self.y):
            raise ValueError("x and y must be same size")
        self.N = len(self.x)
        self.node1, self.node2 = np.array(node1, dtype=int).ravel(), np.array(node2, dtype=int).ravel()
        self._assert_edges_correct_shape()
        self.E = len(self.node1)
        self._assert_edges_contain_existing_nodes()
        self._assert_no_self_loop()

        self.edge_sorter = np.arange(self.edge_count())
        if not _edges_are_sorted:
            self._sort_edges(node1, node2)

        self._assert_nonsimple()
        self.edge_v_array = self._assign_edge_v_array()

        # quantities are computed and stored only when a method needs them.
        self.F = None
        self.faces_v_array = None
        self.face_edges = None
        self.face_nodes = None
        self.face_lengths = None
        self.determinant = None
        self.areas = None
        self.roll_ids = None
        self.boundary_face_indices = None

        if require_single_component:
            self._assert_single_component()
        if require_planar_embedding:
            self._assert_planar_embedding()

    def get_edges(self):
        """
        Return node endpoints of all edges.

        Returns
        -------
        node1, node2: (E,) arrays
            node endpoints of all edges.

        Notes
        -----
        - Edges are sorted such that node1 < node2
        - After that, edges are lexographically sorted first by node1, then by node2.
        """
        return self.node1, self.node2

    def get_edge_ids(self, node1, node2):
        """
        Return ids of edges with given endpoint nodes

        Parameters
        ----------
        node1, node2: arrays in range(N)
            endpoints of edges

        Returns
        -------
        edge_ids: array with same size as node1 in range(E)
            ids of edges

        Raises
        ------
        EdgeNotExistError
            If a queried node-pair does not exist.

        Notes
        -----
        The id of an edge equals the position in the output .get_edges()
        """
        node1, node2 = np.array(node1, dtype=int).ravel(), np.array(node2, dtype=int).ravel()
        mask = node1 < node2
        node1, node2 = np.where(mask, node1, node2), np.where(mask, node2, node1)
        if np.any(node1 > self.node1[-1]):
            raise EdgeNotExistError("queried edge that does not exist")
        starts, ends = self.edge_v_array.row_ranges()
        node1_idx = np.searchsorted(self.node1[starts], node1)
        starts, ends = starts[node1_idx], ends[node1_idx]
        while not np.all(node2 <= self.node2[starts]):
            starts[node2 > self.node2[starts]] += 1
            if not np.all(starts < ends):
                raise EdgeNotExistError("queried edge that does not exist")
        if not np.all((self.node1[starts] == node1) & (self.node2[starts] == node2)):
            raise EdgeNotExistError("queried edge that does not exist")
        return starts

    def add_nodes(self, x, y):
        """
        Add nodes to graph.

        Parameters
        ----------
        x, y: arrays
            coordinates of the nodes to be added to the graph

        Returns
        -------
        new_graph: EmbeddedGraph
            new graph with nodes added to it.
        """
        return EmbeddedGraph(np.append(self.x, np.array(x, dtype=np.double).ravel()), np.append(self.y, np.array(y, dtype=np.double).ravel()), self.node1, self.node2, require_single_component=False)

    def add_edges(self, node1, node2, require_single_component=False, require_planar_embedding=False):
        """
        Add edges to graph.

        Parameters
        ----------
        node1, node2: arrays in range(N)
            endpoints of edges
        require_single_component=False:
            If True; raises error if the resulting graph is not single-component
        require_planar_embedding=False:
            If True; raises error if the resulting graph is not a planar embedding

        Returns
        -------
        new_graph: EmbeddedGraph
            new graph with edges added to it.

        """
        return EmbeddedGraph(self.x, self.y, np.append(self.node1, np.array(node1, dtype=int).ravel()),
                             np.append(self.node2, np.array(node2, dtype=int).ravel()),
                             require_single_component=require_single_component,
                             require_planar_embedding=require_planar_embedding)

    def add_nodes_and_edges(self, x, y, node1, node2, require_single_component=False,
                            require_planar_embedding=False):
        """
        Add nodes and edges to graph.

        Parameters
        ----------
        x, y: arrays
            coordinates of the nodes to be added to the graph
        node1, node2: arrays in range(N + x.size)
            endpoints of edges. The i-th new node must be referred to by index N + i.
        require_single_component=False:
            If True; raises error if the resulting graph is not single-component
        require_planar_embedding=False:
            If True; raises error if the resulting graph is not a planar embedding

        Returns
        -------
        new_graph: EmbeddedGraph
            new graph with nodes and edges added to it.

        """
        return EmbeddedGraph(np.append(self.x, np.array(x, dtype=np.double).ravel()),
                             np.append(self.y, np.array(y, dtype=np.double).ravel()),
                             np.append(self.node1, np.array(node1, dtype=int).ravel()),
                             np.append(self.node2, np.array(node2, dtype=int).ravel()),
                             require_single_component=require_single_component,
                             require_planar_embedding=require_planar_embedding)

    def remove_nodes(self, node_ids, require_single_component=False, require_planar_embedding=False):
        """
        Remove nodes from graph.

        Parameters
        ----------
        node_ids: int array in range(N)
            ids of nodes to be removed
        require_single_component=False:
            If True; raises error if the resulting graph is not single-component
        require_planar_embedding=False:
            If True; raises error if the resulting graph is not a planar embedding

        Returns
        -------
        new_graph: EmbeddedGraph
            new graph with nodes removed from it.
        """
        node_ids = np.array(node_ids, dtype=int).ravel()
        node_map = np.ones(self.node_count(), dtype=int)
        node_map[node_ids] = 0
        node_map = np.cumsum(node_map) - node_map
        node_map[node_ids] = -1
        n1, n2 = node_map[self.node1], node_map[self.node2]
        edge_ids = (n1 >= 0) & (n2 >= 0)
        return EmbeddedGraph(np.delete(self.x, node_ids), np.delete(self.y, node_ids),
                             n1[edge_ids], n2[edge_ids], require_single_component=require_single_component,
                             require_planar_embedding=require_planar_embedding, _edges_are_sorted=True)

    def remove_edges_by_ids(self, edge_ids, require_single_component=False, require_planar_embedding=False):
        """
        Remove edges from graph with edge ids as input.

        Parameters
        ----------
        edge_ids: int array in range(E)
            ids of nodes to be removed
        require_single_component=False:
            If True; raises error if the resulting graph is not single-component
        require_planar_embedding=False:
            If True; raises error if the resulting graph is not a planar embedding

        Returns
        -------
        new_graph: EmbeddedGraph
            new graph with edges removed from it.
        """
        return EmbeddedGraph(self.x, self.y, np.delete(self.node1, edge_ids),
                             np.delete(self.node2, edge_ids), _edges_are_sorted=True,
                             require_single_component=require_single_component,
                             require_planar_embedding=require_planar_embedding)

    def remove_edges(self, node1, node2, require_single_component=False, require_planar_embedding=False):
        """
        Remove edges from graph with node endpoints as input.

        Parameters
        ----------
        node1, node2: int arrays in range(N)
            ids of node endpoints of edges to be removed
        require_single_component=False:
            If True; raises error if the resulting graph is not single-component
        require_planar_embedding=False:
            If True; raises error if the resulting graph is not a planar embedding

        Returns
        -------
        new_graph: EmbeddedGraph
            new graph with edges removed from it.

        Raises
        ------
        EdgeNotExistError
            If a queried node-pair does not exist.
        """
        return self.remove_edges_by_ids(self.get_edge_ids(node1, node2),
                                        require_single_component=require_single_component,
                                        require_planar_embedding=require_planar_embedding)

    def coo(self):
        """
        Returns coordinates of nodes

        Returns
        -------
        x, y: (N,) arrays
            coordinates of nodes in graph
        """
        return self.x, self.y

    def node_count(self):
        """
        Returns number of nodes in graph (abbreviated by N)
        """
        return self.N

    def edge_count(self):
        """
        Returns number of edges in graph (abbreviated by E)
        """
        return self.E


    def get_face_nodes(self, include_boundary_faces=True, to_list=False):
        """
        Returns all faces; a face is a list of all nodes in its boundary.

        Two types of faces exist:
         * boundary-faces: surround whole components. Nodes in clockwise order.
         * internal-faces: surround individual faces. Nodes in counter-clockwise order.

        Parameters
        ----------
        include_boundary_faces=True:
            If True, also returns boundary faces in output. Otherwise only internal faces.
        to_list=False:
            If true, output is in form of list-of-lists, otherwise concatenated array.

        Returns
        -------
        nodes:  list-of-lists or array
            For all faces the nodes in its boundary. Either as list of lists or concatenated
            in a single array.
        lengths: (F,) int array
            Array containing the number of nodes in each face

        """
        if self.face_edges is None:
            self._get_faces()
        if self.face_nodes is None:
            e_idx = self.face_edges % self.edge_count()
            self.face_nodes = np.where(self.face_edges < self.edge_count(), self.node1[e_idx], self.node2[e_idx])
        if include_boundary_faces:
            nodes, lengths = self.face_nodes, self.face_lengths
        else:
            if self.boundary_face_indices is None:
                self.get_areas()
            nodes = self.faces_v_array.delete_rows(self.face_nodes, self.boundary_face_indices)
            lengths = np.delete(self.face_lengths, self.boundary_face_indices)
        if to_list:
            return VarRowArray(lengths).to_list(nodes)
        return nodes, lengths

    def face_count(self, include_boundary_faces=True):
        """
        Returns number of faces in graph (abbreviated by F)

        Two types of faces exist:
         * boundary-faces: surround whole components.
         * internal-faces: surround individual faces.

        Parameters
        ----------
        include_boundary_faces=True:
            If True, also includes boundary faces in count. Otherwise only internal faces.

        Returns
        -------
        face_count:
            Returns number of faces
        """
        if self.F is None:
            self._get_faces()
        if include_boundary_faces:
            return self.F
        else:
            return self.F - len(self.get_boundary_faces())

    def is_planar_embedding(self):
        """
        Returns if graph is planar embedding, which is true if edges only intersect at their endpoints.
        """
        if self.face_edges is None:
            self._get_faces()
        return self.node_count() + self.face_count(False) == self.edge_count() + 1

    def get_areas(self, include_boundary_faces=True):
        """
        Returns signed area of faces in graph

        Two types of faces exist:
         * boundary-faces: surround whole components. Nodes in clockwise order,
           so negative area.
         * internal-faces: surround individual faces. Nodes in counter-clockwise
           order, so positive area.

        Parameters
        ----------
        include_boundary_faces=True:
            If True, also includes boundary faces in output. Otherwise only internal faces.

        Returns
        -------
        areas:  (F,) array
            Returns areas of faces
        """
        if self.areas is None:
            if self.determinant is None:
                self._compute_determinant()
            self.areas = 0.5 * self.faces_v_array.sum(self.determinant)
        if self.boundary_face_indices is None:
            self.boundary_face_indices = np.flatnonzero((self.areas < 0) | np.isclose(self.areas, 0.0))
        if include_boundary_faces:
            return self.areas
        else:
            return np.delete(self.areas, self.boundary_face_indices)

    def get_centroids(self, include_boundary_faces=True):
        """
        Returns centroids of faces in graph

        Two types of faces exist:
         * boundary-faces: surround whole components.
         * internal-faces: surround individual faces.

        Parameters
        ----------
        include_boundary_faces=True:
            If True, also includes boundary faces in output. Otherwise only internal faces.

        Returns
        -------
        x, y:  (F,) arrays
            Returns coordinates of centroids of faces
        """
        six_times_area = 6.0 * self.get_areas()
        mask = np.isclose(six_times_area, 0.0)
        six_times_area[mask] = 2.0 * self.face_lengths[mask]
        long_mask = self.faces_v_array.at_out_index(mask)
        X = self.x[self.face_nodes] + self.x[self.roll_ids]
        Y = self.y[self.face_nodes] + self.y[self.roll_ids]
        centroid_x = self.faces_v_array.sum(np.where(long_mask, X, self.determinant * X)) / six_times_area
        centroid_y = self.faces_v_array.sum(np.where(long_mask, Y, self.determinant * Y)) / six_times_area
        if include_boundary_faces:
            return centroid_x, centroid_y
        else:
            if self.boundary_face_indices is None:
                self.get_boundary_faces()
            return np.delete(centroid_x, self.boundary_face_indices), \
                   np.delete(centroid_y, self.boundary_face_indices)

    def get_num_components(self):
        if self.areas is None:
            self.get_areas()
        mask = np.ones(self.node_count(), dtype=bool)
        mask[self.node1] = False
        mask[self.node2] = False
        return len(self.boundary_face_indices) + np.sum(mask)

    def get_boundary_faces(self):
        if self.areas is None:
            self.get_areas()
        return self.boundary_face_indices

    def cycle_space(self, include_boundary_faces=False):
        face_edges, face_lengths = self._get_faces(include_boundary_faces=include_boundary_faces)
        E, F = self.edge_count(), len(face_lengths)
        indptr = np.append([0], np.cumsum(face_lengths))
        indices, data = face_edges % E, 1 - 2 * (face_edges // E)
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=(F, E)).tocsc()

    def cut_space(self):
        E, N = self.edge_count(), self.node_count()
        row = np.concatenate((self.node1, self.node2))
        col = np.concatenate((np.arange(E), np.arange(E)))
        data = np.concatenate((-np.ones(E), np.ones(E)))
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(N, E)).tocsc()

    def adjacency_matrix(self):
        E, N = self.edge_count(), self.node_count()
        data = np.ones(2 * E)
        row = np.append(self.node1, self.node2)
        col = np.append(self.node2, self.node1)
        return scipy.sparse.coo_matrix((data, (row, col)), shape=(N, N)).tocsc()

    def get_common_edge_of_faces(self, face1, face2, return_orientation=False):
        # returns index of an edge occurring in both faces if it exists; otherwise -1.
        # If multiple exist; returns the lowest index.
        # ({s},) ({s},) -> ({s},) in range(E) or -1
        # optionally return orientation; True if face1 passes edge counterclockwise
        # first encountering its node with lowest index
        if self.face_edges is None:
            self._get_faces()
        f = self.faces_v_array.rows()[np.argsort(self.face_edges)].astype(np.int64)
        E = self.edge_count()
        f1, f2 = f[:E], f[E:2 * E]
        mask = f1 < f2
        f1, f2 = np.where(mask, f1, f2),  np.where(mask, f2, f1)

        sorter = np.lexsort((f2, f1))
        f1, f2 = f1[sorter],  f2[sorter]
        A = np.array(list(zip(f1, f2)), dtype=[('f1', 'int'), ('f2', 'int')])

        face1, face2 = np.array(face1, dtype=int), np.array(face2, dtype=int)
        input_shape = face1.shape
        face1, face2 = face1.ravel(), face2.ravel()
        mask2 = face1 < face2
        fmin, fmax = np.where(mask2, face1, face2).astype(np.int64), np.where(mask2, face2, face1).astype(np.int64)
        V = np.array(list(zip(fmin, fmax)), dtype=[('fmin', 'int'), ('fmax', 'int')])

        edge_ids = np.searchsorted(A, V)
        edge_ids[edge_ids >= E] = E - 1
        if return_orientation:
            orientation = mask[sorter][edge_ids] ^ ~mask2

        found_mask = (f1[edge_ids] != fmin) | (f2[edge_ids] != fmax)
        edge_ids = sorter[edge_ids]

        edge_ids[found_mask] = -1

        if return_orientation:
            orientation[found_mask] = False
            return edge_ids.reshape(input_shape), orientation.reshape(input_shape)
        else:
            return edge_ids.reshape(input_shape)

    def permute_nodes(self, permutation):
        # first precompute faces, because face order will be different if computed after node permutation.
        if self.face_edges is None:
            self._get_faces()
        if self.face_nodes is None:
            self.get_face_nodes()
        if self.areas is None:
            self.get_areas()
        permutation = np.array(permutation, dtype=int)
        if not np.all(np.sort(permutation) == np.arange(self.node_count())):
            raise ValueError("invalid permutation")
        self.x = self.x[permutation]
        self.y = self.y[permutation]
        inv_perm = np.argsort(permutation)
        self.node1 = inv_perm[self.node1]
        self.node2 = inv_perm[self.node2]
        edge_sorter = self._sort_edges(self.node1, self.node2)
        self.edge_v_array = self._assign_edge_v_array()
        inv_edge_sorter = np.argsort(np.append(edge_sorter, (edge_sorter + self.edge_count()) % (2 * self.edge_count())))
        self.face_edges = inv_edge_sorter[self.face_edges]
        self.face_nodes = inv_perm[self.face_nodes]

    def permute_faces(self, permutation):
        permutation = np.array(permutation, dtype=int)
        if not np.all(np.sort(permutation) == np.arange(self.face_count())):
            raise ValueError("invalid permutation")
        inv_permutation = np.argsort(permutation)
        if self.face_edges is None:
            self._get_faces()
        self.face_lengths = self.face_lengths[permutation]
        self.face_edges = self.faces_v_array.permute_rows(permutation, self.face_edges)
        if self.areas is not None:
            self.areas = self.areas[permutation]
            self.determinant = self.faces_v_array.permute_rows(permutation, self.determinant)
            self.roll_ids = self.faces_v_array.permute_rows(permutation, self.roll_ids)
            self.boundary_face_indices = inv_permutation[self.boundary_face_indices]
        if self.face_nodes is not None:
            self.face_nodes = self.faces_v_array.permute_rows(permutation, self.face_nodes)
        self.faces_v_array = VarRowArray(self.faces_v_array.counts[permutation])

    def plot(self, show_faces=True, figsize=[5, 5], show_node_ids=False, show_edge_ids=False,
             show_face_ids=False, face_shrink_factor=0.9, show_boundary_face=False):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        x, y = self.coo()
        self.ax.plot([x[self.node1], x[self.node2]], [y[self.node1], y[self.node2]], color=[0.5,0.5,0.5])
        self.ax.plot([x], [y], color=[0,0,0], marker="o", markerfacecolor=[0,0,0])
        if show_node_ids:
            for i, (xn, yn) in enumerate(zip(x, y)):
                self.ax.annotate(i.__str__(), (xn, yn))
        if show_edge_ids:
            x1, y1,  x2, y2 = x[self.node1], y[self.node1], x[self.node2], y[self.node2]
            for i, (xn, yn) in enumerate(zip(0.5 * (x1 + x2), 0.5 * (y1 + y2))):
                self.ax.annotate(i.__str__(), (xn, yn), color=[0.3, 0.5, 0.9], ha='center', va='center')
        if show_faces:
            xc, yc = self.get_centroids()
            pn = self.get_face_nodes(to_list=True)
            areas = self.get_areas()
            bi = self.boundary_face_indices
            nr=0
            for i, (xcn, ycn, n, area) in enumerate(zip(xc, yc, pn, areas)):
                if np.in1d(i, bi):
                    if show_boundary_face:
                        xp, yp = x[n], y[n]
                        self.ax.plot(np.append(xp, xp[0]), np.append(yp, yp[0]), color=[0.2, 0.5, 1])
                        if show_face_ids:
                            self.ax.annotate(nr.__str__(), (xcn, ycn), color=[0.2, 0.5, 1], ha='center', va='center')
                            nr+=1
                else:
                    xp = face_shrink_factor * x[n] + (1 - face_shrink_factor) * xcn
                    yp = face_shrink_factor * y[n] + (1 - face_shrink_factor) * ycn
                    self.ax.plot(np.append(xp, xp[0]), np.append(yp, yp[0]), color=[1, 0.5, 0.2])
                    if show_face_ids:
                        self.ax.annotate(nr.__str__(), (xcn, ycn), color=[1, 0.5, 0.2], ha='center', va='center')
                        nr += 1
        return self.fig, self.ax

    def _assert_edges_correct_shape(self):
        if len(self.node1) != len(self.node2):
            raise ValueError("node1 and node2 must be same size")

    def _assert_edges_contain_existing_nodes(self):
        if np.any((self.node1 < 0) | (self.node1 >= self.N) | (self.node2 < 0) | (self.node2 >= self.N)):
            raise NodeNotExistError("node1,2 values must be in range(N)")

    def _assert_single_component(self):
        if self.get_num_components() != 1:
            raise NotSingleComponentError()

    def _assert_planar_embedding(self):
        if not self.is_planar_embedding():
            raise NotPlanarEmbeddingError()

    def _assert_no_self_loop(self):
        if np.any(self.node1 == self.node2):
            raise SelfLoopError("no edge is allowed to have identical end nodes.")

    def _assert_nonsimple(self):
        if np.any((self.node1[:-1] == self.node1[1:]) & (self.node2[:-1] == self.node2[1:])):
            raise NonSimpleError("no duplicate edges allowed.")

    def _assign_edge_v_array(self):
        self.edge_v_array = VarRowArray(np.diff(np.append(np.flatnonzero(np.roll(self.node1, 1) - self.node1), len(self.node1))))
        return self.edge_v_array

    def _reset_precomputed_quantities(self):
        self.F = None
        self.faces_v_array = None
        self.face_edges = None
        self.face_nodes = None
        self.face_lengths = None
        self.determinant = None
        self.areas = None
        self.roll_ids = None
        self.boundary_face_indices = None

    def _get_faces(self, include_boundary_faces=True, to_list=False):
        if self.face_edges is None:
            self.face_edges, self.face_lengths = self._construct_faces()
            self.faces_v_array = VarRowArray(self.face_lengths)
            self.F = len(self.face_lengths)
        if include_boundary_faces:
            edges, lengths = self.face_edges, self.face_lengths
        else:
            if self.boundary_face_indices is None:
                self.get_areas()
            edges = self.faces_v_array.delete_rows(self.face_edges, self.boundary_face_indices)
            lengths = np.delete(self.face_lengths, self.boundary_face_indices)
        if to_list:
            return VarRowArray(lengths).to_list(edges)
        return edges, lengths

    def _sort_edges(self, node1, node2):
        node1, node2 = np.array(node1, dtype=int),  np.array(node2, dtype=int)
        mask = node1 < node2
        self.node1, self.node2 = np.where(mask, node1, node2), np.where(mask, node2, node1)
        self.edge_sorter = np.lexsort((self.node2, self.node1))
        self.node1, self.node2 = self.node1[self.edge_sorter], self.node2[self.edge_sorter]
        return self.edge_sorter + (~mask).astype(int) * self.edge_count()

    def _compute_determinant(self):
        if self.roll_ids is None:
            self._compute_roll_ids()
        self.determinant = self.x[self.roll_ids] * self.y[self.face_nodes] - self.x[self.face_nodes] * self.y[self.roll_ids]

    def _compute_roll_ids(self):
        if self.face_edges is None:
            self._get_faces()
        if self.face_nodes is None:
            self.get_face_nodes()
        self.roll_ids = self.face_nodes[self.faces_v_array.roll(1)]

    def _get_edge_map(self):
        """
        Computes a one-to-one map over directed edges, where the image contains the edge
        which is encountered next when traversing the graph moving counter-clockwise.

        Used to generate faces by repeating indexing: e_(i+1) = map[e_i]

        map: np.arange(2*E) -> sorted_out_edge_directed

        Output:
        sorted_out_edge_directed    (2*E,) in range(2*E)
        """

        counter_clockwise = True
        edge_count = self.edge_count()
        ns = np.append(self.node1, self.node2)

        # construct neighbour structure
        nodes, count = np.unique(ns, return_counts=True)
        neighbours = VarRowArray(count)
        neighbour_edges = np.tile(np.arange(edge_count), 2)[np.argsort(ns)]
        neighbour_node_self = nodes[neighbours.rows()]
        neighbour_node_other = np.where(self.node1[neighbour_edges] == neighbour_node_self,
                                        self.node2[neighbour_edges], self.node1[neighbour_edges])

        # sort neighbour in-dimension by ascending angle
        angles = np.arctan2(self.y[neighbour_node_other] - self.y[neighbour_node_self],
                            self.x[neighbour_node_other] - self.x[neighbour_node_self])
        angle_arg_sort = np.argsort(3 * np.pi * neighbour_node_self.astype(np.double) + angles)
        neighbour_edges = neighbour_edges[angle_arg_sort]
        neighbour_node_other = neighbour_node_other[angle_arg_sort]

        def to_directed(edge_nr, edge_direction, edge_count):
            return edge_nr + edge_count * (1 - edge_direction.astype(int))

        # create combined-index for input edges of the map
        neighbour_edges_direction = neighbour_node_other < neighbour_node_self
        in_edge_combined_index = to_directed(neighbour_edges, neighbour_edges_direction, edge_count)

        # find the map from every (combined index) edge to the next edge in (counter-)-clockwise direction
        edges_map = neighbours.roll() if counter_clockwise else neighbours.roll(-1)

        # find the combined index of the output edges of the map
        out_edge_directed = to_directed(neighbour_edges, ~neighbour_edges_direction, edge_count)[edges_map]

        # sort the edge-map based on the (combined index of the) input edges
        sorted_out_edge_directed = out_edge_directed[np.argsort(in_edge_combined_index)]

        return sorted_out_edge_directed

    def _construct_faces(self):
        """
        Computes faces of the embedded graph. This is done using repeated iteration over the
        one-to-one map over directed edges computed with _get_edge_map(). It starts with all
        edges pointing from lowest to highest node to ensure all faces are found.

        Used to generate faces by repeating indexing: e_(i+1) = map[e_i]

        map: np.arange(2*E) -> sorted_out_edge_directed

        Output:
        sorted_out_edge_directed    (2*E,) in range(2*E)
        """

        map = self._get_edge_map()
        edge_ids = np.arange(self.edge_count())
        paths = -np.ones((3, self.edge_count()), dtype=int)
        paths[0, :] = edge_ids
        path_lengths = np.ones(self.edge_count(), dtype=int)
        current_path_length, out_paths, out_path_lengths = 1, np.zeros(0, dtype=int), np.zeros(0, dtype=int)

        # iteration doing counter-clockwise walks through the graphs starting from each junction
        while len(edge_ids) > 0:
            edge_ids = map[edge_ids]
            is_terminated = edge_ids == paths[0, :]
            if np.any(is_terminated):
                out_path_lengths, out_paths = self._store_terminated_paths(paths[:current_path_length, is_terminated].T,
                                                                            out_path_lengths, out_paths)
                paths, path_lengths = paths[:, ~is_terminated], path_lengths[~is_terminated]
                edge_ids = edge_ids[~is_terminated]
            paths[current_path_length, :] = edge_ids
            current_path_length += 1
            if paths.shape[0] == (current_path_length):
                paths = np.append(paths, -np.ones(paths.shape, dtype=int), axis=0)
        return out_paths, out_path_lengths

    def _store_terminated_paths(self, terminated_paths, out_path_lengths, out_paths):
        # roll paths until the lowest node index is in the first column (so its easier to remove duplicate paths)
        path_cnt, path_len = terminated_paths.shape
        paths = terminated_paths[np.arange(path_cnt)[:, None], (
                    np.arange(path_len) + np.argmin(terminated_paths, axis=1)[:, None]) % path_len]

        _, idx = np.unique(paths[:, 0], return_index=True)
        paths = paths[idx, :]
        #
        # paths = np.unique(paths, axis=0)

        out_path_lengths = np.append(out_path_lengths, path_len * np.ones(paths.shape[0], dtype=int))
        out_paths = np.append(out_paths, paths.flatten())
        return out_path_lengths, out_paths

    def _non_boundary_mask(self):
        out = np.ones(self.face_count(include_boundary_faces=True), dtype=bool)
        out[self.boundary_face_indices] = False
        return out

    def _cycle_space_solve_for_integral_x(self, b):
        """
        Solves the equation: A @ x = b (where A = cycle_matrix, without boundary faces).
        If b is integral (contain only integers), the output array x will also be integral.

        input:  b (..., F)
        output: x (..., E)

        Notes:
            - The equation is underdetermined, so the solution x is not unique.
        """
        if (self.get_num_components() != 1) or (not self.is_planar_embedding()):
            raise ValueError("only implemented for single component planar embedding")

        E, F = self.edge_count(), self.face_count(include_boundary_faces=False)
        b = np.array(b, dtype=np.double)
        b_shape = list(b.shape)
        b = b.reshape(-1, F)
        b_tally = b.copy()

        # insert boundary face in b_tally
        b_idx = self.boundary_face_indices[0]
        b_tally = np.concatenate((b_tally[:, :b_idx], np.zeros((b_tally.shape[0], 1)), b_tally[:, b_idx:]), axis=1)

        # do depth first search, resulting in cur (current node of tree) and prev (parent node of cur)
        A = self.cycle_space(include_boundary_faces=True)
        cur, predecessor = scipy.sparse.csgraph.depth_first_order(A @ A.T, b_idx)
        prev = predecessor[cur]
        prev[0] = -1

        # find map of edge between pair of faces. signs returns +1 if face1 counterclockwise passes resulting edge in its own direction.
        juncs, orientation_in_face_1 = self.get_common_edge_of_faces(cur, prev, return_orientation=True)
        sgns = (-1 + 2 * orientation_in_face_1.astype(np.double))

        # construct x at each edge by passing through the tree in reverse.
        x = np.zeros((b.shape[0], E), dtype=b.dtype)
        for i in reversed(range(F+1)):
            if prev[i] >= 0:
                b_tally[:, prev[i]] += b_tally[:, cur[i]]
                x[:, juncs[i]] += b_tally[:, cur[i]] * sgns[i]
                b_tally[:, cur[i]] = 0

        # check if resulting x solves A @ x == b
        b_shape[-1] = E
        mask = self._non_boundary_mask()
        if not np.allclose((A @ x.T)[mask, :], b.T):
            raise ValueError("failed integral solve of cycle space linear problem")

        # return x
        return x.reshape(tuple(b_shape))

class EmbeddedSquareGraph(EmbeddedGraph):

    def __init__(self, count_x, count_y,  x_scale=1.0, y_scale=1.0):
        y, x = np.mgrid[0:count_y, 0:count_x]
        idx = np.arange(count_x * count_y).reshape(count_y, count_x)
        n1 = np.concatenate((idx[:, 0:-1].ravel(), idx[0:-1, :].ravel()))
        n2 = np.concatenate((idx[:, 1:].ravel(), idx[1:, :].ravel()))
        super().__init__(x * x_scale, y * y_scale, n1, n2)


class EmbeddedHoneycombGraph(EmbeddedGraph):

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0):
        y, x = np.mgrid[0:count_y, 0:count_x]
        x1, y1 = 3.0 * x, np.sqrt(3.0) * y
        nodes_x = np.concatenate((x1, x1 + 0.5, x1 + 1.5, x1 + 2), axis=0).ravel()
        nodes_y = np.concatenate((y1, y1 + np.sqrt(0.75), y1 + np.sqrt(0.75), y1), axis=0).ravel()
        idx = np.arange(count_x * count_y).reshape(count_y, count_x)
        s = count_x * count_y
        nodes1 = (idx,   idx[:-1, :]+s, idx+s,   idx[:-1, :]+2*s, idx+2*s, idx[:, :-1]+3*s)
        nodes2 = (idx+s, idx[1:, :],    idx+2*s, idx[1:, :]+3*s,  idx+3*s, idx[:, 1:])
        nodes1 = np.concatenate(tuple([n1.flatten() for n1 in nodes1])).ravel()
        nodes2 = np.concatenate(tuple([n2.flatten() for n2 in nodes2])).ravel()

        remove_node_ids = [0, idx[0, -1] + 3 * s]
        nodes_x, nodes_y, nodes1, nodes2 = remove_nodes(nodes_x, nodes_y, nodes1, nodes2, remove_node_ids)

        super().__init__(nodes_x * x_scale, nodes_y * y_scale, nodes1, nodes2)


class EmbeddedTriangularGraph(EmbeddedGraph):

    def __init__(self, count_x, count_y, x_scale=1.0, y_scale=1.0):
        y, x = np.mgrid[0:count_y, 0:count_x]
        x1, y1 = x, np.sqrt(3.0) * y
        nodes_x = np.concatenate((x1, x1 + 0.5), axis=0)
        nodes_y = np.concatenate((y1, y1 + np.sqrt(0.75)), axis=0)
        idx = np.arange(count_x * count_y).reshape(count_y, count_x)
        s = count_x * count_y
        nodes1 = (idx,   idx[:, :-1], idx[:-1, :]+s, idx[:-1, :-1]+s, idx[:, :-1]+s, idx[:, 1:])
        nodes2 = (idx+s, idx[:, 1:],  idx[1:, :],    idx[1:, 1:],     idx[:, 1:]+s,  idx[:, :-1]+s)
        nodes1 = np.concatenate(tuple([n1.flatten() for n1 in nodes1]))
        nodes2 = np.concatenate(tuple([n2.flatten() for n2 in nodes2]))
        super().__init__(nodes_x * x_scale, nodes_y * y_scale, nodes1, nodes2)

def remove_nodes(x, y, nodes1, nodes2, remove_node_ids):
    remove_node_ids = np.array(remove_node_ids)
    node_map = np.ones(len(x), dtype=int)
    node_map[remove_node_ids] = 0
    node_map = np.cumsum(node_map) - node_map
    remove_edges = np.in1d(nodes1, remove_node_ids) | np.in1d(nodes2, remove_node_ids)
    x = np.delete(x, remove_node_ids)
    y = np.delete(y, remove_node_ids)
    nodes1 = node_map[nodes1[~remove_edges]]
    nodes2 = node_map[nodes2[~remove_edges]]
    return x, y, nodes1, nodes2

def remove_edges(nodes1, nodes2, remove_edge_ids):
    remove_edge_ids = np.array(remove_edge_ids)
    edge_map = np.ones(len(nodes1), dtype=int)
    edge_map[remove_edge_ids] = 0
    edge_map = np.cumsum(edge_map) - edge_map
    nodes1 = edge_map[np.delete(nodes1, remove_edge_ids)]
    nodes2 = edge_map[np.delete(nodes2, remove_edge_ids)]
    return nodes1, nodes2
