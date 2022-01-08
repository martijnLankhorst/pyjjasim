import unittest
import numpy as np
from matplotlib import pyplot as plt

from embedded_graph import EmbeddedGraph, NotSingleComponentError, NotPlanarEmbeddingError, EmbeddedSquareGraph, \
    NonSimpleError, SelfLoopError, EdgeNotExistError, EmbeddedHoneycombGraph
from tests.examples import array_A, array_B, array_C, array_D, array_E, array_F, array_G, array_H, array_I, array_J, \
    array_K, array_L


class TestEmbeddedGraph(unittest.TestCase):

    def setUp(self) -> None:
        self.graph_A = EmbeddedGraph(array_A.x, array_A.y, array_A.n1, array_A.n2)
        self.graph_B = EmbeddedGraph(array_B.x, array_B.y, array_B.n1, array_B.n2)
        self.graph_C = EmbeddedGraph(array_C.x, array_C.y, array_C.n1, array_C.n2)
        self.graph_D = EmbeddedGraph(array_D.x, array_D.y, array_D.n1, array_D.n2)
        self.graph_E = EmbeddedGraph(array_E.x, array_E.y, array_E.n1, array_E.n2)
        self.graph_F = EmbeddedGraph(array_F.x, array_F.y, array_F.n1, array_F.n2)
        self.graph_G = EmbeddedGraph(array_G.x, array_G.y, array_G.n1, array_G.n2)
        self.graph_H = EmbeddedGraph(array_H.x, array_H.y, array_H.n1, array_H.n2)
        self.graph_I = EmbeddedGraph(array_I.x, array_I.y, array_I.n1, array_I.n2)
        self.graph_J = EmbeddedGraph(array_J.x, array_J.y, array_J.n1, array_J.n2)
        self.graph_K = EmbeddedGraph(array_K.x, array_K.y, array_K.n1, array_K.n2)
        self.graph_L = EmbeddedGraph(array_L.x, array_L.y, array_L.n1, array_L.n2)
        self.graph_sq = EmbeddedSquareGraph(3, 4, 0.5, 2)

    def test_graph_creation(self):
        with self.assertRaises(NotSingleComponentError):
            EmbeddedGraph([0, 1, 1.5, 1, 0], [0, 0, 0, 1, 1], [0, 4, 1],
                          [1, 0, 2], require_single_component=True)
        with self.assertRaises(NotSingleComponentError):
            EmbeddedGraph([0, 1, 0, 3, 4, 3], [0, 0, 1, 0, 0, 1], [0, 1, 2, 3, 4, 5],
                          [1, 2, 0, 4, 5, 3], require_single_component=True)
        EmbeddedGraph([0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 2, 3], [1, 2, 3, 0])
        with self.assertRaises(NotPlanarEmbeddingError):
            EmbeddedGraph([0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3],
                          require_planar_embedding=True)

    def test_get_edge_ids(self):
        self.assertEqual(self.graph_sq.get_edge_ids(0, 1), 0)
        self.assertEqual(self.graph_sq.get_edge_ids(1, 0), 0)
        self.assertTrue(np.all(self.graph_sq.get_edge_ids([2, 4, 0], [5, 3, 3]) == [4, 5, 1]))
        with self.assertRaises(EdgeNotExistError):
            self.graph_sq.get_edge_ids(0, 2)
        with self.assertRaises(EdgeNotExistError):
            self.graph_sq.get_edge_ids(0, 0)
        with self.assertRaises(EdgeNotExistError):
            self.graph_sq.get_edge_ids([2, -1], [5, 0])
        with self.assertRaises(EdgeNotExistError):
            self.graph_sq.get_edge_ids([2, 1, 12], [5, 0, 1])
        self.assertTrue(np.all(self.graph_sq.get_edge_ids([2, 4, 0, 5, 7], [5, 3, 3, 2, 8]) == [4, 5, 1, 4, 12]))
        g = EmbeddedHoneycombGraph(200, 200, 1, 1)
        s = np.random.randint(0, g.edge_count(), 100000)
        n1, n2 = g.node1[s], g.node2[s]
        self.assertTrue(np.all(g.get_edge_ids(n1, n2) == s))

    def test_add_nodes(self):
        g = self.graph_A.add_nodes([1.1, 2.2, 3.3], [4.3, 5.2, 6.1])
        self.assertTrue(np.allclose(g.x, [0, 1, 0, 1, 2, 1, 2, 1.1, 2.2, 3.3]))
        self.assertTrue(np.allclose(g.y, [2, 2, 1, 1, 1, 0, 0, 4.3, 5.2, 6.1]))
        self.assertTrue(g.node_count() == 10)
        g = self.graph_A.add_nodes([], [])
        self.assertTrue(np.allclose(g.x, [0, 1, 0, 1, 2, 1, 2]))
        self.assertTrue(np.allclose(g.y, [2, 2, 1, 1, 1, 0, 0]))
        self.assertTrue(g.node_count() == 7)
        g = self.graph_A.add_nodes(2, 3)
        self.assertTrue(np.allclose(g.x, [0, 1, 0, 1, 2, 1, 2, 2]))
        self.assertTrue(np.allclose(g.y, [2, 2, 1, 1, 1, 0, 0, 3]))
        self.assertTrue(g.node_count() == 8)
        g = EmbeddedSquareGraph(3, 4, 1, 2)
        g.add_nodes([1.1, 2.2, 3.3], [4.3, 5.2, 6.1], in_place=True)
        self.assertTrue(np.allclose(g.x, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1.1, 2.2, 3.3]))
        self.assertTrue(g.node_count() == 15)

    def test_add_edges(self):
        g = self.graph_sq.add_edges([0, 1], [4, 5])
        self.assertTrue(np.allclose(g.get_areas(), [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, -6]))
        with self.assertRaises(NonSimpleError):
            self.graph_sq.add_edges([0, 1], [4, 4])
        with self.assertRaises(NonSimpleError):
            g = EmbeddedSquareGraph(3, 4, 1, 2)
            g.add_edges([0, 1], [4, 4], in_place=True)
        g = self.graph_sq.add_edges([[0], [1]], [[4], [5]])
        self.assertTrue(np.allclose(g.get_areas(), [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, -6]))
        g = self.graph_sq.add_edges([], [])
        self.assertTrue(np.allclose(g.get_areas(), [1, 1, 1, 1, 1, 1, -6]))
        with self.assertRaises(SelfLoopError):
            g = EmbeddedSquareGraph(3, 4, 1, 2)
            g.add_edges([0, 2, 1], [4, 2, 5], in_place=True)
        with self.assertRaises(SelfLoopError):
            self.graph_sq.add_edges([0, 2, 1], [4, 2, 5], in_place=False)
        with self.assertRaises(NotPlanarEmbeddingError):
            g = EmbeddedSquareGraph(3, 4, 1, 2)
            g.add_edges([0, 1, 1], [4, 5, 3], in_place=True, require_planar_embedding=True)
        with self.assertRaises(NotPlanarEmbeddingError):
            self.graph_sq.add_edges([0, 1, 1], [4, 5, 3], in_place=False, require_planar_embedding=True)

    def test_add_nodes_and_edges(self):
        g = self.graph_sq.add_nodes_and_edges([0.25], [1], [0, 1, 3, 4], [12, 12, 12, 12])
        self.assertTrue(np.allclose(g.get_areas(), [0.25, 0.25, 0.25, 0.25, 1, 1, 1, 1, 1, -6]))
        g = EmbeddedSquareGraph(3, 4, 1, 2)
        g.add_nodes_and_edges([0.5], [1], [0, 1, 3, 4], [12, 12, 12, 12], in_place=True)
        self.assertTrue(np.allclose(g.get_areas(), [0.5, 0.5, 0.5, 0.5, 2, 2, 2, 2, 2, -12]))

    def test_remove_nodes(self):
        g = self.graph_sq.remove_nodes([1, 3])
        self.assertTrue(np.allclose(g.get_areas(), [1, 1, 1, -3]))
        self.assertEqual(g.get_num_components(), 2)
        with self.assertRaises(NotSingleComponentError):
            self.graph_sq.remove_nodes([1, 3], require_single_component=True)
        g = EmbeddedSquareGraph(3, 4, 1, 2)
        g.remove_nodes([1, 3], in_place=True)
        self.assertTrue(np.allclose(g.get_areas(), [2, 2, 2, -6]))

    def test_remove_edges(self):
        g = self.graph_sq.remove_edges([0, 4], [1, 3], require_planar_embedding=True, require_single_component=True)
        self.assertTrue(np.allclose(g.get_areas(), [1, 1, 1, 1, -4]))
        g = self.graph_sq.remove_edges(1, 4, require_planar_embedding=True, require_single_component=True)
        self.assertTrue(np.allclose(g.get_areas(), [1, 1, 1,1, 2, -6]))
        g = self.graph_sq.remove_edges([], [], require_planar_embedding=True, require_single_component=True)
        self.assertTrue(np.allclose(g.get_areas(), [1, 1, 1,1, 1, 1, -6]))
        with self.assertRaises(NotSingleComponentError):
            self.graph_sq.remove_edges([0, 4, 3], [1, 3, 0],  require_single_component=True)
        g = EmbeddedSquareGraph(3, 4, 1, 2)
        g.remove_edges([0, 4, 3], [1, 3, 0], in_place=True)
        self.assertTrue(np.all(g.node1 == [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 10]))
        self.assertTrue(np.all(g.node2 == [2, 4, 5, 6, 5, 7, 8, 7, 9, 8, 10, 11, 10, 11]))

    def test_remove_edges_by_ids(self):
        g = EmbeddedSquareGraph(3, 4, 1, 2)
        g.remove_edges_by_ids([0, 1, 5], in_place=True)
        self.assertTrue(np.all(g.node1 == [1, 1, 2,  3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 10]))
        self.assertTrue(np.all(g.node2 == [2, 4, 5,  6, 5, 7, 8, 7, 9, 8, 10, 11, 10, 11]))

    def test_permute_nodes(self):
        g = EmbeddedSquareGraph(3, 4, 1, 2)
        self.assertTrue(np.all(g.node1 == [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 10]))
        self.assertTrue(np.all(g.node2 == [1, 3, 2, 4, 5, 4, 6, 5, 7, 8, 7, 9, 8, 10, 11, 10, 11]))
        self.assertTrue(np.all(g._get_faces(include_boundary_faces=False)[0] == [0, 3, 22, 18, 2, 4, 24, 20, 5, 8, 27, 23, 7, 9,
                                                                                 29, 25, 10, 13, 32, 28, 12, 14, 33, 30]))
        self.assertTrue(np.all(g.get_face_nodes(include_boundary_faces=False)[0] == [0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, 6, 7, 10, 9, 7, 8, 11, 10]))
        g.permute_nodes([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertTrue(np.all(g.node1 == [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 10]))
        self.assertTrue(np.all(g.node2 == [1, 3, 2, 4, 5, 4, 6, 5, 7, 8, 7, 9, 8, 10, 11, 10, 11]))
        self.assertTrue(np.all(g.get_face_nodes(include_boundary_faces=False)[0] == [11, 10, 7, 8, 10, 9, 6, 7, 8, 7, 4, 5, 7, 6, 3, 4, 5, 4, 1, 2, 4, 3, 0, 1]))
        self.assertTrue(np.all(g._get_faces(include_boundary_faces=False)[0][:8] == [33, 30, 12, 14, 32, 28, 10, 13]))

    def test_permute_faces(self):
        g = EmbeddedSquareGraph(4, 4, 1, 1)
        g.remove_edges([1, 6, 10], [5, 7, 11], in_place=True)
        g.remove_nodes(9, in_place=True)
        g.permute_faces([1, 2, 3, 0])
        self.assertTrue(np.all(g.get_areas() == [3, 4, -9, 2]))
        self.assertTrue(np.all(g.get_face_nodes()[0] == [2, 3, 7, 10, 14, 13, 9, 6, 4, 5, 6, 9, 13, 12, 11, 8, 0, 4,
                                                         8, 11, 12, 13, 14, 10, 7, 3, 2, 1, 0, 1, 2, 6, 5, 4]))
        self.assertTrue(np.all(g.get_face_nodes()[1] == [8, 8, 12, 6]))
        self.assertTrue(np.all(g.cycle_space(include_boundary_faces=False).todense() ==
                               [[0, 0, 0, 1, -1, 1, 0, 0, 0, -1, 1, 0, -1, 1, 0, 0, -1],
                                [0, 0, 0, 0, 0, 0, 1, -1, 1, 1, 0, -1, 1, 0, -1, -1, 0],
                                [1, -1, 1, 0, 1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]]))

    def test_map(self):
        mapA = self.graph_A._get_edge_map()
        self.assertTrue(np.all(mapA == [2, 3, 4, 10, 6, 7, 15, 14, 1, 0, 8, 9, 5, 11, 12, 13]))
        mapB = self.graph_B._get_edge_map()
        self.assertTrue(np.all(mapB[array_B.inv] == array_B.inv[[1, 3, 5, 7, 13, 8, 9, 19, 16, 17, 2, 10, 0, 11, 6, 12, 4, 14, 15, 18]]))
        mapC = self.graph_C._get_edge_map()
        self.assertTrue(np.all(
            mapC[array_C.inv] == array_C.inv[[1, 6, 8, 5, 19, 21, 13, 23, 14, 11, 27, 12, 15, 31, 28, 29, 2, 3,
                                              0, 16, 7, 20, 17, 4, 18, 10, 9, 25, 26, 22, 24, 30]]))
        mapD = self.graph_D._get_edge_map()
        self.assertTrue(np.all(mapD == [1, 0]))
        mapE = self.graph_E._get_edge_map()
        self.assertTrue(np.all(mapE[array_E.inv] == array_E.inv[[2, 3, 9, 4, 7, 0, 1, 5, 6, 8]]))
        mapF = self.graph_F._get_edge_map()
        self.assertTrue(np.all(mapF == [3, 5, 11, 7, 8, 10, 1, 2, 0, 4, 6, 9]))
        mapG = self.graph_G._get_edge_map()
        self.assertTrue(np.all(mapG == [6, 10, 21, 8, 9, 12, 13, 15, 16, 19, 17, 2, 0, 1, 11, 3, 4, 5, 14, 7, 18, 20]))
        mapH = self.graph_H._get_edge_map()
        self.assertTrue(np.all(mapH == [2, 8, 7, 5, 11, 10, 1, 0, 6, 4, 3, 9]))
        mapI = self.graph_I._get_edge_map()
        self.assertTrue(np.all(mapI[array_I.inv] == array_I.inv[[4, 29, 3, 25, 26, 11, 9, 35, 31, 10, 32, 15, 16, 41,
                                                                 39, 37, 38, 43, 42, 20, 22, 23, 24, 47, 49, 1, 0, 5,
                                                                 27, 6, 2, 28, 7, 12, 13, 14, 30, 8, 33, 34, 17, 19, 36,
                                                                 40, 18, 21, 44, 45, 46, 48]]))
        mapJ = self.graph_J._get_edge_map()
        self.assertTrue(np.all(mapJ == [4, 8, 17, 18, 11, 12, 13, 9, 19, 16, 1, 3, 0, 2, 6, 10, 5, 14, 7, 15]))
        mapL = self.graph_L._get_edge_map()
        self.assertTrue(np.all(mapL[array_L.inv] == array_L.inv[[3, 5, 10, 9, 13, 11, 12, 2, 7, 0, 4, 1, 8, 6]]))

    def test_get_faces(self):
        face_edges, face_lengths = self.graph_A._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [4, 4, 8]))
        self.assertTrue(np.all(face_edges == [1, 3, 10, 8, 5, 7, 14, 12, 0, 2, 4, 6, 15, 13, 11, 9]))
        face_edges, face_lengths = self.graph_A._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == [4, 4]))
        self.assertTrue(np.all(face_edges == [1, 3, 10, 8, 5, 7, 14, 12]))
        face_edges, face_lengths = self.graph_B._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [4, 8, 8]))
        self.assertTrue(np.all(face_edges == [ 6,  9, 17, 15,  0,  2,  3,  7, 19, 18, 14, 11,  1,  4,  8, 16,  5, 13,  12, 10]))
        face_edges, face_lengths = self.graph_B._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == [4, 8]))
        self.assertTrue(np.all(face_edges == [6,  9, 17, 15,  1,  4,  8, 16,  5, 13,  12, 10]))
        face_edges, face_lengths = self.graph_C._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [3, 8, 21]))
        self.assertTrue(np.all(face_edges == array_C.inv[[10, 27, 25, 0, 1, 6, 13, 31, 30, 24, 18, 2, 8, 14, 28, 26,
                                                          9, 11, 12, 15, 29, 22, 17, 3, 5, 21, 20, 7, 23, 4, 19, 16]]))
        face_edges, face_lengths = self.graph_C._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == [3, 21]))
        self.assertTrue(np.all(face_edges == array_C.inv[[10, 27, 25, 2, 8, 14, 28, 26, 9, 11, 12, 15, 29,
                                                          22, 17, 3, 5, 21, 20, 7, 23, 4, 19, 16]]))
        face_edges, face_lengths = self.graph_D._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [2]))
        self.assertTrue(np.all(face_edges == [0, 1]))
        face_edges, face_lengths = self.graph_D._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == []))
        self.assertTrue(np.all(face_edges == []))
        face_edges, face_lengths = self.graph_E._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [10]))
        self.assertTrue(np.all(face_edges == [0, 3, 4, 7, 6, 1, 2, 9, 8, 5]))
        face_edges, face_lengths = self.graph_E._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == []))
        self.assertTrue(np.all(face_edges == []))
        face_edges, face_lengths = self.graph_F._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [4, 8]))
        self.assertTrue(np.all(face_edges == [1, 5, 10, 6, 0, 3, 7, 2, 11, 9, 4, 8]))
        face_edges, face_lengths = self.graph_F._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == [8]))
        self.assertTrue(np.all(face_edges == [0, 3, 7, 2, 11, 9, 4, 8]))
        face_edges, face_lengths = self.graph_G._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [6, 8, 8]))
        self.assertTrue(np.all(face_edges == [2, 21, 20, 18, 14, 11, 0, 6, 13, 1, 10, 17, 5, 12, 3, 8, 16, 4, 9, 19, 7, 15]))
        face_edges, face_lengths = self.graph_G._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == [8, 8]))
        self.assertTrue(np.all(face_edges == [0, 6, 13, 1, 10, 17, 5, 12, 3, 8, 16, 4, 9, 19, 7, 15]))
        face_edges, face_lengths = self.graph_H._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [3, 3, 3, 3]))
        self.assertTrue(np.all(face_edges == [0, 2, 7, 1, 8, 6, 3, 5, 10, 4, 11, 9]))
        face_edges, face_lengths = self.graph_H._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == [3, 3]))
        self.assertTrue(np.all(face_edges == [0, 2, 7, 3, 5, 10]))
        face_edges, face_lengths = self.graph_I._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [3, 3, 4, 4, 8, 28]))
        self.assertTrue(np.all(face_edges == array_I.inv[[0, 4, 26, 17, 43, 40, 12, 16, 38, 33, 21, 23, 47, 45, 5, 11, 15, 37,
                                              8, 31, 28, 27, 1, 29, 6, 9, 10, 32, 7, 35, 14, 39, 34, 13, 41, 19, 20,
                                              22, 24, 49, 48, 46, 44, 18, 42, 36, 30, 2, 3, 25]]))
        face_edges, face_lengths = self.graph_I._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == [3, 3, 4, 4, 8]))
        self.assertTrue(np.all(face_edges == array_I.inv[[0, 4, 26, 17, 43, 40, 12, 16, 38, 33, 21, 23, 47, 45, 5, 11, 15, 37, 8, 31, 28, 27]]))
        face_edges, face_lengths = self.graph_J._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [5, 5, 10]))
        self.assertTrue(np.all(face_edges == [1, 8, 19, 15, 10, 2, 17, 14, 6, 13, 0, 4, 11, 3, 18, 7, 9, 16, 5, 12]))
        face_edges, face_lengths = self.graph_J._get_faces(include_boundary_faces=False)
        self.assertTrue(np.all(face_lengths == [5, 10]))
        self.assertTrue(np.all(face_edges == [2, 17, 14, 6, 13, 0, 4, 11, 3, 18, 7, 9, 16, 5, 12]))
        face_edges, face_lengths = self.graph_K._get_faces(include_boundary_faces=True)
        self.assertTrue(np.all(face_lengths == [3, 3, 3, 3, 4, 6]))
        self.assertTrue(np.all(face_edges == array_K.inv[[0, 3, 13, 1, 6, 16, 4, 18, 14, 5, 19, 15, 7, 8, 10, 20, 2, 9, 21, 17, 12, 11]]))

    def test_get_face_nodes(self):
        face_nodes, face_lengths = self.graph_A.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [0, 2, 3, 1, 3, 5, 6, 4, 0, 1, 3, 4, 6, 5, 3, 2]))
        face_nodes, face_lengths = self.graph_A.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == [0, 2, 3, 1, 3, 5, 6, 4]))
        face_nodes, face_lengths = self.graph_B.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [4, 7, 8, 5, 0, 1, 2, 5, 8, 7, 6, 3, 0, 3, 6, 7, 4, 5, 2, 1]))
        face_nodes, face_lengths = self.graph_B.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == [4, 7, 8, 5, 0, 3, 6, 7, 4, 5, 2, 1]))
        face_nodes, face_lengths = self.graph_C.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [8, 11, 9, 0, 1, 2, 10, 14, 13, 12, 6, 0, 6, 12, 13, 11, 8, 9, 11, 13, 14, 10, 2, 1, 4, 5, 4, 3, 7, 3, 4, 1]))
        face_nodes, face_lengths = self.graph_C.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == [8, 11, 9, 0, 6, 12, 13, 11, 8, 9, 11, 13, 14, 10, 2, 1, 4, 5, 4, 3, 7, 3, 4, 1]))
        face_nodes, face_lengths = self.graph_D.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [0, 1]))
        face_nodes, face_lengths = self.graph_D.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == []))
        face_nodes, face_lengths = self.graph_E.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [0, 3, 4, 5, 2, 1, 2, 5, 4, 3]))
        face_nodes, face_lengths = self.graph_E.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == []))
        face_nodes, face_lengths = self.graph_F.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [0, 2, 3, 1, 0, 1, 2, 0, 3, 2, 1, 3]))
        face_nodes, face_lengths = self.graph_F.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == [0, 1, 2, 0, 3, 2, 1, 3]))
        face_nodes, face_lengths = self.graph_G.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [0, 5, 4, 3, 2, 1, 0, 1, 5, 0, 4, 5, 1, 4, 1, 2, 4, 1, 3, 4, 2, 3]))
        face_nodes, face_lengths = self.graph_G.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == [0, 1, 5, 0, 4, 5, 1, 4, 1, 2, 4, 1, 3, 4, 2, 3]))
        face_nodes, face_lengths = self.graph_H.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [0, 1, 5, 0, 5, 1, 2, 3, 4, 2, 4, 3]))
        face_nodes, face_lengths = self.graph_H.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == [0, 1, 5, 2, 3, 4]))
        face_nodes, face_lengths = self.graph_I.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [0, 3, 4, 11, 15, 12, 7, 12, 13, 8, 16, 18, 19, 17, 1, 6, 11, 12, 7, 8, 3,
                                              2, 0, 4, 3, 8, 9, 10, 5, 10, 9, 14, 9, 8, 13, 12, 16, 17, 19, 20, 19,
                                              18, 16, 12, 15, 11, 6, 1, 2, 3]))
        face_nodes, face_lengths = self.graph_I.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == [0, 3, 4, 11, 15, 12, 7, 12, 13, 8, 16, 18, 19, 17, 1, 6, 11, 12, 7, 8, 3, 2]))
        face_nodes, face_lengths = self.graph_J.get_face_nodes(include_boundary_faces=True)
        self.assertTrue(np.all(face_nodes == [0, 2, 4, 3, 1, 0, 3, 2, 1, 4, 0, 1, 2, 0, 4, 2, 3, 4, 1, 3]))
        face_nodes, face_lengths = self.graph_J.get_face_nodes(include_boundary_faces=False)
        self.assertTrue(np.all(face_nodes == [0, 3, 2, 1, 4, 0, 1, 2, 0, 4, 2, 3, 4, 1, 3]))

    def test_get_face_areas(self):
        face_areas = self.graph_A.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [1, 1, -2]))
        face_areas = self.graph_A.get_areas(include_boundary_faces=False)
        self.assertTrue(np.allclose(face_areas, [1, 1]))
        face_areas = self.graph_B.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [1, -4, 3]))
        face_areas = self.graph_B.get_areas(include_boundary_faces=False)
        self.assertTrue(np.allclose(face_areas, [1, 3]))
        face_areas = self.graph_C.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [0.5, -16, 15.5]))
        face_areas = self.graph_C.get_areas(include_boundary_faces=False)
        self.assertTrue(np.allclose(face_areas, [0.5, 15.5]))
        face_areas = self.graph_D.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [0]))
        face_areas = self.graph_D.get_areas(include_boundary_faces=False)
        self.assertTrue(np.allclose(face_areas, []))
        face_areas = self.graph_E.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [0]))
        face_areas = self.graph_E.get_areas(include_boundary_faces=False)
        self.assertTrue(np.allclose(face_areas, []))
        face_areas = self.graph_F.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [-1, 1]))
        face_areas = self.graph_G.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [-2, 1, 1]))
        face_areas = self.graph_H.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [0.5, -0.5, 0.5, -0.5]))
        face_areas = self.graph_H.get_areas(include_boundary_faces=False)
        self.assertTrue(np.allclose(face_areas, [0.5, 0.5]))
        face_areas = self.graph_I.get_areas(include_boundary_faces=True)
        self.assertTrue(np.allclose(face_areas, [0.5, 0.5, 1, 1, 3, -6]))


    def test_get_face_centroids(self):
        c_x, c_y = self.graph_A.get_centroids(include_boundary_faces=True)
        self.assertTrue(np.allclose(c_x, [0.5, 1.5, 1]))
        self.assertTrue(np.allclose(c_y, [1.5, 0.5, 1]))
        c_x, c_y = self.graph_A.get_centroids(include_boundary_faces=False)
        self.assertTrue(np.allclose(c_x, [0.5, 1.5]))
        self.assertTrue(np.allclose(c_y, [1.5, 0.5]))
        c_x, c_y = self.graph_B.get_centroids(include_boundary_faces=True)
        self.assertTrue(np.allclose(c_x, [1.5, 1, 5/6]))
        self.assertTrue(np.allclose(c_y, [0.5, 1, 7/6]))
        c_x, c_y = self.graph_B.get_centroids(include_boundary_faces=False)
        self.assertTrue(np.allclose(c_x, [1.5, 5/6]))
        self.assertTrue(np.allclose(c_y, [0.5, 7/6]))
        c_x, c_y = self.graph_C.get_centroids(include_boundary_faces=True)
        self.assertTrue(np.allclose(c_x, [7/3, 2, 185/93]))
        self.assertTrue(np.allclose(c_y, [5/3, 2, 187/93]))
        c_x, c_y = self.graph_C.get_centroids(include_boundary_faces=False)
        self.assertTrue(np.allclose(c_x, [7/3, 185/93]))
        self.assertTrue(np.allclose(c_y, [5/3, 187/93]))
        c_x, c_y = self.graph_F.get_centroids(include_boundary_faces=True)
        self.assertTrue(np.allclose(c_x, [0.5, 0.5]))
        self.assertTrue(np.allclose(c_y, [0.5, 0.5]))
        c_x, c_y = self.graph_G.get_centroids(include_boundary_faces=True)
        self.assertTrue(np.allclose(c_x, [1, 0.5, 1.5]))
        self.assertTrue(np.allclose(c_y, [0.5, 0.5, 0.5]))
        c_x, c_y = self.graph_H.get_centroids(include_boundary_faces=True)
        self.assertTrue(np.allclose(c_x, [1/3, 1/3, 5/3, 5/3]))
        self.assertTrue(np.allclose(c_y, [1/3, 1/3, 2/3, 2/3]))
        c_x, c_y = self.graph_I.get_centroids(include_boundary_faces=True)
        self.assertTrue(np.allclose(c_x, [8/3, 1/3, 3/2, 3/2, 5/6, 7/6]))
        self.assertTrue(np.allclose(c_y, [13/3, 5/3, 5/2, 1/2, 19/6, 31/12]))

    def test_get_num_components(self):
        self.assertEqual(self.graph_A.get_num_components(), 1)
        self.assertEqual(self.graph_B.get_num_components(), 1)
        self.assertEqual(self.graph_C.get_num_components(), 1)
        self.assertEqual(self.graph_D.get_num_components(), 1)
        self.assertEqual(self.graph_E.get_num_components(), 1)
        self.assertEqual(self.graph_F.get_num_components(), 1)
        self.assertEqual(self.graph_G.get_num_components(), 1)
        self.assertEqual(self.graph_H.get_num_components(), 2)
        self.assertEqual(self.graph_I.get_num_components(), 1)
        self.assertEqual(self.graph_J.get_num_components(), 1)
        self.assertEqual(self.graph_K.get_num_components(), 1)

    def test_is_planar_embedding(self):
        self.assertEqual(self.graph_A.is_planar_embedding(), True)
        self.assertEqual(self.graph_B.is_planar_embedding(), True)
        self.assertEqual(self.graph_C.is_planar_embedding(), True)
        self.assertEqual(self.graph_D.is_planar_embedding(), True)
        self.assertEqual(self.graph_E.is_planar_embedding(), True)
        self.assertEqual(self.graph_F.is_planar_embedding(), False)
        self.assertEqual(self.graph_G.is_planar_embedding(), False)
        self.assertEqual(self.graph_H.is_planar_embedding(), False)
        self.assertEqual(self.graph_I.is_planar_embedding(), True)
        self.assertEqual(self.graph_J.is_planar_embedding(), False)
        self.assertEqual(self.graph_K.is_planar_embedding(), True)

    def test_cut_space(self):
        A = self.graph_A.cut_space().todense()
        self.assertTrue(np.all(A == [[-1, -1, 0,  0,  0,  0,  0,  0],
                                     [1,  0, -1,  0,  0,  0,  0,  0],
                                     [0,  1,  0, -1,  0,  0,  0,  0],
                                     [0,  0,  1,  1, -1, -1,  0,  0],
                                     [0,  0,  0,  0,  1,  0, -1,  0],
                                     [0,  0,  0,  0,  0,  1,  0, -1],
                                     [0,  0,  0,  0,  0,  0,  1,  1]]))
        A = self.graph_B.cut_space().todense()
        self.assertTrue(np.all(A[:, array_B.rinv] == [[-1,  0, -1,  0,  0,  0,  0,  0,  0,  0],
                                                      [ 1, -1,  0,  0,  0,  0,  0,  0,  0,  0],
                                                      [ 0,  1,  0, -1,  0,  0,  0,  0,  0,  0],
                                                      [ 0,  0,  1,  0,  0, -1,  0,  0,  0,  0],
                                                      [ 0,  0,  0,  0, -1,  0, -1,  0,  0,  0],
                                                      [ 0,  0,  0,  1,  1,  0,  0, -1,  0,  0],
                                                      [ 0,  0,  0,  0,  0,  1,  0,  0, -1,  0],
                                                      [ 0,  0,  0,  0,  0,  0,  1,  0,  1, -1],
                                                      [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  1]]))

    def test_cycle_space(self):
        A = self.graph_A.cycle_space(include_boundary_faces=False).todense()
        self.assertTrue(np.all(A == [[-1, 1, -1, 1, 0, 0, 0, 0], [0, 0, 0, 0, -1, 1, -1, 1]]))
        A = self.graph_B.cycle_space(include_boundary_faces=False).todense()
        self.assertTrue(np.all(A[:, array_B.rinv] == [[0, 0, 0, 0, -1, 0, 1, -1, 0, 1], [-1, -1, 1, -1, 1, 1, -1, 0, 1, 0]]))
        A = self.graph_C.cycle_space(include_boundary_faces=False).todense()
        self.assertTrue(np.all(A[:, array_C.rinv] == [[0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0, 0, 0],
                                                      [-1, -1, 1, 0, 0, 0, -1, 0, 1, 1, -1, 1, 0, -1, 1, 1]]))
        A = self.graph_I.cycle_space(include_boundary_faces=True).todense()
        self.assertTrue(np.all(A[:, array_I.rinv] == [[ 1, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                                      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1, -1,  0,  0,  0,  0,  0,  0],
                                                      [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  1, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
                                                      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1, -1,  1,  0],
                                                      [ 0,  0, -1, -1,  0,  1, -1,  0,  1,  0,  0,  1, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                                      [-1,  1,  1,  1, -1, -1,  1,  0,  0,  0,  0, -1,  0,  1,  0,  0, -1, -1,  1,  0,  1, -1,  1, -1,  0]]))
        A = self.graph_K.cycle_space(include_boundary_faces=False).todense()
        self.assertTrue(np.all(A[:, array_K.rinv] == [[ 1,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0],
                                                      [ 0,  1,  0,  0,  0, -1,  1,  0,  0,  0,  0],
                                                      [ 0,  0,  0, -1,  1,  0,  0, -1,  0,  0,  0],
                                                      [ 0,  0,  0,  0, -1,  1,  0,  0, -1,  0,  0],
                                                      [ 0,  0,  0,  0,  0,  0,  0,  1,  1, -1,  1]]))
        A = self.graph_L.cycle_space(include_boundary_faces=True).todense()
        self.assertTrue(np.all(A[:, array_L.rinv] == [[ 1,  0, -1,  1,  0,  0,  0],
                                                      [ 0,  1,  0,  0, -1,  1,  0],
                                                      [-1, -1,  1, -1,  1, -1,  0]]))

    def test_adjacency_matrix(self):
        A = self.graph_A.adjacency_matrix().todense()
        self.assertTrue(np.all(A == [[0, 1, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 1, 0, 0, 0],
                                     [1, 0, 0, 1, 0, 0, 0],
                                     [0, 1, 1, 0, 1, 1, 0],
                                     [0, 0, 0, 1, 0, 0, 1],
                                     [0, 0, 0, 1, 0, 0, 1],
                                     [0, 0, 0, 0, 1, 1, 0]]))
        A = self.graph_B.adjacency_matrix().todense()
        self.assertTrue(np.all(A == [[0, 1, 0, 1, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 1, 0],
                                     [0, 0, 1, 0, 1, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 0, 1, 0, 1],
                                     [0, 0, 0, 0, 0, 1, 0, 1, 0]]))

    def test_get_common_edge(self):
        common_edges = self.graph_A.get_common_edge_of_faces([0, 1, 2], [1, 2, 0])
        self.assertTrue(np.all(common_edges == [-1, 4, 0]))
        common_edges = self.graph_A.get_common_edge_of_faces([[0, 1], [1, 2]], [[1, 0], [2, 1]])
        self.assertTrue(np.all(common_edges == [[-1, -1], [4, 4]]))
        common_edges = self.graph_B.get_common_edge_of_faces([0, 1, 2], [1, 2, 0])
        self.assertTrue(np.all(common_edges == array_B.inv[[7, 0, 4]]))
        common_edges = self.graph_I.get_common_edge_of_faces([0, 4, 1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 5, 0])
        self.assertTrue(np.all(common_edges == [-1, 15, -1, -1, -1, 2, 0]))

        _, orientation = self.graph_A.get_common_edge_of_faces([0, 1, 2], [1, 2, 0], return_orientation=True)
        self.assertTrue(np.all(orientation == [False, False, True]))
        _, orientation = self.graph_A.get_common_edge_of_faces([[0, 1], [1, 2]], [[1, 0], [2, 1]], return_orientation=True)
        self.assertTrue(np.all(orientation == [[False, False], [False, True]]))
        _, orientation = self.graph_B.get_common_edge_of_faces([0, 1, 2], [1, 2, 0], return_orientation=True)
        self.assertTrue(np.all(orientation == [False, True, True]))
        _, orientation = self.graph_I.get_common_edge_of_faces([0, 4, 1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 5, 0], return_orientation=True)
        self.assertTrue(np.all(orientation == [False, True, False, False, False, False, False]))
        common_edges, orientation= self.graph_K.get_common_edge_of_faces([[i] * 6 for i in range(6)], [[0, 1, 2, 3, 4, 5]] * 6, return_orientation=True)
        self.assertTrue(np.all(common_edges == [[-1, -1,  3, -1, -1,  0],
                                                [-1, -1, -1,  5, -1,  2],
                                                [ 3, -1, -1,  4,  7, -1],
                                                [-1,  5,  4, -1,  9, -1],
                                                [-1, -1,  7,  9, -1,  8],
                                                [ 0,  2, -1, -1,  8, -1]]))
        self.assertTrue(np.all(orientation == [[False, False,  True, False, False,  True],
                                               [False, False, False, False, False,  True],
                                               [False, False, False,  True, False, False],
                                               [False,  True, False, False, False, False],
                                               [False, False,  True,  True, False, False],
                                               [False, False, False, False,  True, False]]))

    def test_cycle_space_solve_for_integral_x(self):
        np.random.seed(0)
        def graph_tester(graph):
            s1 = np.random.randint(0, 100, graph.face_count(include_boundary_faces=False))
            s2 = np.random.randint(0, 100, (1000, graph.face_count(include_boundary_faces=False)))
            x = graph._cycle_space_solve_for_integral_x(s1)
            self.assertTrue(np.all(graph.cycle_space(include_boundary_faces=False) @ x.T == s1.T))
            x = graph._cycle_space_solve_for_integral_x(s2)
            self.assertTrue(np.all(graph.cycle_space(include_boundary_faces=False) @ x.T == s2.T))
        graph_tester(self.graph_A)
        graph_tester(self.graph_B)
        graph_tester(self.graph_C)
        graph_tester(self.graph_I)
        graph_tester(self.graph_K)
        graph_tester(self.graph_L)

if __name__ == "__main__":
    unittest.main()