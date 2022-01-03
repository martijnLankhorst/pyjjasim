import unittest
import numpy as np
from josephson_circuit import *

from tests.examples import array_B, array_A, array_C, array_D, array_E
from variable_row_array import VarRowArray


class TestArray(unittest.TestCase):

    def setUp(self):
        self.sq_array_h = SquareArray(3, 4, current_direction='x')
        self.sq_array_v = SquareArray(3, 4, current_direction='y')
        self.hc_array_h = HoneycombArray(3, 4, current_direction='x')
        self.hc_array_v = HoneycombArray(3, 4, current_direction='y')
        self.tr_array_h = TriangularArray(3, 4, current_direction='x')
        self.tr_array_v = TriangularArray(3, 4, current_direction='y')
        self.custom1 = Circuit([0, 1, 1.5, 1, 0], [0, 0, 0, 1, 1], [0, 4, 1, 1, 4, 2],
                               [1, 0, 2, 3, 3, 3], external_current_basis=[1, 0, -2, 0, 1])

    def test_array_creation(self):
        with self.assertRaises(NotSingleComponentError):
            Circuit([0, 1, 1.5, 1, 0], [0, 0, 0, 1, 1], [0, 4, 1], [1, 0, 2])
        with self.assertRaises(NotSingleComponentError):
            Circuit([0, 1, 0, 3, 4, 3], [0, 0, 1, 0, 0, 1], [0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3])
        Circuit([0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 2, 3], [1, 2, 3, 0])
        with self.assertRaises(NotPlanarError):
            Circuit([0, 1, 1, 0, 2], [0, 0, 1, 1, 0], [0, 1, 2, 3, 0, 1, 1, 2], [1, 2, 3, 0, 2, 3, 4, 4])
        with self.assertRaises(NotPlanarEmbeddingError):
            Circuit([0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3])
        with self.assertRaises(NoCurrentConservationError):
            Circuit([0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 2, 3], [1, 2, 3, 0], external_current_basis=[1, 2, -1, -1])
        Circuit([0, 1], [0, 0], [0], [1], external_current_basis=[1, -1])

    def test_node_count(self):
        self.assertEqual(self.sq_array_h.node_count(), 12)
        self.assertEqual(self.hc_array_h.node_count(), 46)
        self.assertEqual(self.tr_array_h.node_count(), 24)
        self.assertEqual(self.custom1.node_count(), 5)

    def test_junction_count(self):
        self.assertEqual(self.sq_array_h.junction_count(), 3 * 3 + 2 * 4)
        self.assertEqual(self.hc_array_h.junction_count(), 6 * 12 - 4 - 6 - 2)
        self.assertEqual(self.tr_array_h.junction_count(), 6 * 12 - 16 - 6 + 1)
        self.assertEqual(self.custom1.junction_count(), 6)

    def test_face_count(self):
        self.assertEqual(self.sq_array_h.face_count(), 6)
        self.assertEqual(self.hc_array_h.face_count(), (2 * 3 - 1) * (4 - 1))
        self.assertEqual(self.tr_array_h.face_count(), (3 - 1) * (4 * (4 - 0.5)))
        self.assertEqual(self.custom1.face_count(), 2)

    def test_I_base(self):
        self.assertTrue(np.allclose(np.sum(self.sq_array_h.get_external_current_basis()), 0))
        self.assertTrue(np.allclose(np.sum(self.sq_array_v.get_external_current_basis()), 0))
        self.assertTrue(np.allclose(np.sum(self.hc_array_h.get_external_current_basis()), 0))
        self.assertTrue(np.allclose(np.sum(self.hc_array_v.get_external_current_basis()), 0))
        self.assertTrue(np.allclose(np.sum(self.tr_array_h.get_external_current_basis()), 0))
        self.assertTrue(np.allclose(np.sum(self.tr_array_v.get_external_current_basis()), 0))
        def get_in_out(array: Circuit):
            I_base = array.get_external_current_basis()
            return array.x[I_base > 0], array.y[I_base > 0], array.x[I_base < 0], array.y[I_base < 0], np.sum(I_base[I_base > 0])
        self.assertTrue(np.allclose(get_in_out(self.sq_array_h)[0], 0.0))
        self.assertTrue(np.allclose(get_in_out(self.sq_array_h)[2], 2.0))
        self.assertEqual(get_in_out(self.sq_array_h)[4], 4)
        self.assertTrue(np.allclose(get_in_out(self.sq_array_v)[1], 0.0))
        self.assertTrue(np.allclose(get_in_out(self.sq_array_v)[3], 3.0))
        self.assertEqual(get_in_out(self.sq_array_v)[4], 3)

    def test_add_nodes_and_junctions(self):
        with self.assertRaises(NotSingleComponentError):
            self.custom1.add_nodes_and_junctions([2], [2], [], [])
        with self.assertRaises(NodeNotExistError):
            self.custom1.add_nodes_and_junctions([], [], [0], [5])
        custom1 = self.custom1.add_nodes_and_junctions([2], [2], [2, 3], [5, 5])
        self.assertTrue(np.allclose(np.sort(custom1.areas), [0.25, 0.75, 1]))

    def test_remove_nodes(self):
        with self.assertRaises(NoCurrentConservationError):
            sq_array_h = self.sq_array_h.remove_nodes([0], rescale_external_current_basis=False)
        sq_array_h = self.sq_array_h.remove_nodes([0], rescale_external_current_basis=True)
        self.assertEqual(sq_array_h.node_count(), 11)
        self.assertEqual(sq_array_h.junction_count(), 15)
        self.assertEqual(sq_array_h.face_count(), 5)
        self.assertTrue(np.allclose(sq_array_h.get_external_current_basis(),
                                    [0, -1, 4.0/3, 0, -1, 4.0/3, 0, -1, 4.0/3, 0, -1]))
        with self.assertRaises(NoCurrentConservationError):
            sq_array_v = self.sq_array_v.remove_nodes([0], rescale_external_current_basis=False)
        sq_array_v = self.sq_array_v.remove_nodes([0], rescale_external_current_basis=True)
        self.assertEqual(sq_array_v.node_count(), 11)
        self.assertEqual(sq_array_v.junction_count(), 15)
        self.assertEqual(sq_array_v.face_count(), 5)
        self.assertTrue(np.allclose(sq_array_v.get_external_current_basis(),
                                    [1.5, 1.5, 0. , 0. , 0., 0., 0., 0., -1., -1., -1.]))
        sq_array_h = self.sq_array_h.remove_nodes([1,], rescale_external_current_basis=False)
        self.assertEqual(sq_array_h.node_count(), 11)
        self.assertEqual(sq_array_h.junction_count(), 14)
        self.assertEqual(sq_array_h.face_count(), 4)
        sq_array_h = self.sq_array_h.remove_nodes([1, 2, 4, 5], rescale_external_current_basis=True)
        self.assertEqual(sq_array_h.node_count(), 8)
        self.assertEqual(sq_array_h.junction_count(), 9)
        self.assertEqual(sq_array_h.face_count(), 2)
        with self.assertRaises(NotSingleComponentError):
            sq_array_h = self.sq_array_h.remove_nodes([1, 2, 3, 4, 5], rescale_external_current_basis=True)
        sq_array_h = self.sq_array_h.remove_nodes(0, rescale_external_current_basis=True)
        self.assertEqual(sq_array_h.node_count(), 11)
        self.assertEqual(sq_array_h.junction_count(), 15)
        self.assertEqual(sq_array_h.face_count(), 5)
        sq_array_h = self.sq_array_h.remove_nodes([True, False, False, False, False, False, False, False, False, False, False, False],
                                                  rescale_external_current_basis=True)
        self.assertEqual(sq_array_h.node_count(), 11)
        self.assertEqual(sq_array_h.junction_count(), 15)
        self.assertEqual(sq_array_h.face_count(), 5)

    def test_remove_junctions(self):
        sq_array_h = self.sq_array_h.remove_junctions([0])
        self.assertEqual(sq_array_h.node_count(), 12)
        self.assertEqual(sq_array_h.junction_count(), 16)
        self.assertEqual(sq_array_h.face_count(), 5)
        with self.assertRaises(NotSingleComponentError):
            sq_array_h = self.sq_array_h.remove_junctions([0, 8])
        sq_array_v = self.sq_array_v.remove_junctions([0, 3, 5, 15])
        self.assertEqual(sq_array_v.node_count(), 12)
        self.assertEqual(sq_array_v.junction_count(), 13)
        self.assertEqual(sq_array_v.face_count(), 2)
        sq_array_v = self.sq_array_v.remove_junctions([True, False, False, True, False, True, False, False, False,
                                                       False, False, False, False, False, False, True, False])
        self.assertEqual(sq_array_v.node_count(), 12)
        self.assertEqual(sq_array_v.junction_count(), 13)
        self.assertEqual(sq_array_v.face_count(), 2)
        sq_array_v = self.sq_array_v.remove_junctions(1)
        self.assertEqual(sq_array_v.node_count(), 12)
        self.assertEqual(sq_array_v.junction_count(), 16)
        self.assertEqual(sq_array_v.face_count(), 5)

    def test_get_node_coordinates(self):
        self.assertTrue(np.allclose(self.sq_array_h.get_node_coordinates()[0],
                                    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        self.assertTrue(np.allclose(self.sq_array_h.get_node_coordinates()[1],
                                    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))

    def test_critical_current_factors(self):
        sq_array_h = self.sq_array_h.copy()
        sq_array_h.set_critical_current_factors(2)
        self.assertEqual(sq_array_h.get_critical_current_factors(), 2)
        sq_array_h.set_critical_current_factors([2])
        self.assertEqual(sq_array_h.get_critical_current_factors(), 2)
        sq_array_h.set_critical_current_factors(np.arange(17))
        self.assertTrue(np.allclose(sq_array_h.get_critical_current_factors(), np.arange(17)))

    def test_resistance_factors(self):
        sq_array_h = self.sq_array_h.copy()
        sq_array_h.set_resistance_factors(2)
        self.assertEqual(sq_array_h.get_resistance_factors(), 2)
        sq_array_h.set_resistance_factors([2])
        self.assertEqual(sq_array_h.get_resistance_factors(), 2)
        with self.assertRaises(ValueError):
            sq_array_h.set_resistance_factors(0.0)
        with self.assertRaises(ValueError):
            sq_array_h.set_resistance_factors(-1.0)
        sq_array_h.set_resistance_factors(np.arange(1, 18))
        self.assertTrue(np.allclose(sq_array_h.get_resistance_factors(), np.arange(1, 18)))
        self.assertTrue(np.allclose(sq_array_h.R(), np.arange(1, 18)))

    def test_capacitance_factors(self):
        sq_array_h = self.sq_array_h.copy()
        sq_array_h.set_capacitance_factors(2)
        self.assertEqual(sq_array_h.get_capacitance_factors(), 2)
        sq_array_h.set_capacitance_factors([2])
        self.assertEqual(sq_array_h.get_capacitance_factors(), 2)
        sq_array_h.set_capacitance_factors(0.0)
        self.assertEqual(sq_array_h.get_capacitance_factors(), 0.0)
        with self.assertRaises(ValueError):
            sq_array_h.set_capacitance_factors(-1.0)
        sq_array_h.set_capacitance_factors(np.arange(1, 18))
        self.assertTrue(np.allclose(sq_array_h.get_capacitance_factors(), np.arange(1, 18)))

    def test_C(self):
        sq_array_h = self.sq_array_h.copy()
        sq_array_h.set_capacitance_factors(2)
        sq_array_h.set_beta_C(3)
        self.assertEqual(sq_array_h.get_beta_C(), 3)
        self.assertEqual(sq_array_h.C(), 6)
        sq_array_h.set_capacitance_factors(np.arange(1, 18))
        self.assertTrue(np.allclose(sq_array_h.C(), 3 * np.arange(1, 18)))

    def test_get_faces(self):
        # incomplete
        self.assertEqual(len(self.sq_array_h.get_faces()), self.sq_array_h.face_count())
        self.assertEqual(len(self.tr_array_h.get_faces()), self.tr_array_h.face_count())
        self.assertEqual(len(self.hc_array_v.get_faces()), self.hc_array_v.face_count())
        self.assertEqual(len(self.custom1.get_faces()), self.custom1.face_count())

    def test_face_areas(self):
        pass

    def test_face_centroids(self):
        pass

    def test_face_common_junctions(self):
        pass

    def test_locate_faces(self):
        pass

    def test_approximate_inductance(self):
        pass

    # def test_junction_inductance_matrix(self):
    #     pass
    #
    # def test_beta_C(self):
    #     # get_beta_C
    #     # set_beta_C
    #     # has_capacitance
    #     pass
    #
    # def test_beta_L(self):
    #     # get_beta_L
    #     # set_beta_L
    #     # has_inductance
    #     pass
    #
    # def test_cut_matrix(self):
    #     pass
    #
    # def test_cycle_matrix(self):
    #     pass
    #
    # def test_reduced_cut_matrix(self):
    #     pass
    #
    # def test_permute_nodes(self):
    #     pass
    #
    # def test_permute_nodes_lexsort(self):
    #     pass
    #
    # def test_all_same_junction_self_L(self):
    #     pass
    #
    # def test_junction_hash(self):
    #     pass
    #
    # def test_Ibase(self):
    #     pass
    #
    # def test_is_orthogonal(self):
    #     pass
    #
    # def test_is_single_component(self):
    #     pass
    #
    # def test_circuit_rank(self):
    #     pass
    #
    # def test_IbaseJ(self):
    #     pass
    #
    # def test_Ibase_total(self):
    #     pass
    #
    # def test_compress_np_array_internal(self):
    #     pass
    #
    # def test_assert_junction_id_exists_internal(self):
    #     pass
    #
    # def test_append_quantities_internal(self):
    #     pass
    #
    # def test_prepare_junction_quantity_internal(self):
    #     pass
    #
    # def test_junction_centers_internal(self):
    #     pass
    #
    # def test_junction_lengths_internal(self):
    #     pass
    #
    # def test_junction_inner_internal(self):
    #     pass
    #
    # def test_get_path_junctions_internal(self):
    #     pass
    #
    # def test_reorder_paths_internal(self):
    #     pass
    #
    # def test_assign_path_map_internal(self):
    #     pass
    #

    def test_get_face_areas_and_centroids_internal(self):
        x1, y1 = [0, 1, 1],  [0, 0, 1]
        n1, cnt1 = [0, 1, 2], [3]
        areas1 = Circuit._get_face_areas_2(VarRowArray(cnt1), n1, x1, y1)
        self.assertTrue(np.allclose(areas1, [0.5]))
        cx1, cy1 = Circuit._get_face_centroids_2(VarRowArray(cnt1), n1, x1, y1)
        self.assertTrue(np.allclose(cx1, [2/3]))
        self.assertTrue(np.allclose(cy1, [1/3]))
        x2, y2 = [0, 1, 2, 0, 1],  [0, 0, 0, 1, 1]
        n2, cnt2 = [0, 1, 4, 3, 1, 2, 4], [4, 3]
        areas2 = Circuit._get_face_areas_2(VarRowArray(cnt2), n2, x2, y2)
        self.assertTrue(np.allclose(areas2, [1.0, 0.5]))
        cx2, cy2 = Circuit._get_face_centroids_2(VarRowArray(cnt2), n2, x2, y2)
        self.assertTrue(np.allclose(cx2, [0.5, 4/3]))
        self.assertTrue(np.allclose(cy2, [0.5, 1/3]))
        x3, y3 = [0, 1, 2, 1, 2, 0, 2], [0, 0, 0, 1, 1, 2, 2]
        n3, cnt3 = [0, 1, 3, 4, 6, 5, 1, 2, 4, 3, 1, 3, 4, 2], [6, 4, 4]
        areas3 = Circuit._get_face_areas_2(VarRowArray(cnt3), n3, x3, y3)
        self.assertTrue(np.allclose(areas3, [3.0, 1.0, -1.0]))
        cx3, cy3 = Circuit._get_face_centroids_2(VarRowArray(cnt3), n3, x3, y3)
        self.assertTrue(np.allclose(cx3, [5/6, 1.5, 1.5]))
        self.assertTrue(np.allclose(cy3, [7/6, 0.5, 0.5]))

    def test_make_cycle_structure_internal(self):
        mapA = Circuit._make_cycle_structure2(array_A.x, array_A.y, array_A.n1, array_A.n2)
        self.assertTrue(np.all(mapA == [2, 3, 4, 10, 6, 7, 15, 14, 1, 0, 8, 9, 5, 11, 12, 13]))
        mapB = Circuit._make_cycle_structure2(array_B.x, array_B.y, array_B.n1, array_B.n2)
        self.assertTrue(np.all(mapB == [1, 3, 5, 7, 13, 8, 9, 19, 16, 17, 2, 10, 0, 11, 6, 12, 4, 14, 15, 18]))
        mapC = Circuit._make_cycle_structure2(array_C.x, array_C.y, array_C.n1, array_C.n2)
        self.assertTrue(np.all(mapC == [1, 6, 8, 5, 19, 21, 13, 23, 14, 11, 27, 12, 15, 31, 28, 29, 2, 3, 0, 16, 7, 20, 17, 4, 18, 10, 9, 25, 26, 22, 24, 30]))
        mapD = Circuit._make_cycle_structure2(array_D.x, array_D.y, array_D.n1, array_D.n2)
        self.assertTrue(np.all(mapD == [1, 0]))
        mapE = Circuit._make_cycle_structure2(array_E.x, array_E.y, array_E.n1, array_E.n2)
        self.assertTrue(np.all(mapE == [2, 3, 9, 4, 7, 0, 1, 5, 6, 8]))

    def test_retrace_paths_internal(self):
        paths1 = np.array([[1, 4, 6, 8, 8, 4, 3, 5, 7, 3]])
        retraced_paths1, retraced_path_lengths1 = Circuit._retrace_paths(paths1)
        self.assertTrue(np.all(retraced_paths1 == [1, 4, 3]))
        self.assertTrue(np.all(retraced_path_lengths1 == [3]))
        paths2 = np.array([[1, 4, 6, 8, 8, 4, 3, 5, 7, 3], [2, 3, 9, 5, 4, 3, 7, 10, 7, 8]])
        retraced_paths2, retraced_path_lengths2 = Circuit._retrace_paths(paths2)
        self.assertTrue(np.all(retraced_paths2 == [1, 4, 3, 2, 3, 7, 8]))
        self.assertTrue(np.all(retraced_path_lengths2 == [3, 4]))
        paths3 = np.array([[4, 6, 4, 5]])
        retraced_paths3, retraced_path_lengths3 = Circuit._retrace_paths(paths3)
        self.assertTrue(np.all(retraced_paths3 == [4, 5]))
        self.assertTrue(np.all(retraced_path_lengths3 == [2]))

    def test_store_terminated_paths_internal(self):
        p, pl = [], []
        pl, p = Circuit._store_terminated_paths([[2, 4, 5], [7, 6, 5], [4, 5, 2]], pl, p)
        self.assertTrue(np.all(pl == [3, 3]))
        self.assertTrue(np.all(p == [2, 4, 5, 5, 7, 6]))
        pl, p = Circuit._store_terminated_paths([[6, 5, 7, 8], [1, 3, 2, 4], [5, 4, 6, 4], [7, 8, 6, 5]], pl, p)
        self.assertTrue(np.all(pl == [3, 3, 4, 4, 2]))
        self.assertTrue(np.all(p == [2, 4, 5, 5, 7, 6, 1, 3, 2, 4, 5, 7, 8, 6, 4, 5]))
        pl, p = Circuit._store_terminated_paths([[6, 5, 7, 8, 1, 3, 2, 4], [5, 4, 6, 4, 7, 8, 6, 5]], pl, p)
        self.assertTrue(np.all(pl == [3, 3, 4, 4, 2]))
        self.assertTrue(np.all(p == [2, 4, 5, 5, 7, 6, 1, 3, 2, 4, 5, 7, 8, 6, 4, 5, 1, 3, 2, 4, 6, 5, 7, 8]))

    def test_construct_faces_internal(self):
        pass


    #
    # def check_planarity_internal(self):
    #     pass
    #
    # def test_junction_remove_mask_internal(self):
    #     pass
    #
    # def test_apply_matrix_format_internal(self):
    #     pass
    #
    # def test_is_sparse_diag_internal(self):
    #     pass
    #
    # def test_is_diagonal_internal(self):
    #     pass
    #
    # def test_is_sparse_symmetric_internal(self):
    #     pass
    #
    # def test_is_symmetric_internal(self):
    #     pass


if __name__ == "__main__":
    unittest.main()
