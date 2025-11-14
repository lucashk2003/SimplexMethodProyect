import unittest
import numpy as np
from services.simplex_methods import SimplexSimple, SimplexIntermediate, SimplexAdvanced

class TestSimplexMethods(unittest.TestCase):
    def test_simplex_simple_basic(self):
        # Problema: Max Z = 3x1 + 5x2, 2x1 + 3x2 <= 8, 2x1 + x2 <= 4
        c = np.array([3, 5], dtype=float)
        A = np.array([[2, 3], [2, 1]], dtype=float)
        b = np.array([8, 4], dtype=float)
        constraints_types = ['<=', '<=']
        is_max = True
        simplex = SimplexSimple()
        solution, z = simplex.solve(c, A, b, constraints_types, is_max)
        expected_solution = np.array([0, 8/3], dtype=float)
        expected_z = 40/3
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)
        self.assertAlmostEqual(z, expected_z, places=5)

    def test_simplex_intermediate_with_artificial(self):
        # Problema: Max Z = 2x1 + x2, x1 + x2 >= 2, x1 + 2x2 = 4
        c = np.array([2, 1], dtype=float)
        A = np.array([[1, 1], [1, 2]], dtype=float)
        b = np.array([2, 4], dtype=float)
        constraints_types = ['>=', '=']
        is_max = True
        simplex = SimplexIntermediate()
        solution, z = simplex.solve(c, A, b, constraints_types, is_max)
        expected_solution = np.array([4, 0], dtype=float)
        expected_z = 8.0
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)
        self.assertAlmostEqual(z, expected_z, places=5)

    def test_simplex_advanced_minimization(self):
        # Problema: Min Z = 3x1 + 2x2, x1 + x2 <= 4, x1 + 3x2 >= 3
        c = np.array([3, 2], dtype=float)
        A = np.array([[1, 1], [1, 3]], dtype=float)
        b = np.array([4, 3], dtype=float)
        constraints_types = ['<=', '>=']
        is_max = False
        simplex = SimplexAdvanced()
        solution, z = simplex.solve(c, A, b, constraints_types, is_max)
        expected_solution = np.array([0, 1], dtype=float)
        expected_z = 2.0
        np.testing.assert_array_almost_equal(solution, expected_solution, decimal=5)
        self.assertAlmostEqual(z, expected_z, places=5)

    def test_no_feasible_solution(self):
        # Problema sin solución: Max Z = x1 + x2, x1 + x2 >= 5, x1 + x2 <= 3
        c = np.array([1, 1], dtype=float)
        A = np.array([[1, 1], [1, 1]], dtype=float)
        b = np.array([5, 3], dtype=float)
        constraints_types = ['>=', '<=']
        is_max = True
        simplex = SimplexIntermediate()
        with self.assertRaises(ValueError):
            simplex.solve(c, A, b, constraints_types, is_max)

    def test_unbounded_problem(self):
        # Problema no acotado: Max Z = x1 + x2, -x1 - x2 <= -2
        c = np.array([1, 1], dtype=float)
        A = np.array([[-1, -1]], dtype=float)
        b = np.array([-2], dtype=float)
        constraints_types = ['<=']
        is_max = True
        simplex = SimplexSimple()
        with self.assertRaises(ValueError):
            simplex.solve(c, A, b, constraints_types, is_max)

if __name__ == "__main__":
    unittest.main()