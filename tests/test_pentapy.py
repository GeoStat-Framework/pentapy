# -*- coding: utf-8 -*-
"""
This is the unittest for pentapy.
"""
from __future__ import division, absolute_import, print_function

# import platform
import warnings
import unittest
import numpy as np
import pentapy as pp


warnings.simplefilter("always")


class TestPentapy(unittest.TestCase):
    def setUp(self):
        self.seed = 19031977
        self.size = 1000
        self.rand = np.random.RandomState(self.seed)
        self.mat = (self.rand.rand(5, self.size) - 0.5) * 1e-5
        self.rhs = self.rand.rand(self.size) * 1e5

    def test_tools(self):
        self.mat_int = np.zeros((100, 100), dtype=int)
        # fill bands of pentadiagonal matrix
        self.mat_int[pp.diag_indices(100, 0)] = self.rand.randint(
            1, 1000, size=100
        )
        self.mat_int[pp.diag_indices(100, 1)] = self.rand.randint(
            1, 1000, size=99
        )
        self.mat_int[pp.diag_indices(100, 2)] = self.rand.randint(
            1, 1000, size=98
        )
        self.mat_int[pp.diag_indices(100, -1)] = self.rand.randint(
            1, 1000, size=99
        )
        self.mat_int[pp.diag_indices(100, -2)] = self.rand.randint(
            1, 1000, size=98
        )
        # create banded
        self.mat_int_col = pp.create_banded(self.mat_int)
        self.mat_int_row = pp.create_banded(self.mat_int, col_wise=False)
        # create full
        self.mat_int_col_ful = pp.create_full(self.mat_int_col, col_wise=True)
        self.mat_int_row_ful = pp.create_full(self.mat_int_row, col_wise=False)
        # shifting
        self.mat_shift_cr = pp.shift_banded(self.mat_int_col)
        self.mat_shift_rc = pp.shift_banded(self.mat_int_row, col_to_row=False)
        # in place shifting
        self.mat_int_col_ip = pp.create_banded(self.mat_int)
        self.mat_int_row_ip = pp.create_banded(self.mat_int, col_wise=False)
        pp.shift_banded(self.mat_int_col_ip, copy=False)
        pp.shift_banded(self.mat_int_row_ip, copy=False, col_to_row=False)
        # checking
        self.assertEqual(np.sum(self.mat_int > 0), 494)
        self.assertTrue(np.array_equal(self.mat_int_col, self.mat_shift_rc))
        self.assertTrue(np.array_equal(self.mat_int_row, self.mat_shift_cr))
        self.assertTrue(np.array_equal(self.mat_int_col, self.mat_int_row_ip))
        self.assertTrue(np.array_equal(self.mat_int_row, self.mat_int_col_ip))
        self.assertTrue(np.array_equal(self.mat_int, self.mat_int_col_ful))
        self.assertTrue(np.array_equal(self.mat_int, self.mat_int_row_ful))

    def test_solve1(self):
        self.mat_col = pp.shift_banded(self.mat, col_to_row=False)
        self.mat_ful = pp.create_full(self.mat, col_wise=False)

        sol_row = pp.solve(self.mat, self.rhs, is_flat=True, solver=1)
        sol_col = pp.solve(
            self.mat_col,
            self.rhs,
            is_flat=True,
            index_row_wise=False,
            solver=1,
        )
        sol_ful = pp.solve(self.mat_ful, self.rhs, solver=1)

        diff_row = np.max(np.abs(np.dot(self.mat_ful, sol_row) - self.rhs))
        diff_col = np.max(np.abs(np.dot(self.mat_ful, sol_col) - self.rhs))
        diff_ful = np.max(np.abs(np.dot(self.mat_ful, sol_ful) - self.rhs))

        diff_row_col = np.max(
            np.abs(self.mat_ful - pp.create_full(self.mat_col))
        )
        self.assertAlmostEqual(diff_row * 1e-5, 0.0)
        self.assertAlmostEqual(diff_col * 1e-5, 0.0)
        self.assertAlmostEqual(diff_ful * 1e-5, 0.0)
        self.assertAlmostEqual(diff_row_col * 1e5, 0.0)

    def test_solve2(self):
        self.mat_col = pp.shift_banded(self.mat, col_to_row=False)
        self.mat_ful = pp.create_full(self.mat, col_wise=False)

        sol_row = pp.solve(self.mat, self.rhs, is_flat=True, solver=2)
        sol_col = pp.solve(
            self.mat_col,
            self.rhs,
            is_flat=True,
            index_row_wise=False,
            solver=2,
        )
        sol_ful = pp.solve(self.mat_ful, self.rhs, solver=2)

        diff_row = np.max(np.abs(np.dot(self.mat_ful, sol_row) - self.rhs))
        diff_col = np.max(np.abs(np.dot(self.mat_ful, sol_col) - self.rhs))
        diff_ful = np.max(np.abs(np.dot(self.mat_ful, sol_ful) - self.rhs))

        diff_row_col = np.max(
            np.abs(self.mat_ful - pp.create_full(self.mat_col))
        )
        self.assertAlmostEqual(diff_row * 1e-5, 0.0)
        self.assertAlmostEqual(diff_col * 1e-5, 0.0)
        self.assertAlmostEqual(diff_ful * 1e-5, 0.0)
        self.assertAlmostEqual(diff_row_col * 1e5, 0.0)

    def test_error(self):
        self.err_mat = np.array(
            [[3, 2, 1, 0], [-3, -2, 7, 1], [3, 2, -1, 5], [0, 1, 2, 3]]
        )
        self.err_rhs = np.array([6, 3, 9, 6])
        sol_2 = pp.solve(self.err_mat, self.err_rhs, is_flat=False, solver=2)
        diff_2 = np.max(np.abs(np.dot(self.err_mat, sol_2) - self.err_rhs))
        self.assertAlmostEqual(diff_2, 0.0)


if __name__ == "__main__":
    unittest.main()
