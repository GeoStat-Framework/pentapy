# -*- coding: utf-8 -*-
"""
2. Example.

Here we compare different algorithms for solving pentadiagonal systems.

To use this script you need to have the following packages installed:

    * scipy
    * scikit-umfpack
    * perfplot
    * matplotlib
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from scipy import sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from pentapy.py_solver import penta_solver1 as ps1
from pentapy.core import solve as ps2
from pentapy import tools
from scipy.linalg.lapack import dgbsv
import perfplot


def get_les(size):
    mat = (np.random.random((5, size)) - 0.5) * 1e-5
    V = np.array(np.random.random(size) * 1e5)
    return mat, V


def solve_1(in_val):
    mat, V = in_val
    size = mat.shape[1]
    M = sps.spdiags(mat, [2, 1, 0, -1, -2], size, size, format="csc")
    return spsolve(M, V, use_umfpack=False)


def solve_2(in_val):
    mat, V = in_val
    size = mat.shape[1]
    M = sps.spdiags(mat, [2, 1, 0, -1, -2], size, size, format="csc")
    return spsolve(M, V, use_umfpack=True)


def solve_3(in_val):
    mat, V = in_val
    M = tools.shift_banded(mat, col_to_row=True)
    return ps1(M, V)


def solve_4(in_val):
    mat, V = in_val
    return ps2(mat, V, is_flat=True, index_row_wise=False)


def solve_5(in_val):
    mat, V = in_val
    M = np.vstack((np.zeros((2, mat.shape[1])), mat))
    return dgbsv(2, 2, M, V)[2]


def solve_6(in_val):
    mat, V = in_val
    return solve_banded((2, 2), mat, V)


def solve_7(in_val):
    mat, V = in_val
    M = tools.create_full(mat)
    return np.linalg.solve(M, V)


perfplot.show(
    setup=get_les,
    kernels=[solve_1, solve_2, solve_3, solve_4, solve_5, solve_6, solve_7],
    labels=[
        "scipy sparse",
        "umfpack",
        "penta_py",
        "penta_cy",
        "lapack dgbsv",
        "scipy solve_banded",
        "numpy",
    ],
    n_range=[2 ** k for k in range(2, 13)],
    xlabel="Size [n]",
    logy=True,
)
