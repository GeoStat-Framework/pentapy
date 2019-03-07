# -*- coding: utf-8 -*-
"""
The core module of pentapy

The following functions are provided

.. currentmodule:: pentapy.core

.. autosummary::
   solve
"""
from __future__ import division, absolute_import, print_function

import numpy as np

try:
    from pentapy.solver import penta_solver1
except ImportError: # pragma: no cover
    print("pentapy Warning: No Cython functions imported")
    from pentapy.py_solver import penta_solver1


def solve(mat, rhs, is_flat=False, index_row_wise=True, solver=1):
    """
    Solver for a pentadiagonal system

    The Matrix can be given as a full n x n Matrix or as a flattend one.
    The flattend matrix can be given in a row-wise flattend form::

      [[Dup2[0]  Dup2[1]  Dup2[2]  ... Dup2[N-2]  0          0       ]
       [Dup1[0]  Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  0       ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [0        Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] Dlow1[N]]
       [0        0        Dlow2[2] ... Dlow2[N-2] Dlow2[N-2] Dlow2[N]]]

    Or a column-wise flattend form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Dup1 and Dup2 or the first and second upper minor-diagonals and Dlow1 resp.
    Dlow2 are the lower ones.
    If you provide a column-wise flattend matrix, you have to set::

      index_row_wise=False


    Parameters
    ----------
    mat : :class:`numpy.ndarray`
        The Matrix or the flattened Version of the pentadiagonal matrix.
    rhs : :class:`numpy.ndarray`
        The right hand side of the equation system.
    is_flat : :class:`bool`, optional
        State if the matrix is already flattend. Default: ``False``
    index_row_wise : :class:`bool`, optional
        State if the flattend matrix is row-wise flattend. Default: ``True``
    solver : :class:`int`, optional
        Which solver should be used. Default: ``1``

    Returns
    -------
    result : :class:`numpy.ndarray`
        Result of the equation system
    """

    if is_flat and index_row_wise:
        mat_flat = np.array(mat, dtype=np.double)
    elif is_flat:
        mat_flat = np.array(mat, dtype=np.double)
        mat_flat[0, :-2] = mat_flat[0, 2:]
        mat_flat[1, :-1] = mat_flat[1, 1:]
        mat_flat[3, 1:] = mat_flat[3, :-1]
        mat_flat[4, 2:] = mat_flat[4, :-2]
    else:
        mat = np.asanyarray(mat)
        mat_flat = np.zeros((5, mat.shape[0]), dtype=np.double)
        mat_flat[0, :-2] = mat.diagonal(2)
        mat_flat[1, :-1] = mat.diagonal(1)
        mat_flat[2, :] = mat.diagonal(0)
        mat_flat[3, 1:] = mat.diagonal(-1)
        mat_flat[4, 2:] = mat.diagonal(-2)

    rhs = np.array(rhs, dtype=np.double)

    if solver==1:
        return penta_solver1(mat_flat, rhs)
    else:
        raise ValueError("pentapy.solve: unknown solver (" + str(solver) + ")")
