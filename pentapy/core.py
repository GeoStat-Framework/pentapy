# -*- coding: utf-8 -*-
"""
The core module of pentapy.

The following functions are provided

.. currentmodule:: pentapy.core

.. autosummary::
   solve
"""
import warnings
import numpy as np
from pentapy.tools import shift_banded, create_banded, _check_penta
from pentapy.solver import penta_solver1, penta_solver2


def solve(mat, rhs, is_flat=False, index_row_wise=True, solver=1):
    """
    Solver for a pentadiagonal system.

    The matrix can be given as a full n x n matrix or as a flattend one.
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

    Dup1 and Dup2 are the first and second upper minor-diagonals
    and Dlow1 resp. Dlow2 are the lower ones.
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
    solver : :class:`int` or :class:`str`, optional
        Which solver should be used. The following are provided:

            * ``[1, "1", "PTRANS-I"]`` : The PTRANS-I algorithm
            * ``[2, "2", "PTRANS-II"]`` : The PTRANS-II algorithm
            * ``[3, "3", "lapack", "solve_banded"]`` :
              scipy.linalg.solve_banded
            * ``[4, "4", "spsolve"]`` :
              The scipy sparse solver without umf_pack
            * ``[5, "5", "spsolve_umf", "umf", "umf_pack"]`` :
              The scipy sparse solver with umf_pack

        Default: ``1``

    Returns
    -------
    result : :class:`numpy.ndarray`
        Solution of the equation system
    """

    if solver in [1, "1", "PTRANS-I"]:
        if is_flat and index_row_wise:
            mat_flat = np.array(mat, dtype=np.double)
            _check_penta(mat_flat)
        elif is_flat:
            mat_flat = np.array(mat, dtype=np.double)
            _check_penta(mat_flat)
            shift_banded(mat_flat, copy=False)
        else:
            mat_flat = create_banded(mat, col_wise=False, dtype=np.double)
        rhs = np.array(rhs, dtype=np.double)
        try:
            return penta_solver1(mat_flat, rhs)
        except ZeroDivisionError:
            warnings.warn("pentapy: PTRANS-I not suitable for input-matrix.")
            return np.full_like(rhs, np.nan)
    elif solver in [2, "2", "PTRANS-II"]:
        if is_flat and index_row_wise:
            mat_flat = np.array(mat, dtype=np.double)
            _check_penta(mat_flat)
        elif is_flat:
            mat_flat = np.array(mat, dtype=np.double)
            _check_penta(mat_flat)
            shift_banded(mat_flat, copy=False)
        else:
            mat_flat = create_banded(mat, col_wise=False, dtype=np.double)
        rhs = np.array(rhs, dtype=np.double)
        try:
            return penta_solver2(mat_flat, rhs)
        except ZeroDivisionError:
            warnings.warn("pentapy: PTRANS-II not suitable for input-matrix.")
            return np.full_like(rhs, np.nan)
    elif solver in [3, "3", "lapack", "solve_banded"]:  # pragma: no cover
        try:
            from scipy.linalg import solve_banded
        except ImportError:  # pragma: no cover
            raise ValueError(
                "pentapy.solve: "
                + "scipy.linalg.solve_banded could not be imported"
            )
        if is_flat and index_row_wise:
            mat_flat = np.array(mat)
            _check_penta(mat_flat)
            shift_banded(mat_flat, col_to_row=False, copy=False)
        elif is_flat:
            mat_flat = np.array(mat)
        else:
            mat_flat = create_banded(mat)
        return solve_banded((2, 2), mat_flat, rhs)
    elif solver in [4, "4", "spsolve"]:  # pragma: no cover
        try:
            from scipy import sparse as sps
            from scipy.sparse.linalg import spsolve
        except ImportError:
            raise ValueError(
                "pentapy.solve: scipy.sparse could not be imported"
            )
        if is_flat and index_row_wise:
            mat_flat = np.array(mat)
            _check_penta(mat_flat)
            shift_banded(mat_flat, col_to_row=False, copy=False)
        elif is_flat:
            mat_flat = np.array(mat)
        else:
            mat_flat = create_banded(mat)
        size = mat_flat.shape[1]
        M = sps.spdiags(mat_flat, [2, 1, 0, -1, -2], size, size, format="csc")
        return spsolve(M, rhs, use_umfpack=False)
    elif solver in [
        5,
        "5",
        "spsolve_umf",
        "umf",
        "umf_pack",
    ]:  # pragma: no cover
        try:
            from scipy import sparse as sps
            from scipy.sparse.linalg import spsolve
        except ImportError:
            raise ValueError(
                "pentapy.solve: scipy.sparse could not be imported"
            )
        if is_flat and index_row_wise:
            mat_flat = np.array(mat)
            _check_penta(mat_flat)
            shift_banded(mat_flat, col_to_row=False, copy=False)
        elif is_flat:
            mat_flat = np.array(mat)
        else:
            mat_flat = create_banded(mat)
        size = mat_flat.shape[1]
        M = sps.spdiags(mat_flat, [2, 1, 0, -1, -2], size, size, format="csc")
        return spsolve(M, rhs, use_umfpack=True)
    else:  # pragma: no cover
        raise ValueError("pentapy.solve: unknown solver (" + str(solver) + ")")
