"""The core module of pentapy."""

# pylint: disable=C0103, C0415, R0911, E0611

# === Imports ===

import warnings
from typing import Literal

import numpy as np

from pentapy import _models as pmodels
from pentapy import solver as psolver  # type: ignore
from pentapy import tools as ptools

# === Solver ===


def solve(
    mat: np.ndarray,
    rhs: np.ndarray,
    is_flat: bool = False,
    index_row_wise: bool = True,
    solver: Literal[
        1,
        "1",
        "PTRANS-I",
        "ptrans-i",
        2,
        "2",
        "PTRANS-II",
        "ptrans-ii",
        3,
        "3",
        "lapack",
        4,
        "4",
        "spsolve",
        5,
        "5",
        "spsolve_umf",
        "umf",
        "umf_pack",
    ] = 1,
) -> np.ndarray:
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
    mat : :class:`numpy.ndarray` of shape (m, m) or (5, m)
        The full or flattened version of the pentadiagonal matrix.
    rhs : :class:`numpy.ndarray` of shape (m,) or (m, n)
        The right hand side(s) of the equation system. Its shape is preserved.
    is_flat : :class:`bool`, default=False
        State if the matrix is already flattend. Default: ``False``
    index_row_wise : :class:`bool`, default=True
        State if the flattend matrix is row-wise flattend. Default: ``True``
    solver : :class:`int` or :class:`str`, default=1
        Which solver should be used. The following are provided:

            * ``[1, "1", "PTRANS-I"]`` : The PTRANS-I algorithm (default)
            * ``[2, "2", "PTRANS-II"]`` : The PTRANS-II algorithm
            * ``[3, "3", "lapack", "solve_banded"]`` : :func:`scipy.linalg.solve_banded`
            * ``[4, "4", "spsolve"]`` : :func:`scipy.sparse.linalg.spsolve(..., use_umfpack=False)`
            * ``[5, "5", "spsolve_umf", "umf", "umf_pack"]`` : :func:`scipy.sparse.linalg.spsolve(..., use_umfpack=False)`

        Strings are not case-sensitive.

    Returns
    -------
    result : :class:`numpy.ndarray` of shape (m,) or (m, n)
        Solution of the equation system with the same shape as ``rhs``.

    """

    # first, the solver is converted to the internal name to avoid confusion
    solver_inter = pmodels._SOLVER_ALIAS_CONVERSIONS[str(solver).lower()]

    if solver_inter in {
        pmodels.PentaSolverAliases.PTRANS_I,
        pmodels.PentaSolverAliases.PTRANS_II,
    }:
        if is_flat and index_row_wise:
            mat_flat = np.asarray(mat, dtype=np.double)
            ptools._check_penta(mat_flat)
        elif is_flat:
            mat_flat = np.array(mat, dtype=np.double)
            ptools._check_penta(mat_flat)
            ptools.shift_banded(mat_flat, copy=False)
        else:
            mat_flat = ptools.create_banded(mat, col_wise=False, dtype=np.double)

        rhs = np.asarray(rhs, dtype=np.double)

        # Special case: Early exit when the matrix has only 3 rows/columns
        # NOTE: this avoids memory leakage in the Cython-solver that will iterate over
        #       at least 4 rows/columns no matter what
        if mat_flat.shape[1] == 3:
            return np.linalg.solve(
                a=ptools.create_full(mat_flat, col_wise=False),
                b=rhs,
            )

        # if there is only a single right-hand side, it has to be reshaped to a 2D array
        # NOTE: this has to be reverted at the end
        single_rhs = rhs.ndim == 1
        rhs_og_shape = rhs.shape
        if single_rhs:
            rhs = rhs[:, np.newaxis]

        try:
            solver_func = (
                psolver.penta_solver1
                if solver_inter == pmodels.PentaSolverAliases.PTRANS_I
                else psolver.penta_solver2
            )

            # if there was only a 1D right-hand side, the result has to be flattened
            if single_rhs:
                return solver_func(mat_flat, rhs).ravel()

            return solver_func(mat_flat, rhs)

        except ZeroDivisionError:
            warnings.warn("pentapy: PTRANS-I not suitable for input-matrix.")
            return np.full(shape=rhs_og_shape, fill_value=np.nan)

    elif solver_inter == pmodels.PentaSolverAliases.LAPACK:  # pragma: no cover
        try:
            from scipy.linalg import solve_banded
        except ImportError as imp_err:  # pragma: no cover
            msg = "pentapy.solve: scipy.linalg.solve_banded could not be imported"
            raise ValueError(msg) from imp_err
        if is_flat and index_row_wise:
            mat_flat = np.array(mat)
            ptools._check_penta(mat_flat)
            ptools.shift_banded(mat_flat, col_to_row=False, copy=False)
        elif is_flat:
            mat_flat = np.asarray(mat)
        else:
            mat_flat = ptools.create_banded(mat)
        return solve_banded((2, 2), mat_flat, rhs)

    elif solver_inter == pmodels.PentaSolverAliases.SUPER_LU:  # pragma: no cover
        try:
            from scipy import sparse as sps
            from scipy.sparse.linalg import spsolve
        except ImportError as imp_err:
            msg = "pentapy.solve: scipy.sparse could not be imported"
            raise ValueError(msg) from imp_err
        if is_flat and index_row_wise:
            mat_flat = np.array(mat)
            ptools._check_penta(mat_flat)
            ptools.shift_banded(mat_flat, col_to_row=False, copy=False)
        elif is_flat:
            mat_flat = np.asarray(mat)
        else:
            mat_flat = ptools.create_banded(mat)
        size = mat_flat.shape[1]
        M = sps.spdiags(mat_flat, [2, 1, 0, -1, -2], size, size, format="csc")
        return spsolve(M, rhs, use_umfpack=False)

    elif solver_inter == pmodels.PentaSolverAliases.UMFPACK:  # pragma: no cover
        try:
            from scipy import sparse as sps
            from scipy.sparse.linalg import spsolve
        except ImportError as imp_err:
            msg = "pentapy.solve: scipy.sparse could not be imported"
            raise ValueError(msg) from imp_err
        if is_flat and index_row_wise:
            mat_flat = np.array(mat)
            ptools._check_penta(mat_flat)
            ptools.shift_banded(mat_flat, col_to_row=False, copy=False)
        elif is_flat:
            mat_flat = np.asarray(mat)
        else:
            mat_flat = ptools.create_banded(mat)
        size = mat_flat.shape[1]
        M = sps.spdiags(mat_flat, [2, 1, 0, -1, -2], size, size, format="csc")
        return spsolve(M, rhs, use_umfpack=True)

    else:  # pragma: no cover
        msg = f"pentapy.solve: unknown solver ({solver})"
        raise ValueError(msg)
