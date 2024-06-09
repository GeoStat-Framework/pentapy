"""The core module of pentapy."""

# pylint: disable=C0103, C0415, R0911, E0611

# === Imports ===

import warnings
from typing import Literal

import numpy as np
import psutil

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
    workers: int = 1,
) -> np.ndarray:
    """
    Solver for a pentadiagonal system.

    The matrix can be given as a full (n x n) matrix or as a flattened one.
    The flattened matrix can be given in a row-wise flattened form::

      [[Dup2[0]  Dup2[1]  Dup2[2]  ... Dup2[N-2]  0          0       ]
       [Dup1[0]  Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  0       ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [0        Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] Dlow1[N]]
       [0        0        Dlow2[2] ... Dlow2[N-2] Dlow2[N-2] Dlow2[N]]]

    Or a column-wise flattened form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Dup1 and Dup2 are the first and second upper minor-diagonals
    and Dlow1 resp. Dlow2 are the lower ones.
    If you provide a column-wise flattened matrix, you have to set::

      index_row_wise=False


    Parameters
    ----------
    mat : :class:`numpy.ndarray` of shape (m, m) or (5, m)
        The full or flattened version of the pentadiagonal matrix.
    rhs : :class:`numpy.ndarray` of shape (m,) or (m, n)
        The right hand side(s) of the equation system. Its shape determines the shape
        of the output as they will be identical.
    is_flat : :class:`bool`, optional
        State if the matrix is already flattened. Default: ``False``
    index_row_wise : :class:`bool`, optional
        State if the flattened matrix is row-wise flattened. Default: ``True``
    solver : :class:`int` or :class:`str`, optional
        Which solver should be used. The following are provided:

            * ``[1, "1", "PTRANS-I"]`` : The PTRANS-I algorithm (default)
            * ``[2, "2", "PTRANS-II"]`` : The PTRANS-II algorithm
            * ``[3, "3", "lapack", "solve_banded"]`` : :func:`scipy.linalg.solve_banded`
            * ``[4, "4", "spsolve"]`` : :func:`scipy.sparse.linalg.spsolve(..., use_umfpack=False)`
            * ``[5, "5", "spsolve_umf", "umf", "umf_pack"]`` : :func:`scipy.sparse.linalg.spsolve(..., use_umfpack=False)`

        Strings are not case-sensitive.
    workers : :class:`int`, optional
        Number of workers used in the PTRANS-I and PTRANS-II solvers for parallel
        processing of multiple right-hand sides. Parallelisation overhead can be
        significant for small systems. If set to ``-1``, the number of workers is
        automatically determined. Default: ``1``

    Returns
    -------
    result : :class:`numpy.ndarray` of shape (m,) or (m, n)
        Solution of the equation system with the same shape as ``rhs``.

    """

    # first, the solver is converted to the internal name to avoid confusion
    solver_inter = pmodels._SOLVER_ALIAS_CONVERSIONS[str(solver).lower()]

    # Case 1: the pentapy solvers
    if solver_inter in {
        pmodels.PentaSolverAliases.PTRANS_I,
        pmodels.PentaSolverAliases.PTRANS_II,
    }:
        # the matrix is checked and shifted if necessary ...
        if is_flat and index_row_wise:
            mat_flat = np.asarray(mat, dtype=np.double)
            ptools._check_penta(mat_flat)
        elif is_flat:
            mat_flat = np.array(mat, dtype=np.double)  # NOTE: this is a copy
            ptools._check_penta(mat_flat)
            ptools.shift_banded(mat_flat, copy=False)
        else:
            mat_flat = ptools.create_banded(mat, col_wise=False, dtype=np.double)

        # ... followed by the conversion of the right-hand side
        rhs = np.asarray(rhs, dtype=np.double)

        # Special case: Early exit when the matrix has only 3 rows/columns
        # NOTE: this avoids memory leakage in the Cython-solver that will iterate over
        #       at least 4 rows/columns no matter what
        if mat_flat.shape[1] == 3:
            return np.linalg.solve(
                a=ptools.create_full(mat_flat, col_wise=False),
                b=rhs,
            )

        # now, the number of workers for multithreading has to be determined if
        # necessary
        # NOTE: the following does not count the number of total threads, but the number
        #       of threads available for the solver
        if workers < -1:
            raise ValueError(
                f"pentapy.solve: workers has to be -1 or greater, not {workers=}"
            )

        if workers == -1:
            proc = psutil.Process()
            workers = len(proc.cpu_affinity())  # type: ignore
            del proc

        elif workers == 0:
            workers = 1

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
            sol = solver_func(
                np.ascontiguousarray(mat_flat),
                np.ascontiguousarray(rhs),
                workers,
            )
            if single_rhs:
                sol = sol.ravel()

            return sol

        except ZeroDivisionError:
            warnings.warn("pentapy: PTRANS-I not suitable for input-matrix.")
            return np.full(shape=rhs_og_shape, fill_value=np.nan)

    # Case 2: LAPACK's banded solver
    elif solver_inter == pmodels.PentaSolverAliases.LAPACK:
        try:
            from scipy.linalg import solve_banded
        except ImportError as imp_err:  # pragma: no cover
            msg = "pentapy.solve: scipy.linalg.solve_banded could not be imported"
            raise ValueError(msg) from imp_err

        if is_flat and index_row_wise:
            mat_flat = np.array(mat)  # NOTE: this is a copy
            ptools._check_penta(mat_flat)
            ptools.shift_banded(mat_flat, col_to_row=False, copy=False)
        elif is_flat:
            mat_flat = np.asarray(mat)
        else:
            mat_flat = ptools.create_banded(mat)

        # NOTE: since this is a general banded solver, the number of sub- and super-
        #       diagonals has to be provided
        # NOTE: LAPACK handles all the reshaping and flattening internally
        try:
            return solve_banded(
                l_and_u=(2, 2),
                ab=mat_flat,
                b=rhs,
            )

        except np.linalg.LinAlgError:
            warnings.warn("pentapy: LAPACK solver encountered singular matrix.")
            return np.full(shape=rhs.shape, fill_value=np.nan)

    # Case 3: SciPy's sparse solver with or without UMFPACK
    elif solver_inter in {
        pmodels.PentaSolverAliases.SUPER_LU,
        pmodels.PentaSolverAliases.UMFPACK,
    }:
        try:
            from scipy import sparse as sps
            from scipy.sparse.linalg import spsolve
        except ImportError as imp_err:  # pragma: no cover
            msg = "pentapy.solve: scipy.sparse could not be imported"
            raise ValueError(msg) from imp_err

        if is_flat and index_row_wise:
            mat_flat = np.array(mat)  # NOTE: this is a copy
            ptools._check_penta(mat_flat)
            ptools.shift_banded(mat_flat, col_to_row=False, copy=False)
        elif is_flat:
            mat_flat = np.asarray(mat)
        else:
            mat_flat = ptools.create_banded(mat)

        # the solvers require a sparse left-hand side matrix, so this is created here
        # NOTE: the UMFPACK solver will not be triggered for multiple right-hand sides
        use_umfpack = solver_inter == pmodels.PentaSolverAliases.UMFPACK
        size = mat_flat.shape[1]
        M = sps.spdiags(
            data=mat_flat,
            diags=[2, 1, 0, -1, -2],
            m=size,
            n=size,
            format="csc",
        )

        sol = spsolve(
            A=M,
            b=rhs,
            use_umfpack=use_umfpack,
        )

        # NOTE: spsolve flattens column-vectors, thus their shape has to be restored
        # NOTE: it already fills the result vector with NaNs if the matrix is singular
        if rhs.ndim == 2 and 1 in rhs.shape:
            sol = sol[::, np.newaxis]

        return sol

    else:  # pragma: no cover
        msg = f"pentapy.solve: unknown solver ({solver})"
        raise ValueError(msg)
