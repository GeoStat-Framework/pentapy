"""
This test suite implements the utility functions for testing the ``pentapy`` package.

"""

# === Imports ===

from functools import partial
from typing import Tuple

import numpy as np
from scipy import linalg as spla
from scipy import sparse as sprs

import pentapy as pp

# === Constants ===

_MIN_DIAG_VAL = 1e-3


# === Utility Functions ===


def get_diag_indices(
    n: int,
    offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the row and column indices of the diagonal of a matrix ``mat``.

    This answer is based on the Stack Overflow answer that can be found at:
    https://stackoverflow.com/a/18081653/14814813

    Doctests
    --------
    >>> # Setting up a test matrix
    >>> n_rows = 5
    >>> mat = np.arange(start=0, stop=n_rows * n_rows).reshape(n_rows, n_rows)

    >>> # Getting the main diagonal indices
    >>> row_idxs, col_idxs = get_diag_indices(n=n_rows, offset=0)
    >>> row_idxs
    array([0, 1, 2, 3, 4])
    >>> col_idxs
    array([0, 1, 2, 3, 4])
    >>> mat[row_idxs, col_idxs]
    array([ 0,  6, 12, 18, 24])

    >>> # Getting the first upper diagonal indices
    >>> row_idxs, col_idxs = get_diag_indices(n=n_rows, offset=1)
    >>> row_idxs
    array([0, 1, 2, 3])
    >>> col_idxs
    array([1, 2, 3, 4])
    >>> mat[row_idxs, col_idxs]
    array([ 1,  7, 13, 19])

    >>> # Getting the second upper diagonal indices
    >>> row_idxs, col_idxs = get_diag_indices(n=n_rows, offset=2)
    >>> row_idxs
    array([0, 1, 2])
    >>> col_idxs
    array([2, 3, 4])
    >>> mat[row_idxs, col_idxs]
    array([ 2,  8, 14])

    >>> # Getting the first lower diagonal indices
    >>> row_idxs, col_idxs = get_diag_indices(n=n_rows, offset=-1)
    >>> row_idxs
    array([1, 2, 3, 4])
    >>> col_idxs
    array([0, 1, 2, 3])
    >>> mat[row_idxs, col_idxs]
    array([ 5, 11, 17, 23])

    >>> # Getting the second lower diagonal indices
    >>> row_idxs, col_idxs = get_diag_indices(n=n_rows, offset=-2)
    >>> row_idxs
    array([2, 3, 4])
    >>> col_idxs
    array([0, 1, 2])
    >>> mat[row_idxs, col_idxs]
    array([10, 16, 22])

    """

    row_idxs, col_idxs = np.diag_indices(n=n, ndim=2)
    if offset < 0:
        row_idx_from = -offset
        row_idx_to = None
        col_idx_from = 0
        col_idx_to = offset
    elif offset > 0:
        row_idx_from = 0
        row_idx_to = -offset
        col_idx_from = offset
        col_idx_to = None
    else:
        row_idx_from = None
        row_idx_to = None
        col_idx_from = None
        col_idx_to = None

    return (
        row_idxs[row_idx_from:row_idx_to],
        col_idxs[col_idx_from:col_idx_to],
    )


def gen_rand_penta_matrix_dense_int(
    n_rows: int,
    seed: int,
    with_pentapy_indices: bool,
) -> np.ndarray:
    """
    Generates a random dense pentadiagonal matrix with shape ``(n_rows, n_rows)`` and
    data type ``int64``.

    Doctests
    --------
    >>> # Generating a random pentadiagonal matrix with NumPy indices
    >>> n_rows = 5
    >>> seed = 19_031_977
    >>> with_pentapy_indices = False

    >>> mat_no_pentapy = gen_rand_penta_matrix_dense_int(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     with_pentapy_indices=with_pentapy_indices
    ... )
    >>> mat_no_pentapy
    array([[117, 499,  43,   0,   0],
           [378, 149, 857, 353,   0],
           [285, 769, 767, 229, 484],
           [  0, 717, 214, 243, 877],
           [  0,   0, 410, 611,  79]], dtype=int64)

    >>> # Generating a random pentadiagonal matrix with pentapy indices
    >>> mat_with_pentapy = gen_rand_penta_matrix_dense_int(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     with_pentapy_indices=True
    ... )
    >>> mat_with_pentapy
    array([[117, 499,  43,   0,   0],
           [378, 149, 857, 353,   0],
           [285, 769, 767, 229, 484],
           [  0, 717, 214, 243, 877],
           [  0,   0, 410, 611,  79]], dtype=int64)

    >>> # Checking if the two matrices are equal
    >>> np.array_equal(mat_no_pentapy, mat_with_pentapy)
    True

    """

    # first, a matrix of zeros is initialised ...
    mat = np.zeros((n_rows, n_rows), dtype=np.int64)
    # ... together with a partially specified random vector generator
    # NOTE: this ensures consistent random numbers for both cases
    gen_rand_int = partial(np.random.randint, low=1, high=1_000)

    # then, the diagonal index function is obtained
    diag_idx_func = get_diag_indices
    if with_pentapy_indices:
        diag_idx_func = pp.diag_indices

    # then, the diagonals are filled with random integers
    np.random.seed(seed=seed)
    for offset in range(-2, 3):
        row_idxs, col_idxs = diag_idx_func(n=n_rows, offset=offset)
        mat[row_idxs, col_idxs] = gen_rand_int(size=n_rows - abs(offset))

    return mat


def gen_conditioned_rand_penta_matrix_dense(
    n_rows: int,
    seed: int,
    ill_conditioned: bool,
) -> np.ndarray:
    """
    Generates a well- or ill-conditioned random banded pentadiagonal matrix with shape
    ``(n_rows, n_rows)``.

    This is achieved as follows:
    - a fake LDU decomposition is generated where ``L`` and ``U`` are unit lower and
      upper triangular matrices, respectively, and ``D`` is a diagonal matrix
    - the matrix is then reconstructed by multiplying the three matrices and converting
      the result to a banded matrix

    If ``D`` does not have any zeros or values of small magnitude compared to the
    largest value, the matrix should be well-conditioned.
    Otherwise, it is ill-conditioned.

    Doctests
    --------
    >>> # 1) Generating a super small well-conditioned random pentadiagonal matrix
    >>> n_rows = 3
    >>> seed = 19_031_977

    >>> mat = gen_conditioned_rand_penta_matrix_dense(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     ill_conditioned=False,
    ... )
    >>> mat
    array([[ 0.92453713,  0.28308514, -0.09972199],
           [-0.09784268,  0.2270634 , -0.1509019 ],
           [-0.23431267,  0.00468463,  0.22991003]])
    >>> # it has to be square and its bandwidth is computed and should be equal to 2
    >>> mat.shape[0] == mat.shape[1]
    True
    >>> spla.bandwidth(mat)
    (2, 2)
    >>> # its condition number is computed and values below 1e10 can be considered good
    >>> np.linalg.cond(mat)
    4.976880305142543

    >>> # 2) Generating a super small ill-conditioned random pentadiagonal matrix
    >>> mat = gen_conditioned_rand_penta_matrix_dense(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     ill_conditioned=True,
    ... )
    >>> mat
    array([[ 0.92453713,  0.28308514, -0.09972199],
           [-0.09784268,  0.2270634 , -0.1509019 ],
           [-0.23431267,  0.00468463, -0.02273771]])
    >>> # it has to be square and its bandwidth is computed and should be equal to 2
    >>> mat.shape[0] == mat.shape[1]
    True
    >>> spla.bandwidth(mat)
    (2, 2)
    >>> # its condition number is computed and its value should be close to the
    >>> # reciprocal floating point precision, i.e., ~1e16
    >>> np.linalg.cond(mat)
    1.493156437173682e+17

    >>> # 3) Generating a small well-conditioned random pentadiagonal matrix
    >>> n_rows = 7

    >>> mat = gen_conditioned_rand_penta_matrix_dense(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     ill_conditioned=False,
    ... )
    >>> np.round(mat, 2)
    array([[ 0.92, -0.72,  0.73,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.83, -0.02,  1.08,  0.41,  0.  ,  0.  ,  0.  ],
           [-0.58,  0.13, -0.13, -0.37,  0.18,  0.  ,  0.  ],
           [ 0.  , -0.07, -0.58,  0.46, -0.31,  0.28,  0.  ],
           [ 0.  ,  0.  ,  0.43,  0.13,  0.39, -0.1 , -0.15],
           [ 0.  ,  0.  ,  0.  ,  0.06, -0.14,  0.4 ,  0.28],
           [ 0.  ,  0.  ,  0.  ,  0.  , -0.14,  0.36,  0.53]])
    >>> # it has to be square and its bandwidth is computed and should be equal to 2
    >>> mat.shape[0] == mat.shape[1]
    True
    >>> spla.bandwidth(mat)
    (2, 2)
    >>> # its condition number is computed and values below 1e10 can be considered good
    >>> np.linalg.cond(mat)
    42.4847446467131

    >>> # 4) Generating a small ill-conditioned random pentadiagonal matrix
    >>> mat = gen_conditioned_rand_penta_matrix_dense(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     ill_conditioned=True,
    ... )
    >>> np.round(mat, 2)
    array([[ 0.92, -0.72,  0.73,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.83, -0.02,  1.08,  0.41,  0.  ,  0.  ,  0.  ],
           [-0.58,  0.13, -0.13, -0.37,  0.18,  0.  ,  0.  ],
           [ 0.  , -0.07, -0.58,  0.46, -0.31,  0.28,  0.  ],
           [ 0.  ,  0.  ,  0.43,  0.13,  0.39, -0.1 , -0.15],
           [ 0.  ,  0.  ,  0.  ,  0.06, -0.14,  0.4 ,  0.28],
           [ 0.  ,  0.  ,  0.  ,  0.  , -0.14,  0.36,  0.28]])
    >>> # it has to be square and its bandwidth is computed and should be equal to 2
    >>> mat.shape[0] == mat.shape[1]
    True
    >>> spla.bandwidth(mat)
    (2, 2)
    >>> # its condition number is computed and its value should be close to the
    >>> # reciprocal floating point precision, i.e., ~1e16
    >>> np.linalg.cond(mat)
    1.1079218802103074e+17

    >>> # 5) Generating a large well-conditioned random pentadiagonal matrix
    >>> n_rows = 1_000

    >>> mat = gen_conditioned_rand_penta_matrix_dense(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     ill_conditioned=False,
    ... )
    >>> # it has to be square and its bandwidth is computed and should be equal to 2
    >>> mat.shape[0] == mat.shape[1]
    True
    >>> spla.bandwidth(mat)
    (2, 2)
    >>> # its condition number is computed and values below 1e10 can be considered good
    >>> np.linalg.cond(mat)
    9570.995402466417

    >>> # 6) Generating a large ill-conditioned random pentadiagonal matrix
    >>> mat = gen_conditioned_rand_penta_matrix_dense(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     ill_conditioned=True,
    ... )
    >>> # it has to be square and its bandwidth is computed and should be equal to 2
    >>> mat.shape[0] == mat.shape[1]
    True
    >>> spla.bandwidth(mat)
    (2, 2)
    >>> # its condition number is computed and its value should be close to the
    >>> # reciprocal floating point precision, i.e., ~1e16
    >>> np.linalg.cond(mat)
    1.7137059583101745e+19

    """

    # first, the fake diagonal matrix is generated whose entries are strictly
    # positive and sorted in descending order
    np.random.seed(seed=seed)
    d_diag = np.flip(np.sort(np.random.rand(n_rows)))

    # the conditioning is achieved by manipulating the smallest diagonal entry
    # Case 1: well-conditioned matrix
    if not ill_conditioned:
        # here, the smallest diagonal entry is set to a value that is enforced to have
        # a minimum magnitude
        d_diag = np.maximum(d_diag, _MIN_DIAG_VAL)

    # Case 2: ill-conditioned matrix
    else:
        # here, the smallest diagonal entry is set to zero
        d_diag[n_rows - 1] = 0.0

    # ... followed by a unit lower triangular matrix with 2 sub-diagonals, but here
    # the entries may be negative ...
    diagonals = [
        1.0 - 2.0 * np.random.rand(n_rows - 2),
        1.0 - 2.0 * np.random.rand(n_rows - 1),
        np.ones(n_rows),
    ]
    l_mat = sprs.diags(
        diagonals=diagonals,
        offsets=[-2, -1, 0],  # type: ignore
        shape=(n_rows, n_rows),
        format="csr",
        dtype=np.float64,
    )

    # ... and an upper triangular matrix with 2 super-diagonals
    diagonals = [
        np.ones(n_rows),
        1.0 - 2.0 * np.random.rand(n_rows - 1),
        1.0 - 2.0 * np.random.rand(n_rows - 2),
    ]
    u_mat = sprs.diags(
        diagonals=diagonals,
        offsets=[0, 1, 2],  # type: ignore
        shape=(n_rows, n_rows),
        format="csr",
        dtype=np.float64,
    )

    # finally, the matrix is reconstructed by multiplying the three matrices
    return (l_mat.multiply(d_diag[np.newaxis, ::]).dot(u_mat)).toarray()


def solve_penta_matrix_dense_scipy(
    mat: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solves a pentadiagonal matrix system using SciPy's banded solver.

    Doctests
    --------
    >>> # Setting up a small test matrix and right-hand side
    >>> n_rows = 5
    >>> seed = 19_031_977

    >>> mat = gen_conditioned_rand_penta_matrix_dense(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     ill_conditioned=False,
    ... )
    >>> rhs = np.random.rand(n_rows, 5)

    >>> # Solving the system using SciPy's banded solver
    >>> sol = solve_penta_matrix_dense_scipy(mat=mat, rhs=rhs)
    >>> np.round(sol, 2)
    array([[-2.16, -0.36,  0.72,  0.23, -0.2 ],
           [ 4.07,  1.3 ,  0.81,  1.31,  0.48],
           [ 4.05,  0.33,  2.19,  1.22,  0.58],
           [-1.9 , -0.79,  1.02, -0.39,  1.02],
           [ 6.31,  1.81,  1.29,  1.41,  0.37]])

    >>> # the solution is checked by verifying that the residual is close to zero
    >>> np.max(np.abs(mat @ sol - rhs)) <= np.finfo(np.float64).eps * n_rows
    True

    >>> # Setting up a large test matrix and right-hand side
    >>> n_rows = 1_000

    >>> mat = gen_conditioned_rand_penta_matrix_dense(
    ...     n_rows=n_rows,
    ...     seed=seed,
    ...     ill_conditioned=False,
    ... )
    >>> rhs = np.random.rand(n_rows, 5)

    >>> # Solving the system using SciPy's banded solver
    >>> sol = solve_penta_matrix_dense_scipy(mat=mat, rhs=rhs)
    >>> # the solution is checked by verifying that the residual is close to zero
    >>> np.max(np.abs(mat @ sol - rhs)) <= np.finfo(np.float64).eps * n_rows
    True

    """

    # first, the matrix is converted to LAPACK banded storage format
    mat_banded = pp.create_banded(mat=mat, col_wise=True)

    # then, the system is solved using SciPy's banded solver
    return spla.solve_banded(
        l_and_u=(2, 2),
        ab=mat_banded,
        b=rhs,
    )


# === Doctests ===

if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
