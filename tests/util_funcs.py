"""
This test suite implements the utility functions for testing the ``pentapy`` package.

"""

### Imports ###

from functools import partial
from typing import Tuple

import numpy as np
import pentapy as pp

### Utility Functions ###


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


### Doctests ###

if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
