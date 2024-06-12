"""
This test suite implements the test for the ``tools`` module of the ``pentapy`` package.

"""

# === Imports ===

import warnings
from typing import Optional, Tuple, Type

import numpy as np
import pytest
import util_funcs as uf

import pentapy as pp
from pentapy.tools import _check_penta

warnings.simplefilter("always")

# === Constants ===

SEED = 19_031_977
N_ROWS = [
    3,
    4,
    5,
    10,
    11,
    25,
    26,
    50,
    51,
    100,
    101,
    250,
    251,
    500,
    501,
    1_000,
    1_001,
    2500,
    2501,
    5_000,
    5_001,
    10_000,
    10_001,
]

# === Tests ===


@pytest.mark.parametrize("offset", [0, 1, 2, -1, -2])
@pytest.mark.parametrize("n_rows", N_ROWS)
def test_diag_indices(n_rows: int, offset: int) -> None:
    """
    Tests the generation of the diagonal indices via the function
    ``pentapy.diag_indices``.

    """

    # the diagonal indices are obtained with NumPy and pentapy
    row_idxs_ref, col_idxs_ref = uf.get_diag_indices(n=n_rows, offset=offset)
    row_idxs, col_idxs = pp.diag_indices(n=n_rows, offset=offset)

    # the diagonal indices are compared
    assert np.array_equal(row_idxs_ref, row_idxs)
    assert np.array_equal(col_idxs_ref, col_idxs)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("with_shift", [True, False])
@pytest.mark.parametrize("col_wise", [True, False])
@pytest.mark.parametrize("n_rows", N_ROWS)
def test_penta_generators(
    n_rows: int,
    col_wise: bool,
    with_shift: bool,
    copy: bool,
) -> None:
    """
    Tests the generation of pentadiagonal matrices where the matrix.

    """

    # a reference matrix is initialised
    mat_full_ref = uf.gen_rand_penta_matrix_dense_int(
        n_rows=n_rows,
        seed=SEED,
        with_pentapy_indices=False,
    )

    # then, it is turned into a banded matrix ...
    mat_banded = pp.create_banded(mat_full_ref, col_wise=col_wise)

    # ... which is maybe shifted
    # Case 1: copied shift
    if with_shift and copy:
        mat_banded = pp.shift_banded(mat_banded, col_to_row=col_wise, copy=True)
        col_wise = not col_wise

    # Case 2: in-place shift
    if with_shift and not copy:
        mat_banded = pp.shift_banded(mat_banded, col_to_row=col_wise, copy=False)
        col_wise = not col_wise

    # ... from which a full matrix is created again
    mat_full = pp.create_full(mat_banded, col_wise=col_wise)

    # the matrices are compared
    assert np.array_equal(mat_full_ref, mat_full)


@pytest.mark.parametrize(
    "shape, exception",
    [
        ((5, 5), None),  # Valid 2D Array with 5 rows and 5 rows
        ((5, 2), ValueError),  # 2D Array with 5 rows but only 2 columns
        ((2, 5), ValueError),  # 2D Array with 2 rows but 5 columns
        ((5,), ValueError),  # 1D Array
    ],
)
def test_create_banded_raises(
    shape: Tuple[int, ...],
    exception: Optional[Type[Exception]],
) -> None:
    """
    Test if the function ``pentapy.create_banded`` raises the expected exceptions.

    """

    # the test matrix is initialised
    np.random.seed(SEED)
    mat = np.random.rand(*shape)

    # Case 1: no exception should be raised
    if exception is None:
        pp.create_banded(mat)
        return

    # Case 2: an exception should be raised
    with pytest.raises(exception):
        pp.create_banded(mat)


@pytest.mark.parametrize(
    "shape, exception",
    [
        ((5, 5), None),  # Valid 2D Array with 5 bands and 5 columns
        ((5, 10), None),  # Valid 2D Array with 5 bands and 10 columns
        ((5, 3), None),  # 2D Array with 5 bands and the minimum number of columns
        ((6, 20), ValueError),  # 2D Array does not have 5 bands
        ((4, 30), ValueError),  # 2D Array does not have 5 bands
        ((5, 1), ValueError),  # 2D Array with 5 bands but too little columns
        ((5, 2), ValueError),  # 2D Array with 5 bands but too little columns
        ((5,), ValueError),  # 1D Array
    ],
)
def test_create_full_raises(
    shape: Tuple[int, ...],
    exception: Optional[Type[Exception]],
) -> None:
    """
    Test if the function ``pentapy.create_full`` raises the expected exceptions.

    """

    # the test matrix is initialised
    np.random.seed(SEED)
    mat = np.random.rand(*shape)

    # Case 1: no exception should be raised
    if exception is None:
        pp.create_full(mat)
        return

    # Case 2: an exception should be raised
    with pytest.raises(exception):
        pp.create_full(mat)


@pytest.mark.parametrize(
    "shape, exception",
    [
        ((5, 3), None),  # Valid 2D Array with 5 bands and 3 rows
        ((5, 2), ValueError),  # 2D Array with 5 bands but less than 3 rows
        ((4, 3), ValueError),  # 2D Array with less than 5 bands
        ((5,), ValueError),  # 1D Array
    ],
)
def test_check_penta(
    shape: Tuple[int, ...],
    exception: Optional[Type[Exception]],
) -> None:
    """
    Test if the function ``pentapy.tools._check_penta`` raises the expected exceptions.

    """

    # the test matrix is initialised
    np.random.seed(SEED)
    mat = np.random.rand(*shape)

    # Case 1: no exception should be raised
    if exception is None:
        _check_penta(mat)
        return

    # Case 2: an exception should be raised
    with pytest.raises(exception):
        _check_penta(mat)
