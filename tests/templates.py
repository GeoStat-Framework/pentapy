"""
This test suite implements reusable templates for testing the pentadiagonal solver based
on either Algorithm PTRANS-I or PTRANS-II.
"""

# === Imports ===

from typing import Dict, Literal

import numpy as np
import pentapy as pp
import pytest
import util_funcs as uf

# === Constants ===

SEED = 19_031_977
SINGULAR_WARNING_REF_CONTENT = "solver encountered singular matrix"
SHAPE_MISMATCH_ERROR_REF_CONTENT = "shape mismatch between the number of equations"
N_ROWS = [
    3,  # important edge case
    4,  # important edge case
    5,  # important edge case
    10,  # even
    11,  # odd
    50,  # even
    51,  # odd
    100,  # ...
    101,
    1_000,
    1_001,
]
SOLVER_ALIASES_PTRANS_I = [1, "1", "pTrAnS-I"]
SOLVER_ALIASES_PTRANS_II = [2, "2", "pTrAnS-Ii"]

PARAM_DICT = {
    "n_rows": N_ROWS,
    "n_rhs": [None, 1, 10],
    "input_layout": ["full", "banded_row_wise", "banded_col_wise"],
    "solver_alias": SOLVER_ALIASES_PTRANS_I + SOLVER_ALIASES_PTRANS_II,
    "induce_error": [False, True],
    "from_order": ["C", "F"],
    "num_threads": [1],
}

# === Auxiliary functions ===


def convert_matrix_to_layout(
    mat: np.ndarray,
    input_layout: Literal["full", "banded_row_wise", "banded_col_wise"],
) -> tuple[np.ndarray, Dict[str, bool]]:
    """
    Converts a dense pentadiagonal matrix to the desired layout.

    """

    if input_layout == "full":
        return (
            mat,
            dict(is_flat=False),
        )

    elif input_layout == "banded_row_wise":
        return (
            pp.create_banded(mat, col_wise=False),
            dict(
                is_flat=True,
                index_row_wise=True,
            ),
        )

    elif input_layout == "banded_col_wise":
        return (
            pp.create_banded(mat, col_wise=True),
            dict(
                is_flat=True,
                index_row_wise=False,
            ),
        )

    else:
        raise ValueError(f"Invalid input layout: {input_layout}")


def convert_matrix_to_order(
    mat: np.ndarray,
    from_order: Literal["C", "F"],
) -> np.ndarray:
    """
    Converts a dense pentadiagonal matrix to the desired order.

    """

    if from_order == "C":
        return np.ascontiguousarray(mat)

    elif from_order == "F":
        return np.asfortranarray(mat)

    else:
        raise ValueError(f"Invalid from order: {from_order=}")


# === Templates ===


def pentapy_solvers_extended_template(
    n_rows: int,
    n_rhs: int,
    input_layout: Literal["full", "banded_row_wise", "banded_col_wise"],
    solver_alias: Literal[
        1,
        "1",
        "PTRANS-I",
        "pTrAnS-I",
        2,
        "2",
        "PTRANS-II",
        "pTrAnS-Ii",
    ],
    induce_error: bool,
    from_order: Literal["C", "F"],
    num_threads: int,
) -> None:
    """
    Tests the pentadiagonal solvers when starting from different input layouts, number
    of right-hand sides, number of rows, and also when inducing an error by making the
    first or last diagonal element exactly zero.
    It has to be ensured that the edge case of ``n_rows = 3`` is also covered.

    For ``n_rows = 3``, the error is induced by initialising a matrix of zeros.

    """

    # first, a random pentadiagonal matrix is generated
    mat_full = uf.gen_conditioned_rand_penta_matrix_dense(
        n_rows=n_rows,
        seed=SEED,
        ill_conditioned=False,
    )

    # an error is induced by setting the first or last diagonal element to zero
    if induce_error:
        # the induction of the error is only possible if the matrix does not have
        # only 3 rows
        if n_rows == 3:
            mat_full = np.zeros_like(mat_full)

        elif solver_alias in SOLVER_ALIASES_PTRANS_I:
            mat_full[0, 0] = 0.0
        else:
            mat_full[n_rows - 1, n_rows - 1] = 0.0

    # the right-hand side is generated
    np.random.seed(SEED)
    if n_rhs is not None:
        rhs = np.random.rand(n_rows, n_rhs)
        result_shape = (n_rows, n_rhs)
    else:
        rhs = np.random.rand(n_rows)
        result_shape = (n_rows,)

    # the matrix is converted to the desired layout
    mat, kwargs = convert_matrix_to_layout(mat_full, input_layout)

    # the left-hand side matrix and right-hand side is converted to the desired order
    mat = convert_matrix_to_order(mat=mat, from_order=from_order)
    rhs = convert_matrix_to_order(mat=rhs, from_order=from_order)

    # the solution is computed
    # Case 1: in case of an error, a warning has to be issued and the result has to
    # be NaN
    if induce_error:
        with pytest.warns(UserWarning, match=SINGULAR_WARNING_REF_CONTENT):
            mat_ref_copy = mat.copy()
            sol = pp.solve(
                mat=mat,
                rhs=rhs,
                solver=solver_alias,  # type: ignore
                num_threads=num_threads,
                **kwargs,
            )

        assert sol.shape == result_shape
        assert np.isnan(sol).all()
        assert np.array_equal(mat, mat_ref_copy)

        return

    # Case 2: in case of no error, the solution can be computed without any issues
    mat_ref_copy = mat.copy()
    sol = pp.solve(
        mat=mat,
        rhs=rhs,
        solver=solver_alias,  # type: ignore
        num_threads=num_threads,
        **kwargs,
    )
    assert sol.shape == result_shape
    assert np.array_equal(mat, mat_ref_copy)

    # if no error was induced, the reference solution is computed with SciPy
    sol_ref = uf.solve_penta_matrix_dense_scipy(
        mat=mat_full,
        rhs=rhs,
    )

    # the solutions are compared
    assert np.allclose(sol, sol_ref)

    return


def pentapy_solvers_shape_mismatch_template(
    n_rows: int,
    n_rhs: int,
    input_layout: Literal["full", "banded_row_wise", "banded_col_wise"],
    solver_alias: Literal[
        1,
        "1",
        "PTRANS-I",
        "pTrAnS-I",
        2,
        "2",
        "PTRANS-II",
        "pTrAnS-Ii",
    ],
    from_order: Literal["C", "F"],
    num_threads: int,
) -> None:
    """
    Tests the pentadiagonal solvers when the shape of the right-hand side is incorrect,
    starting from different input layouts, number of right-hand sides, and number of
    rows.

    """

    # first, a random pentadiagonal matrix is generated
    mat_full = uf.gen_conditioned_rand_penta_matrix_dense(
        n_rows=n_rows,
        seed=SEED,
        ill_conditioned=False,
    )

    # the right-hand side is generated with a wrong shape (rows + 10)
    np.random.seed(SEED)
    if n_rhs is not None:
        rhs = np.random.rand(n_rows + 10, n_rhs)
    else:
        rhs = np.random.rand(n_rows + 10)

    # the matrix is converted to the desired layout
    mat, kwargs = convert_matrix_to_layout(mat_full, input_layout)

    # the left-hand side matrix and right-hand side is converted to the desired order
    mat = convert_matrix_to_order(mat=mat, from_order=from_order)
    rhs = convert_matrix_to_order(mat=rhs, from_order=from_order)

    # the solution is computed, but due to the wrong shape of the right-hand side, an
    # error has to be raised
    with pytest.raises(ValueError, match=SHAPE_MISMATCH_ERROR_REF_CONTENT):
        pp.solve(
            mat=mat,
            rhs=rhs,
            solver=solver_alias,  # type: ignore
            num_threads=num_threads,
            **kwargs,
        )

    return
