"""
This test suite implements reusable templates for testing the pentadiagonal solver based
on either Algorithm PTRANS-I or PTRANS-II.
"""

# === Imports ===

from typing import Literal

import numpy as np
import pytest
import util_funcs as uf

import pentapy as pp

# === Constants ===

SEED = 19_031_977
REF_WARNING_CONTENT = "not suitable for input-matrix."
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
    500,
    501,
    1_000,
    1_001,
    5_000,
    5_001,
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
    "workers": [1],
}

# === Templates ===


def pentapy_solvers_template(
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
    workers: int,
) -> None:
    """
    Tests the pentadiagonal solver based on Algorithm PTRANS-I when starting from
    different input layouts, number of right-hand sides, number of rows, and also
    when inducing an error by making the first diagonal element zero.
    It has to be ensured that the edge case of ``n_rows = 3`` is also covered.

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
            pytest.skip(
                "Only 3 rows, cannot induce error because this will not go into "
                "PTRANS-I, but NumPy."
            )

        if solver_alias in SOLVER_ALIASES_PTRANS_I:
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
    if input_layout == "full":
        mat = mat_full
        kwargs = dict(is_flat=False)

    elif input_layout == "banded_row_wise":
        mat = pp.create_banded(mat_full, col_wise=False)
        kwargs = dict(
            is_flat=True,
            index_row_wise=True,
        )

    elif input_layout == "banded_col_wise":
        mat = pp.create_banded(mat_full, col_wise=True)
        kwargs = dict(
            is_flat=True,
            index_row_wise=False,
        )

    else:
        raise ValueError(f"Invalid input layout: {input_layout}")

    # the matrix is converted to the desired order
    if from_order == "C":
        mat = np.ascontiguousarray(mat)
        rhs = np.ascontiguousarray(rhs)
    elif from_order == "F":
        mat = np.asfortranarray(mat)
        rhs = np.asfortranarray(rhs)
    else:
        raise ValueError(f"Invalid from order: {from_order=}")

    # the solution is computed
    # Case 1: in case of an error, a warning has to be issued and the result has to
    # be NaN
    if induce_error:
        with pytest.warns(UserWarning, match=REF_WARNING_CONTENT):
            mat_ref_copy = mat.copy()
            sol = pp.solve(
                mat=mat,
                rhs=rhs,
                solver=solver_alias,  # type: ignore
                workers=workers,
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
        workers=workers,
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
