"""
Test suite for testing the external solvers that can be called via pentapy. The tests
are not exhaustive and only check whether the solvers can be called and return a
solution.

"""

# === Imports ===

from typing import Literal

import numpy as np
import pentapy as pp
import pytest
import util_funcs as uf

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
]
REF_WARNING_CONTENT = "singular"
SOLVER_ALIASES_LAPACK = [3, "3", "lapack", "LaPaCk"]
SOLVER_ALIASES_SPSOLVE = [4, "4", "spsolve", "SpSoLvE"]

# === Tests ===


@pytest.mark.parametrize("induce_error", [False, True])
@pytest.mark.parametrize("solver_alias", SOLVER_ALIASES_LAPACK + SOLVER_ALIASES_SPSOLVE)
@pytest.mark.parametrize("input_layout", ["full", "banded_row_wise", "banded_col_wise"])
@pytest.mark.parametrize("n_rhs", [None, 1, 10])
@pytest.mark.parametrize("n_rows", N_ROWS)
def test_external_solvers(
    n_rows: int,
    n_rhs: int,
    input_layout: Literal["full", "banded_row_wise", "banded_col_wise"],
    solver_alias: Literal[1, "1", "PTRANS-I"],
    induce_error: bool,
) -> None:
    """
    Tests the external bindings for solving pentadiagonal systems starting from
    different input layouts, number of right-hand sides, number of rows, and when an
    error is induced by a zero matrix.
    It has to be ensured that the edge case of ``n_rows = 3`` is also covered.

    """

    # first, a random pentadiagonal matrix is generated
    mat_full = np.zeros(shape=(n_rows, n_rows))
    if not induce_error:
        mat_full[::, ::] = uf.gen_conditioned_rand_penta_matrix_dense(
            n_rows=n_rows,
            seed=SEED,
            ill_conditioned=False,
        )

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

    # the solution is computed
    # Case 1: in case of an error, a warning has to be issued and the result has to
    # be NaN
    if induce_error:
        with pytest.warns(UserWarning, match=REF_WARNING_CONTENT):
            sol = pp.solve(
                mat=mat,
                rhs=rhs,
                solver=solver_alias,  # type: ignore
                **kwargs,
            )
            assert sol.shape == result_shape
            assert np.isnan(sol).all()

        return

    # Case 2: in case of no error, the solution can be computed without any issues
    sol = pp.solve(
        mat=mat,
        rhs=rhs,
        solver=solver_alias,  # type: ignore
        **kwargs,
    )
    assert sol.shape == result_shape
