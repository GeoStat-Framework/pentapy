"""
Test suite for testing the pentadiagonal solver based on either Algorithm PTRANS-I or
PTRANS-II.

It tests them in PARALLEL mode.

"""

# === Imports ===

from copy import deepcopy
from typing import Literal, Optional

import pytest
import templates

# === Tests ===

# the following series of decorators parametrize the tests for the pentadiagonal solver
# based on either Algorithm PTRANS-I or PTRANS-II in parallel mode
param_dict = deepcopy(templates.PARAM_DICT)
param_dict["from_order"] = ["C"]
param_dict["num_threads"] = [-1]

# --- Extended solve test ---


def test_pentapy_solvers_parallel(
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

    templates.pentapy_solvers_extended_template(
        n_rows=n_rows,
        n_rhs=n_rhs,
        input_layout=input_layout,
        solver_alias=solver_alias,
        induce_error=induce_error,
        from_order=from_order,
        num_threads=num_threads,
    )


for key, value in param_dict.items():
    test_pentapy_solvers_parallel = pytest.mark.parametrize(key, value)(
        test_pentapy_solvers_parallel
    )


# --- Different number of threads test ---


@pytest.mark.parametrize("num_threads", [0, 1, -1, -2, None])
def test_pentapy_solvers_parallel_different_num_threads(
    num_threads: Optional[int],
) -> None:
    """
    Tests that the parallel solvers run properly with different numbers of threads.

    """

    kwargs = dict(
        n_rows=10,
        n_rhs=1,
        input_layout="full",
        solver_alias=1,
        induce_error=False,
        from_order="C",
        num_threads=num_threads,
    )

    # NOTE: if there is no crash, the test is successful
    templates.pentapy_solvers_extended_template(**kwargs)  # type: ignore


# --- Shape mismatch test ---


def test_pentapy_solvers_shape_mismatch_parallel(
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

    templates.pentapy_solvers_shape_mismatch_template(
        n_rows=n_rows,
        n_rhs=n_rhs,
        input_layout=input_layout,
        solver_alias=solver_alias,
        from_order=from_order,
        num_threads=num_threads,
    )


params_dict_without_induce_error = deepcopy(templates.PARAM_DICT)
params_dict_without_induce_error["num_threads"] = [-1]
params_dict_without_induce_error.pop("induce_error")

for key, value in params_dict_without_induce_error.items():
    test_pentapy_solvers_shape_mismatch_parallel = pytest.mark.parametrize(key, value)(
        test_pentapy_solvers_shape_mismatch_parallel
    )
