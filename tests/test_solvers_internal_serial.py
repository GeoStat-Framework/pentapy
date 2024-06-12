"""
Test suite for testing the pentadiagonal solver based on either Algorithm PTRANS-I or
PTRANS-II.

It tests them in SERIAL mode only.

"""

# === Imports ===

from copy import deepcopy
from typing import Literal

import pytest
import templates

# === Tests ===

# the following series of decorators parametrize the tests for the pentadiagonal solver
# based on either Algorithm PTRANS-I or PTRANS-II in serial mode


# --- Extended solve test ---


def test_pentapy_solvers_extended_serial(
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

    templates.pentapy_solvers_extended_template(
        n_rows=n_rows,
        n_rhs=n_rhs,
        input_layout=input_layout,
        solver_alias=solver_alias,
        induce_error=induce_error,
        from_order=from_order,
        workers=workers,
    )


for key, value in templates.PARAM_DICT.items():
    test_pentapy_solvers_extended_serial = pytest.mark.parametrize(key, value)(
        test_pentapy_solvers_extended_serial
    )


# --- Shape mismatch test ---


def test_pentapy_solvers_shape_mismatch_serial(
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
    workers: int,
) -> None:

    templates.pentapy_solvers_shape_mismatch_template(
        n_rows=n_rows,
        n_rhs=n_rhs,
        input_layout=input_layout,
        solver_alias=solver_alias,
        from_order=from_order,
        workers=workers,
    )


params_dict_without_induce_error = deepcopy(templates.PARAM_DICT)
params_dict_without_induce_error.pop("induce_error")


for key, value in params_dict_without_induce_error.items():
    test_pentapy_solvers_shape_mismatch_serial = pytest.mark.parametrize(key, value)(
        test_pentapy_solvers_shape_mismatch_serial
    )
