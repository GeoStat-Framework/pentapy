"""
Test suite for testing the pentadiagonal solver based on either Algorithm PTRANS-I or
PTRANS-II.

It tests them in PARALLEL mode.

"""

# === Imports ===

from copy import deepcopy
from typing import Literal, Optional, Type

import pytest
import templates

# === Tests ===

# the following series of decorators parametrize the tests for the pentadiagonal solver
# based on either Algorithm PTRANS-I or PTRANS-II in parallel mode
param_dict = deepcopy(templates.PARAM_DICT)
param_dict["from_order"] = ["C"]
param_dict["workers"] = [-1]


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
    workers: int,
) -> None:

    templates.pentapy_solvers_template(
        n_rows=n_rows,
        n_rhs=n_rhs,
        input_layout=input_layout,
        solver_alias=solver_alias,
        induce_error=induce_error,
        from_order=from_order,
        workers=workers,
    )


for key, value in param_dict.items():
    test_pentapy_solvers_parallel = pytest.mark.parametrize(key, value)(
        test_pentapy_solvers_parallel
    )


@pytest.mark.parametrize(
    "workers, expected", [(0, None), (1, None), (-1, None), (-2, ValueError)]
)
def test_pentapy_solvers_parallel_different_workers(
    workers: int, expected: Optional[Type[Exception]]
) -> None:
    """
    Tests the parallel solver with different number of workers, which might be wrong.

    """

    kwargs = dict(
        n_rows=10,
        n_rhs=1,
        input_layout="full",
        solver_alias=1,
        induce_error=False,
        from_order="C",
        workers=workers,
    )

    # Case 1: the test should fail
    if expected is not None:
        with pytest.raises(expected):
            templates.pentapy_solvers_template(**kwargs)  # type: ignore

        return

    # Case 2: the test should pass
    templates.pentapy_solvers_template(**kwargs)  # type: ignore
