"""
Auxiliary models for the pentapy package.

"""

# === Imports ===

from enum import IntEnum
from typing import Dict

# === Models ===


class Infos(IntEnum):
    """
    Defines the possible returns for ``info`` of the low level pentadiagonal solvers,
    namely

    - ``SUCCESS``: the solver has successfully solved the system
    - ``SHAPE_MISMATCH``: the shape of the input arrays is incorrect
    - ``WRONG_SOLVER``: the solver alias is the solver alias is incorrect on C-level
        (internal error, should not occur)

    """

    SUCCESS = 0
    SHAPE_MISMATCH = -1
    WRONG_SOLVER = -2


class PentaSolverAliases(IntEnum):
    """
    Defines all available solver aliases for pentadiagonal systems, namely

    - ``PTRANS_I``: the PTRANS-I algorithm
    - ``PTRANS_II``: the PTRANS-II algorithm
    - ``LAPACK``: Scipy's LAPACK solver :func:`scipy.linalg.solve_banded`
    - ``SUPER_LU``: Scipy's SuperLU solver :func:`scipy.sparse.linalg.spsolve(..., use_umfpack=False)`
    - ``UMFPACK``: Scipy's UMFpack solver :func:`scipy.sparse.linalg.spsolve(..., use_umfpack=True)`

    """  # noqa: E501

    PTRANS_I = 1
    PTRANS_II = 2
    LAPACK = 3
    SUPER_LU = 4
    UMFPACK = 5


# === Constants ===

_SOLVER_ALIAS_CONVERSIONS: Dict[str, PentaSolverAliases] = {
    "1": PentaSolverAliases.PTRANS_I,
    "ptrans-i": PentaSolverAliases.PTRANS_I,
    "2": PentaSolverAliases.PTRANS_II,
    "ptrans-ii": PentaSolverAliases.PTRANS_II,
    "3": PentaSolverAliases.LAPACK,
    "lapack": PentaSolverAliases.LAPACK,
    "solve_banded": PentaSolverAliases.LAPACK,
    "4": PentaSolverAliases.SUPER_LU,
    "spsolve": PentaSolverAliases.SUPER_LU,
    "5": PentaSolverAliases.UMFPACK,
    "spsolve_umf": PentaSolverAliases.UMFPACK,
    "umf": PentaSolverAliases.UMFPACK,
    "umf_pack": PentaSolverAliases.UMFPACK,
}
