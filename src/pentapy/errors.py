"""
Auxiliary errors for the pentapy package.

"""

# === Imports ===

from enum import Enum


class PentaPyErrorMessages(str, Enum):
    """
    Defines the possible error messages for the pentapy package, namely

    - ``WRONG_WORKERS``: the number of workers is incorrect
    - ``SINGULAR_MATRIX``: the matrix is singular
    - ``SHAPE_MISMATCH``: the shape of the input arrays is incorrect
    - ``WRONG_SOLVER``: the solver alias is incorrect on C-level (internal error,
        should not occur)
    - ``UNKNOWN_ERROR``: an unknown error occurred

    """

    WRONG_WORKERS = (
        "pentapy.solve: workers has to be -1 or greater, but got workers={workers}"
    )
    SINGULAR_MATRIX = (
        "pentapy: {solver_inter_name} solver encountered singular matrix at "
        "row index {row_idx}. Returning NaNs."
    )
    SHAPE_MISMATCH = (
        "pentapy.solve: shape mismatch between the number of equations in the "
        "left-hand side matrix ({lhs_n_cols}) and the number of right-hand sides "
        "({rhs_n_rows})."
    )
    WRONG_SOLVER = "pentapy.solve: failure in determining the solver internally."
    UNKNOWN_ERROR = "pentapy.solve: unknown error in the pentadiagonal solver."
