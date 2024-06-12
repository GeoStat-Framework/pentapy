# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
This is a solver linear equation systems with a penta-diagonal matrix,
implemented in Cython.

"""

# Imports

import numpy as np

cimport numpy as np

from cython cimport view
from cython.parallel import prange
from libc.stdint cimport int64_t


# === Constants ===

cdef enum: MAT_FACT_N_COLS = 5

cdef enum Solvers:
    PTRRANS_1 = 1
    PTRRANS_2 = 2

cdef enum Infos:
    SUCCESS = 0
    SHAPE_MISMATCH = -1
    WRONG_SOLVER = -2

# === Main Python Interface ===


def penta_solver1(
    double[::, ::1] mat_flat,
    double[::, ::1] rhs,
    int workers,
):

    # NOTE: info is defined to be overwritten for possible future validations
    cdef int info

    return (
        np.asarray(
            c_penta_solver1(
                mat_flat,
                rhs,
                workers,
                &info,
            )
        ),
        info,
    )


def penta_solver2(
    double[::, ::1] mat_flat,
    double[::, ::1] rhs,
    int workers,
):

    # NOTE: info is defined to be overwritten for possible future validations
    cdef int info

    return (
        np.asarray(
            c_penta_solver2(
                mat_flat,
                rhs,
                workers,
                &info,
            )
        ),
        info,
    )


# === Solver Algorithm 1 ===

cdef double[::, ::1] c_penta_solver1(
    double[::, ::1] mat_flat,
    double[::, ::1] rhs,
    int workers,
    int* info,
):
    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the matrix ``A`` and
    the right-hand side ``b`` by

    - factorizing the matrix ``A`` into auxiliary coefficients and a unit upper
        triangular matrix ``U``
    - transforming the right-hand side into a vector ``zeta``
    - solving the system of equations ``Ux = zeta`` by backward substitution

    """

    # --- Initial checks ---

    # if the number of columns in the flattened matrix is not equal to the number of
    # rows in the right-hand side, the function exits early to avoid memory errors
    if mat_flat.shape[1] != rhs.shape[0]:
        info[0] = Infos.SHAPE_MISMATCH
        return np.empty_like(rhs)

    # --- Solving the system of equations ---

    # first, the matrix is factorized
    cdef double[::, ::1] mat_factorized = _c_interf_factorize(
        mat_flat,
        info,
        Solvers.PTRRANS_1,
    )

    # in case of an error during factorization, the function exits early
    if info[0] != Infos.SUCCESS:
        return np.empty_like(rhs)

    # then, all the right-hand sides are solved
    return _c_interf_factorize_solve(
        mat_factorized,
        rhs,
        workers,
        info,
        Solvers.PTRRANS_1,
    )



cdef double[::, ::1] _c_interf_factorize(
    double[::, ::1] mat_flat,
    int* info,
    int solver,
):
    """
    This function serves as the interface that takes the memoryview of the flattened
    matrix and returns the freshly allocated factorized matrix.

    """

    # --- Variable declarations ---

    cdef int64_t mat_n_cols = mat_flat.shape[1]
    tmp = view.array(
        shape=(mat_n_cols, MAT_FACT_N_COLS),
        itemsize=sizeof(double),
        format="d",
    )
    cdef double[::, ::1] mat_factorized = tmp

    # --- Factorization ---

    # the solver algorithm is chosen based on the input parameter
    # Case 1: PTRRANS-I
    if solver == Solvers.PTRRANS_1:
        info[0] = _c_core_factorize_algo_1(
            &mat_flat[0, 0],
            mat_n_cols,
            &mat_factorized[0, 0],
        )
        return mat_factorized

    # Case 2: PTRRANS-II
    elif solver == Solvers.PTRRANS_2:
        info[0] = _c_core_factorize_algo_2(
            &mat_flat[0, 0],
            mat_n_cols,
            &mat_factorized[0, 0],
        )
        return mat_factorized

    # Case 3: the wrong solver is chosen
    else:
        info[0] = Infos.WRONG_SOLVER
        return mat_factorized


cdef double[::, ::1] _c_interf_factorize_solve(
    double[::, ::1] mat_factorized,
    double[::, ::1] rhs,
    int workers,
    int* info,
    int solver,
):
    """
    This function serves as the interface that takes the factorized matrix and the
    right-hand sides and returns the freshly allocated solution vector obtained by
    solving the system of equations via backward substitution.

    """

    # --- Variable declarations ---

    cdef int64_t mat_n_cols = mat_factorized.shape[0]
    cdef int64_t rhs_n_cols = rhs.shape[1]
    cdef int64_t iter_col
    tmp = view.array(
        shape=(mat_n_cols, rhs_n_cols),
        itemsize=sizeof(double),
        format="d",
    )
    cdef double[::, ::1] result = tmp

    # --- Solving the system of equations ---

    # the solver algorithm is chosen based on the input parameter
    # Case 1: PTRRANS-I
    if solver == Solvers.PTRRANS_1:
        for iter_col in prange(
            rhs_n_cols,
            nogil=True,
            num_threads=workers,
        ):
            info[0] = _c_core_factorize_solve_algo_1(
                mat_n_cols,
                &mat_factorized[0, 0],
                &rhs[0, iter_col],
                rhs_n_cols,
                &result[0, iter_col],
            )

        return result

    # Case 2: PTRRANS-II
    elif solver == Solvers.PTRRANS_2:
        for iter_col in prange(
            rhs_n_cols,
            nogil=True,
            num_threads=workers,
        ):
            info[0] = _c_core_factorize_solve_algo_2(
                mat_n_cols,
                &mat_factorized[0, 0],
                &rhs[0, iter_col],
                rhs_n_cols,
                &result[0, iter_col],
            )

        return result

    # Case 3: the wrong solver is chosen
    else:
        info[0] = Infos.WRONG_SOLVER
        return result


cdef int _c_core_factorize_algo_1(
    double* mat_flat,
    int64_t mat_n_cols,
    double* mat_factorized,
):
    """
    Factorizes the pentadiagonal matrix ``A`` into

    - auxiliary coefficients ``e``, ``mu`` and ``gamma`` (``ga``) for the transformation
        of the right-hand side
    - a unit upper triangular matrix with the main diagonals ``alpha``(``al``) and
        ``beta`` (``be``) for the following backward substitution. Its unit main
        diagonal is implicit.

    They are overwriting the memoryview ``mat_factorized`` as follows:

    ```bash
    [[   *           *          mu_0        al_0        be_0      ]
     [   *          ga_1        mu_1        al_1        be_1      ]
     [  e_2         ga_2        mu_2        al_2        be_2      ]
                                ...
     [  e_i         ga_i        mu_i        al_i        be_i  ]
     [  e_{n-3}     ga_{n-3}    mu_{n-3}    al_{n-3}    be_{n-3}  ]                                ...
     [  e_{n-2}     ga_{n-2}    mu_{n-2}    al_{n-2}      *       ]
     [  e_{n-1}     ga_{n-1}    mu_{n-1}      *           *       ]]
    ```

    where the entries marked with ``*`` are not used by design, but overwritten with
    zeros.

    """

    # --- Variable declarations ---

    cdef int64_t iter_row, fact_curr_base_idx
    cdef int64_t mat_row_base_idx_1 = mat_n_cols  # base index for the second row
    cdef int64_t mat_row_base_idx_2 = 2 * mat_n_cols  # base index for the third row
    cdef int64_t mat_row_base_idx_3 = 3 * mat_n_cols  # base index for the fourth row
    cdef int64_t mat_row_base_idx_4 = 4 * mat_n_cols  # base index for the fifth row
    cdef double mu_i, ga_i, e_i  # mu, gamma, e
    cdef double al_i, al_i_minus_1, al_i_plus_1  # alpha
    cdef double be_i, be_i_minus_1, be_i_plus_1  # beta

    # --- Factorization ---

    # NOTE: in the following mu is manually checked for zero-division to extract the
    #       proper value of ``info`` and exit early in case of failure;
    #       ``info`` is set to the row count where the error occured as for LAPACK ``pbtrf``

    # First row
    mu_i = mat_flat[mat_row_base_idx_2]
    if mu_i == 0.0:
        return 1

    al_i_minus_1 = mat_flat[mat_row_base_idx_1] / mu_i
    be_i_minus_1 = mat_flat[0] / mu_i


    mat_factorized[0] = 0.0
    mat_factorized[1] = 0.0
    mat_factorized[2] = mu_i
    mat_factorized[3] = al_i_minus_1
    mat_factorized[4] = be_i_minus_1

    # Second row
    ga_i = mat_flat[mat_row_base_idx_3 + 1]
    mu_i = mat_flat[mat_row_base_idx_2 + 1] - al_i_minus_1 * ga_i
    if mu_i == 0.0:
        return 2

    al_i = (mat_flat[mat_row_base_idx_1 + 1] - be_i_minus_1 * ga_i) / mu_i
    be_i = mat_flat[1] / mu_i

    mat_factorized[5] = 0.0
    mat_factorized[6] = ga_i
    mat_factorized[7] = mu_i
    mat_factorized[8] = al_i
    mat_factorized[9] = be_i

    # Central rows
    fact_curr_base_idx = 10
    for iter_row in range(2, mat_n_cols-2):
        e_i = mat_flat[mat_row_base_idx_4 + iter_row]
        ga_i = mat_flat[mat_row_base_idx_3 + iter_row] - al_i_minus_1 * e_i
        mu_i = mat_flat[mat_row_base_idx_2 + iter_row] - be_i_minus_1 * e_i - al_i * ga_i
        if mu_i == 0.0:
            return iter_row + 1

        al_i_plus_1 = (mat_flat[mat_row_base_idx_1 + iter_row] - be_i * ga_i) / mu_i
        al_i_minus_1 = al_i
        al_i = al_i_plus_1

        be_i_plus_1 = mat_flat[iter_row] / mu_i
        be_i_minus_1 = be_i
        be_i = be_i_plus_1

        mat_factorized[fact_curr_base_idx] = e_i
        mat_factorized[fact_curr_base_idx + 1] = ga_i
        mat_factorized[fact_curr_base_idx + 2] = mu_i
        mat_factorized[fact_curr_base_idx + 3] = al_i
        mat_factorized[fact_curr_base_idx + 4] = be_i

        fact_curr_base_idx += MAT_FACT_N_COLS

    # Second to last row
    e_i = mat_flat[mat_row_base_idx_4 + mat_n_cols - 2]
    ga_i = mat_flat[mat_row_base_idx_3 + mat_n_cols - 2] - al_i_minus_1 * e_i
    mu_i = mat_flat[mat_row_base_idx_2 + mat_n_cols - 2] - be_i_minus_1 * e_i - al_i * ga_i
    if mu_i == 0.0:
        return mat_n_cols - 1

    al_i_plus_1 = (mat_flat[mat_row_base_idx_1 + mat_n_cols - 2] - be_i * ga_i) / mu_i

    mat_factorized[fact_curr_base_idx] = e_i
    mat_factorized[fact_curr_base_idx + 1] = ga_i
    mat_factorized[fact_curr_base_idx + 2] = mu_i
    mat_factorized[fact_curr_base_idx + 3] = al_i_plus_1
    mat_factorized[fact_curr_base_idx + 4] = 0.0

    # Last Row
    e_i = mat_flat[mat_row_base_idx_4 + mat_n_cols - 1]
    ga_i = mat_flat[mat_row_base_idx_3 + mat_n_cols - 1] - al_i * e_i
    mu_i = mat_flat[mat_row_base_idx_2 + mat_n_cols - 1] - be_i * e_i - al_i_plus_1 * ga_i
    if mu_i == 0.0:
        return mat_n_cols

    mat_factorized[fact_curr_base_idx + 5] = e_i
    mat_factorized[fact_curr_base_idx + 6] = ga_i
    mat_factorized[fact_curr_base_idx + 7] = mu_i
    mat_factorized[fact_curr_base_idx + 8] = 0.0
    mat_factorized[fact_curr_base_idx + 9] = 0.0

    return 0


cdef int _c_core_factorize_solve_algo_1(
    int64_t mat_n_cols,
    double* mat_factorized,
    double* rhs_single,
    int64_t rhs_n_cols,
    double* result_view,
) except * nogil:
    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the factorized
    unit upper triangular matrix ``U`` and the right-hand side ``b``.
    It overwrites the right-hand side ``b`` first with the transformed vector ``zeta``
    and then with the solution vector ``x`` for ``Ux = zeta``.

    """

    # --- Variable declarations ---

    cdef int64_t iter_row, fact_curr_base_idx, res_curr_base_idx
    cdef double ze_i, ze_i_minus_1, ze_i_plus_1  # zeta

    # --- Transformation ---

    # first, the right-hand side is transformed into the vector ``zeta``
    # First row

    ze_i_minus_1 = rhs_single[0] / mat_factorized[2]
    result_view[0] = ze_i_minus_1

    # Second row
    ze_i = (rhs_single[rhs_n_cols] - ze_i_minus_1 * mat_factorized[6]) / mat_factorized[7]
    result_view[rhs_n_cols] = ze_i

    # Central rows
    fact_curr_base_idx = 10
    res_curr_base_idx = rhs_n_cols + rhs_n_cols

    for iter_row in range(2, mat_n_cols-2):
        ze_i_plus_1 = (
            rhs_single[res_curr_base_idx]
            - ze_i_minus_1 * mat_factorized[fact_curr_base_idx]
            - ze_i * mat_factorized[fact_curr_base_idx + 1]
        ) / mat_factorized[fact_curr_base_idx + 2]
        ze_i_minus_1 = ze_i
        ze_i = ze_i_plus_1
        result_view[res_curr_base_idx] = ze_i_plus_1

        fact_curr_base_idx += MAT_FACT_N_COLS
        res_curr_base_idx += rhs_n_cols

    # Second to last row
    ze_i_plus_1 = (
        rhs_single[res_curr_base_idx]
        - ze_i_minus_1 * mat_factorized[fact_curr_base_idx]
        - ze_i * mat_factorized[fact_curr_base_idx + 1]
    ) / mat_factorized[fact_curr_base_idx + 2]
    ze_i_minus_1 = ze_i
    ze_i = ze_i_plus_1
    result_view[res_curr_base_idx] = ze_i_plus_1

    # Last row
    ze_i_plus_1 = (
        rhs_single[res_curr_base_idx + rhs_n_cols]
        - ze_i_minus_1 * mat_factorized[fact_curr_base_idx + 5]
        - ze_i * mat_factorized[fact_curr_base_idx + 6]
    ) / mat_factorized[fact_curr_base_idx + 7]
    result_view[res_curr_base_idx + rhs_n_cols] = ze_i_plus_1

    # --- Backward substitution ---

    # The solution vector is calculated by backward substitution that overwrites the
    # right-hand side vector with the solution vector
    ze_i -= mat_factorized[fact_curr_base_idx + 3] * ze_i_plus_1
    result_view[res_curr_base_idx] = ze_i

    for iter_row in range(mat_n_cols-3, -1, -1):
        fact_curr_base_idx -= MAT_FACT_N_COLS
        res_curr_base_idx -= rhs_n_cols

        result_view[res_curr_base_idx] -= (
            mat_factorized[fact_curr_base_idx + 3] * ze_i
            + mat_factorized[fact_curr_base_idx + 4] * ze_i_plus_1
        )
        ze_i_plus_1 = ze_i
        ze_i = result_view[res_curr_base_idx]

    return 0

# === Solver Algorithm 2 ===


cdef double[::, ::1] c_penta_solver2(
    double[::, ::1] mat_flat,
    double[::, ::1] rhs,
    int workers,
    int* info,
):
    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the matrix ``A`` and
    the right-hand side ``b`` by

    - factorizing the matrix ``A`` into auxiliary coefficients and a unit lower
        triangular matrix ``L``
    - transforming the right-hand side into a vector ``omega``
    - solving the system of equations ``Lx = omega`` by backward substitution

    """

    # --- Initial checks ---

    # if the number of columns in the flattened matrix is not equal to the number of
    # rows in the right-hand side, the function exits early to avoid memory errors
    if mat_flat.shape[1] != rhs.shape[0]:
        info[0] = Infos.SHAPE_MISMATCH
        return np.empty_like(rhs)

    # --- Solving the system of equations ---

    # first, the matrix is factorized
    cdef double[::, ::1] mat_factorized = _c_interf_factorize(
        mat_flat,
        info,
        Solvers.PTRRANS_2,
    )

    # in case of an error during factorization, the function exits early
    if info[0] != Infos.SUCCESS:
        return np.empty_like(rhs)

    # then, all the right-hand sides are solved
    return _c_interf_factorize_solve(
        mat_factorized,
        rhs,
        workers,
        info,
        Solvers.PTRRANS_2,
    )


cdef int _c_core_factorize_algo_2(
    double* mat_flat,
    int64_t mat_n_cols,
    double* mat_factorized,
):
    """
    Factorizes the pentadiagonal matrix ``A`` into

    - auxiliary coefficients ``psi`` (``ps``), ``rho`` and ``b`` for the transformation
        of the right-hand side
    - a unit lower triangular matrix with the main diagonals ``phi`` and ``sigma``
       (``si``) for the following forward substitution. Its unit main diagonal is
       implicit.

    They are overwriting the memoryview ``mat_factorized`` as follows:

    ```bash
    [[    *           *         ps_0        rho_0       b_i      ]
     [    *         si_1        ps_1        rho_1       b_1      ]
     [  phi_2       si_2        ps_2        rho_2       b_2      ]
                                ...
     [  phi_i       si_i        ps_i        rho_i       b_i      ]
                                ...
     [  phi_{n-3}   si_{n-3}    ps_{n-3}    rho_{n-3}   b_{n-3}  ]
     [  phi_{n-2}   si_{n-2}    ps_{n-2}    rho_{n-2}     *      ]
     [  phi_{n-1}   si_{n-1}    ps_{n-1}      *           *      ]]
    ```

    where the entries marked with ``*`` are not used by design, but overwritten with
    zeros.

    """

    # --- Variable declarations ---

    cdef int64_t iter_row, fact_curr_base_idx
    cdef int64_t mat_row_base_idx_1 = mat_n_cols  # base index for the second row
    cdef int64_t mat_row_base_idx_2 = 2 * mat_n_cols  # base index for the third row
    cdef int64_t mat_row_base_idx_3 = 3 * mat_n_cols  # base index for the fourth row
    cdef int64_t mat_row_base_idx_4 = 4 * mat_n_cols  # base index for the fifth row
    cdef double ps_i, rho_i  # psi, rho
    cdef double si_i, si_i_minus_1, si_i_plus_1  # sigma
    cdef double phi_i, phi_i_minus_1, phi_i_plus_1  # phi

    # --- Factorization ---

    # NOTE: in the following ps is manually checked for zero-division to extract the
    #       proper value of ``info`` and exit early in case of failure;
    #       ``info`` is set to the row count where the error occured as for LAPACK ``pbtrf``

    # First row

    ps_i = mat_flat[mat_row_base_idx_2 + mat_n_cols - 1]
    if ps_i == 0.0:
        return mat_n_cols

    si_i_plus_1 = mat_flat[mat_row_base_idx_3 + mat_n_cols - 1] / ps_i
    phi_i_plus_1 = mat_flat[mat_row_base_idx_4 + mat_n_cols - 1] / ps_i

    fact_curr_base_idx = (mat_n_cols - 1) * MAT_FACT_N_COLS
    mat_factorized[fact_curr_base_idx + 4] = 0.0
    mat_factorized[fact_curr_base_idx + 3] = 0.0
    mat_factorized[fact_curr_base_idx + 2] = ps_i
    mat_factorized[fact_curr_base_idx + 1] = si_i_plus_1
    mat_factorized[fact_curr_base_idx] = phi_i_plus_1

    # Second row
    rho_i = mat_flat[mat_row_base_idx_1 + mat_n_cols - 2]
    ps_i = mat_flat[mat_row_base_idx_2 + mat_n_cols - 2] - si_i_plus_1 * rho_i
    if ps_i == 0.0:
        return mat_n_cols - 1

    si_i = (mat_flat[mat_row_base_idx_3 + mat_n_cols - 2] - phi_i_plus_1 * rho_i) / ps_i
    phi_i = mat_flat[mat_row_base_idx_4 + mat_n_cols - 2] / ps_i

    fact_curr_base_idx -= MAT_FACT_N_COLS
    mat_factorized[fact_curr_base_idx + 4] = 0.0
    mat_factorized[fact_curr_base_idx + 3] = rho_i
    mat_factorized[fact_curr_base_idx + 2] = ps_i
    mat_factorized[fact_curr_base_idx + 1] = si_i
    mat_factorized[fact_curr_base_idx] = phi_i

    # Central rows
    for iter_row in range(mat_n_cols - 3, 1, -1):
        b_i = mat_flat[iter_row]
        rho_i = mat_flat[mat_row_base_idx_1 + iter_row] - si_i_plus_1 * b_i
        ps_i = mat_flat[mat_row_base_idx_2 + iter_row] - phi_i_plus_1 * b_i - si_i * rho_i
        if ps_i == 0.0:
            return iter_row + 1

        si_i_minus_1 = (mat_flat[mat_row_base_idx_3 + iter_row] - phi_i * rho_i) / ps_i
        si_i_plus_1 = si_i
        si_i = si_i_minus_1
        phi_i_minus_1 = mat_flat[mat_row_base_idx_4 + iter_row] / ps_i
        phi_i_plus_1 = phi_i
        phi_i = phi_i_minus_1

        fact_curr_base_idx -= MAT_FACT_N_COLS
        mat_factorized[fact_curr_base_idx + 4] = b_i
        mat_factorized[fact_curr_base_idx + 3] = rho_i
        mat_factorized[fact_curr_base_idx + 2] = ps_i
        mat_factorized[fact_curr_base_idx + 1] = si_i
        mat_factorized[fact_curr_base_idx] = phi_i

    # Second to last row
    b_i = mat_flat[1]
    rho_i = mat_flat[mat_row_base_idx_1 + 1] - si_i_plus_1 * b_i
    ps_i = mat_flat[mat_row_base_idx_2 + 1] - phi_i_plus_1 * b_i - si_i * rho_i
    if ps_i == 0.0:
        return 2

    si_i_minus_1 = (mat_flat[mat_row_base_idx_3 + 1] - phi_i * rho_i) / ps_i
    si_i_plus_1 = si_i
    si_i = si_i_minus_1

    mat_factorized[9] = b_i
    mat_factorized[8] = rho_i
    mat_factorized[7] = ps_i
    mat_factorized[6] = si_i
    mat_factorized[5] = 0.0

    # Last row
    b_i = mat_flat[0]
    rho_i = mat_flat[mat_row_base_idx_1 + 0] - si_i_plus_1 * b_i
    ps_i = mat_flat[mat_row_base_idx_2 + 0] - phi_i * b_i - si_i * rho_i
    if ps_i == 0.0:
        return 1

    mat_factorized[4] = b_i
    mat_factorized[3] = rho_i
    mat_factorized[2] = ps_i
    mat_factorized[1] = 0.0
    mat_factorized[0] = 0.0

    return 0


cdef int _c_core_factorize_solve_algo_2(
    int64_t mat_n_cols,
    double* mat_factorized,
    double* rhs_single,
    int64_t rhs_n_cols,
    double* result_view,
) except * nogil:

    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the factorized
    unit lower triangular matrix ``L`` and the right-hand side ``b``.
    It overwrites the right-hand side ``b`` first with the transformed vector ``omega``
    and then with the solution vector ``x`` for ``Lx = omega``.

    """

    # --- Variable declarations ---

    cdef int64_t iter_row, fact_curr_base_idx, res_curr_base_idx
    cdef double om_i, om_i_minus_1, om_i_plus_1  # omega

    # --- Transformation ---

    # first, the right-hand side is transformed into the vector ``omega``
    # First row
    fact_curr_base_idx = (mat_n_cols - 1) * MAT_FACT_N_COLS
    res_curr_base_idx = (mat_n_cols - 1) * rhs_n_cols

    om_i_plus_1 = rhs_single[res_curr_base_idx] / mat_factorized[fact_curr_base_idx + 2]
    result_view[res_curr_base_idx] = om_i_plus_1

    # Second row
    fact_curr_base_idx -= MAT_FACT_N_COLS
    res_curr_base_idx -= rhs_n_cols

    om_i = (
        rhs_single[res_curr_base_idx]
        - om_i_plus_1 * mat_factorized[fact_curr_base_idx + 3]
    ) / mat_factorized[fact_curr_base_idx + 2]
    result_view[res_curr_base_idx] = om_i

    # Central rows
    for iter_row in range(mat_n_cols - 3, 1, -1):
        fact_curr_base_idx -= MAT_FACT_N_COLS
        res_curr_base_idx -= rhs_n_cols

        om_i_minus_1 = (
            rhs_single[res_curr_base_idx]
            - om_i_plus_1 * mat_factorized[fact_curr_base_idx + 4]
            - om_i * mat_factorized[fact_curr_base_idx + 3]
        ) / mat_factorized[fact_curr_base_idx + 2]
        om_i_plus_1 = om_i
        om_i = om_i_minus_1
        result_view[res_curr_base_idx] = om_i

    # Second to last row
    fact_curr_base_idx -= MAT_FACT_N_COLS
    res_curr_base_idx -= rhs_n_cols

    om_i_minus_1 = (
        rhs_single[res_curr_base_idx]
        - om_i_plus_1 * mat_factorized[fact_curr_base_idx + 4]
        - om_i * mat_factorized[fact_curr_base_idx + 3]
    ) / mat_factorized[fact_curr_base_idx + 2]
    om_i_plus_1 = om_i
    om_i = om_i_minus_1
    result_view[res_curr_base_idx] = om_i

    # Last row
    om_i_minus_1 = (
        rhs_single[0]
        - om_i_plus_1 * mat_factorized[4]
        - om_i * mat_factorized[3]
    ) / mat_factorized[2]
    result_view[0] = om_i_minus_1

    # --- Forward substitution ---

    # The solution vector is calculated by forward substitution that overwrites the
    # right-hand side vector with the solution vector
    om_i -= mat_factorized[fact_curr_base_idx + 1] * om_i_minus_1
    result_view[res_curr_base_idx] = om_i

    for iter_row in range(2, mat_n_cols):
        fact_curr_base_idx += MAT_FACT_N_COLS
        res_curr_base_idx += rhs_n_cols

        result_view[res_curr_base_idx] = (
            result_view[res_curr_base_idx]
            - mat_factorized[fact_curr_base_idx] * om_i_minus_1
            - mat_factorized[fact_curr_base_idx + 1] * om_i
        )
        om_i_minus_1 = om_i
        om_i = result_view[res_curr_base_idx]

    return 0
