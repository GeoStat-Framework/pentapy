# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=False

"""
This is a solver linear equation systems with a penta-diagonal matrix,
implemented in Cython.

"""

# Imports

import numpy as np

cimport numpy as np
from libc.stdint cimport int64_t


# === Main Python Interface ===


def penta_solver1(double[::, ::] mat_flat, double[::, ::] rhs):
    return np.asarray(c_penta_solver1(mat_flat, rhs))


def penta_solver2(double[::, ::] mat_flat, double[::, ::] rhs):
    return np.asarray(c_penta_solver2(mat_flat, rhs))


# === Solver Algorithm 1 ===


cdef double[::, ::] c_penta_solver1(double[::, ::] mat_flat, double[::, ::] rhs):
    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the matrix ``A`` and
    the right-hand side ``b`` by

    - factorizing the matrix ``A`` into auxiliary coefficients and a unit upper
        triangular matrix ``U``
    - transforming the right-hand side into a vector ``zeta``
    - solving the system of equations ``Ux = zeta`` by backward substitution

    """

    # === Variable declarations ===

    cdef int64_t mat_n_rows = mat_flat.shape[1]
    cdef int64_t rhs_n_cols = rhs.shape[1]
    cdef int64_t iter_col
    cdef double[::, ::1] result = np.empty(shape=(mat_n_rows, rhs_n_cols))
    cdef double[::, ::1] mat_factorized = np.empty(shape=(mat_n_rows, 5))

    # === Solving the system of equations ===

    # first, the matrix is factorized
    c_penta_factorize_algo1(
        mat_flat,
        mat_n_rows,
        mat_factorized,
    )

    # then, all the right-hand sides are solved
    for iter_col in range(rhs_n_cols):
        c_solve_penta_from_factorize_algo_1(
            mat_n_rows,
            mat_factorized,
            rhs[::, iter_col],
            result[::, iter_col],
        )

    return result


cdef void c_penta_factorize_algo1(
    double[::, ::] mat_flat,
    int64_t mat_n_rows,
    double[::, ::1] mat_factorized,
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
    [[   *          mu_0         *          al_0        be_0      ]
     [   *          mu_1        ga_1        al_1        be_1      ]
     [  e_2         mu_2        ga_2        al_2        be_2      ]
                                ...
     [  e_i         mu_i        ga_i        al_i        be_i  ]
     [  e_{n-3}     mu_{n-3}    ga_{n-3}    al_{n-3}    be_{n-3}  ]                                ...
     [  e_{n-2}     mu_{n-2}    ga_{n-2}    al_{n-2}      *       ]
     [  e_{n-1}     mu_{n-1}    ga_{n-1}      *           *       ]]
    ```

    where the entries marked with ``*`` are not used by design, but overwritten with
    zeros.

    """

    # === Variable declarations ===

    cdef int64_t iter_row
    cdef double mu_i, ga_i, e_i  # mu, gamma, e
    cdef double al_i, al_i_minus_1, al_i_plus_1  # alpha
    cdef double be_i, be_i_minus_1, be_i_plus_1  # beta

    # === Factorization ===

    # First row
    mu_i = mat_flat[2, 0]
    al_i_minus_1 = mat_flat[1, 0] / mu_i
    be_i_minus_1 = mat_flat[0, 0] / mu_i

    mat_factorized[0, 0] = 0.0
    mat_factorized[0, 1] = mu_i
    mat_factorized[0, 2] = 0.0
    mat_factorized[0, 3] = al_i_minus_1
    mat_factorized[0, 4] = be_i_minus_1

    # Second row
    ga_i = mat_flat[3, 1]
    mu_i = mat_flat[2, 1] - al_i_minus_1 * ga_i
    al_i = (mat_flat[1, 1] - be_i_minus_1 * ga_i) / mu_i
    be_i = mat_flat[0, 1] / mu_i

    mat_factorized[1, 0] = 0.0
    mat_factorized[1, 1] = mu_i
    mat_factorized[1, 2] = ga_i
    mat_factorized[1, 3] = al_i
    mat_factorized[1, 4] = be_i

    # Central rows
    for iter_row in range(2, mat_n_rows - 2):
        e_i = mat_flat[4, iter_row]
        ga_i = mat_flat[3, iter_row] - al_i_minus_1 * e_i
        mu_i = mat_flat[2, iter_row] - be_i_minus_1 * e_i - al_i * ga_i

        al_i_plus_1 = (mat_flat[1, iter_row] - be_i * ga_i) / mu_i
        al_i_minus_1 = al_i
        al_i = al_i_plus_1

        be_i_plus_1 = mat_flat[0, iter_row] / mu_i
        be_i_minus_1 = be_i
        be_i = be_i_plus_1

        mat_factorized[iter_row, 0] = e_i
        mat_factorized[iter_row, 1] = mu_i
        mat_factorized[iter_row, 2] = ga_i
        mat_factorized[iter_row, 3] = al_i
        mat_factorized[iter_row, 4] = be_i

    # Second to last row
    e_i = mat_flat[4, mat_n_rows - 2]
    ga_i = mat_flat[3, mat_n_rows - 2] - al_i_minus_1 * e_i
    mu_i = mat_flat[2, mat_n_rows - 2] - be_i_minus_1 * e_i - al_i * ga_i
    al_i_plus_1 = (mat_flat[1, mat_n_rows - 2] - be_i * ga_i) / mu_i

    mat_factorized[mat_n_rows - 2, 0] = e_i
    mat_factorized[mat_n_rows - 2, 1] = mu_i
    mat_factorized[mat_n_rows - 2, 2] = ga_i
    mat_factorized[mat_n_rows - 2, 3] = al_i_plus_1
    mat_factorized[mat_n_rows - 2, 4] = 0.0

    # Last Row
    e_i = mat_flat[4, mat_n_rows - 1]
    ga_i = mat_flat[3, mat_n_rows - 1] - al_i * e_i
    mu_i = mat_flat[2, mat_n_rows - 1] - be_i * e_i - al_i_plus_1 * ga_i

    mat_factorized[mat_n_rows - 1, 0] = e_i
    mat_factorized[mat_n_rows - 1, 1] = mu_i
    mat_factorized[mat_n_rows - 1, 2] = ga_i
    mat_factorized[mat_n_rows - 1, 3] = 0.0
    mat_factorized[mat_n_rows - 1, 4] = 0.0

    return


cdef void c_solve_penta_from_factorize_algo_1(
    int64_t mat_n_rows,
    double[::, ::1] mat_factorized,
    double[::] rhs_single,
    double[::] result_view,
):
    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the factorized
    unit upper triangular matrix ``U`` and the right-hand side ``b``.
    It overwrites the right-hand side ``b`` first with the transformed vector ``zeta``
    and then with the solution vector ``x`` for ``Ux = zeta``.

    """

    # === Variable declarations ===

    cdef int64_t iter_row
    cdef double ze_i, ze_i_minus_1, ze_i_plus_1

    # === Transformation ===

    # first, the right-hand side is transformed into the vector ``zeta``
    # First row

    ze_i_minus_1 = rhs_single[0] / mat_factorized[0, 1]
    result_view[0] = ze_i_minus_1

    # Second row
    ze_i = (rhs_single[1] - ze_i_minus_1 * mat_factorized[1, 2]) / mat_factorized[1, 1]
    result_view[1] = ze_i

    # Central rows
    for iter_row in range(2, mat_n_rows - 2):
        ze_i_plus_1 = (
            rhs_single[iter_row]
            - ze_i_minus_1 * mat_factorized[iter_row, 0]
            - ze_i * mat_factorized[iter_row, 2]
        ) / mat_factorized[iter_row, 1]
        ze_i_minus_1 = ze_i
        ze_i = ze_i_plus_1
        result_view[iter_row] = ze_i_plus_1

    # Second to last row
    ze_i_plus_1 = (
        rhs_single[mat_n_rows - 2]
        - ze_i_minus_1 * mat_factorized[mat_n_rows - 2, 0]
        - ze_i * mat_factorized[mat_n_rows - 2, 2]
    ) / mat_factorized[mat_n_rows - 2, 1]
    ze_i_minus_1 = ze_i
    ze_i = ze_i_plus_1
    result_view[mat_n_rows - 2] = ze_i_plus_1

    # Last row
    ze_i_plus_1 = (
        rhs_single[mat_n_rows - 1]
        - ze_i_minus_1 * mat_factorized[mat_n_rows - 1, 0]
        - ze_i * mat_factorized[mat_n_rows - 1, 2]
    ) / mat_factorized[mat_n_rows - 1, 1]
    result_view[mat_n_rows - 1] = ze_i_plus_1

    # === Backward substitution ===

    # The solution vector is calculated by backward substitution that overwrites the
    # right-hand side vector with the solution vector
    ze_i -= mat_factorized[mat_n_rows - 2, 3] * ze_i_plus_1
    result_view[mat_n_rows - 2] = ze_i

    for iter_row in range(mat_n_rows - 3, -1, -1):
        result_view[iter_row] -= (
            mat_factorized[iter_row, 3] * ze_i
            + mat_factorized[iter_row, 4] * ze_i_plus_1
        )
        ze_i_plus_1 = ze_i
        ze_i = result_view[iter_row]

    return


# === Solver Algorithm 2 ===


cdef double[::, ::] c_penta_solver2(double[::, ::] mat_flat, double[::, ::] rhs):
    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the matrix ``A`` and
    the right-hand side ``b`` by

    - factorizing the matrix ``A`` into auxiliary coefficients and a unit lower
        triangular matrix ``L``
    - transforming the right-hand side into a vector ``omega``
    - solving the system of equations ``Lx = omega`` by backward substitution

    """

    # Variable declarations

    cdef int64_t mat_n_rows = mat_flat.shape[1]
    cdef int64_t rhs_n_cols = rhs.shape[1]
    cdef int64_t iter_col
    cdef double[::, ::1] result = np.empty(shape=(mat_n_rows, rhs_n_cols))
    cdef double[::, ::1] mat_factorized = np.empty(shape=(mat_n_rows, 5))

    # first, the matrix is factorized
    c_penta_factorize_algo2(
        mat_flat,
        mat_n_rows,
        mat_factorized,
    )

    # then, all the right-hand sides are solved
    for iter_col in range(rhs_n_cols):
        c_solve_penta_from_factorize_algo_2(
            mat_n_rows,
            mat_factorized,
            rhs[::, iter_col],
            result[::, iter_col],
        )

    return result

cdef void c_penta_factorize_algo2(
    double[::, ::] mat_flat,
    int64_t mat_n_rows,
    double[::, ::1] mat_factorized,
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

    # === Variable declarations ===

    cdef int64_t iter_row
    cdef double ps_i, rho_i  # psi, rho
    cdef double si_i, si_i_minus_1, si_i_plus_1  # sigma
    cdef double phi_i, phi_i_minus_1, phi_i_plus_1  # phi

    # === Factorization ===

    # First row
    ps_i = mat_flat[2, mat_n_rows - 1]
    si_i_plus_1 = mat_flat[3, mat_n_rows - 1] / ps_i
    phi_i_plus_1 = mat_flat[4, mat_n_rows - 1] / ps_i

    mat_factorized[mat_n_rows - 1, 0] = phi_i_plus_1
    mat_factorized[mat_n_rows - 1, 1] = si_i_plus_1
    mat_factorized[mat_n_rows - 1, 2] = ps_i
    mat_factorized[mat_n_rows - 1, 3] = 0.0
    mat_factorized[mat_n_rows - 1, 4] = 0.0

    # Second row
    rho_i = mat_flat[1, mat_n_rows-2]
    ps_i = mat_flat[2, mat_n_rows-2] - si_i_plus_1 * rho_i
    si_i = (mat_flat[3, mat_n_rows-2] - phi_i_plus_1 * rho_i) / ps_i
    phi_i = mat_flat[4, mat_n_rows-2] / ps_i

    mat_factorized[mat_n_rows - 2, 0] = phi_i
    mat_factorized[mat_n_rows - 2, 1] = si_i
    mat_factorized[mat_n_rows - 2, 2] = ps_i
    mat_factorized[mat_n_rows - 2, 3] = rho_i
    mat_factorized[mat_n_rows - 2, 4] = 0.0

    # Central rows
    for iter_row in range(mat_n_rows-3, 1, -1):
        b_i = mat_flat[0, iter_row]
        rho_i = mat_flat[1, iter_row] - si_i_plus_1 * b_i
        ps_i = mat_flat[2, iter_row] - phi_i_plus_1 * b_i - si_i * rho_i
        si_i_minus_1 = (mat_flat[3, iter_row] - phi_i * rho_i) / ps_i
        si_i_plus_1 = si_i
        si_i = si_i_minus_1
        phi_i_minus_1 = mat_flat[4, iter_row] / ps_i
        phi_i_plus_1 = phi_i
        phi_i = phi_i_minus_1

        mat_factorized[iter_row, 0] = phi_i
        mat_factorized[iter_row, 1] = si_i
        mat_factorized[iter_row, 2] = ps_i
        mat_factorized[iter_row, 3] = rho_i
        mat_factorized[iter_row, 4] = b_i

    # Second to last row
    b_i = mat_flat[0, 1]
    rho_i = mat_flat[1, 1] - si_i_plus_1 * b_i
    ps_i = mat_flat[2, 1] - phi_i_plus_1 * b_i - si_i * rho_i
    si_i_minus_1 = (mat_flat[3, 1] - phi_i * rho_i) / ps_i
    si_i_plus_1 = si_i
    si_i = si_i_minus_1

    mat_factorized[1, 0] = 0.0
    mat_factorized[1, 1] = si_i
    mat_factorized[1, 2] = ps_i
    mat_factorized[1, 3] = rho_i
    mat_factorized[1, 4] = b_i

    # Last row
    b_i = mat_flat[0, 0]
    rho_i = mat_flat[1, 0] - si_i_plus_1 * b_i
    ps_i = mat_flat[2, 0] - phi_i * b_i - si_i * rho_i

    mat_factorized[0, 0] = 0.0
    mat_factorized[0, 1] = 0.0
    mat_factorized[0, 2] = ps_i
    mat_factorized[0, 3] = rho_i
    mat_factorized[0, 4] = b_i

    return


cdef void c_solve_penta_from_factorize_algo_2(
    int64_t mat_n_rows,
    double[::, ::1] mat_factorized,
    double[::] rhs_single,
    double[::] result_view,
):
    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the factorized
    unit lower triangular matrix ``L`` and the right-hand side ``b``.
    It overwrites the right-hand side ``b`` first with the transformed vector ``omega``
    and then with the solution vector ``x`` for ``Lx = omega``.

    """

    # === Variable declarations ===

    cdef int64_t iter_row
    cdef double om_i, om_i_minus_1, om_i_plus_1  # omega

    # === Transformation ===

    # first, the right-hand side is transformed into the vector ``omega``
    # First row
    om_i_plus_1 = rhs_single[mat_n_rows-1] / mat_factorized[mat_n_rows - 1, 2]
    result_view[mat_n_rows-1] = om_i_plus_1

    # Second row
    om_i = (
        rhs_single[mat_n_rows-2]
        - om_i_plus_1 * mat_factorized[mat_n_rows - 2, 3]
    ) / mat_factorized[mat_n_rows - 2, 2]
    result_view[mat_n_rows-2] = om_i

    # Central rows
    for iter_row in range(mat_n_rows-3, 1, -1):
        om_i_minus_1 = (
            rhs_single[iter_row]
            - om_i_plus_1 * mat_factorized[iter_row, 4]
            - om_i * mat_factorized[iter_row, 3]
        ) / mat_factorized[iter_row, 2]
        om_i_plus_1 = om_i
        om_i = om_i_minus_1
        result_view[iter_row] = om_i

    # Second to last row
    om_i_minus_1 = (
        rhs_single[1]
        - om_i_plus_1 * mat_factorized[1, 4]
        - om_i * mat_factorized[1, 3]
    ) / mat_factorized[1, 2]
    om_i_plus_1 = om_i
    om_i = om_i_minus_1
    result_view[1] = om_i

    # Last row
    om_i_minus_1 = (
        rhs_single[0]
        - om_i_plus_1 * mat_factorized[0, 4]
        - om_i * mat_factorized[0, 3]
    ) / mat_factorized[0, 2]
    result_view[0] = om_i_minus_1

    # === Forward substitution ===

    # The solution vector is calculated by forward substitution that overwrites the
    # right-hand side vector with the solution vector
    om_i -= mat_factorized[1, 1] * om_i_minus_1
    result_view[1] = om_i

    for iter_row in range(2, mat_n_rows):
        result_view[iter_row] = (
            result_view[iter_row]
            - mat_factorized[iter_row, 0] * om_i_minus_1
            - mat_factorized[iter_row, 1] * om_i
        )
        om_i_minus_1 = om_i
        om_i = result_view[iter_row]

    return
