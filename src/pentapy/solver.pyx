# cython: language_level=3, boundscheck=True, wraparound=False, cdivision=False

"""
This is a solver linear equation systems with a penta-diagonal matrix,
implemented in Cython.

"""

# Imports

import numpy as np

cimport numpy as np
from libc.stdint cimport int64_t, uint64_t


# === Main Python Interface ===


def penta_solver1(double[:, :] mat_flat, double[:, :] rhs):
    return np.asarray(c_penta_solver1(mat_flat, rhs))


def penta_solver2(double[:, :] mat_flat, double[:] rhs):
    return np.asarray(c_penta_solver2(mat_flat, rhs))


# === Solver Algorithm 1 ===


cdef double[:, :] c_penta_solver1(double[:, :] mat_flat, double[:, :] rhs):
    """
    Solves the pentadiagonal system of equations ``Ax = b`` with the matrix ``A`` and
    the right-hand side ``b`` by

    - factorizing the matrix ``A`` into auxiliary coefficients and a unit upper
        triangular matrix ``U``
    - transforming the right-hand side into a vector ``zeta``
    - solving the system of equations ``Ux = zeta`` by backward substitution

    """

    cdef uint64_t mat_n_rows = mat_flat.shape[1]
    cdef uint64_t rhs_n_cols = rhs.shape[1]
    cdef uint64_t iter_col
    cdef double[::, ::1] result = np.empty(shape=(mat_n_rows, rhs_n_cols))
    cdef double[::, ::1] mat_factorized = np.empty(shape=(mat_n_rows, 5))

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
    double[:, :] mat_flat,
    uint64_t mat_n_rows,
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
    [[  *           mu_0        *           al_0        be_0  ]
     [  *           mu_1        ga_1        al_1        be_1  ]
     [  e_2         mu_2        ga_2        al_2        be_2  ]
                                ...
     [  e_i         mu_i        ga_i        al_i        be_i  ]
                                ...
     [  e_{n-2}     mu_{n-2}    ga_{n-2}    al_{n-2}    *     ]
     [  e_{n-1}     mu_{n-1}    ga_{n-1}    *           *     ]]
    ```

    where the entries marked with ``*`` are not used by design, but overwritten with
    zeros.

    """

    # === Variable declarations ===

    cdef uint64_t iter_row
    cdef double mu_i, ga_i, e_i # mu, gamma, e
    cdef double al_i, al_i_minus_1, al_i_plus_1 # alpha
    cdef double be_i, be_i_minus_1, be_i_plus_1 # beta

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
    for iter_row in range(2, mat_n_rows-2):
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
    e_i = mat_flat[4, mat_n_rows-2]
    ga_i = mat_flat[3, mat_n_rows-2] - al_i_minus_1 * e_i
    mu_i = mat_flat[2, mat_n_rows-2] - be_i_minus_1 * e_i - al_i * ga_i
    al_i_plus_1 = (mat_flat[1, mat_n_rows-2] - be_i * ga_i) / mu_i

    mat_factorized[mat_n_rows-2, 0] = e_i
    mat_factorized[mat_n_rows-2, 1] = mu_i
    mat_factorized[mat_n_rows-2, 2] = ga_i
    mat_factorized[mat_n_rows-2, 3] = al_i_plus_1
    mat_factorized[mat_n_rows-2, 4] = 0.0

    # Last Row
    e_i = mat_flat[4, mat_n_rows-1]
    ga_i = mat_flat[3, mat_n_rows-1] - al_i * e_i
    mu_i = mat_flat[2, mat_n_rows-1] - be_i * e_i - al_i_plus_1 * ga_i

    mat_factorized[mat_n_rows-1, 0] = e_i
    mat_factorized[mat_n_rows-1, 1] = mu_i
    mat_factorized[mat_n_rows-1, 2] = ga_i
    mat_factorized[mat_n_rows-1, 3] = 0.0
    mat_factorized[mat_n_rows-1, 4] = 0.0

    return


cdef void c_solve_penta_from_factorize_algo_1(
    uint64_t mat_n_rows,
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
    for iter_row in range(2, mat_n_rows-2):
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
        rhs_single[mat_n_rows-2]
        - ze_i_minus_1 * mat_factorized[mat_n_rows-2, 0]
        - ze_i * mat_factorized[mat_n_rows-2, 2]
    ) / mat_factorized[mat_n_rows-2, 1]
    ze_i_minus_1 = ze_i
    ze_i = ze_i_plus_1
    result_view[mat_n_rows-2] = ze_i_plus_1

    # Last row
    ze_i_plus_1 = (
        rhs_single[mat_n_rows-1]
        - ze_i_minus_1 * mat_factorized[mat_n_rows-1, 0]
        - ze_i * mat_factorized[mat_n_rows-1, 2]
    ) / mat_factorized[mat_n_rows-1, 1]
    result_view[mat_n_rows-1] = ze_i_plus_1

    # === Backward substitution ===

    # The solution vector is calculated by backward substitution that overwrites the
    # right-hand side vector with the solution vector
    ze_i -= mat_factorized[mat_n_rows-2, 3] * ze_i_plus_1
    result_view[mat_n_rows-2] = ze_i

    for iter_row in range(mat_n_rows-3, -1, -1):
        result_view[iter_row] -= (
            mat_factorized[iter_row, 3] * ze_i
            + mat_factorized[iter_row, 4] * ze_i_plus_1
        )
        ze_i_plus_1 = ze_i
        ze_i = result_view[iter_row]

    return


# === Solver Algorithm 2 ===


cdef double[:] c_penta_solver2(double[:, :] mat_flat, double[:] rhs):

    cdef int mat_j = mat_flat.shape[1]

    cdef double[:] result = np.zeros(mat_j)

    cdef double[:] ps = np.zeros(mat_j)  # psi
    cdef double[:] si = np.zeros(mat_j)  # sigma
    cdef double[:] ph = np.zeros(mat_j)  # phi
    cdef double[:] ro = np.zeros(mat_j)  # rho
    cdef double[:] we = np.zeros(mat_j)  # w

    cdef int i

    ps[mat_j-1] = mat_flat[2, mat_j-1]
    si[mat_j-1] = mat_flat[3, mat_j-1] / ps[mat_j-1]
    ph[mat_j-1] = mat_flat[4, mat_j-1] / ps[mat_j-1]
    we[mat_j-1] = rhs[mat_j-1] / ps[mat_j-1]

    ro[mat_j-2] = mat_flat[1, mat_j-2]
    ps[mat_j-2] = mat_flat[2, mat_j-2] - si[mat_j-1] * ro[mat_j-2]
    si[mat_j-2] = (mat_flat[3, mat_j-2] - ph[mat_j-1] * ro[mat_j-2]) / ps[mat_j-2]
    ph[mat_j-2] = mat_flat[4, mat_j-2] / ps[mat_j-2]
    we[mat_j-2] = (rhs[mat_j-2] - we[mat_j-1] * ro[mat_j-2]) / ps[mat_j-2]

    for i in range(mat_j-3, 1, -1):
        ro[i] = mat_flat[1, i] - si[i+2] * mat_flat[0, i]
        ps[i] = mat_flat[2, i] - ph[i+2] * mat_flat[0, i] - si[i+1] * ro[i]
        si[i] = (mat_flat[3, i] - ph[i+1] * ro[i]) / ps[i]
        ph[i] = mat_flat[4, i] / ps[i]
        we[i] = (rhs[i] - we[i+2] * mat_flat[0, i] - we[i+1] * ro[i]) / ps[i]

    ro[1] = mat_flat[1, 1] - si[3] * mat_flat[0, 1]
    ps[1] = mat_flat[2, 1] - ph[3] * mat_flat[0, 1] - si[2] * ro[1]
    si[1] = (mat_flat[3, 1] - ph[2] * ro[1]) / ps[1]

    ro[0] = mat_flat[1, 0] - si[2] * mat_flat[0, 0]
    ps[0] = mat_flat[2, 0] - ph[2] * mat_flat[0, 0] - si[1] * ro[0]

    we[1] = (rhs[1] - we[3] * mat_flat[0, 1] - we[2] * ro[1]) / ps[1]
    we[0] = (rhs[0] - we[2] * mat_flat[0, 0] - we[1] * ro[0]) / ps[0]

    # Foreward substitution
    result[0] = we[0]
    result[1] = we[1] - si[1] * result[0]

    for i in range(2, mat_j):
        result[i] = we[i] - si[i] * result[i-1] - ph[i] * result[i-2]

    return result
