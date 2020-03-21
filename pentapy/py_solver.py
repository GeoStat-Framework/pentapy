# -*- coding: utf-8 -*-
"""
A solver for linear equation systems with a penta-diagonal matrix.

This is the python implementation.

The following functions are provided

.. currentmodule:: pentapy.py_solver

.. autosummary::
   penta_solver1
   penta_solver2

"""
import numpy as np


def penta_solver1(mat_flat, rhs):
    """
    Solver for a pentadiagonal system.

    The Matrix has to be given in a row-wise flattend form::

      [[Dup2[0]  Dup2[1]  Dup2[2]  ... Dup2[N-2]  0          0       ]
       [Dup1[0]  Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  0       ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [0        Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] Dlow1[N]]
       [0        0        Dlow2[2] ... Dlow2[N-2] Dlow2[N-2] Dlow2[N]]]

    This routine was presented in [Askar2015]_.

    Parameters
    ----------
    mat_flat : :class:`numpy.ndarray`
        The flattened Version of the pentadiagonal matrix.
    rhs : :class:`numpy.ndarray`
        The right hand side of the equation system.

    Returns
    -------
    result : :class:`numpy.ndarray`
        Result of the equation system

    References
    ----------
    .. [Askar2015] S. S. Askar and A. A. Karawia,
       ''On Solving Pentadiagonal Linear Systems via Transformations''
       Mathematical Problems in Engineering, vol. 2015, Article ID 232456,
       9 pages, 2015. https://doi.org/10.1155/2015/232456.
    """
    mat_j = mat_flat.shape[1]
    result = np.zeros(mat_j)

    al = np.zeros(mat_j)
    be = np.zeros(mat_j)
    ze = np.zeros(mat_j)
    ga = np.zeros(mat_j)
    mu = np.zeros(mat_j)

    mu[0] = mat_flat[2, 0]
    al[0] = mat_flat[1, 0] / mu[0]
    be[0] = mat_flat[0, 0] / mu[0]
    ze[0] = rhs[0] / mu[0]

    ga[1] = mat_flat[3, 1]
    mu[1] = mat_flat[2, 1] - al[0] * ga[1]
    al[1] = (mat_flat[1, 1] - be[0] * ga[1]) / mu[1]
    be[1] = mat_flat[0, 1] / mu[1]
    ze[1] = (rhs[1] - ze[0] * ga[1]) / mu[1]

    for i in range(2, mat_j - 2):
        ga[i] = mat_flat[3, i] - al[i - 2] * mat_flat[4, i]
        mu[i] = mat_flat[2, i] - be[i - 2] * mat_flat[4, i] - al[i - 1] * ga[i]
        al[i] = (mat_flat[1, i] - be[i - 1] * ga[i]) / mu[i]
        be[i] = mat_flat[0, i] / mu[i]
        ze[i] = (rhs[i] - ze[i - 2] * mat_flat[4, i] - ze[i - 1] * ga[i]) / mu[
            i
        ]

    ga[mat_j - 2] = (
        mat_flat[3, mat_j - 2] - al[mat_j - 4] * mat_flat[4, mat_j - 2]
    )
    mu[mat_j - 2] = (
        mat_flat[2, mat_j - 2]
        - be[mat_j - 4] * mat_flat[4, mat_j - 2]
        - al[mat_j - 3] * ga[mat_j - 2]
    )
    al[mat_j - 2] = (
        mat_flat[1, mat_j - 2] - be[mat_j - 3] * ga[mat_j - 2]
    ) / mu[mat_j - 2]

    ga[mat_j - 1] = (
        mat_flat[3, mat_j - 1] - al[mat_j - 3] * mat_flat[4, mat_j - 1]
    )
    mu[mat_j - 1] = (
        mat_flat[2, mat_j - 1]
        - be[mat_j - 3] * mat_flat[4, mat_j - 1]
        - al[mat_j - 2] * ga[mat_j - 1]
    )

    ze[mat_j - 2] = (
        rhs[mat_j - 2]
        - ze[mat_j - 4] * mat_flat[4, mat_j - 2]
        - ze[mat_j - 3] * ga[mat_j - 2]
    ) / mu[mat_j - 2]
    ze[mat_j - 1] = (
        rhs[mat_j - 1]
        - ze[mat_j - 3] * mat_flat[4, mat_j - 1]
        - ze[mat_j - 2] * ga[mat_j - 1]
    ) / mu[mat_j - 1]

    # Backward substitution
    result[mat_j - 1] = ze[mat_j - 1]
    result[mat_j - 2] = ze[mat_j - 2] - al[mat_j - 2] * result[mat_j - 1]

    for i in range(mat_j - 3, -1, -1):
        result[i] = ze[i] - al[i] * result[i + 1] - be[i] * result[i + 2]

    return np.asarray(result)


def penta_solver2(mat_flat, rhs):
    """
    Solver for a pentadiagonal system.

    The Matrix has to be given in a row-wise flattend form::

      [[Dup2[0]  Dup2[1]  Dup2[2]  ... Dup2[N-2]  0          0       ]
       [Dup1[0]  Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  0       ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [0        Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] Dlow1[N]]
       [0        0        Dlow2[2] ... Dlow2[N-2] Dlow2[N-2] Dlow2[N]]]

    This routine was presented in [Askar2015]_.

    Parameters
    ----------
    mat_flat : :class:`numpy.ndarray`
        The flattened Version of the pentadiagonal matrix.
    rhs : :class:`numpy.ndarray`
        The right hand side of the equation system.

    Returns
    -------
    result : :class:`numpy.ndarray`
        Result of the equation system

    References
    ----------
    .. [Askar2015] S. S. Askar and A. A. Karawia,
       ''On Solving Pentadiagonal Linear Systems via Transformations''
       Mathematical Problems in Engineering, vol. 2015, Article ID 232456,
       9 pages, 2015. https://doi.org/10.1155/2015/232456.
    """
    mat_j = mat_flat.shape[1]

    result = np.zeros(mat_j)

    ps = np.zeros(mat_j)  # psi
    si = np.zeros(mat_j)  # sigma
    ph = np.zeros(mat_j)  # phi
    ro = np.zeros(mat_j)  # rho
    we = np.zeros(mat_j)  # w

    ps[mat_j - 1] = mat_flat[2, mat_j - 1]
    si[mat_j - 1] = mat_flat[3, mat_j - 1] / ps[mat_j - 1]
    ph[mat_j - 1] = mat_flat[4, mat_j - 1] / ps[mat_j - 1]
    we[mat_j - 1] = rhs[mat_j - 1] / ps[mat_j - 1]

    ro[mat_j - 2] = mat_flat[1, mat_j - 2]
    ps[mat_j - 2] = mat_flat[2, mat_j - 2] - si[mat_j - 1] * ro[mat_j - 2]
    si[mat_j - 2] = (
        mat_flat[3, mat_j - 2] - ph[mat_j - 1] * ro[mat_j - 2]
    ) / ps[mat_j - 2]
    ph[mat_j - 2] = mat_flat[4, mat_j - 2] / ps[mat_j - 2]
    we[mat_j - 2] = (rhs[mat_j - 2] - we[mat_j - 1] * ro[mat_j - 2]) / ps[
        mat_j - 2
    ]

    for i in range(mat_j - 3, 1, -1):
        ro[i] = mat_flat[1, i] - si[i + 2] * mat_flat[0, i]
        ps[i] = mat_flat[2, i] - ph[i + 2] * mat_flat[0, i] - si[i + 1] * ro[i]
        si[i] = (mat_flat[3, i] - ph[i + 1] * ro[i]) / ps[i]
        ph[i] = mat_flat[4, i] / ps[i]
        we[i] = (rhs[i] - we[i + 2] * mat_flat[0, i] - we[i + 1] * ro[i]) / ps[
            i
        ]

    ro[1] = mat_flat[1, 1] - si[3] * mat_flat[0, 1]
    ps[1] = mat_flat[2, 1] - ph[3] * mat_flat[0, 1] - si[2] * ro[1]
    si[1] = (mat_flat[3, 1] - ph[2] * ro[1]) / ps[1]

    ro[0] = mat_flat[1, 0] - si[2] * mat_flat[0, 0]
    ps[0] = mat_flat[2, 0] - ph[2] * mat_flat[0, 0] - si[1] * ro[0]

    we[1] = (rhs[1] - we[3] * mat_flat[0, 1] - we[2] * ro[1]) / ps[1]
    we[0] = (rhs[0] - we[2] * mat_flat[0, 0] - we[1] * ro[0]) / ps[0]

    # Backward substitution
    result[0] = we[0]
    result[1] = we[1] - si[1] * result[0]

    for i in range(2, mat_j):
        result[i] = we[i] - si[i] * result[i - 1] - ph[i] * result[i - 2]

    return np.asarray(result)
