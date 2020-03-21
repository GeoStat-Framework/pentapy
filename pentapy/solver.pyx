#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# -*- coding: utf-8 -*-
"""
This is a solver linear equation systems with a penta-diagonal matrix,
implemented in cython.
"""
import numpy as np

cimport cython
cimport numpy as np


def penta_solver1(double[:,:] mat_flat, double[:] rhs):

    cdef int mat_j = mat_flat.shape[1]

    cdef double[:] result = np.zeros(mat_j)

    cdef double[:] al = np.zeros(mat_j)
    cdef double[:] be = np.zeros(mat_j)
    cdef double[:] ze = np.zeros(mat_j)
    cdef double[:] ga = np.zeros(mat_j)
    cdef double[:] mu = np.zeros(mat_j)

    cdef int i

    mu[0] = mat_flat[2, 0]
    al[0] = mat_flat[1, 0] / mu[0]
    be[0] = mat_flat[0, 0] / mu[0]
    ze[0] = rhs[0] / mu[0]

    ga[1] = mat_flat[3, 1]
    mu[1] = mat_flat[2, 1] - al[0] * ga[1]
    al[1] = (mat_flat[1, 1] - be[0] * ga[1]) / mu[1]
    be[1] = mat_flat[0, 1] / mu[1]
    ze[1] = (rhs[1] - ze[0] * ga[1]) / mu[1]

    for i in range(2, mat_j-2):
        ga[i] = mat_flat[3, i] - al[i-2] * mat_flat[4, i]
        mu[i] = mat_flat[2, i] - be[i-2] * mat_flat[4, i] - al[i-1] * ga[i]
        al[i] = (mat_flat[1, i] - be[i-1] * ga[i]) / mu[i]
        be[i] = mat_flat[0, i] / mu[i]
        ze[i] = (rhs[i] - ze[i-2] * mat_flat[4, i] - ze[i-1] * ga[i]) / mu[i]

    ga[mat_j-2] = mat_flat[3, mat_j-2] - al[mat_j-4] * mat_flat[4, mat_j-2]
    mu[mat_j-2] = mat_flat[2, mat_j-2] - be[mat_j-4] * mat_flat[4, mat_j-2] - al[mat_j-3] * ga[mat_j-2]
    al[mat_j-2] = (mat_flat[1, mat_j-2] - be[mat_j-3] * ga[mat_j-2]) / mu[mat_j-2]

    ga[mat_j-1] = mat_flat[3, mat_j-1] - al[mat_j-3] * mat_flat[4, mat_j-1]
    mu[mat_j-1] = mat_flat[2, mat_j-1] - be[mat_j-3] * mat_flat[4, mat_j-1] - al[mat_j-2] * ga[mat_j-1]

    ze[mat_j-2] = (rhs[mat_j-2] - ze[mat_j-4] * mat_flat[4, mat_j-2] - ze[mat_j-3] * ga[mat_j-2]) / mu[mat_j-2]
    ze[mat_j-1] = (rhs[mat_j-1] - ze[mat_j-3] * mat_flat[4, mat_j-1] - ze[mat_j-2] * ga[mat_j-1]) / mu[mat_j-1]

    # Backward substitution
    result[mat_j-1] = ze[mat_j-1]
    result[mat_j-2] = ze[mat_j-2] - al[mat_j-2] * result[mat_j-1]

    for i in range(mat_j-3, -1, -1):
        result[i] = ze[i] - al[i] * result[i+1] - be[i] * result[i+2]

    return np.asarray(result)


def penta_solver2(double[:,:] mat_flat, double[:] rhs):

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

    return np.asarray(result)
