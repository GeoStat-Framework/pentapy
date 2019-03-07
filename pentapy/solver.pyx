# cython: language_level=2
# -*- coding: utf-8 -*-
"""
This is a solver linear equation systems with a penta-diagonal matrix,
implemented in cython.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

cimport cython
cimport numpy as np


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
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
