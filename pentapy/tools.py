# -*- coding: utf-8 -*-
"""
The tools module of pentapy.

The following functions are provided

.. currentmodule:: pentapy.tools

.. autosummary::
   diag_indices
   shift_banded
   create_banded
   create_full
"""
import numpy as np


def diag_indices(n, offset=0):
    """
    Indices for the main or minor diagonals of a matrix.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array `a` with ``a.ndim == 2`` dimensions and shape
    (n, n).

    Parameters
    ----------
    n : int
      The size, along each dimension, of the arrays for which the returned
      indices can be used.
    offset : int, optional
      The diagonal offset.

    Returns
    -------
    idx : :class:`numpy.ndarray`
        row indices
    idy : :class:`numpy.ndarray`
        col indices

    """
    idx = np.arange(n - np.abs(offset)) - np.min((0, offset))
    idy = np.arange(n - np.abs(offset)) + np.max((0, offset))
    return idx, idy


def shift_banded(mat, up=2, low=2, col_to_row=True, copy=True):
    """Shift rows of a banded matrix.

    Either from column-wise to row-wise storage or vice versa.

    The Matrix has to be given as a flattend matrix.
    Either in a column-wise flattend form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Then use::

      col_to_row=True

    Or in a row-wise flattend form::

      [[Dup2[0]  Dup2[1]  Dup2[2]  ... Dup2[N-2]  0          0       ]
       [Dup1[0]  Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  0       ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [0        Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] Dlow1[N]]
       [0        0        Dlow2[2] ... Dlow2[N-2] Dlow2[N-2] Dlow2[N]]]

    Then use::

      col_to_row=False

    Dup1 and Dup2 are the first and second upper minor-diagonals
    and Dlow1 resp. Dlow2 are the lower ones.
    The number of upper and lower minor-diagonals can be altered.

    Parameters
    ----------
    mat : :class:`numpy.ndarray`
        The Matrix or the flattened Version of the pentadiagonal matrix.
    up : :class:`int`
        The number of upper minor-diagonals. Default: 2
    low : :class:`int`
        The number of lower minor-diagonals. Default: 2
    col_to_row : :class:`bool`, optional
        Shift from column-wise to row-wise storage or vice versa.
        Default: ``True``
    copy : :class:`bool`, optional
        Copy the input matrix or overwrite it. Default: ``True``

    Returns
    -------
    :class:`numpy.ndarray`
        Shifted bandend matrix
    """
    if copy:
        mat_flat = np.copy(mat)
    else:
        mat_flat = mat
    if col_to_row:
        for i in range(up):
            mat_flat[i, : -(up - i)] = mat_flat[i, (up - i) :]
            mat_flat[i, -(up - i) :] = 0
        for i in range(low):
            mat_flat[-i - 1, (low - i) :] = mat_flat[-i - 1, : -(low - i)]
            mat_flat[-i - 1, : (low - i)] = 0
    else:
        for i in range(up):
            mat_flat[i, (up - i) :] = mat_flat[i, : -(up - i)]
            mat_flat[i, : (up - i)] = 0
        for i in range(low):
            mat_flat[-i - 1, : -(low - i)] = mat_flat[-i - 1, (low - i) :]
            mat_flat[-i - 1, -(low - i) :] = 0
    return mat_flat


def create_banded(mat, up=2, low=2, col_wise=True, dtype=None):
    """Create a banded matrix from a given quadratic Matrix.

    The Matrix will to be returned as a flattend matrix.
    Either in a column-wise flattend form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Then use::

      col_wise=True

    Or in a row-wise flattend form::

      [[Dup2[0]  Dup2[1]  Dup2[2]  ... Dup2[N-2]  0          0       ]
       [Dup1[0]  Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  0       ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [0        Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] Dlow1[N]]
       [0        0        Dlow2[2] ... Dlow2[N-2] Dlow2[N-2] Dlow2[N]]]

    Then use::

      col_wise=False

    Dup1 and Dup2 or the first and second upper minor-diagonals and Dlow1 resp.
    Dlow2 are the lower ones. The number of upper and lower minor-diagonals can
    be altered.

    Parameters
    ----------
    mat : :class:`numpy.ndarray`
        The full (n x n) Matrix.
    up : :class:`int`
        The number of upper minor-diagonals. Default: 2
    low : :class:`int`
        The number of lower minor-diagonals. Default: 2
    col_wise : :class:`bool`, optional
        Use column-wise storage. If False, use row-wise storage.
        Default: ``True``

    Returns
    -------
    :class:`numpy.ndarray`
        Bandend matrix
    """
    mat = np.asanyarray(mat)
    if mat.ndim != 2:
        raise ValueError("create_banded: matrix has to be 2D")
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("create_banded: matrix has to be n x n")

    size = mat.shape[0]
    mat_flat = np.zeros((5, size), dtype=dtype)
    mat_flat[up, :] = mat.diagonal()

    if col_wise:
        for i in range(up):
            mat_flat[i, (up - i) :] = mat.diagonal(up - i)
        for i in range(low):
            mat_flat[-i - 1, : -(low - i)] = mat.diagonal(-(low - i))
    else:
        for i in range(up):
            mat_flat[i, : -(up - i)] = mat.diagonal(up - i)
        for i in range(low):
            mat_flat[-i - 1, (low - i) :] = mat.diagonal(-(low - i))
    return mat_flat


def create_full(mat, up=2, low=2, col_wise=True):
    """Create a (n x n) Matrix from a given banded matrix.

    The given Matrix has to be a flattend matrix.
    Either in a column-wise flattend form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Then use::

      col_wise=True

    Or in a row-wise flattend form::

      [[Dup2[0]  Dup2[1]  Dup2[2]  ... Dup2[N-2]  0          0       ]
       [Dup1[0]  Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  0       ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [0        Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] Dlow1[N]]
       [0        0        Dlow2[2] ... Dlow2[N-2] Dlow2[N-2] Dlow2[N]]]

    Then use::

      col_wise=False

    Dup1 and Dup2 or the first and second upper minor-diagonals and Dlow1 resp.
    Dlow2 are the lower ones. The number of upper and lower minor-diagonals can
    be altered.

    Parameters
    ----------
    mat : :class:`numpy.ndarray`
        The flattened Matrix.
    up : :class:`int`
        The number of upper minor-diagonals. Default: 2
    low : :class:`int`
        The number of lower minor-diagonals. Default: 2
    col_wise : :class:`bool`, optional
        Input is in column-wise storage. If False, use as row-wise storage.
        Default: ``True``

    Returns
    -------
    :class:`numpy.ndarray`
        Full matrix.
    """
    mat = np.asanyarray(mat)
    if mat.ndim != 2:
        raise ValueError("create_full: matrix has to be 2D")
    if mat.shape[0] != up + low + 1:
        raise ValueError("create_full: matrix has wrong count of bands")
    if mat.shape[1] < max(up, low) + 1:
        raise ValueError("create_full: matrix has to few information")
    size = mat.shape[1]
    mat_full = np.diag(mat[up])
    if col_wise:
        for i in range(up):
            mat_full[diag_indices(size, up - i)] = mat[i, (up - i) :]
        for i in range(low):
            mat_full[diag_indices(size, -(low - i))] = mat[
                -i - 1, : -(low - i)
            ]
    else:
        for i in range(up):
            mat_full[diag_indices(size, up - i)] = mat[i, : -(up - i)]
        for i in range(low):
            mat_full[diag_indices(size, -(low - i))] = mat[-i - 1, (low - i) :]
    return mat_full


def _check_penta(mat):
    if mat.ndim != 2:
        raise ValueError("pentapy: matrix has to be 2D")
    if mat.shape[0] != 5:
        raise ValueError("pentapy: matrix needs 5 bands")
    if mat.shape[1] < 3:
        raise ValueError("pentapy: matrix needs at least 3 rows")
