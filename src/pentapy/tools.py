"""
The tools module of pentapy.

The following functions are provided

.. currentmodule:: pentapy.tools

.. autosummary::
   :toctree:

   diag_indices
   shift_banded
   create_banded
   create_full
"""

# === Imports ===

from typing import Optional, Tuple, Type

import numpy as np

# === Functions ===


def diag_indices(
    n: int,
    offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Get indices for the main or minor diagonals of a matrix.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array `a` with ``a.ndim == 2`` dimensions and shape
    (n, n).

    Parameters
    ----------
    n : :class:`int`
      The size, along each dimension, of the arrays for which the returned
      indices can be used.
    offset : :class:`int`, optional
      The diagonal offset. Default: 0

    Returns
    -------
    idx : :class:`numpy.ndarray` of shape (n - abs(offset),)
        row indices
    idy : :class:`numpy.ndarray` of shape (n - abs(offset),)
        col indices

    """
    idx = np.arange(n - np.abs(offset)) - np.min((0, offset))
    idy = np.arange(n - np.abs(offset)) + np.max((0, offset))
    return idx, idy


def shift_banded(
    mat: np.ndarray,
    up: int = 2,
    low: int = 2,
    col_to_row: bool = True,
    copy: bool = True,
) -> np.ndarray:
    """
    Shift rows of a banded matrix.

    Either from column-wise to row-wise storage or vice versa.

    The Matrix has to be given as a flattened matrix.
    Either in a column-wise flattened form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Then use::

      col_to_row=True

    Or in a row-wise flattened form::

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
    mat : :class:`numpy.ndarray` of shape (5, n)
        The Matrix or the flattened Version of the pentadiagonal matrix.
    up : :class:`int`, optional
        The number of upper minor-diagonals. Default: 2
    low : :class:`int`, optional
        The number of lower minor-diagonals. Default: 2
    col_to_row : :class:`bool`, optional
        Shift from column-wise to row-wise storage or vice versa.
        Default: ``True``
    copy : :class:`bool`, optional
        Copy the input matrix or overwrite it. Default: ``True``

    Returns
    -------
    :class:`numpy.ndarray` of shape (5, n)
        Shifted banded matrix

    """

    # first, the matrix is copied if required
    if copy:
        mat_flat = np.copy(mat)
    else:
        mat_flat = mat

    # then, the shifting is performed
    # Case 1: Column-wise to row-wise
    if col_to_row:
        for i in range(up):
            mat_flat[i, : -(up - i)] = mat_flat[i, (up - i) :]
            mat_flat[i, -(up - i) :] = 0
        for i in range(low):
            mat_flat[-i - 1, (low - i) :] = mat_flat[-i - 1, : -(low - i)]
            mat_flat[-i - 1, : (low - i)] = 0

        return mat_flat

    # Case 2: Row-wise to column-wise
    for i in range(up):
        mat_flat[i, (up - i) :] = mat_flat[i, : -(up - i)]
        mat_flat[i, : (up - i)] = 0
    for i in range(low):
        mat_flat[-i - 1, : -(low - i)] = mat_flat[-i - 1, (low - i) :]
        mat_flat[-i - 1, -(low - i) :] = 0

    return mat_flat


def create_banded(
    mat: np.ndarray,
    up: int = 2,
    low: int = 2,
    col_wise: bool = True,
    dtype: Optional[Type] = None,
) -> np.ndarray:
    """
    Create a banded matrix from a given square Matrix.

    The Matrix will to be returned as a flattened matrix.
    Either in a column-wise flattened form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Then use::

      col_wise=True

    Or in a row-wise flattened form::

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
    mat : :class:`numpy.ndarray` of shape (n, n)
        The full (n x n) Matrix.
    up : :class:`int`, optional
        The number of upper minor-diagonals. Default: 2
    low : :class:`int`, optional
        The number of lower minor-diagonals. Default: 2
    col_wise : :class:`bool`, optional
        Use column-wise storage. If False, use row-wise storage.
        Default: ``True``
    dtype : :class:`type` or ``None``, optional
        The data type of the returned matrix. If ``None``, the data type of the
        input matrix is preserved. Default: ``None``

    Returns
    -------
    :class:`numpy.ndarray` of shape (5, n)
        Banded matrix

    """

    # first, the matrix is checked
    mat = np.asanyarray(mat)
    if mat.ndim != 2:
        msg = f"create_banded: matrix has to be 2D, got {mat.ndim}D"
        raise ValueError(msg)

    if mat.shape[0] != mat.shape[1]:
        msg = (
            f"create_banded: matrix has to be n x n, "
            f"got {mat.shape[0]} x {mat.shape[1]}"
        )
        raise ValueError(msg)

    # then, the matrix is created
    dtype = mat.dtype if dtype is None else dtype
    size = mat.shape[0]
    mat_flat = np.zeros(shape=(5, size), dtype=dtype)
    mat_flat[up, :] = mat.diagonal()

    # Case 1: Column-wise storage
    if col_wise:
        for i in range(up):
            mat_flat[i, (up - i) :] = mat.diagonal(up - i)
        for i in range(low):
            mat_flat[-i - 1, : -(low - i)] = mat.diagonal(-(low - i))

        return mat_flat

    # Case 2: Row-wise storage
    for i in range(up):
        mat_flat[i, : -(up - i)] = mat.diagonal(up - i)
    for i in range(low):
        mat_flat[-i - 1, (low - i) :] = mat.diagonal(-(low - i))

    return mat_flat


def create_full(
    mat: np.ndarray,
    up: int = 2,
    low: int = 2,
    col_wise: bool = True,
) -> np.ndarray:
    """Create a (n x n) Matrix from a given banded matrix.

    The given Matrix has to be a flattened matrix.
    Either in a column-wise flattened form::

      [[0        0        Dup2[2]  ... Dup2[N-2]  Dup2[N-1]  Dup2[N] ]
       [0        Dup1[1]  Dup1[2]  ... Dup1[N-2]  Dup1[N-1]  Dup1[N] ]
       [Diag[0]  Diag[1]  Diag[2]  ... Diag[N-2]  Diag[N-1]  Diag[N] ]
       [Dlow1[0] Dlow1[1] Dlow1[2] ... Dlow1[N-2] Dlow1[N-1] 0       ]
       [Dlow2[0] Dlow2[1] Dlow2[2] ... Dlow2[N-2] 0          0       ]]

    Then use::

      col_wise=True

    Or in a row-wise flattened form::

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
    mat : :class:`numpy.ndarray` of shape (5, n)
        The flattened Matrix.
    up : :class:`int`, optional
        The number of upper minor-diagonals. Default: 2
    low : :class:`int`, optional
        The number of lower minor-diagonals. Default: 2
    col_wise : :class:`bool`, optional
        Input is in column-wise storage. If False, use as row-wise storage.
        Default: ``True``

    Returns
    -------
    :class:`numpy.ndarray` of shape (n, n)
        Full matrix.

    """

    # first, the matrix is checked
    mat = np.asanyarray(mat)
    if mat.ndim != 2:
        msg = f"create_full: matrix has to be 2D, got {mat.ndim}D"
        raise ValueError(msg)

    if mat.shape[0] != up + low + 1:
        msg = (
            f"create_full: matrix has wrong count of bands, required "
            f"{up} + {low} + 1 = {up + low + 1}, got {mat.shape[0]} bands"
        )
        raise ValueError(msg)

    if mat.shape[1] < max(up, low) + 1:
        msg = (
            f"create_full: matrix has to few information, required "
            f"{max(up, low) + 1} columns, got {mat.shape[1]} columns"
        )
        raise ValueError(msg)

    # then, the matrix is created
    size = mat.shape[1]
    mat_full = np.diag(mat[up])

    # Case 1: Column-wise storage
    if col_wise:
        for i in range(up):
            mat_full[diag_indices(size, up - i)] = mat[i, (up - i) :]
        for i in range(low):
            mat_full[diag_indices(size, -(low - i))] = mat[-i - 1, : -(low - i)]

        return mat_full

    # Case 2: Row-wise storage
    for i in range(up):
        mat_full[diag_indices(size, up - i)] = mat[i, : -(up - i)]
    for i in range(low):
        mat_full[diag_indices(size, -(low - i))] = mat[-i - 1, (low - i) :]

    return mat_full


def _check_penta(mat: np.ndarray) -> None:
    if mat.ndim != 2:
        msg = f"pentapy: matrix has to be 2D, got {mat.ndim}D"
        raise ValueError(msg)
    if mat.shape[0] != 5:
        msg = f"pentapy: matrix needs 5 bands, got {mat.shape[0]} bands"
        raise ValueError(msg)
    if mat.shape[1] < 3:
        msg = f"pentapy: matrix needs at least 3 rows, got {mat.shape[1]} rows"
        raise ValueError(msg)
