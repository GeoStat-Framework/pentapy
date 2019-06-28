Tutorial 1: Solving a pentadiagonal system
==========================================

Pentadiagonal systems arise in many areas of science and engineering,
for example in solving differential equations with a finite difference sceme.


Theoretical Background
----------------------

A pentadiagonal system is given by the equation: :math:`M\cdot X = Y`, where
:math:`M` is a quadratic :math:`n\times n` matrix given by:

.. math::

    M=\left(\begin{matrix}d_{1} & d_{1}^{\left(1\right)} & d_{1}^{\left(2\right)} & 0 & \cdots & \cdots & \cdots & \cdots & \cdots & 0\\
    d_{2}^{\left(-1\right)} & d_{2} & d_{2}^{\left(1\right)} & d_{2}^{\left(2\right)} & 0 & \cdots & \cdots & \cdots & \cdots & 0\\
    d_{3}^{\left(-2\right)} & d_{3}^{\left(-1\right)} & d_{3} & d_{3}^{\left(1\right)} & d_{3}^{\left(2\right)} & 0 & \cdots & \cdots & \cdots & 0\\
    0 & d_{4}^{\left(-2\right)} & d_{4}^{\left(-1\right)} & d_{4} & d_{4}^{\left(1\right)} & d_{4}^{\left(2\right)} & 0 & \cdots & \cdots & 0\\
    \vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots\\
    \vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots\\
    \vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots\\
    0 & \cdots & \cdots & \cdots & 0 & d_{n-2}^{\left(-2\right)} & d_{n-2}^{\left(-1\right)} & d_{n-2} & d_{n-2}^{\left(1\right)} & d_{n-2}^{\left(2\right)}\\
    0 & \cdots & \cdots & \cdots & \cdots & 0 & d_{n-1}^{\left(-2\right)} & d_{n-1}^{\left(-1\right)} & d_{n-1} & d_{n-1}^{\left(1\right)}\\
    0 & \cdots & \cdots & \cdots & \cdots & \cdots & 0 & d_{n}^{\left(-2\right)} & d_{n}^{\left(-1\right)} & d_{n}
    \end{matrix}\right)

The aim is now, to solve this equation for :math:`X`.


Memory efficient storage
------------------------

To store a pentadiagonal matrix memory efficient, there are two options:

1. row-wise storage:

.. math::

    M_{\mathrm{row}}=\left(\begin{matrix}d_{1}^{\left(2\right)} & d_{2}^{\left(2\right)} & d_{3}^{\left(2\right)} & \cdots & d_{n-2}^{\left(2\right)} & 0 & 0\\
    d_{1}^{\left(1\right)} & d_{2}^{\left(1\right)} & d_{3}^{\left(1\right)} & \cdots & d_{n-2}^{\left(1\right)} & d_{n-1}^{\left(1\right)} & 0\\
    d_{1} & d_{2} & d_{3} & \cdots & d_{n-2} & d_{n-1} & d_{n}\\
    0 & d_{2}^{\left(-1\right)} & d_{3}^{\left(-1\right)} & \cdots & d_{n-2}^{\left(-1\right)} & d_{n-1}^{\left(-1\right)} & d_{n}^{\left(-1\right)}\\
    0 & 0 & d_{3}^{\left(-2\right)} & \cdots & d_{n-2}^{\left(-2\right)} & d_{n-1}^{\left(-2\right)} & d_{n}^{\left(-2\right)}
    \end{matrix}\right)

Here we see, that the numbering in the above given matrix was aiming at the
row-wise storage. That means, the indices were taken from the row-indices of
the entries.


2. column-wise storage:

.. math::

    M_{\mathrm{col}}=\left(\begin{matrix}0 & 0 & d_{1}^{\left(2\right)} & \cdots & d_{n-4}^{\left(2\right)} & d_{n-3}^{\left(2\right)} & d_{n-2}^{\left(2\right)}\\
    0 & d_{1}^{\left(1\right)} & d_{2}^{\left(1\right)} & \cdots & d_{n-3}^{\left(1\right)} & d_{n-2}^{\left(1\right)} & d_{n-1}^{\left(1\right)}\\
    d_{1} & d_{2} & d_{3} & \cdots & d_{n-2} & d_{n-1} & d_{n}\\
    d_{2}^{\left(-1\right)} & d_{3}^{\left(-1\right)} & d_{4}^{\left(-1\right)} & \cdots & d_{n-1}^{\left(-1\right)} & d_{n}^{\left(-1\right)} & 0\\
    d_{3}^{\left(-2\right)} & d_{4}^{\left(-2\right)} & d_{5}^{\left(-2\right)} & \cdots & d_{n}^{\left(-2\right)} & 0 & 0
    \end{matrix}\right)

The numbering here is a bit confusing, but in the column-wise storage, all
entries written in one column were in the same column in the original matrix.


Solving the system using pentapy
--------------------------------

To solve the system you can either provide :math:`M` as a full matrix or as
a flattened matrix in row-wise resp. col-wise flattened form.

If M is a full matrix, you call the following:

.. code-block:: python

    import pentapy as pp

    M = ...  # your matrix
    Y = ...  # your right hand side

    X = pp.solve(M, Y)

If M is flattend in row-wise order you have to set the keyword argument ``is_flat=True``:

.. code-block:: python

    import pentapy as pp

    M = ...  # your flattened matrix
    Y = ...  # your right hand side

    X = pp.solve(M, Y, is_flat=True)

If you got a col-wise flattend matrix you have to set ``index_row_wise=False``:

.. code-block:: python

    X = pp.solve(M, Y, is_flat=True, index_row_wise=False)


Tools
-----

pentapy provides some tools to convert a pentadiagonal matrix.

.. currentmodule:: pentapy.tools

.. autosummary::
   diag_indices
   shift_banded
   create_banded
   create_full


Example
-------

This is an example of how to solve a LES with a pentadiagonal matrix.
The matrix is given as a row-wise flattened matrix, that is filled with
random numbers.
Afterwards the matrix is transformed to the full quadradratic matrix to check
the result.

.. code-block:: python

    import numpy as np
    import pentapy as pp

    size = 1000
    # create a flattened pentadiagonal matrix
    M_flat = (np.random.random((5, size)) - 0.5) * 1e-5
    V = np.random.random(size) * 1e5
    # solve the LES with M_flat as row-wise flattened matrix
    X = pp.solve(M_flat, V, is_flat=True)

    # create the corresponding matrix for checking
    M = pp.create_full(M_flat, col_wise=False)
    # calculate the error
    print(np.max(np.abs(np.dot(M, X) - V)))


This should give something small like:

.. code-block:: python

    4.257890395820141e-08


.. raw:: latex

    \clearpage
