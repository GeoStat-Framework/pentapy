==================
pentapy Quickstart
==================

pentapy is a toolbox to deal with pentadiagonal matrices in Python and solve
the corresponding linear equation systems.

Installation
============

The package can be installed via `pip <https://pypi.org/project/pentapy/>`_.
On Windows you can install `WinPython <https://winpython.github.io/>`_ to get
Python and pip running.

.. code-block:: none

    pip install pentapy


References
==========

The solver is based on the algorithms ``PTRANS-I`` and ``PTRANS-II``
presented by `Askar et al. 2015 <http://dx.doi.org/10.1155/2015/232456>`_.


Examples
========

Solving a pentadiagonal linear equation system
----------------------------------------------

This is an example of how to solve a LES with a pentadiagonal matrix.

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


This should give something like:

.. code-block:: python

    4.257890395820141e-08


Performance
-----------

In the following, a couple of solvers for pentadiagonal systems are compared:

* Solver 1: Python implementation of ``pentapy.solve``
* Solver 2: Cython implementation of ``pentapy.solve``
* Solver 3: ``scipy.sparse.linalg.spsolve`` (`link <http://scipy.github.io/devdocs/generated/scipy.sparse.linalg.spsolve.html>`__) with ``use_umfpack=False``
* Solver 4: ``scipy.sparse.linalg.spsolve`` (`link <http://scipy.github.io/devdocs/generated/scipy.sparse.linalg.spsolve.html>`__) with ``use_umfpack=True`` (`link <https://scikit-umfpack.github.io/scikit-umfpack/>`__)
* Solver 5: `Lapack solver <http://www.netlib.org/lapack/explore-html/d3/d49/group__double_g_bsolve_gafa35ce1d7865b80563bbed6317050ad7.html>`_ for diagonal matrices ``scipy.linalg.lapack.dgbsv`` (`link <scipy.github.io/devdocs/generated/scipy.linalg.lapack.dgbsv.html>`__)
* Solver 6: Scipy banded solver ``scipy.linalg.solve_banded`` (`link <scipy.github.io/devdocs/generated/scipy.linalg.solve_banded.html>`__)
* Solver 7: Standard solver of Numpy ``np.linalg.solve`` (`link <https://www.numpy.org/devdocs/reference/generated/numpy.linalg.solve.html>`__)

.. image:: https://raw.githubusercontent.com/GeoStat-Framework/pentapy/master/examples/perfplot.png
   :width: 400px
   :align: center

The performance plot was created with ``perfplot`` (`link <https://github.com/nschloe/perfplot>`__).

Requirements
============

- `Numpy >= 1.14.5 <http://www.numpy.org>`_


License
=======

`MIT <https://github.com/GeoStat-Framework/pentapy/blob/master/LICENSE>`_ © 2019
