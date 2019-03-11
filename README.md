# Welcome to pentapy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2587158.svg)](https://doi.org/10.5281/zenodo.2587158)
[![PyPI version](https://badge.fury.io/py/pentapy.svg)](https://badge.fury.io/py/pentapy)
[![Build Status](https://travis-ci.org/GeoStat-Framework/pentapy.svg?branch=master)](https://travis-ci.org/GeoStat-Framework/pentapy)
[![Build status](https://ci.appveyor.com/api/projects/status/yyfgn9dgxcoolp97/branch/master?svg=true)](https://ci.appveyor.com/project/GeoStat-Framework/pentapy/branch/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## Purpose

pentapy is a toolbox to deal with pentadiagonal matrices in Python.


## Installation

The package can be installed via [pip][pip_link].
On Windows you can install [WinPython][winpy_link] to get
Python and pip running.

    pip install pentapy


## References

The solver is based on the algorithms PTRANS-I and PTRANS-II
presented by [Askar et al. 2015][ref_link].

[ref_link]: http://dx.doi.org/10.1155/2015/232456


## Examples

### Solving a pentadiagonal linear equation system

This is an example of how to solve a LES with a pentadiagonal matrix.

```python
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
```

This should give something like:
```python
4.257890395820141e-08
```

### Performance

In the following a couple of solvers for pentadiagonal systems is compared:

* Solver 1: Python implementation of ``pentapy.solve``
* Solver 2: Cython implementation of ``pentapy.solve``
* Solver 3: [``scipy.sparse.linalg.spsolve``](http://scipy.github.io/devdocs/generated/scipy.sparse.linalg.spsolve.html) with ``use_umfpack=False``
* Solver 4: [``scipy.sparse.linalg.spsolve``](http://scipy.github.io/devdocs/generated/scipy.sparse.linalg.spsolve.html) with [``use_umfpack=True``](https://scikit-umfpack.github.io/scikit-umfpack/)
* Solver 5: [Lapack solver](http://www.netlib.org/lapack/explore-html/d3/d49/group__double_g_bsolve_gafa35ce1d7865b80563bbed6317050ad7.html) for diagonal matrices [``scipy.linalg.lapack.dgbsv``](scipy.github.io/devdocs/generated/scipy.linalg.lapack.dgbsv.html)
* Solver 6: Scipy banded solver [``scipy.linalg.solve_banded``](scipy.github.io/devdocs/generated/scipy.linalg.solve_banded.html)
* Solver 7: Standard solver of Numpy [``np.linalg.solve``](https://www.numpy.org/devdocs/reference/generated/numpy.linalg.solve.html)

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/pentapy/master/examples/perfplot.png" alt="Performance" width="600px"/>
</p>

The performance plot was created with [``perfplot``](https://github.com/nschloe/perfplot).
Have a look at the script: [``examples/02_perform.py``](https://github.com/GeoStat-Framework/pentapy/blob/master/examples/02_perform.py).



## Requirements:

- [NumPy >= 1.13.0](https://www.numpy.org)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[GPL][gpl_link] © 2019

[pip_link]: https://pypi.org/project/pentapy
[winpy_link]: https://winpython.github.io/
[gpl_link]: https://github.com/GeoStat-Framework/pentapy/blob/master/LICENSE
