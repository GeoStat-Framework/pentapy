# Welcome to pentapy

[![status](https://joss.theoj.org/papers/57c3bbdd7b7f3068dd1e669ccbcf107c/status.svg)](https://joss.theoj.org/papers/57c3bbdd7b7f3068dd1e669ccbcf107c)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2587158.svg)](https://doi.org/10.5281/zenodo.2587158)
[![PyPI version](https://badge.fury.io/py/pentapy.svg)](https://badge.fury.io/py/pentapy)
[![Build Status](https://travis-ci.com/GeoStat-Framework/pentapy.svg?branch=master)](https://travis-ci.com/GeoStat-Framework/pentapy)
[![Coverage Status](https://coveralls.io/repos/github/GeoStat-Framework/pentapy/badge.svg?branch=master)](https://coveralls.io/github/GeoStat-Framework/pentapy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pentapy/badge/?version=stable)](https://geostat-framework.readthedocs.io/projects/pentapy/en/stable/?badge=stable)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## Purpose

pentapy is a toolbox to deal with pentadiagonal matrices in Python.

Pentadiagonal linear equation systems arise in many areas of science and engineering:
e.g. when solving differential equations, in interpolation problems, or in numerical schemes like finite difference.


## Installation

The package can be installed via [pip][pip_link].
On Windows you can install [WinPython][winpy_link] to get Python and pip running.

    pip install pentapy

There are pre-built wheels for Linux, MacOS and Windows for most Python versions (2.7, 3.4-3.7).

If your system is not supported and you want to have the Cython routines of
pentapy installed, you have to provide a c-compiler and run:

    pip install numpy cython
    pip install pentapy

To get the scipy solvers running, you have to install scipy or you can use the
following extra argument:

    pip install pentapy[all]

Instead of "all" you can also typ "scipy" or "umfpack" to get one of these specific packages.


## Citation

If you use `pentapy` in your publication, please cite it:

> Müller, (2019). pentapy: A Python toolbox for pentadiagonal linear systems. Journal of Open Source Software, 4(42), 1759, https://doi.org/10.21105/joss.01759

To cite a certain release, have a look at the Zenodo site: https://doi.org/10.5281/zenodo.2587158


## References

The solver is based on the algorithms PTRANS-I and PTRANS-II
presented by [Askar et al. 2015][ref_link].


## Documentation and Examples

You can find the documentation under [geostat-framework.readthedocs.io][doc_link].

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

In the following a couple of solvers for pentadiagonal systems are compared:

* Solver 1: Standard linear algebra solver of Numpy [``np.linalg.solve``](https://www.numpy.org/devdocs/reference/generated/numpy.linalg.solve.html)
* Solver 2: [``scipy.sparse.linalg.spsolve``](http://scipy.github.io/devdocs/generated/scipy.sparse.linalg.spsolve.html)
* Solver 3: Scipy banded solver [``scipy.linalg.solve_banded``](scipy.github.io/devdocs/generated/scipy.linalg.solve_banded.html)
* Solver 4: pentapy.solve with ``solver=1``
* Solver 5: pentapy.solve with ``solver=2``

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/pentapy/master/examples/perfplot_simple.png" alt="Performance" width="600px"/>
</p>

The implementations of pentapy are almost one order of magnitude faster than the
scipy algorithms for banded or sparse matrices.

The performance plot was created with [``perfplot``](https://github.com/nschloe/perfplot).
Have a look at the script: [``examples/03_perform_simple.py``](https://github.com/GeoStat-Framework/pentapy/blob/master/examples/03_perform_simple.py).



## Requirements:

- [NumPy >= 1.14.5](https://www.numpy.org)

### Optional

- [SciPy](https://www.scipy.org/)
- [scikit-umfpack](https://github.com/scikit-umfpack/scikit-umfpack)

## Contact

You can contact us via <info@geostat-framework.org>.


## License

[MIT][licence_link] © 2019 - 2020

[ref_link]: http://dx.doi.org/10.1155/2015/232456
[pip_link]: https://pypi.org/project/pentapy
[winpy_link]: https://winpython.github.io/
[licence_link]: https://github.com/GeoStat-Framework/pentapy/blob/master/LICENSE
[doc_link]: https://pentapy.readthedocs.org