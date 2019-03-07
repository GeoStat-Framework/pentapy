# Welcome to pentapy


## Purpose

pentapy is a toolbox to deal with pentadiagonal matrizes in Python.


## Installation

The package can be installed via [pip][pip_link].
On Windows you can install [WinPython][winpy_link] to get
Python and pip running.

    pip install pentapy


## References

The solver is based on the algorithms PTRANS-I and PTRANS-II
presented by [Askar et al. 2015][ref_link].

[ref_link]: http://dx.doi.org/10.1155/2015/232456


### Examples

#### Solving a pentadiagonal linear equation system

This is an example of how to solve a LES with a pentadiagonal matrix.

```python
import numpy as np
from pentapy import solve
size = 1000
# create a flattened pentadiagonal matrix
M_flat = (np.random.random((5, size)) - 0.5) * 1e-5
V = np.random.random(size) * 1e5
# solve the LES
X = solve(M_flat, V, is_flat=True)

# create the corresponding matrix for checking
M = (
     np.diag(M_flat[0, :-2], 2)
     + np.diag(M_flat[1, :-1], 1)
     + np.diag(M_flat[2, :], 0)
     + np.diag(M_flat[3, 1:], -1)
     + np.diag(M_flat[4, 2:], -2)
)
# calculate the error
print(np.max(np.abs(np.dot(M, X) - V)))
```

This should give something like:
```python
4.257890395820141e-08
```


## Requirements:

- [NumPy >= 1.13.0](https://www.numpy.org)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[GPL][gpl_link] © 2019

[pip_link]: https://pypi.org/project/pentapy
[winpy_link]: https://winpython.github.io/
[gpl_link]: https://github.com/GeoStat-Framework/pentapy/blob/master/LICENSE
