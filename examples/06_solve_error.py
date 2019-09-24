# -*- coding: utf-8 -*-
"""
6. Example.

Here we demonstrate that the solver PTRANS-I can fail to solve a given system.
A warning is given in that case and the output will be a nan-array.
"""
import numpy as np
import pentapy as pp

# create a full pentadiagonal matrix
mat = np.array([[3, 2, 1, 0], [-3, -2, 7, 1], [3, 2, -1, 5], [0, 1, 2, 3]])
V = np.array([6, 3, 9, 6])
# solve the LES with mat as a qudratic input matrix
X = pp.solve(mat, V, is_flat=False, solver=1)
print(X)
