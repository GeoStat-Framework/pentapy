# -*- coding: utf-8 -*-
"""
1. Example.

Here we create a random row wise flattened matrix M_flat and a random
right hand side for the pentadiagonal equation system.

After solving we calculate the absolute difference between the right hand side
and the product of the matrix (which is transformed to a full quadratic one)
and the solution of the system.
"""
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
