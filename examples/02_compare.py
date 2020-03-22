# -*- coding: utf-8 -*-
"""
2. Example.

Here we compare the outcome of the PTRANS-I and PTRANS-II algorithm for
a random input.
"""
import numpy as np
import pentapy as pp

size = 1000
# create a flattened pentadiagonal matrix
M_flat = (np.random.random((5, size)) - 0.5) * 1e-5
V = np.random.random(size) * 1e5
# compare the two solvers
X1 = pp.solve(M_flat, V, is_flat=True, solver=1)
X2 = pp.solve(M_flat, V, is_flat=True, solver=2)

# calculate the error
print("rel. diff X1 X2: ", np.max(np.abs(X1 - X2)) / np.max(np.abs(X1 + X2)))
print("max X1: ", np.max(np.abs(X1)))
print("max X2: ", np.max(np.abs(X2)))

M = pp.create_full(M_flat, col_wise=False)
# calculate the error
print("max |M*X1 - V|: ", np.max(np.abs(np.dot(M, X1) - V)))
print("max |M*X2 - V|: ", np.max(np.abs(np.dot(M, X2) - V)))
