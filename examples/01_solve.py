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
