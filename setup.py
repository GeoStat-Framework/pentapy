# -*- coding: utf-8 -*-
"""pentapy: A toolbox for pentadiagonal matrizes."""
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# cython extensions ###########################################################

CY_MODULES = []
CY_MODULES.append(
    Extension(
        "pentapy.solver",
        [os.path.join("pentapy", "solver.pyx")],
        include_dirs=[np.get_include()],
    )
)
EXT_MODULES = cythonize(CY_MODULES)  # annotate=True

# embed signatures for sphinx
for ext_m in EXT_MODULES:
    ext_m.cython_directives = {"embedsignature": True}

# setup #######################################################################

setup(ext_modules=EXT_MODULES, include_dirs=[np.get_include()])
