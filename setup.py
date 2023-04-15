"""pentapy: A toolbox for pentadiagonal matrizes."""
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# cython extensions
CY_MODULES = [
    Extension(
        name=f"pentapy.solver",
        sources=[os.path.join("src", "pentapy", "solver.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(CY_MODULES),
    package_data={"pentapy": ["*.pxd"]},  # include pxd files
    include_package_data=False,  # ignore other files
    zip_safe=False,
)
