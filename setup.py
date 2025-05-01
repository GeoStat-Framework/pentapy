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

compiler_directives = {}
if int(os.getenv("PENTAPY_CY_DOCS", "0")):
    print(f"## pentapy setup: embed signatures for documentation")
    compiler_directives["embedsignature"] = True
if int(os.getenv("PENTAPY_CY_COV", "0")):
    print(f"## pentapy setup: enable line-trace for coverage")
    compiler_directives["linetrace"] = True

options = {"compiler_directives": compiler_directives}
setup(
    ext_modules=cythonize(CY_MODULES, **options),
    package_data={"pentapy": ["*.pxd"]},  # include pxd files
    include_package_data=False,  # ignore other files
    zip_safe=False,
)
