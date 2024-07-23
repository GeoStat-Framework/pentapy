"""pentapy: A toolbox for pentadiagonal matrices."""

# === Imports ===

import os

import numpy as np
from Cython.Build import cythonize
from extension_helpers import add_openmp_flags_if_available
from setuptools import Extension, setup

# === Constants ===

# the environment variable key for the build of the serial/parallel version
PENTAPY_BUILD_PARALLEL = "PENTAPY_BUILD_PARALLEL"
# the compiler flags for the OpenMP parallelization
OPENMP = "OPENMP"
# the number of threads for the Cython build
CYTHON_BUILD_NUM_THREADS = 1

# === Setup ===

# cython extensions
CY_MODULES = [
    Extension(
        name="pentapy.solver",
        sources=[os.path.join("src", "pentapy", "solver.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

# the OpenMP link is added if available/requested
# the environment variables can be PENTAPY_BUILD_PARALLEL = 0 (builds serial version) or
# PENTAPY_BUILD_PARALLEL != 0 (builds parallel version)
with_open_mp = False
if int(os.environ.get(PENTAPY_BUILD_PARALLEL, "0")):
    openmp_added = [add_openmp_flags_if_available(mod) for mod in CY_MODULES]
    with_open_mp = any(openmp_added)
    if with_open_mp:
        open_mp_str = "linking OpenMP (parallel version)"
    else:
        open_mp_str = "not linking OpenMP (serial version)"

    print(f"PENTAPY SETUP - {open_mp_str}")

else:
    print("PENTAPY SETUP - OpenMP not requested (serial version)")

setup(
    ext_modules=cythonize(
        CY_MODULES,
        nthreads=CYTHON_BUILD_NUM_THREADS,
        compile_time_env={OPENMP: with_open_mp},
    ),
    package_data={"pentapy": ["*.pxd"]},  # include pxd files
    include_package_data=False,  # ignore other files
    zip_safe=False,
)
