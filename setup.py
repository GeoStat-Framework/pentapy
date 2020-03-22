# -*- coding: utf-8 -*-
"""pentapy: A toolbox for pentadiagonal matrizes."""
import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))

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

# This is an important part. By setting this compiler directive, cython will
# embed signature information in docstrings. Sphinx then knows how to extract
# and use those signatures.
# python setup.py build_ext --inplace --> then sphinx build
for ext_m in EXT_MODULES:
    ext_m.cython_directives = {"embedsignature": True}

# setup #######################################################################

with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    README = f.read()
with open(os.path.join(HERE, "requirements.txt"), encoding="utf-8") as f:
    REQ = f.read().splitlines()
with open(os.path.join(HERE, "requirements_setup.txt"), encoding="utf-8") as f:
    REQ_SETUP = f.read().splitlines()
with open(os.path.join(HERE, "requirements_test.txt"), encoding="utf-8") as f:
    REQ_TEST = f.read().splitlines()
with open(
    os.path.join(HERE, "docs", "requirements_doc.txt"), encoding="utf-8"
) as f:
    REQ_DOC = f.read().splitlines()

REQ_DEV = REQ_SETUP + REQ_TEST + REQ_DOC

DOCLINE = __doc__.split("\n")[0]
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
    "Topic :: Utilities",
]

setup(
    name="pentapy",
    description=DOCLINE,
    long_description=README,
    long_description_content_type="text/markdown",
    maintainer="Sebastian Mueller",
    maintainer_email="info@geostat-framework.org",
    author="Sebastian Mueller",
    author_email="info@geostat-framework.org",
    url="https://github.com/GeoStat-Framework/pentapy",
    license="MIT",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    include_package_data=True,
    python_requires=">=3.5",
    use_scm_version={
        "relative_to": __file__,
        "write_to": "pentapy/_version.py",
        "write_to_template": "__version__ = '{version}'",
        "local_scheme": "no-local-version",
        "fallback_version": "0.0.0.dev0",
    },
    setup_requires=REQ_SETUP,
    install_requires=REQ,
    extras_require={
        "scipy": ["scipy"],
        "umfpack": ["scikit-umfpack"],
        "all": ["scipy", "scikit-umfpack"],
        "doc": REQ_DOC,
        "test": REQ_TEST,
        "dev": REQ_DEV,
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    ext_modules=EXT_MODULES,
    include_dirs=[np.get_include()],
)
