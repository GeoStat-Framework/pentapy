"""
Purpose
=======

pentapy is a toolbox to deal with pentadiagonal matrizes in Python.


Subpackages
^^^^^^^^^^^

.. autosummary::
   :toctree: api

   tools


Solver
^^^^^^

Solver for a pentadiagonal equations system.

.. autosummary::
   :toctree: api

   solve


Tools
^^^^^

.. currentmodule:: pentapy.tools

The following tools are provided:

.. autosummary::
   diag_indices
   shift_banded
   create_banded
   create_full
"""

from pentapy.core import solve
from pentapy.tools import create_banded, create_full, diag_indices, shift_banded

try:
    from pentapy._version import __version__
except ImportError:  # pragma: nocover
    # package is not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
__all__ += ["solve"]
__all__ += ["create_banded", "create_full", "shift_banded", "diag_indices"]
