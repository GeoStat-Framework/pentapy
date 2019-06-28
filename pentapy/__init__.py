# -*- coding: utf-8 -*-
"""
Purpose
=======

pentapy is a toolbox to deal with pentadiagonal matrizes in Python.

Solver
^^^^^^

.. currentmodule:: pentapy.core

Solver for a linear equations system with a pentadiagonal matrix.

.. autosummary::
   solve


Tools
^^^^^

.. currentmodule:: pentapy.tools

The following tools are provided

.. autosummary::
   diag_indices
   shift_banded
   create_banded
   create_full
"""
from __future__ import absolute_import

from pentapy._version import __version__
from pentapy.core import solve
from pentapy.tools import (
    create_banded,
    create_full,
    shift_banded,
    diag_indices,
)

__all__ = ["__version__"]
__all__ += ["solve"]
__all__ += ["create_banded", "create_full", "shift_banded", "diag_indices"]
