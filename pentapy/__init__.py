# -*- coding: utf-8 -*-
"""
Purpose
=======

pentapy is a toolbox to deal with pentadiagonal matrizes in Python.

Functions
^^^^^^^^^

.. currentmodule:: pentapy.core

Solver for a linear equations system with a pentadiagonal matrix.

.. autosummary::
   solve
"""
from __future__ import absolute_import

from pentapy._version import __version__
from pentapy.core import solve

__all__ = ["__version__"]
__all__ += ["solve"]
