# Changelog

All notable changes to **pentapy** will be documented in this file.

## [1.4.0] - 2024-06

See [#22](https://github.com/GeoStat-Framework/pentapy/pull/22)

### Enhancements

- added support for multiple right-hand sides (currently serial)
- improved error handling and added debug information to error messages

### Changes

- shotgun refactored and documented the Cython implementation of PTRANS-I and PTRANS-II for single and multiple right-hand sides support
- fully typed the function ``pentapy.solve``
- made internal solver alias handling of ``pentapy.solve`` smarter, more robust, and removed all duplicate code
- gave all solvers a consistent interface
- made code in ``pentapy.core`` more human-readable and maintainable and added comments
- fixed typos in documentation

### Bugfixes

- fixed error handling in case of zero-division to trigger dead error handling branch (see [Issue 23](https://github.com/GeoStat-Framework/pentapy/issues/23))
- fixed edge case error for row/column of 3 (see [Issue 24](https://github.com/GeoStat-Framework/pentapy/issues/24))

### Tests

- transitioned from ``unittest``-based testing to fully ``pytest``-based testing with parametrized and parallelized exhaustive testing (see [Issue 25](https://github.com/GeoStat-Framework/pentapy/issues/25))
- made actual tests more meaningful by comparing them to LAPACK as reference standard (see [Issue 25](https://github.com/GeoStat-Framework/pentapy/issues/25))
- included external solver bindings accessible via ``pentapy.solve`` as part of the test suite
- increased true coverage (not line-hit coverage) close to 100%

### Packaging

- made dependency specification file-based and dynamic

## [1.3.0] - 2024-04

See [#21](https://github.com/GeoStat-Framework/pentapy/pull/21)

### Enhancements
- added support for python 3.12
- added support for numpy 2
- build extensions with numpy 2 and cython 3

### Changes
- dropped python 3.7 support
- dropped 32bit builds
- linted cython files
- increase maximal line length to 88 (black default)


## [1.2.0] - 2023-04

See [#19](https://github.com/GeoStat-Framework/pentapy/pull/19)

### Enhancements
- added support for python 3.10 and 3.11
- add wheels for arm64 systems
- created `solver.pxd` file to be able to cimport the solver module
- added a `CITATION.bib` file

### Changes
- move to `src/` based package structure
- dropped python 3.6 support
- move meta-data to pyproject.toml
- simplified documentation

### Bugfixes
- determine correct version when installing from archive

## [1.1.2] - 2021-07

### Changes
- new package structure with `pyproject.toml` ([#15](https://github.com/GeoStat-Framework/pentapy/pull/15))
- Sphinx-Gallery for Examples
- Repository restructuring: use a single `main` branch
- use `np.asarray` in `solve` to speed up computation ([#17](https://github.com/GeoStat-Framework/pentapy/pull/17))


## [1.1.1] - 2021-02

### Enhancements
- Python 3.9 support

### Changes
- GitHub Actions for CI


## [1.1.0] - 2020-03-22

### Enhancements
- Python 3.8 support

### Changes
- python only builds are no longer available
- Python 2.7 and 3.4 support dropped


## [1.0.3] - 2019-11-10

### Enhancements
- the algorithms `PTRANS-I` and `PTRANS-II` now raise a warning when they can not solve the given system
- there are now switches to install scipy and umf solvers as extra requirements

### Bugfixes
- multiple minor bugfixes


## [1.0.0] - 2019-09-18

### Enhancements
- the second algorithm `PTRANS-II` from *Askar et al. 2015* is now implemented and can be used by `solver=2`
- the package is now tested and a coverage is calculated
- there are now pre-built binaries for Python 3.7
- the documentation is now available under https://geostat-framework.readthedocs.io/projects/pentapy

### Changes
- pentapy is now licensed under the MIT license


## [0.1.1] - 2019-03-08

### Bugfixes
- MANIFEST.in was missing in the 0.1.0 version


## [0.1.0] - 2019-03-07

This is the first release of pentapy, a python toolbox for solving pentadiagonal linear equation systems.
The solver is implemented in cython, which makes it really fast.


[1.4.0]: https://github.com/GeoStat-Framework/pentapy/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/GeoStat-Framework/pentapy/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/GeoStat-Framework/pentapy/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/GeoStat-Framework/pentapy/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/GeoStat-Framework/pentapy/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/GeoStat-Framework/pentapy/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/GeoStat-Framework/pentapy/compare/v1.0.0...v1.0.3
[1.0.0]: https://github.com/GeoStat-Framework/pentapy/compare/v0.1.1...v1.0.0
[0.1.1]: https://github.com/GeoStat-Framework/pentapy/compare/v0.1...v0.1.1
[0.1.0]: https://github.com/GeoStat-Framework/pentapy/releases/tag/v0.1
