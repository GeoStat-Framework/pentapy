# Changelog

All notable changes to **pentapy** will be documented in this file.


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


[1.1.0]: https://github.com/GeoStat-Framework/gstools/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/GeoStat-Framework/gstools/compare/v1.0.0...v1.0.3
[1.0.0]: https://github.com/GeoStat-Framework/gstools/compare/v0.1.1...v1.0.0
[0.1.1]: https://github.com/GeoStat-Framework/gstools/compare/v0.1...v0.1.1
[0.1.0]: https://github.com/GeoStat-Framework/gstools/releases/tag/v0.1
