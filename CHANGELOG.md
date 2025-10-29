# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Unreleased

### Added
- Data checker: standard compliance, spatial and temporal completeness, physical plausibility, and consistency of variables and coordinates checks when loading new datasets.
- Scientific skill metrics: bimodal ISO indices Tropical IntraSeasonal Oscillation (ISO) evaluation and Tropical Cyclones (TCs) metrics.
- Documentation 


### Changed

### Removed


## [0.1.0] - 2025-07-28

### Added
- Initial release of the package.
- Core features:
    - Support loading and manipulating climate simulation ensembles in .netcdf format and as xarray.Dataset objects.
    - Visualization diagnostics: generate time series, absolute difference and effect size plots.
    - Replicability test: perform test and visualize results together with effect size.