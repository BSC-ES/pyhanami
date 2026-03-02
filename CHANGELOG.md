# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Unreleased

### Added
- Data checker: automatically run standard compliance, spatial and temporal completeness, physical plausibility, and consistency of variables and coordinates checks when loading new datasets.
- Visualization diagnostics: included spatial bias plots comparing simulations with observations.
- Replicability test: added option to access the numerical output of the test.
- Scientific skill metrics: implemented Tropical IntraSeasonal Oscillation (ISO), Madden-Julian Oscillation (MJO), and Tropical Cyclones (TCs) analyses comparing simulations with observations.
- Configuration files: created default configuration files for parameters and variables.
- Flexible data management: included option to add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization.
- Documentation: 
    - Added licenses and references sections to README.
    - Created a project in Read the Docs for the package's documentation.

### Changed

- Visualization diagnostics: 
    - Added options for different frequency and observations in time series plots.
    - Separated absolute difference and effect size plots into individual functions for more flexibility.
- Installation: smoothed installation process with conda `environment.yaml` file including all dependencies.

### Removed


## [0.1.0] - 2025-07-28

### Added
- Initial release of the package.
- Core features:
    - Support loading and manipulating climate simulation ensembles in .netcdf format and as xarray.Dataset objects.
    - Visualization diagnostics: generate time series, absolute difference and effect size plots.
    - Replicability test: perform test and visualize results together with effect size.