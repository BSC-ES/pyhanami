# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Unreleased

### Added

### Changed
- Replicability test: 
    - Added new threshold for the effect size between the distributions of scores to determine whether the test is passed or failed based on the minimum detectable effect size for a _t_-test.
    - Added option to select the year range used for the test.
- Scientific skill scores:
    - Modified computation of mean MJO amplitude and active MJO days per phase: now computed as annual averages (climatologies) instead of totals over the entire analyzed period to be able to compare these scores across different years.
- Table plots: added option to disable first reference row and show only the relative scores of the other datasets with respect to the reference dataset.

### Removed


## [0.2.0] - 2026-04-07

### Added
- Data checker: automatically run standard compliance, spatial and temporal completeness, physical plausibility, and consistency of variables and coordinates checks when loading new datasets.
- Visualization diagnostics: 
    - Added options for different frequency and observations in time series plots.
    - Included spatial bias plots comparing simulations with observations.
- Replicability test: added option to access the numerical output of the test.
- Scientific skill metrics: implemented general, Tropical IntraSeasonal Oscillation (ISO), Madden-Julian Oscillation (MJO), and Tropical Cyclones (TCs) analyses comparing simulations with observations. This includes:
    - A general `ScientificEvaluation` class with one method for each type of analysis.
    - Specific classes for each type of analysis (`GeneralEvaluation`, `ISOEvaluation`, `MJOEvaluation`, and `TCEvaluation`) which automatically compute the relevant metrics and include methods for visualizing the results.
    - Configuration classes (`ISOConfig`, `MJOConfig`, and `TCConfig`) to modify the default parameters for each analysis on a case-by-case basis.

- Configuration files: created default configuration files for parameters and variables.
- Flexible data management: included option to add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization.
- Documentation: 
    - Added licenses and references sections to README.
    - Created and documented a project in Read the Docs for the package's documentation.

### Changed

- Visualization diagnostics: separated absolute difference and effect size plots into individual functions for more flexibility.
- Replicability test: separated the execution of the test and the output visualization into individual functions.
- Installation: smoothed installation process with the conda `environment.yaml` file including all dependencies.
- Authors: moved contributors' names to README and removed `AUTHORS.md` and `generate_authors.sh` files.


## [0.1.0] - 2025-07-28

### Added
- Initial release of the package.
- Core features:
    - Support loading and manipulating climate simulation ensembles in .netcdf format and as xarray.Dataset objects.
    - Visualization diagnostics: generate time series, absolute difference and effect size plots.
    - Replicability test: perform test and visualize results together with effect size.