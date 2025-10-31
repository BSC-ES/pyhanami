# Home

Welcome to the Read the Docs page for _pyhanami_. This documentation contains detailed information on how to use the _pyhanami_ Python package, a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).

## Description

_pyhanami_ is a Python package that provides tools for:
1. Generating diagnostic visualizations (e.g., time series and spatial plots)
2. Evaluating the replicability of ESMs (e.g., identify significant differences between experiments)
3. Assessing the scientific skill of ESMs (e.g., metrics related to ISO and Tropical Cyclones)

Key features of the package include:

- **Easy input handling:** load climate simulation ensembles from a NetCDF file or an `xarray.Dataset` object (in future releases, also from an intake catalogue) using the `SimulationData` class. 
- **Diagnostics plotting:** generate visualizations comparing two previously loaded simulation ensembles for selected variables using the `DataDiagnostics` class. 
These include time series plots (with the `time_series_plots` method) and spatial plots (with the `spatial_plots` method). The latter generates two plots, one for the absolute difference and another for the effect size (Cohen's _d)_ between both ensembles.
- **Replicability testing:** perform a replicability test checking the statistical indistinguishability between two previously loaded simulation ensembles using the `ReplicabilityTest` class.
- **Scientific skill evaluation:** compute metrics evaluating the Tropical IntraSeasonal Oscillation (ISO) using the `ScientificEvaluation` class. These include the computation of the bimodal ISO indices (for MJO and BSISO), as well as the calculation of related statistics comparing the indices between simulations and observations (temporal correlation (R), standard deviation ratio (σ), and Taylor Skill Score (TSS)).
- **Flexible data management:** add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization.

Future releases will include:
- **Additional scientific skill evaluation:** compute additional metrics evaluating several climate phenomena, such as TCs and precipitation.
- **Automated report generation:** produce reports including plots and statistics summary.

For quick start instructions, see the README on the [project's GitHub repository](https://github.com/BSC-ES/pyhanami/tree/main). This documentation provides in-depth information and advanced usage details.


## Contact people

Main developer: 
- Marta Alerany Solé (BSC-CNS): marta.alerany@bsc.es

Significant contributors:
- Kai Keller (BSC-CNS): kai.keller@bsc.es
- Masuo Nakano (JAMSTEC): masuo@jamstec.go.jp

This work was developed as part of the [Hpc AlliaNce for Applications and supercoMputing Innovation (HANAMI) project](https://hanami-project.com/), which received funding from the European High Performance Computing Joint Undertaking (EuroHPC JU) under the European Union’s Horizon Europe framework program for research and innovation and Grant Agreement No. 101136269.