# Home

Welcome to the Read the Docs page for _pyhanami_. This documentation contains detailed information on how to use the _pyhanami_ Python package, a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).

## Description

_pyhanami_ is a Python package that provides tools for:
1. Generating diagnostic visualizations (e.g., time series and spatial plots)
2. Evaluating the replicability of ESMs (e.g., identify significant differences between experiments)
3. Assessing the scientific skill of ESMs (e.g., scalar scores related to ISO and Tropical Cyclones)

Key features of the package include:

- **Easy input handling:** load climate simulation ensembles from a NetCDF file or an `xarray.Dataset` object (in future releases, also from an intake catalogue) using the `SimulationData` class. 
- **Diagnostics plotting:** generate visualizations comparing two previously loaded simulation ensembles for selected variables using the `DataDiagnostics` class. 
These include time series and spatial plots. The latter involves plots displaying the absolute difference and the effect size (Cohen's _d)_ between two ensembles, and bias plots comparing the simulations to reference observations.
- **Replicability testing:** perform a replicability test checking the statistical indistinguishability between two previously loaded simulation ensembles using the `ReplicabilityTest` class.
- **Scientific skill evaluation:** compute scalar scores evaluating the following phenomena using the `ScientificEvaluation` class:
    - <u> General skill:</u> this includes the computation of general scalar scores (bias (BIAS), Root Mean Square Error (RMSE), and spatial Pearson correlation ($r_{xy}$)) for any variable in the simulation dataset comparing to observations.
    - <u>Tropical IntraSeasonal Oscillation (ISO):</u> this includes the computation of the bimodal ISO indices (for MJO and BSISO), as well as the calculation of related scalar scores comparing the indices between simulations and observations (amplitude ratio ($\alpha$), temporal correlation ($R$), standard deviation ratio ($\sigma$), and Taylor Skill Score (TSS)).
    - <u>Madden-Julian Oscillation (MJO):</u> this includes the computation of the Real-Time Multivariate MJO (RMM) indices, as well as the calculation of related scalar scores (Pearson correlation ($r_{\text{var,mode}}$) and bias ($\bar{b}$)) comparing the MJO spatial patterns and activity between simulations and observations.
    - <u>Tropical Cyclones (TCs):</u> this includes the computation of various scalar scores (bias ($\bar{b}$), spatial Pearson correlation ($r_{xy}$), and temporal Spearman rank correlation ($\rho_s$)) for several TC metrics (counts, TC days (TCD), accumulated cyclone energy (ACE), pressure ACE (PACE), and latitude of lifetime-maximum intensity (LMI)).
- **Flexible data management:** add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization.

Future releases will include:
- **Additional scientific skill evaluation:** compute additional scalar scores evaluating several climate phenomena, such as precipitation.
- **Automated report generation:** produce reports including plots and scores summary.

For quick start instructions, see the README on the [project's GitHub repository](https://github.com/BSC-ES/pyhanami/tree/main). This documentation provides in-depth information and advanced usage details.


## Contact people

Main developer:
- Marta Alerany Solé (BSC-CNS): [marta.alerany@bsc.es](mailto:marta.alerany@bsc.es)

<!--Significant contributors:
- Kai Keller (BSC-CNS): kai.keller@bsc.es
- Masuo Nakano (JAMSTEC): masuo@jamstec.go.jp-->

This work was developed as part of the [Hpc AlliaNce for Applications and supercoMputing Innovation (HANAMI) project](https://hanami-project.com/), which received funding from the European High Performance Computing Joint Undertaking (EuroHPC JU) under the European Union’s Horizon Europe framework program for research and innovation and Grant Agreement No. 101136269.