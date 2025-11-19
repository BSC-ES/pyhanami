<p align="center">
  <img width="256" height="256" alt="logo" src="https://github.com/user-attachments/assets/80d42e32-53ae-44d8-9f19-581d1c1a60ac" />
</p>

# pyhanami

_pyhanami_ is a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).

## Features

Key features of the package include:

- **Easy input handling:** load climate simulation ensembles from a NetCDF file or an `xarray.Dataset` object (in future releases, also from an intake catalogue) using the `SimulationData` class. 
- **Diagnostics plotting:** generate visualizations comparing two previously loaded simulation ensembles for selected variables using the `DataDiagnostics` class. 
These include time series plots (with the `time_series_plots` method) and spatial plots (with the `spatial_plots` method). The latter generates two plots, one for the absolute difference and another for the effect size (Cohen's _d)_ between both ensembles.
- **Replicability testing:** perform a replicability test checking the statistical indistinguishability between two previously loaded simulation ensembles using the `ReplicabilityTest` class.
- **Scientific skill evaluation:** compute metrics evaluating the following phenomena using the `ScientificEvaluation` class:
    - Tropical IntraSeasonal Oscillation (ISO): this includes the computation of the bimodal ISO indices (for MJO and BSISO), as well as the calculation of related statistics comparing the indices between simulations and observations (temporal correlation ($R$), standard deviation ratio ($\sigma$), and Taylor Skill Score (TSS)).
    - Tropical Cyclones (TCs): this includes the computation of various scalar statistics (bias ($\bar{b}$), spatial Pearson correlation ($r_{xy}$), and temporal Spearman rank correlation ($\rho_s$)) for several TC metrics (counts, TC days (TCD), accumulated cyclone energy (ACE), pressure ACE (PACE), and latitude of lifetime-maximum intensity (LMI)).
- **Flexible data management:** add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization.

Future releases will include:
- **Additional scientific skill evaluation:** compute additional metrics evaluating several climate phenomena, such as TCs and precipitation.
- **Automated report generation:** produce reports including plots and statistics summary.



## Installation

### Prerequisites
1. **TempestExtremes** package:

    TempestExtremes can be installed with conda:
    ```bash
    conda install -c conda-forge tempest-extremes
    ```
    or following the instructions in: [https://github.com/ClimateGlobalChange/tempestextremes](https://github.com/ClimateGlobalChange/tempestextremes).

    Please, ensure that the TempestExtremes binaries are in your system PATH.

### Installing pyhanami

The package can be installed from the source using `pip`:
```bash
git clone https://earth.bsc.es/gitlab/ces/hanami/pyhanami.git
cd pyhanami
pip install .
```



## Basic Usage

The following example demonstrates the main functionalities of _pyhanami_. For detailed usage instructions, see the project's documentation in [https://pyhanami.readthedocs.io/](https://pyhanami.readthedocs.io/)

```python
import pyhanami as hnmi

# Load simulation data 
sim_1 = hnmi.SimulationData('source_simulation_1', name='name_sim_1')
sim_2 = hnmi.SimulationData('source_simulation_2', name='name_sim_2')


# Generate diagnostics plots comparing both simulation datasets
diags = hnmi.DataDiagnostics([sim_1, sim_2])

# Create time series plot (simulations + observations)
diags.time_series_plots(
    'variable_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path',
    obs=True,
    obs_paths='path_obs',
    obs_names='name_obs',
    start_year=year_init,
    end_year=year_end
)

# Create spatial plots (absolute difference + effect size)
diags.spatial_plots(
    'variable_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path'
)


# Perform replicability test comparing both simulation datasets
tester = hnmi.ReplicabilityTest([sim_1, sim_2], obs_path='path_obs')

tester.matrix_plot(
    ['name_sim_1', 'name_sim_2'],
    'output_path'
)


# Generate scientific skill evaluation for one simulation dataset
sciskill = hnmi.ScientificEvaluation(sim_1)

# Compute bimodal ISO indices and related statistics for one simulation dataset
corr, std_dev, tss = sciskill.bimodal_ISO(
    'name_sim_1',
    'output_path',
    start_year_eeof=year_init,
    end_year_eeof=year_end,
    years_pc=years,
    obs=True,
    obs_path='path_obs',
    obs_name='name_obs'
)

# Compute TC metrics statistics for one simulation dataset
clim_bias, storm_bias, seas_corr, spat_corr = sciskill.tc_metrics(
    'name_sim_1',
    'output_path',
    start_year=year_init,
    end_year=year_end,
    obs=True,
    obs_path='path_obs',
    obs_name='name_obs',
    obs_wind_factor=wind_factor
)
```



## License

This package is under the GPLv3 license, which you can find in the [LICENSE](./LICENSES/LICENSE) file.

### Third-party licenses
This project includes code and resources from the following sources:

#### Adapted code:
- [CyMeP package](https://github.com/zarzycki/cymep):
    - Used in: `src/pyhanami/utils/tcs_metrics/tcs_cymep_main.py`, `src/pyhanami/utils/tcs_metrics/tcs_cymep_funcs.py`
    - License: MIT License
    - Copyright (c) 2021 Colin Zarzycki

#### Code dependencies:
- [TempestExtremes package](https://github.com/ClimateGlobalChange/tempestextremes):
    - Used in: `src/pyhanami/utils/tcs_metrics/tcs_tempestextremes.py`
    - License: BSD 2-Clause License
    - Copyright (c) 2025, Paul Ullrich

#### Data sources:
- [GSV-Interface/gsv/dqc/profiles/config/variables.yaml](https://github.com/DestinE-Climate-DT/GSV-Interface/blob/master/gsv/dqc/profiles/config/variables.yaml):
    - Used for: data ranges and boundaries for plausibility checks in `src/pyhanami/config/variables.yaml`
    - License: Apache License, Version 2.0, January 2004
- [GEBCO_2024 Grid](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2024-grid):
    - Used for: computing surface geopotential in `src/pyhanami/utils/tcs_tempestextremes.py`
    - Reference: GEBCO Compilation Group (2025) GEBCO 2025 Grid (doi:10.5285/ 37c52e96-24ea-67ce-e063-7086abc05f29)
- [IBTrACS Version 4.01](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01552):
    - Used for: reference Tropical Cyclone data in `src/pyhanami/utils/tcs_ibtracs.py`
    - Reference: NCEI DSI 9637_02 (doi:10.25921/82ty-9e16)

For full license files, see the [LICENSES](./LICENSES) directory.



## Support

Create an issue or contact the authors below.



## Authors and acknowledgements

Main developer: 
- Marta Alerany Solé (BSC-CNS): marta.alerany@bsc.es

Significant contributors:
- Kai Keller (BSC-CNS): kai.keller@bsc.es
- Masuo Nakano (JAMSTEC): masuo@jamstec.go.jp

Thanks to:
- Bernardo Maraldi (BSC-CNS): bernardo.maraldi@bsc.es
- Chihiro Kodama (JAMSTEC): kodamac@jamstec.go.jp
- Iker Gonzalez (BSC-CNS): iker.gonzalez@bsc.es
- Tomoe Nasuno (JAMSTEC): nasuno@jamstec.go.jp
<!-- - JAMSTEC - Research Center for Environmental Modeling and Application (CEMA) -->

This work, along with the replicability test methodology implemented in this package, were developed as part of the [Hpc AlliaNce for Applications and supercoMputing Innovation (HANAMI) project](https://hanami-project.com/), which received funding from the European High Performance Computing Joint Undertaking (EuroHPC JU) under the European Union’s Horizon Europe framework program for research and innovation and Grant Agreement No. 101136269.

### Scientific References

Gahtan, J., Knapp, K.R., Schreck, C.J., Diamond, H.J., Kossin, J.P., & Kruk, M.C. International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01, Subset since1980. NOAA National Centers for Environmental Information (2024). https://doi.org/10.25921/82ty-9e16 (access date: 2025-11-31)

Keller, K.R., Alerany Solé, M., & Acosta, M., Replicability in Earth System Models. EGUsphere [preprint] (2025). https://doi.org/10.5194/egusphere-2025-1367

Kikuchi, K., Wang, B. & Kajikawa, Y., Bimodal representation of the tropical intraseasonal oscillation. Clim Dyn 38, 1989–2000 (2012). https://doi.org/10.1007/s00382-011-1159-1

Kikuchi, K., Extension of the bimodal intraseasonal oscillation index using JRA-55 reanalysis. Clim Dyn 54, 919–933 (2020). https://doi.org/10.1007/s00382-019-05037-z

Knapp, K.R., Kruk, M.C., Levinson, D.H., Diamond, H.J., & Neumann, C.J., The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data. Bulletin of the American Meteorological Society, 91, 363-376 (2010). https://doi.org/10.1175/2009BAMS2755.1

Nakano, M., & Kikuchi, K., Seasonality of intraseasonal variability in global climate models. Geophysical Research Letters, 46, 4441–4449 (2019). https://doi.org/10.1029/2019GL082443

Walsh, K.J.E., Fiorino, M., Landsea, C.W., & McInnes, K.L., Objectively Determined Resolution-Dependent Threshold Criteria for the Detection of Tropical Cyclones in Climate Models and Reanalyses. J. Climate, 20, 2307–2314 (2007). https://doi.org/10.1175/JCLI4074.1

Zarzycki, C.M., & Ullrich, P.A., Assessing sensitivities in algorithmic detection of tropical cyclones in climate data. Geophys. Res. Lett., 44, 1141–1149 (2017). https://doi.org/10.1002/2016GL071606.
