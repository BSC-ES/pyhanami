<p align="center">
  <img width="256" height="256" alt="logo" src="https://github.com/user-attachments/assets/80d42e32-53ae-44d8-9f19-581d1c1a60ac" />
</p>

# pyhanami

_pyhanami_ is a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).

## Features

Key features of the package include:

- **Easy input handling:** load climate simulation ensembles from a NetCDF file or an `xarray.Dataset` object (in future releases, also from an intake catalogue) using the `SimulationData` class. 
- **Diagnostics plotting:** generate visualizations comparing two previously loaded simulation ensembles for selected variables using the `DataDiagnostics` class. 
These include time series plots (with the `time_series_plot` method) and spatial plots (with the `spatial_plots` method). The latter generates two plots, one for the absolute difference and another for the effect size (Cohen's _d)_ between both ensembles.
- **Replicability testing:** perform a replicability test checking the statistical indistinguishability between two previously loaded simulation ensembles using the `ReplicabilityTest` class.
- **Scientific skill evaluation:** compute metrics evaluating the following phenomena using the `ScientificEvaluation` class:
    - Tropical IntraSeasonal Oscillation (ISO): this includes the computation of the bimodal ISO indices (for MJO and BSISO), as well as the calculation of related statistics comparing the indices between simulations and observations (temporal correlation ($R$), standard deviation ratio ($\sigma$), and Taylor Skill Score (TSS)).
    - Tropical Cyclones (TCs): this includes the computation of various scalar statistics (bias ($\bar{b}$), spatial Pearson correlation ($r_{xy}$), and temporal Spearman rank correlation ($\rho_s$)) for several TC metrics (counts, TC days (TCD), accumulated cyclone energy (ACE), pressure ACE (PACE), and latitude of lifetime-maximum intensity (LMI)).
- **Flexible data management:** add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization.

Future releases will include:
- **Additional scientific skill evaluation:** compute additional metrics evaluating several climate phenomena, such as precipitation.
- **Automated report generation:** produce reports including plots and statistics summary.



## Installation
The package and the necessary dependencies can be installed from the source using the conda environment provided in this repository (`environment.yaml`):
```bash
git clone https://github.com/BSC-ES/pyhanami.git
cd pyhanami
conda env create -f environment.yaml
conda activate pyhanami-env_v0.1.0
```

<!--
It is also possible to install the package directly from the source using pip:
```bash
git clone https://github.com/BSC-ES/pyhanami.git
cd pyhanami
pip install .
```
-->



## Basic Usage

The following examples demonstrate the main functionalities of _pyhanami_. For detailed usage instructions, see the project's documentation in [https://pyhanami.readthedocs.io/](https://pyhanami.readthedocs.io/)

### Load simulation data
```python
import pyhanami

# Load two simulation datasets from paths to NetCDF files or xarray.Dataset objects 
# ('source_simulation_1' and 'source_simulation_2') and assign a name to each of 
# them ('name_sim_1' and 'name_sim_2')
sim_1 = pyhanami.SimulationData('source_simulation_1', name='name_sim_1')
sim_2 = pyhanami.SimulationData('source_simulation_2', name='name_sim_2')
```

### Generate visualization diagnostics (time series and spatial plots)
```python
# Generate diagnostics plots comparing both simulation datasets
diags = pyhanami.DataDiagnostics([sim_1, sim_2])

# Create a time series plot (simulations + observations) for one climate variable 
# ('variable_name') over a specified time period (from 'year_init' to 'year_end') 
# and save it to 'output_path'
diags.time_series_plot(
    'variable_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path',
    obs=True,
    obs_paths='path_obs',
    obs_names='name_obs',
    start_year=year_init,
    end_year=year_end
)

# Create spatial plots (absolute difference + effect size) for one climate variable 
# ('variable_name') and save them to 'output_path'
diags.abs_diff_plot(
    'variable_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path'
)

diags.eff_size_plot(
    'variable_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path'
)
```

### Perform replicability test
```python
# Perform a replicability test comparing the two simulation datasets and save results 
# to 'output_path'
tester = pyhanami.ReplicabilityTest([sim_1, sim_2], obs_path='path_obs')

tester.matrix_plot(
    ['name_sim_1', 'name_sim_2'],
    'output_path'
)
```

### Evaluate scientific skill
```python
# Evaluate the scientific skill of one simulation dataset
sciskill = pyhanami.ScientificEvaluation(sim_1)

# Assess simulation of the Tropical IntraSeasonal Oscillation (ISO) by computing the 
# bimodal ISO indices and related scalar scores comparing to observations
iso_analysis = sciskill.compute_iso_scores(
    'name_sim_1',
    start_year_pc=year_init_pc,
    end_year_pc=year_end_pc,
    obs = True
)

# Save results of the analysis to 'output_path'
iso_analysis.save_data('output_path')

# Create plots for EEOFs, PCs (bimodal ISO indices) and frequency of ISO events and
# save them to 'output_path'
iso_analysis.eeof_plots('output_path')
iso_analysis.pc_plots('output_path', years=[year_1, year_2, year_3])
iso_analysis.freq_plot('output_path')

# Display computed scores (scalar values measuring how well simulations match 
# observations)
iso_analysis.scores

# Assess simulation of Tropical Cyclones (TCs) by computing TC metrics and derived 
# scalar scores for one simulation dataset comparing to observations and reanalyses
tc_analysis = sciskill.compute_tc_scores(
    'name_sim_1',
    start_year_tc=year_init_tc,
    end_year_tc=year_end_tc
)

# Save results of the analysis to 'output_path'
tc_analysis.save_data('output_path')

# Create table plots summarizing the computed scores (biases, and temporal and 
# spatial correlations) and save them to 'output_path'
tc_analysis.clim_bias_table('output_path')
tc_analysis.storm_bias_table('output_path')
tc_analysis.temp_corr_table('output_path')
tc_analysis.spatial_corr_table('output_path')
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
- [NOAA Interpolated Outgoing Longwave Radiation (OLR)](https://psl.noaa.gov/data/gridded/data.olrcdr.interp.html):
    - Used for: reference Tropical IntraSeasonal Oscillation data in `src/pyhanami/diags/ScientificSkill.py`
    - Reference: Liebmann, B., & Smith, C.A., Description of a Complete (Interpolated) Outgoing Longwave Radiation Dataset. Bulletin of the American Meteorological Society, 77, 1275-1277 (1996)
    - Acknowledgment: NOAA Interpolated Outgoing Longwave Radiation (OLR) data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov
- [IBTrACS Version 4.01](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01552):
    - Used for: reference Tropical Cyclones data in `src/pyhanami/diags/ScientificSkill.py`
    - Reference: NCEI DSI 9637_02 (doi:10.25921/82ty-9e16)
- [GEBCO_2024 Grid](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2024-grid):
    - Used for: computing surface geopotential in `src/pyhanami/utils/tcs_tempestextremes.py`
    - Reference: GEBCO Compilation Group (2024) GEBCO 2024 Grid (doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f)

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

Liebmann, B., & Smith, C.A., Description of a Complete (Interpolated) Outgoing Longwave Radiation Dataset. Bulletin of the American Meteorological Society, 77, 1275-1277 (1996).
<!-- No DOI found for this paper??? -->

Nakano, M., & Kikuchi, K., Seasonality of intraseasonal variability in global climate models. Geophysical Research Letters, 46, 4441–4449 (2019). https://doi.org/10.1029/2019GL082443

Taylor, K.E., Summarizing multiple aspects of model performance in a single diagram. J. Geophys. Res., 106(D7), 7183–7192, (2001). https://doi.org/10.1029/2000JD900719

Walsh, K.J.E., Fiorino, M., Landsea, C.W., & McInnes, K.L., Objectively Determined Resolution-Dependent Threshold Criteria for the Detection of Tropical Cyclones in Climate Models and Reanalyses. J. Climate, 20, 2307–2314 (2007). https://doi.org/10.1175/JCLI4074.1

Zarzycki, C.M., & Ullrich, P.A., Assessing sensitivities in algorithmic detection of tropical cyclones in climate data. Geophys. Res. Lett., 44, 1141–1149 (2017). https://doi.org/10.1002/2016GL071606.
