<p align="center">
  <img width="256" height="256" alt="logo" src="https://github.com/user-attachments/assets/80d42e32-53ae-44d8-9f19-581d1c1a60ac" />
</p>

# pyhanami

_pyhanami_ is a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).

## Features

Key features of the package include:

- **Easy input handling:** load climate simulation ensembles from a NetCDF file or an `xarray.Dataset` object (in future releases, also from an intake catalogue) using the `SimulationData` class. 
- **Diagnostics plotting:** generate visualizations comparing two previously loaded simulation ensembles for selected variables using the `DataDiagnostics` class. 
These include time series and spatial plots. The latter involves plots displaying the absolute difference and the effect size (Cohen's _d)_ between two ensembles, and bias plots comparing the simulations to reference observations.
- **Replicability testing:** perform a replicability test checking the statistical indistinguishability between two previously loaded simulation ensembles using the `ReplicabilityTest` class.
- **Scientific skill evaluation:** compute scalar scores evaluating the following phenomena using the `ScientificEvaluation` class:
    - Tropical IntraSeasonal Oscillation (ISO): this includes the computation of the bimodal ISO indices (for MJO and BSISO), as well as the calculation of related scalar scores comparing the indices between simulations and observations (amplitude ratio ($\alpha$), temporal correlation ($R$), standard deviation ratio ($\sigma$), and Taylor Skill Score (TSS)).
    - Madden-Julian Oscillation (MJO): this includes the computation of the Real-Time Multivariate MJO (RMM) indices and the MJO power spectrum, as well as the calculation of related scalar scores (Pearson correlation ($r_{\text{var,mode}}$) and bias ($b$)) comparing the MJO spatial patterns, activity and power between simulations and observations. 
    - Tropical Cyclones (TCs): this includes the computation of various scalar scores (bias ($\bar{b}$), spatial Pearson correlation ($r_{xy}$), and temporal Spearman rank correlation ($\rho_s$)) for several TC metrics (counts, TC days (TCD), accumulated cyclone energy (ACE), pressure ACE (PACE), and latitude of lifetime-maximum intensity (LMI)).
- **Flexible data management:** add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization.

Future releases will include:
- **Additional scientific skill evaluation:** compute additional scalar scores evaluating several climate phenomena, such as precipitation.
- **Automated report generation:** produce reports including plots and scores summary.



## Installation
The package and its dependencies can be installed using the conda environment provided in this repository (`environment.yaml`):
```bash
# Clone the repository to your current directory
git clone https://github.com/BSC-ES/pyhanami.git

# Enter the repository folder
cd pyhanami

# Create the conda environment 
conda env create -f environment.yaml

# Activate the newly created environment
conda activate pyhanami-env_v0.2.0
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
Load two simulation datasets from either paths to NetCDF files or already loaded `xarray.Dataset` objects (`source_simulation_1` and `source_simulation_2`) and assign a name to each of them (`name_sim_1` and `name_sim_2`):
```python
import pyhanami

# Initialize the SimulationData class for both datasets
sim_1 = pyhanami.SimulationData(data_source='source_simulation_1', name='name_sim_1')
sim_2 = pyhanami.SimulationData(data_source='source_simulation_2', name='name_sim_2')
```

### Generate visualization diagnostics (time series and spatial plots)
Generate diagnostics plots comparing both simulation datasets:
```python
# Initialize the DataDiagnostics class with the two simulation datasets
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

# Create a spatial bias plot (simulations - observations) for one climate variable 
# ('variable_name') and save it to 'output_path'
diags.bias_plot(
    'variable_name',
    'name_sim_1',
    'output_path',
    obs_path='path_obs',
    obs_name='name_obs',
    start_year=year_init,
    end_year=year_end
)
```

### Perform replicability test
Perform a replicability test comparing both simulation datasets:
```python
# Initialize the ReplicabilityTest class with the two simulation datasets
tester = pyhanami.ReplicabilityTest([sim_1, sim_2], obs_path='path_obs')

# Perform replicability test
tester.perform_rep_test(['name_sim_1', 'name_sim_2'])

# Save outcome of the test to 'output_path'
tester.save_data(['name_sim_1', 'name_sim_2'], 'output_path')

# Create plot summarizing the outcome of the test and save it to 'output_path'
tester.matrix_plot(['name_sim_1', 'name_sim_2'], 'output_path')
```

### Evaluate scientific skill
Evaluate the scientific skill of one simulation dataset:
```python
# Initialize the ScientificEvaluation class with one simulation dataset
sciskill = pyhanami.ScientificEvaluation(sim_1)
```

#### Tropical IntraSeasonal Oscillation (ISO)
Assess simulation of the Tropical IntraSeasonal Oscillation (ISO) by computing the bimodal ISO indices and related scalar scores comparing to observations:
```python
# Perform the ISO analysis
iso_analysis = sciskill.compute_iso_scores(
    'name_sim_1',
    start_year_pc=year_init_pc,
    end_year_pc=year_end_pc,
    obs = True
)

# Save results of the analysis to 'output_path'
iso_analysis.save_data('output_path')

# Create plots for EEOFs, PCs (bimodal ISO indices) and monthly frequency of 
# ISO events (seasonality) and save them to 'output_path'
iso_analysis.eeof_plots('output_path')
iso_analysis.pc_plots('output_path', years=[year_1, year_2, year_3])
iso_analysis.freq_plot('output_path')

# Create table plot summarizing the computed scalar scores and save them to 'output_path'
# (scalar values measuring how well simulations match observations)
iso_analysis.scores_table('output_path')
```

#### Madden-Julian Oscillation (MJO)
Assess simulation of the Madden-Julian Oscillation (MJO) by computing the Real-time Multivariate MJO (RMM) indices, the power spectrum and related scalar scores comparing to observations:
```python
# Perform the MJO analysis
mjo_analysis = sciskill.compute_mjo_scores(
    'name_sim_1',
    start_year_mjo=year_init_mjo,
    end_year_mjo=year_end_mjo,
    threshold_active_days=1
)

# Save results of the analysis to 'output_path'
mjo_analysis.save_data('output_path')

# Create plots for CEOFs, lead-lag correlation, MJO activity per phase and power spectrum 
# and save them to 'output_path'
mjo_analysis.ceof_plots('output_path')
mjo_analysis.lead_lag_corr_plot('output_path')
mjo_analysis.activity_per_phase_plots('output_path')
mjo_analysis.power_spectrum_plots('output_path')

# Create table plots summarizing the computed scalar scores and save them to 'output_path'
mjo_analysis.ceof_bias_table('output_path')
mjo_analysis.ceof_corr_table('output_path')
mjo_analysis.activity_per_phase_bias_tables('output_path')
mjo_analysis.power_bias_table('output_path')
```

#### Tropical Cyclones (TCs)
Assess simulation of Tropical Cyclones (TCs) by computing TC metrics and derived scalar scores comparing to observations and reanalyses:
```python
# Perform the TC analysis
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

<!--
# Assess general scientific skill by computing general scalar scores (bias, RMSE, and 
# spatial Pearson correlation) for one variable comparing to observations
general_anlysis = sciskill.compute_general_scores(
    'var_name',
    'name_sim_1',
    start_year=year_init,
    end_year=year_end,
    obs_path='path_obs',
    obs_name='name_obs',
    start_year=year_init,
    end_year=year_end
)

# Save results of the analysis to 'output_path'
general_anlysis.save_data('output_path')

# Create table plot summarizing the computed scores for one variable and save it 
# to 'output_path'
general_anlysis.scores_table('var_name', 'output_path')
-->


## License

This package is under the GPLv3 license, which you can find in the [LICENSE](./LICENSES/LICENSE) file.

### Third-party licenses
This project includes code and resources from the following sources:

#### Adapted code:
- [CyMeP package](https://github.com/zarzycki/cymep):
    - Used in: `src/pyhanami/utils/tcs_scores/tcs_cymep_main.py`, `src/pyhanami/utils/tcs_scores/tcs_cymep_funcs.py`
    - License: MIT License
    - Copyright (c) 2021 Colin Zarzycki
- [wavenumber_frequency GitHub repository](https://github.com/brianpm/wavenumber_frequency):
    - Used in: `src/pyhanami/utils/mjo_scores/mjo_spectrum_funcs.py`
    - License: MIT License
    - Copyright (c) 2024 Brian Medeiros
#### Code dependencies:
- [TempestExtremes package](https://github.com/ClimateGlobalChange/tempestextremes):
    - Used in: `src/pyhanami/utils/tcs_scores/tcs_tempestextremes.py`
    - License: BSD 2-Clause License
    - Copyright (c) 2025, Paul Ullrich

#### Data sources:
- [GSV-Interface/gsv/dqc/profiles/config/variables.yaml](https://github.com/DestinE-Climate-DT/GSV-Interface/blob/master/gsv/dqc/profiles/config/variables.yaml):
    - Used for: data ranges and boundaries for plausibility checks in `src/pyhanami/config/variables.yaml`
    - License: Apache License, Version 2.0, January 2004
- [NOAA Interpolated Outgoing Longwave Radiation (OLR)](https://psl.noaa.gov/data/gridded/data.olrcdr.interp.html):
    - Used for: reference top of atmosphere outgoing longwave radiation data in `src/pyhanami/diags/ScientificSkill.py`
    - Reference: Liebmann, B., & Smith, C.A., Description of a Complete (Interpolated) Outgoing Longwave Radiation Dataset. Bulletin of the American Meteorological Society, 77, 1275-1277 (1996)
    - Acknowledgment: NOAA Interpolated Outgoing Longwave Radiation (OLR) data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov
- [ERA5 hourly data on pressure levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview):
    - Used for: reference eastward wind data in `src/pyhanami/diags/ScientificSkill.py` 
    - References:
        - Copernicus Climate Change Service, Climate Data Store, ERA5 hourly data on pressure levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS) (2023). https://doi.org/10.24381/cds.bd0915c6 (access date: 2025-06-18)
        - Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., & Thépaut, J-N., ERA5 hourly data on pressure levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS) (2023). https://doi.org/10.24381/cds.bd0915c6 (access date: 2025-06-18)
    - Acknowledgement: (H. Hersbach et al., 2023) was downloaded from the Copernicus Climate Change Service (2023). The results contain modified Copernicus Climate Change Service information (2023). Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
<!-- NOTE: check https://confluence.ecmwf.int/display/CKB/Use+Case+2%3A+ERA5+hourly+data+on+single+levels+from+1940+to+present for proper ERA5 data citation -->
- [IBTrACS Version 4.01](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01552):
    - Used for: reference Tropical Cyclones data in `src/pyhanami/diags/ScientificSkill.py`
    - Reference: NCEI DSI 9637_02 (doi:10.25921/82ty-9e16)
- [GEBCO_2024 Grid](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2024-grid):
    - Used for: computing surface geopotential in `src/pyhanami/utils/tcs_scores/tcs_tempestextremes.py`
    - Reference: GEBCO Compilation Group (2024) GEBCO 2024 Grid (doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f)

For full license files, see the [LICENSES](./LICENSES) directory.



## Support

Create an issue or contact the authors below.



## Authors and acknowledgements

Main developers: 
- Marta Alerany Solé (BSC-CNS): marta.alerany@bsc.es
- Kai Keller (BSC-CNS): kai.keller@bsc.es

Significant contributors:
- Bernardo Maraldi (BSC-CNS): bernardo.maraldi@bsc.es
- Masuo Nakano (JAMSTEC): masuo@jamstec.go.jp

Thanks to:
- Chihiro Kodama (JAMSTEC): kodamac@jamstec.go.jp
- Iker Gonzalez Yeregui (BSC-CNS): iker.gonzalez@bsc.es
- Tomoe Nasuno (JAMSTEC): nasuno@jamstec.go.jp
<!-- - JAMSTEC - Research Center for Environmental Modeling and Application (CEMA) -->

This work, along with the replicability test methodology implemented in this package, were developed as part of the [Hpc AlliaNce for Applications and supercoMputing Innovation (HANAMI) project](https://hanami-project.com/), which received funding from the European High Performance Computing Joint Undertaking (EuroHPC JU) under the European Union’s Horizon Europe framework program for research and innovation and Grant Agreement No. 101136269.

### Scientific References

Ahn, M.-S., Kim, D., Sperber, K.R., Kang, I.-S., Maloney, E., Waliser, D., & Hendon, H., MJO simulation in CMIP5 climate models: MJO skill metrics and process-oriented diagnosis. Clim. Dyn., 49, 4023–4045 (2017). https://doi.org/10.1007/s00382-017-3558-4

Gahtan, J., Knapp, K.R., Schreck, C.J., Diamond, H.J., Kossin, J.P., & Kruk, M.C. International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01, Subset since1980. NOAA National Centers for Environmental Information (2024). https://doi.org/10.25921/82ty-9e16 (access date: 2025-11-31)

<!--
Keller, K.R., Alerany Solé, M., & Acosta, M., Replicability in Earth System Models. EGUsphere [preprint] (2025). https://doi.org/10.5194/egusphere-2025-1367
-->

Keller, K.R., Alerany Solé, M., & Acosta, M., Replicability in Earth System Models. Geosci. Model Dev., 18, 10221-10243 (2025) https://doi.org/10.5194/gmd-18-10221-2025

Kikuchi, K., Wang, B., & Kajikawa, Y., Bimodal representation of the tropical intraseasonal oscillation. Clim. Dyn. 38, 1989–2000 (2012). https://doi.org/10.1007/s00382-011-1159-1

Kikuchi, K., Extension of the bimodal intraseasonal oscillation index using JRA-55 reanalysis. Clim. Dyn. 54, 919–933 (2020). https://doi.org/10.1007/s00382-019-05037-z

Knapp, K.R., Kruk, M.C., Levinson, D.H., Diamond, H.J., & Neumann, C.J., The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data. Bulletin of the American Meteorological Society, 91, 363-376 (2010). https://doi.org/10.1175/2009BAMS2755.1

Lee, J., Gleckler, P.J., Ahn, M.-S., Ordonez, A., Ullrich, P.A., Sperber, K.R., Taylor, K.E., Planton, Y.Y., Guilyardi, E., Durack, P., Bonfils, C., Zelinka, M.D., Chao, L.-W., Dong, B., Doutriaux, C., Zhang, C., Vo, T., Boutte, J., Wehner, M.F., Pendergrass, A.G., Kim, D., Xue, Z., Wittenberg, A.T., & Krasting, J., Systematic and objective evaluation of Earth system models: PCMDI Metrics Package (PMP) version 3. Geosci. Model Dev., 17, 3919–3948 (2024). https://doi.org/10.5194/gmd-17-3919-2024

Liebmann, B., & Smith, C.A., Description of a Complete (Interpolated) Outgoing Longwave Radiation Dataset. Bulletin of the American Meteorological Society, 77, 1275-1277 (1996).
<!-- No DOI found for this paper??? -->

Nakano, M., & Kikuchi, K., Seasonality of intraseasonal variability in global climate models. Geophysical Research Letters, 46, 4441–4449 (2019). https://doi.org/10.1029/2019GL082443

Planton, Y.Y., Guilyardi, E., Wittenberg, A.T., Lee, J., Gleckler, P.J., Bayr, T., McGregor, S., McPhaden, M.J., Power, S., Roehrig, R., Vialard, J., & Voldoire, A., Evaluating Climate Models with the CLIVAR 2020 ENSO Metrics Package. Bull. Amer. Meteor. Soc., 102, E193–E217 (2021). https://doi.org/10.1175/BAMS-D-19-0337.1

Taylor, K.E., Summarizing multiple aspects of model performance in a single diagram. J. Geophys. Res., 106(D7), 7183–7192, (2001). https://doi.org/10.1029/2000JD900719

Walsh, K.J.E., Fiorino, M., Landsea, C.W., & McInnes, K.L., Objectively Determined Resolution-Dependent Threshold Criteria for the Detection of Tropical Cyclones in Climate Models and Reanalyses. J. Climate, 20, 2307–2314 (2007). https://doi.org/10.1175/JCLI4074.1

Wheeler, M.C., & Hendon, H.H., An All-Season Real-Time Multivariate MJO Index: Development of an Index for Monitoring and Prediction. Mon. Wea. Rev., 132, 1917–1932 (2004). https://doi.org/10.1175/1520-0493(2004)132%3C1917:AARMMI%3E2.0.CO;2

Wheeler, M., & Kiladis, G.N., Convectively Coupled Equatorial Waves: Analysis of Clouds and Temperature in the Wavenumber-Frequency Domain. J. Atmos.Sci., 56, 374-399 (1999). [https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2](https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2)

Zarzycki, C.M., & Ullrich, P.A., Assessing sensitivities in algorithmic detection of tropical cyclones in climate data. Geophys. Res. Lett., 44, 1141–1149 (2017). https://doi.org/10.1002/2016GL071606