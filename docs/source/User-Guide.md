# User Guide

This guide provides detailed instructions and examples for using _pyhanami_ to evaluate several features of Earth System Models (ESMs). 


 ## Set up configuration files

Before using _pyhanami_, ensure that the configuration files explained in the [Configuration](./Configuration.md) guide are properly set up. In particular, pay attention to the following aspects:

- **Variables:** for each variable to be analyzed, ensure it is defined in `src/pyhanami/config/variables.yaml` following CMIP conventions (see [Configuration](./Configuration.md#variables.yaml)). It is required that each variable (as a xarray.DataArray) has as attribute the corresponding `units`.

<!-- TO DO: finish explanation of configuration files.-->


## Load simulation data

To load simulation data, create a `SimulationData` object for each dataset you want to analyze. Each simulation dataset requires two parameters: a `data_source` (the data location/object) and a `name` (a unique identifier for the dataset):

```python
import pyhanami

# Load two simulation datasets
sim_1 = pyhanami.SimulationData('source_sim_1', name='name_sim_1')
sim_2 = pyhanami.SimulationData('source_sim_2', name='name_sim_2')
```

The `data_source` parameter can be either:
- A path to a NetCDF file containing the simulation data (as string)
- An `xarray.Dataset` object already loaded in memory

The `name` parameter is a user-defined string that uniquely identifies the dataset during the analysis.

Example with file path:
```python
sim_1 = pyhanami.SimulationData('/path/to/simulation_1.nc', name='name_sim_1')
```

Example with `xarray.Dataset` object:
```python
import xarray as xr

data_sim_1 = xr.open_dataset('/path/to/simulation_1.nc')
sim_1 = pyhanami.SimulationData(data_sim_1, name='name_sim_1')
```


## Time series plots

To generate **time series plots** between `year_init` and `year_end` for a given climate variable `var_name`, initialize the `DataDiagnostics` class with the `SimulationData` objects that you want to analyze and use the `time_series_plot` method:

```python
# Initialize DataDiagnostics class with a SimulationData object
diags = pyhanami.DataDiagnostics(sim_1)

# Plot mean time series for one dataset
diags.time_series_plot(
    'var_name', 
    'name_sim_1', 
    'output_path', 
    start_year=year_init, 
    end_year=year_end
)

# Add another SimulationData object to the DataDiagnostics object 
diags.add_datasets(sim_2)

# Plot mean time series for two datasets together
diags.time_series_plot(
    'var_name', 
    ['name_sim_1','name_sim_2'], 
    'output_path', 
    start_year=year_init, 
    end_year=year_end
)

# Plot mean time series for two datasets together with observations
diags.time_series_plot(
    'var_name', 
    ['name_sim_1','name_sim_2'], 
    'output_path', 
    obs=True, 
    obs_paths='path_obs', 
    obs_names='name_obs', 
    start_year=year_init, 
    end_year=year_end
)
```

To take into account when using this `time_series_plot` method:
- It plots **annual mean** time series by default, but it also supports **monthly and daily mean** time series by passing the argument `time_freq='monthly'` and `time_freq='daily'`, respectively. 
- It is possible to include in the plot the trajectories of **individual ensemble members** together with the mean by passing the argument `plot_ens=True`.
- If no `start_year` and `end_year` are specified, the whole period covered by the dataset(s) is used by default. In this case, when plotting for multiple datasets, they must have overlapping time periods.


## Spatial plots

To generate **spatial plots** comparing two datasets between `year_init` and `year_end`, initialize the `DataDiagnostics` class with the `SimulationData` objects that you want to analyze and use the following methods:

```python
# Initialize DataDiagnostics class with two SimulationData objects
# (If you have already added these datasets to an existing DataDiagnostics instance, 
# you can skip this step)
diags = pyhanami.DataDiagnostics([sim_1, sim_2])

# Plot spatial absolute difference between both datasets
diags.abs_diff_plot(
    'var_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path', 
    start_year=year_init, 
    end_year=year_end
)

# Plot spatial effect size between both datasets
diags.eff_size_plot(
    'var_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path', 
    start_year=year_init, 
    end_year=year_end
)

# Plot spatial bias between one dataset and observations
diags.bias_plot(
    'var_name',
    'name_sim_1',
    'output_path',
    obs_path='path_obs', 
    obs_name='name_obs', 
    start_year=year_init,
    end_year=year_end
)
```
The `abs_diff_plot`, `eff_size_plot`, and `bias_plot` methods generate the following visualization outputs, respectively:
1. **Absolute difference plot**: spatial plot displaying the absolute average difference between two simulation datasets for the given variable at the grid point level. 
2. **Effect size plot**: spatial plot showing the effect size (Cohen's _d_) between two simulation datasets for the given variable at the grid point level. Grid points in which the difference between the datasets is statistically significant (based on the _t_-test) are highlighted in the plot.
3. **Bias plot**: spatial plot of the average difference between a simulation dataset and observations for the given variable at the grid point level.

To take into account when using these methods:
- The central longitude in the plots is set to 0º by default, but it can be modified with the argument `clon` (e.g., `clon=180`).
- If no `start_year` and `end_year` are specified, the whole period covered by the datasets is used by default. In this case, both datasets must have overlapping time periods.


## Replicability test

To perform and plot results of a **replicability test** comparing two simulation datasets, initialize the `ReplicabilityTest` class with the `SimulationData` objects that you want to compare and use the `matrix_plot` method:

```python
# Initialize ReplicabilityTest class with two SimulationData objects
tester = pyhanami.ReplicabilityTest([sim_1, sim_2], obs_path='path_obs')

# Perform test
tester.matrix_plot(
    ['name_sim_1', 'name_sim_2'], 
    'output_path'
)
```

This `matrix_plot` method uses all variables present in the simulation datasets as long as they are listed in `src/pyhanami/config/variables.yaml` to perform the replicability test.


## General scientific skill analysis

To perform a general scientific skill evaluation of a simulation dataset, initialize the `ScientificEvaluation` class with the `SimulationData` object that you want to analyze and use the `compute_general_scores` method. This method computes several scalar scores related to the general scientific skill of the model by comparing to a reference observational dataset. It can be performed on any of the variables listed in `src/pyhanami/config/variables.yaml`. 

The following snippet creates a `GeneralEvaluation` instance that computes the general scientific skill scores for the variable `var_name` between `year_init` and `year_end`:

```python
# Initialize ScientificEvaluation class with a SimulationData object
sciskill = pyhanami.ScientificEvaluation(sim_1)

# Compute general scalar scores for one simulation dataset comparing to observations
general_analysis = sciskill.compute_general_scores(
    'var_name',
    'name_sim_1',
    obs_path='path_obs',
    obs_name = 'name_obs',
    start_year=year_init,
    end_year=year_end
```
Note that `var_name` can either be a single variable name (as string) or a list of variable names. If no variable is specified, all variables in the simulation dataset are considered for the analysis. Besides, if no years are passed, the whole period covered by the simulation dataset is used by default.

Moreover, the `GeneralEvaluation` class includes methods to visualize and save the results of the analysis. The following shows how to save the computed scalar scores and create a summary table plot for a given variable:

```python
# Save and plot outcome of the general scientific skill analysis
general_analysis.save_data('output_path')
general_analysis.scores_table('var_name', 'output_path')
```

<!-- TO DO: Explain the colors in the summary table plot.-->


## Tropical IntraSeasonal Oscillation (ISO) analysis

To evaluate the simulation of the ISO, initialize the `ScientificEvaluation` class with the `SimulationData` object that you want to analyze and use the `compute_iso_scores` method. This method computes scalar scores related to ISO and requires **daily Top of Atmosphere Outgoing Longwave Radiation** (`rlut`) data, preferably covering a period of 10 years or more (ideally, at least 30 years).

The following snippet creates an `ISOEvaluation` instance that:
1. Performs an **Extended Empirical Orthogonal Function (EEOF)** analysis between `year_init_eeof` and `year_end_eeof`
2. Uses the EEOFs to compute the **first two Principal Components (PCs) (bimodal ISO indices)** between `year_init_pc` and `year_end_pc`
3. Calculates the **mean monthly frequency (seasonality)** of ISO events using all the bimodal ISO indices computed in step 2

```python
# Initialize ScientificEvaluation class with a SimulationData object
# (If you have already added this dataset to an existing ScientificEvaluation object, 
# you can skip this step)
sciskill = pyhanami.ScientificEvaluation(sim_1)

# Compute bimodal ISO indices performing an EEOF analysis on simulated data
iso_analysis = sciskill.compute_iso_scores(
    'name_sim_1',
    start_year_eeof=year_init_eeof,
    end_year_eeof=year_end_eeof,
    start_year_pc=year_init_pc,
    end_year_pc=year_end_pc,
)
```
If no years are passed for the EEOFs or the PCs computation, the whole period covered by the simulation dataset is used by default.

Moreover, the `ISOEvaluation` class includes methods to visualize and save the results of the analysis. The following shows how to save the computed data and create plots for the EEOFs, the PCs (bimodal indices) for the selected years (`[year_1, year_2, year_3]`), and the mean monthly frequency (seasonality) of ISO events:

```python
# Save and plot outcome of the ISO analysis
iso_analysis.save_data('output_path')
iso_analysis.eeof_plots('output_path')
iso_analysis.pc_plots('output_path', years=[year_1, year_2, year_3])
iso_analysis.freq_plot('output_path')
```

By passing the argument `obs=True`, simulations are compared against observations to compute several scalar scores:

```python
# Compute bimodal ISO indices and related scalar scores comparing to observations
iso_analysis_obs = sciskill.compute_iso_scores(
    'name_sim_1',
    start_year_pc=year_init_pc,
    end_year_pc=year_end_pc,
    obs = True
)
```
Note that, in this case, it is not necessary to specify `start_year_eeof`and `end_year_eeof`, as the ones used for the reference observational dataset ([NOAA](https://psl.noaa.gov/data/gridded/data.olrcdr.interp.html) by default) are applied automatically.

The same way as before, it is possible to save the computed data and create plots for the EEOFs, the PCs (bimodal indices) for the selected years (`[year_1, year_2, year_3]`), and the mean monthly frequency (seasonality) of ISO events. In this case, the monthly frequency is compared between simulations and observations, and plotted together with the **scalar scores**. The scores can also be retrieved with the `ISOEvaluation.scores` attribute:

```python
# Check computed scalar scores
iso_analysis_obs.scores
```

When `obs=True`, the simulated PCs can be adjusted before computing the scores to account for amplitude differences between simulations and observations (see [Methodology](./Methodology.md#tropical-intraseasonal-oscillation-iso) for more details). This correction can be turned on by passing the argument `correct_pc=True`.

To summarize, the `ISOEvaluation` class includes methods to generate the following visualization outputs:
1. **EEOF plots** (`ISOEvaluation.eeof_plots`): spatial patterns of the first two EEOFs for boreal winter and boreal summer. The central longitude for these plots is set to 0º by default, but it can be modified with the argument `clon` (e.g., `clon=180`).
2. **PC plots** (`ISOEvaluation.pc_plots`, passing one year or a list of years): time series of the first two PCs and their amplitude for both MJO and BSISO for the specified years.
3. **Frequency plot** (`ISOEvaluation.freq_plot`): mean monthly frequency (seasonality) of ISO events for both MJO and BSISO, computed over the entire period covered by the dataset. If observations are set to `True`, these are included in the  frequency plot, which also shows the scalar scores ($\alpha$, $R$, $\sigma$ and $\text{TSS}$) comparing simulations and observations.


## Tropical Cyclones (TCs) analysis

To evaluate the simulation of TCs, initialize the `ScientificEvaluation` class with the `SimulationData` object that you want to analyze and use the `compute_tc_scores` method. This method computes several scalar scores related to TCs (see [Methodology](./Methodology.md#tropical-cyclones-tcs)) and requires the following variables with a 6-hourly frequency:
- **Sea level pressure** (`psl`)
- **Zonal and meridional wind at 10 m** (`uas` and `vas`)
- **Geopotential height at 300 hPa and 500 hPa** (`zg300` and `zg500`)

The following snippet creates a `TCEvaluation` instance that:
1. Detects and tracks TCs between `year_init` and `year_end`.
2. Computes several **TC metrics**, including number of TCs, their lifetime and intensity.
3. Computes various global temporal and spatial **scalar scores** from the TC metrics for both the provided simulation data and the reference observational data ([IBTrACS](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01552) by default).

```python
# Initialize ScientificEvaluation class with a SimulationData object
# (If you have already added this dataset to an existing ScientificEvaluation object, 
# you can skip this step)
sciskill = pyhanami.ScientificEvaluation(sim_1)

# Compute TC metrics and related scalar scores for one simulation dataset
tc_analysis= sciskill.compute_tc_scores(
    'name_sim_1',
    start_year_tc=year_init_tc,
    end_year_tc=year_end_tc
)
```

If no years are passed, the whole period covered by the simulation dataset is used by default.

Moreover, the `TCEvaluation` class includes methods to visualize and save the results of the analysis. The following shows how to save the computed data, create linear and spatial plots displaying the computed TC metrics, and create table plots summarizing the scalar score for each TC metric:

```python
# Save outcome of the TC analysis
tc_analysis.save_data('output_path')

# Plot the resulting TC metrics
tc_analysis.linear_plots('output_path')
tc_analysis.spatial_plots('output_path')

# Plot summary tables of the TC scalar scores
tc_analysis.clim_bias_table('output_path')
tc_analysis.storm_bias_table('output_path')
tc_analysis.temp_corr_table('output_path')
tc_analysis.spatial_corr_table('output_path')
```

Considerations regarding the 10 m wind speed:
- By default, a threshold of 10 m/s is used for TC detection. However, we recommend adjusting it according to the model's (or reanalysis') horizontal 
resolution following the criteria established in [(K.J.E. Walsh et al., 2007)](https://doi.org/10.1175/JCLI4074.1). This can be done by passing the argument `min_wind` when calling the `compute_tc_scores` method.
- By default, it is assumed that the wind provided is at 10 m height. If the wind data corresponds to a different height, it can still be used by passing the argument `wind_factor` when calling the `compute_tc_scores` method. This factor will be used to scale the wind data to approximate the 10 m wind speed. 


## General considerations
- The `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes can all be initialized without providing any `SimulationData` objects; datasets can be added later with the `add_datasets` method.
- The climate variable name `var_name` must be listed in the configuration file `src/pyhanami/config/variables.yaml`.  
- `output_path` for functions that generate a single plot can be either a directory path or a full file path including the filename. For functions that generate multiple plots, `output_path` must be a directory path. If `output_path` is not provided, the plots are displayed interactively.
- `path_obs` must be a path to a directory containing observation datasets, with files named following the pattern `data_obs*_{var_name}.nc`, where `var_name` matches the corresponding variable name in `src/pyhanami/config/variables.yaml`. 
- For all the spatial plots, the central longitude is set to 0º by default, but it can be modified with the argument `clon` (e.g., `clon=180`).