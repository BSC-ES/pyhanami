# User Guide

This guide provides detailed instructions and examples for using _pyhanami_ to evaluate several features of ESMs. 

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

The `data_source` parameter can be:
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

To generate **time series plots** between `year_init` and `year_end` for a given climate variable `var_name`, create a `DataDiagnostics` object with the `SimulationData` objects that you want to analyze and use the `time_series_plot` method:

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

This `time_series_plots` method plots **annual mean** time series by default, but it also supports **monthly and daily mean** time series by passing the argument `time_freq='monthly'` and `time_freq='daily'`, respectively. Besides, it is possible to include in the plot the trajectories of **individual ensemble members** together with the mean by passing the argument `plot_ens=True`.


## Spatial plots

To generate **spatial plots** comparing two simulation datasets, create a `DataDiagnostics` object with the `SimulationData` objects that you want to analyze and use the `spatial_plots` method:

```python
# Initialize DataDiagnostics class with two SimulationData objects
# (If you have already added these datasets to an existing DataDiagnostics object, you can skip this step)
diags = pyhanami.DataDiagnostics([sim_1, sim_2])

# Plot spatial absolute difference between both datasets
diags.abs_diff_plot(
    'var_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path'
)

# Plot spatial effect size between both datasets
diags.eff_size_plot(
    'var_name',
    ['name_sim_1', 'name_sim_2'],
    'output_path'
)
```
The `abs_diff_plot` and `eff_size_plot` methods generate the following visualization outputs, respectively:
1. **Absolute difference plot**: spatial plot displaying the absolute average difference between two simulation datasets for the given variable at the grid point level. 
2. **Effect size plot**: spatial plot showing the effect size (Cohen's _d_) between two simulation datasets for the given variable at the grid point level. Grid points in which the difference between the datasets is statistically significant (based on the _t_-test) are highlighted in the plot.

Note that the central longitude for these plots is set to 0º by default, but it can be modified with the argument `clon`.


## Replicability test

To perform and plot results of a **replicability test** comparing two simulation datasets, create a `ReplicabilityTest` object with the `SimulationData` objects that you want to compare and use the `matrix_plot` method:

```python
# Initialize ReplicabilityTest class with two SimulationData objects
tester = pyhanami.ReplicabilityTest([sim_1, sim_2], obs_path='path_obs')

# Perform test
tester.matrix_plot(
    ['name_sim_1', 'name_sim_2'], 
    'output_path'
)
```

This `matrix_plot` method uses all variables from the simulation datasets that are listed in `src/pyhanami/config/variables.yaml` to perform the replicability test.


## Tropical IntraSeasonal Oscillation (ISO) evaluation

To evaluate the simulation of ISOs, create a `ScientificEvaluation` object with the `SimulationData` object that you want to analyze and use the `compute_bimodal_ISO` method. This method computes the **bimodal ISO indices** and requires **daily TOA outgoing longwave radiation** (`rlut`) data, preferably covering a period of 10 years or more (ideally, at least 30 years).

The following snippet creates a `BimodalISO` instance that:
1. Performs an Extended Empirical Orthogonal Function (EEOF) analysis between `year_init_eeof` and `year_end_eeof`
2. Uses the EEOFs to compute bimodal ISO indices (first two Principal Components (PCs)) between `year_init_pc` and `year_end_pc`
3. Calculates mean monthly frequency of ISO events using all the bimodal ISO indices computed in step 2

```python
# Initialize ScientificEvaluation class with a SimulationData object
sciskill = pyhanami.ScientificEvaluation(sim_1)

# Compute and plot bimodal ISO indices performing an EEOF analysis on simulated data
bimodal_indices = sciskill.compute_bimodal_ISO(
    'name_sim_1',
    start_year_eeof=year_init_eeof,
    end_year_eeof=year_end_eeof,
    start_year_pc=year_init_pc,
    end_year_pc=year_end_pc,
)
```
If no years are passed for the EEOFs or the PCs computation, the whole period covered by the simulation dataset is used by default.

Moreover, the `BimodalISO` class includes methods to visualize and save the results of the analysis. The following shows how to save the computed data and create plots for the EEOFs, the PCs (bimodal indices) for the selected years (`[year_1, year_2, year_3]`), and the frequency of ISO events:

```python
# Save and plot outcome of the bimodal ISO analysis
bimodal_indices.save_data('output_path')
bimodal_indices.eeof_plots('output_path')
bimodal_indices.pc_plots('output_path', years=[year_1, year_2, year_3])
bimodal_indices.freq_ISO_plot('output_path')
```

By passing the argument `obs=True`, the `BimodalISO` class performs the same analysis as above but using precomputed EEOFs from NOAA data ([NOAA Interpolated OLR dataset](https://psl.noaa.gov/data/gridded/data.olrcdr.interp.html)) to generate the PCs and the frequency of ISO events for the simulation data: 
<!-- Mention that either the NOAA EEOFs or the simulated data is regridded based on the relative resolutions. -->

```python
# Compute bimodal ISO indices and related statistics comparing to observations
bimodal_indices_obs = sciskill.compute_bimodal_ISO(
    'name_sim_1',
    start_year_pc=year_init_pc,
    end_year_pc=year_end_pc,
    obs = True
)
```
Note that, in this case, it is not necessary to specify `start_year_eeof`and `end_year_eeof`, as the ones used for the NOAA dataset will be applied automatically.

The same way as before, it is possible to save the computed data and create plots for the EEOFs, the PCs (bimodal indices) for the selected years (`[year_1, year_2, year_3]`), and the frequency of ISO events. In this case, the mean monthly frequency (seasonality) of ISO events is compared between simulations and observations, and plotted together with the **TSS statistics**. The statistics can also be retrieved with the `BimodalISO.stats` attribute:

```python
# Check computed statistics
bimodal_indices_obs.stats
```

When `obs=True`, the simulated PCs can be adjusted before computing the TSS statistics to account for amplitude differences between simulations and observations (see [Methodology](./Methodology.md#tropical-intraseasonal-oscillation-iso-bimodal-iso-indices) for more details). This correction can be turned on by passing the argument `correct_pc=True`.

To summarize, the `BimodalISO` class includes methods to generate the following visualization outputs:
1. **EEOF plots** (`BimodalISO.eeof_plots`): spatial patterns of the first two EEOFs for boreal winter and boreal summer. The central longitude for these plots is set to 0º by default, but it can be modified with the argument `clon`.
2. **PC plots** (`BimodalISO.pc_plots`, passing one year or a list of years): time series of the first two PCs and their amplitude for both MJO and BSISO for the specified years.
3. **Frequency plot** (`BimodalISO.freq_ISO_plot`): mean monthly frequency (seasonality) of ISO events for both MJO and BSISO, computed over the entire period covered by the dataset. If observations are set to `True`, these are included in the  frequency plot, which also shows the TSS statistics (R, σ and TSS) comparing simulations and observations.


## Tropical Cyclones (TCs) evaluation

To evaluate the simulation of TCs, create a `ScientificEvaluation` object with the `SimulationData` object that you want to analyze and use the `tc_metrics` method. This method computes several **TC metrics** (see [Methodology](./Methodology.md#tropical-cyclones-tcs-tc-metrics)) and requires the following variables with a 6-hourly frequency:
- **Sea level pressure** (`psl`)
- **Zonal and meridional wind at 10 m** (`uas` and `vas`)
- **Geopotential height at 300 hPa and 500 hPa** (`zg300` and `zg500`)

With the following, the `tc_metrics` method will detect and track TCs between `year_init` and `year_end`, and compute various global temporal and spatial statistics for several TC metrics for both the provided simulation data and IBTrACS data. Apart from the scalar values, it will also output summary tables with the computed statistics:

```python
# Initialize ScientificEvaluation class with a SimulationData object
# (If you have already added this dataset to an existing ScientificEvaluation object, you can skip this step)
sciskill = pyhanami.ScientificEvaluation(sim_1)

# Compute TC metrics statistics for one simulation dataset
clim_bias, storm_bias, seas_corr, spat_corr = sciskill.tc_metrics(
    'name_sim_1',
    'output_path',
    start_year=year_init,
    end_year=year_end
)
```
Besides, it is possible to include observational values in the tables by adding the following arguments:

```python
# Compute TC metrics statistics for one simulation dataset together with observations
clim_bias, storm_bias, seas_corr, spat_corr = sciskill.tc_metrics(
    'name_sim_1',
    'output_path',
    start_year=year_init,
    end_year=year_end,
    obs=True, 
    obs_path='path_obs', 
    obs_name='name_obs'
)
```

Finally, by passing the argument `full_output=True`, it will also generate spatial plots of the absolute values and biases with respect to an observational reference (by default, IBTrACS) for the TC metrics.

Considerations regarding the 10 m wind speed:
- By default, a threshold of 10 m/s is used for TC detection. However, we recommend adjusting it according to the model's (or reanalysis') horizontal 
resolution following the criteria established in [(K.J.E. Walsh et al., 2007)](https://doi.org/10.1175/JCLI4074.1). This can be done by passing the argument `min_wind` when calling the `tc_metrics` method.
- By default, it is assumed that the wind passed is at 10 m height. If the wind data corresponds to a different height, it can still be used by passing the argument `wind_factor` when calling the `tc_metrics` method. This factor will be used to scale the wind data to approximate the 10 m wind speed. 


## General considerations:  
- The `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes can all be initialized without providing any `SimulationData` objects; datasets can be added later with the `add_datasets` method.
- The climate variable name `var_name` must be listed in the configuration file `src/pyhanami/config/variables.yaml`.  
- `output_path` can be either a directory or a full file path including the file name. If `output_path` is not provided, the plots are displayed interactively.  
- `path_obs` must be a path to a directory containing observation datasets, with files named following the pattern `data_obs*_{var_name}.nc`, where `var_name` matches the corresponding variable name in `src/pyhanami/config/variables.yaml`. 
- For all the spatial plots, the central longitude is set to 0º by default, but it can be modified with the argument `clon`.