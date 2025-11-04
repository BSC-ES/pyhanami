# User Guide

This guide provides detailed instructions and examples for using _pyhanami_ to evaluate several features of ESMs. 

<!-- ## Set up configuration files

TO DO: add explanation of configuration files.-->

## Load simulation data

To load simulation data, create `SimulationData` objects for each dataset you want to analyze:

```python
import pyhanami as hnmi

sim_1 = hnmi.SimulationData('source_sim_1', name='name_sim_1')
sim_2 = hnmi.SimulationData('source_sim_2', name='name_sim_2')
```

Note that `source_sim_1` and `source_sim_2` must be paths to NetCDF files or `xarray.Dataset` objects.


## Time series plots

To generate **time series plots** between `year_init` and `year_end` for a given climate variable `var_name`, create a `DataDiagnostics` object with the `SimulationData` objects that you want to analyze and use the `time_series_plots` method:

```python
# Initialize DataDiagnostics class with a SimulationData object
diags = hnmi.DataDiagnostics(sim_1)

# Plot mean time series for one dataset
diags.time_series_plots(
    'var_name', 
    'name_sim_1', 
    'output_path', 
    start_year=year_init, 
    end_year=year_end
)

# Add SimulationData object to the DataDiagnostics object 
diags.add_datasets(sim_2)

# Plot mean time series for two datasets together
diags.time_series_plots(
    'var_name', 
    ['name_sim_1','name_sim_2'], 
    'output_path', 
    start_year=year_init, 
    end_year=year_end
)

# Plot mean time series for two datasets together with observations
diags.time_series_plots(
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

This `time_series_plots` method plots annual mean time series by default, but it also supports monthly and daily mean time series by passing the argument `time_freq='monthly'` and `time_freq='daily'`, respectively. Besides, it is possible to include in the plot the trajectories of individual ensemble members together with the mean by passing the argument `plot_ens=True`.


## Spatial plots

To generate **spatial plots** comparing two simulation datasets, create a `DataDiagnostics` object with the `SimulationData` objects that you want to analyze and use the `spatial_plots` method:

```python
# Initialize DataDiagnostics class with two SimulationData objects
# (If you have already added these datasets to an existing DataDiagnostics object, you can skip this step)
diags = hnmi.DataDiagnostics([sim_1, sim_2])

# Plot spatial plots comparing both datasets
diags.spatial_plots(
    'var_name', 
    ['name_sim_1','name_sim_2'], 
    'output_path'
)
```
The `spatial_plots` method generates the following visualization outputs:
1. **Absolute difference plot**: spatial plot displaying the absolute average difference between two simulation datasets for the given variable at the grid point level. 
2. **Effect size plot**: spatial plot showing the effect size (Cohen's _d_) between two simulation datasets for the given variable at the grid point level. Grid points in which the difference between the datasets is statistically significant (based on the _t_-test) are highlighted in the plot.

Note that the central longitude for these plots is set to 0º by default, but it can be modified with the argument `clon`.


## Replicability test

To perform and plot results of a **replicability test** comparing two simulation datasets, create a `ReplicabilityTest` object with the `SimulationData` objects that you want to compare and use the `matrix_plot` method:

```python
# Initialize ReplicabilityTest class with two SimulationData objects
tester = hnmi.ReplicabilityTest([sim_1, sim_2], obs_path='path_obs')

# Perform test
tester.matrix_plot(
    ['name_sim_1', 'name_sim_2'], 
    'output_path'
)
```

This `matrix_plot` method uses all variables from the simulation datasets that are listed in `src/pyhanami/config/variables.yaml` to perform the replicability test.


## Tropical IntraSeasonal Oscillation (ISO) evaluation

To evaluate the simulation of ISOs, create a `ScientificEvaluation` object with the `SimulationData` object that you want to analyze and use the `bimodal_ISO` method. This method computes the **bimodal ISO indices** and requires **daily TOA outgoing longwave radiation** (`olr`) data, preferably covering a period of 10 years or more (ideally, at least 30 years).

With the following snippet, the `bimodal_ISO` method will perform an EEOF analysis between `year_init` and `year_end`, and use the resulting EEOFs to compute the bimodal ISO indices (first two PCs) for the entire period covered by the provided dataset. Then, it will plot these indices for `years`. Finally, it will use the indices to calculate and plot the mean monthly frequency (seasonality) of ISO events for the full dataset period:

```python
# Initialize ScientificEvaluation class with a SimulationData object
sciskill = hnmi.ScientificEvaluation(sim_1)

# Compute and plot bimodal ISO indices performing an EEOF analysis on simulated data
sciskill.bimodal_ISO(
    'name_sim_1', 
    'output_path', 
    start_year_eeof=year_init, 
    end_year_eeof=year_end, 
    years_pc=years
)
```

With the following, the `bimodal_ISO` method will perform the same analysis as above but using observational data to compute the EEOFs, and then calculate the bimodal ISO indices for both simulations and observations. In this case, the mean monthly frequency (seasonality) of ISO events is compared between simulations and observations, and plotted together with the **TSS statistics**:

```python
# Compute and plot bimodal ISO indices and TSS statistics performing an EEOF analysis on observational data
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
```

The `bimodal_ISO` method generates the following visualization outputs:
1. **EEOF plots** (when `plot_eeofs=True` is passed): spatial patterns of the first two EEOFs for boreal winter and boreal summer. The central longitude for these plots is set to 0º by default, but it can be modified with the argument `clon`.
2. **PC plots** (when `years_pc=years` is passed, where `years` can be just one year or a list of years): time series of the first two PCs and their amplitude for both MJO and BSISO for the specified `years`. If `years_pc` is not passed as an argument, no PCs are plotted, but they are still computed for all the years present in the dataset.
3. **Frequency plot**: mean monthly frequency (seasonality) of ISO events for both MJO and BSISO, computed over the entire period covered by the dataset. If observational data is provided, the frequency plots also include the TSS statistics (R, σ and TSS) comparing simulations and observations.


## Tropical Cyclones (TCs) evaluation

To evaluate the simulation of TCs, create a `ScientificEvaluation` object with the `SimulationData` object that you want to analyze and use the `tc_metrics` method. This method computes several **TC metrics** (see [Methodology](./Methodology.md#tc-metrics)) and requires the following variables with a 6-hourly frequency:
- **Sea level pressure** (`psl`)
- **Zonal and meridional wind at 10 m** (`uas` and `vas`)
- **Geopotential height at 300 hPa and 500 hPa** (`zg300` and `zg500`)

With the following, the `tc_metrics` method will detect and track TCs between `year_init` and `year_end`, and compute various global temporal and spatial statistics for several TC metrics for both the provided simulation data and IBTrACS data. Apart from the scalar values, it will also output summary tables with the computed statistics:

```python
# Initialize ScientificEvaluation class with a SimulationData object
# (If you have already added this dataset to an existing ScientificEvaluation object, you can skip this step)
sciskill = hnmi.ScientificEvaluation(sim_1)

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