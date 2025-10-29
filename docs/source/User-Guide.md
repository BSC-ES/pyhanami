# User Guide

This guide provides detailed instructions and examples for using _pyhanami_ to evaluate several features of ESMs. 

<!-- ## Set up configuration files

TO DO: add explanation of configuration files.-->

## Load simulation data

```python
import pyhanami as hnmi

sim_1 = hnmi.SimulationData('source_sim_1', name='name_sim_1')
sim_2 = hnmi.SimulationData('source_sim_2', name='name_sim_2')
```

Note that `source_sim_1` and `source_sim_2` must be paths to NetCDF files or `xarray.Dataset` objects.


## Time series plots

Generate **time series plots** between `year_init` and `year_end` for a given climate variable `var_name`:

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

Generate **spatial plots** comparing two simulation datasets:

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

Perform and plot results of a **replicability test** comparing two simulation datasets:

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

The computation of the **bimodal ISO indices** requires daily TOA outgoing longwave radiation (`olr`) data, preferably covering a period of 10 years or more (ideally, at least 30 years).

Perform an EEOF analysis between `year_init` and `year_end`, and use the resulting EEOFs to compute the bimodal ISO indices (first two PCs) for the entire period covered by the provided dataset. Then, plot these indices for `years`. Finally, use the indices to calculate and plot the mean monthly frequency (seasonality) of ISO events for the full dataset period:

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

Perform the same analysis but using observational data to compute the EEOFs, and then calculate the bimodal ISO indices for both simulations and observations. In this case, the mean monthly frequency (seasonality) of ISO events is compared between simulations and observations, and plotted together with the **TSS statistics**:

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

The computation of the **TC metrics** requires the following variables with a 6-hourly frequency:
- Sea level pressure (`psl`)
- Zonal and meridional wind at 10 m (`uas` and `vas`)
- Geopotential height at 300 hPa and 500 hPa (`zg300` and `zg500`)
- Surface geopotential  (`phis`)




### General considerations:  
- The `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes can all be initialized without providing any `SimulationData` objects; datasets can be added later with the `add_datasets` method.
- The climate variable name `var_name` must be listed in the configuration file `src/pyhanami/config/variables.yaml`.  
- `output_path` can be either a directory or a full file path including the file name. If `output_path` is not provided, the plots are displayed interactively.  
- `path_obs` must be a path to a directory containing observation datasets, with files named following the pattern `data_obs*_{var_name}.nc`, where `var_name` matches the corresponding variable name in `src/pyhanami/config/variables.yaml`. 
- For all the spatial plots, the central longitude is set to 0º by default, but it can be modified with the argument `clon`.