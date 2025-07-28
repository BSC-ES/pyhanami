# pyhanami

pyhanami is a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).


## Replicability

An ESM is replicable if performing the same experiment (with the same model and forcing) using different computing environments or compilers leads to _identical_ results representing the same climate. In practice, bit-for-bit replicability is not feasible due to the chaotic nature of this type of models. However, we can aim to achieve statistically indistinguishable results. pyhanami provides a replicability test to assess whether this indistinguishability holds between two given ensembles of simulated data, following the methodology presented in ([[Preprint] K. Keller et al., 2025](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1367/)).


## Scientific skill

Scientific model skill refers to the ability of an ESM to accurately represent and predict various aspects of the climate system, including its capacity to forecast future climate changes or to capture complex patterns and relationships within the system. (This functionality is not implemented yet)


## Features

Key features include:

- **Easy input handling:** load climate simulation ensembles from a NetCDF file or an `xarray.Dataset` object (in future releases, also from an intake catalogue) using the `SimulationData` class. 
- **Diagnostics plotting:** generate visualizations comparing two previously loaded simulation ensembles for selected variables using the `DataDiagnostics` class. 
These include time series plots (with the `time_series_plots` method) and spatial plots (with the `spatial_plots` method). The latter generates two plots, one for the absolute difference and another for the effect size (Cohen's _d)_ between both ensembles.
- **Replicability testing:** perform a replicability test checking the statistical indistinguishability between two previously loaded simulation ensembles using the `ReplicabilityTest` class.

Future releases will include:
- **Flexible data management:** add and compare datasets at a later stage. 
- **Scientific skill evaluation:** compute metrics evaluating several climate phenomena.
- **Automated report generation:** produce reports including plots and summary statistics.


## Example: Performing a replicability test
#### Input parameters
- `source_ref` and `source_test`: paths to NetCDF files or `xarray.Dataset` objects.
- `var_name`: name of climate variable to evaluate. It must be listed in `src/pyhanami/config.py`.
- `output_path`: path to save the corresponding plots. It can be either a directory or a full file path including the file name.
- `obs_path`: path to a directory containing observation datasets, stored in files following the naming pattern `data_obs*_{var_name}`.nc`.


#### Full example workflow
```python
import pyhanami as hnmi

# Retrieve data
ref = hnmi.SimulationData('source_ref', name='ref')
test = hnmi.SimulationData('source_test', name='test')

# Create plots
diags = hnmi.DataDiagnostics(ref, test)
diags.time_series_plots('var_name', 'output_path')
diags.spatial_plots('var_name', 'output_path')

# Run replicability test
tester = hnmi.ReplicabilityTest(ref, test, 'obs_path')
tester.matrix_plot('output_path')
```