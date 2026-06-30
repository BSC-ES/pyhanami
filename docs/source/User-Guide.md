# User Guide

This guide provides detailed instructions and examples for using _pyhanami_ to evaluate several features of Earth System Models (ESMs). 


 ## Set up configuration files

Before using _pyhanami_, ensure that the configuration files explained in the [Configuration](./Configuration.md) guide are properly set up. In particular, pay attention to the following aspects:

- **Variables:** for each variable to be analyzed, ensure it is defined in `src/pyhanami/config/variables.yaml` following CMIP conventions (see [Configuration](./Configuration.md#variables.yaml)). It is required that each variable (as a xarray.DataArray) has as attribute the corresponding `units`.
- **Scientific evaluation parameters:** for each climate phenomenon to be evaluated, the relevant parameters and their default values are defined in `src/pyhanami/config/scientific_evaluation_parameters.yaml` (see [Configuration](./Configuration.md#scientific_evaluation_parameters.yaml)). Ensure that the default values for the parameters are appropriate for your analysis. If necessary, modify them permanently in the configuration file or temporarily when calling the corresponding method (see [Configuration of scientific evaluation parameters](./User-Guide.md#configuration-of-scientific-evaluation-parameters)). 

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

## Diagnostic visualizations

The following subsections demonstrate how to use the diagnostic visualization methods included in the `DataDiagnostics` class.

### Time series plots

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
    obs_paths='path/to/obs', 
    obs_names='name_obs', 
    start_year=year_init, 
    end_year=year_end
)
```

To take into account when using this `time_series_plot` method:
- It plots **annual mean** time series by default, but it also supports **monthly and daily mean** time series by passing the argument `time_freq='monthly'` and `time_freq='daily'`, respectively. 
- It is possible to include in the plot the trajectories of **individual ensemble members** together with the mean by passing the argument `plot_ens=True`.
- If no `start_year` and `end_year` are specified, the whole period covered by the dataset(s) is used by default. In this case, when plotting for multiple datasets, they must have overlapping time periods.


### Spatial plots

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
    obs_path='path/to/obs', 
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

To perform a replicability test comparing two simulation datasets, initialize the `ReplicabilityTest` class with the `SimulationData` objects that you want to compare and use the `perform_rep_test` method:

```python
# Initialize ReplicabilityTest class with two SimulationData objects
tester = pyhanami.ReplicabilityTest([sim_1, sim_2])

# Perform replicability test between both datasets
tester.perform_rep_test(
    ['name_sim_1', 'name_sim_2'],
    obs_path='path/to/obs'
    start_year=year_init,
    end_year=year_end
)
```

To take into account when using this `perform_rep_test` method:
- The test is performed on all variables present in the simulation datasets as long as they are listed in `src/pyhanami/config/variables.yaml`.
- By default, the test uses a significant level of $\alpha=0.05$ and a statistical power of $P=0.80$. These values can be modified by passing the arguments `alpha` and `power`.
- If no `start_year` and `end_year` are specified, the whole period covered by the datasets is used by default. In this case, both datasets must have overlapping time periods.

Moreover, the `ReplicabilityTest` class includes methods to visualize and save the results of the test. The following shows how to retrieve, save and plot the outcome of the test for both datasets:

```python
# Load test outcome (effect size between the replicability test scores and 
# test results for all variables, seasons and regions)
effect_size_scores = tester.get_effect_sizes(
    ['name_sim_1', 'name_sim_2'],
    start_year=year_init,
    end_year=year_end
)
test_results = tester.get_test_results(
    ['name_sim_1', 'name_sim_2'],
    start_year=year_init,
    end_year=year_end
)

# Save and plot test outcome
tester.save_data(
    ['name_sim_1', 'name_sim_2'], 
    'output_path',
    start_year=year_init,
    end_year=year_end
)
tester.matrix_plot(
    ['name_sim_1', 'name_sim_2'],
    'output_path',
    start_year=year_init,
    end_year=year_end
)
```
Note that, if no years are passed when retrieving, saving or plotting the test outcome, the first replicability test performed for the selected datasets is used by default.


## Scientific skill

The following subsections demonstrate how to use the methods in the `ScientificEvaluation` class to evaluate the scientific skill of simulation datasets, i.e., how well they reproduce real-world observations.

To perform an analysis, initialize the `ScientificEvaluation` class with one or more `SimulationData` objects:

```python
# Initialize ScientificEvaluation class with one SimulationData object
skill_eval_one = pyhanami.ScientificEvaluation(sim_1)

# Initialize ScientificEvaluation class with multiple SimulationData objects
skill_eval_multiple = pyhanami.ScientificEvaluation([sim_1, sim_2])
```
It is also possible to add additional simulation datasets to an existing `ScientificEvaluation` instance with the `add_datasets` method:

```python
# Add another SimulationData object to the ScientificEvaluation instance
skill_eval_one.add_datasets(sim_2)
```

Then, the `ScientificEvaluation` class provides several methods to perform different analyses. All follow the same naming convention: `compute_<phenomenon>_scores(*args, **kwargs)`, where `<phenomenon>` is the climate phenomenon being evaluated. Each method can be called multiple times for different simulation datasets, and the results are stored internally in the same `ScientificEvaluation` instance under the corresponding simulation name.

Finally, the outcome of each analysis can be accessed later using the corresponding `<phenomenon>_scores('name_sim')` method, which returns an object providing methods to save and visualize the results. These methods accept either a single simulation dataset name or a list of dataset names. When multiple names are passed, most methods operate independently on each selected dataset, generating one separate plot per simulation dataset. However, summary table plots combine the selected datasets into a single table for easier comparison. If no simulation name is specified, all results stored in the `ScientificEvaluation` instance for the corresponding phenomenon are returned.

The next subsections explain how to use these methods and the corresponding visualization outputs for each available phenomenon.



### General analysis

To perform a general scientific skill evaluation of a simulation dataset, use the `compute_general_scores` method. This method computes several scalar scores that characterize the general scientific skill of the model by comparing to a reference observational dataset. It can be applied on any of the variables listed in `src/pyhanami/config/variables.yaml`. 

The following snippet computes the general scientific skill scores for the variable `var_name` between `year_init` and `year_end`:

```python
# Initialize ScientificEvaluation class with a SimulationData object
skill_eval = pyhanami.ScientificEvaluation(sim_1)

# Compute general scalar scores for one simulation dataset comparing to observations
skill_eval.compute_general_scores(
    'var_name',
    'name_sim_1',
    obs_path='path/to/obs',
    obs_name='name_obs',
    start_year=year_init,
    end_year=year_end
)
```
Note that the `var_name` argument can either be a single variable name (as a string) or a list of variable names. If no variable is specified, all variables in the simulation dataset are included in the analysis. Likewise, if no years (`start_year` and `end_year`) are passed, the full period covered by the simulation dataset is used by default.

Once the analysis has been performed, the outcome can be accessed using the `general_scores()` method. The following shows how to save the computed scalar scores and generate a summary table plot for the variable `var_name`:

```python
# Save and plot outcome of the general scientific skill analysis for one dataset
skill_eval.general_scores('name_sim_1').save_data('output_path')
skill_eval.general_scores('name_sim_1').scores_table('var_name', 'output_path')
```
When multiple simulation datasets are selected, the scalar scores are displayed together in a single summary table.

<!-- TO DO: Explain the colors in the summary table plot.-->


### Tropical IntraSeasonal Oscillation (ISO) analysis

To evaluate the simulation of the ISO, use the `compute_iso_scores` method. This method computes scalar scores related to ISO and requires **daily Top of Atmosphere Outgoing Longwave Radiation** (`rlut`) data, preferably covering a period of 10 years or more (ideally, at least 30 years).

The following snippet:
1. Performs an **Extended Empirical Orthogonal Function (EEOF)** analysis between `year_init_eeof` and `year_end_eeof`
2. Uses the EEOFs to compute the first two Principal Components (PCs) (**bimodal ISO indices**) between `year_init_pc` and `year_end_pc`
3. Calculates the **mean monthly frequency (seasonality)** of ISO events using the bimodal ISO indices computed in step 2

```python
# Initialize ScientificEvaluation class with a SimulationData object
# (If you have already added this dataset to an existing ScientificEvaluation object, 
# you can skip this step)
skill_eval = pyhanami.ScientificEvaluation(sim_1)

# Compute bimodal ISO indices performing an EEOF analysis on simulated data
skill_eval.compute_iso_scores(
    'name_sim_1',
    start_year_eeof=year_init_eeof,
    end_year_eeof=year_end_eeof,
    start_year_pc=year_init_pc,
    end_year_pc=year_end_pc,
)
```
If no years are passed for the EEOFs or the PCs computation, the whole period covered by the simulation dataset is used by default.

Once the analysis has been performed, the outcome can be accessed using the `iso_scores()` method. The following shows how to save the computed data and generate plots for the EEOFs, the PCs (bimodal indices) for selected years (`[year_1, year_2, year_3]`), and the mean monthly frequency (seasonality) of ISO events:

```python
# Save and plot outcome of the ISO analysis
skill_eval.iso_scores('name_sim_1').save_data('output_path')
skill_eval.iso_scores('name_sim_1').eeof_plots('output_path')
skill_eval.iso_scores('name_sim_1').pc_plots('output_path', years=[year_1, year_2, year_3])
skill_eval.iso_scores('name_sim_1').freq_plot('output_path')
```
<!-- For the future, methods such as `save_data()` operate on each selected dataset individually, while plotting methods combine the selected datasets into a single figure whenever applicable.-->

By passing the argument `obs=True` when performing the evaluation, the simulated bimodal ISO indices are compared against observations to compute several scalar scores:

```python
# Compute bimodal ISO indices and related scalar scores comparing to observations
skill_eval.compute_iso_scores(
    'name_sim_1',
    start_year_pc=year_init_pc,
    end_year_pc=year_end_pc,
    obs = True
)
```
In this case, it is not necessary to specify `start_year_eeof`and `end_year_eeof`, as the values used for the reference observational dataset ([NOAA](https://psl.noaa.gov/data/gridded/data.olrcdr.interp.html) by default) are applied automatically.

The computed data can be saved and visualized the same way as before. When observations are included, the monthly frequency of ISO events is compared between simulations and observations, and plotted together with the **scalar scores** when calling the `iso_scores().freq_plot()` method. The scores can also be retrieved with the `iso_scores().scores` attribute and summarized in a table as follows:

```python
# Plot table with computed scalar scores
skill_eval.iso_scores('name_sim_1').scores_table('output_path')
```
When multiple simulation datasets are selected, the scalar scores are displayed together in a single summary table.

In addition, when `obs=True`, the simulated PCs can optionally be adjusted before computing the scores to account for amplitude differences between simulations and observations (see [Methodology](./Methodology.md#tropical-intraseasonal-oscillation-iso) for more details). This correction can be turned on by passing the argument `correct_pc=True`.

To summarize, the object returned by `ScientificEvaluation.iso_scores()` includes methods to generate the following visualization outputs:
1. **EEOF plots** (`.iso_scores().eeof_plots`): spatial patterns of the first two EEOFs for boreal winter and boreal summer. The central longitude for these plots is set to 0º by default, but it can be modified with the argument `clon` (e.g., `clon=180`).
2. **PC plots** (`.iso_scores().pc_plots`, passing one year or a list of years): time series of the first two PCs and their amplitude for both MJO and BSISO for the specified years.
3. **Frequency plot** (`.iso_scores().freq_plot`): mean monthly frequency (seasonality) of ISO events for both MJO and BSISO, computed over the entire period covered by the dataset. If observations are set to `True`, these are included in the  frequency plot, which also shows the scalar scores ($\alpha$, $R$, $\sigma$ and $\text{TSS}$) comparing simulations and observations.
4. **Scalar scores table** (`.iso_scores().scores_table`): table summarizing the computed scalar scores. Each column is colored independently, with the best- and worst-performing dataset in a column determining the limits of the colorbar for that column.


### Specific Madden-Julian Oscillation (MJO) analysis

To evaluate the simulation of the MJO specifically, use the `compute_mjo_scores` method. This method computes several scalar scores related to MJO (see [Methodology](./Methodology.md#madden-julian-oscillation-mjo)) and requires the following variables with a **daily** frequency:
- **Top of atmosphere outgoing longwave radiation** (`rlut`)
- **Eastward wind at 200 and 850 hPa** (`ua200` and `ua850`)

The following snippet:
1. Performs a **Combined Empirical Orthogonal Function (CEOF)** analysis between `year_init_mjo` and `year_end_mjo`
2. Uses the CEOFs to compute the first two Principal Components (PCs) (**Real-Time Multivariate MJO (RMM) indices**) for the same period, projecting simulations both on the observed and simulated CEOFs
3. Computes the **lead-lag correlation** between the RMM indices (RMM1 vs RMM2)
4. Determines the climatological (annually averaged) **mean MJO amplitude and active days per phase** based on the RMM indices amplitude
5. Calculates the **MJO power spectrum** (both symmetric and antisymmetric components) between `year_init_mjo` and `year_end_mjo`
6. Computes various **scalar scores** comparing the simulated CEOFs, RMM indices, MJO activity and power to the observed ones

```python
# Initialize ScientificEvaluation class with a SimulationData object
# (If you have already added this dataset to an existing ScientificEvaluation object,
# you can skip this step)
skill_eval = pyhanami.ScientificEvaluation(sim_1)

# Compute MJO scalar scores performing a CEOF analysis and a power spectrum analysis
skill_eval.compute_mjo_scores(
    'name_sim_1',
    start_year_mjo=year_init_mjo,
    end_year_mjo=year_end_mjo,
    threshold_active_days=1
)
```

If `start_year_mjo` and `end_year_mjo` are omitted, the full period covered by the simulation dataset is used by default. Besides, `threshold_active_days` argument specifies the minimum RMM amplitude required for the MJO to be considered active on a given day. If it is not specified, the total mean RMM amplitude across the entire period is used by default as a threshold. 

Once the analysis has been performed, the results can be accessed using the `mjo_scores()` method. The following shows how to save the computed data, create plots of the CEOFs, lead-lag correlation, MJO activity, and power spectrum, and create summary tables of the scalar scores:

```python
# Save outcome of the MJO analysis
skill_eval.mjo_scores('name_sim_1').save_data('output_path')

# Plot visual outputs
skill_eval.mjo_scores('name_sim_1').ceof_plots('output_path')
skill_eval.mjo_scores('name_sim_1').lead_lag_corr_plot('output_path')
skill_eval.mjo_scores('name_sim_1').activity_per_phase_plots('output_path')
skill_eval.mjo_scores('name_sim_1').power_spectrum_plots('output_path')

# Plot summary tables of the scalar scores
skill_eval.mjo_scores('name_sim_1').ceof_corr_table('output_path')
skill_eval.mjo_scores('name_sim_1').ceof_bias_table('output_path')
skill_eval.mjo_scores('name_sim_1').activity_per_phase_bias_tables('output_path')
skill_eval.mjo_scores('name_sim_1').power_bias_table('output_path')
```
When multiple simulation datasets are selected, all table-based methods (e.g., `ceof_corr_table`) display the results for all datasets together in a single summary table, allowing direct comparison across simulations, while the other plotting methods operate on each dataset independently (generating a separate plot for each dataset).

To summarize, the object returned by `ScientificEvaluation.mjo_scores()` includes methods to generate the following visualization outputs:
1. **CEOF plots** (`MJOEvaluation.ceof_plots`): longitudinal patterns of the first two Combined EOFs for the three considered variables comparing observations and simulations.
2. **RMM indices lead-lag correlation plot** (`MJOEvaluation.lead_lag_corr_plot`): correlation between the RMM indices in terms of the termporal lag between them comparing observations and simulations.
3. **MJO activity per phase plots** (`MJOEvaluation.activity_per_phase_plots`): climatological (annually averaged) mean MJO amplitude and number of active MJO days per phase for both observations and simulations. Both quantities are plotted separately using two bar plots by default, but they can also be plotted together in the same bar plot by passing the argument `layout='together'`, allowing to more easily identify the relationship between both quantities.
4. **Power spectrum plots** (`MJOEvaluation.power_spectrum_plots`): wavenumber-frequency spectrum comparing simulations and observations. By default, the symmetric component of the spectrum is plotted, but it is also possible to plot the antisymmetric component with the argument `component='antisymmetric'`. Besides, a dashed box around the MJO region <!--(wavenumbers 0-4, and frequencies 1/80-1/30) but why??--> is included in the plots by default, pass the argument `mjo_box=False` to remove it. Finally, the plot is created for the radiation `'rlut'` by default. It is also possible to generate it for the wind variables with the argument `spectrum_var='ua200'` or `spectrum_var='ua850'`.
5. **Scalar score tables** (`MJOEvaluation.coef_corr_table`, `MJOEvaluation.coef_bias_table`, `MJOEvaluation.activity_per_phase_bias_tables`, `MJOEvaluation.power_bias_table`): tables summarizing the computed scalar scores. In each table, columns are colored independently. In the case of bias scores, white (zero bias) indicates the best-performing dataset in that column relative to the reference dataset in the first row. While in the case of correlation scores, dark green (correlation of 1) indicates the best performance.


### Tropical Cyclones (TCs) analysis

To evaluate the simulation of TCs, use the `compute_tc_scores` method. This method computes several scalar scores related to TCs (see [Methodology](./Methodology.md#tropical-cyclones-tcs)) and requires the following variables with a **6-hourly** frequency:
- **Sea level pressure** (`psl`)
- **Eastward and northward wind at 10 m** (`uas` and `vas`)
- **Geopotential height at 300 hPa and 500 hPa** (`zg300` and `zg500`)

The following snippet:
1. Detects and tracks TCs between `year_init` and `year_end`.
2. Computes several **TC metrics**, including number of TCs, their lifetime and intensity.
3. Computes various global temporal and spatial **scalar scores** from the TC metrics, comparing simulations with reference observational data ([IBTrACS](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01552) by default).

```python
# Initialize ScientificEvaluation class with a SimulationData object
# (If you have already added this dataset to an existing ScientificEvaluation object, 
# you can skip this step)
skill_eval = pyhanami.ScientificEvaluation(sim_1)

# Compute TC metrics and related scalar scores for one simulation dataset
skill_eval.compute_tc_scores(
    'name_sim_1',
    start_year_tc=year_init_tc,
    end_year_tc=year_end_tc
)
```

If no years are passed, the whole period covered by the simulation dataset is used by default.

Once the analysis has been performed, the results can be accessed using the `tc_scores()` method. The following shows how to save the computed data, create linear and spatial plots displaying the computed TC metrics, and create summary tables of the scalar score for each TC metric:

```python
# Save outcome of the TC analysis
skill_eval.tc_scores('name_sim_1').save_data('output_path')

# Plot the resulting TC metrics
skill_eval.tc_scores('name_sim_1').linear_plots('output_path')
skill_eval.tc_scores('name_sim_1').spatial_plots('output_path')

# Plot summary tables of the TC scalar scores
skill_eval.tc_scores('name_sim_1').clim_bias_table('output_path')
skill_eval.tc_scores('name_sim_1').storm_bias_table('output_path')
skill_eval.tc_scores('name_sim_1').temp_corr_table('output_path')
skill_eval.tc_scores('name_sim_1').spatial_corr_table('output_path')
```
When multiple simulation datasets are selected, all table-based methods (e.g., `clim_bias_table`) and the `linear_plots` method display the results for all datasets together in a single plot, allowing direct comparison across simulations, while the other methods operate on each dataset independently (generating a separate plot per dataset).

Note that, in the bias table plots, each column is colored independently, with the largest bias value in a column determining the limits of the colorbar for that column. In contrast, in the correlation tables, all columns share a common colorbar, ranging from -1 to 1.

Considerations regarding the 10 m wind speed:
- By default, a threshold of 10 m/s is used for TC detection. However, we recommend adjusting it according to the model's (or reanalysis') horizontal 
resolution following the criteria established in [(K.J.E. Walsh et al., 2007)](https://doi.org/10.1175/JCLI4074.1) (see [Methodology](./Methodology.md#tropical-cyclones-tcs)). This can be done by passing the argument `min_wind` when calling the `compute_tc_scores` method.
- By default, it is assumed that the wind provided is at 10 m height. If the wind data corresponds to a different height, it can still be used by passing the argument `wind_factor` when calling the `compute_tc_scores` method. This factor will be used to scale the wind data to approximate the 10 m wind speed. 


### Configuration of scientific evaluation parameters

All climate phenomena analyses require several parameters, whose default values are defined in the configuration file `src/pyhanami/config/scientific_evaluation_parameters.yaml`. These parameters can be modified permanently in the configuration file or on a case-by-case basis when calling the corresponding methods. To do so, specific configuration dataclasses are available for each phenomenon:
- Tropical IntraSeasonal Oscillation (ISO): `ISOConfig`
- Madden-Julian Oscillation (MJO): `MJOConfig`
- Tropical Cyclones (TCs): `TCConfig`

The following snippet exemplifies how to modify default parameters for the ISO analysis when calling the `compute_iso_scores` method. Parameters for other phenomena can be modified in a similar way by using the corresponding dataclass and method:

```python
# Initialize ScientificEvaluation class with a SimulationData object
# (If you have already added this dataset to an existing ScientificEvaluation object, 
# you can skip this step)
skill_eval = pyhanami.ScientificEvaluation(sim_1)

# Create a custom ISO configuration dataclass with modified parameters
iso_config = pyhanami.ISOConfig(
    lat_range=(-15,15),
    lag=10,
    n_lags=4
)

# Compute bimodal ISO indices performing an EEOF analysis on simulated data
skill_eval.compute_iso_scores(
    'name_sim_1',
    iso_config=iso_config
)
```

Note that parameter values can also be modified after creating the `ISOConfig` instance by directly updating the corresponding attributes, as shown below:

```python
# Create an ISO configuration dataclass with default parameters
iso_config = pyhanami.ISOConfig()

# Modify the default value of the lag parameter
iso_config.lag = 20
```


## General considerations
- The `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes can all be initialized without providing any `SimulationData` objects; datasets can be added later with the `add_datasets` method.
- The climate variable name `var_name` must be listed in the configuration file `src/pyhanami/config/variables.yaml`.  
- `output_path` for functions that generate a single plot can be either a directory path or a full file path including the filename. For functions that generate multiple plots, `output_path` must be a directory path. If `output_path` is not provided, the plots are displayed interactively.
- `path/to/obs` must be a path to a directory containing observation datasets, with files named following the pattern `data_obs*_{var_name}.nc`, where `var_name` matches the corresponding variable name in `src/pyhanami/config/variables.yaml`. 
- For all the spatial plots, the central longitude is set to 0º by default, but it can be modified with the argument `clon` (e.g., `clon=180`).
- For all table plots, the first row displays values for the reference dataset by default, and subsequent rows show the scores relative to the reference. However, this row can be disabled by passing the argument `reference=False` when calling the corresponding method.