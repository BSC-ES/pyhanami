# pyhanami

_pyhanami_ is a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).


### Replicability

An ESM is replicable if performing the same experiment (with the same model and forcings) using different computing environments or compilers leads to _identical_ results representing the same climate. In practice, bit-for-bit replicability is not feasible due to the chaotic nature of this type of models. However, we can aim to achieve statistically indistinguishable results. _pyhanami_ provides a replicability test to assess whether this indistinguishability holds between two given ensembles of simulated data, following the methodology presented in ([[Preprint] K. Keller et al., 2025](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1367/)).


### Scientific skill

Scientific model skill refers to the ability of an ESM to accurately represent and predict various aspects of the climate system, including its capacity to forecast future climate changes or to capture complex patterns and relationships within the system. 

Currently, evaluation of the following phenomena is available within _pyhanami_:
- **Tropical IntraSeasonal Oscillation (ISO):** large-scale convective anomalies that modulate tropical atmospheric circulation on 30-90 day timescales. This is the predominant phenomenon in the tropics throughout the year. Following [(K. Kikuchi et al., 2012)](https://link.springer.com/article/10.1007/s00382-011-1159-1), we differentiate between two modes of ISO: the **Madden-Jullian Oscillation (MJO)** and the **Boreal Summer ISO (BSISO)**. 

<!--
- **Tropical Cyclones (TCs):** warm-core, cyclonic storms characterized by heavy precipitation and strong winds that begin over tropical oceans.
-->



## Features

Key features include:

- **Easy input handling:** load climate simulation ensembles from a NetCDF file or an `xarray.Dataset` object (in future releases, also from an intake catalogue) using the `SimulationData` class. 
- **Diagnostics plotting:** generate visualizations comparing two previously loaded simulation ensembles for selected variables using the `DataDiagnostics` class. 
These include time series plots (with the `time_series_plots` method) and spatial plots (with the `spatial_plots` method). The latter generates two plots, one for the absolute difference and another for the effect size (Cohen's _d)_ between both ensembles.
- **Replicability testing:** perform a replicability test checking the statistical indistinguishability between two previously loaded simulation ensembles using the `ReplicabilityTest` class.
- **Scientific skill evaluation:** compute metrics evaluating ISO using the `ScientificEvaluation` class.
- **Flexible data management:** add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization. 


Available scientific skill metrics:
- **Bimodal ISO indices:** two separate indices are defined for the MJO and the BSISO, following [(K. Kikuchi, 2020)](https://link.springer.com/article/10.1007/s00382-019-05037-z). These indices capture the ISO behaviour during boreal winter and boreal summer, respectively. They are computed by performing an Extended Empirical Orthogonal Function (EEOF) analysis using TOA Outgoing Longwave Radiation (OLR) data, and then projecting the OLR data onto the first two EEOFs. This results in two Principal Components (PCs) for MJO and two for BSISO, which together represent the bimodal ISO indices. 

    In order to obtain scalar metrics, we also compute the **temporal correlation**, **standard deviation ratio**, and **Taylor Skill Score (TSS)** between simulations and observations using the PCs' amplitude, following [(M. Nakano et al., 2019)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL082443). Specifically, the mean monthly frequency of MJO and BSISO events (ISO  seasonality) is calculated using the amplitude of the corresponding PCs. The MJO frequency is then subtracted from the BSISO frequency, and this difference is compared between simulations and observations.

    The correlation indicates how well the phase of the ISO seasonality is reproduced by a model. While the ratio of standard deviations (model/observations) provides information about the amplitude (MJO/BSISO contrast) of the seasonality. Finally, the TSS combines both the correlation and the standard deviation, allowing to assess how well a model matches the ISO seasonality of the observations with a single score.


Future releases will include:
- **Additional scientific skill evaluation:** compute additional metrics evaluating several climate phenomena, such as TCs and precipitation.
- **Automated report generation:** produce reports including plots and statistics summary.



## Installation
The package can be installed from the source using `pip`:
```bash
git clone https://earth.bsc.es/gitlab/ces/hanami/pyhanami.git
cd pyhanami
pip install .
```



## Usage

### **1. Load simulation data** 

```python
import pyhanami as hnmi

sim_1 = hnmi.SimulationData('source_sim_1', name='name_sim_1')
sim_2 = hnmi.SimulationData('source_sim_2', name='name_sim_2')
```

Note that `source_sim_1` and `source_sim_2` must be paths to NetCDF files or `xarray.Dataset` objects.


### **2. Time series plots**

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

# Plot mean time series for both datasets together
diags.time_series_plots(
    'var_name', 
    ['name_sim_1','name_sim_2'], 
    'output_path', 
    start_year=year_init, 
    end_year=year_end
)

# Plot mean time series for both datasets together with observations
diags.time_series_plots(
    'var_name', 
    ['name_sim_1','name_sim_2'], 
    'output_path', 
    obs=True, 
    obs_paths='obs_path', 
    obs_names='obs', 
    start_year=year_init, 
    end_year=year_end
)
```

This `time_series_plots` method plots annual mean time series by default, but it also supports monthly and daily mean time series by passing the argument `time_freq='monthly'` and `time_freq='daily'`, respectively. 


### **3. Spatial plots**

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


### **4. Replicability test**

Perform and plot results of a **replicability test** comparing two simulation datasets:

```python
# Initialize ReplicabilityTest class with two SimulationData objects
tester = hnmi.ReplicabilityTest([sim_1, sim_2], obs_path='obs_path')

# Perform test
tester.matrix_plot(
    ['name_sim_1', 'name_sim_2'], 
    'output_path'
)
```

This `matrix_plot` method uses all variables from the simulation datasets that are listed in `src/pyhanami/config.py` to perform the replicability test.


### **5. Tropical IntraSeasonal Oscillation (ISO) evaluation**

The computation of the **bimodal ISO indices** requires daily TOA outgoing longwave radiation (`olr`) data, preferably covering a period of 10 years or more (ideally, at least 30 years).

Perform an EEOF analysis between `year_init` and `year_end`, and use the resulting EEOFs to compute the bimodal ISO indices (first two PCs) for the entire period covered by the provided dataset. Then, plot these indices for `years` (which can be just one year or a list of years). Finally, use the indices to calculate and plot the mean monthly frequency (seasonality) of ISO events for the full dataset period:

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
    obs_path='obs_path', 
    obs_name='obs'
)
```

By default, the `bimodal_ISO` method does not spatially plot the first two EEOFs for MJO and BSISO; to enable this, pass the argument `plot_eeofs=True`. Besides, the central longitude for the spatial EEOF plots is set to 0º by default; this can be modified with the argument `clon`. Finally, if `years_pc` is not passed as an argument, no PCs are plotted, but they are still computed for all the years present in the dataset. 


#### General considerations:  
- The climate variable name `var_name` must be listed in the `VARIABLES` dictionary in `src/pyhanami/config.py`.  
- `output_path` can be either a directory or a full file path including the file name. If `output_path` is not provided, the plots are displayed interactively.  
- `obs_path` must be a path to a directory containing observation datasets, with files named following the pattern `data_obs*_{var_name}.nc`, where `var_name` matches the corresponding variable name in `src/pyhanami/config.py`.  
- The `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes can all be initialized without providing any `SimulationData` objects; datasets can be added later with the `add_datasets` method.



## License:

This package is under GPLv3.



## Support

Create an issue or contact the authors below.



## Authors and acknowledgements

Main developers of _pyhanami_ at BSC:
- Marta Alerany Solé: [marta.alerany@bsc.es](marta.alerany@bsc.es)
- Kai Keller: [kai.keller@bsc.es](kai.keller@bsc.es)

Significant contributors at JAMSTEC:
- Masuo Nakano: [masuo@jamstec.go.jp](masuo@jamstec.go.jp)

Thanks to researchers at BSC:
- Bernardo Maraldi: [bernardo.maraldi@bsc.es](bernardo.maraldi@bsc.es)

Thanks to researchers at JAMSTEC:
- Chihiro Kodama: [kodamac@jamstec.go.jp](kodamac@jamstec.go.jp)
- Tomoe Nasuno: [nasuno@jamstec.go.jp](nasuno@jamstec.go.jp)
- JAMSTEC - Research Center for Environmental Modeling and Application (CEMA)
