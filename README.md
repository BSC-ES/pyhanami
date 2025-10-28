# pyhanami

_pyhanami_ is a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).


## Features

Key features of the package include:

- **Easy input handling:** load climate simulation ensembles from a NetCDF file or an `xarray.Dataset` object (in future releases, also from an intake catalogue) using the `SimulationData` class. 
- **Diagnostics plotting:** generate visualizations comparing two previously loaded simulation ensembles for selected variables using the `DataDiagnostics` class. 
These include time series plots (with the `time_series_plots` method) and spatial plots (with the `spatial_plots` method). The latter generates two plots, one for the absolute difference and another for the effect size (Cohen's _d)_ between both ensembles.
- **Replicability testing:** perform a replicability test checking the statistical indistinguishability between two previously loaded simulation ensembles using the `ReplicabilityTest` class.
- **Scientific skill evaluation:** compute metrics evaluating the Tropical IntraSeasonal Oscillation (ISO) using the `ScientificEvaluation` class. These include the computation of the bimodal ISO indices (for MJO and BSISO), as well as the calculation of related statistics comparing the indices between simulations and observations (temporal correlation (R), standard deviation ratio (σ), and Taylor Skill Score (TSS)).
- **Flexible data management:** add and compare datasets in the `DataDiagnostics`, `ReplicabilityTest`, and `ScientificEvaluation` classes even after initialization.

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



## Basic Usage

The following example demonstrates the main functionalities of _pyhanami_. For detailed usage instructions, see the [User Guide](https://earth.bsc.es/gitlab/ces/hanami/pyhanami/-/wikis/User-Guide) in the Wiki page of the project.

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


# Compute bimodal ISO indices and related statistics for one simulation dataset
sciskill = hnmi.ScientificEvaluation(sim_1)

corr, std_dev, tss = sciskill.bimodal_ISO(
    'name_sim_1',
    'output_path',
    start_year_eeof=year_init,
    end_year_eeof=year_end,
    years_pc = years,
    obs = True,
    obs_path = 'path_obs',
    obs_name = 'name_obs'
)
```



## License

This package is under the GPLv3 license, which you can find in the [LICENSE](./LICENSES/LICENSE) file.

### Third-party licenses
This project includes code and resources from the following sources:

#### Adapted code:
- [CyMeP package](https://github.com/zarzycki/cymep):
    - Used in: `src/pyhanami/utils/tcs_cymep.py`, `src/pyhanami/utils/tcs_cymep_funcs.py`
    - License: MIT License
    - Copyright (c) 2021 Colin Zarzycki

#### Code dependencies:
- [TempestExtremes package](https://github.com/ClimateGlobalChange/tempestextremes):
    - Used in: `src/pyhanami/utils/tcs_tempestextremes.py`
    - License: BSD 2-Clause License
    - Copyright (c) 2025, Paul Ullrich

#### Data sources:
- [GSV-Interface/gsv/dqc/profiles/config/variables.yaml](https://github.com/DestinE-Climate-DT/GSV-Interface/blob/master/gsv/dqc/profiles/config/variables.yaml):
    - Used for: data ranges and boundaries for plausibility checks in `src/pyhanami/config/variables.yaml`
    - License: Apache License, Version 2.0, January 2004

For full license files, see the [LICENSES](./LICENSES) directory.



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
- Iker Gonzalez: [iker.gonzalez@bsc.es](iker.gonzalez@bsc.es)

Thanks to researchers at JAMSTEC:
- Chihiro Kodama: [kodamac@jamstec.go.jp](kodamac@jamstec.go.jp)
- Tomoe Nasuno: [nasuno@jamstec.go.jp](nasuno@jamstec.go.jp)
- JAMSTEC - Research Center for Environmental Modeling and Application (CEMA)


This work, along with the replicability test methodology, were developed as part of the Hpc AlliaNce for Applications and supercoMputing Innovation (HANAMI) project, which received funding from the European High Performance Computing Joint Undertaking (EuroHPC JU) under the European Union’s Horizon Europe framework program for research and innovation and Grant Agreement No. 101136269.

### Related references

Keller, K.R., Alerany Solé, M., & Acosta, M. Replicability in Earth System Models. EGUsphere [preprint] (2025). https://doi.org/10.5194/egusphere-2025-1367

Kikuchi, K., Wang, B. & Kajikawa, Y. Bimodal representation of the tropical intraseasonal oscillation. Clim Dyn 38, 1989–2000 (2012). https://doi.org/10.1007/s00382-011-1159-1

Kikuchi, K. Extension of the bimodal intraseasonal oscillation index using JRA-55 reanalysis. Clim Dyn 54, 919–933 (2020). https://doi.org/10.1007/s00382-019-05037-z

Nakano, M., & Kikuchi, K. Seasonality of intraseasonal variability in global climate models. Geophysical Research Letters, 46, 4441–4449 (2019). https://doi.org/10.1029/2019GL082443
