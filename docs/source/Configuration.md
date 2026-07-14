# Configuration

_pyhanami_ uses the configuration files in `src/pyhanami/config` to manage analysis parameters and settings. This guide explains how to set up and customize these configuration files.


## variables.yaml

The `variables.yaml` file is the main configuration file that defines climate variables and their corresponding metadata. Both the variable's shortname and long name follow CMIP conventions (see [CMIP6 Data Request](https://clipc-services.ceda.ac.uk/dreq/index/var.html)).

### Variable definitions
Required fields for each variable:
```yaml
variable_name:  # Variable shortname
  check_enabled: bool # Whether to perform physical plausibility checks
  long_name: str  # Descriptive variable name
  mask: str  # Masking information (e.g., atm/oce)
  units: str  # Units of the variable
```

Note that additional fields to check the physical plausibility of a variable, such as `max`, `min` and/or `boundaries`, can be added.


## scientific_evaluation_parameters.yaml

The `scientific_evaluation_parameters.yaml` file contains default parameters and their associated metadata for the scientific evaluation of several climate phenomena. It is organized into sections for each phenomenon, including:
- Tropical Intraseasonal Oscillation (ISO): `iso`
- Madden-Julian Oscillation (MJO): `mjo`
- Tropical Cyclones (TCs): `tc`

### Parameter definitions
Within each section, parameters are defined with the following required fields:
```yaml
parameter_name:  # Parameter shortname
  value: int | float | str  # Default value
  type: str  # Data type (e.g., int, float, tuple, str)
  units: str  # Units (if applicable)
  description: str  # Description of the parameter
```


## tc_metrics.yaml

The `tc_metrics.yaml` file contains default Tropical Cyclone (TC) metrics and their default associated metadata for the TCs scientific evaluation. It is used to:
- Determine which metrics are included in temporal and spatial scalar score calculations (`temporal`, `spatial`), and therefore which metrics appear in each type of plot.
- Build labels in plots and summary tables (`short_name`, `long_name`, `units`).

### Metric definitions
Required fields for each TC metric:
```yaml
metric_name:  # Metric shortname
  short_name: str  # Compact display name for tables/plots
  long_name: str  # Descriptive metric name
  units: str  # Units
  temporal: bool  # Whether to compute temporal scores
  spatial: bool  # Whether to compute spatial scores
```

When modifying `temporal` or `spatial`, ensure the choice is physically and methodologically consistent for that metric. These fields control which scores are computed and visualized, hence, enabling a score type for a metric where it does not make sense may produce misleading outputs.


## config_params.py (internal)

The `config_params.py` file centralizes internal package constants used across _pyhanami_. Unlike the YAML configuration files above, this module mainly defines package-level paths and default values used by the implementation.

It includes, among others:
- Paths to configuration files (e.g., `VARIABLES_PATH`, `SCI_EVAL_PARAMS_PATH`, `TCS_METRICS_PATH`).
- Paths and metadata for built-in reference data (e.g., NOAA, MJO, IBTrACS, topography).
- Replicability test defaults (e.g., metrics, tests, seasons, regions).
- Parallelization defaults (`MAX_WORKERS_VARS`, `MAX_WORKERS_GRID`).

Most users **do not need to edit** this file directly. However, advanced users may customize some constants (for example, data paths such as topography files, or default parallelization limits) when adapting _pyhanami_ to a specific local setup.

<!-- TO DO: add explanation of `config_params.py`. -->

For practical examples of using these configurations, see the [User Guide](User-Guide.md).