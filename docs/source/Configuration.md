# Configuration

_pyhanami_ uses the configuration files in `src/pyhanami/config` to manage analysis parameters and settings. This guide explains how to set up and customize these configuration files.

## variables.yaml

The `variables.yaml` file is the main configuration file that defines climate variables and their corresponding metadata.

### Variable Definitions
Required fields for each variable:
```yaml
variable_name:  # Variable shortname
  check_enabled: bool # Whether to perform physical plausibility checks
  long_name: str  # Descriptive name of the variable
  mask: str  # Masking information (e.g., atm/oce)
  units: str  # Units of the variable
```

Note that additional fields to check the physical plausibility of the variable can be added (e.g., max, min, boundaries, ...).

TO DO: add explanation of `config_params.py`.

For practical examples of using these configurations, see the [User Guide](User-Guide.md).
