# Troubleshooting and Known Issues

This page collects known issues that may appear when using the _pyhanami_ package, along with their causes and practical workarounds. Use it as a troubleshooting reference when an analysis fails, produces unexpected warnings, or returns incorrect output types. Each entry includes the problem description, its root cause, an occurrence example, and a recommended fix you can apply immediately.

<!-- REMEMBER TO UPDATE THE LAST UPDATE DATE WHEN MAKING CHANGES -->
_Last update:_ Oct 05, 2026

<!--
Page index:
- [OpenBLAS thread overflow during ISO evaluation](#openblas-thread-overflow-during-iso-evaluation)
- [TCD bias decoded as timedelta after reload](#tcd-bias-decoded-as-timedelta-after-reload)
-->

## During ISO evaluation: OpenBLAS thread overflow (`xeofs` fails due to `NUM_THREADS`)

### Problem
When performing a [Tropical IntraSeasonal Oscillation (ISO) evaluation](./User-Guide.md#tropical-intraseasonal-oscillation-iso-analysis) on high-resolution data, an OpenBLAS warning appears, followed by memory allocation warnings that accumulate until the process crashes.

**Observed on:** 2025-11-26

### Cause
This is an OpenBLAS threading conflict warning. OpenBLAS is compiled with a maximum thread count, but when the `xeofs` package is used to perform an Empirical Orthogonal Function analysis, the underlying `numpy`/`scipy` operations attempt to use more threads than OpenBLAS was compiled to handle. This causes a thread overflow, triggering `Bad memory unallocation!` warnings that accumulate until the process crashes.

### Example
```python
import pyhanami

# Running ISO analysis with xeofs on high-resolution data
skill_eval = pyhanami.ScientificEvaluation(data_sim)
skill_eval.compute_iso_scores('data_sim_name')

# Produces a series of warnings like:
# OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
# To avoid this warning, please rebuild your copy of OpenBLAS with a larger NUM_THREADS setting
# or set the environment variable OPENBLAS_NUM_THREADS to 64 or lower
# BLAS : Bad memory unallocation! : ...
# BLAS : Bad memory unallocation! : ...
# BLAS : Bad memory unallocation! : ...
# ... (until the process crashes)
```

### Solution
Set the `OPENBLAS_NUM_THREADS` environment variable before importing scientific libraries (for example, `numpy`, `scipy`, `xeofs`, or `pyhanami`) to limit the number of threads used by OpenBLAS:

```python
# Preset environment variable
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Now import pyhanami and run the analysis
import pyhanami

skill_eval = pyhanami.ScientificEvaluation(data_sim)
skill_eval.compute_iso_scores('data_sim_name')
```
Note that setting `OPENBLAS_NUM_THREADS=1` disables multi-threading for OpenBLAS operations, which may slow down the computation but prevents the memory allocation errors.


## During TCs evaluation: TCD bias scores decoded as `timedelta` instead of `float` after reload

### Problem
When performing a [Tropical Cyclones (TCs) evaluation](./User-Guide.md#tropical-cyclones-tcs-analysis), the bias scores for **Tropical Cyclone Days (TCD)** are incorrectly loaded as `timedelta64[ns]` instead of `float` after saving and reloading data.

**Affected scores:**
- TCD climatological bias (`clim_bias_tcd`)
- TCD storm bias (`storm_bias_tcd`)

**Observed on:** 2025-12-20

### Cause
These scores represent values in days and are saved with a `units` attribute set to `'days'`. When `xarray` detects this attribute during loading, it automatically interprets the data as time duration and converts it to `timedelta64[ns]`.

### Example
```python
import pyhanami
import xarray as xr

# After running TC analysis and saving data
skill_eval.tc_scores().save_data('output_path')

# Loading the data normally causes the issue
loaded_data = xr.open_dataset('output_path/clim_bias.nc')

print(type(loaded_data['clim_bias_tcd'].values[0]))
# Output: <class 'numpy.timedelta64'>  ← This is wrong!
```

### Solution
Use `decode_times=False` when loading the data to prevent automatic time conversion:

```python
# Load data without time decoding
loaded_data = xr.open_dataset('output_path/clim_bias.nc', decode_times=False)

print(type(loaded_data['clim_bias_tcd'].values[0]))
# Output: <class 'numpy.float64'>  ← This is correct!
```

It is also possible to convert the specific variables back to `float` once loaded:

```python
# Load normally
loaded_data = xr.open_dataset('output_path/clim_bias.nc')

# Convert problematic variables
loaded_data['clim_bias_tcd'] = loaded_data['clim_bias_tcd'].astype(float)
```

<!-- Be aware that `decode_times=False` disables automatic time decoding for the entire dataset, not only for TCD-related variables. -->