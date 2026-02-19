# This script prepares NOAA TOA Outgoing Longwave Radiation (rlut) data together with ERA5
# Eastward wind at 850 hPa and 200 hPa (ua850 and ua200) data for the MJO analysis, creates a
# CEOF model using xeofs, and saves the fitted model to the path specified in `config_params.py`.

import pyhanami
import xarray as xr


# Load raw data
vars_names = ['ua850', 'ua200', 'rlut']
vars_units = ['m s**-1', 'm s**-1', 'W m**-2']
paths_obs = [
    '',
    '',
    '',
             ]  # Add path to raw NOAA and ERA5 data NetCDF files
data_obs_raw = [xr.open_dataset(path_obs) for path_obs in paths_obs]

# Prepare data
start_year = 1975
end_year = 2020

data_obs_all = []
for dataset in data_obs_raw:
    dataset = dataset.sel(time=slice(str(start_year), str(end_year)))
    dataset = dataset.sortby("lat")

    data_obs_all.append(dataset)
data_obs_nans = xr.merge(data_obs_all)

data_unfiltered_obs = data_obs_nans.interpolate_na(dim='time', method='linear', fill_value='extrapolate')
for var_name, unit in zip(vars_names, vars_units):
    data_unfiltered_obs[var_name].attrs['units'] = unit


# Filter data and remove longer-time-scale components
lat_range = (-15, 15)
rolling_window_size = 120
n_harmonics = 3
normalize_std = False

data_filtered_obs, _, _ = pyhanami.utils.mjo_scores.remove_longer_time_scale_components(data_unfiltered_obs, start_year, end_year, lat_range,
                                                                                        rolling_window_size, n_harmonics, normalize_std)
        

# Create and fit CEOF model
n_modes = 2
ceof_model = pyhanami.utils.mjo_scores.fit_CEOF_model_xeofs(data_filtered_obs, n_modes)

# Save fitted model
ceof_model.save(pyhanami.config.config_params.MJO_MODEL_PATH)


# Save reference grid
data_grid = data_unfiltered_obs.isel(time=0, drop=True).drop_vars(vars_names)
data_grid.to_netcdf(pyhanami.config.config_params.NOAA_GRID_PATH)

print("Observational MJO data preparation completed.")