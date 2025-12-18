# This script prepares NOAA TOA Outgoing Longwave Radiation (rlut) data for the ISO analysis, peforms
# an EEOF analysis (both for boreal winter and boreal summer), computes the corresponding PCs, and 
# saves the results to the paths specified in `config_params.py`.

import pyhanami
import xarray as xr


# Load raw data
var_name = 'rlut'
path_noaa = ''  # Add path to raw NOAA TOA Outgoing Longwave Radiation (OLR) data NetCDF file
data_noaa_aux = xr.open_dataset(path_noaa)

# Prepare data
start_year = 1975
end_year = 2020
data_noaa_nans = data_noaa_aux.sel(time=slice(f'{start_year}-01-01',f'{end_year}-12-31'))

data_noaa = data_noaa_nans.interpolate_na(dim='time', method='linear', fill_value='extrapolate')
data_noaa[var_name].attrs['units'] = 'W m**-2'


# Filter data using Lanczos bandpass filter
lat_range = (-30, 30)
window = 141
low_freq = 1/90
high_freq = 1/25

data_unfiltered_obs = data_noaa[var_name].sortby("lat").sel(lat=slice(*lat_range)).compute()
data_filtered_obs = pyhanami.utils.iso_metrics.apply_lanczos_bandpass_filter(data_unfiltered_obs, window, low_freq, high_freq)


# Perform EEOF analysis
lag = 5
n_lags = 3
n_modes = 2

data_eeof = data_filtered_obs
eeof_summer = pyhanami.utils.iso_metrics.perform_EEOF_analysis(data_eeof, start_year, end_year, 'boreal_summer', lag, n_lags, n_modes)
eeof_winter = pyhanami.utils.iso_metrics.perform_EEOF_analysis(data_eeof, start_year, end_year, 'boreal_winter', lag, n_lags, n_modes)

# Save EEOFs
eeof_winter.to_netcdf(pyhanami.config.config_params.NOAA_EEOF_WINTER_PATH)
eeof_summer.to_netcdf(pyhanami.config.config_params.NOAA_EEOF_SUMMER_PATH)


# Compute PCs
pcs = pyhanami.utils.iso_metrics.compute_PCs(data_eeof, [eeof_winter, eeof_summer])

# Save PCs
pcs.to_netcdf(pyhanami.config.config_params.NOAA_PC_PATH)


# Save reference grid
data_grid = data_noaa.isel(time=0, drop=True).drop_vars(var_name)
data_grid.to_netcdf(pyhanami.config.config_params.NOAA_GRID_PATH)

print("NOAA ISO data preparation completed.")