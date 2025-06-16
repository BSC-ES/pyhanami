import numpy as np

# Available variables (name: long_name, units)
VARIABLES = {
    'hus300': ['Specific humidity at 300hPa', 'kg kg-1'],
    'hus850': ['Specific humidity at 850hPa', 'kg kg-1'],
    'net_sfc': ['Net surface heat flux', 'W m-2'],
    'pr': ['Total precipitation rate', 'kg m-2 s-1'],
    'psl': ['Sea level pressure', 'Pa'],
    'ta200': ['Air temperature at 200hPa', 'K'],
    'ta850': ['Air temperature at 850hPa', 'K'],
    'tas': ['Air surface temperature', 'K'],
    'tauu': ['Surface downward eastward stress', 'Pa'],
    'tauv': ['Surface downward northward stress', 'Pa'],
    'ua200': ['Zonal wind at 200hPa', 'm s-1'],
    'ua850': ['Zonal wind at 850hPa', 'm s-1'],
    'va200': ['Meridional wind at 200hPa', 'm s-1'],
    'va850': ['Meridional wind at 850hPa', 'm s-1'],
    'siconc': ['Sea ice concentration', '-'],
    'sos': ['Sea surface salinity', 'g kg-1'],
    'tos': ['Sea surface temperature', 'K']
}


# Paths


# General parameters
SEASONS = ['All', 'DJF', 'MAM', 'JJA', 'SON']
REGIONS = {'Global':slice(90,-90), 'Tropics':slice(30,-30), 'Extratropics':np.r_[slice(-90,-30), slice(30,90)]}