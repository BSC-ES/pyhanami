import numpy as np

from pyhanami.utils import statistics
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu


# Available variables (name: long_name, units)
VARIABLES = {
    'hus300': ['Specific humidity at 300hPa', 'kg kg-1'],
    'hus850': ['Specific humidity at 850hPa', 'kg kg-1'],
    'net_sfc': ['Net surface heat flux', 'W m-2'],
    'olr': ['TOA outgoing longwave radiation', 'W m-2'],
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


# General parameters
METRICS = np.array([
        ('RK08', [statistics.exp_RK_index], True),
        ('Bias', [statistics.ilamb_weighted_bias], True), 
        ('RMSE', [statistics.ilamb_weighted_RMSE], True)
    ], dtype=[('name', 'U10'), ('functions', 'O'), ('obs_needed', 'bool')])  

TESTS = {
        'KS-test': lambda ref, test: ks_2samp(ref, test)[1],
        'T-test': lambda ref, test: ttest_ind(ref, test, equal_var=False)[1], 
        'U-test': lambda ref, test: mannwhitneyu(ref, test)[1],
        'B-test': lambda ref, test: statistics.bootstrap_test(ref, test)[2]
    }

SEASONS = ['All', 'DJF', 'MAM', 'JJA', 'SON']
REGIONS = {
        'Global': lambda lat: (lat >= -90) & (lat <= 90),
        'Tropics': lambda lat: (lat >= -30) & (lat <= 30),
        'Extratropics': lambda lat: (lat < -30) | (lat > 30)
    }


# Parallelization parameters
MAX_WORKERS_VARS = None
MAX_WORKERS_GRID = None