import numpy as np

from pathlib import Path
from pyhanami.utils import statistics
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu


# Path to available variables and metadata
VARIABLES_PATH = Path(__file__).parent / "variables.yaml"


# Available datasets parameters
DATA_PATH = Path(__file__).parent.parent / "data"

NOAA_GRID_PATH = DATA_PATH / "noaa/noaa_grid.nc"
NOAA_EEOF_SUMMER_PATH = DATA_PATH / "noaa/eeof_boreal_summer_noaa_1975-2020.nc"
NOAA_EEOF_WINTER_PATH = DATA_PATH / "noaa/eeof_boreal_winter_noaa_1975-2020.nc"
NOAA_PC_PATH = DATA_PATH / "noaa/pc_noaa_1975-2020.nc"
NOAA_START_YEAR = 1975
NOAA_END_YEAR = 2020


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
MAX_WORKERS_VARS = None         # Used in 'pyhanami/diags/Replicability.py'
MAX_WORKERS_GRID = None         # Used in 'pyhanami/diags/Diagnostics.py'
MAX_WORKERS_LAGBLOCKS = None    # Used in 'pyhanami/utils/iso_metrics.py'