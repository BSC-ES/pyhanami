import numpy as np

from pathlib import Path
from pyhanami.utils import statistics
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu


# Path to available variables and metadata
VARIABLES_PATH = Path(__file__).parent / "variables.yaml"

# Path to TCs metrics and metadata
TCS_METRICS_PATH = Path(__file__).parent / "tcs_metrics.yaml"


# Available datasets parameters
DATA_PATH = Path(__file__).parent.parent / "data"

IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"
# NOTE: when updating the URL above, please also update the version, start year and path below
IBTRACS_VERSION = "v4"
IBTRACS_START_YEAR = 1980
IBTRACS_PATH = DATA_PATH / "IBTrACS.since1980.v04r01.nc"
IBTRACS_DATASET = 'wmo'

TOPOG_URL = "https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2024-grid"
TOPOG_PATH = DATA_PATH / "topog_GEBCO.nc"
TOPOG_VARNAME = 'elevation'
G = 9.80665

CYMEP_CONFIGS_PATH = DATA_PATH / "cymep_configs.csv"


# General parameters replicability test
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