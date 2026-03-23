import numpy as np

from pathlib import Path
from pyhanami.utils import statistics
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu


# Path to available variables and metadata
VARIABLES_PATH = Path(__file__).parent / "variables.yaml"

# Path to scientific evaluation parameters and metadata
SCI_EVAL_PARAMS_PATH = Path(__file__).parent / "scientific_evaluation_parameters.yaml"

# Path to TCs metrics and metadata
TCS_METRICS_PATH = Path(__file__).parent / "tc_metrics.yaml"


# General parameters

# Available datasets parameters
DATA_PATH = Path(__file__).parent.parent / "data"

# Related to general scalar evaluation data
GEN_OBS_NAME = "ERA5"
GEN_OBS_PATH = "Not available yet"  #DATA_PATH / "era5"

# Related to NOAA data
NOAA_PATH = "Not available yet"     #DATA_PATH / "noaa/data_obs_noaa_1974-2022_rlut.nc"
NOAA_GRID_PATH = DATA_PATH / "noaa/noaa_grid.nc"
NOAA_EEOF_SUMMER_PATH = DATA_PATH / "noaa/eeof_boreal_summer_noaa_1975-2020.nc"
NOAA_EEOF_WINTER_PATH = DATA_PATH / "noaa/eeof_boreal_winter_noaa_1975-2020.nc"
NOAA_PC_PATH = DATA_PATH / "noaa/pc_noaa_1975-2020.nc"
NOAA_START_YEAR = 1975
NOAA_END_YEAR = 2020

# Related to MJO data
MJO_DATA_PATH = DATA_PATH / "mjo"
MJO_VARS_PATH = "Not available yet"  #MJO_DATA_PATH / "data_obs_1975-2022_mjo.nc"
MJO_MODEL_PATH = "Not available yet"  #MJO_DATA_PATH / "mjo_obs_model_1975-2020"
MJO_GRID_PATH = NOAA_GRID_PATH
MJO_OBS_RES = 2.5
MJO_START_YEAR = 1975
MJO_END_YEAR = 2022

# Related to TCs data
TC_DATA_PATH = DATA_PATH / "tropical_cyclones"

IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc"
# NOTE: when updating the URL above, please also update the version, start year and path below
IBTRACS_VERSION = "v4"
IBTRACS_START_YEAR = 1980
IBTRACS_PATH = TC_DATA_PATH / "IBTrACS.since1980.v04r01.nc"
IBTRACS_DATASET = 'wmo'

TOPOG_URL = "https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2024-grid"
TOPOG_PATH = TC_DATA_PATH / "topog_GEBCO.nc"
TOPOG_VARNAME = 'elevation'
G = 9.80665

CYMEP_CONFIGS_PATH = TC_DATA_PATH / "cymep_configs.csv"


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
MAX_WORKERS_VARS = None         # Used in 'pyhanami/diags/Replicability.py'
MAX_WORKERS_GRID = None         # Used in 'pyhanami/diags/Diagnostics.py'