import xarray as xr

from pathlib import Path
from pyhanami.utils import data


class SimulationData:
    """
    Loads and processes climate simulation data from a NetCDF file or a catalogue interface.

    This class provides functionality to read input data, perform validation, and store
    metadata such as the simulation name and file path.

    Parameters
    ----------
    data_path : str
        Path to the dataset file or catalogue interface.
    name : str
        Name of the simulation instance.

    Attributes
    ----------
    data_path : Path
        Path to the dataset file or catalogue interface.
    name : str
        Name of the simulation instance.
    data : xarray.Dataset
        Loaded dataset object with climate variables.

    Methods
    -------
    _prepare_data():
        Loads the data and applies any preprocessing.

    check_data():
        Validates that the dataset has required dimensions and variables.
    """

    def __init__(self, data_path: str, name: str = 'sim'):
        self.data_path = Path(data_path)
        self.name = name
        self.data = self._prepare_data()
        #self.check_data()   


    def _prepare_data(self):
        """ 
        Load data from .netcdf file or catalogue interface.
        
        Returns
        -------
        data_sim (xr.Dataset): Loaded simulation data.
        """

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} not found.")

        data_sim = data.prepare_data(self.data_path)
        return data_sim


    def check_data(self):
        """ 
        Check provided data (available variables, units, coordinates names, ...). 
        """

        data.check_data(self.data)
        return