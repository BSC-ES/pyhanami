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
    data_source : Union[str, Path, xr.Dataset]
        Path to a dataset file or catalogue interface, or an already loaded xarray.Dataset object.
    name : str
        Name of the simulation instance (default: 'sim').

    Attributes
    ----------
    data_path : Path or None
        Path to the dataset file or catalogue interface if provided; None if dataset was passed directly.
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

    def __init__(self, data_source: str, name: str = 'sim'):
        if isinstance(data_source, (str, Path)):
            self.data_path = Path(data_source)
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data path {self.data_path} not found.")
            self.data = self._prepare_data()
        elif isinstance(data_source, xr.Dataset):
            self.data_path = None
            self.data = data_source.chunk("auto")
        else:
            raise TypeError("'data_source' must be a path string, Path object, or an xarray.Dataset")
        
        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError("'name' must be a string.")
        
        self.check_data()


    def _prepare_data(self):
        """ 
        Load data from .netcdf file or catalogue interface.
        
        Returns
        -------
        data_sim (xr.Dataset): Loaded simulation data.
        """

        data_sim = data.prepare_data(self.data_path)
        return data_sim


    def check_data(self):
        """ 
        Check provided data (available variables, units, coordinates names, ...). 
        """

        self.data = data.check_data(self.data)
        return