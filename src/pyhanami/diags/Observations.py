import warnings
warnings.simplefilter("always")

import numpy as np
import xarray as xr

from pathlib import Path
from pyhanami.utils import data


class ObservationData:
    """
    Retrieves and processes observational datasets for evaluation of simulations.

    This class interfaces with an external observational data source to retrieve datasets
    that match the variables and time period of a given simulation dataset. Retrieved data
    are then regridded to match the spatial resolution of the input simulation data.

    Parameters
    ----------
    data_path : str
        Path to an observations database.
    sim : xr.Dataset
        Input simulation dataset.
    name : str
        Name of the observations instance.
    regrid_method : str
        Regridding method.

    Attributes
    ----------
    data_path : Path
        Path to the observations database.
    data : xr.Dataset
        Processed observational data, regridded to match the input simulation.
    name : str
        Name of the observations instance (default: obs).
    regrid_method : str
        Regridding method.

    Methods
    -------
    _retrieve_obs(sim)
        (Not implemented) Intended to retrieve raw observations from a database.
    load_and_process(sim)
        Retrieves observational data and regrids it to match the input simulation.
    """

    def __init__(self, data_path: str, sim: xr.Dataset, name: str = 'obs', regrid_method: str = 'conservative'):
        if isinstance(data_path, (str, Path)):
            self.data_path = Path(data_path)
        else:
            raise TypeError("'data_path' must be a string or Path object.")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Observational data path {self.data_path} not found.")
        
        if not isinstance(sim, xr.Dataset):
            raise TypeError("Input simulation must be an xarray.Dataset.")
        if not sim.data_vars:
            raise ValueError("Input simulation must contain at least one climate variable.")

        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError("'name' must be a string.")
        if isinstance(regrid_method, str):
            self.regrid_method = regrid_method
        else:
            raise TypeError("'regrid_method' must be a string.")
        
        self.data = self.load_and_process(sim)


    def _retrieve_obs(self, sim):
        """ 
        Retrieve observations from database for the variables and period
        available in the given simulation ensemble. 
        
        Parameters
        ----------
        sim (xr.Dataset): Input simulation dataset.

        Returns
        -------
        obs (xr.Dataset): Dataset containing observational data for the variables in sim.
        """

        # Validate input
        if not isinstance(sim, xr.Dataset):
            raise TypeError("Input simulation must be an xarray.Dataset.")
        if not sim.data_vars:
            raise ValueError("Input simulation must contain at least one climate variable.")
        

        # Load observational data
        data_obs_vars = []
        for var in sim.data_vars:
            var_path = next(self.data_path.glob(f"data_obs*_{var}.nc"))
            data_obs_aux = xr.open_dataset(var_path, chunks="auto")

            # Align the time range with the simulations
            if "time" not in data_obs_aux.coords or "time" not in sim.coords:
                raise ValueError(f"'time' coordinate missing in either simulations or observations for variable {var}.")
            data_obs_time = data.normalize_time_format(data_obs_aux)
            
            try:
                data_obs_sel = data_obs_time.sel(time=sim.time)
            except KeyError:
                raise KeyError(f"Observations missing for some time points in variable {var}.")
            data_obs_vars.append(data_obs_sel)
        
        data_obs = xr.merge(data_obs_vars)
        return data_obs


    def load_and_process(self, sim):
        """ 
        Retrieve and regrid observational data for the variables and period
        available in the given simulation ensemble. 
        
        Parameters
        ----------
        sim (xr.Dataset): Input simulation dataset.     

        Returns
        -------
        data_new_grid (xr.Dataset): Regridded observational dataset matching the input simulation.
        """

        # Validate input
        if not isinstance(sim, xr.Dataset):
            raise TypeError("Input simulation must be an xarray.Dataset.")
        if not sim.data_vars:
            raise ValueError("Input simulation must contain at least one climate variable.")

        data_old_grid = self._retrieve_obs(sim)
        data_new_grid = data.regrid_data(data_old_grid, sim, method=self.regrid_method)

        return data_new_grid 