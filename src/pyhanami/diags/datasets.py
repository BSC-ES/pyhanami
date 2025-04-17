import xarray as xr
from pathlib import Path
import pyhanami.utils as utils


class SimulationData:
    def __init__(self, interface_path: str, name: str = 'sim'):
        self.interface_path = Path(interface_path)
        self.name = name
        self.data = self._prepare_data()
        self.check_data()   

    def _prepare_data(self) -> xr.Dataset:
        """ Load data from catalogue interface."""
        if not self.interface_path.exists():
            raise FileNotFoundError(f"Interface path {self.interface_path} not found.")
        
        data_sim = utils.prepare_data(self.interface_path)
        return data_sim

    def check_data(self):
        """ Check provided data (available variables, units, ...). """  
        utils.check_data(self.data)
        return


class ObservationData:
    def __init__(self, sim: xr.Dataset):
        self.name = 'obs'
        self.data = self.load_and_process(sim)


    def _retrieve_obs(self, sim: xr.Dataset) -> xr.Dataset:
        """ Retrieve observations from database for the variables
        and period available in sim. """
        raise NotImplementedError("This function is not implemented yet.")

    def load_and_process(self, sim: xr.Dataset) -> xr.Dataset:
        """ Retrieve and regrid observational data. """
        data_old_grid = self._retrieve_obs(sim)
        return utils.regrid_data(data_old_grid, sim)