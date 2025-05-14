import xarray as xr
from pathlib import Path
from pyhanami.utils import data


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
        
        data_sim = data.prepare_data(self.interface_path)
        return data_sim

    def check_data(self):
        """ Check provided data (available variables, units, ...). """  
        data.check_data(self.data)
        return