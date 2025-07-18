import xarray as xr

from pyhanami.utils import data


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
        return data.regrid_data(data_old_grid, sim)