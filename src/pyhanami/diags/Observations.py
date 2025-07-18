import xarray as xr

from pyhanami.utils import data


class ObservationData:
    """
    Retrieves and processes observational datasets for evaluation of simulations.

    This class interfaces with an external observational data source to retrieve datasets
    that match the variables and time period of a given simulation dataset. Retrieved data
    are then regridded to match the spatial resolution of the input simulation data.

    Parameters
    ----------
    sim : SimulationData
        Ensemble containing simulation data and metadata.

    Attributes
    ----------
    name : str
        Name of the observations instance.
    data : xr.Dataset
        Processed observational data, regridded to match the input simulation.

    Methods
    -------
    _retrieve_obs(sim)
        (Not implemented) Intended to retrieve raw observations from a database.
    load_and_process(sim)
        Retrieves observational data and regrids it to match the input simulation.
    """

    def __init__(self, sim: xr.Dataset):
        self.name = 'obs'
        self.data = self.load_and_process(sim)


    def _retrieve_obs(self, sim):
        """ 
        Retrieve observations from database for the variables
        and period available in the given simulation ensemble. 
        
        Parameters
        ----------
        sim (xr.Dataset): Input simulation dataset.

        Returns
        -------
        obs (xr.Dataset): Dataset containing observational data for the variables in sim.
        """
        raise NotImplementedError("This function is not implemented yet.")


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

        data_old_grid = self._retrieve_obs(sim)
        data_new_grid = data.regrid_data(data_old_grid, sim)

        return data_new_grid 