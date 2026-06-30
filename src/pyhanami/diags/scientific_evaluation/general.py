import numpy as np
import xarray as xr
import concurrent.futures

from pathlib import Path

from pyhanami.config import config_params
from pyhanami.utils import data_general, statistics
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData
from pyhanami.utils.plots import plots_scientific_evaluation_tables

VARIABLES = data_general.load_yaml_file(config_params.VARIABLES_PATH)


class GeneralEvaluation:
    """
    Compute general scientific skill scalar scores.

    This class provides functionality for computing several scalar scores comparing simulations
    and observational data:
        - Bias (absolute and relative)
        - Centralized Root Mean Square Error (RMSE) (absolute and relative)
        - Pearson correlation coefficient

    Parameters
    ----------
    data_sim : SimulationData
        Simulation dataset to use.
    var_names : str or list[str], optional
        Climate variable(s) name(s). If None, all variables in the simulated dataset will be used.
    obs_name : str
        Name of the observational dataset to compare to (default: config_params.GEN_OBS_NAME).
    obs_path : str
        Path to the observations database (default: config_params.GEN_OBS_PATH).
    start_year, end_year : int, optional
        Initial and end years to perform the general analysis for.

    Attributes
    ----------
    var_names : list[str]
        List of climate variables names used in the analysis.
    sim_name : str
        Name of the simulation dataset.
    obs_name : str
        Name of the observational dataset
    max_workers_grid : int
        Number of parallel workers used for variable-wise computations (default:
        config_params.MAX_WORKERS_VARS).
    ensemble : bool
        Whether the simulation dataset is a single member or an ensemble.
    start_year, end_year : int
        Initial and end years to perform the general analysis for.
    scores : xr.Dataset
        Dataset containing various scalar scores comparing simulations and observational data:
            bias_abs : np.ndarray
                Area-weighted mean absolute bias.
            bias_rel : np.ndarray
                Area-weighted mean relative bias
            rmse_abs : np.ndarray
                Area-weighted mean absolute RMSE.
            rmse_rel : np.ndarray
                Area-weighted mean relative RMSE.
            pcorr : np.ndarray
                Area-weighted Pearson correlation coefficient.
    """

    def __init__(self, data_sim, var_names=None, obs_name=config_params.GEN_OBS_NAME,
                 obs_path=config_params.GEN_OBS_PATH, start_year=None, end_year=None):

        # Validate input
        if not isinstance(data_sim, SimulationData):
            raise TypeError("'data_sim' must be an instance of SimulationData.")
        if var_names is None:
            var_names = list(data_sim.data.data_vars.keys())
        if isinstance(var_names, str):
            var_names = [var_names]
        for var_name in var_names:
            if var_name not in data_sim.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset '{data_sim.name}'. "
                    f"Available variables: {list(data_sim.data.data_vars.keys())}"
                )

        # Prepare attributes
        self.var_names = var_names
        self.sim_name = data_sim.name
        self.obs_name = obs_name

        self.ensemble = True if "realization" in data_sim.data.dims else False
        self.max_workers_vars = config_params.MAX_WORKERS_VARS


        # Select year for general analysis
        start_year, end_year = data_general.validate_year_range(
            data_sim, start_year, end_year, process_name="general scalar"
        )
        self.start_year, self.end_year = start_year, end_year
        print(
            f"\tYears selected for general scalar scores computation: {self.start_year}-{self.end_year}.",
            flush=True,
        )
        data_sim_filtered = data_sim.data.sel(time=slice(str(self.start_year), str(self.end_year))).compute()

        # Load observational data
        if obs_path is None or obs_name is None:
            raise ValueError(
                "Automatic selection of observations is not implemented yet. "
                "Please provide a path and a name for the observations database."
            )
        if not isinstance(obs_path, (str, Path)) or not isinstance(obs_name, str):
            raise TypeError(
                "'obs_path' and 'obs_name' must be strings representing the observations database "
                "path and name, respectively."
            )
        data_obs = ObservationData(obs_path, data_sim_filtered[var_names], obs_name)
        data_obs_filtered = data_obs.data.compute()


        # Compute scalar scores
        bias_abs, bias_rel = self._compute_bias(data_sim_filtered, data_obs_filtered)
        rmse_abs, rmse_rel = self._compute_rmse(data_sim_filtered, data_obs_filtered)
        pcorr = self._compute_pcorr(data_sim_filtered, data_obs_filtered)

        # Store all scores in an xarray Dataset
        self.scores = xr.Dataset(
            data_vars={
                "bias_abs": (["variable"], bias_abs),
                "bias_rel": (["variable"], bias_rel),
                "rmse_abs": (["variable"], rmse_abs),
                "rmse_rel": (["variable"], rmse_rel),
                "pcorr": (["variable"], pcorr),
            },
            coords={"variable": self.var_names},
        )

        print(
            f"\nGeneral scalar scores computation completed between years {self.start_year} and {self.end_year}. "
            f"See attribute 'scores' for results.",
            flush=True,
        )
        return


    def _compute_bias_one_var(self, args):
        """
        Compute bias between simulations and observations as the area-weighted
        mean of the absolute and relative differences for the given variable.

        Parameters
        ----------
        args : tuple
            List containing:
                data_sim_mean : xr.DataArray
                    Time averaged simulation data.
                data_obs : xr.DataArray
                    Observational data.
                var_name : str
                    Climate variable.

        Returns
        -------
        bias_abs_one_var : float
            Absolute bias.
        bias_rel_one_var : float
            Relative bias.
        """

        data_sim_mean, data_obs, var_name = args

        # Compute absolute bias
        bias_abs_one_var = statistics.abs_weighted_bias(data_sim_mean, data_obs, var_name)

        # Compute relative bias
        bias_rel_one_var = statistics.ilamb_weighted_bias(data_sim_mean, data_obs, var_name)

        # Take ensemble mean when more than one member is present
        if self.ensemble:
            bias_abs_one_var = np.mean(bias_abs_one_var)
            bias_rel_one_var = np.mean(bias_rel_one_var)

        return bias_abs_one_var, bias_rel_one_var


    def _compute_bias(self, data_sim, data_obs):
        """
        Compute bias between simulations and observations as the area-weighted mean
        of the absolute and relative differences, for all variables in parallel.

        Parameters
        ----------
        data_sim : xr.DataArray
            Simulation data.
        data_obs : xr.DataArray
            Observational data.

        Returns
        -------
        bias_abs : np.ndarray
            Absolute bias.
        bias_rel : np.ndarray
            Relative bias.
        """

        # Prepare data
        data_sim_mean = data_sim.mean(dim="time")

        # Compute biases for all variables in parallel
        bias_abs = np.empty(len(self.var_names))
        bias_rel = np.empty(len(self.var_names))
        tasks = [
            (data_sim_mean[[var_name]], data_obs[[var_name]], var_name)
            for var_name in self.var_names
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_vars) as executor:
            for idx, value in enumerate(executor.map(self._compute_bias_one_var, tasks)):
                bias_abs[idx], bias_rel[idx] = value

        return bias_abs, bias_rel


    def _compute_rmse_one_var(self, args):
        """
        Compute root mean square error (RMSE) between simulations and observations
        as the area-weighted mean of the absolute and relative centralized RMSE
        for the given variable.

        Parameters
        ----------
        args : tuple
            List containing:
                data_sim : xr.DataArray
                    Simulation data.
                data_obs : xr.DataArray
                    Observational data.
                var_name : str
                    Climate variable.

        Returns
        -------
        rmse_abs : float
            Absolute RMSE.
        rmse_rel : float
            Relative RMSE.
        """
        data_sim, data_obs, var_name = args

        # Compute absolute RMSE
        rmse_abs = statistics.abs_weighted_RMSE(data_sim, data_obs, var_name)

        # Compute relative RMSE
        rmse_rel = statistics.ilamb_weighted_RMSE(data_sim, data_obs, var_name)

        # Take ensemble mean when more than one member is present
        if self.ensemble:
            rmse_abs = np.mean(rmse_abs)
            rmse_rel = np.mean(rmse_rel)

        return rmse_abs, rmse_rel


    def _compute_rmse(self, data_sim, data_obs):
        """
        Compute root mean square error (RMSE) between simulations and observations
        as the area-weighted mean of the absolute and relative centralized RMSE
        for all variables in parallel

        Parameters
        ----------
        data_sim : xr.DataArray
            Simulation data.
        data_obs : xr.DataArray
            Observational data.

        Returns
        -------
        rmse_abs : np.ndarray
            Absolute RMSE.
        rmse_rel : np.ndarray
            Relative RMSE.
        """

        # Compute RMSE for all variables in parallel
        rmse_abs = np.empty(len(self.var_names))
        rmse_rel = np.empty(len(self.var_names))
        tasks = [
            (data_sim[[var_name]], data_obs[[var_name]], var_name)
            for var_name in self.var_names
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_vars) as executor:
            for idx, value in enumerate(executor.map(self._compute_rmse_one_var, tasks)):
                rmse_abs[idx], rmse_rel[idx] = value

        return rmse_abs, rmse_rel


    def _compute_pcorr_one_var(self, args):
        """
        Compute area-weighted Pearson correlation coefficient between simulations
        and observations for the given variable.

        Parameters
        ----------
        args : tuple
            List containing:
                data_sim : xr.DataArray
                    Simulation data.
                data_obs : xr.DataArray
                    Observational data.
                var_name : str
                    Climate variable.

        Returns
        -------
        pcorr : float
            Area-weighted Pearson correlation coefficient.
        """

        # With scipy.stats.pearsonr (not used anymore, kept for reference)
        # # Prepare data
        # data_sim_mean = data_sim[var_name].mean(dim='time').values.flatten()
        # data_obs_mean = data_obs[var_name].mean(dim='time').values.flatten()

        # # Compute Pearson correlation coefficient
        # pcorr, _ = pearsonr(data_sim_mean, data_obs_mean)

        data_sim, data_obs, var_name = args

        # With xarray.corr
        # Prepare data
        data_sim_mean = data_sim[var_name].mean(dim="time")  # .stack(spatial=['lat', 'lon'])
        data_obs_mean = data_obs[var_name].mean(dim="time")  # .stack(spatial=['lat', 'lon'])

        # Compute area-weighted Pearson correlation coefficient
        weights = statistics.area_weights(data_sim_mean)
        pcorr = xr.corr(data_sim_mean, data_obs_mean, dim=["lat", "lon"], weights=weights).values

        # Take ensemble mean when more than one member is present
        if self.ensemble:
            pcorr = np.mean(pcorr)

        return pcorr


    def _compute_pcorr(self, data_sim, data_obs):
        """
        Compute Pearson correlation coefficient between simulations and observations
        for all variables in parallel.

        Parameters
        ----------
        data_sim : xr.DataArray
            Simulation data.
        data_obs : xr.DataArray
            Observational data.

        Returns
        -------
        pcorr : np.ndarray
            Pearson correlation coefficient.
        """

        # Compute Pearson correlation coefficient for all variables in parallel
        pcorr = np.empty(len(self.var_names))
        tasks = [
            (data_sim[[var_name]], data_obs[[var_name]], var_name)
            for var_name in self.var_names
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_vars) as executor:
            for idx, value in enumerate(executor.map(self._compute_pcorr_one_var, tasks)):
                pcorr[idx] = value

        return pcorr


    def save_data(self, output_path):
        """
        Save computed scalar scores to a Numpy file.

        Parameters
        ----------
        output_path : str
            Path to save the data files.
        """

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save scalar scores
        path_sim_name = self.sim_name.replace(" ", "_")
        path_obs_name = self.obs_name.replace(" ", "_")
        scores_path = (
            output_path /
            f"general_scalar_scores_{path_sim_name}-{path_obs_name}_{self.start_year}-{self.end_year}.nc"
        )

        self.scores.to_netcdf(scores_path)
        print(f"General scalar scores saved to '{scores_path}'.", flush=True)

        return


    def scores_table(self, var_names=None, output_path=None, reference=True):
        """
        Generate and save/display table plot with general scalar scores for the given variable(s).

        Parameters
        ----------
        var_names : str or list[str], optional
            Climate variable(s) name(s). If None, all variables in the analysis will be used.
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) (default: True).
        """

        # Validate input
        if var_names is None:
            var_names = self.var_names
        if isinstance(var_names, str):
            var_names = [var_names]
        for var_name in var_names:
            if var_name not in self.var_names:
                raise ValueError(
                    f"Variable '{var_name}' was not used in the general scalar analysis. "
                    f"Available variables: {self.var_names}"
                )

        # Prepare data and plotting parameters
        data = [self.scores]
        data_names = [self.obs_name, self.sim_name]
        year_range = f"{self.start_year}-{self.end_year}"

        # Create and save/display plot
        plots_scientific_evaluation_tables.general_evaluation_scores_table(
            data,
            data_names=data_names,
            year_range=year_range,
            var_names=var_names,
            ensemble=self.ensemble,
            output_path=output_path,
            reference=reference,
        )

        return
