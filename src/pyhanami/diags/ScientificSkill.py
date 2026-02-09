import warnings
warnings.simplefilter("always")

import os
import re
import shutil
import cmocean
import numpy as np
import xarray as xr
import concurrent.futures
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import pearsonr
from collections.abc import Iterable
from matplotlib.colors import LinearSegmentedColormap

from pyhanami.config import config_params
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData
from pyhanami.utils import data_general, iso_scores, plot, statistics
from pyhanami.utils.tcs_scores import tcs_tempestextremes, tcs_ibtracs, tcs_cymep_main

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
    start_year, end_year : int
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
                Pearson correlation coefficient.
    """

    def __init__(self,  data_sim : SimulationData, var_names : str | list[str] = None, obs_name : str = config_params.GEN_OBS_NAME, 
                 obs_path : str = config_params.GEN_OBS_PATH, start_year : int = None, end_year : int = None):

        # Validate input
        if not isinstance(data_sim, SimulationData):
            raise TypeError("'data_sim' must be an instance of SimulationData.")
        if var_names is None:
            var_names = list(data_sim.data.data_vars.keys())
        if isinstance(var_names, str):
            var_names = [var_names]
        for var_name in var_names:
            if var_name not in data_sim.data.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{data_sim.name}'. "
                                f"Available variables: {list(data_sim.data.data_vars.keys())}")
            
        # Prepare attributes
        self.var_names = var_names
        self.sim_name = data_sim.name
        self.obs_name = obs_name

        self.ensemble = True if 'realization' in data_sim.data.dims else False
        self.max_workers_vars = config_params.MAX_WORKERS_VARS

        # Load observational data
        if obs_path is None or obs_name is None:
            raise ValueError('Automatic selection of observations is not implemented yet. '
                             'Please provide a path and a name for the observations database.')
        elif not isinstance(obs_path, (str, Path)) or not isinstance(obs_name, str):
            raise TypeError("'obs_path' and 'obs_name' must be strings representing the observations database path and name, respectively.")
        else:
            data_obs = ObservationData(obs_path, data_sim.data[[var_name]], obs_name)
        
        # Select year for general analysis
        for dataset in [data_sim, data_obs]:
            start_year, end_year = data_general.validate_year_range(dataset, start_year, end_year, process_name='general scalar')
        self.start_year, self.end_year = start_year, end_year
        data_sim_filtered = data_sim.data.sel(time=slice(str(self.start_year), str(self.end_year))).compute()
        data_obs_filtered = data_obs.data.sel(time=slice(str(self.start_year), str(self.end_year))).compute()


        # Compute scalar scores
        bias_abs, bias_rel = self._compute_bias(data_sim_filtered, data_obs_filtered)
        rmse_abs, rmse_rel = self._compute_rmse(data_sim_filtered, data_obs_filtered)
        pcorr = self._compute_pcorr(data_sim_filtered, data_obs_filtered)

        # Store all scores in an xarray Dataset
        self.scores = xr.Dataset(
            data_vars = {
                'bias_abs': (['variable'], bias_abs),
                'bias_rel': (['variable'], bias_rel),
                'rmse_abs': (['variable'], rmse_abs),
                'rmse_rel': (['variable'], rmse_rel),
                'pcorr': (['variable'], pcorr)
            },
            coords = {'variable': self.var_names},
        )

        print("\nGeneral scalar analysis computation completed.", flush=True)

        return
 

    def _compute_bias_one_var(self, args):
        """
        Compute bias between simulations and observations as the area-weighted
        mean of the absolute and relative differences for the given variable.

        Parameters
        ----------
        args : tuple
            List containing:
                data_sim_mean : xarray.DataArray
                    Time averaged simulation data.
                data_obs : xarray.DataArray
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
        data_sim : xarray.DataArray
            Simulation data.
        data_obs : xarray.DataArray
            Observational data.

        Returns
        -------
        bias_abs : np.ndarray
            Absolute bias.
        bias_rel : np.ndarray
            Relative bias.
        """

        # Prepare data
        data_sim_mean = data_sim.mean(dim='time')

        # Compute biases for all variables in parallel
        bias_abs = np.empty(len(self.var_names))
        bias_rel = np.empty(len(self.var_names))
        tasks = [(data_sim_mean[[var_name]], data_obs[[var_name]], var_name) for var_name in self.var_names]

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
                data_sim : xarray.DataArray
                    Simulation data.
                data_obs : xarray.DataArray
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
        data_sim : xarray.DataArray
            Simulation data.
        data_obs : xarray.DataArray
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
        tasks = [(data_sim[[var_name]], data_obs[[var_name]], var_name) for var_name in self.var_names]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_vars) as executor:
            for idx, value in enumerate(executor.map(self._compute_rmse_one_var, tasks)):
                rmse_abs[idx], rmse_rel[idx] = value

        return rmse_abs, rmse_rel


    def _compute_pcorr_one_var(self, args):
        """
        Compute Pearson correlation coefficient between simulations and observations
        for the given variable.

        Parameters
        ----------
        args : tuple
            List containing:
                data_sim : xarray.DataArray
                    Simulation data.
                data_obs : xarray.DataArray
                    Observational data.
                var_name : str
                    Climate variable.

        Returns
        -------     
        pcorr : float
            Pearson correlation coefficient.
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
        data_sim_mean = data_sim[var_name].mean(dim='time').stack(spatial=['lat', 'lon'])
        data_obs_mean = data_obs[var_name].mean(dim='time').stack(spatial=['lat', 'lon'])

        # Compute Pearson correlation coefficient
        pcorr = xr.corr(data_sim_mean, data_obs_mean, dim='spatial').values

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
        data_sim : xarray.DataArray
            Simulation data.
        data_obs : xarray.DataArray
            Observational data.

        Returns
        -------     
        pcorr : np.ndarray
            Pearson correlation coefficient.
        """

        # Compute Pearson correlation coefficient for all variables in parallel
        pcorr = np.empty(len(self.var_names))
        tasks = [(data_sim[[var_name]], data_obs[[var_name]], var_name) for var_name in self.var_names]

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
        path_sim_name = self.sim_name.replace(' ', '_')
        path_obs_name = self.obs_name.replace(' ', '_')      
        scores_path = output_path / f"general_scalar_scores_{path_sim_name}-{path_obs_name}_{self.start_year}-{self.end_year}.nc"
        
        self.scores.to_netcdf(scores_path)
        print(f"General scalar scores saved to '{scores_path}'.", flush=True)

        return

    
    def scores_table(self, var_names=None, output_path=None):
        """
        Generate and save/display table plot with general scalar scores for the given variable(s).

        Parameters
        ----------
        var_names : str or list[str], optional
            Climate variable(s) name(s). If None, all variables in the analysis will be used.
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Validate input
        if var_names is None:
            var_names = self.var_names
        if isinstance(var_names, str):
            var_names = [var_names]
        for var_name in var_names:
            if var_name not in self.var_names:
                raise ValueError(f"Variable '{var_name}' was not used in the general scalar analysis. "
                                 f"Available variables: {self.var_names}")


        # Prepare plot parameters        
        rows = [self.obs_name, self.sim_name]
        cbar_ticks = ['Worse performance', ' ', 'Better performance']
        colors = ("RedGreen", ['tab:red', 'white', 'tab:green'])
        data_ref = np.array([1, 1, 1])

        for var_name in var_names:
            var_name_title = VARIABLES[var_name]['long_name']
            year_range = f"{self.start_year}-{self.end_year}"
            if self.ensemble:
                # cols = [' ', r'$\overline{\text{BIAS}}$', r'$\overline{\text{eBIAS}}$', r'$\overline{\text{RMSE}}$', 
                #         r'$\overline{\text{eRMSE}}$', r'$\overline{r}_{xy}$']
                cols = [' ', r'$\overline{\text{eBIAS}}$', r'$\overline{\text{eRMSE}}$', r'$\overline{r}_{xy}$']
                title = f"Scalar scores for {var_name_title} (ensemble mean) ({year_range})"
            else:
                # cols = [' ', 'BIAS', 'eBIAS', 'RMSE', 'eRMSE', r'$r_{xy}$']
                cols = [' ', 'eBIAS', 'eRMSE', r'$r_{xy}$']
                title = f"Scalar scores for {var_name_title} ({year_range})"

            # Prepare plot data
            data_sim = self.scores[['bias_rel', 'rmse_rel', 'pcorr']].sel(variable=var_name).to_array().values
            data_plot = np.stack([data_ref, data_sim])

            # Generate and save/display plot
            general_scores_plot, _ = plot.plot_table(data_plot, title=title, col_labels=cols, row_labels=rows, cbar_ticks=cbar_ticks, colors=colors, decimals=3)

            plot.save_or_show_plot(general_scores_plot, output_path, plot_filename=f"general_scalar_scores_table_{var_name}_{self.sim_name.replace(' ', '-')}_{year_range}",
                                   plot_name="General scalar scores table plot")

        return


class ISOEvaluation:
    """
    Compute bimodal ISO indices and derived scalar scores.

    This class provides functionality for computing the bimodal ISO indices following (K. Kikuchi, 2020) and 
    plotting the results for the selected years, as well as, computing scalar scores comparing simulations and 
    observational data following (M. Nakano et al., 2019).

    Parameters
    ----------
    data_sim : SimulationData
        Simulation dataset to use.
    var_name : str
        Name given to the TOA Outgoing Longwave Radiation (OLR) variable (default: 'rlut').
    start_year_eeof, end_year_eeof : int
        Initial and end years to perform Extended Empirical Orthogonal Function (EEOF) analysis for.
    start_year_pc, end_year_pc : int
        Initial and end years to compute Principal Components (PCs) for.
    obs : bool
        If True, also plot observational data if available (default: False).
    correct_pc : bool
        Whether to adjust simulated PCs by dividing by alpha (default: False).
    lat_range : tuple
        Geographic latitude bounds (default: (-30, 30)).
    lag : int
        Lag timesteps (default: 5).
    n_lags : int
        Number of lag copies (default: 3).
    n_modes : int
        Number of EEOFs modes to compute (default: 2).
    window_size : int
        Length of the filter kernel (default: 141).
    low_freq : float
        Lower cutoff frequency (default: 1/90).
    high_freq : float
        Upper cutoff frequency (default: 1/25).

    Attributes
    ----------
    sim_name : str
        Name of the simulation dataset.
    obs_name : str or None
        Name of the observational dataset, if requested.
    obs : bool
        Whether to use observational data.
    correct_pc : bool
        Whether to adjust simulated PCs with observational data.
    eeof_summer : xarray.DataArray
        EEOFs for boreal summer.
    eeof_winter : xarray.DataArray
        EEOFs for boreal winter.
    start_year_eeof, end_year_eeof : int
        Initial and end years to perform Extended Empirical Orthogonal Function (EEOF) analysis for.
    pcs_sim : xarray.DataArray
        PCs computed from the simulation data.
    pcs_obs : xarray.DataArray or None
        PCs from observational data, or None if not computed.
    start_year_pc, end_year_pc : int
        Initial and end years to compute Principal Components (PCs) for.
    freq_sim : xarray.DataArray
        Mean monthly frequency of occurrence for simulation data.
    freq_obs : xarray.DataArray or None
        Mean monthly frequency of occurrence for observational data, or None if not computed.
    scores : dict
        Dictionary containing various scalar scores comparing simulations and observational data:
            alpha : float
                Ratio of PCs' amplitude (model/obs).
            corr : float
                Temporal correlation coefficient of the seasonality.
            sigma : float
                Ratio of the standard deviations (model/obs) of the seasonality.
            tss : float
                Taylor Skill Score.
    """

    def __init__(self, data_sim : SimulationData, var_name : str = 'rlut', start_year_eeof : int = None, end_year_eeof : int = None, 
                 start_year_pc : int = None, end_year_pc : int = None, obs : bool = False, correct_pc : bool = False,
                 lat_range : tuple = (-30, 30), lag : int = 5, n_lags : int = 3, n_modes : int = 2, window : int = 141, 
                 low_freq : float = 1/90, high_freq : float = 1/25):

        # Validate input
        if not isinstance(data_sim, SimulationData):
            raise TypeError("'data_sim' must be an instance of SimulationData.")
        if var_name not in data_sim.data.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{data_sim.name}'. "
                            f"Available variables: {list(data_sim.data.data_vars.keys())}")
        self.sim_name = data_sim.name
        self.obs_name = None
        self.obs = obs
        self.correct_pc = correct_pc

        # Select years for EEOF analysis and PCs computation
        if self.obs:
            if start_year_eeof is not None or end_year_eeof is not None:
                warnings.warn("\t'start_year_eeof' and 'end_year_eeof' are ignored when `obs=True`. "
                            f"Using predefined observational period for EEOFs: {config_params.NOAA_START_YEAR}-{config_params.NOAA_END_YEAR}.")
            self.start_year_eeof = config_params.NOAA_START_YEAR
            self.end_year_eeof = config_params.NOAA_END_YEAR
        else:
            self.start_year_eeof, self.end_year_eeof = data_general.validate_year_range(data_sim, start_year_eeof, end_year_eeof, process_name="EEOF")
            if self.start_year_eeof == self.end_year_eeof:
                raise ValueError("More than one year is needed for the EEOF analysis (at least 10 years is recommended, ideally ~ 30 years).")

        self.start_year_pc, self.end_year_pc = data_general.validate_year_range(data_sim, start_year_pc, end_year_pc, process_name="PC")
        

        # Compute/load EEOFs
        if self.obs:
            self.obs_name = 'NOAA'
            data_sim.data, self.eeof_summer, self.eeof_winter, self.pcs_obs = self._load_and_regrid_obs_data(data_sim.data)
            print(f"\tEEOF analysis loaded for '{self.obs_name}' observations between {self.start_year_eeof} and {self.end_year_eeof}.")

        # Filter simulation data
        data_unfiltered_sim = data_sim.data[var_name].sortby("lat").sel(lat=slice(*lat_range)).compute()
        data_filtered_sim = iso_scores.apply_lanczos_bandpass_filter(data_unfiltered_sim, window, low_freq, high_freq)
        print("\tSimulation data filtered for ISO timescales.", flush=True)
        
        if not self.obs:
            # Compute EEOFs from simulation data
            eeof_summer, eeof_winter = self._compute_EEOFs(data_filtered_sim, lag, n_lags, n_modes)
            self.pcs_obs = None
            self.eeof_summer = eeof_summer.compute()
            self.eeof_winter = eeof_winter.compute()
            print(f"\tEEOF analysis completed for '{self.sim_name}' data between {self.start_year_eeof} and {self.end_year_eeof}."
                  " See attributes `eeof_summer` and `eeof_winter` for results.", flush=True)


        # Compute PCs
        self.scores = {}
        self.pcs_sim, self.scores['alpha'] = self._compute_PCs(data_filtered_sim)
        print(f"\tPCs (bimodal ISO indices) computed between {self.start_year_pc} and {self.end_year_pc}."
              " See attribute `pcs_sim` (and `pcs_obs` if `obs=True`) for results.", flush=True)

        # Compute monthly frequency and scalar scores
        self.freq_sim, self.freq_obs, self.scores['R'], self.scores['sigma'], self.scores['TSS'] = self._compute_freq_and_scores()
        print(f'\tMean monthly frequency computed between {self.start_year_pc} and {self.end_year_pc}.'
              ' See attributes `freq_sim` (and `freq_obs` if `obs=True`) for results.', flush=True)
        
        # Print scalar scores
        if self.obs:
            print(f"\tTaylor Skill Score (TSS) between simulations and observations computed (stored in attribute `scores`):"
                  f"\n\t\tRatio PCs amplitudes ($\\alpha$): {self.scores['alpha']:.2f}"
                  f"\n\t\tTemporal correlation (R): {self.scores['R']:.2f}"
                  f"\n\t\tRatio standard deviations ($\\sigma$): {self.scores['sigma']:.2f}"
                  f"\n\t\tTaylor Skill Score (TSS): {self.scores['TSS']:.2f}", flush=True)
        print("\nTropical Intraseasonal Oscillation scores computation completed.", flush=True)

        return


    def _load_and_regrid_obs_data(self, data_sim):
        """
        Load observational data and regrid simulation or observational data if needed
        to match resolutions.

        Parameters
        ----------
        data_sim : xr.Dataset
            Simulation data.

        Returns
        -------
        data_sim : xr.Dataset
            Regridded simulation data if regridding was necessary, otherwise the original data.
        eeof_summer : xarray.DataArray
            EEOFs for boreal summer from observations.
        eeof_winter : xarray.DataArray
            EEOFs for boreal winter from observations.
        pcs_obs : xarray.DataArray
            PCs from observations.
        """

        # Load observational data grid 
        try:
            noaa_grid = xr.open_dataset(config_params.NOAA_GRID_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(f"NOAA grid file not found at '{config_params.NOAA_GRID_PATH}'.")

        # Compute resolutions
        sim_lat_res = abs(data_sim.lat[1] - data_sim.lat[0]).values
        sim_lon_res = abs(data_sim.lon[1] - data_sim.lon[0]).values
        obs_lat_res = abs(noaa_grid.lat[1] - noaa_grid.lat[0]).values
        obs_lon_res = abs(noaa_grid.lon[1] - noaa_grid.lon[0]).values
        
        sim_resolution = (sim_lat_res + sim_lon_res) / 2
        obs_resolution = (obs_lat_res + obs_lon_res) / 2
            
            
        # Regrid simulations if their resolution is higher
        if sim_resolution < obs_resolution:  
            try: 
                eeof_summer = xr.open_dataset(config_params.NOAA_EEOF_SUMMER_PATH)
                eeof_winter = xr.open_dataset(config_params.NOAA_EEOF_WINTER_PATH)
                pcs_obs = xr.open_dataset(config_params.NOAA_PC_PATH)
            except FileNotFoundError:   
                raise FileNotFoundError(f"Some or all required NOAA analysis files not found: "
                                        f"'{config_params.NOAA_EEOF_SUMMER_PATH}', "
                                        f"'{config_params.NOAA_EEOF_WINTER_PATH}', "
                                        f"'{config_params.NOAA_PC_PATH}'.")

            data_sim = data_general.regrid_data(data_sim, noaa_grid)
            print(f"\tSimulation data regridded to match observations' resolution (~{obs_resolution:.2f}°).")

        # Keep original grids if both resolutions are equal
        elif sim_resolution == obs_resolution:  
            try: 
                eeof_summer = xr.open_dataset(config_params.NOAA_EEOF_SUMMER_PATH)
                eeof_winter = xr.open_dataset(config_params.NOAA_EEOF_WINTER_PATH)
                pcs_obs = xr.open_dataset(config_params.NOAA_PC_PATH)
            except FileNotFoundError:   
                raise FileNotFoundError(f"Some or all required NOAA analysis files not found: "
                                        f"'{config_params.NOAA_EEOF_SUMMER_PATH}', "
                                        f"'{config_params.NOAA_EEOF_WINTER_PATH}', "
                                        f"'{config_params.NOAA_PC_PATH}'.")
        
        # Regrid observations if their resolution is higher
        elif obs_resolution < sim_resolution:
            try:
                data_obs = xr.open_dataset(config_params.NOAA_PATH)
            except FileNotFoundError:
                raise FileNotFoundError(f"NOAA observations data file not found at '{config_params.NOAA_PATH}'")
            data_obs_regrid = data_general.regrid_data(data_obs, data_sim)

            eeof_summer, eeof_winter, pcs_obs = iso_scores.prepare_NOAA_iso_data(data_obs_regrid)
            del data_obs, data_obs_regrid

            # Directly regrid EEOFs (not used anymore as redoing the EEOF analysis is now computationally affordable)
            # eeof_summer_regrid = data_general.regrid_data(eeof_summer, data_sim.sortby("lat").sel(lat=slice(*lat_range)), var='eeof')
            # eeof_winter_regrid = data_general.regrid_data(eeof_winter, data_sim.sortby("lat").sel(lat=slice(*lat_range)), var='eeof')

            # eeof_summer_copy = eeof_summer.copy()
            # eeof_winter_copy = eeof_winter.copy()
            # eeof_summer_no_eeof = eeof_summer_copy.drop_vars('eeof').drop_dims(['lat', 'lon'])
            # eeof_winter_no_eeof = eeof_winter_copy.drop_vars('eeof').drop_dims(['lat', 'lon'])

            # eeof_summer = xr.merge([eeof_summer_regrid, eeof_summer_no_eeof])
            # eeof_winter = xr.merge([eeof_winter_regrid, eeof_winter_no_eeof])

            print(f"\tObservational data regridded to match simulations' resolution (~{sim_resolution:.2f}°).")

        return data_sim, eeof_summer, eeof_winter, pcs_obs


    def _compute_EEOFs(self, data_sim, lag=5, n_lags=3, n_modes=2):
        """
        Compute Extended Empirical Orthogonal Functions (EEOFs) from the simulation data for the requested years.
        
        Parameters
        ----------
        data_sim : xarray.DataArray
            Filtered simulation data.
        lag : int
            Lag timesteps (default: 5).
        n_lags : int
            Number of lag copies (default: 3).
        n_modes : int
            Number of EEOFs modes to compute (default: 2).

        Returns
        -------
        eeof_summer : xarray.DataArray
            EEOFs for boreal summer for simulated data.
        eeof_winter : xarray.DataArray
            EEOFs for boreal winter for simulated data.
        """

        data_eeof_sim = data_sim.sel(time=slice(f'{self.start_year_eeof}-01-01', f'{self.end_year_eeof}-12-31'))
        eeof_summer = iso_scores.perform_EEOF_analysis(data_eeof_sim, self.start_year_eeof, self.end_year_eeof, 'boreal_summer', lag, n_lags, n_modes)
        eeof_winter = iso_scores.perform_EEOF_analysis(data_eeof_sim, self.start_year_eeof, self.end_year_eeof, 'boreal_winter', lag, n_lags, n_modes)

        return eeof_summer, eeof_winter


    def _compute_PCs(self, data_sim):
        """
        Compute Principal Components (PCs) from the simulation data for the requested years, 
        and adjust them with observational data if requested.
        
        Parameters
        ----------
        data_sim : xarray.DataArray
            Filtered simulation data.

        Returns
        -------
        pcs_sim : xarray.DataArray
            PCs computed from the simulation data.
        alpha : float
            Ratio between simulated and observed PCs' amplitude.
        """

        data_pcs_sim = data_sim.sel(time=slice(f'{self.start_year_pc}-01-01', f'{self.end_year_pc}-12-31'))
        pcs_sim = iso_scores.compute_PCs(data_pcs_sim, [self.eeof_winter, self.eeof_summer])

        alpha = None
        if self.obs: 
            # Compute alpha (only with raw PCs' amplitudes)
            alpha_num = pcs_sim[f'amp_MJO_raw'].mean(dim='time') + pcs_sim[f'amp_BSISO_raw'].mean(dim='time')
            alpha_den = self.pcs_obs[f'amp_MJO_raw'].mean(dim='time') + self.pcs_obs[f'amp_BSISO_raw'].mean(dim='time')
            alpha = ((alpha_num / alpha_den).values).item()

            # Adjust PCs if requested
            if self.correct_pc:
                pcs_sim = iso_scores.adjust_PCs(pcs_sim, alpha)
                pcs_sim.attrs['alpha'] = alpha
                print(f"\tSimulated PCs have been adjusted using the '{self.obs_name}' observations.", flush=True)
        elif self.correct_pc:
            warnings.warn("Simulated PCs cannot be adjusted without observations. Continuing without modification.")

        return pcs_sim, alpha


    def _compute_freq_and_scores(self):
        """
        Compute monthly frequency of ISO events, and statistics comparing simulation and
        observational data if requested.
        
        Returns
        -------
        freq_sim : xr.DataArray
            Mean monthly frequency of occurrence for simulation data.
        freq_obs : xr.DataArray or None
            Mean monthly frequency of occurrence for observational data, or None if not computed.
        corr : float
            Temporal correlation coefficient of the seasonality.
        sigma : float
            Ratio of the standard deviations (model/obs) of the seasonality.
        tss : float
            Taylor Skill Score.
        """

        events_sim = self.pcs_sim['label']
        freq_sim = iso_scores.compute_freq_ISO(events_sim)

        if self.obs:
            # TO ADD: precompute freq_obs and just load it here?
            events_obs = self.pcs_obs['label']
            freq_obs = iso_scores.compute_freq_ISO(events_obs)
            corr, sigma, tss = iso_scores.compute_TSS(freq_sim, freq_obs)

            # Add statistics as attributes to freq_sim
            freq_sim.attrs['R'] = corr
            freq_sim.attrs['sigma'] = sigma
            freq_sim.attrs['TSS'] = tss
        else:
            freq_obs, corr, sigma, tss = None, None, None, None

        return freq_sim, freq_obs, corr, sigma, tss


    def save_data(self, output_path):
        """
        Save computed EEOFs, PCs, and frequency of ISO events to NetCDF files.

        Parameters
        ----------
        output_path : str
            Path to save the data files.
        """

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        year_range_eeof = f"{self.start_year_eeof}-{self.end_year_eeof}"
        year_range_pcs = f"{self.start_year_pc}-{self.end_year_pc}"

        # Determine dataset names for filenames
        sim_name_file = self.sim_name.replace(' ', '-') 

        if self.obs:
            obs_name_file = self.obs_name.replace(' ', '-')
            name = self.obs_name
        else:
            name = self.sim_name
        name_file = name.replace(' ', '-') 


        # Save EEOFS
        eeof_summer_path = output_path / f"eeof_boreal_summer_{name_file}_{year_range_eeof}.nc"
        self.eeof_summer.to_netcdf(eeof_summer_path)
        print(f"EEOFs computed from '{name}' for boreal summer saved to '{eeof_summer_path}'.", flush=True)

        eeof_winter_path = output_path / f"eeof_boreal_winter_{name_file}_{year_range_eeof}.nc"
        self.eeof_winter.to_netcdf(eeof_winter_path)
        print(f"EEOFs computed from '{name}' for boreal winter saved to '{eeof_winter_path}'.", flush=True)

        # Save PCs
        if self.obs:
            pcs_sim_path = output_path / f"pcs_{sim_name_file}_projected_on_{obs_name_file}_{year_range_pcs}.nc"
            self.pcs_sim.to_netcdf(pcs_sim_path)
            print(f"PCs (bimodal ISO indices) computed for '{self.sim_name}' simulations saved to '{pcs_sim_path}'.", flush=True)

            pcs_obs_path = output_path / f"pcs_{obs_name_file}_projected_{config_params.NOAA_START_YEAR}-{config_params.NOAA_END_YEAR}.nc"
            self.pcs_obs.to_netcdf(pcs_obs_path)
            print(f"PCs (bimodal ISO indices) computed for '{self.obs_name}' observations saved to '{pcs_obs_path}'.", flush=True)
        else:
            pcs_sim_path = output_path / f"pcs_{sim_name_file}_projected_{year_range_pcs}.nc"
            self.pcs_sim.to_netcdf(pcs_sim_path)
            print(f"PCs (bimodal ISO indices) computed for '{self.sim_name}' simulations saved to '{pcs_sim_path}'.", flush=True)

        # Save frequency of ISO events
        if self.obs:
            freq_sim_path = output_path / f"freq_ISO_{sim_name_file}_projected_on_{obs_name_file}_{year_range_pcs}.nc"
            self.freq_sim.to_netcdf(freq_sim_path)
            print(f"Mean monthly frequency of ISO events computed for '{self.sim_name}' saved to '{freq_sim_path}'.", flush=True)

            freq_obs_path = output_path / f"freq_ISO_{obs_name_file}_projected_{config_params.NOAA_START_YEAR}-{config_params.NOAA_END_YEAR}.nc"
            self.freq_obs.to_netcdf(freq_obs_path)
            print(f"Mean monthly frequency of ISO events computed for '{self.obs_name}' observations saved to '{freq_obs_path}'.", flush=True)
        else:
            freq_sim_path = output_path / f"freq_ISO_{sim_name_file}_projected_{year_range_pcs}.nc"
            self.freq_sim.to_netcdf(freq_sim_path)
            print(f"Mean monthly frequency of ISO events computed for '{self.sim_name}' saved to '{freq_sim_path}'.", flush=True)
        return


    def eeof_plots(self, output_path=None, var_name='rlut', clon=0):
        """
        Generate and save/display plots of the EEOFs for boreal summer and winter
        for the simulation data or, if `obs=True`, for the observational data.

        Parameters
        ----------
        output_path : str, optional
            Path to save plots. If None, plots are displayed but not saved.
        var_name : str
            Name given to the TOA Outgoing Longwave Radiation (OLR) variable (default: 'rlut').
        clon : int
            Central longitude for the plots (default: 0).
        """

        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        
        year_range = f"{self.start_year_eeof}-{self.end_year_eeof}"

        # Determine dataset name for titles
        if self.obs:
            name = self.obs_name
        else:
            name = self.sim_name
        name_file = name.replace(' ', '-')


        # Plot EEOFs for borean summer
        eeofs_plot, _ = plot.plot_eeofs(self.eeof_summer, clon=clon, title=f"BSISO convective pattern '{name}' (JJASO {year_range})",
                                cb_label=f"scaled EEOF ({VARIABLES[var_name]['units']})", cmap=LinearSegmentedColormap.from_list("GreenOrange", ['tab:green', 'white', 'tab:orange']))       
        
        plot.save_or_show_plot(eeofs_plot, output_path, plot_filename=f"eeof_boreal_summer_{name_file}_{year_range}_clon_{clon}",
                               plot_name="BSISO EEOFs plot", custom_name=False)


        # Plot EEOFs for borean winter
        eeofw_plot, _ = plot.plot_eeofs(self.eeof_winter, clon=clon, title=f"MJO convective pattern '{name}' (DJFMA {year_range})",
                                        cb_label=f"scaled EEOF ({VARIABLES[var_name]['units']})", cmap=LinearSegmentedColormap.from_list("BlueRed", ['tab:blue', 'white', 'tab:red']))
        
        plot.save_or_show_plot(eeofw_plot, output_path, plot_filename=f"eeof_boreal_winter_{name_file}_{year_range}_clon_{clon}",
                               plot_name="MJO EEOFs plot", custom_name=False)

        return


    def pc_plots(self, output_path=None, years=None, data='sim'):
        """
        Generate and save/display plots of the PCs (bimodal ISO indices) for the selected years.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save plots. If None, plots are displayed but not saved.
        years : int or list[int]
            Years to plot the PCs for.
        data : str
            Whether to plot 'sim' (simulation) or 'obs' (observations) PCs (default: 'sim').
        """

        # Determine dataset to plot
        if data == 'sim':
            pcs_data = self.pcs_sim
            if self.obs:
                name_title = f"'{self.sim_name}' projected on '{self.obs_name}'" 
                name_projected = f"_{self.obs_name.replace(' ', '-')}"
            else:
                name_title = f"'{self.sim_name}'"
                name_projected = ''
            name_file = self.sim_name.replace(' ', '-')
        elif data == 'obs':
            if not self.obs:
                raise ValueError("Observational PCs are not available. Set 'obs=True' when initializing the bimodalISO object to load them.")
            pcs_data = self.pcs_obs
            name_title = f"'{self.obs_name}'"
            name_file = self.obs_name.replace(' ', '-')
            name_projected = ''

        # Validate years input
        if years is None:
            raise ValueError(f"Provide at least one year to plot the PCs for.")
        if isinstance(years, int):
            years = [years]
        if not all(year in pcs_data.time.dt.year for year in years):
            raise ValueError(f"Some years are missing in the PCs data.")

        # Plot PCs for selected years
        custom_name = True if len(years)==1 else False
        for year in years:
            pcs_year = pcs_data.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
            pcs_plot, _ = plot.plot_pcs(pcs_year, title=f"Bimodal ISO indices {name_title} ({year})")

            plot.save_or_show_plot(pcs_plot, output_path, plot_filename=f"pcs_{name_file}_{year}_projected{name_projected}_{self.start_year_eeof}-{self.end_year_eeof}",
                                   plot_name=f"PCs (bimodal ISO indices) for year {year} plot", custom_name=custom_name)  

        return


    def freq_plot(self, output_path=None):
        """
        Generate and save/display plots of the mean monthly frequency of ISO events
        for the simulation data or, if `obs=True`, for the observational data.

        Parameters
        ----------
        output_path : str, optional
            Path to save plots. If None, plots are displayed but not saved.
        """

        plot_title = f'Mean monthly frequency of ISO events'
        if self.obs:
            name_title = f"'{self.sim_name}'_vs_'{self.obs_name}'"
            name_file = f"{self.sim_name.replace(' ', '-')}_vs_{self.obs_name.replace(' ', '-')}"

            if self.correct_pc:
                plot_title += f' (corrected PCs)'
                name_file += f"_corrected_PCs"
        else:
            name_title = f"'{self.sim_name}'"
            name_file = f"{self.sim_name.replace(' ', '-')}"

        # Plot frequency of ISO events
        freq_plot, _ = plot.plot_freq_ISO(self.freq_sim, self.freq_obs, alpha=self.scores['alpha'], corr=self.scores['R'],
                                           sigma=self.scores['sigma'], tss=self.scores['TSS'],
                                           title=plot_title, sim_label=self.sim_name, obs_label=self.obs_name)

        plot.save_or_show_plot(freq_plot, output_path, plot_filename=f"freq_ISO_{name_file}_{self.start_year_pc}-{self.end_year_pc}_projected_{self.start_year_eeof}-{self.end_year_eeof}",
                               plot_name=f"Mean monthly frequency of ISO events for {name_title} plot")

        return


class TCEvaluation:
    """
    Compute Tropical Cyclones (TCs) metrics and derived scalar scores.
    
    This class provides functionality for computing various TC metrics and derived scalar scores following 
    (C.M. Zarzycki et al., 2021) and plotting the results comparing simulations to IBTrACS observational 
    data, as well as, several reanalysis datasets.

    Parameters
    ----------
    data_sim: SimulationData
        Simulation dataset to use.
    start_year_tc, end_year_tc : int, optional
        Initial and end years to compute the TCs metrics for.
    obs : bool
        If True, also consider obsrvational data if available (default: True).
    wind_factor : float
        Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
    min_wind : float
        Minimum 10 m wind speed in m/s for TCs detection (default: 10.0).
    bin_size : float
        Size of the bins in degrees for computing the TCs metrics with CyMeP (default: 2.5).

    Attributes
    ----------
    sim_name : str
        Name of the simulation dataset.
    obs_names : str
        Name of the observational dataset, if requested.
    obs : bool
        Whether to use observational data.
    start_year_tc, end_year_tc : int
        Initial and end years to compute the TCs metrics for.
    min_wind : float
        Minimum 10 m wind speed in m/s for TCs detection.
    config_cymep : dict
        Dictionary with configuration parameters for each dataset (containing traj_filename,
        short_name, unstructured, ens_members, years_per_member, wind_speed_correction),
        necessary for the CyMeP package.
    tracks_path : str
        Path to save temporary files.
    bin_size : float
        Size of the bins in degrees for computing the TCs metrics with CyMeP.
    metrics_metadata : dict
        Metadata for the TC metrics.
    data_cymep : xr.Dataset
        TCs metrics computed with CyMeP.
    model_names : list[str]
        List of model names included in the TCs metrics dataset.
    clim_bias : xr.Dataset
        Global mean climatological bias for each TC metric.
    storm_bias : xr.Dataset
        Global mean storm bias for each TC metric.
    cbar_ticks_bias : list[str]
        Colorbar ticks labels for bias tables.
    colors_bias : tuple
        Colorbar colors for bias tables.
    temp_corr : xr.Dataset
        Seasonal correlation for each TC metric.
    spatial_corr : xr.Dataset
        Spatial correlation for each TC metric.
    cbar_ticks_corr : list[str]
        Colorbar ticks labels for correlation tables.
    colors_corr : tuple
        Colorbar colors for correlation tables.
    """

    def __init__(self, data_sim : SimulationData, start_year_tc: int = None, end_year_tc: int = None, obs: bool = True, 
                 wind_factor: float = 1.0, min_wind: float = 10.0, bin_size : float = 2.5):
        
        # Validate input
        if not isinstance(data_sim, SimulationData):
            raise TypeError("'data_sim' must be an instance of SimulationData.")   
        self.sim_name = data_sim.name
        self.obs_names = []
        self.obs = obs

        if not isinstance(min_wind, (int, float)) or min_wind < 10.0:
            raise ValueError("The minimum 10 m wind speed for TCs detection 'min_wind' must be a numeric value of at least 10 m/s.")
        self.min_wind = min_wind

        self.config_cymep = {}
        self.bin_size = bin_size
        self.metrics_metadata = data_general.load_yaml_file(config_params.TCS_METRICS_PATH)


        # Select years for TCs analysis
        self.start_year_tc, self.end_year_tc = data_general.validate_year_range(data_sim, start_year_tc, end_year_tc, process_name="TC")
        print(f"\tYears selected for TCs metrics computation: {self.start_year_tc}-{self.end_year_tc}.", flush=True)
        data_sim_tcs = data_sim.data.sel(time=slice(np.datetime64(f"{self.start_year_tc}-01-01"), np.datetime64(f"{self.end_year_tc}-12-31")))

        # Prepare output folder for temporary files
        self.tracks_path = config_params.TC_DATA_PATH / "temp_tracks"
        self.tracks_path.mkdir(parents=True, exist_ok=True)


        # # Prepare IBTrACS data, and observational data if requested
        self._prepare_IBTrACS_data()
        print(f"\tIBTrACS TCs data preprocessed and saved to '{self.config_cymep['IBTrACS'][0]}'.", flush=True)

        if self.obs:
            self.obs_names = ['JRA55']
            self._prepare_obs_data()
            print(f"\tObservational TCs data preprocessed and saved to '{self.tracks_path}'.", flush=True)

        # Prepare simulation data
        print("\tStarting simulation data TCs tracking. TempestExtremes output:", flush=True)
        self._prepare_sim_data(data_sim_tcs, wind_factor=wind_factor)
        print(f"\tSimulation data TCs tracking completed and saved to '{self.config_cymep[self.sim_name][0]}'.", flush=True)
        

        # Compute TCs metrics
        print("\tStarting Tropical Cyclones metrics computation. CyMeP output:", flush=True)
        self.data_cymep, self.model_names = self._compute_cymep_metrics()
        print(f"\tTCs metrics computed. See attribute `data_cymep`.", flush=True)

        self.clim_bias, self.storm_bias, self.cbar_ticks_bias, self.colors_bias = self._retrieve_biases()
        print("\tBiases computation completed. See attributes `clim_bias` and `storm_bias`.", flush=True)

        self.temp_corr, self.spatial_corr, self.cbar_ticks_corr, self.colors_corr =  self._retrieve_correlations()
        print("\tCorrelations computation completed. See attributes `temp_corr` and `spatial_corr`.", flush=True)

        # # Delete intermediate files
        # shutil.rmtree(self.tracks_path)

        print(f"\nTropical Cyclones scores computation completed between years {self.start_year_tc} and {self.end_year_tc}.", flush=True)
        return


    def _prepare_IBTrACS_data(self):
        """
        Prepare IBTrACS TCs data to use as a reference for the TCs metrics computation.
        """

        # Retrieve IBTrACS data metadata
        ib_path, ib_end_year = tcs_ibtracs.check_ibtracs_file(self.start_year_tc, self.end_year_tc, output_path=self.tracks_path,
                                                              min_wind=self.min_wind)  

        # Add IBTrACS parameters to CyMeP configuration file
        year_range = ib_end_year - config_params.IBTRACS_START_YEAR + 1
        ib_config = [str(ib_path), "IBTrACS", False, 1, year_range, 1.0]
        self.config_cymep["IBTrACS"] = ib_config

        return 


    def _prepare_obs_data(self):
        """
        Prepare observational TCs data for the TCs metrics computation.
        """

        data_path = config_params.TC_DATA_PATH
        for name in self.obs_names:
            # Look for observational data
            obs_files = list(data_path.glob(f'{name.lower()}_*.txt'))
            if obs_files is None:
                raise FileNotFoundError(f"No observational TCs data found for '{name}' in '{data_path}'.")
            obs_path = obs_files[0]
            
            # Check whether the current version covers the selected period
            match = re.search(rf'{name.lower()}_(\d+)-(\d+)', obs_path.name)
            obs_start_year = int(match.group(1))
            obs_end_year = int(match.group(2))

            if self.start_year_tc < obs_start_year  or obs_end_year < self.end_year_tc:
                warnings.warn(f"The available observational TCs data for '{name}' ({obs_start_year}-{obs_end_year}) does not "
                              f"cover the selected period for TCs metrics computation ({self.start_year_tc}-{self.end_year_tc})."
                              f" This dataset will not be considered for the TCs metrics computation.")
                break

            # Apply wind threshold
            if self.min_wind > 10.0:
                new_obs_path = self.tracks_path / f'{name.lower()}_{obs_start_year}_{obs_end_year}_{self.min_wind:.1f}_False_1_1.0.txt'
                obs_path = tcs_tempestextremes.filter_tracks_by_wind(obs_path, new_obs_path, cutoff_wind=self.min_wind)
            else:
                new_obs_path = self.tracks_path / obs_path.name
                shutil.copy2(obs_path, new_obs_path)

            year_range = obs_end_year - obs_start_year + 1
            obs_config = [str(new_obs_path), name, False, 1, year_range, 1.0]
            self.config_cymep[name] = obs_config

        return


    def _prepare_sim_data(self, data_sim, wind_factor=1.0):
        """
        Prepare simulation data for the TCs metrics computation (detect 
        and track TCs with TempestExtremes).

        Parameters
        ----------
        data_sim : xr.Dataset
            Simulation data.
        wind_factor : float
            Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
        """

        # Run TempestExtremes tracking on simulated data
        tracks_sim_path = tcs_tempestextremes.run_tempestExtremes(data_sim, self.sim_name, self.tracks_path, min_wind=self.min_wind)
        new_sim_path = self.tracks_path / f"{self.sim_name.replace(' ', '-')}_{self.start_year_tc}-{self.end_year_tc}_{self.min_wind:.1f}_False_1_{wind_factor:.1f}.txt"
        os.rename(tracks_sim_path, new_sim_path)

        # Add simulation data parameters to CyMeP configuration file
        year_range = self.end_year_tc - self.start_year_tc + 1
        self.config_cymep[self.sim_name] = [str(new_sim_path), self.sim_name, False, 1, year_range, wind_factor]

        return


    def _compute_cymep_metrics(self):
        """
        Compute TCs metrics using the CyMeP package.

        Returns 
        -------
        data_cymep : xr.Dataset
            TCs metrics computed with CyMeP.
        model_names : list[str]
            List of model names included in the TCs metrics dataset.
        """

        # Prepare CyMeP configuration file
        config_cymep_path = self.tracks_path / "cymep_config.csv"
        tcs_cymep_main.prepare_configs_file(self.config_cymep, output_path=config_cymep_path)

        # Run CyMeP TCs metrics computation
        data_cymep = tcs_cymep_main.run_cymep_pyhanami(self.start_year_tc, self.end_year_tc, output_path=self.tracks_path, 
                                                            gridsize=self.bin_size, csvfilename=config_cymep_path)
        model_names = data_cymep.model.values 

        return data_cymep, model_names


    def _retrieve_biases(self):
        """
        Retrieve climatological and storm biases from CyMeP output data,
        and define related plotting parameters.

        Returns
        -------
        clim_bias : xr.Dataset
            Global mean climatological bias for each TC metric.
        storm_bias : xr.Dataset
            Global mean storm bias for each TC metric.
        cbar_ticks_bias : list[str]
            Colorbar ticks labels for bias tables.
        colors_bias : tuple
            Colorbar colors for bias tables.
        """

        # With numpy arrays (not used anymore, kept for reference)
        # Retrieve biases
        # clim_mean = np.transpose([self.data_cymep[f'clim_mean_{metric}'].values for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True])   
        # clim_bias = np.concatenate(([clim_mean[0]], clim_mean[1:] - clim_mean[0]))

        # storm_mean = np.transpose([self.data_cymep[f'storm_mean_{metric}'].values for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True 
        #               and metric!='count'])
        # storm_bias = np.concatenate(([storm_mean[0]], storm_mean[1:] - storm_mean[0]))


        # Retrieve biases as xarray.Datasets
        clim_metrics = [metric for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True]
        clim_mean = self.data_cymep[[f'clim_mean_{metric}' for metric in clim_metrics]]
        clim_bias = xr.concat([
            clim_mean.isel(model=0).expand_dims('model'),
            clim_mean.isel(model=slice(1, None)) - clim_mean.isel(model=0)
        ], dim='model')
        clim_bias = clim_bias.rename({f'clim_mean_{metric}': f'clim_bias_{metric}' for metric in clim_metrics})
        
        storm_metrics = [metric for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True and metric!='count']
        storm_mean = self.data_cymep[[f'storm_mean_{metric}' for metric in storm_metrics]]
        storm_bias = xr.concat([
            storm_mean.isel(model=0).expand_dims('model'),
            storm_mean.isel(model=slice(1, None)) - storm_mean.isel(model=0)
        ], dim='model')
        storm_bias = storm_bias.rename({f'storm_mean_{metric}': f'storm_bias_{metric}' for metric in storm_metrics})

        # Define plotting parameters
        cbar_ticks_bias = ['Negative bias', 'No bias', 'Positive bias']
        colors_bias = ("RedGreen", ['tab:red', 'white', 'tab:green'])   #("BlueRed", ['tab:blue', 'white', 'tab:red'])

        return clim_bias, storm_bias, cbar_ticks_bias, colors_bias
    

    def _retrieve_correlations(self):
        """
        Retrieve temporal and spatial correlations from CyMeP output data,
        and define related plotting parameters.

        Returns
        -------
        temp_corr : xr.Dataset
            Seasonal correlation for each TC metric.
        spatial_corr : xr.Dataset
            Spatial correlation for each TC metric.
        cbar_ticks_corr : list[str]
            Colorbar ticks labels for correlation tables.
        colors_corr : tuple
            Colorbar colors for correlation tables.
        """

        # With numpy arrays (not used anymore, kept for reference)
        # Retrieve correlations
        # temp_corr = np.transpose([self.data_cymep[var].values for var in self.data_cymep.data_vars if var.startswith('temporal_scorr_')])
        # spatial_corr = np.transpose([self.data_cymep[var].values for var in self.data_cymep.data_vars if var.startswith('spatial_pcorr_')])


        # Retrieve correlations as xarray.Datasets
        temp_corr = self.data_cymep[[var for var in self.data_cymep.data_vars if var.startswith('temporal_scorr_')]]
        spatial_corr = self.data_cymep[[var for var in self.data_cymep.data_vars if var.startswith('spatial_pcorr_')]]

        # Define plotting parameters
        cbar_ticks_corr = ['Negative correlation (-1)', 'No correlation (0)', 'Positive correlation (1)']
        colors_corr = ("RedGreen", ['tab:red', 'white', 'tab:green'])   #("OrangeGreen", ['tab:orange', 'white', 'tab:green'])

        return temp_corr, spatial_corr, cbar_ticks_corr, colors_corr


    def save_data(self, output_path):
        """
        Save computed EEOFs, PCs, and frequency of ISO events to NetCDF files.

        Parameters
        ----------
        output_path : str
            Path to save the data files.
        """

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"

        # Prepare dataset names for filenames
        name = f"IBTrACS_{self.sim_name.replace(' ', '-')}"
        if self.obs:
            name += '_obs'


        # Save all CyMeP output data to NetCDF file
        data_cymep_path = output_path / f'tcs_metrics_{name}_{year_range}.nc'
        self.data_cymep.to_netcdf(data_cymep_path)
        print(f"TCs metrics computed for '{self.sim_name}' simulations saved to '{data_cymep_path}'.", flush=True)


        # Save biases and correlations to Numpy files
        clim_bias_path = output_path / f'tcs_clim_bias_{name}_{year_range}.nc'
        self.clim_bias.to_netcdf(clim_bias_path)
        print(f"Global climatological mean bias saved to '{clim_bias_path}'.", flush=True)

        storm_bias_path = output_path / f'tcs_storm_bias_{name}_{year_range}.nc'
        self.storm_bias.to_netcdf(storm_bias_path)
        print(f"Global storm mean bias saved to '{storm_bias_path}'.", flush=True)

        temp_corr_path = output_path / f'tcs_temp_corr_{name}_{year_range}.nc'
        self.temp_corr.to_netcdf(temp_corr_path)
        print(f"Global seasonal correlation saved to '{temp_corr_path}'.", flush=True)

        spatial_corr_path = output_path / f'tcs_spatial_corr_{name}_{year_range}.nc'
        self.spatial_corr.to_netcdf(spatial_corr_path)
        print(f"Global spatial correlation saved to '{spatial_corr_path}'.", flush=True)

        return


    def clim_bias_table(self, output_path=None):
        """
        Generate and save/display table plot with global climatological mean bias for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_clim_bias = self.clim_bias.to_array().values.transpose()

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        cols_clim_bias = np.append([f'{self.bin_size}° x {self.bin_size}°'], [fr'$\overline{{b}}_{{clim,{metric}}}$ ({self.metrics_metadata[metric]["units"]})' 
                                                                            for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True])
        maxs_clim_bias = np.max(np.abs(data_clim_bias[1:, :]), axis=0)
        limits_clim_bias = np.stack([-maxs_clim_bias, maxs_clim_bias], axis=1)

        # Generate table plot
        clim_bias_table_plot, _ = plot.plot_table(data_clim_bias, title=f'Global climatological mean bias ({year_range})', col_labels=cols_clim_bias, 
                                                  row_labels=self.model_names, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias,
                                                  limits=limits_clim_bias)
        
        plot.save_or_show_plot(clim_bias_table_plot, output_path, plot_filename=f"tcs_climatological_bias_table_{self.sim_name.replace(' ', '-')}_{year_range}",
                               plot_name="Climatological bias table for TCs metrics plot")
        
        return
    

    def storm_bias_table(self, output_path=None):
        """
        Generate and save/display table plot with global storm mean bias for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_storm_bias = self.storm_bias.to_array().values.transpose()

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        cols_storm_bias = np.append([f'{self.bin_size}° x {self.bin_size}°'], [fr'$\overline{{b}}_{{storm,{metric}}}$ ({self.metrics_metadata[metric]["units"]})' 
                                                                               for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True 
                                                                               and metric!='count'])
        maxs_storm_bias = np.max(np.abs(data_storm_bias[1:, :]), axis=0)
        limits_storm_bias = np.stack([-maxs_storm_bias, maxs_storm_bias], axis=1)

        # Generate table plot
        storm_bias_table_plot, _ = plot.plot_table(data_storm_bias, title=f'Global storm mean bias ({year_range})', col_labels=cols_storm_bias, 
                                                   row_labels=self.model_names, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias,
                                                   limits=limits_storm_bias)
        
        plot.save_or_show_plot(storm_bias_table_plot, output_path, plot_filename=f"tcs_storm_bias_table_{self.sim_name.replace(' ', '-')}_{year_range}",
                               plot_name="Storm bias table for TCs metrics plot")
        
        return

    
    def temp_corr_table(self, output_path=None):
        """
        Generate and save/display table plot with global seasonal correlation for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_temp_corr = self.temp_corr.to_array().values.transpose()

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        cols_temp_corr = np.append([f'{self.bin_size}° x {self.bin_size}°'], [fr'$\rho_{{s,{metric}}}$' for metric in self.metrics_metadata 
                                                                              if self.metrics_metadata[metric]['temporal']==True])
        limits_temp_corr = np.repeat([[-1, 1]], len(cols_temp_corr)-1, axis=0)

        # Generate table plot
        temp_corr_table_plot, _ = plot.plot_table(data_temp_corr, title=f'Global seasonal correlation ({year_range})', col_labels=cols_temp_corr, 
                                                  row_labels=self.model_names, cbar_ticks=self.cbar_ticks_corr, colors=self.colors_corr, 
                                                  limits=limits_temp_corr, decimals=2)
        
        plot.save_or_show_plot(temp_corr_table_plot, output_path, plot_filename=f"tcs_seasonal_corr_table_{self.sim_name.replace(' ', '-')}_{year_range}",
                               plot_name="Seasonal correlation table for TCs metrics plot")

        return
    
    
    def spatial_corr_table(self, output_path=None):
        """
        Generate and save/display table plot with global spatial correlation for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_spatial_corr = self.spatial_corr.to_array().values.transpose()

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        cols_spatial_corr = np.append([f'{self.bin_size}° x {self.bin_size}°'], [fr'$r_{{xy,{metric}}}$' for metric in self.metrics_metadata 
                                                                                 if self.metrics_metadata[metric]['spatial']==True])
        limits_spatial_corr = np.repeat([[-1, 1]], len(cols_spatial_corr)-1, axis=0)
        
        # Generate table plot
        spatial_corr_table_plot, _ = plot.plot_table(data_spatial_corr, title=f'Global spatial correlation ({year_range})', col_labels=cols_spatial_corr,
                                                     row_labels=self.model_names, cbar_ticks=self.cbar_ticks_corr, colors=self.colors_corr, 
                                                     limits=limits_spatial_corr, decimals=2)
        
        plot.save_or_show_plot(spatial_corr_table_plot, output_path, plot_filename=f"tcs_spatial_corr_table_{self.sim_name.replace(' ', '-')}_{year_range}",
                               plot_name="Spatial correlation table for TCs metrics plot")

        return


    def linear_plots(self, output_path=None):
        """
        Generate and save/display linear plots comparing all datasets for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the linear plots. If None, the plots are displayed but not saved.
        """

        # Prepare metrics metadata
        temporal_metrics = [metric for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True]
        linear_metrics_names = [self.metrics_metadata[metric]['short_name'] for metric in temporal_metrics]
        linear_metrics_units = [self.metrics_metadata[metric]['units'] for metric in temporal_metrics]

        # Prepare labels and titles
        linear_ylabel = [f'{name} ({unit})' for name, unit in zip(linear_metrics_names, linear_metrics_units)]
        linear_month_titles = [f'{name} seasonal cycle' for name in linear_metrics_names] 
        linear_year_titles = [f'{name} interannual cycle' for name in linear_metrics_names]

        # Prepare invariant arrays/strings
        months = np.arange(1, 13, dtype=int)
        years = np.arange(self.start_year_tc, self.end_year_tc+1, dtype=int)  

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        name_file = self.sim_name.replace(' ', '-')


        # Create plots
        for i, name in enumerate(linear_metrics_names):
            # Create line plot for monthly cycles
            linear_month_data = self.data_cymep[f'per_month_{temporal_metrics[i]}'].rename({'month': 'time'})
            linear_month_data_list = [linear_month_data.sel(model=model) for model in self.model_names]

            plt.figure(figsize=(10, 6))
            for j, month_data in enumerate(linear_month_data_list):
                plt.plot(months, month_data, 'o-', markersize=4, linewidth=1.2, label=self.model_names[j])
            plt.xticks(months, months)
            plt.xlabel('month', fontsize=12)
            plt.ylabel(linear_ylabel[i], fontsize=12)
            plt.title(linear_month_titles[i], fontsize=16)
            plt.grid(True)
            plt.legend()

            plot.save_or_show_plot(plt.gcf(), output_path, plot_name=f"Linear seasonal cycle plot for TC {name}",
                                   plot_filename=f"tcs_{name.lower()}_seasonal_cycle_plot_{name_file}_{year_range}", 
                                   custom_name=False)

            # Create line plot for interannual cycles
            linear_year_data = self.data_cymep[f'per_year_{temporal_metrics[i]}'].rename({'year': 'time'})
            linear_year_data_list = [linear_year_data.sel(model=model) for model in self.model_names]              

            plt.figure(figsize=(10, 6))
            for j, year_data in enumerate(linear_year_data_list):
                plt.plot(years, year_data, 'o-', markersize=4, linewidth=1.2, label=self.model_names[j])
            plt.xlabel('year', fontsize=12)
            plt.ylabel(linear_ylabel[i], fontsize=12)
            plt.title(linear_year_titles[i], fontsize=16)
            plt.grid(True)
            plt.legend()

            plot.save_or_show_plot(plt.gcf(), output_path, plot_name=f"Linear interannual cycle plot for TC {name}",
                                   plot_filename=f"tcs_{name.lower()}_interannual_cycle_plot_{name_file}_{year_range}", 
                                   custom_name=False)

        return
    

    def spatial_plots(self, output_path=None, clon=0):
        """
        Generate and save/display spatial plots comparing simulations with IBTrACS data for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the spatial plots. If None, the plots are displayed but not saved.
        clon : int
            Central longitude for the spatial maps (default: 0).
        """

        # Prepare metrics metadata
        spatial_metrics = [metric for metric in self.metrics_metadata if self.metrics_metadata[metric]['spatial']==True]
        spatial_metrics_names = [self.metrics_metadata[metric]['short_name'] for metric in spatial_metrics]
        spatial_metrics_units = [self.metrics_metadata[metric]['units'] for metric in spatial_metrics]

        # Prepare labels and titles
        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        spatial_titles = [f'TC {name} density for {self.bin_size}°x{self.bin_size}° cells ({year_range})' for name in spatial_metrics_names]
        spatial_bias_titles = [f"TC {name} bias ('{self.sim_name}' vs 'IBTrACS') for {self.bin_size}°x{self.bin_size}° cells ({year_range})" for name in spatial_metrics_names]
        spatial_cb_labels = [f'{name} ({unit})' for name, unit in zip(spatial_metrics_names, spatial_metrics_units)]
        name_file = self.sim_name.replace(' ', '-')

        # Create plots
        for i, name in enumerate(spatial_metrics_names):

            # Create a modified colormap with white for NaN values
            cmap_modified = cmocean.cm.thermal_r.copy()
            # cmap_modified.set_bad('white')


            spatial_abs_data = self.data_cymep[f'spatial_abs_{spatial_metrics[i]}']
            spatial_abs_data = spatial_abs_data.where(spatial_abs_data != 0)  # Set zero values to NaN for better visualization
            spatial_abs_plot, _ = plot.two_spatial_plots(spatial_abs_data.sel(model='IBTrACS'), spatial_abs_data.sel(model=self.sim_name), clon=clon,
                                                         title_1='IBTrACS', title_2=self.sim_name, suptitle=spatial_titles[i], cb_label=spatial_cb_labels[i],
                                                         cmap=cmap_modified)
            plot.save_or_show_plot(spatial_abs_plot, output_path, plot_filename=f"tcs_{name.lower()}_spatial_abs_plot_{name_file}_{year_range}_clon_{clon}",
                                    plot_name=f"Spatial plot for TC {name}", custom_name=False)

            spatial_bias_data = self.data_cymep[f'spatial_bias_{spatial_metrics[i]}'].sel(model=self.sim_name)
            limit = np.ceil(np.nanmax(np.abs(spatial_bias_data.values)))
            levels = np.linspace(-limit, limit, 13)
            spatial_bias_plot, _ = plot.plot_spatial(spatial_bias_data, clon=clon, title=spatial_bias_titles[i], cb_label=f'bias in {spatial_cb_labels[i]}', 
                                                     cmap=LinearSegmentedColormap.from_list(*self.colors_bias), levels=levels)
                                                     #cmap=cmocean.cm.diff)
            plot.save_or_show_plot(spatial_bias_plot, output_path, plot_filename=f"tcs_{name.lower()}_spatial_bias_plot_{name_file}_{year_range}_clon_{clon}",
                                   plot_name=f"Spatial bias plot for TC {name}", custom_name=False)
            
        return
    

class ScientificEvaluation:
    """
    Compute and plot metrics for scientific model skill evaluation.

    This class provides functionality for computing and visualizing metrics to evaluate how well a model 
    reproduces several phenomena. Currently, it includes methods for bimodal ISO indices.
    
    Parameters
    ----------
    datasets : SimulationData or Iterable[SimulationData], optional
        Ensemble or list of ensembles containing simulation data and metadata.

    Attributes
    ----------
    datasets : list[SimulationData]
        List of ensembles containing simulation data and metadata.
    variables : dict
        Configuration dictionary mapping variable names to display metadata.
    """    

    def __init__(self, datasets: Iterable[SimulationData] = None):        
        if datasets is None:
            self.datasets = []
        else:
            if isinstance(datasets, SimulationData):
                self.datasets = [datasets]
            elif isinstance(datasets, Iterable) and not isinstance(datasets, (str, bytes)) \
                and all(isinstance(ds, SimulationData) for ds in datasets):
                self.datasets = list(datasets)
            else:
                raise TypeError("Input must be a SimulationData object or an iterable of SimulationData objects.")

        return


    def add_datasets(self, datasets):
        """ 
        Add new datasets to the ScientificEvaluation object.

        Parameters
        ----------
        datasets : SimulationData or Iterable[SimulationData])
            Ensemble or list of ensembles containing simulation data and metadata to add.
        """

        # Validate input
        if isinstance(datasets, SimulationData):
            datasets = [datasets]
        elif not isinstance(datasets, Iterable) or isinstance(datasets, (str, bytes)) \
            or not all(isinstance(ds, SimulationData) for ds in datasets):
            raise TypeError("Input must be a SimulationData object or an iterable of SimulationData objects.")
        
        # Check for duplicate datasets
        for dataset in datasets:
            if not any(ds.name == dataset.name for ds in self.datasets):
                self.datasets.append(dataset)
            else:
                warnings.warn(f"Dataset with name '{dataset.name}' already exists in the ScientificEvaluation object. Skipping addition.")

        return
    
    
    def compute_general_scores(self, var_names=None, data_name=None, obs_name=None, obs_path=None, #config_params.GEN_OBS_NAME, obs_path=config_params.GEN_OBS_PATH, 
                               start_year=None, end_year=None):
        """
        Initialize and compute general model skill evaluation scores for a selected dataset.
        
        Parameters
        ----------
        var_names : str or list[str], optional
            Climate variable(s) name(s). If None, all variables in the simulated dataset will be used.
        data_name : str, optional
            Name of simulation ensemble to use. If None, the first dataset in the ScientificEvaluation 
            object is used.
        obs_name : str
            Name of the observational dataset to compare to (default: config_params.GEN_OBS_NAME).
        obs_path : str
            Path to the observations database (default: config_params.GEN_OBS_PATH).
        start_year, end_year : int
            Initial and end years to compute the general scores for.

        Returns
        -------
        general_analysis : GeneralEvaluation
            GeneralEvaluation object containing the computed general scientific skill scalar scores.
        """

        # Validate input
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError("At least one dataset is required for the general evaluation.")
            data_general = self.datasets[0]
            data_name = data_general.name
        elif isinstance(data_name, str):
            data_general = [ds for ds in self.datasets if ds.name == data_name]
            if not data_general:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_general = data_general[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")
        
        # Create GeneralEvaluation object and compute scores
        print(f"Performing general scalar analysis for dataset '{data_name}':", flush=True)
        general_analysis = GeneralEvaluation(data_sim=data_general, var_names=var_names, obs_name=obs_name, 
                                             obs_path=obs_path, start_year=start_year, end_year=end_year)

        return general_analysis


    def compute_iso_scores(self, data_name=None, start_year_eeof=None, end_year_eeof=None, start_year_pc=None, end_year_pc=None, obs=False, 
                            correct_pc=False, lat_range=(-30, 30), lag=5, n_lags=3, n_modes=2, window=141, low_freq=1/90, high_freq=1/25):
        """
        Initialize and compute bimodal ISO indices (following (K. Kikuchi, 2020)) and derive scalar 
        scores (following (M. Nakano et al., 2019)) for a selected dataset.

        Parameters
        ----------
        data_name : str, optional
            Name of simulation ensemble to use. If None, the first dataset in the ScientificEvaluation 
            object is used.
        start_year_eeof, end_year_eeof : int
            Initial and end years to perform the Extended Empirical Orthogonal Function (EEOF) analysis for 
            (not needed if `obs=True`). 
        start_year_pc, end_year_pc : int
            Initial and end years to compute Principal Components (PCs) for.
        obs : bool
            If True, use EEOFs from observational data (default: False).
        correct_pc : bool
            Whether to adjust simulated PCs by dividing by alpha (default: False).
        lat_range : tuple
            Geographic latitude bounds (default: (-30, 30)).
        lag : int
            Lag timesteps (default: 5).
        n_lags : int
            Number of lag copies (default: 3).
        n_modes : int)
            Number of EEOFs modes to compute (default: 2).
        window_size : int
            Length of the filter kernel (default: 141).
        low_freq : float
            Lower cutoff frequency (default: 1/90).
        high_freq : float
            Upper cutoff frequency (default: 1/25).

        Returns
        -------
        iso_analysis : ISOEvaluation
            ISOEvaluation object containing the computed bimodal ISO indices and scalar scores.
        """

        # Validate input
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError("At least one dataset is required for the ISO evaluation.")
            data_ISO = self.datasets[0]
            data_name = data_ISO.name
        elif isinstance(data_name, str):
            data_ISO = [ds for ds in self.datasets if ds.name == data_name]
            if not data_ISO:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_ISO = data_ISO[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")
        
        # Create ISOEvaluation object and compute scores
        print(f"Performing ISO analysis for dataset '{data_name}':", flush=True)
        iso_analysis = ISOEvaluation(data_sim=data_ISO, start_year_eeof=start_year_eeof, end_year_eeof=end_year_eeof, start_year_pc=start_year_pc, 
                                      end_year_pc=end_year_pc, obs=obs, correct_pc=correct_pc, lat_range=lat_range, lag=lag, n_lags=n_lags,
                                      n_modes=n_modes, window=window, low_freq=low_freq, high_freq=high_freq)

        return iso_analysis
    

    def compute_tc_scores(self, data_name=None, start_year_tc=None, end_year_tc=None, obs=True, wind_factor=1.0, min_wind=10, 
                           bin_size=2.5):
        """
        Compute Tropical Cyclones (TCs) metrics and derive scalar scores following (C.M. Zarzycki et al., 2021) 
        and plot results.

        Parameters
        ----------
        data_name : str, optional
            Name of simulation ensemble to use. If None, the first dataset in the ScientificEvaluation 
            object is used.
        start_year_tc, end_year_tc : int, optional
            Initial and end years to compute the TCs metrics for.
        obs : bool
            If True, include observational data if available (default: True).
        wind_factor : float
            Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
        min_wind : float
            Minimum 10 m wind speed in m/s for TCs detection (default: 10.0).
        bin_size : float
            Size of the bins in degrees for computing the TCs metrics with CyMeP (default: 2.5).

        Returns
        -------
        tc_analysis : TCEvaluation
            TCEvaluation object containing the computed TCs metrics and scalar scores.
        """

        # Validate input
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError("At least one dataset is required for the TCs evaluation.")
            data_TC = self.datasets[0]
            data_name = data_TC.name
        elif isinstance(data_name, str):
            data_TC = [ds for ds in self.datasets if ds.name == data_name]
            if not data_TC:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_TC = data_TC[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")

        # Create a TCEvaluation object and compute scores
        print(f"Performing TCs analysis for dataset '{data_name}':", flush=True)
        tc_analysis = TCEvaluation(data_sim=data_TC, start_year_tc=start_year_tc, end_year_tc=end_year_tc, obs=obs, 
                                   wind_factor=wind_factor, min_wind=min_wind, bin_size=bin_size)

        return tc_analysis

        # input_path = data_TC.data_path

        # if obs:
        #     if obs_path is None or obs_name is None or obs_wind_factor is None:
        #         raise NotImplementedError('Automatic selection of observations is not implemented yet. '
        #                                   'Please provide at least one path, one name and the corresponding wind factor if you want to include observations.')
            
        #     # Convert to lists if single values are provided                
        #     obs_path = [obs_path] if isinstance(obs_path, (str, Path)) else list(obs_path)
        #     obs_name = [obs_name] if isinstance(obs_name, str) else list(obs_name)
        #     obs_wind_factor = [obs_wind_factor] if isinstance(obs_wind_factor, (int, float)) else list(obs_wind_factor)

        #     # Validate lengths match
        #     if not (len(obs_path) == len(obs_name) == len(obs_wind_factor)):
        #         raise ValueError("'obs_path', 'obs_name' and 'obs_wind_factor' must have the same length.")

                




        # # Plot TC genesis and trajectory density if requested
        # if full_output:
        #     # Get simulated tracks and counts
        #     sim_tracks = tcs_tempestextremes.read_tracks_tempestExtremes(tracks_sim_path)
        #     sim_counts_gen, sim_counts_traj = tcs_tempestextremes.compute_tc_counts(sim_tracks, start_year, end_year, bin_size=bin_size, cutoff_wind=min_wind)

        #     # Get IBTrACS tracks and counts
        #     ib_tracks = tcs_tempestextremes.read_tracks_tempestExtremes(ib_path)
        #     ib_counts_gen, ib_counts_traj = tcs_tempestextremes.compute_tc_counts(ib_tracks, start_year, end_year, bin_size=bin_size, cutoff_wind=min_wind)



        # # Prepare observations TCs data if requested
        # if obs:
        #     obs_tracks_path = config_params.TC_DATA_PATH
        #     for name, path, wind in zip(obs_name, obs_path, obs_wind_factor):
        #         # Check if TCs data is already present for the selected years, minimum wind and observations dataset
        #         obs_files = list(obs_tracks_path.glob(f"{name}_*.txt"))

        #         found = False
        #         for obs_file in obs_files:
        #             parts = obs_file.stem.split('_')

        #             if len(parts) >= 2 and '-' in parts[1]:
        #                 year_range = parts[1]
        #                 try:
        #                     # Check if the file covers the selected period
        #                     file_start, file_end = map(int, year_range.split('-'))                            
        #                     if file_start <= start_year and file_end >= end_year:

        #                         # Check if the file matches the selected min_wind
        #                         obs_min_wind = float(parts[2])
        #                         if abs(obs_min_wind - min_wind) < 1e-6:
        #                             unstructured = parts[3].lower() == 'true'
        #                             ens_members = int(parts[4])
        #                             aux_wind_factor = float(parts[5])
        #                             found = True
        #                             break
        #                 except ValueError:
        #                     continue
        #         if found:
        #             configs[name] = [obs_file.name, name.lower(), unstructured, ens_members, years, aux_wind_factor]

        #         # Compute TCs data for observations if not already present
        #         else:
        #             # Load observations
        #             var_names = ["psl", "uas", "vas", "zg300", "zg500"]
        #             data_sim_selected = data_sim_all[var_names]
        #             data_obs = ObservationData(path, data_sim_selected, name=name)
                    
        #             ens_members = 1 if 'realization' not in data_obs.data.dims else data_obs.data.dims['realization']
        #             unstructured = False

        #             # Run TempestExtremes tracking on observational data
        #             print(f'Starting Tropical Cyclones tracking using TempestExtremes for {name} observations...', flush=True)
        #             tracks_obs_path = tcs_tempestextremes.run_tempestExtremes(data_obs, name, obs_tracks_path, min_wind=min_wind)
        #             tracks_obs_path = tracks_obs_path.rename(tracks_obs_path.with_name(f"{name}_{start_year}-{end_year}_{min_wind:.1f}_{unstructured}_{ens_members}_{wind:.1f}.txt"))
        #             print(f"Tropical Cyclones tracking completed for {name} observations. Output files added to '{obs_tracks_path}'.", flush=True)

        #             configs[name] = [tracks_obs_path, name, unstructured, ens_members, years, wind]