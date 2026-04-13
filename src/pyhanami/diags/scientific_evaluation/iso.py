import warnings
warnings.simplefilter("always")

import numpy as np
import xarray as xr

from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

from pyhanami.config import config_params
from pyhanami.diags.Simulations import SimulationData
from pyhanami.utils import data_general, config_scores, iso_scores, plot

VARIABLES = data_general.load_yaml_file(config_params.VARIABLES_PATH)


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
    start_year_eeof, end_year_eeof : int, optional
        Initial and end years to perform Extended Empirical Orthogonal Function (EEOF) analysis for.
    start_year_pc, end_year_pc : int, optional
        Initial and end years to compute Principal Components (PCs) for.
    obs : bool
        If True, also plot observational data if available (default: False).
    obs_path : str
        Path to the observational NOAA data file (default: config_params.NOAA_PATH). As of now, only 
        necessary if the resolution of the NOAA data (2.5°x2.5°) is higher than that of the simulation data.
    correct_pc : bool
        Whether to adjust simulated PCs by dividing by alpha (default: False).
    iso_config : ISOConfig
        Configuration dataclass with parameters necessary for the ISO evaluation. If None, default values 
        from the configuration file `pyhanami.config.scientific_evaluation_parameters.yaml` will be used.


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
    eeof_summer : xr.DataArray
        EEOFs for boreal summer.
    eeof_winter : xr.DataArray
        EEOFs for boreal winter.
    start_year_eeof, end_year_eeof : int
        Initial and end years to perform Extended Empirical Orthogonal Function (EEOF) analysis for.
    pcs_sim : xr.DataArray
        PCs computed from the simulation data.
    pcs_obs : xr.DataArray or None
        PCs from observational data, or None if not computed.
    start_year_pc, end_year_pc : int
        Initial and end years to compute Principal Components (PCs) for.
    freq_sim : xr.DataArray
        Mean monthly frequency of occurrence for simulation data.
    freq_obs : xr.DataArray or None
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

    def __init__(self, data_sim, var_name='rlut', start_year_eeof=None, end_year_eeof=None, start_year_pc=None, 
                 end_year_pc=None, obs=False, obs_path=config_params.NOAA_PATH, correct_pc=False, iso_config=None):

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

        # Load ISO evaluation parameters
        if iso_config is None:
            iso_config = config_scores.ISOConfig()
        elif not isinstance(iso_config, config_scores.ISOConfig):
            raise TypeError("'iso_config' must be an instance of the ISOConfig dataclass defined in 'pyhanami.utils.config_scores'.")


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
        print(f"\tYears selected for EEOF analysis: {self.start_year_eeof}-{self.end_year_eeof}.", flush=True)

        self.start_year_pc, self.end_year_pc = data_general.validate_year_range(data_sim, start_year_pc, end_year_pc, process_name="PC")
        print(f"\tYears selected for PCs computation: {self.start_year_pc}-{self.end_year_pc}.", flush=True)
        

        # Compute/load EEOFs
        if self.obs:
            self.obs_name = 'NOAA'
            data_sim.data, self.eeof_summer, self.eeof_winter, self.pcs_obs = self._load_and_regrid_obs_data(data_sim.data, obs_path)
            print(f"\tEEOF analysis loaded for '{self.obs_name}' observations between {self.start_year_eeof} and {self.end_year_eeof}."
                  " See attributes `eeof_summer`, `eeof_winter` and `pcs_obs` for results.", flush=True)

        # Filter simulation data
        data_unfiltered_sim = data_sim.data[var_name].sortby("lat").sel(lat=slice(*iso_config.lat_range)).compute()
        data_filtered_sim = iso_scores.apply_lanczos_bandpass_filter(data_unfiltered_sim, iso_config.window_size, iso_config.low_freq, iso_config.high_freq)
        print("\tSimulation data filtered for ISO timescales.", flush=True)
        
        if not self.obs:
            # Compute EEOFs from simulation data
            eeof_summer, eeof_winter = self._compute_EEOFs(data_filtered_sim, iso_config.lag, iso_config.n_lags, iso_config.n_modes)
            self.pcs_obs = None
            self.eeof_summer = eeof_summer.compute()
            self.eeof_winter = eeof_winter.compute()
            print(f"\tEEOF analysis completed for '{self.sim_name}' data between {self.start_year_eeof} and {self.end_year_eeof}."
                  " See attributes `eeof_summer` and `eeof_winter` for results.", flush=True)


        # Compute PCs
        self.scores = {}
        self.pcs_sim, self.scores['alpha'] = self._compute_PCs(data_filtered_sim)
        print(f"\tPCs (bimodal ISO indices) computed for '{self.sim_name}' data between {self.start_year_pc} and {self.end_year_pc}."
              " See attribute `pcs_sim` for results.", flush=True)

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


    def _load_and_regrid_obs_data(self, data_sim, obs_path=None):
        """
        Load observational data and regrid simulation or observational data if needed
        to match resolutions.

        Parameters
        ----------
        data_sim : xr.Dataset
            Simulation data.
        obs_path : str
            Path to the observational NOAA data file. As of now, only necessary if the 
            resolution of the NOAA data (2.5°x2.5°) is higher than that of the simulation data.

        Returns
        -------
        data_sim : xr.Dataset
            Regridded simulation data if regridding was necessary, otherwise the original data.
        eeof_summer : xr.DataArray
            EEOFs for boreal summer from observations.
        eeof_winter : xr.DataArray
            EEOFs for boreal winter from observations.
        pcs_obs : xr.DataArray
            PCs from observations.
        """
        # TO DO: since the EEOF analysis with xeofs takes just a few seconds, the code would be clearer and
        # more efficient if we always loaded the NOAA observational data, regridded it if needed, and 
        # then computed the EEOFs, PCs, etc. Perhaps even using the ObservationData class?

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
                data_obs = xr.open_dataset(obs_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"NOAA observations data file not found at '{obs_path}'")
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
        data_sim : xr.DataArray
            Filtered simulation data.
        lag : int
            Lag timesteps (default: 5).
        n_lags : int
            Number of lag copies (default: 3).
        n_modes : int
            Number of EEOFs modes to compute (default: 2).

        Returns
        -------
        eeof_summer : xr.DataArray
            EEOFs for boreal summer for simulated data.
        eeof_winter : xr.DataArray
            EEOFs for boreal winter for simulated data.
        """

        data_eeof_sim = data_sim.sel(time=slice(str(self.start_year_eeof), str(self.end_year_eeof)))
        eeof_summer = iso_scores.perform_EEOF_analysis(data_eeof_sim, self.start_year_eeof, self.end_year_eeof, 'boreal_summer', lag, n_lags, n_modes)
        eeof_winter = iso_scores.perform_EEOF_analysis(data_eeof_sim, self.start_year_eeof, self.end_year_eeof, 'boreal_winter', lag, n_lags, n_modes)

        return eeof_summer, eeof_winter


    def _compute_PCs(self, data_sim):
        """
        Compute Principal Components (PCs) from the simulation data for the requested years, 
        and adjust them with observational data if requested.
        
        Parameters
        ----------
        data_sim : xr.DataArray
            Filtered simulation data.

        Returns
        -------
        pcs_sim : xr.DataArray
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
                raise ValueError("Observational PCs are not available. "
                                 "Set `obs=True` when calling the `compute_iso_scores` method to load them.")
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
            name_file = f"{self.sim_name.replace(' ', '-')}_{self.obs_name.replace(' ', '-')}"

            if self.correct_pc:
                plot_title += f' (corrected PCs)'
                name_file += f"_corrected_PCs"
        else:
            name_file = f"{self.sim_name.replace(' ', '-')}"

        # Plot frequency of ISO events
        freq_plot, _ = plot.plot_freq_ISO(self.freq_sim, self.freq_obs, alpha=self.scores['alpha'], corr=self.scores['R'],
                                           sigma=self.scores['sigma'], tss=self.scores['TSS'],
                                           title=plot_title, sim_label=self.sim_name, obs_label=self.obs_name)

        plot.save_or_show_plot(freq_plot, output_path, plot_filename=f"freq_ISO_{name_file}_{self.start_year_pc}-{self.end_year_pc}_projected_{self.start_year_eeof}-{self.end_year_eeof}",
                               plot_name=f"Mean monthly frequency of ISO events plot")

        return
    
    
    def scores_table(self, output_path=None, reference=True):
        """
        Generate and save/display table plot with ISO scalar scores comparing simulations
        and observations.

        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) (default: True).
        """

        # Validate that scores are available
        if not self.obs:
            raise ValueError("Scalar scores cannot be computed without observations. "
                             "Set `obs=True` when calling the `compute_iso_scores` method to compute them.")


        # Prepare data and plotting parameters  
        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_pc}-{self.end_year_pc}"
        plot_title = f"ISO scalar scores ({year_range})"

        scalar_scores = [self.scores['alpha'], self.scores['R'], self.scores['sigma'], self.scores['TSS']]
        if reference:
            data_scalar_scores = np.stack([[1.0, 1.0, 1.0, 1.0], scalar_scores])
            rows_scalar_scores = [self.obs_name, self.sim_name]
            name_file = f"{sim_name_file}_ref_{obs_name_file}"
        else:
            data_scalar_scores = np.array(scalar_scores).reshape(1, -1)
            rows_scalar_scores = [self.sim_name]
            name_file = f"{sim_name_file}_no_ref_{obs_name_file}"

        if self.correct_pc:
            plot_title += f' (corrected PCs)'
            name_file += f"_corrected_PCs"

        cols_scalar_scores = ["", r"  $\alpha$  ", r"    $R$    ", r"  $\sigma$  ", r"$\text{TSS}$"]    # Same number of characters needed to get same column width
        cbar_ticks = ['Worse performance', ' ', 'Better performance']
        colors = ("RedGreen", ['tab:red', 'white', 'tab:green'])


        # Generate and save/display table plot
        scalar_scores_table, _ = plot.plot_table(data_scalar_scores, title=plot_title, col_labels=cols_scalar_scores, row_labels=rows_scalar_scores,
                                                 cbar_ticks=cbar_ticks, colors=colors, reference=reference, decimals=2)

        plot.save_or_show_plot(scalar_scores_table, output_path, plot_filename=f"ISO_scalar_scores_table_{name_file}_{year_range}_projected_{self.start_year_eeof}-{self.end_year_eeof}",
                               plot_name=f"ISO scalar scores table plot")

        return