import warnings
warnings.simplefilter("always")

import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from collections.abc import Iterable
from matplotlib.colors import LinearSegmentedColormap

from pyhanami.config import config_params
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData
from pyhanami.utils import data_general, iso_metrics, plot

VARIABLES = data_general.load_yaml_file(config_params.VARIABLES_PATH)

class BimodalISO:
    """
    Compute bimodal ISO indices and derived statistics.

    This class provides functionality for computing the bimodal ISO indices following (K. Kikuchi, 2020) and 
    plotting the results for the selected years, as well as, computing statistics comparing simulation and 
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
    freq_ISO_sim : xarray.DataArray
        Mean monthly frequency of occurrence for simulation data.
    freq_ISO_obs : xarray.DataArray or None
        Mean monthly frequency of occurrence for observational data, or None if not computed.
    stats : dict
        Dictionary containing various statistics comparing simulations and observational data:
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
        years = data_sim.data.time.dt.year
        start_year = int(years.min())
        end_year = int(years.max())

        if self.obs:
            if start_year_eeof is not None or end_year_eeof is not None:
                warnings.warn("\t'start_year_eeof' and 'end_year_eeof' are ignored when `obs=True`. "
                            f"Using predefined observational period for EEOFs: {config_params.NOAA_START_YEAR}-{config_params.NOAA_END_YEAR}.")
            self.start_year_eeof = config_params.NOAA_START_YEAR
            self.end_year_eeof = config_params.NOAA_END_YEAR
        else:
            if start_year_eeof is None:
                start_year_eeof = start_year
                print(f"\tAs no start year was provided for the EEOF analysis, the first year available in the '{self.sim_name}'",
                      f" dataset ({start_year}) will be used.", flush=True)
            if end_year_eeof is None:
                end_year_eeof = end_year
                print(f"\tAs no end year was provided for the EEOF analysis, the last year available in the '{self.sim_name}'",
                      f" dataset ({end_year}) will be used.", flush=True)
            if start_year_eeof == end_year_eeof:
                raise ValueError("More than one year is needed for the EEOF analysis (at least 10 years is recommended, ideally ~ 30 years).")
            if start_year_eeof > end_year_eeof:
                raise ValueError(f"'start_year_eeof' ({start_year_eeof}) must be less than 'end_year_eeof' ({end_year_eeof}).")
            if start_year_eeof < start_year or end_year_eeof > end_year:
                raise ValueError(f"EEOF years ({start_year_eeof}-{end_year_eeof}) must be within the available simulation data range ({start_year}-{end_year}).")
            self.start_year_eeof = start_year_eeof
            self.end_year_eeof = end_year_eeof

        if start_year_pc is None:
            start_year_pc = start_year
        if end_year_pc is None:
            end_year_pc = end_year
        if start_year_pc < start_year or end_year_pc > end_year:
            raise ValueError(f"PC years ({start_year_pc}-{end_year_pc}) must be within the available simulation data range ({start_year}-{end_year}).")
        self.start_year_pc = start_year_pc
        self.end_year_pc = end_year_pc
        

        # Compute/load EEOFs
        if self.obs:
            self.obs_name = 'NOAA'
            
            # Load observational data grid 
            try:
                noaa_grid = xr.open_dataset(config_params.NOAA_GRID_PATH)
            except FileNotFoundError:
                raise FileNotFoundError(f"NOAA grid file not found at '{config_params.NOAA_GRID_PATH}'.")
            
            # Compute resolutions
            sim_lat_res = abs(data_sim.data.lat[1] - data_sim.data.lat[0]).values
            sim_lon_res = abs(data_sim.data.lon[1] - data_sim.data.lon[0]).values
            obs_lat_res = abs(noaa_grid.lat[1] - noaa_grid.lat[0]).values
            obs_lon_res = abs(noaa_grid.lon[1] - noaa_grid.lon[0]).values
            
            sim_resolution = (sim_lat_res + sim_lon_res) / 2
            obs_resolution = (obs_lat_res + obs_lon_res) / 2
            
            # Regrid simulations if their resolution is higher
            if sim_resolution < obs_resolution:  
                try: 
                    self.eeof_summer = xr.open_dataset(config_params.NOAA_EEOF_SUMMER_PATH)
                    self.eeof_winter = xr.open_dataset(config_params.NOAA_EEOF_WINTER_PATH)
                    self.pcs_obs = xr.open_dataset(config_params.NOAA_PC_PATH)
                except FileNotFoundError:   
                    raise FileNotFoundError(f"Some or all required NOAA analysis files not found: "
                                            f"'{config_params.NOAA_EEOF_SUMMER_PATH}', "
                                            f"'{config_params.NOAA_EEOF_WINTER_PATH}', "
                                            f"'{config_params.NOAA_PC_PATH}'.")

                data_sim.data = data_general.regrid_data(data_sim.data, noaa_grid)
                print(f"\tSimulation data regridded to match observations' resolution (~{obs_resolution:.2f}°).")
                
            # Regrid observations if their resolution is higher
            elif obs_resolution < sim_resolution:
                try:
                    data_obs = xr.open_dataset(config_params.NOAA_PATH)
                except FileNotFoundError:
                    raise FileNotFoundError(f"NOAA observations data file not found at '{config_params.NOAA_PATH}'")
                data_obs_regrid = data_general.regrid_data(data_obs, data_sim.data)

                self.eeof_summer, self.eeof_winter, self.pcs_obs = iso_metrics.prepare_NOAA_iso_data(data_obs_regrid)
                del data_obs, data_obs_regrid

                # Directly regrid EEOFs (not used anymore as redoing the EEOF analysis is now computationally affordable)
                # eeof_summer_regrid = data_general.regrid_data(self.eeof_summer, data_sim.data.sortby("lat").sel(lat=slice(*lat_range)), var='eeof')
                # eeof_winter_regrid = data_general.regrid_data(self.eeof_winter, data_sim.data.sortby("lat").sel(lat=slice(*lat_range)), var='eeof')

                # eeof_summer_copy = self.eeof_summer.copy()
                # eeof_winter_copy = self.eeof_winter.copy()

                # eeof_summer_no_eeof = eeof_summer_copy.drop_vars('eeof').drop_dims(['lat', 'lon'])
                # eeof_winter_no_eeof = eeof_winter_copy.drop_vars('eeof').drop_dims(['lat', 'lon'])

                # self.eeof_summer = xr.merge([eeof_summer_regrid, eeof_summer_no_eeof])
                # self.eeof_winter = xr.merge([eeof_winter_regrid, eeof_winter_no_eeof])

                print(f"\tObservational data regridded to match simulations' resolution (~{sim_resolution:.2f}°).")

            print(f"\tEEOF analysis loaded for '{self.obs_name}' observations between {self.start_year_eeof} and {self.end_year_eeof}.")

        # Filter simulation data
        data_unfiltered_sim = data_sim.data[var_name].sortby("lat").sel(lat=slice(*lat_range)).compute()
        data_filtered_sim = iso_metrics.apply_lanczos_bandpass_filter(data_unfiltered_sim, window, low_freq, high_freq)
        print("\tSimulation data filtered for ISO timescales.", flush=True)
        
        if not self.obs:
            # Compute EEOFs from simulation data
            eeof_summer, eeof_winter = self._compute_EEOFs(data_filtered_sim, lag, n_lags, n_modes)
            self.pcs_obs = None
            self.eeof_summer = eeof_summer.compute()
            self.eeof_winter = eeof_winter.compute()
            print(f"\tEEOF analysis completed for '{self.sim_name}' data between {self.start_year_eeof} and {self.end_year_eeof}."
                  " See attributes `eeof_summer` and `eeof_winter` for results.", flush=True)


        # Compute PCs and ISO statistics
        self.stats = {}
        self.pcs_sim, self.stats['alpha'] = self._compute_PCs(data_filtered_sim)
        print(f"\tPCs (bimodal ISO indices) computed between {self.start_year_pc} and {self.end_year_pc}."
              " See attribute `pcs_sim` (and `pcs_obs` if `obs=True`) for results.", flush=True)

        self.freq_ISO_sim, self.freq_ISO_obs, self.stats['R'], self.stats['sigma'], self.stats['TSS'] = self._compute_ISO_stats()
        print(f'\tMean monthly frequency computed between {self.start_year_pc} and {self.end_year_pc}.'
              ' See attributes `freq_ISO_sim` (and `freq_ISO_obs` if `obs=True`) for results.', flush=True)
        
        if self.obs:
            print(f"\tTaylor Skill Score (TSS) between simulations and observations computed (stored in attribute `stats`):"
                  f"\n\t\tRatio PCs amplitudes ($\\alpha$): {self.stats['alpha']:.2f}"
                  f"\n\t\tTemporal correlation (R): {self.stats['R']:.2f}"
                  f"\n\t\tRatio standard deviations ($\\sigma$): {self.stats['sigma']:.2f}"
                  f"\n\t\tTaylor Skill Score (TSS): {self.stats['TSS']:.2f}", flush=True)
        print("\nBimodal ISO indices computation completed.", flush=True)

        return


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
        eeof_summer = iso_metrics.perform_EEOF_analysis(data_eeof_sim, self.start_year_eeof, self.end_year_eeof, 'boreal_summer', lag, n_lags, n_modes)
        eeof_winter = iso_metrics.perform_EEOF_analysis(data_eeof_sim, self.start_year_eeof, self.end_year_eeof, 'boreal_winter', lag, n_lags, n_modes)

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
        pcs_sim = iso_metrics.compute_PCs(data_pcs_sim, [self.eeof_winter, self.eeof_summer])

        alpha = None
        if self.obs: 
            # Compute alpha (only with raw PCs' amplitudes)
            alpha_num = pcs_sim[f'amp_MJO_raw'].mean(dim='time') + pcs_sim[f'amp_BSISO_raw'].mean(dim='time')
            alpha_den = self.pcs_obs[f'amp_MJO_raw'].mean(dim='time') + self.pcs_obs[f'amp_BSISO_raw'].mean(dim='time')
            alpha = ((alpha_num / alpha_den).values).item()

            # Adjust PCs if requested
            if self.correct_pc:
                pcs_sim = iso_metrics.adjust_PCs(pcs_sim, alpha)
                pcs_sim.attrs['alpha'] = alpha
                print(f"\tSimulated PCs have been adjusted using the '{self.obs_name}' observations.", flush=True)
        elif self.correct_pc:
            warnings.warn("Simulated PCs cannot be adjusted without observations. Continuing without modification.")

        return pcs_sim, alpha


    def _compute_ISO_stats(self):
        """
        Compute monthly frequency of ISO events, and statistics comparing simulation and
        observational data if requested.
        
        Returns
        -------
        freq_ISO_sim : xr.DataArray
            Mean monthly frequency of occurrence for simulation data.
        freq_ISO_obs : xr.DataArray or None
            Mean monthly frequency of occurrence for observational data, or None if not computed.
        corr : float
            Temporal correlation coefficient of the seasonality.
        sigma : float
            Ratio of the standard deviations (model/obs) of the seasonality.
        tss : float
            Taylor Skill Score.
        """

        events_sim = self.pcs_sim['label']
        freq_ISO_sim = iso_metrics.compute_freq_ISO(events_sim)

        if self.obs:
            # TO ADD: precompute freq_ISO_obs and just load it here?
            events_obs = self.pcs_obs['label']
            freq_ISO_obs = iso_metrics.compute_freq_ISO(events_obs)
            corr, sigma, tss = iso_metrics.compute_TSS(freq_ISO_sim, freq_ISO_obs)

            # Add statistics as attributes to freq_ISO_sim
            freq_ISO_sim.attrs['R'] = corr
            freq_ISO_sim.attrs['sigma'] = sigma
            freq_ISO_sim.attrs['TSS'] = tss
        else:
            freq_ISO_obs, corr, sigma, tss = None, None, None, None

        return freq_ISO_sim, freq_ISO_obs, corr, sigma, tss


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

        # Determine dataset name for titles
        if self.obs:
            name = self.obs_name
        else:
            name = self.sim_name

        # Save EEOFS
        eeof_summer_path = output_path / f"eeof_boreal_summer_{('_').join(name.split())}_{self.start_year_eeof}-{self.end_year_eeof}.nc"
        self.eeof_summer.to_netcdf(eeof_summer_path)
        print(f"EEOFs computed from '{name}' for boreal summer saved to '{eeof_summer_path}'.", flush=True)

        eeof_winter_path = output_path / f"eeof_boreal_winter_{('_').join(name.split())}_{self.start_year_eeof}-{self.end_year_eeof}.nc"
        self.eeof_winter.to_netcdf(eeof_winter_path)
        print(f"EEOFs computed from '{name}' for boreal winter saved to '{eeof_winter_path}'.", flush=True)

        # Save PCs
        if self.obs:
            pcs_sim_path = output_path / f"pcs_{('_').join(self.sim_name.split())}_projected_on_{('_').join(self.obs_name.split())}_{self.start_year_pc}-{self.end_year_pc}.nc"
            self.pcs_sim.to_netcdf(pcs_sim_path)
            print(f"PCs (bimodal ISO indices) computed for '{self.sim_name}' simulations saved to '{pcs_sim_path}'.", flush=True)

            pcs_obs_path = output_path / f"pcs_{('_').join(self.obs_name.split())}_projected_{config_params.NOAA_START_YEAR}-{config_params.NOAA_END_YEAR}.nc"
            self.pcs_obs.to_netcdf(pcs_obs_path)
            print(f"PCs (bimodal ISO indices) computed for '{self.obs_name}' observations saved to '{pcs_obs_path}'.", flush=True)
        else:
            pcs_sim_path = output_path / f"pcs_{('_').join(self.sim_name.split())}_projected_{self.start_year_pc}-{self.end_year_pc}.nc"
            self.pcs_sim.to_netcdf(pcs_sim_path)
            print(f"PCs (bimodal ISO indices) computed for '{self.sim_name}' simulations saved to '{pcs_sim_path}'.", flush=True)

        # Save frequency of ISO events
        if self.obs:
            freq_ISO_sim_path = output_path / f"freq_ISO_{('_').join(self.sim_name.split())}_projected_on_{('_').join(self.obs_name.split())}_{self.start_year_pc}-{self.end_year_pc}.nc"
            self.freq_ISO_sim.to_netcdf(freq_ISO_sim_path)
            print(f"Mean monthly frequency of ISO events computed for '{self.sim_name}' saved to '{freq_ISO_sim_path}'.", flush=True)

            freq_ISO_obs_path = output_path / f"freq_ISO_{('_').join(self.obs_name.split())}_projected_{config_params.NOAA_START_YEAR}-{config_params.NOAA_END_YEAR}.nc"
            self.freq_ISO_obs.to_netcdf(freq_ISO_obs_path)
            print(f"Mean monthly frequency of ISO events computed for '{self.obs_name}' observations saved to '{freq_ISO_obs_path}'.", flush=True)
        else:
            freq_ISO_sim_path = output_path / f"freq_ISO_{('_').join(self.sim_name.split())}_projected_{self.start_year_pc}-{self.end_year_pc}.nc"
            self.freq_ISO_sim.to_netcdf(freq_ISO_sim_path)
            print(f"Mean monthly frequency of ISO events computed for '{self.sim_name}' saved to '{freq_ISO_sim_path}'.", flush=True)

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

        # Determine dataset name for titles
        if self.obs:
            name = self.obs_name
        else:
            name = self.sim_name


        # Plot EEOFs for borean summer
        eeofs_plot, _ = plot.plot_eeofs(self.eeof_summer, clon=clon, title=f"BSISO convective pattern '{name}' (JJASO {self.start_year_eeof}-{self.end_year_eeof})",
                                cb_label=f"scaled EEOF ({VARIABLES[var_name]['units']})", cmap=LinearSegmentedColormap.from_list("GreenOrange", ['tab:green', 'white', 'tab:orange']))       
        if output_path is None:
            plt.show()
            print("BSISO EEOFs plot created and displayed.", flush=True)
        else:
            eeofs_path = output_path / f"eeof_boreal_summer_{('_').join(name.split())}_{self.start_year_eeof}-{self.end_year_eeof}_clon_{clon}.png"

            eeofs_plot.savefig(eeofs_path, bbox_inches='tight', dpi=150)
            print(f"BSISO EEOFs plot created and saved to '{eeofs_path}'.", flush=True)


        # Plot EEOFs for borean winter
        eeofw_plot, _ = plot.plot_eeofs(self.eeof_winter, clon=clon, title=f"MJO convective pattern '{name}' (DJFMA {self.start_year_eeof}-{self.end_year_eeof})",
                                        cb_label=f"scaled EEOF ({VARIABLES[var_name]['units']})", cmap=LinearSegmentedColormap.from_list("BlueRed", ['tab:blue', 'white', 'tab:red']))
        
        if output_path is None:
            plt.show()
            print("MJO EEOFs plot created and displayed.", flush=True)
        else:
            eeofw_path = output_path / f"eeof_boreal_winter_{('_').join(name.split())}_{self.start_year_eeof}-{self.end_year_eeof}_clon_{clon}.png"

            eeofw_plot.savefig(eeofw_path, bbox_inches='tight', dpi=150)
            print(f"MJO EEOFs plot created and saved to '{eeofw_path}'.", flush=True)

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

        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        # Determine dataset to plot
        if data == 'sim':
            pcs_data = self.pcs_sim
            if self.obs:
                name_title = f"'{self.sim_name}' projected on '{self.obs_name}'" 
                name_projected = f"_{('_').join(self.obs_name.split())}"
            else:
                name_title = f"'{self.sim_name}'"
                name_projected = ''
            name_file = ('_').join(self.sim_name.split())
        elif data == 'obs':
            if not self.obs:
                raise ValueError("Observational PCs are not available. Set 'obs=True' when initializing the bimodalISO object to load them.")
            pcs_data = self.pcs_obs
            name_title = f"'{self.obs_name}'"
            name_file = ('_').join(self.obs_name.split())
            name_projected = ''

        # Validate years input
        if years is None:
            raise ValueError(f"Provide at least one year to plot the PCs for.")
        if isinstance(years, int):
            years = [years]
        if not any(year in pcs_data.time.dt.year for year in years):
            raise ValueError(f"Some years are missing in the PCs data.")

        # Plot PCs for selected years
        for year in years:
            pcs_year = pcs_data.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
            pcs_plot, _ = plot.plot_pcs(pcs_year, title=f"Bimodal ISO indices {name_title} ({year})")

            if output_path is None:
                plt.show()
                print(f"PCs (bimodal ISO indices) for year {year} plot created and displayed.", flush=True)
            else:
                pcs_path = output_path / f"pcs_{name_file}_{year}_projected{name_projected}_{self.start_year_eeof}-{self.end_year_eeof}.png"

                # pcs_year.to_netcdf(pcs_path.with_suffix('.nc'))
                pcs_plot.savefig(pcs_path, bbox_inches='tight', dpi=150)
                print(f"PCs (bimodal ISO indices) plot for year {year} created and saved to '{pcs_path}'.", flush=True)

        return


    def freq_ISO_plot(self, output_path=None):
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
            name_file = f"{('_').join(self.sim_name.split())}_vs_{('_').join(self.obs_name.split())}"

            if self.correct_pc:
                plot_title += f' (corrected PCs)'
                name_file += f"_corrected_PCs"
        else:
            name_title = f"'{self.sim_name}'"
            name_file = f"{('_').join(self.sim_name.split())}"

        # Plot frequency of ISO events
        freq_plot, _ = plot.plot_freq_ISO(self.freq_ISO_sim, self.freq_ISO_obs, alpha=self.stats['alpha'], corr=self.stats['R'],
                                           sigma=self.stats['sigma'], tss=self.stats['TSS'],
                                           title=plot_title, sim_label=self.sim_name, obs_label=self.obs_name)
        
        if output_path is None:
            plt.show()
            print(f"Mean monthly frequency of ISO events for {name_title} plot created and displayed.", flush=True)
        else:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            freq_path = output_path / f"freq_ISO_{name_file}_{self.start_year_pc}-{self.end_year_pc}_projected_{self.start_year_eeof}-{self.end_year_eeof}.png"
            freq_plot.savefig(freq_path, bbox_inches='tight', dpi=150)
            print(f"Mean monthly frequency of ISO events for {name_title} plot created and saved to '{freq_path}'.", flush=True)

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
    
    
    def compute_bimodal_ISO(self, data_name=None, start_year_eeof=None, end_year_eeof=None, start_year_pc=None, end_year_pc=None, obs=False, 
                            correct_pc=False, lat_range=(-30, 30), lag=5, n_lags=3, n_modes=2, window=141, low_freq=1/90, high_freq=1/25):
        """
        Initialize and compute bimodal ISO indices (following (K. Kikuchi, 2020)) and derived 
        statistics (following (M. Nakano et al., 2019)) for a selected dataset.

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
        """

        # Validate input
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError("At least one dataset is required for the Bimodal ISO indices.")
            data_ISO = self.datasets[0]
            data_name = data_ISO.name
        elif isinstance(data_name, str):
            data_ISO = [ds for ds in self.datasets if ds.name == data_name]
            if not data_ISO:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_ISO = data_ISO[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")
        
        # Create BimodalISO object and compute indices/statistics
        print(f"Performing ISO analysis for dataset '{data_name}':", flush=True)
        bimodal_indices = BimodalISO(data_ISO, start_year_eeof=start_year_eeof, end_year_eeof=end_year_eeof, start_year_pc=start_year_pc, 
                                      end_year_pc=end_year_pc, obs=obs, correct_pc=correct_pc, lat_range=lat_range, lag=lag, n_lags=n_lags,
                                      n_modes=n_modes, window=window, low_freq=low_freq, high_freq=high_freq)

        return bimodal_indices
