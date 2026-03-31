import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pyhanami.config import config_params
from pyhanami.diags.Simulations import SimulationData
from pyhanami.utils import data_general, config_scores, plot
from pyhanami.utils.mjo_scores import mjo_ceof_funcs, mjo_spectrum_funcs


class MJOEvaluation:
    """
    Compute Real-Time Multivariate MJO (RMM) indices and MJO wavenumber-frequency power spectra, 
    and derived scalar scores.

    This class provides functionality for computing the RMM MJO indices and derived scalar scores 
    following (M.C. Wheeler & H.H. Hendon, 2004), as well as the MJO wavenumber-frequency power  
    spectra following (M.C. Wheeler & G.N. Kiladis, 1999) and derived scalar scores following
    (M.-S. Ahn et al., 2017). It also includes tools to plot the results comparing simulations 
    to observational data.

    Parameters
    ----------
    data_sim : SimulationData
        Simulation dataset to use.
    obs_path : str
        Path to the observational data file with the necessary variables for the MJO analysis
        (default: config_params.MJO_VARS_PATH).
    start_year_mjo, end_year_mjo : int, optional
        Initial and end years to perform the analysis for.
    start_year_ref, end_year_ref : int, optional
        Initial and end years for computing the reference seasonal cycle. If None, taken as
        the initial and end years for the whole MJO analysis.
    threshold_active_days : float
        Threshold for the amplitude of the RMM indices (first two PCs) to consider the MJO active at 
        a given day. If None, the mean MJO amplitude over the entire considered time period is used 
        as a threshold.
    spectrum_var : str
        Variable to be used for the spectral analysis (default: 'rlut').
    mjo_config : MJOConfig
        Configuration dataclass with parameters necessary for the MJO evaluation. If None, default values 
        from the configuration file `pyhanami.config.scientific_evaluation_parameters.yaml` will be used.


    Attributes
    ----------
    sim_name : str
        Name of the simulation dataset.
    obs_name : str
        Name of the observational dataset.
    data_res : float
        Resolution of the data used for the MJO analysis.
    start_year_mjo, end_year_mjo : int
        Initial and end years to perform the MJO analysis for.
    data_ceof_sim : xr.DataArray
        Simulation data with longer-time-scale components removed for all three variables.
    data_ceof_obs : xr.DataArray
        Observational data with longer-time-scale components removed for all three variables.
    ceof_obs : xr.Dataset
        Output of CEOF analysis for observational data ('ceof', 'eigval', 'var_frac' 
        and 'pc' for the first 'n_modes').
    ceof_sim_on_obs : xr.Dataset
        Output of CEOF analysis for simulated data projected on observed CEOFs ('ceof', 
        'eigval', 'var_frac' and 'pc' for the first 'n_modes').
    ceof_sim_on_sim : xr.Dataset
        Output of CEOF analysis for simulated data projected on simulated CEOFs ('ceof', 
        'eigval', 'var_frac' and 'pc' for the first 'n_modes').
    ceof_scores : xr.Dataset
        Scalar scores related to the CEOF analysis, including the correlation between simulated and
        observed CEOFs ('ceof_corr') per mode, the explained variance for each CEOF ('explained_var',
        'explained_var_bias') for variable and mode, the lead-lag correlation between the RMM indices 
        ('lead_lag_corr'), its maximum value ('max_lead_lag_corr', 'max_lead_lag_corr_bias') and the
        MJO period ('pceof', 'pceof_bias').
    activity_per_phase : xr.Dataset
        Absolute values and bias in mean amplitude and days per phase (total and only for active MJO 
        days). It contains the following variables: 'mean_amplitude', 'mean_amplitude_bias', 
        'mean_active_amplitude', 'mean_active_amplitude_bias', 'total_counts', 'total_counts_bias',
        'active_counts' and 'active_counts_bias' per phase for observations ('obs'), simulations 
        projected on observed CEOFs ('sim_on_obs'), and simulations projected on their own CEOFs 
        ('sim_on_sim').
    spectrum_var : str
        Climate variable used for the spectral analysis.
    data_spectra_sim : xr.DataArray
        Simulation data with seasonal cycle removed for the selected variable.
    data_spectra_obs : xr.DataArray
        Observational data with seasonal cycle removed for the selected variable.
    power_spectra : xr.Dataset
        Power spectra. It contains the following variables: 'sym_spec' (normalized 
        symmetric spectrum), 'asym_spec' (normalized antisymmetric spectrum), and 
        'background' (smoothed background spectrum) for both observations ('obs')
        and simulations ('sim').
    mjo_freq_bounds : tuple
        Frequency bounds corresponding to the MJO band in the wavenumber-frequency space.
    mjo_wavenum_bounds : tuple
        Wavenumber bounds corresponding to the MJO band in the wavenumber-frequency space.
    power_scores : xr.Dataset
        Scalar scores related to the symmetric power spectra, including the eastward/westward 
        power ratio ('ew_ratio', 'ew_ratio_bias'), the eastward/observed power ratio ('eo_ratio', 
        'eo_ratio_bias') and the dominant eastward period from the wavenumber-frequency power 
        spectra ('pwfps', 'pwfps_bias') in the MJO band, for both observations ('obs') and 
        simulations ('sim').
    cbar_ticks_bias : list[str]
        Colorbar ticks labels for bias tables.
    colors_bias : tuple
        Colorbar colors for bias tables.
    """

    def __init__(self, data_sim, obs_path=config_params.MJO_VARS_PATH, start_year_mjo=None, end_year_mjo=None, start_year_ref=None, 
                 end_year_ref=None, threshold_active_days=None, spectrum_var='rlut', mjo_config=None):

        # Validate input
        if not isinstance(data_sim, SimulationData):
            raise TypeError("'data_sim' must be an instance of SimulationData.")
        self.sim_name = data_sim.name
        self.obs_name = 'Obs'    #'NOAA+ERA5'
        self.data_res = config_params.MJO_OBS_RES
        self.spectrum_var = spectrum_var

        # Load MJO evaluation parameters
        if mjo_config is None:
            mjo_config = config_scores.MJOConfig()
        elif not isinstance(mjo_config, config_scores.MJOConfig):
            raise TypeError("'mjo_config' must be an instance of the MJOConfig dataclass defined in 'pyhanami.utils.config_scores'.")


        # Select years for MJO analysis
        self.start_year_mjo, self.end_year_mjo = data_general.validate_year_range(data_sim, start_year_mjo, end_year_mjo, process_name='MJO')
        if self.start_year_mjo < config_params.MJO_START_YEAR or config_params.MJO_END_YEAR < self.end_year_mjo:
            raise ValueError(f"Selected years for MJO analysis must be within the available observational period"
                             f" ({config_params.MJO_START_YEAR} and {config_params.MJO_END_YEAR}).")
        print(f"\tYears selected for MJO scores computation: {self.start_year_mjo}-{self.end_year_mjo}.", flush=True)
        data_sim_filtered_time = data_sim.data.sel(time=slice(str(self.start_year_mjo), str(self.end_year_mjo))).compute()    


        # Prepare data for the CEOF analysis (regrid simulations to match observations, if needed, and filter seasonal cycle and interannual variability)
        self.data_ceof_sim, _, _, self.data_ceof_obs, _, _ = self._prepare_ceof_data(data_sim_filtered_time, obs_path, start_year_ref, end_year_ref,
                                                                                     mjo_config.lat_range, mjo_config.rolling_window_size, 
                                                                                     mjo_config.n_harmonics, mjo_config.normalize_std)
        print(f'\tObservations and simulations data prepared for the CEOF analysis by removing longer-time-scale components between {self.start_year_mjo}'
              f' and {self.end_year_mjo}. See attributes `data_ceof_sim` and `data_ceof_obs` for results.', flush=True)

        # Perform CEOF analysis (projecting on observed and simulated EOFs)
        self.ceof_obs, self.ceof_sim_on_obs, self.ceof_sim_on_sim = self._perform_CEOF_analysis(mjo_config.n_modes)
        print(f"\tCEOF analyses completed. See attributes `ceof_obs`, `ceof_sim_on_obs`, and `ceof_sim_on_sim` for results.", flush=True)
        
        # Compute scalar scores related to the CEOFs
        self.ceof_scores = self._compute_CEOF_scores()
        print(f"\tScalar scores related to CEOFs computed. See attribute `ceof_scores` for results.", flush=True)

        
        # Compute MJO mean amplitude and days per phase (absolute value and bias)
        self.activity_per_phase = self._compute_activity_per_phase(threshold=threshold_active_days)
        # self.activity_per_phase = self._compute_phase_counts(threshold=threshold_active_days)
        # self.activity_per_phase_bias, self.cbar_ticks_bias, self.colors_bias = self._compute_phase_counts_bias()
        print(f"\tAbsolute values and bias in MJO activity (mean amplitude and days) per phase computation completed."
              f" See attribute `activity_per_phase` for results.", flush=True)


        # Prepare data for the power spectra analysis (regrid simulations, if needed, and remove seasonal cycle)
        self.data_spectra_sim, self.data_spectra_obs = self._prepare_spectra_data(data_sim_filtered_time, obs_path, start_year_ref, end_year_ref, 
                                                                                  self.spectrum_var, mjo_config.n_harmonics)
        print(f"\tObservations and simulations data prepared for the power spectra analysis by removing the seasonal cycle between "
              f"{self.start_year_mjo} and {self.end_year_mjo}. See attributes `data_spectra_sim` and `data_spectra_obs` for results.", flush=True)
        
        # Compute wavenumber-frequency power spectra and related scalar scores
        self.power_spectra = self._compute_power_spectra(mjo_config.seg_size, mjo_config.n_overlap, mjo_config.lat_range)
        print(f"\tWavenumber-frequency power spectra computation completed. See attribute `power_spectra` for results.", flush=True)

        self.mjo_freq_bounds = mjo_config.mjo_freq_bounds
        self.mjo_wavenum_bounds = mjo_config.mjo_wavenum_bounds
        self.power_scores = self._compute_power_scores(self.mjo_freq_bounds, self.mjo_wavenum_bounds)
        print(f"\tScalar scores related to power spectra computed. See attribute `power_scores` for results.", flush=True)
        
        
        # Define plotting parameters
        self.cbar_ticks_bias = ['Negative bias', 'No bias', 'Positive bias']
        self.colors_bias = ("RedGreen", ['tab:red', 'white', 'tab:green'])   #("BlueRed", ['tab:blue', 'white', 'tab:red'])

        print(f"\nMadden-Julian Oscillation scores computation completed between years {self.start_year_mjo} and {self.end_year_mjo}.", flush=True)
        return
    

    def _match_obs_resolution(self, data_sim):
        """
        Regrid simulations if needed to match the resolution of the observations
        (defined in `config_params`).

        Parameters
        ----------
        data_sim : xr.Dataset
            Simulation data to compare with observations.

        Returns
        -------
        data_sim_regrid : xr.Dataset
            Regridded simulation data to match the observational data resolution.
        """

        # Check simulations' resolution and regrid if needed
        obs_res = self.data_res
        sim_lat_res = abs(data_sim.lat[1] - data_sim.lat[0]).values
        sim_lon_res = abs(data_sim.lon[1] - data_sim.lon[0]).values

        # Regrid simulations if the resolutions do not match
        if sim_lat_res != obs_res or sim_lon_res != obs_res:
            data_obs_grid = xr.open_dataset(config_params.MJO_GRID_PATH)
            data_sim_regrid = data_general.regrid_data(data_sim, data_obs_grid)
        else:
            data_sim_regrid = data_sim

        return data_sim_regrid


    def _prepare_ceof_data(self, data_sim, obs_path, start_year_ref, end_year_ref, lat_range=(-15, 15), 
                          rolling_window_size=120, n_harmonics=3, normalize_std=False):
        """
        Regrid simulation data if needed to match the resolution of the observations, then filter the data to
        remove longer-time-scale components (seasonal cycle and interannual variability) at the grid point 
        level and concatenate into a single dataset all three variables necessary for the CEOF analysis.

        Parameters
        ----------
        data_sim : xr.Dataset
            Simulation data containing variables ('ua850', 'ua200', 'rlut').
        obs_path : str
            Path to the observational data file with the necessary variables for the CEOF analysis.
        start_year_ref, end_year_ref : int
            Initial and end years for computing the reference seasonal cycle.
        lat_range : tuple
            Geographic latitude bounds (default: (-15, 15)).
        rolling_window_size : int
            Window size for rolling mean to remove low-frequency variability (default: 120 days).
        n_harmonics : int
            Number of harmonics to remove from the seasonal cycle (default: 3).
        normalize_std : bool
            Whether to normalize anomalies by fixed standard deviations when removing the 
            seasonal cycle (default: False).

        Returns 
        -------
        filtered_sim : xr.DataArray
            Filtered simulation data for each variable.
        anom_sim : xr.DataArray
            Anomalies in simulation data for each variable.
        std_sim : xr.DataArray
            Standard deviations in simulation data for each variable.
        filtered_obs : xr.DataArray
            Filtered observational data for each variable.
        anom_obs : xr.DataArray
            Anomalies in observational data for each variable.
        std_obs : xr.DataArray
            Standard deviations in observational data for each variable.
        """

        vars_mjo = ['ua850', 'ua200', 'rlut']


        # Load observational data (NOAA + ERA5)
        try:
            data_obs_vars = xr.open_dataset(obs_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Observations MJO data file not found at '{obs_path}'")
        data_obs_vars = data_obs_vars[vars_mjo].sel(time=slice(str(self.start_year_mjo), str(self.end_year_mjo)))


        # Extract MJO variables from simulation data
        for var_name in vars_mjo:
            if var_name not in data_sim.data_vars:
                raise ValueError(f"Variable '{var_name}' required for the CEOF analysis not found in the simulated dataset '{self.sim_name}'.")
        data_sim_vars = data_sim[vars_mjo]

        # Regrid simulations if needed to match observations' resolution
        data_sim_vars_regrid = self._match_obs_resolution(data_sim_vars)


        # Prepare reference years
        if start_year_ref is None:
            start_year_ref = self.start_year_mjo
        if end_year_ref is None:
            end_year_ref = self.end_year_mjo

        # Filter data and remove longer-time-scale components
        filtered_obs, anom_obs, std_obs = mjo_ceof_funcs.remove_longer_time_scale_components(data_obs_vars, start_year_ref, end_year_ref, lat_range, 
                                                                                             rolling_window_size, n_harmonics, normalize_std)
        filtered_sim, anom_sim, std_sim = mjo_ceof_funcs.remove_longer_time_scale_components(data_sim_vars_regrid, start_year_ref, end_year_ref, lat_range,
                                                                                             rolling_window_size, n_harmonics, normalize_std)

        return filtered_sim, anom_sim, std_sim, filtered_obs, anom_obs, std_obs
    

    def _perform_CEOF_analysis(self, n_modes=2):
        """
        Perform Combined Empirical Orthogonal Function (CEOF) analyses on observational and 
        simulation data. For the latter, projecting the data both on the observed CEOFs and
        on its own CEOFs.

        Parameters
        ----------
        n_modes : int
            Number of CEOF modes to compute (default: 2).

        Returns
        -------
        ceof_obs : xr.Dataset
            Output of CEOF analysis for observational data ('eof', 'eigval', 'var_frac' 
            and 'pc' for the first 'n_modes').
        ceof_sim_on_obs : xr.Dataset
            Output of CEOF analysis for simulated data projected on observational CEOFs 
            ('eof', 'eigval', 'var_frac' and 'pc' for the first 'n_modes').
        ceof_sim_on_sim : xr.Dataset
            Output of CEOF analysis for simulated data projected on its own CEOFs 
            ('ceof', 'eigval', 'var_frac' and 'pc' for the first 'n_modes').
        """

        # Load observational model data (not used anymore, kept for reference)
        # try:
        #     model_mjo_obs = xeofs.single.EOF.load(config_params.MJO_MODEL_PATH)
        # except FileNotFoundError:
        #     raise FileNotFoundError(f"Observations MJO CEOF analysis file not found at "
        #                             f"'{config_params.MJO_MODEL_PATH}'")
        

        # Perform observational CEOF analysis (correct sign of the first mode to match the typical
        # MJO pattern from (M.Wheeler et al., (2004))
        model_mjo_obs = mjo_ceof_funcs.fit_CEOF_model_xeofs(self.data_ceof_obs, n_modes)
        ceof_obs = mjo_ceof_funcs.perform_CEOF_analysis(None, model_mjo_obs, n_modes)
        ceof_obs['ceof'].loc[dict(mode=0)] *= -1
        ceof_obs['pc'].loc[dict(mode=0)] *= -1
        ceof_obs.attrs['EOFs source'] = f"Observations '{self.obs_name}'"
        ceof_obs.attrs['PCs source'] = f"Observations '{self.obs_name}' projected on Observations CEOFs"
        

        # Compute PCs for simulations projecting on observed CEOFs
        ceof_sim_on_obs = ceof_obs.copy(deep=True)
        ceof_sim_on_obs.attrs['EOFs source'] = f"Observations '{self.obs_name}'"
        ceof_sim_on_obs.attrs['PCs source'] = f"Simulations '{self.sim_name}' projected on Observations '{self.obs_name}' CEOFs"

        pc_sim_on_obs = model_mjo_obs.transform(self.data_ceof_sim)

        # Correct PCs to match eofs.xarray.Eof output
        pc_sim_on_obs = pc_sim_on_obs.assign_coords(mode=[0, 1]).transpose('time', 'mode')
        pc_sim_on_obs = pc_sim_on_obs/np.sqrt(ceof_sim_on_obs['eigval'])
        pc_sim_on_obs.loc[dict(mode=0)] *= -1
        pc_sim_on_obs.attrs.pop('solver_kwargs', None)
        ceof_sim_on_obs['pc'] = pc_sim_on_obs


        # Perform and correct CEOF analysis on simulations projecting on themselves
        ceof_sim_on_sim = mjo_ceof_funcs.perform_CEOF_analysis(self.data_ceof_sim, None, n_modes)
        ceof_sim_on_sim = mjo_ceof_funcs.correct_CEOFs(ceof_sim_on_sim, ceof_obs)
        ceof_sim_on_sim.attrs['EOFs source'] = f"Simulations '{self.sim_name}'"
        ceof_sim_on_sim.attrs['PCs source'] = f"Simulations '{self.sim_name}' projected on Simulations '{self.sim_name}' CEOFs"

        return ceof_obs, ceof_sim_on_obs, ceof_sim_on_sim
    
    
    def _compute_CEOF_scores(self):
        """
        Compute scalar scores related to the CEOF analysis, including the Pearson 
        correlation between observed and simulated CEOFs, the bias in the associated 
        explained variance per variable and mode, the lead-lag correlation between  
        the first two PCs, its maximum value and the MJO period.

        Alternative: compute variance explained by the first two modes together (add 1
        and 2), and average the spatial correlation coefficients between both modes to 
        produce a single scalar metric.

        Returns
        -------
        ceof_scores : xr.Dataset
            Scalar scores related to the CEOF analysis ('ceof_corr', 'explained_var', 
            'explained_var_bias', 'lead_lag_corr', 'max_lead_lag_corr', 
            'max_lead_lag_corr_bias', 'pceof', 'pceof_bias').
        """

        # Compute correlation between observed and simulated CEOFs
        ceof_obs_corr = mjo_ceof_funcs.compute_CEOFs_corr(self.ceof_obs['ceof'], self.ceof_obs['ceof'])
        ceof_sim_corr = mjo_ceof_funcs.compute_CEOFs_corr(self.ceof_obs['ceof'], self.ceof_sim_on_sim['ceof'])
    
        # Retrieve explained variance of the CEOFs (absolute value and bias)
        ceof_obs_expl_var = self.ceof_obs['var_frac'].rename('explained_var')
        ceof_sim_expl_var = self.ceof_sim_on_sim['var_frac'].rename('explained_var')

        ceof_obs_expl_var_bias = ceof_obs_expl_var.rename('explained_var_bias')
        ceof_sim_expl_var_bias = (ceof_sim_expl_var - ceof_obs_expl_var).rename('explained_var_bias') 


        # Compute lead-lag correlation between the RMM indices (first two PCs)
        lead_lag_corr_obs = mjo_ceof_funcs.compute_lead_lag_correlation(self.ceof_obs['pc'])
        lead_lag_corr_sim = mjo_ceof_funcs.compute_lead_lag_correlation(self.ceof_sim_on_sim['pc'])

        # Compute absolute value and vias in maximum lead-lag correlation
        max_corr_obs_value = mjo_ceof_funcs.compute_max_correlation(lead_lag_corr_obs['lead_lag_corr'])
        max_corr_sim_value = mjo_ceof_funcs.compute_max_correlation(lead_lag_corr_sim['lead_lag_corr'])

        max_corr_obs = xr.DataArray(max_corr_obs_value, name='max_lead_lag_corr')
        max_corr_sim = xr.DataArray(max_corr_sim_value, name='max_lead_lag_corr')

        max_corr_obs_bias = max_corr_obs.rename('max_lead_lag_corr_bias')
        max_corr_sim_bias = xr.DataArray(max_corr_sim - max_corr_obs, name='max_lead_lag_corr_bias')

        
        # Compute absolute value and bias in periodicity from the ceof analysis
        pceof_obs = mjo_ceof_funcs.compute_ceof_periodicity(lead_lag_corr_obs['lead_lag_corr'])
        pceof_sim = mjo_ceof_funcs.compute_ceof_periodicity(lead_lag_corr_sim['lead_lag_corr'])

        pceof_obs_ds = xr.DataArray(pceof_obs, name='pceof')
        pceof_sim_ds = xr.DataArray(pceof_sim, name='pceof')

        pceof_obs_bias = pceof_obs_ds.rename('pceof_bias')
        pceof_sim_bias = (pceof_sim_ds - pceof_obs_ds).rename('pceof_bias')


        # Store all scores in a single dataset
        obs_ds = xr.merge([
            ceof_obs_corr,
            ceof_obs_expl_var.to_dataset(),
            ceof_obs_expl_var_bias.to_dataset(),
            lead_lag_corr_obs,
            max_corr_obs.to_dataset(),
            max_corr_obs_bias.to_dataset(),
            pceof_obs_ds.to_dataset(),
            pceof_obs_bias.to_dataset()
        ])

        sim_ds = xr.merge([
            ceof_sim_corr,
            ceof_sim_expl_var.to_dataset(),
            ceof_sim_expl_var_bias.to_dataset(),
            lead_lag_corr_sim,
            max_corr_sim.to_dataset(),
            max_corr_sim_bias.to_dataset(),
            pceof_sim_ds.to_dataset(),
            pceof_sim_bias.to_dataset()
        ])

        ceof_scores = xr.concat([obs_ds, sim_ds], dim='dataset')
        ceof_scores = ceof_scores.assign_coords(dataset=['obs', 'sim'])

        return ceof_scores


    def _compute_activity_per_phase(self, threshold=None):
        """
        Compute absolute value and bias in the number of MJO days and MJO active
        days per phase for both simulations and observations, including simulations 
        projected on observed CEOFs and on their own CEOFs.

        Parameters
        ----------
        threshold : float
            Threshold for the amplitude of the first two PCs to consider the  
            MJO active at a given day. If None, the mean MJO amplitude over 
            the entire time period is used as a threshold.

        Returns
        -------
        activity_per_phase : xr.Dataset
            Mean amplitude and days per phase (total and only for active MJO days). 
            It contains the following variables: 'mean_amplitude', 'mean_amplitude_bias',
            'mean_active_amplitude', 'mean_active_amplitude_bias', 'total_counts',
            'total_counts_bias', 'active_counts' and 'active_counts_bias' per phase 
            for observations ('obs'), simulations projected on observed CEOFs 
            ('sim_on_obs'), and simulations projected on their own CEOFs ('sim_on_sim').
        """

        # Compute days for observations
        pcs_obs = self.ceof_obs['pc'].sel(time=slice(str(self.start_year_mjo), str(self.end_year_mjo)))
        phase_counts_obs = mjo_ceof_funcs.compute_phase_counts(pcs_obs, threshold)
        # phase_counts_obs.attrs['PCs source'] = self.ceof_obs.attrs['PCs source']

        # Compute days for simulations projected on observed CEOFs
        pcs_sim_on_obs = self.ceof_sim_on_obs['pc']
        phase_counts_sim_on_obs = mjo_ceof_funcs.compute_phase_counts(pcs_sim_on_obs, threshold)
        # phase_counts_sim_on_obs.attrs['PCs source'] = self.ceof_sim_on_obs.attrs['PCs source']
        
        # Compute days for simulations projected on their own CEOFs
        pcs_sim_on_sim = self.ceof_sim_on_sim['pc']
        phase_counts_sim_on_sim = mjo_ceof_funcs.compute_phase_counts(pcs_sim_on_sim, threshold)
        # phase_counts_sim_on_sim.attrs['PCs source'] = self.ceof_sim_on_sim.attrs['PCs source']

        
        # Compile all phase counts into a single dataset
        phase_counts = xr.concat([phase_counts_obs, phase_counts_sim_on_obs, phase_counts_sim_on_sim], dim='dataset')
        phase_counts = phase_counts.assign_coords(dataset=['obs', 'sim_on_obs', 'sim_on_sim'])


        # Commpute biases 
        phase_counts_bias = xr.concat([
            phase_counts.isel(dataset=0).expand_dims('dataset'),
            phase_counts.isel(dataset=slice(1, None)) - phase_counts.isel(dataset=0),
        ], dim='dataset') 


        # Save results in a single dataset
        activity_per_phase = xr.merge([phase_counts, phase_counts_bias.rename({var: var + '_bias' for var in phase_counts_bias.data_vars})])
        
        return activity_per_phase


    def _prepare_spectra_data(self, data_sim, obs_path, start_year_ref, end_year_ref, spectrum_var='rlut',
                              n_harmonics=3):
        """
        Regrid simulation data if needed to match the resolution of the observations, then filter the data to
        remove the seasonal cycle at the grid point level for the selected variable for the spectral analysis.

        Parameters
        ----------
        data_sim : xr.Dataset
            Simulation data containing the selected variable.
        obs_path : str
            Path to the observational data file with the necessary variables for the spectral analysis.
        start_year_ref, end_year_ref : int
            Initial and end years for computing the reference seasonal cycle.
        spectrum_var : str
            Variable to be used for the spectral analysis (default: 'rlut').
        n_harmonics : int
            Number of harmonics to remove from the seasonal cycle (default: 3).

        Returns 
        -------
        filtered_sim : xr.DataArray
            Filtered simulation data for the selected variable.
        filtered_obs : xr.DataArray
            Filtered observational data for the selected variable.
        """
        # Validate input
        vars_mjo = ['ua850', 'ua200', 'rlut']
        if spectrum_var not in vars_mjo:
            raise ValueError(f"Variable '{spectrum_var}' not valid for the spectral analysis. Choose one of {vars_mjo}.")

        # Load observational data (NOAA)
        try:
            data_obs_vars = xr.open_dataset(obs_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Observations MJO data file not found at '{obs_path}'")
        data_obs_vars = data_obs_vars[[spectrum_var]].sel(time=slice(str(self.start_year_mjo), str(self.end_year_mjo)))


        # Extract MJO variable from simulation data
        if spectrum_var not in data_sim.data_vars:
            raise ValueError(f"Variable '{spectrum_var}' required for the spectral analysis not found in the simulated dataset '{self.sim_name}'.")
        data_sim_vars = data_sim[[spectrum_var]]

        # Regrid simulations if needed to match observations' resolution
        data_sim_vars_regrid = self._match_obs_resolution(data_sim_vars)


        # Prepare reference years
        if start_year_ref is None:
            start_year_ref = self.start_year_mjo
        if end_year_ref is None:
            end_year_ref = self.end_year_mjo

        # Filter data and remove seasonal cycle
        filtered_obs = mjo_ceof_funcs.remove_seasonal_cycle(data_obs_vars[spectrum_var], start_year_ref, end_year_ref, n_harmonics)
        filtered_sim = mjo_ceof_funcs.remove_seasonal_cycle(data_sim_vars_regrid[spectrum_var], start_year_ref, end_year_ref, n_harmonics)

        return filtered_sim, filtered_obs


    def _compute_power_spectra(self, seg_size=96, n_overlap=60, lat_range=(-15, 15)):
        """
        Perform wavenumber-frequency analysis and return the normalized spectral symmetric 
        and antisymmetric components obtained dividing by a smoothed background following 
        (M.C. Wheeler & G.N. Kiladis, 1999).

        Parameters
        ----------
        data : xr.DataArray
            Data to compute the power spectra of.
        seg_size : int
            Size of the segments to perform the spectral analysis on, in days (default: 96).
        n_overlap : int
            Number of overlapping points between segments, in days (default: 60).
        lat_range : tuple
            Geographic latitude bounds (default: (-15, 15)).

        Returns
        -------
        power_spectra : xr.Dataset
            Power spectra. It contains the following variables: 'sym_spec' (normalized 
            symmetric spectrum), 'asym_spec' (normalized antisymmetric spectrum), and 
            'background' (smoothed background spectrum) for both observations ('obs')
            and simulations ('sim').            
        """

        # Compute power spectra for observations
        spec_obs_sim, spec_obs_asym, _, _, background_obs = mjo_spectrum_funcs.wavenum_freq_analysis(self.data_spectra_obs, seg_size, 
                                                                                                     n_overlap, lat_range)
        power_spectra_obs = xr.merge([spec_obs_sim.drop_vars('component'), spec_obs_asym.drop_vars('component'), background_obs])

        # Compute power spectra for simulations
        spec_sim_sim, spec_sim_asym, _, _, background_sim = mjo_spectrum_funcs.wavenum_freq_analysis(self.data_spectra_sim, seg_size, 
                                                                                                     n_overlap, lat_range)
        power_spectra_sim = xr.merge([spec_sim_sim.drop_vars('component'), spec_sim_asym.drop_vars('component'), background_sim])


        # Store all spectra in a single dataset
        power_spectra = xr.concat([power_spectra_obs, power_spectra_sim], dim='dataset')
        power_spectra = power_spectra.assign_coords(dataset=['obs', 'sim'])

        return power_spectra


    def _compute_power_scores(self, freq_bounds=None, wavenum_bounds=None, freq_dim='frequency', 
                              wavenum_dim='wavenumber'):
        """
        Compute scalar scores related to the power spectra, including the eastward/westward power ratio 
        (E/W ratio) and the eastward/observed power ratio (E/O ratio) in the wavenumber-frequency space, 
        and the dominant eastward period from the wavenumber-frequency power spectra (P_WFPS) in the MJO 
        band, for both simulations and observations, using the symmetric power spectra.

        Parameters
        ----------
        freq_bounds : tuple
            Frequency bounds corresponding to the MJO band in the wavenumber-frequency space.
        wavenum_bounds : tuple
            Wavenumber bounds corresponding to the MJO band in the wavenumber-frequency space.
        freq_dim : str
            Name of the frequency dimension (default: 'frequency').
        wavenum_dim : str
            Name of the wavenumber dimension (default: 'wavenumber').

        Returns
        -------
        power_scores : xr.Dataset
            Scalar scores related to power spectra ('ew_ratio', 'ew_ratio_bias', 'eo_ratio', 
            'eo_ratio_bias', 'pwfps' and 'pwfps_bias').
        """

        # Compute ratio and eastward power for observations
        ratio_obs, eastward_obs, _ = mjo_spectrum_funcs.compute_eastward_westward_ratio(self.power_spectra['sym_spec'].sel(dataset='obs'), 
                                                                                        freq_bounds, wavenum_bounds, freq_dim, wavenum_dim)
        if eastward_obs == 0:
            raise ValueError("Eastward power in observations is zero in the selected MJO band, cannot compute eastward/observed power ratio.")

        # Compute ratio and eastward power for simulations
        ew_ratios = [ratio_obs]
        eo_ratios = [1.0]

        n_datasets = self.power_spectra.dataset.size
        for i in range(1, n_datasets):
            ratio_sim, eastward_sim, _ = mjo_spectrum_funcs.compute_eastward_westward_ratio(self.power_spectra['sym_spec'].isel(dataset=i), 
                                                                                            freq_bounds, wavenum_bounds, freq_dim, wavenum_dim)
            ew_ratios.append(ratio_sim)
            eo_ratios.append(eastward_sim/eastward_obs)


        # Compute power-weighted mean period (MJO periodicity) for all datasets
        pwfps = [mjo_spectrum_funcs.compute_power_periodicity(self.power_spectra['sym_spec'].isel(dataset=i), freq_bounds, wavenum_bounds, 
                                                              freq_dim, wavenum_dim) for i in range(n_datasets)]


        # Compute bias for all scores
        ew_ratios_bias = [ratio_obs] + [ratio_sim - ratio_obs for ratio_sim in ew_ratios[1:]]
        eo_ratios_bias = [1.0] + [eo_sim - 1.0 for eo_sim in eo_ratios[1:]]
        pwfps_bias = [pwfps[0]] + [pwfps_sim - pwfps[0] for pwfps_sim in pwfps[1:]]


        # Store all scores in a single dataset
        power_scores = xr.Dataset(
            data_vars = {
                'ew_ratio': (['dataset'], ew_ratios),
                'eo_ratio': (['dataset'], eo_ratios),
                'pwfps': (['dataset'], pwfps),
                'ew_ratio_bias': (['dataset'], ew_ratios_bias),
                'eo_ratio_bias': (['dataset'], eo_ratios_bias),
                'pwfps_bias': (['dataset'], pwfps_bias)
            }, 
            coords={'dataset': self.power_spectra.dataset.values}
        )

        return power_scores


    def save_data(self, output_path):
        """
        Save computed output of CEOF analysis to NetCDF files.

        Parameters
        ----------
        output_path : str
            Path to save the data files.
        """

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare parameters for file names
        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        obs_name_file = self.obs_name.replace(' ', '-')
        sim_name_file = self.sim_name.replace(' ', '-') 


        # Save output of CEOF analyses
        ceof_obs_path = output_path / f"ceof_mjo_{obs_name_file}_{year_range}.nc"
        self.ceof_obs.to_netcdf(ceof_obs_path)
        print(f"Output of CEOF analysis for '{self.obs_name}' saved to '{ceof_obs_path}'.", flush=True)

        ceof_sim_on_obs_path = output_path / f"ceof_mjo_{sim_name_file}_projected_on_{obs_name_file}_{year_range}.nc"
        self.ceof_sim_on_obs.to_netcdf(ceof_sim_on_obs_path)
        print(f"Output of CEOF analysis for '{self.sim_name}' projected on '{self.obs_name}' saved to '{ceof_sim_on_obs_path}'.", flush=True)

        ceof_sim_on_sim_path = output_path / f"ceof_mjo_{sim_name_file}_projected_on_{sim_name_file}_{year_range}.nc"
        self.ceof_sim_on_sim.to_netcdf(ceof_sim_on_sim_path)
        print(f"Output of CEOF analysis for '{self.sim_name}' projected on itself saved to '{ceof_sim_on_sim_path}'.", flush=True)


        # Save CEOF scalar scores
        ceof_scores_path = output_path / f"ceof_scores_{sim_name_file}_{obs_name_file}_{year_range}.nc"
        self.ceof_scores.to_netcdf(ceof_scores_path)
        print(f"Scalar scores related to CEOFs for all datasets saved to '{ceof_scores_path}'.", flush=True)


        # Save total and active MJO days (absolute values and bias)
        activity_per_phase_path = output_path / f"activity_per_phase_{sim_name_file}_{obs_name_file}_{year_range}.nc"
        self.activity_per_phase.to_netcdf(activity_per_phase_path)
        print(f"MJO activity (mean amplitude and days) per phase for all datasets saved to '{activity_per_phase_path}'.", flush=True)


        # Save power spectra and related scalar scores
        power_spectra_path = output_path / f"power_spectra_{self.spectrum_var}_{sim_name_file}_{obs_name_file}_{year_range}.nc"
        self.power_spectra.to_netcdf(power_spectra_path)
        print(f"Power spectra for both observations and simulations saved to '{power_spectra_path}'.", flush=True)

        power_scores_path = output_path / f"power_scores_{self.spectrum_var}_{sim_name_file}_{obs_name_file}_{year_range}.nc"
        self.power_scores.to_netcdf(power_scores_path)
        print(f"Scalar scores related to power spectra for both observations and simulations saved to '{power_scores_path}'.", flush=True)
        
        return


    def ceof_plots(self, output_path=None):
        """
        Generate and save/display plots of CEOFs for simulations and observations together.

        Parameters
        ----------
        output_path : str, optional
            Path to save plots. If None, plots are displayed but not saved.
        """

        # Prepare plotting parameters
        # vars_colors = {
        #     'ua850': '#e41a1c',  # red
        #     'ua200': '#4daf4a',  # green
        #     'rlut':  '#377eb8'   # blue
        # }
        labels_linestyles = {
            self.obs_name: '-',
            self.sim_name: '--'
        }

        # Prepare parameters for titles and file names
        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')
        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"

        # Generate CEOFs plot
        ceof_sim_plot, _ = plot.plot_ceofs([self.ceof_obs['ceof'], self.ceof_sim_on_sim['ceof']], title=f"Combined MJO EOFs ({year_range})",
                                          labels_linestyles=labels_linestyles)
        
        plot.save_or_show_plot(ceof_sim_plot, output_path, plot_filename=f"ceof_{sim_name_file}_{obs_name_file}_projected_on_sim_{year_range}",
                               plot_name=f"Combined MJO CEOFs plot")
        return


    def lead_lag_corr_plot(self, output_path=None): 
        """
        Generate and save/display lead-lag correlation plot between the RMM indices for 
        simulations and observations together.

        Parameters
        ----------
        output_path : str, optional
            Path to save the plot. If None, the plot is displayed but not saved.
        """

        # Prepare parameters for titles and file names
        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')
        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"

        # Prepare data to plot
        lags = self.ceof_scores.lag.values
        lead_lag_corrs = self.ceof_scores['lead_lag_corr']

        # Generate lead-lag correlation plot
        plt.figure(figsize=(10, 6))
        for lead_lag_corr, name in zip([lead_lag_corrs.sel(dataset='obs'), lead_lag_corrs.sel(dataset='sim')], [self.obs_name, self.sim_name]):
            plt.plot(lags, lead_lag_corr, '-', linewidth=2, label=name)

        plt.axhline(0, color='black', linestyle='-', zorder=1)
        plt.axvline(0, color='black', linestyle='-', zorder=1)
        plt.ylim(-1,1)

        plt.xlabel("lag (days)", fontsize=12)
        plt.ylabel("correlation coefficient", fontsize=12)
        plt.title(f"Lead-lag correlation of the RMM indices ({year_range})", fontsize=16, pad=15)
        plt.grid(True, linestyle=':')
        plt.legend(fontsize=12)
        
        # Save/display plot
        plot.save_or_show_plot(plt.gcf(), output_path, plot_filename=f"lead_lag_corr_rmm_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Lead-lag correlation of the RMM indices plot")
        return


    def ceof_corr_table(self, output_path=None):
        """
        Generate and save/display table with CEOF correlation for each dataset together.

        Parameters
        ----------
        output_path : str, optional
            Path to save the table. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        ceof_corr = self.ceof_scores['ceof_corr'].values
        data_ceof_corr = ceof_corr.reshape(ceof_corr.shape[0], -1)
        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        cols_ceof_corr = [f'{self.data_res}° x {self.data_res}°'] + [fr'$r_{{\text{{{var}}}, {mode}}}$' for var in self.ceof_scores['variable'].values 
                                                                     for mode in self.ceof_scores['mode'].values]
        rows_ceof_corr = [self.obs_name, self.sim_name]

        cbar_ticks_corr = ['Negative correlation (-1)', 'No correlation (0)', 'Positive correlation (1)']
        colors_corr = ("RedGreen", ['tab:red', 'white', 'tab:green']) 
        limits_corr = np.repeat([[-1, 1]], len(cols_ceof_corr)-1, axis=0)

        # Generate CEOF scores table
        ceof_scores_table, _ = plot.plot_table(data_ceof_corr, title=f"Correlation between Combined EOFs ({year_range})", col_labels=cols_ceof_corr,
                                               row_labels=rows_ceof_corr, cbar_ticks=cbar_ticks_corr, colors=colors_corr, limits=limits_corr,
                                               decimals=2)
        
        plot.save_or_show_plot(ceof_scores_table, output_path, plot_filename=f"ceof_corr_table_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Correlation in Combined EOFs table plot")

        return


    def ceof_bias_table(self, output_path=None):
        """
        Generate and save/display table plot with the bias scalar scores derived from
        the CEOF analysis.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        expl_var_bias = self.ceof_scores['explained_var_bias'].values
        data_expl_var_bias = expl_var_bias.reshape(expl_var_bias.shape[0], -1)

        data_ceof_bias = np.concatenate([data_expl_var_bias, self.ceof_scores['max_lead_lag_corr_bias'].values.reshape(-1, 1), 
                                         self.ceof_scores['pceof_bias'].values.reshape(-1, 1)], axis=1)

        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        cols_ceof_bias = [
            f'{self.data_res}° x {self.data_res}°',
            *[fr'$b_{{\text{{expl var}}, {mode}}}$ (%)' for mode in self.ceof_scores['mode'].values],
            fr'$b_{{\text{{max lead-lag corr}}}}$ (-)', 
            fr'$b_{{\text{{MJO period}}}}$ (days)'
        ]
        rows_ceof_bias = [self.obs_name, self.sim_name]

        maxs_ceof_bias = np.max(np.abs(data_ceof_bias[1:, :]), axis=0)
        limits_ceof_bias = np.stack([-maxs_ceof_bias, maxs_ceof_bias], axis=1)
        
        # Generate table plot
        ceof_bias_table_plot, _ = plot.plot_table(data_ceof_bias, title=f'Bias derived from Combined EOF analysis ({year_range})', col_labels=cols_ceof_bias, 
                                                  row_labels=rows_ceof_bias, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias, limits=limits_ceof_bias, 
                                                  decimals=2)
        
        plot.save_or_show_plot(ceof_bias_table_plot, output_path, plot_filename=f"ceof_bias_table_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Bias derived from Combined EOF analysis table plot")

        return


    def mean_active_amplitude_plot(self, output_path=None):
        """
        Generate and save/display bar plot with mean MJO amplitude in the active 
        days per phase for each dataset.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the bar plot. If None, the plot is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_mean_active_amp = self.activity_per_phase['mean_active_amplitude'].values
        x_values = self.activity_per_phase.phase.values

        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        labels_mean_amp = [f"{self.obs_name}", f"{self.sim_name} on {self.obs_name}", f"{self.sim_name} on {self.sim_name}"]

        # Generate bar plot
        mean_amp_bar_plot, _ = plot.plot_grouped_bars(data_mean_active_amp, x_values=x_values, title=f'Mean MJO amplitude per phase ({year_range})', 
                                                     x_label='MJO phase', y_label='mean amplitude', labels=labels_mean_amp)
        
        plot.save_or_show_plot(mean_amp_bar_plot, output_path, plot_filename=f"mean_amplitude_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Mean MJO amplitude in the active days per phase bar plot")

        return
    
    
    def active_days_plot(self, output_path=None):
        """
        Generate and save/display bar plot with active MJO days per phase for each
        dataset.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the bar plot. If None, the plot is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_active_days = self.activity_per_phase['active_counts'].values
        x_values = self.activity_per_phase.phase.values

        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        labels_active_days = [f"{self.obs_name}", f"{self.sim_name} on {self.obs_name}", f"{self.sim_name} on {self.sim_name}"]

        # Generate bar plot
        active_days_bar_plot, _ = plot.plot_grouped_bars(data_active_days, x_values=x_values, title=f'Active MJO days per phase ({year_range})', 
                                                     x_label='MJO phase', y_label='number of active days (days)', labels=labels_active_days)
        
        plot.save_or_show_plot(active_days_bar_plot, output_path, plot_filename=f"active_days_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Active MJO days per phase bar plot")

        return


    def activity_per_phase_plots(self, output_path=None, layout='separate'):
        """
        Generate and save/display plot with mean MJO amplitude in the active days 
        and active MJO days per phase for each dataset together.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the plot. If None, the plot is displayed but not saved.
        layout: str
            Whether to plot mean amplitude and active days in 'separate' bar 
            subplots or 'together' in the same dots plot sharing the x-axis 
            (default: 'separate').
        """

        # Prepare data and plotting parameters
        data_mean_active_amp = self.activity_per_phase['mean_active_amplitude'].values
        data_active_days = self.activity_per_phase['active_counts'].values
        
        x_values = self.activity_per_phase.phase.values
        x_values_minor = np.append([0.5], x_values + 0.5) 
        x_label = 'MJO phase'

        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        labels = [f"{self.obs_name}", f"{self.sim_name} on {self.obs_name}", f"{self.sim_name} on {self.sim_name}"]

        # Generate bar plot
        if layout == 'separate':
            mean_amp_active_days_plot, _ = plot.plot_two_grouped_bars(data_mean_active_amp, data_active_days, x1_values=x_values, x2_values=x_values,
                                                                        suptitle=f'MJO activity per phase ({year_range})', title_1='Mean MJO amplitude per phase', 
                                                                        title_2='Active MJO days per phase', x1_label=x_label, x2_label=x_label, 
                                                                        y1_label='mean amplitude', y2_label='number of active days (days)', labels=labels)
        elif layout == 'together':
            mean_amp_active_days_plot, _ = plot.plot_dots_two_axes(data_mean_active_amp, data_active_days, x_values=x_values, x_values_minor=x_values_minor, 
                                                                       title=f'MJO activity per phase ({year_range})', x_label=x_label, y1_label='mean amplitude', 
                                                                       y2_label='number of active days (days)', labels=labels)
        else:
            raise ValueError(f"Invalid layout option '{layout}'. Choose either 'separate' or 'together'.")
        
        plot.save_or_show_plot(mean_amp_active_days_plot, output_path, plot_filename=f"activity_per_phase_{layout}_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"MJO activity (mean MJO amplitude and active days) per phase {layout} plot")

        return


    def mean_amplitude_bias_table(self, output_path=None):
        """
        Generate and save/display table plot with mean MJO amplitude per phase bias.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_mean_amp_bias = self.activity_per_phase['mean_active_amplitude_bias'].values
        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        cols_mean_amp_bias = [f'{self.data_res}° x {self.data_res}°'] + [fr'$\overline{{b}}_{{ph\, {phase}}}$ (-)' for phase in self.activity_per_phase.phase.values]
                               #(['Dataset \ Phase'], [str(phase) for phase in self.activity_per_phase.phase.values])
        rows_mean_amp_bias = [self.obs_name, f"{self.sim_name} on {self.obs_name}", f"{self.sim_name} on {self.sim_name}"]

        maxs_mean_amp_bias = np.max(np.abs(data_mean_amp_bias[1:, :]), axis=0)
        limits_mean_amp_bias = np.stack([-maxs_mean_amp_bias, maxs_mean_amp_bias], axis=1)
        
        # Generate table plot
        mean_amp_bias_table_plot, _ = plot.plot_table(data_mean_amp_bias, title=f'Bias in mean MJO amplitude per phase ({year_range})', col_labels=cols_mean_amp_bias, 
                                                      row_labels=rows_mean_amp_bias, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias, 
                                                      limits=limits_mean_amp_bias, decimals=2)
        
        plot.save_or_show_plot(mean_amp_bias_table_plot, output_path, plot_filename=f"mean_amplitude_bias_table_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Bias in mean MJO amplitude per phase table plot")

        return


    def active_days_bias_table(self, output_path=None):
        """
        Generate and save/display table plot with active MJO days per phase bias.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_active_days_bias = self.activity_per_phase['active_counts_bias'].values
        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        cols_active_days_bias = [f'{self.data_res}° x {self.data_res}°'] + [fr'$\overline{{b}}_{{ph\, {phase}}}$ (days)' for phase in self.activity_per_phase.phase.values]
                                #(['Dataset \ Phase'], [str(phase) for phase in self.activity_per_phase.phase.values])
        rows_active_days_bias = [self.obs_name, f"{self.sim_name} on {self.obs_name}", f"{self.sim_name} on {self.sim_name}"]

        maxs_active_days_bias = np.max(np.abs(data_active_days_bias[1:, :]), axis=0)
        limits_active_days_bias = np.stack([-maxs_active_days_bias, maxs_active_days_bias], axis=1)
        
        # Generate table plot
        active_days_bias_table_plot, _ = plot.plot_table(data_active_days_bias, title=f'Bias in active MJO days per phase ({year_range})', col_labels=cols_active_days_bias, 
                                                         row_labels=rows_active_days_bias, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias, 
                                                         limits=limits_active_days_bias, decimals=0)
        
        plot.save_or_show_plot(active_days_bias_table_plot, output_path, plot_filename=f"active_days_bias_table_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Bias in active MJO days per phase table plot")

        return


    def activity_per_phase_bias_tables(self, output_path=None):
        """
        Generate and save/display table plots with mean MJO amplitude and active MJO
        days per phase bias.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        data_mean_amp_bias = self.activity_per_phase['mean_active_amplitude_bias'].values
        data_active_days_bias = self.activity_per_phase['active_counts_bias'].values

        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')
        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"

        # Column labels
        cols_mean_amp_bias = [f'Bias in Mean amplitude'] + [fr'$\overline{{b}}_{{ph\, {phase}}}$ (-)' for phase in self.activity_per_phase.phase.values]
        cols_active_days_bias = [f'Bias in Active days'] + [fr'$\overline{{b}}_{{ph\, {phase}}}$ (days)' for phase in self.activity_per_phase.phase.values]
        cols = [cols_mean_amp_bias, cols_active_days_bias]

        # Row labels
        rows_mean_amp_bias = [self.obs_name, f"{self.sim_name} on {self.obs_name}", f"{self.sim_name} on {self.sim_name}"]
        rows_active_days_bias = rows_mean_amp_bias
        rows = [rows_mean_amp_bias, rows_active_days_bias]


        # Limits per column
        maxs_mean_amp_bias = np.max(np.abs(data_mean_amp_bias[1:, :]), axis=0)
        limits_mean_amp_bias = np.stack([-maxs_mean_amp_bias, maxs_mean_amp_bias], axis=1)

        maxs_active_days_bias = np.max(np.abs(data_active_days_bias[1:, :]), axis=0)
        limits_active_days_bias = np.stack([-maxs_active_days_bias, maxs_active_days_bias], axis=1)

        limits = [limits_mean_amp_bias, limits_active_days_bias]


        # Generate table plots
        mean_amp_active_days_table_plot, _ = plot.plot_two_tables(data_mean_amp_bias, data_active_days_bias, title=f'Bias in MJO activity per phase ({year_range})',
                                                                  col_labels=cols, row_labels=rows, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias,
                                                                  limits=limits, decimals=[2, 0])

        plot.save_or_show_plot(mean_amp_active_days_table_plot, output_path, plot_filename=f"activity_per_phase_bias_tables_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Bias in MJO activity per phase table plot")
        
        return


    def power_spectrum_plots(self, output_path=None, component='symmetric', x_lim=[-10, 10], y_lim=[0.01, 0.25],
                            levels=[1.1, 1.4, 1.7, 2, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8], mjo_box=True):
        """
        Generate and save/display wavenumber-frequency power spectrum plots for the selected 
        component for both observations and simulations.
        

        Parameters
        ----------
        output_path : str, optional
            Path to save the plot. If None, the plot is displayed but not saved.
        component : str
            Component to plot, either 'symmetric' or 'antisymmetric' (default: 'symmetric').
        x_lim : list[float]
            Limits for the x-axis (default: [-10, 10]).
        y_lim : list[float]
            Limits for the y-axis (default: [0.01, 0.25]).
        levels : list[float]
            Contour levels (default: [1.1, 1.4, 1.7, 2, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8]).
        mjo_box : bool
            Whether to draw a dashed box around the MJO region (default: True).
        """

        # Prepare data and plotting parameters
        if component == 'symmetric':
            name_spec = 'sym_spec'
            name_file = 'sym'
        elif component == 'antisymmetric':
            name_spec = 'asym_spec'
            name_file = 'asym'
        else:
            raise ValueError(f"Invalid component option '{component}'. Choose either 'symmetric' or 'antisymmetric'.")
        
        data_spec_obs = self.power_spectra[name_spec].sel(dataset='obs')
        data_spec_sim = self.power_spectra[name_spec].sel(dataset='sim')
        
        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"


        # Generate power spectrum plot
        power_spectrum_plot, _ = plot.plot_power_spectrum_two(data_spec_obs, data_spec_sim, component=component, x_lim=x_lim, y_lim=y_lim, title_1=self.obs_name, 
                                                              title_2=self.sim_name, suptitle=f"{component.capitalize()} '{self.spectrum_var}' power spectrum ({year_range})", 
                                                              levels=levels, mjo_box=mjo_box, mjo_freq_bounds=self.mjo_freq_bounds, 
                                                              mjo_wavenum_bounds=self.mjo_wavenum_bounds)

        plot.save_or_show_plot(power_spectrum_plot, output_path, plot_filename=f"power_spectrum_{self.spectrum_var}_{name_file}_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"{component.capitalize()} power spectrum plot")

        return


    def power_bias_table(self, output_path=None):
        """
        Generate and save/display table with the biases related to the power spectra
        for both observations and simulations.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table. If None, the table is displayed but not saved.
        """

        # Prepare data and plotting parameters
        power_bias = self.power_scores[['ew_ratio_bias', 'eo_ratio_bias', 'pwfps_bias']].to_array().values
        data_power_bias = power_bias.transpose()
        sim_name_file = self.sim_name.replace(' ', '-')
        obs_name_file = self.obs_name.replace(' ', '-')

        year_range = f"{self.start_year_mjo}-{self.end_year_mjo}"
        cols_power_bias = [
            f'{self.data_res}° x {self.data_res}°', 
            fr'$b_{{\text{{E/W, {self.spectrum_var}}}}}$ (-)', 
            fr'$b_{{\text{{E/O, {self.spectrum_var}}}}}$ (-)', 
            fr'$b_{{\text{{MJO period, {self.spectrum_var}}}}}$ (days)'
        ]
        rows_power_bias = [self.obs_name, self.sim_name]

        maxs_power_bias = np.max(np.abs(data_power_bias[1:, :]), axis=0)
        limits_power_bias = np.stack([-maxs_power_bias, maxs_power_bias], axis=1)

        # Generate power bias table
        power_bias_table, _ = plot.plot_table(data_power_bias, title=f"Bias derived from '{self.spectrum_var}' power spectra ({year_range})", col_labels=cols_power_bias, 
                                              row_labels=rows_power_bias, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias, limits=limits_power_bias, 
                                              decimals=2)
        plot.save_or_show_plot(power_bias_table, output_path, plot_filename=f"power_bias_table_{self.spectrum_var}_{sim_name_file}_{obs_name_file}_{year_range}",
                               plot_name=f"Bias derived from '{self.spectrum_var}' power spectra table plot")
        return
    