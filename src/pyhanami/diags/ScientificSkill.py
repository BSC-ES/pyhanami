import warnings
warnings.simplefilter("always")

from collections.abc import Iterable

from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.scientific_evaluation import general, iso, mjo, tc
   

class ScientificEvaluation:
    """
    Compute and plot scores for scientific model skill evaluation.

    This class provides functionality for computing and visualizing metric to evaluate how well a model 
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

    def __init__(self, datasets=None):        
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
            data_General = self.datasets[0]
            data_name = data_General.name
        elif isinstance(data_name, str):
            data_General = [ds for ds in self.datasets if ds.name == data_name]
            if not data_General:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_General = data_General[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")
        
        # Create GeneralEvaluation object and compute scores
        print(f"Performing general scalar analysis for dataset '{data_name}':", flush=True)
        general_analysis = general.GeneralEvaluation(data_sim=data_General, var_names=var_names, obs_name=obs_name, 
                                                     obs_path=obs_path, start_year=start_year, end_year=end_year)

        return general_analysis


    def compute_iso_scores(self, data_name=None, start_year_eeof=None, end_year_eeof=None, start_year_pc=None, end_year_pc=None, obs=False, 
                           obs_path=None, correct_pc=False, iso_config=None):
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
        obs_path : str
            Path to the observational NOAA data file. As of now, only necessary if the resolution of the 
            NOAA data (2.5°x2.5°) is higher than that of the simulation data.
        correct_pc : bool
            Whether to adjust simulated PCs by dividing by alpha (default: False).
        iso_config : ISOConfig
            Configuration dataclass with parameters necessary for the ISO evaluation. If None, default values 
            from the configuration file `pyhanami.config.scientific_evaluation_parameters.yaml` will be used.

        Returns
        -------
        iso_analysis : ISOEvaluation
            ISOEvaluation object containing the computed bimodal ISO indices and related scalar scores.
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
        iso_analysis = iso.ISOEvaluation(data_sim=data_ISO, start_year_eeof=start_year_eeof, end_year_eeof=end_year_eeof, start_year_pc=start_year_pc,
                                         end_year_pc=end_year_pc, obs=obs, obs_path=obs_path, correct_pc=correct_pc, iso_config=iso_config)

        return iso_analysis
    

    def compute_mjo_scores(self, data_name=None, obs_path=None, start_year_mjo=None, end_year_mjo=None, start_year_ref=None, end_year_ref=None,
                           threshold_active_days=None, mjo_config=None, mjo_vars=['ua850', 'ua200', 'rlut']):
        """
        Initialize and compute Real-Time Multivariate MJO (RMM) indices following (M.C. Wheeler & 
        H.H. Hendon, 2004) and MJO wavenumber-frequency power spectra following (M.C. Wheeler & 
        G.N. Kiladis, 1999) and derived scalar scores following (M.-S. Ahn et al., 2017) for a 
        selected dataset.
        
        Parameters
        ----------
        data_name : str, optional
            Name of simulation ensemble to use. If None, the first dataset in the ScientificEvaluation 
            object is used.
        obs_path : str
            Path to the observational data file with the necessary variables for the MJO analysis.
        start_year_mjo, end_year_mjo : int
            Initial and end years to perform the analysis for.
        start_year_ref, end_year_ref : int
            Initial and end years for computing the reference seasonal cycle. If None, taken as the
            initial and end years for the whole MJO analysis.
        threshold_active_days : float
            Threshold for the amplitude of the first two PCs to consider the MJO active at a given 
            day. If None, the mean MJO amplitude over the entire period is used as a threshold.
        mjo_config : MJOConfig
            Configuration dataclass with parameters necessary for the MJO evaluation. If None, default values 
            from the configuration file `pyhanami.config.scientific_evaluation_parameters.yaml` will be used.
        mjo_vars : list[str]
            Variables to be usd for the MJO analysis (default: ['ua850', 'ua200', 'rlut']).

        Returns
        -------
        mjo_analysis : MJOEvaluation
            MJOEvaluation object containing the computed RMM MJO indices, power spectra and scalar scores.
        """

        # Validate input
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError("At least one dataset is required for the MJO evaluation.")
            data_MJO = self.datasets[0]
            data_name = data_MJO.name
        elif isinstance(data_name, str):
            data_MJO = [ds for ds in self.datasets if ds.name == data_name]
            if not data_MJO:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_MJO = data_MJO[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")
        
        # Create MJOEvaluation object and compute scores
        print(f"Performing MJO analysis for dataset '{data_name}':", flush=True)
        mjo_analysis = mjo.MJOEvaluation(data_sim=data_MJO, obs_path=obs_path, start_year_mjo=start_year_mjo, end_year_mjo=end_year_mjo,
                                         start_year_ref=start_year_ref, end_year_ref=end_year_ref, threshold_active_days=threshold_active_days,
                                         mjo_config=mjo_config, mjo_vars=mjo_vars)

        return mjo_analysis
    

    def compute_tc_scores(self, data_name=None, start_year_tc=None, end_year_tc=None, obs=True, wind_factor=1.0, min_wind=10, 
                           bin_size=2.5, tc_config=None):
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
        tc_config : TCConfig
            Configuration dataclass with parameters necessary for the TC evaluation. If None, default values 
            from the configuration file `pyhanami.config.scientific_evaluation_parameters.yaml` will be used.

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
        tc_analysis = tc.TCEvaluation(data_sim=data_TC, start_year_tc=start_year_tc, end_year_tc=end_year_tc, obs=obs,
                                      wind_factor=wind_factor, min_wind=min_wind, bin_size=bin_size, tc_config=tc_config)

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