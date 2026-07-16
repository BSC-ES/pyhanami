import sys

from collections.abc import Iterable

from pyhanami.utils import data_general
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.scientific_evaluation import general, iso, mjo, tc, wrapper


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
    _general_scores : dict[str, general.GeneralEvaluation]
        Dictionary to store general evaluation output for each dataset.
    _iso_scores : dict[str, iso.ISOEvaluation]
        Dictionary to store ISO evaluation output for each dataset.
    _mjo_scores : dict[str, mjo.MJOEvaluation]
        Dictionary to store MJO evaluation output for each dataset.
    _tc_scores : dict[str, tc.TCEvaluation]
        Dictionary to store TCs evaluation output for each dataset.
    """

    def __init__(self, datasets=None):
        if datasets is None:
            self.datasets = []
        else:
            if isinstance(datasets, SimulationData):
                self.datasets = [datasets]
            elif (
                isinstance(datasets, Iterable)
                and not isinstance(datasets, (str, bytes))
                and all(isinstance(ds, SimulationData) for ds in datasets)
            ):
                self.datasets = list(datasets)
            else:
                raise TypeError(
                    "Input must be a SimulationData object or an iterable of SimulationData objects."
                )

        # Create placeholders to store analyses outputs as: {dataset_name: evaluation_instance}
        self._general_scores = {}
        self._iso_scores = {}
        self._mjo_scores = {}
        self._tc_scores = {}

        return


    def _validate_input_compute_methods(self, data_names, evaluation_type):
        """
        Validate input for the methods that compute scores.

        Parameters
        ----------
        data_name : str
            Name/s of the dataset/s to compute scores for.
        evaluation_type : str
            Type of evaluation.

        Returns
        -------
        data_evaluation : list[SimulationData]
            Validated dataset/s.
        data_names_validated : list[str]
            Name/s of the validated dataset/s.
        """

        # Use all loaded datasets if no names are provided
        if data_names is None:
            if len(self.datasets) < 1:
                raise ValueError(f"At least one dataset is required for the {evaluation_type} evaluation.")
            data_evaluation = self.datasets
            data_names_validated = [ds.name for ds in data_evaluation]

        # Look for datasets in the class instance by name
        elif isinstance(data_names, str) or (
            isinstance(data_names, list) and all(isinstance(name, str) for name in data_names)
        ):
            if isinstance(data_names, str):
                data_names = [data_names]

            evaluation_data = getattr(self, f"_{evaluation_type}_scores")
            datasets_dict = {ds.name: ds for ds in self.datasets}
            data_evaluation = []
            data_names_validated = []

            for name in data_names:
                if name not in datasets_dict:
                    data_general.warn_always(
                        f"Dataset with name '{name}' not found in the ScientificEvaluation object. "
                        f"Skipping this dataset. Available datasets: {list(datasets_dict.keys())}\n"
                    )
                    continue

                # Check whether the scores have already been computed for the given dataset and evaluation type
                if name in evaluation_data:
                    data_general.warn_always(
                        f"{evaluation_type} scores for dataset '{name}' have already been computed."
                    )

                    # Check if running interactively
                    if sys.stdin.isatty():
                        try:
                            # Ask user for confirmation
                            response = input("Do you want to overwrite the existing scores? (y/n):").strip().lower()
                        except EOFError:
                            print(f"\n{evaluation_type} scores computation cancelled for dataset '{name}'.\n")
                            continue

                        if response not in ['y', 'yes']:
                            print(
                                "Skipping scores computation for this dataset. You can access the existing scores "
                                f"using the '{evaluation_type}_scores' method.\n"
                            )
                            continue

                    # Non-interactive mode: auto-overwrite warning
                    else:
                        data_general.warn_always(
                            "Non-interactive mode detected. Existing scores will be overwritten automatically."
                        )

                    print(f"Overwriting the existing {evaluation_type} scores for dataset '{name}'.\n")

                data_evaluation.append(datasets_dict[name])
                data_names_validated.append(name)

            if not data_evaluation:
                raise ValueError(f"No valid datasets selected for {evaluation_type} analysis.")

        # Issue with the input type
        else:
            raise TypeError("'data_names' must be a string or a list of strings representing dataset names.")

        return data_evaluation, data_names_validated


    def _validate_input_access_methods(self, data_names, evaluation_type):
        """
        Validate input for the methods that access the evaluation outputs.

        Parameters
        ----------
        data_names : str or list[str]
            Name/s of the dataset/s to access the evaluation output for.
        evaluation_type : str
            Type of evaluation to handle.

        Returns
        -------
        selected_evaluation_outputs : list[x.XEvaluation]
            List of evaluation outputs for the validated dataset/s.
        """

        # Validate dataset names format
        if data_names is None:
            # Use all names if no names are provided
            data_names = [ds.name for ds in self.datasets]
        elif isinstance(data_names, str):
            data_names = [data_names]
        elif not isinstance(data_names, list) or not all(isinstance(name, str) for name in data_names):
            raise TypeError("'data_names' must be a string or a list of dataset name strings.")


        # Check availability of the requested evaluation for the given dataset/s
        evaluation_outputs = getattr(self, f"_{evaluation_type}_scores")
        selected_evaluation_outputs = []

        for name in data_names:
            if name not in evaluation_outputs:
                data_general.warn_always(
                    f"{evaluation_type} scores for dataset '{name}' not found. Skipping this dataset.\n"
                    f"Please compute the {evaluation_type} scores for this dataset first using the "
                    f"'compute_{evaluation_type}_scores' method if you want to access them.\n"
                )
            else:
                selected_evaluation_outputs.append(evaluation_outputs[name])

        if not selected_evaluation_outputs:
            raise ValueError(f"No {evaluation_type} scores are available for the provided dataset/s.")

        return selected_evaluation_outputs


    def _access_scores(self, data_names, evaluation_type):
        """
        Access evaluation output and its corresponding methods for the given dataset/s and
        evaluation type.

        Parameters
        ----------
        data_names : str or list[str]
            Name/s of the dataset/s to access the evaluation output for.
        evaluation_type : str
            Type of evaluation to access.
        """

        # Validate input
        evaluation_instances= self._validate_input_access_methods(data_names, evaluation_type)

        # Create instance of wrapper class with all evaluations to be accessed
        evaluation_handlers = wrapper.ScientificEvaluationWrapper(evaluation_instances, evaluation_type)

        return evaluation_handlers


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
        elif (
            not isinstance(datasets, Iterable)
            or isinstance(datasets, (str, bytes))
            or not all(isinstance(ds, SimulationData) for ds in datasets)
        ):
            raise TypeError(
                "Input must be a SimulationData object or an iterable of SimulationData objects."
            )

        # Check for duplicate datasets
        for dataset in datasets:
            if not any(ds.name == dataset.name for ds in self.datasets):
                self.datasets.append(dataset)
            else:
                data_general.warn_always(
                    f"Dataset with name '{dataset.name}' already exists in the ScientificEvaluation object. "
                    "Skipping addition."
                )

        return


    def compute_general_scores(self, var_names=None, data_names=None, obs_name=None, obs_path=None,
                               start_year=None, end_year=None):
                                #config_params.GEN_OBS_NAME, obs_path=config_params.GEN_OBS_PATH,
        """
        Initialize and compute general model skill evaluation scores for selected dataset/s.

        Parameters
        ----------
        var_names : str or list[str], optional
            Climate variable/s name/s. If None, all variables in the simulated dataset will be used.
        data_names : str or list[str], optional
            Name/s of simulation ensemble/s to use. If None, all datasets in the ScientificEvaluation
            object are used.
        obs_name : str
            Name of the observational dataset to compare to (default: config_params.GEN_OBS_NAME).
        obs_path : str
            Path to the observations database (default: config_params.GEN_OBS_PATH).
        start_year, end_year : int
            Initial and end years to compute the general scores for.
        """

        # Validate input
        evaluation_type = "general"
        data_General, data_names = self._validate_input_compute_methods(data_names, evaluation_type)

        # Create GeneralEvaluation object and compute scores for each dataset
        for data, name in zip(data_General, data_names):
            print(f"Performing general scalar analysis for dataset '{name}':", flush=True)
            general_evaluation = general.GeneralEvaluation(
                data_sim=data,
                var_names=var_names,
                obs_name=obs_name,
                obs_path=obs_path,
                start_year=start_year,
                end_year=end_year,
            )

            # Store analysis output in object attribute
            self._general_scores[name] = general_evaluation

        return


    def compute_iso_scores(self, data_names=None, start_year_eeof=None, end_year_eeof=None,
                           start_year_pc=None, end_year_pc=None, obs=False, obs_path=None,
                           correct_pc=False, iso_config=None):
        """
        Initialize and compute bimodal ISO indices (following (K. Kikuchi, 2020)) and derive scalar
        scores (following (M. Nakano et al., 2019)) for selected datasets.

        Parameters
        ----------
        data_names : str or list[str], optional
            Name/s of simulation ensemble/s to use. If None, all datasets in the ScientificEvaluation
            object are used.
        start_year_eeof, end_year_eeof : int
            Initial and end years to perform the Extended Empirical Orthogonal Function (EEOF) analysis
            for (not needed if `obs=True`).
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
        """

        # Validate input
        evaluation_type = "iso"
        data_ISO, data_names = self._validate_input_compute_methods(data_names, evaluation_type)

        # Create ISO object and compute scores for each dataset
        for data, name in zip(data_ISO, data_names):
            print(f"Performing ISO analysis for dataset '{name}':", flush=True)
            iso_evaluation = iso.ISOEvaluation(
                data_sim=data,
                start_year_eeof=start_year_eeof,
                end_year_eeof=end_year_eeof,
                start_year_pc=start_year_pc,
                end_year_pc=end_year_pc,
                obs=obs,
                obs_path=obs_path,
                correct_pc=correct_pc,
                iso_config=iso_config,
            )

            # Store analysis output in object attribute
            self._iso_scores[name] = iso_evaluation

        return


    def compute_mjo_scores(self, data_names=None, obs_path=None, start_year_mjo=None, end_year_mjo=None,
                           start_year_ref=None, end_year_ref=None, threshold_active_days=None,
                           mjo_config=None, mjo_vars=['ua850', 'ua200', 'rlut']):
        """
        Initialize and compute Real-Time Multivariate MJO (RMM) indices following (M.C. Wheeler &
        H.H. Hendon, 2004) and MJO wavenumber-frequency power spectra following (M.C. Wheeler &
        G.N. Kiladis, 1999) and derived scalar scores following (M.-S. Ahn et al., 2017) for a
        selected dataset.

        Parameters
        ----------
        data_names : str or list[str], optional
            Name/s of simulation ensemble/s to use. If None, all datasets in the ScientificEvaluation
            object are used.
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
        evaluation_type = "mjo"
        data_MJO, data_names = self._validate_input_compute_methods(data_names, evaluation_type)

        # Create MJO object and compute scores for each dataset
        for data, name in zip(data_MJO, data_names):
            print(f"Performing MJO analysis for dataset '{name}':", flush=True)
            mjo_analysis = mjo.MJOEvaluation(
                data_sim=data,
                obs_path=obs_path,
                start_year_mjo=start_year_mjo,
                end_year_mjo=end_year_mjo,
                start_year_ref=start_year_ref,
                end_year_ref=end_year_ref,
                threshold_active_days=threshold_active_days,
                mjo_config=mjo_config,
                mjo_vars=mjo_vars,
            )

            # Store analysis output in object attribute
            self._mjo_scores[name] = mjo_analysis

        return


    def compute_tc_scores(self, data_names=None, start_year_tc=None, end_year_tc=None, obs=True,
                          wind_factor=1.0, min_wind=10, basin=-1, bin_size=2.5, tc_config=None):
        """
        Compute Tropical Cyclones (TCs) metrics and derive scalar scores following (C.M. Zarzycki et al., 2021)
        and plot results.

        Parameters
        ----------
        data_names : str or list[str], optional
            Name/s of simulation ensemble/s to use. If None, all datasets in the ScientificEvaluation
            object are used.
        start_year_tc, end_year_tc : int, optional
            Initial and end years to compute the TCs metrics for.
        obs : bool
            If True, include observational data if available (default: True).
        wind_factor : float
            Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
        min_wind : float
            Minimum 10 m wind speed in m/s for TCs detection (default: 10.0).
        basin : int
            Basin/hemisphere to consider for the analysis (default: -1). Codes are:
                - <0 → GLOB (Global domain)
                - 1  → NATL (North Atlantic)
                - 2  → EPAC (Eastern Pacific)
                - 3  → CPAC (Central Pacific)
                - 4  → WPAC (Western Pacific)
                - 5  → NIO (North Indian Ocean)
                - 6  → SIO (South Indian Ocean)
                - 7  → SPAC (South Pacific)
                - 8  → SATL (South Atlantic)
                - 9  → FLA (Florida)
                - 20 → NHEMI (Northern Hemisphere)
                - 21 → SHEMI (Southern Hemisphere)
                - otherwise → NONE (unrecognized)
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
        evaluation_type = "tc"
        data_TC, data_names = self._validate_input_compute_methods(data_names, evaluation_type)

        # Create TC object and compute scores for each dataset
        for data, name in zip(data_TC, data_names):
            print(f"Performing TCs analysis for dataset '{name}':", flush=True)
            tc_analysis = tc.TCEvaluation(
                data_sim=data,
                start_year_tc=start_year_tc,
                end_year_tc=end_year_tc,
                obs=obs,
                wind_factor=wind_factor,
                min_wind=min_wind,
                basin=basin,
                bin_size=bin_size,
                tc_config=tc_config,
            )

            # Store analysis output in object attribute
            self._tc_scores[name] = tc_analysis

        return


    def general_scores(self, data_names=None):
        """
        Access general evaluation output and its corresponding methods for the given dataset/s.

        Parameters
        ----------
        data_names : str or list[str]
            Name/s of the dataset/s to access the general evaluation output for.

        Returns
        -------
        general_handlers : ScientificEvaluationWrapper
            Instance of the ScientificEvaluationWrapper class containing the general evaluation
            output for the given dataset/s and providing access to its corresponding methods.
        """

        evaluation_type = "general"
        general_handlers = self._access_scores(data_names, evaluation_type)

        return general_handlers


    def iso_scores(self, data_names=None):
        """
        Access ISO evaluation output and its corresponding methods for the given dataset/s.

        Parameters
        ----------
        data_names : str or list[str]
            Name/s of the dataset/s to access the ISO evaluation output for.

        Returns
        -------
        iso_handlers : ScientificEvaluationWrapper
            Instance of the ScientificEvaluationWrapper class containing the ISO evaluation
            output for the given dataset/s and providing access to its corresponding methods.
        """

        evaluation_type = "iso"
        iso_handlers = self._access_scores(data_names, evaluation_type)

        return iso_handlers


    def mjo_scores(self, data_names=None):
        """
        Access MJO evaluation output and its corresponding methods for the given dataset/s.

        Parameters
        ----------
        data_names : str or list[str]
            Name/s of the dataset/s to access the MJO evaluation output for.

        Returns
        -------
        mjo_handlers : ScientificEvaluationWrapper
            Instance of the ScientificEvaluationWrapper class containing the MJO evaluation
            output for the given dataset/s and providing access to its corresponding methods.
        """

        evaluation_type = "mjo"
        mjo_handlers = self._access_scores(data_names, evaluation_type)

        return mjo_handlers


    def tc_scores(self, data_names=None):
        """
        Access TC evaluation output and its corresponding methods for the given dataset/s.

        Parameters
        ----------
        data_names : str or list[str]
            Name/s of the dataset/s to access the TC evaluation output for.

        Returns
        -------
        tc_handlers : ScientificEvaluationWrapper
            Instance of the ScientificEvaluationWrapper class containing the TC evaluation
            output for the given dataset/s and providing access to its corresponding methods.
        """

        evaluation_type = "tc"
        tc_handlers = self._access_scores(data_names, evaluation_type)

        return tc_handlers
