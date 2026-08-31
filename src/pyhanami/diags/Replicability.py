import sys
import copy

import numpy as np
import xarray as xr
import concurrent.futures

from pathlib import Path
from scipy.stats import bootstrap
from statsmodels.stats.power import TTestIndPower

from pyhanami.config import config_params
from pyhanami.utils.plots import plots_general
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData
from pyhanami.utils import data_general, report, statistics


class ReplicabilityTest:
    """
    Perform replicability test between two climate simulation ensembles.

    This class compares two climate simulation ensembles using a variety of metrics and
    statistical tests to assess whether both climates are statistically significantly
    different. The test is conducted over multiple variables, regions, seasons, and
    ensemble members. It also supports plotting results and generating summary reports.

    Parameters
    ----------
    datasets : Iterable[SimulationData], optional
        Ensemble or list of ensembles containing simulation data and metadata.
    alpha : float
        Significance level for the statistical tests (default: 0.05).
    power : float
        Statistical power to compute minimum detectable effect size for the t-test
        (default: 0.8).

    Attributes
    ----------
    datasets : list[SimulationData]
        List of ensembles containing simulation data and metadata.
    variables : dict
        Configuration dictionary mapping variable names to display metadata.
    alpha : float
        Significance level for the statistical tests.
    power : float
        Statistical power to compute minimum detectable effect size for the t-test.
    max_workers_grid : int
        Number of parallel workers used for variable-wise computations.
    metrics : list of dict
        List of metrics with names and corresponding functions to compute scores.
    tests : dict
        Dictionary of statistical tests for comparing score distributions.
    seasons : list of str
        List of seasons to compute scores over.
    regions : dict
        Dictionary mapping region names to latitude bounds.
    effect_sizes : dict[tuple[str, str, int, int], xr.DataArray]
        Dictionary to store effect sizes between the replicability test scores for
        each pair of datasets for all variables, seasons, regions, and metrics.
    p_values : dict[tuple[str, str, int, int], xr.DataArray]
        Dictionary to store p-values obtained during the replicability test for each pair
        of datasets for all variables, seasons, regions, metrics, and statistical tests.
    test_results : dict[tuple[str, str, int, int], xr.DataArray]
        Dictionary to store results of the replicability test for each pair of datasets
        for all variables, seasons, regions, metrics, and statistical tests.
    """

    def __init__(self, datasets=None, alpha=0.05, power=0.8):

        # Validate input dataset/s and initialize attributes
        if datasets is None:
            self.datasets = []
            self.variables = None
        else:
            if isinstance(datasets, SimulationData):
                self.datasets = [datasets]
            elif (isinstance(datasets, list) and all(isinstance(ds, SimulationData) for ds in datasets)):
                self.datasets = datasets
            else:
                raise TypeError("Input must be a SimulationData object or a list of SimulationData objects.")

            # Check that all datasets have the same variables and lat-lon coordinates
            self._compare_ensembles()

            # Load metadata for the variables in the datasets from the configuration file
            expected_vars = data_general.load_yaml_file(config_params.VARIABLES_PATH)
            self.variables = {
                var: info
                for var, info in expected_vars.items()
                if var in self.datasets[0].data.data_vars
            }

        # Validate significance level and statistical power
        if not isinstance(alpha, (int, float)) or not isinstance(power, (int, float)):
            raise TypeError("The significance level 'alpha' and statistical power 'power' must be numeric.")
        if not (0 <= alpha <= 1) or not (0 <= power <= 1):
            raise ValueError("'alpha' and 'power' must be between 0 and 1.")
        self.alpha = alpha
        self.power = power

        # Load config parameters once
        self.max_workers_vars = config_params.MAX_WORKERS_VARS
        self.metrics = config_params.METRICS
        self.tests = config_params.TESTS
        self.seasons = config_params.SEASONS
        self.regions = config_params.REGIONS

        # Create placeholders for effect sizes and replicability test results
        self.effect_sizes = {}
        self.p_values = {}
        self.test_results = {}

        return


    def _compare_ensembles(self):
        """Check that all provided ensembles are equivalent, i.e. same
        variables and lat-lon coordinates."""

        if len(self.datasets) < 2:
            return

        # Get reference variables and coordinates
        ref = self.datasets[0]
        ref_vars = set(ref.data.data_vars)
        ref_lat = ref.data.coords["lat"]
        ref_lon = ref.data.coords["lon"]

        # Compare with all other datasets
        for dataset in self.datasets[1:]:
            dataset_vars = set(dataset.data.data_vars)
            if ref_vars != dataset_vars:
                raise ValueError(f"Ensembles '{ref.name}' and '{dataset.name}' have different variables.")

            dataset_lat = dataset.data.coords["lat"]
            if not np.array_equal(ref_lat, dataset_lat):
                raise ValueError(f"Ensembles '{ref.name}' and '{dataset.name}' have different latitude coordinates.")

            dataset_lon = dataset.data.coords["lon"]
            if not np.array_equal(ref_lon, dataset_lon):
                raise ValueError(f"Ensembles '{ref.name}' and '{dataset.name}' have different longitude coordinates.")

        return


    def _compute_scores_one_var(self, args):
        """
        Compute scores for the given variable in both simulation ensembles.

        Parameters
        ----------
        args : tuple
            List containing:
                var_name (str): Climate variable name.
                data_names (list[str]): List of two simulation ensemble names to compare.
                data_plot (list[xr.Dataset]): List of two simulation ensembles to compare.
                data_obs (xr.Dataset): Observational dataset for comparison.

        Returns
        -------
        scores_dataset : tuple[str, xr.Dataset]
            Variable name and dataset containing computed scores.
        """
        var_name, data_names, data_plot, data_obs = args

        # Prepare datasets and check matching time coordinates
        datasets = [data_plot[0].persist(), data_plot[1].persist()]
        data_obs = data_obs.resample(time="1MS").sum().persist()

        if not datasets[0].time.equals(datasets[1].time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n  "
                f"{data_names[0]} has time from {datasets[0].time.min().item()} "
                f"to {datasets[0].time.max().item()}\n  "
                f"{data_names[1]} has time from {datasets[1].time.min().item()} "
                f"to {datasets[1].time.max().item()}"
            )


        # Initialize scores dictionary
        length_seasons = len(self.seasons)
        length_regions = len(self.regions)
        length_realizations = datasets[0].sizes["realization"]
        scores_dict = {
            metric["name"]: np.zeros((2, length_seasons, length_regions, length_realizations))
            for metric in self.metrics
        }

        # Process each dataset
        for dataset_idx, data_sim in enumerate(datasets):
            lat = data_sim["lat"]

            # Process each metric
            for metric_idx, metric in enumerate(self.metrics):
                metric_label = metric["name"]
                metric_funcs = metric["functions"]
                obs_needed = metric["obs_needed"]

                # Compute scores with annual and seasonal climatology
                for season_idx, season in enumerate(self.seasons):
                    if season_idx == 0:
                        data_sim_season = data_sim
                        data_obs_season = data_obs if obs_needed else None
                    else:
                        data_sim_season = data_sim.groupby("time.season")[season]
                        data_obs_season = data_obs.groupby("time.season")[season] if obs_needed else None
                    if metric_idx < 2:
                        data_sim_season = data_sim_season.mean(dim="time")

                    # Global, tropical and extratropical regions
                    for region_idx, region in enumerate(self.regions.values()):
                        mask = region(lat)
                        data_sim_region = data_sim_season.where(mask, drop=True)
                        data_obs_region = data_obs_season.where(mask, drop=True) if obs_needed else None

                        scores_region = 0
                        for metric_func in metric_funcs:
                            if metric_func.__code__.co_argcount == 3:
                                scores = metric_func(data_sim_region, data_obs_region, var_name)
                            else:
                                scores = metric_func(data_sim_region, var_name)
                            scores_region += scores
                        scores_dict[metric_label][dataset_idx, season_idx, region_idx,:] = scores_region
                        del mask


        # Add combined metric
        combined_scores = np.mean(
            np.stack([scores_dict[metric["name"]] for metric in self.metrics], axis=0),
            axis=0,
        )
        scores_dict["Combined"] = combined_scores

        # Create xarray.Dataset with scores for all metrics
        coords = {
            "dataset": [data_names[0], data_names[1]],
            "season": self.seasons,
            "region": list(self.regions.keys()),
            "realization": np.arange(length_realizations),
        }

        scores_var = {}
        for metric_name, scores in scores_dict.items():
            scores_var[metric_name] = (["dataset", "season", "region", "realization"], scores)

        scores_dataset = xr.Dataset(data_vars=scores_var, coords=coords)
        scores_dataset.attrs["variable"] = var_name
        scores_dataset.attrs["long_name"] = self.variables[var_name]["long_name"]

        print(f"\tComputed scores for variable '{var_name}'...", flush=True)
        return scores_dataset


    def _compute_scores(self, data_plot, data_obs):
        """
        Compute scores for all variables in both simulation ensembles in parallel.

        Parameters
        ----------
        data_plot : list[SimulationData])
            List of two simulation ensembles to compare.
        data_obs : ObservationData
            Observational dataset for comparison.

        Returns
        -------
        scores_all : dict[str, xr.Dataset]
            Dictionary of scores datasets for each variable.
        """

        # Validate inputs
        if (not isinstance(data_plot, list) or len(data_plot) == 0
            or not all(isinstance(ds, SimulationData) for ds in data_plot)):
            raise TypeError("'data_plot' must be a non-empty list of SimulationData instances.")

        # Check that all variables are present in both datasets
        var_names = list(self.variables.keys())
        for dataset in data_plot:
            for var_name in var_names:
                if var_name not in dataset.data.data_vars:
                    raise ValueError(
                        f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                        f"Available variables: {list(dataset.data.data_vars.keys())}"
                    )


        # Compute scores for each variable in parallel
        scores_all = {}
        data_names = [data_plot[0].name, data_plot[1].name]
        tasks = [
            (var_name, data_names, [data_plot[0].data[[var_name]], data_plot[1].data[[var_name]]], data_obs.data[[var_name]])
            for var_name in var_names
        ]
        # for task in tasks:
        #     scores_all[task[0][1]] = self._compute_scores_one_var(task[0], task[1])
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_vars) as executor:
            for idx, value in enumerate(executor.map(self._compute_scores_one_var, tasks)):
                scores_all[tasks[idx][0]] = value

        print("Computed scores for all variables...", flush=True)
        return scores_all


    def _validate_scores_and_data_names(self, scores, data_names):
        """
        Validate the format of 'scores' and 'data_names' and check the presence
        of all datasets in the 'scores' dictionary.

        Parameters
        ----------
        scores : dict[str, xr.Dataset]
            Dictionary of scores datasets for each variable.
        data_names : list[str]
            List of two simulation ensemble names to compare.
        """

        # Check format of 'scores'
        if (not isinstance(scores, dict)
            or not all(isinstance(key, str) and isinstance(value, xr.Dataset) for key, value in scores.items())):
            raise TypeError("'scores' must be a dictionary with variable names as keys and xr.Dataset as values.")

        # Check format of 'data_names'
        if (not isinstance(data_names, list) or len(data_names) != 2
            or not all(isinstance(name, str) for name in data_names)):
            raise TypeError("'data_names' must be a list of two strings representing simulation dataset names.")


        # Check that all datasets in 'data_names' are present in 'scores'
        for var_name, scores_var in scores.items():
            for name in data_names:
                if name not in scores_var.coords["dataset"].values:
                    raise ValueError(
                        f"Dataset '{name}' not found in scores for variable '{var_name}'. "
                        f"Available datasets: {scores_var.coords['dataset'].values}"
                    )

        return


    def _compute_eff_sizes(self, scores_all, data_names):
        """
        Compute effect size (Cohen's d) between the pre-computed scores separating
        by season and region, for all available variables.

        Parameters
        ----------
        scores_all : dict[str, xr.Dataset])
            Dictionary of scores datasets for each variable.
        data_names : list[str])
            List of two simulation ensemble names to compare.

        Returns
        -------
        effect_sizes : xr.DataArray
            Effect sizes for all variables, seasons, regions, and metrics.
        """

        # Validate inputs
        self._validate_scores_and_data_names(scores_all, data_names)


        # Initialize array
        names_metrics = np.append(self.metrics["name"], "Combined")
        eff_sizes_array = np.empty((
            len(self.variables),
            len(self.seasons),
            len(self.regions),
            len(self.metrics) + 1  # +1 for combined metric
        ))

        # Loop over all scores sets
        for var_idx, var in enumerate(self.variables):
            scores_var = scores_all[var]

            for metric_idx, metric_name in enumerate(names_metrics):
                for season_idx, season in enumerate(self.seasons):
                    for region_idx, region in enumerate(list(self.regions.keys())):
                        scores = scores_var[metric_name].sel(season=season, region=region)

                        scores_ref = scores.sel(dataset=data_names[0]).compute().values
                        scores_test = scores.sel(dataset=data_names[1]).compute().values

                        # Compute effect size with bootstrapping
                        bootstrap_res = bootstrap(
                            (scores_ref, scores_test),
                            statistics.cp_effect_size,
                            confidence_level=0.95,
                            n_resamples=10000,
                        )
                        eff_sizes_array[var_idx, season_idx, region_idx, metric_idx] = np.mean(
                            bootstrap_res.bootstrap_distribution
                        )


        # Save all effect sizes as a xr.DataArray
        effect_sizes = xr.DataArray(
            data=eff_sizes_array,
            dims=["variable", "season", "region", "metric"],
            coords={
                "variable": list(self.variables.keys()),
                "season": self.seasons,
                "region": list(self.regions.keys()),
                "metric": names_metrics,
            },
            attrs={"datasets": " - ".join(data_names)},
            name="effect_size",
        )

        print("Computed effect sizes between scores distributions for all variables...", flush=True)
        return effect_sizes


    def _apply_tests(self, scores_all, data_names, datasets_key):
        """
        Compare scores with statistical tests separating by season, region and metric, for all 
        available variables.

        Parameters
        ----------
        scores_all : dict[str, xr.Dataset])
            Dictionary of scores datasets for each variable.
        data_names : list[str]
            List of two simulation ensemble names to compare.
        datasets_key : str
            Key representing the combination of datasets being compared and their time period.

        Returns
        -------
        p_values : xr.DataArray
            Raw p-values for all variables, seasons, regions, metrics, and statistical tests.
        test_results : xr.DataArray
            Test results for all variables, seasons, regions, metrics, and statistical tests.
        """

        # Validate inputs
        self._validate_scores_and_data_names(scores_all, data_names)


        # Initialize array
        names_metrics = np.append(self.metrics["name"], "Combined")
        p_values = np.full(
            (
                len(self.variables),
                len(self.seasons),
                len(self.regions),
                len(self.metrics) + 1,  # +1 for combined metric
                len(self.tests)
            ),
            np.nan,
            dtype=float,
        )
        test_results = np.zeros(
            (
                len(self.variables),
                len(self.seasons),
                len(self.regions),
                len(self.metrics) + 1,  # +1 for combined metric
                len(self.tests)
            ),
            dtype=bool
        )

        # Loop over all scores sets
        power_analysis = TTestIndPower()
        for var_idx, var in enumerate(self.variables):
            scores_var = scores_all[var]

            for metric_idx, metric_name in enumerate(names_metrics):
                for season_idx, season in enumerate(self.seasons):
                    for region_idx, region in enumerate(list(self.regions.keys())):
                        scores = scores_var[metric_name].sel(season=season, region=region)

                        scores_ref = scores.sel(dataset=data_names[0]).compute().values
                        scores_test = scores.sel(dataset=data_names[1]).compute().values

                        # Check that the effect size is not too small to apply the statistical tests
                        effect_size = self.effect_sizes[datasets_key][
                            var_idx, season_idx, region_idx, metric_idx
                        ]
                        min_detectable_effect_size = power_analysis.solve_power(
                            effect_size=None,
                            nobs1=len(scores_ref),
                            alpha=self.alpha,
                            power=self.power,
                        )

                        if np.abs(effect_size) < min_detectable_effect_size:
                            test_results[var_idx, season_idx, region_idx, metric_idx, :] = True
                            continue

                        # Apply statistical tests
                        for test_idx, test_name in enumerate(self.tests):
                            p_value = self.tests[test_name](scores_ref, scores_test)
                            p_values[var_idx, season_idx, region_idx, metric_idx, test_idx] = p_value

                            test_results[var_idx, season_idx, region_idx, metric_idx, test_idx] = (
                                p_value <= self.alpha
                            )


        # Save all p-values and test results as xr.DataArray
        p_values = xr.DataArray(
            data=p_values,
            dims=["variable", "season", "region", "metric", "test"],
            coords={
                "variable": list(self.variables.keys()),
                "season": self.seasons,
                "region": list(self.regions.keys()),
                "metric": names_metrics,
                "test": list(self.tests.keys())
            },
            attrs={"datasets": " - ".join(data_names)},
            name="p_value",
        )

        test_results = xr.DataArray(
            data=test_results,
            dims=["variable", "season", "region", "metric", "test"],
            coords={
                "variable": list(self.variables.keys()),
                "season": self.seasons,
                "region": list(self.regions.keys()),
                "metric": names_metrics,
                "test": list(self.tests.keys())
            },
            attrs={"datasets": " - ".join(data_names)},
            name="test_result",
        )

        print("Performed replicability test for all variables.", flush=True)
        return p_values, test_results


    def _create_datasets_key(self, data_name_1, data_name_2, start_year, end_year):
        """ Create key for a pair of datasets together with the year range. """

        # return f"{data_name_1} - {data_name_2} ({start_year}-{end_year})"
        return (*sorted((data_name_1, data_name_2)), start_year, end_year)


    def _find_datasets_pair(self, data, data_names, start_year=None, end_year=None):
        """
        Look for the given pair of simulation ensembles in the provided data dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing precomputed data for a given dataset pair.
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        start_year : int
            Start year for filtering data.
        end_year : int
            End year for filtering data.

        Returns
        -------
        found_data : xr.DataArray or None
            Precomputed data for the given dataset pair, or None if not found.
        """

        # Validate input
        if (not isinstance(data_names, list) or len(data_names) != 2
            or not all(isinstance(name, str) for name in data_names)):
            raise TypeError("'data_names' must be a list of two strings representing simulation dataset names.")

        # Look for data
        found_data = None
        if start_year is not None and end_year is not None:
            # Exact match with date range
            key = self._create_datasets_key(*data_names, start_year, end_year)
            found_data = data.get(key)

        else:
            data_name_1, data_name_2 = sorted(data_names)

            # Match any date range
            for key, value in data.items():
                # if (key.startswith(f"{data_names[0]} - {data_names[1]}")
                #     or key.startswith(f"{data_names[1]} - {data_names[0]}")):
                #     found_data = data[key]
                if key[:2] == (data_name_1, data_name_2):
                    # Match specified start year
                    if start_year is not None and key[2] != start_year:
                        continue
                    # Match specified end year
                    if end_year is not None and key[3] != end_year:
                        continue

                    # Warn about selected years
                    data_general.warn_always(
                        f"Year range not fully specified. Using the first matching dataset: {key}."
                    )

                    found_data = value
                    break

        return found_data


    def _get_datasets_pair(self, data, data_names, start_year, end_year, attribute):

        """
        Look for the given pair of simulation ensembles in the provided data dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing precomputed data for a given dataset pair.
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        start_year : int
            Start year for filtering data.
        end_year : int
            End year for filtering data.
        attribute : str
            Name of the attribute to retrieve for the dataset pair.

        Returns
        -------
        attribute_ds: xr.DataArray
            Requested attribute for the given dataset pair.
        """

        attribute_ds = self._find_datasets_pair(data, data_names, start_year, end_year)
        if attribute_ds is None:
            raise ValueError(
                f"{attribute.capitalize()} between the selected datasets ('{data_names[0]}' and '{data_names[1]}') "
                f"and year range ({start_year}-{end_year}) not found. Please, run the 'perform_rep_test' "
                f"method with the selected datasets and years to compute the {attribute}."
            )
        
        return attribute_ds


    def add_datasets(self, datasets):
        """
        Add new datasets to the ReplicabilityTest object.

        Parameters
        ----------
        datasets : SimulationData or Iterable[SimulationData]
            Ensemble or list of ensembles containing simulation data and metadata to add.
        """

        # Validate input
        if isinstance(datasets, SimulationData):
            datasets = [datasets]
        elif not isinstance(datasets, list) or not all(isinstance(ds, SimulationData) for ds in datasets):
            raise TypeError("Input must be a SimulationData object or an list of SimulationData objects.")

        # Check for duplicate datasets
        added = False
        for dataset in datasets:
            if not any(ds.name == dataset.name for ds in self.datasets):
                self.datasets.append(dataset)
                added = True
            else:
                data_general.warn_always(
                    f"Dataset with name '{dataset.name}' already exists in the ReplicabilityTest object. "
                    "Skipping addition."
                )
        if added:
            self._compare_ensembles()

        # Add variables to the corresponding attribute if not already present
        if self.variables is None:
            expected_vars = data_general.load_yaml_file(config_params.VARIABLES_PATH)
            self.variables = {
                var: info
                for var, info in expected_vars.items()
                if var in self.datasets[0].data.data_vars
            }

        return


    def perform_rep_test(self, data_names=None, obs_path=None, start_year=None, end_year=None):
        """
        Perform replicability test comparing the given simulation ensembles.

        Parameters
        ----------
        data_names : list[str], optional
            List of names of two simulation ensembles to compare. If None, the first
            two datasets in the ReplicabilityTest object are used.
        obs_path : str
            Path to the observations database.
        start_year : int
            Start year for the test.
        end_year : int
            End year for the test.
        """

        # Validate inputs
        if data_names is None:
            if len(self.datasets) < 2:
                raise ValueError("At least two datasets are required for the replicability test.")

            # Get datasets from 'datasets' attribute if names not provided
            data_plot = [self.datasets[0], self.datasets[1]]
            data_names = [self.datasets[0].name, self.datasets[1].name]
            data_general.warn_always(
                f"As no dataset names were provided, the first two datasets in the ReplicabilityTest object "
                f"('{data_names[0]}' and '{data_names[1]}') will be used for the test."
            )

        elif (isinstance(data_names, list) and len(data_names) == 2
            and all(isinstance(name, str) for name in data_names)):

            # Look for the datasets in the 'datasets' attribute if names provided
            existing_names = [ds.name for ds in self.datasets]
            missing_names = [name for name in data_names if name not in existing_names]
            if missing_names:
                raise ValueError(
                    f"The following dataset names were not found in the ReplicabilityTest object: {missing_names}."
                )

            data_plot = [next(ds for ds in self.datasets if ds.name == name) for name in data_names]
            if not all("realization" in ds.data.coords for ds in data_plot):
                raise ValueError(
                    "All selected datasets must contain a 'realization' coordinate for ensemble computations."
                )
        else:
            raise TypeError("'data_names' must be a list of two strings representing simulation dataset names.")

        # Validate year range
        for dataset in data_plot:
            start_year, end_year = data_general.validate_year_range(
                dataset, start_year, end_year, process_name="replicability test"
            )
        data_plot_filtered = copy.deepcopy(data_plot)
        for i, dataset in enumerate(data_plot):
            data_plot_filtered[i].data = dataset.data.sel(time=slice(str(start_year), str(end_year)))


        # Check whether the test has already been performed for the given datasets and year range
        previous_tests = self._find_datasets_pair(self.test_results, data_names, start_year, end_year)

        if previous_tests is not None:
            data_general.warn_always(
                f"A replicability test between the selected datasets ('{data_names[0]}' and '{data_names[1]}') "
                f"and year range ({start_year}-{end_year}) has already been performed. "
            )

            # Check if running interactively
            if sys.stdin.isatty():
                try:
                    # Ask user for confirmation
                    response = input(
                        "Do you want to recompute the test anyway? This will overwrite existing results. (y/n): "
                    ).strip().lower()
                except EOFError:
                    print("\nReplicability test cancelled.")
                    return

                if response not in ['y', 'yes']:
                    print("Replicability test cancelled.")
                    return

            # Non-interactive mode: auto-recompute warning
            else:
                data_general.warn_always(
                    "Non-interactive mode detected. Existing results will be overwritten automatically."
                )

            print("Recomputing replicability test...")

        # Prepare observational data for the test
        if obs_path is None:
            raise ValueError(
                "Automatic selection of observations is not implemented yet. "
                "Please, provide a path to the observations database."
            )
        if not isinstance(obs_path, str):
            raise TypeError("'obs_path' must be a string representing the path to the observations database.")

        data_obs = ObservationData(obs_path, data_plot_filtered[0].data)


        # Run replicability test
        print(
            f"Started replicability test with significance level {self.alpha:.2f} and power {self.power:.2f} "
            f"to compare ensembles '{data_names[0]}' and '{data_names[1]}' for years {start_year}-{end_year}:",
            flush=True,
        )

        # Compute scores
        scores = self._compute_scores(data_plot_filtered, data_obs)
        datasets_key = self._create_datasets_key(data_names[0], data_names[1], start_year, end_year)

        # Compute effect sizes between the scores and store them in the object attribute
        eff_sizes = self._compute_eff_sizes(scores, data_names)
        eff_sizes.attrs["start_year"] = start_year
        eff_sizes.attrs["end_year"] = end_year
        self.effect_sizes[datasets_key] = eff_sizes

        # Apply statistical tests to the scores and store the results in the object attributes
        p_values, test_results = self._apply_tests(scores, data_names, datasets_key)

        p_values.attrs["start_year"] = start_year
        p_values.attrs["end_year"] = end_year
        self.p_values[datasets_key] = p_values

        test_results.attrs["start_year"] = start_year
        test_results.attrs["end_year"] = end_year
        self.test_results[datasets_key] = test_results

        return


    def get_effect_sizes(self, data_names, start_year=None, end_year=None):
        """
        Return precomputed effect sizes between the replicability test
        scores for the given simulation ensembles.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        start_year : int
            Start year for effect sizes.
        end_year : int
            End year for effect sizes.

        Returns
        -------
        effect_sizes_ds : xr.DataArray
            Effect sizes for all variables, seasons, regions, and metrics.
        """

        effect_sizes_ds = self._get_datasets_pair(self.effect_sizes, data_names, start_year, end_year, 'effect sizes')
        # if effect_sizes_ds is None:
        #     raise ValueError(
        #         f"Effect sizes between the selected datasets ('{data_names[0]}' and '{data_names[1]}') "
        #         f"and year range ({start_year}-{end_year}) not found. Please, run the 'perform_rep_test' "
        #         f"method with the selected datasets and years to compute the effect sizes."
        #     )

        return effect_sizes_ds


    def get_p_values(self, data_names, start_year=None, end_year=None):
        """
        Return precomputed p-values for the replicability test for the given
        simulation ensembles.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        start_year : int
            Start year for p-values.
        end_year : int
            End year for p-values.

        Returns
        -------
        p_values_ds : xr.DataArray
            Raw p-values for all variables, seasons, regions, metrics, and tests.
        """

        p_values_ds = self._get_datasets_pair(self.p_values, data_names, start_year, end_year, 'p-values')
        # if p_values_ds is None:
        #     raise ValueError(
        #         f"p-values for the selected datasets ('{data_names[0]}' and '{data_names[1]}') and "
        #         f"year range ({start_year}-{end_year}) not found. Please, run the 'perform_rep_test' "
        #         f"method with the selected datasets and years to compute the p-values."
        #     )

        return p_values_ds


    def get_test_results(self, data_names, start_year=None, end_year=None):
        """
        Return replicability test results for the given simulation ensembles.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        start_year : int
            Start year for test results.
        end_year : int
            End year for test results.

        Returns
        -------
        test_results_ds : xr.DataArray
            Results of the replicability test for all variables, seasons,
            regions, metrics, and tests.
        """

        test_results_ds = self._get_datasets_pair(self.test_results, data_names, start_year, end_year, 'test results')
        # if test_results_ds is None:
        #     raise ValueError(
        #         f"Replicability test results between the selected datasets ('{data_names[0]}' and '{data_names[1]}') "
        #         f"and year range ({start_year}-{end_year}) not found. Please, run the 'perform_rep_test' method "
        #         f"with the selected datasets and years to compute the results."
        #     )

        return test_results_ds


    def save_data(self, data_names, output_path, start_year=None, end_year=None):
        """
        Save computed effect size between the replicability test
        scores and test results to NetCDF files.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        output_path : str
            Path to save the data files.
        start_year : int
            Start year for test output.
        end_year : int
            End year for test output.
        """

        # Look for results in stored attributes
        eff_sizes = self.get_effect_sizes(data_names, start_year, end_year)
        test_results = self.get_test_results(data_names, start_year, end_year)

        # Prepare output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)


        # Save data (save p-values too?)
        data_names_file = "-".join([name.replace(" ", "_") for name in data_names])
        year_range_str = f"{eff_sizes.attrs['start_year']}-{eff_sizes.attrs['end_year']}"

        eff_sizes_path = output_path / f"effect_size_scores_{data_names_file}_{year_range_str}.nc"
        eff_sizes.to_netcdf(eff_sizes_path)
        print(f"Effect size between the replicability test scores for '{data_names[0]}' and '{data_names[1]}' "
              f"and years {year_range_str} saved to '{eff_sizes_path}'.", flush=True)

        test_results_path = output_path / f"replicability_test_results_{data_names_file}_{year_range_str}.nc"
        test_results.to_netcdf(test_results_path)
        print(f"Replicability test results for '{data_names[0]}' and '{data_names[1]}' and years {year_range_str} "
              f"saved to '{test_results_path}'.", flush=True)

        return


    def matrix_plot(self, data_names, output_path=None, start_year=None, end_year=None):
        """
        Generate matrix plot with effect sizes and replicability test results.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        output_path : str, optional
            Path to save the matrix plot.
        start_year : int
            Start year for test output.
        end_year : int
            End year for test output.
        """

        # Find results for the given datasets in stored attributes
        effect_sizes_ds = self._find_datasets_pair(self.effect_sizes, data_names, start_year, end_year)
        test_results_ds = self._find_datasets_pair(self.test_results, data_names, start_year, end_year)
        if effect_sizes_ds is None or test_results_ds is None:
            raise ValueError(
                f"Replicability test output between the selected datasets ('{data_names[0]}' and '{data_names[1]}') "
                f"and year range ({start_year}-{end_year}) not found. Please, run the 'perform_rep_test' method "
                f"with the selected datasets and years to perform the replicability test."
            )

        # Prepare data for plotting
        effect_sizes_array = effect_sizes_ds.values
        n_vars, n_seasons, n_regions, n_metrics = effect_sizes_array.shape
        effect_sizes_array_reshaped = effect_sizes_array.reshape(n_vars, n_seasons * n_regions, n_metrics)

        test_results_array = np.any(test_results_ds.values, axis=3)
        n_tests = test_results_array.shape[-1]
        test_results_array_reshaped = test_results_array.reshape(n_vars, n_seasons * n_regions, n_tests)


        # Generate matrix plot
        year_range_str = f"{effect_sizes_ds.attrs['start_year']}-{effect_sizes_ds.attrs['end_year']}"
        matrix, _ = plots_general.plot_matrix(
            effect_sizes_array_reshaped,
            test_results_array_reshaped,
            title=f"Outcome of the replicability test for '{data_names[0]}' vs '{data_names[1]}' ({year_range_str})",
            variables=self.variables,
        )

        data_names_str = "-".join([name.replace(" ", "_") for name in data_names])
        plots_general.save_or_show_plot(
            matrix,
            output_path,
            plot_filename=f"replicability_test_matrix_{data_names_str}_{year_range_str}",
            plot_name="Replicability test matrix plot",
        )

        return


    def report(self, output_path, time_series=False, spatial=False):
        """
        Generate a summary report with the results of the replicability test
        and the selected plots.

        Parameters
        ----------
        output_path : str
            Path to save the report.
        time_series : bool
            Whether to include time series plots in the report (default: False).
        spatial : bool
            Whether to include spatial plots in the report (default: False).
        """

        generated_plots = {"time_series": False, "spatial": False, "matrix": False}

        # Check if the selected plots have already been generated
        # Generate missing plots and save all in generated_plots

        report.pdf_replicability(output_path, generated_plots)
        raise NotImplementedError("This function is not implemented yet.")
