import warnings
warnings.simplefilter("always")

import numpy as np
import xarray as xr
import concurrent.futures

from pathlib import Path
from scipy.stats import bootstrap
from collections.abc import Iterable
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
    obs_path : str
        Path to the observations database.
    alpha : float
        Significance level for the statistical tests (default: 0.05).
    power : float
        Statistical power to compute minimum detectable effect size for the t-test
        (default: 0.8).

    Attributes
    ----------
    datasets : list[SimulationData]
        List of ensembles containing simulation data and metadata.
    obs_path : str
        Path to the observations database.
    obs : ObservationData
        Instance containing observational data for comparison.
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
    effect_sizes : dict[str, xr.DataArray]
        Dictionary to store effect sizes between the replicability test scores for
        each pair of datasets for all variables, seasons, regions, and metrics.
    test_results : dict[str, xr.DataArray]
        Dictionary to store results of the replicability test for each pair of datasets
        for all variables, seasons, regions, metrics, and statistical tests.
    """

    def __init__(self, datasets=None, obs_path=None, alpha=0.05, power=0.8):

        # Validate inputs and prepare observational data and variables
        self.obs_path = obs_path
        if datasets is None:
            self.datasets = []
            self.obs = None
            self.variables = None
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

            self._compare_ensembles()
            self.obs = ObservationData(self.obs_path, self.datasets[0].data)

            expected_vars = data_general.load_yaml_file(config_params.VARIABLES_PATH)
            self.variables = {
                var: info
                for var, info in expected_vars.items()
                if var in datasets[0].data.data_vars
            }

        # Validate significance level and statistical power
        if not isinstance(alpha, (int, float)) or not isinstance(power, (int, float)):
            raise TypeError(
                "The significance level 'alpha' and statistical power 'power' must be numeric."
            )
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
                raise ValueError(
                    f"Ensembles '{ref.name}' and '{dataset.name}' have different variables."
                )

            dataset_lat = dataset.data.coords["lat"]
            if not np.array_equal(ref_lat, dataset_lat):
                raise ValueError(
                    f"Ensembles '{ref.name}' and '{dataset.name}' have different latitude coordinates."
                )

            dataset_lon = dataset.data.coords["lon"]
            if not np.array_equal(ref_lon, dataset_lon):
                raise ValueError(
                    f"Ensembles '{ref.name}' and '{dataset.name}' have different longitude coordinates."
                )

        return


    def _compute_scores_one_var(self, args):
        """
        Compute scores for the given variable in both simulation ensembles.

        Parameters
        ----------
        args : tuple
            List containing:
                var_name (str): Climate variable name.
                data_plot (list[SimulationData]): List of two simulation ensembles to compare.

        Returns
        -------
        scores_dataset : tuple[str, xr.Dataset]
            Variable name and dataset containing computed scores.
        """

        # Validate inputs
        var_name, data_plot = args

        if (
            not isinstance(data_plot, list)
            or len(data_plot) == 0
            or not all(isinstance(ds, SimulationData) for ds in data_plot)
        ):
            raise TypeError("'data_plot' must be a non-empty list of SimulationData instances.")

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )

        # Prepare datasets and check matching time coordinates
        datasets = [
            data_plot[0].data[[var_name]].persist(),
            data_plot[1].data[[var_name]].persist(),
        ]
        data_obs = self.obs.data[[var_name]].resample(time="1MS").sum().persist()

        if not datasets[0].time.equals(datasets[1].time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n"
                f"  {data_plot[0].name} has time from {datasets[0].time.min().item()} "
                f"to {datasets[0].time.max().item()}\n"
                f"  {data_plot[1].name} has time from {datasets[1].time.min().item()} "
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
                        data_obs_season = (
                            data_obs.groupby("time.season")[season]
                            if obs_needed
                            else None
                        )
                    if metric_idx < 2:
                        data_sim_season = data_sim_season.mean(dim="time")

                    # Global, tropical and extratropical regions
                    for region_idx, region in enumerate(self.regions.values()):
                        mask = region(lat)
                        data_sim_region = data_sim_season.where(mask, drop=True)
                        data_obs_region = (
                            data_obs_season.where(mask, drop=True)
                            if obs_needed
                            else None
                        )

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
            "dataset": [data_plot[0].name, data_plot[1].name],
            "season": self.seasons,
            "region": list(self.regions.keys()),
            "realization": np.arange(length_realizations),
        }

        scores_var = {}
        for metric_name, scores in scores_dict.items():
            scores_var[metric_name] = (
                ["dataset", "season", "region", "realization"],
                scores,
            )

        scores_dataset = xr.Dataset(data_vars=scores_var, coords=coords)
        scores_dataset.attrs["variable"] = var_name
        scores_dataset.attrs["long_name"] = self.variables[var_name]["long_name"]

        print(f"\tComputed scores for variable '{var_name}'...", flush=True)
        return scores_dataset


    def _compute_scores(self, data_plot):
        """
        Compute scores for all variables in both simulation ensembles in parallel.

        Parameters
        ----------
        data_plot : list[SimulationData])
            List of two simulation ensembles to compare.

        Returns
        -------
        scores_all : dict[str, xr.Dataset]
            Dictionary of scores datasets for each variable.
        """

        # Validate inputs
        if (
            not isinstance(data_plot, list)
            or len(data_plot) == 0
            or not all(isinstance(ds, SimulationData) for ds in data_plot)
        ):
            raise TypeError("'data_plot' must be a non-empty list of SimulationData instances.")

        # Compute scores for each variable in parallel
        scores_all = {}
        var_names = list(self.variables.keys())
        tasks = [(var, data_plot) for var in var_names]
        # for task in tasks:
        #     scores_all[task[0][1]] = self._compute_scores_one_var(task[0], task[1])
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_vars) as executor:
            for idx, value in enumerate(executor.map(self._compute_scores_one_var, tasks)):
                scores_all[tasks[idx][0]] = value

        print("Computed scores for all variables...", flush=True)
        return scores_all


    def _validate_scores_and_data_names(self, scores_all, data_names):
        """
        Validate the format of scores_all and data_names and check the presence
        of all datasets in the scores_all dictionary.

        Parameters
        ----------
        scores_all : dict[str, xr.Dataset]
            Dictionary of scores datasets for each variable.
        data_names : list[str]
            List of two simulation ensemble names to compare.
        """

        # Check format of scores_all
        if not isinstance(scores_all, dict) or not all(
            isinstance(key, str) and isinstance(value, xr.Dataset)
            for key, value in scores_all.items()
        ):
            raise TypeError(
                "'scores_all' must be a dictionary with variable names as keys and xarray.Dataset as values."
            )

        # Check format of data_names
        if (
            not isinstance(data_names, list)
            or len(data_names) != 2
            or not all(isinstance(name, str) for name in data_names)
        ):
            raise TypeError(
                "'data_names' must be a list of two strings representing simulation dataset names."
            )

        # Check that all datasets in data_names are present in scores_all
        for var_name, scores in scores_all.items():
            for name in data_names:
                if name not in scores.coords["dataset"].values:
                    raise ValueError(
                        f"Dataset '{name}' not found in scores for variable '{var_name}'. "
                        f"Available datasets: {scores.coords['dataset'].values}"
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

        print(
            "Computed effect sizes between scores distributions for all variables...",
            flush=True,
        )
        return effect_sizes


    def _apply_tests(self, scores_all, data_names):
        """
        Compare scores with statistical tests separating by season, region and metric, for all 
        available variables.

        Parameters
        ----------
        scores_all : dict[str, xr.Dataset])
            Dictionary of scores datasets for each variable.
        data_names : list[str]
            List of two simulation ensemble names to compare.

        Returns
        -------
        test_results : xr.DataArray
            Test results for all variables, seasons, regions, metrics, and statistical tests.
        """

        # Validate inputs
        self._validate_scores_and_data_names(scores_all, data_names)


        # Initialize array
        names_metrics = np.append(self.metrics["name"], "Combined")
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
                        effect_size = self.effect_sizes[" - ".join(data_names)][
                            var_idx, season_idx, region_idx, metric_idx
                        ]
                        min_detectable_effect_size = power_analysis.solve_power(
                            effect_size=None,
                            nobs1=len(scores_ref),
                            alpha=self.alpha,
                            power=self.power,
                        )

                        if effect_size < min_detectable_effect_size:
                            test_results[var_idx, season_idx, region_idx, metric_idx, :] = True
                            continue

                        # Apply statistical tests
                        for test_idx, test_name in enumerate(self.tests):
                            p_value = self.tests[test_name](scores_ref, scores_test)
                            test_results[var_idx, season_idx, region_idx, metric_idx, test_idx] = (
                                p_value <= self.alpha
                            )


        # Save all test results as a xr.DataArray
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
        return test_results


    def _find_dataset_pair(self, data, data_names):
        """
        Look for the given pair of simulation ensembles in the provided data dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing precomputed data for a given dataset pair.
        data_names : list[str]
            List of names of two simulation ensembles to compare.

        Returns
        -------
        found_data : xr.DataArray or None
            Precomputed data for the given dataset pair, or None if not found.
        """

        # Validate input
        if (
            not isinstance(data_names, list)
            or len(data_names) != 2
            or not all(isinstance(name, str) for name in data_names)
        ):
            raise TypeError(
                "'data_names' must be a list of two strings representing simulation dataset names."
            )

        # Look for data
        if " - ".join(data_names) in data:
            datasets_name = " - ".join(data_names)
            found_data = data[datasets_name]
        elif " - ".join(data_names[::-1]) in data:
            data_names = data_names[::-1]
            datasets_name = " - ".join(data_names)
            found_data = data[datasets_name]
        else:
            found_data = None

        return found_data


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
        elif (
            not isinstance(datasets, Iterable)
            or isinstance(datasets, (str, bytes))
            or not all(isinstance(ds, SimulationData) for ds in datasets)
        ):
            raise TypeError(
                "Input must be a SimulationData object or an iterable of SimulationData objects."
            )

        # Check for duplicate datasets
        added = False
        for dataset in datasets:
            if not any(ds.name == dataset.name for ds in self.datasets):
                self.datasets.append(dataset)
                added = True
            else:
                warnings.warn(
                    f"Dataset with name '{dataset.name}' already exists in the ReplicabilityTest object. "
                    "Skipping addition."
                )
        if added:
            self._compare_ensembles()

        # Add observation data and variables if not already present
        if self.obs is None:
            self.obs = ObservationData(self.obs_path, self.datasets[0].data)
            expected_vars = data_general.load_yaml_file(config_params.VARIABLES_PATH)
            self.variables = {
                var: info
                for var, info in expected_vars.items()
                if var in self.datasets[0].data.data_vars
            }

        return


    def perform_rep_test(self, data_names=None):
        """
        Perform replicability test comparing the given simulation ensembles.

        Parameters
        ----------
        data_names : list[str], optional
            List of names of two simulation ensembles to compare. If None, the first
            two datasets in the ReplicabilityTest object are used.
        """

        # Validate inputs
        if data_names is None:
            if len(self.datasets) < 2:
                raise ValueError("At least two datasets are required for the replicability test.")
            data_plot = [self.datasets[0], self.datasets[1]]
            data_names = [self.datasets[0].name, self.datasets[1].name]
        elif (
            isinstance(data_names, list)
            and len(data_names) == 2
            and all(isinstance(name, str) for name in data_names)
        ):
            existing_names = [ds.name for ds in self.datasets]
            missing_names = [name for name in data_names if name not in existing_names]
            if missing_names:
                raise ValueError(
                    f"The following dataset names were not found in the ReplicabilityTest object: {missing_names}."
                )

            data_plot = [next(ds for ds in self.datasets if ds.name == name) for name in data_names]
            if (
                "realization" not in data_plot[0].data.coords
                or "realization" not in data_plot[1].data.coords
            ):
                raise ValueError(
                    "All selected datasets must contain a 'realization' coordinate for ensemble computations."
                )
        else:
            raise TypeError(
                "'data_names' must be a list of two strings representing simulation dataset names."
            )
        

        # Run replicability test
        print(
            f"Started replicability test with significance level {self.alpha} to compare ensembles "
            f"'{data_names[0]}' and '{data_names[1]}':",
            flush=True,
        )

        # Compute scores
        scores = self._compute_scores(data_plot)
        datasets_name = " - ".join(data_names)

        # Compute effect sizes between the scores and store them in the object attribute
        eff_sizes = self._compute_eff_sizes(scores, data_names)
        self.effect_sizes[datasets_name] = eff_sizes

        # Apply statistical tests to the scores and store the results in the object attribute
        test_results = self._apply_tests(scores, data_names)
        self.test_results[datasets_name] = test_results

        return


    def get_effect_sizes(self, data_names):
        """
        Return precomputed effect sizes between the replicability test
        scores for the given simulation ensembles.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.

        Returns
        -------
        effect_sizes_ds : xr.DataArray
            Effect sizes for all variables, seasons, regions, and metrics.
        """

        # Find effect sizes for the given datasets in the stored attributes
        effect_sizes_ds = self._find_dataset_pair(self.effect_sizes, data_names)
        if effect_sizes_ds is None:
            raise ValueError(
                f"Effect sizes between the selected datasets ('{data_names[0]}' and '{data_names[1]}') not found."
                f" Please, run 'perform_rep_test' method with the selected datasets to compute the effect sizes."
            )

        return effect_sizes_ds


    def get_test_results(self, data_names):
        """
        Return replicability test results for the given simulation ensembles.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.

        Returns
        -------
        test_results_ds : xr.DataArray
            Results of the replicability test for all variables, seasons,
            regions, metrics, and tests.
        """

        # Find test results for the given datasets in the stored attributes
        test_results_ds = self._find_dataset_pair(self.test_results, data_names)
        if test_results_ds is None:
            raise ValueError(
                f"Replicability test results for the selected datasets ('{data_names[0]}' and '{data_names[1]}') "
                "not found. Please, run 'perform_rep_test' method with the selected datasets to compute the results."
            )

        return test_results_ds


    def save_data(self, data_names, output_path):
        """
        Save computed effect size between the replicability test
        scores and test results to NetCDF files.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        output_path : str
            Path to save the data files.
        """

        # Look for results in stored attributes
        datasets_name = " - ".join(data_names)
        eff_sizes = self.get_effect_sizes(data_names)
        test_results = self.get_test_results(data_names)

        # Prepare output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)


        # Save data
        data_names_file = "-".join([name.replace(" ", "_") for name in data_names])

        eff_sizes_path = output_path / f"effect_size_scores_{data_names_file}.nc"
        eff_sizes.to_netcdf(eff_sizes_path)
        print(
            f"Effect size between the replicability test scores for '{datasets_name}' saved to '{eff_sizes_path}'",
            flush=True,
        )

        test_results_path = output_path / f"replicability_test_results_{data_names_file}.nc"
        test_results.to_netcdf(test_results_path)
        print(
            f"Replicability test results for '{datasets_name}' saved to '{test_results_path}'",
            flush=True,
        )

        return


    def matrix_plot(self, data_names, output_path=None):
        """
        Generate matrix plot with effect sizes and replicability test results.

        Parameters
        ----------
        data_names : list[str]
            List of names of two simulation ensembles to compare.
        output_path : str, optional
            Path to save the matrix plot.
        """

        # Find results for the given datasets in stored attributes
        effect_sizes_ds = self._find_dataset_pair(self.effect_sizes, data_names)
        test_results_ds = self._find_dataset_pair(self.test_results, data_names)
        if effect_sizes_ds is None or test_results_ds is None:
            raise ValueError(
                f"Replicability test output between the selected datasets ('{data_names[0]}' and '{data_names[1]}') "
                "not found. Please, run 'perform_rep_test' method with the selected datasets to perform "
                "the replicability test."
            )

        # Prepare data for plotting
        effect_sizes_array = effect_sizes_ds.values
        n_vars, n_seasons, n_regions, n_metrics = effect_sizes_array.shape
        effect_sizes_array_reshaped = effect_sizes_array.reshape(n_vars, n_seasons * n_regions, n_metrics)

        test_results_array = np.any(test_results_ds.values, axis=3)
        n_tests = test_results_array.shape[-1]
        test_results_array_reshaped = test_results_array.reshape(n_vars, n_seasons *n_regions, n_tests)

        # Generate matrix plot
        matrix, _ = plots_general.plot_matrix(
            effect_sizes_array_reshaped,
            test_results_array_reshaped,
            title=f"Outcome of the replicability test ({data_names[0]} vs {data_names[1]})",
            variables=self.variables,
        )

        data_names_str = "-".join([name.replace(" ", "_") for name in data_names])
        plots_general.save_or_show_plot(
            matrix,
            output_path,
            plot_filename=f"replicability_test_matrix_{data_names_str}",
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
