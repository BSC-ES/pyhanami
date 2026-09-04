import copy
import cmocean

import numpy as np
import xarray as xr
import concurrent.futures
import multiprocessing as mp

from tqdm import tqdm
from scipy.stats import ttest_ind
from collections.abc import Iterable
from matplotlib.colors import LinearSegmentedColormap

from pyhanami.config import config_params
from pyhanami.utils.plots import plots_general
from pyhanami.utils import data_general, statistics
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData


class DataDiagnostics:
    """
    Perform diagnostic comparisons between climate simulation ensembles.

    This class provides functionality for computing and visualizing differences in climate
    variables between simulation ensembles. It includes methods for computing annual time
    series, absolute differences, effect sizes and significance differences at grid point
    level.

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
    max_workers_grid : int
        Number of parallel workers used for grid-level computations.
    """

    def __init__(self, datasets=None):
        if datasets is None:
            self.datasets = []
        elif isinstance(datasets, SimulationData):
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

        # Load config parameters once
        self.variables = data_general.load_yaml_file(config_params.VARIABLES_PATH)
        self.max_workers_grid = config_params.MAX_WORKERS_GRID


    def _compute_time_series(self, var_name, data_plot, time_freq="1YS"):
        """
        Compute time series for the given simulation ensembles and variable with the selected time frequency.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_plot : list[SimulationData and/or ObservationData]
            List of ensembles to compute time series for.
        time_freq : str
            Resampling frequency for averaging (default: '1YS').

        Returns
        -------
        time_series : list[xr.DataArray])
            List of mean time series.
        """

        # Validate inputs
        if (
            not isinstance(data_plot, list)
            or len(data_plot) == 0
            or not all(isinstance(ds, (SimulationData, ObservationData)) for ds in data_plot)
        ):
            raise TypeError("'data_plot' must be a non-empty list of SimulationData instances.")

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )


        # Compute mean time series for each dataset
        time_series = []
        for dataset in [ds.data for ds in data_plot]:
            data_var = dataset[var_name]
            weights = statistics.area_weights(data_var)
            data_weighted = data_var.weighted(weights)

            data_area_mean = data_weighted.mean(["lat", "lon"])
            data_time_mean = data_area_mean.resample(time=time_freq).mean().compute()
            time_series.append(data_time_mean)

        return time_series


    def _compute_abs_diff(self, var_name, data_plot):
        """
        Compute absolute average difference between two simulation ensembles
        for the given variable and time range at the grid point level.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_plot : list[SimulationData]
            List of two simulation ensembles to compute the absolute difference for.

        Returns
        -------
        data_diff : xr.DataArray
            Absolute difference between the two ensembles.
        """

        # Validate inputs
        if (
            not isinstance(data_plot, list)
            or len(data_plot) != 2
            or not all(isinstance(ds, SimulationData) for ds in data_plot)
        ):
            raise TypeError(
                "'data_plot' must be a non-empty list with two SimulationData instances."
            )

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )

        # Validate time coordinates match
        data_sim_1 = data_plot[0].data.persist()
        data_sim_2 = data_plot[1].data.persist()
        if not data_sim_1.time.equals(data_sim_2.time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n"
                f"  {data_plot[0].name} has time from {str(data_sim_1.time.min().values)[:19]} "
                f"to {str(data_sim_1.time.max().values)[:19]}\n"
                f"  {data_plot[1].name} has time from {str(data_sim_2.time.min().values)[:19]} "
                f"to {str(data_sim_2.time.max().values)[:19]}"
            )


        # Compute mean absolute difference
        data_sim_mean_1 = data_sim_1[var_name].mean(["time"])
        data_sim_mean_2 = data_sim_2[var_name].mean(["time"])
        if "realization" in data_sim_1.coords:
            data_sim_mean_1 = data_sim_mean_1.mean(["realization"])
        if "realization" in data_sim_2.coords:
            data_sim_mean_2 = data_sim_mean_2.mean(["realization"])

        if data_sim_mean_1.shape != data_sim_mean_2.shape:
            raise ValueError(
                "Averaged data shapes of the two datasets do not match "
                f"({data_sim_mean_1.shape} vs {data_sim_mean_2.shape})."
            )

        data_diff = (data_sim_mean_1 - data_sim_mean_2).compute()
        if var_name in ["siconc", "sos", "tos"]:
            data_diff = xr.where((np.isnan(data_diff)) | (data_diff == 0), 10**-6, data_diff)
            # NOTE: values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0

        print(
            f"Computed absolute difference for variable '{var_name}' between "
            f"'{data_plot[0].name}' and '{data_plot[1].name}'.",
            flush=True,
        )
        return data_diff


    def _compute_eff_size_ens(self, var_name, data_plot):
        """
        Compute average effect size (Cohen's d) between two simulation ensembles
        in parallel for the given variable at the grid point level.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_plot : list[SimulationData]
            List of two simulation ensembles to compute the absolute difference for.

        Returns
        -------
        data_effect_size : xr.DataArray)
            Effect size between the two ensembles.
        """

        # Validate inputs
        if (
            not isinstance(data_plot, list)
            or len(data_plot) != 2
            or not all(isinstance(ds, SimulationData) for ds in data_plot)
        ):
            raise TypeError(
                "'data_plot' must be a non-empty list with two SimulationData instances."
            )

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )
            if "realization" not in dataset.data.coords:
                raise ValueError(
                    f"Dataset '{dataset.name}' must contain a 'realization' coordinate for ensemble computations."
                )

        # Validate time coordinates match
        data_sim_1 = data_plot[0].data.persist()
        data_sim_2 = data_plot[1].data.persist()
        if not data_sim_1.time.equals(data_sim_2.time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n"
                f"  {data_plot[0].name} has time from {str(data_sim_1.time.min().values)[:19]} "
                f"to {str(data_sim_1.time.max().values)[:19]}\n"
                f"  {data_plot[1].name} has time from {str(data_sim_2.time.min().values)[:19]} "
                f"to {str(data_sim_2.time.max().values)[:19]}"
            )


        # Prepare data
        data_sim_flat_1 = data_sim_1[var_name].mean("time").stack(ngrid=["lat", "lon"]).compute()
        data_sim_flat_2 = data_sim_2[var_name].mean("time").stack(ngrid=["lat", "lon"]).compute()
        if data_sim_flat_1.shape != data_sim_flat_2.shape:
            raise ValueError(
                "Averaged data shapes of the two datasets do not match "
                f"({data_sim_flat_1.shape} vs {data_sim_flat_2.shape})."
            )

        #  Compute effect sizes in parallel
        tasks = [
            (data_sim_flat_1.sel(ngrid=i).values, data_sim_flat_2.sel(ngrid=i).values)
            for i in data_sim_flat_1.ngrid
        ]
        effect_size = np.empty(len(tasks))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers_grid, mp_context=mp.get_context("spawn")
        ) as executor:
            for idx, value in enumerate(
                tqdm(
                    executor.map(statistics.cp_effect_size_bootstrap, tasks),
                    total=len(tasks),
                    desc=f"Computing effect sizes for variable '{var_name}'",
                    unit=" grid points",
                )
            ):
                effect_size[idx] = value

        # Convert to xarray.DataArray
        effect_size_reshaped = effect_size.reshape(data_sim_1.sizes['lat'], data_sim_1.sizes['lon'])
        data_effect_size = xr.DataArray(
            effect_size_reshaped,
            dims=["lat", "lon"],
            coords={"lat": data_sim_1["lat"], "lon": data_sim_1["lon"]},
            name=var_name,
        )

        print(
            f"Computed effect size for variable '{var_name}' between "
            f"'{data_plot[0].name}' and '{data_plot[1].name}'.",
            flush=True,
        )
        return data_effect_size


    def _compute_significant_diff(self, var_name, data_plot, alpha=0.05, stat=ttest_ind):
        """
        Compute significant difference between two simulation ensembles
        in parallel for the given variable at the grid point level.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_plot : list[SimulationData]
            List of two simulation ensembles to compute the significant differences for.
        alpha : float
            Significance level for the statistical test (default: 0.05).
        stat : Callable
            Statistical test function to use (default: ttest_ind).

        Returns
        -------
        significant : np.ndarray
            Boolean array indicating significant differences between the two ensembles.
        """

        # Validate input
        if (
            not isinstance(data_plot, list)
            or len(data_plot) != 2
            or not all(isinstance(ds, SimulationData) for ds in data_plot)
        ):
            raise TypeError(
                "'data_plot' must be a non-empty list with two SimulationData instances."
            )

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )
            if "realization" not in dataset.data.coords:
                raise ValueError(
                    f"Dataset '{dataset.name}' must contain a 'realization' coordinate for ensemble computations."
                )
        if not isinstance(alpha, (int, float)):
            raise TypeError("The significance level 'alpha' must be numeric.")
        if not (0 <= alpha <= 1):
            raise ValueError("'alpha' must be between 0 and 1.")
        if not callable(stat):
            raise TypeError("'stat' must be callable.")

        data_sim_1 = data_plot[0].data.persist()
        data_sim_2 = data_plot[1].data.persist()
        if not data_sim_1.time.equals(data_sim_2.time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n"
                f"  {data_plot[0].name} has time from {str(data_sim_1.time.min().values)[:19]} "
                f"to {str(data_sim_1.time.max().values)[:19]}\n"
                f"  {data_plot[1].name} has time from {str(data_sim_2.time.min().values)[:19]} "
                f"to {str(data_sim_2.time.max().values)[:19]}"
            )


        # Prepare data
        n_lats = data_sim_1.sizes["lat"]
        n_lons = data_sim_1.sizes["lon"]
        n_points = n_lats * n_lons
        n_realizations = data_sim_1.sizes["realization"]

        data_sim_flat_1 = data_sim_1[var_name].mean("time").data.compute().reshape((n_realizations,n_points))
        data_sim_flat_2 = data_sim_2[var_name].mean("time").data.compute().reshape((n_realizations,n_points))

        if data_sim_flat_1.shape != data_sim_flat_2.shape:
            raise ValueError(
                "Averaged data shapes of the two datasets do not match "
                f"({data_sim_flat_1.shape} vs {data_sim_flat_2.shape})."
            )
        d1, d2 = (data_sim_flat_1, data_sim_flat_2)

        # Compute significant differences in parallel
        tasks = [(d1[:, n], d2[:, n], alpha, stat) for n in range(n_points)]
        significant = np.zeros(n_points)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers_grid, mp_context=mp.get_context("spawn")
        ) as executor:
            for idx, value in enumerate(executor.map(statistics.significant_diff, tasks)):
                significant[idx] = value
        significant_reshaped = significant.reshape((n_lats, n_lons))

        print(
            f"Computed significant difference for variable '{var_name}' between "
            f"'{data_plot[0].name}' and '{data_plot[1].name}'.",
            flush=True,
        )
        return significant_reshaped


    def _compute_bias(self, var_name, data_plot):
        """
        Compute average bias between a simulation ensemble and observations
        for the given variable at the grid point level.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_plot : list[SimulationData and ObservationData]
            List containing the simulation ensemble and the observational dataset.

        Returns
        -------
        data_bias : xr.DataArray
            Bias between the simulation ensemble and observations.
        """

        # Validate inputs
        if (
            not isinstance(data_plot, list)
            or len(data_plot) != 2
            or not isinstance(data_plot[0], SimulationData)
            or not isinstance(data_plot[1], ObservationData)
        ):
            raise TypeError(
                "'data_plot' must be a list with a SimulationData instance and an ObservationData instance."
            )

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the dataset '{dataset.name}'. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )

        # Validate time coordinates match
        data_sim = data_plot[0].data.persist()
        data_obs = data_plot[1].data.persist()
        if not data_sim.time.equals(data_obs.time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n"
                f"  {data_plot[0].name} has time from {str(data_sim.time.min().values)[:19]} "
                f"to {str(data_sim.time.max().values)[:19]}\n"
                f"  {data_plot[1].name} has time from {str(data_obs.time.min().values)[:19]} "
                f"to {str(data_obs.time.max().values)[:19]}"
            )


        # Compute mean bias
        data_sim_mean = data_sim[var_name].mean(["time"])
        data_obs_mean = data_obs[var_name].mean(["time"])
        if "realization" in data_sim.coords:
            data_sim_mean = data_sim_mean.mean(["realization"])
        if data_sim_mean.shape != data_obs_mean.shape:
            raise ValueError(
                "Averaged data shapes of the two datasets do not match "
                f"({data_sim_mean.shape} vs {data_obs_mean.shape})."
            )

        data_bias = (data_sim_mean - data_obs_mean).compute()
        return data_bias


    def add_datasets(self, datasets):
        """
        Add new datasets to the DataDiagnostics object.

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
                    f"\nDataset with name '{dataset.name}' already exists in the DataDiagnostics object. "
                    "Skipping addition."
                )

        return


    def time_series_plot(self, var_name, data_names=None, output_path=None, obs=False, obs_paths=None,
                         obs_names=None, time_freq='annual', start_year=None, end_year=None,
                         plot_ens=False):
        """
        Generate time series plot for the given datasets and variable for the selected period
        and time frequency.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_names : str or list[str], optional
            Name or list of names of simulation ensembles to plot. If None, all datasets
            in the DataDiagnostics object are used.
        output_path : str, optional
            Path to save the time series plot.
        obs : bool
            If True, also plot observational data if available (default: False).
        obs_paths : str or list[str], optional
            Path to the observations database/s.
        obs_names : str or list[str], optional
            Name of the observational dataset/s.
        time_freq : str
            Resampling frequency (default: 'annual').
        start_year : int
            Start year to plot.
        end_year : int
            End year to plot.
        plot_ens : bool
            Whether to plot individual ensemble members trajectories (default: False).
        """

        # Validate inputs
        if data_names is None:
            data_plot = self.datasets
            data_names = [ds.name for ds in data_plot]
        elif isinstance(data_names, str):
            data_plot = [ds for ds in self.datasets if ds.name == data_names]
            if not data_plot:
                raise ValueError(
                    f"Dataset with name '{data_names}' not found in the DataDiagnostics object."
                )
            data_names = [data_names]
        elif isinstance(data_names, list) and all(isinstance(name, str) for name in data_names):
            existing_names = [ds.name for ds in self.datasets]
            missing_names = [name for name in data_names if name not in existing_names]
            if missing_names:
                raise ValueError(
                    f"The following dataset names were not found in the DataDiagnostics object: {missing_names}."
                )

            data_plot = [next(ds for ds in self.datasets if ds.name == name) for name in data_names]
        else:
            raise TypeError(
                "'data_names' must be a string or a list of strings representing dataset names."
            )

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )

        if obs:
            if obs_paths is None or obs_names is None:
                raise NotImplementedError(
                    "Automatic selection of observations is not implemented yet. "
                    "Please provide at least one path and one name for the observations database."
                )

            if isinstance(obs_paths, str):
                if not isinstance(obs_names, str):
                    raise TypeError("'obs_names' must be a string if 'obs_paths' is a string.")
                obs_paths = [obs_paths]
                obs_names = [obs_names]
            elif isinstance(obs_names, str):
                raise TypeError("'obs_paths' must be a string if 'obs_names' is a string.")
            if len(obs_paths) != len(obs_names):
                raise ValueError("'obs_paths' and 'obs_names' must have the same length.")


        # Validate year range
        for dataset in data_plot:
            start_year, end_year = data_general.validate_year_range(
                dataset, start_year, end_year, process_name="time series"
            )

        # Filter simulation data to the selected year range
        data_plot_filtered = copy.deepcopy(data_plot)
        for i, dataset in enumerate(data_plot):
            data_plot_filtered[i].data = dataset.data.sel(time=slice(str(start_year), str(end_year)))

        # Load observations for the selected year range if requested
        if obs:
            data_obs = [
                ObservationData(path, data_plot_filtered[0].data[[var_name]], name)
                for path, name in zip(obs_paths, obs_names)
            ]
            data_plot_filtered.extend(copy.deepcopy(data_obs))
            data_names.extend(obs_names)


        # Prepare time frequency for resampling
        if time_freq == "annual":
            time_freq_unit = "1YS"
        elif time_freq == "monthly":
            time_freq_unit = "1MS"
        elif time_freq == "daily":
            time_freq_unit = "1D"
        else:
            raise ValueError(
                "Incorrect time frequency, supported values are 'annual', 'monthly' and 'daily'"
            )

        # Compute and plot time series
        time_series = self._compute_time_series(var_name, data_plot_filtered, time_freq_unit)
        time_series_plot, _ = plots_general.plot_time_series(
            time_series,
            title=f"{time_freq.capitalize()} mean time series of {self.variables[var_name]['long_name']}",
            y_label=f"{var_name} ({self.variables[var_name]['units']})",
            labels=data_names,
            time_freq=time_freq,
            start_year=start_year,
            end_year=end_year,
            plot_ens=plot_ens,
        )

        # Save plot to path if given
        data_names_str = "_".join([name.replace(" ", "-") for name in data_names])
        ens_suffix = "_all_members" if plot_ens else ""
        plots_general.save_or_show_plot(
            time_series_plot,
            output_path,
            plot_filename=f"{time_freq}_time_series_{var_name}_{data_names_str}_{start_year}-{end_year}{ens_suffix}",
            plot_name=f"{time_freq.capitalize()} mean time series plot",
        )

        return


    def abs_diff_plot(self, var_name, data_names=None, output_path=None, start_year=None, end_year=None,
                      clon=0, cbar_limit=None):
        """
        Generate absolute difference plot for the given datasets and variable.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_names : list[str], optional
            List of names of two simulation ensembles to compare. If None, the first two datasets
            in the diagnostics object are used.
        output_path : str, optional
            Path to save the spatial plots.
        start_year : int
            Start year to plot.
        end_year : int
            End year to plot.
        clon : int
            Central longitude for the spatial map (default: 0).
        cbar_limit : float, optional
            Limit for the colorbar in the spatial plot. If None, it is taken as the
            maximum absolute value in the plot.
        """

        # Validate inputs
        if data_names is None:
            if len(self.datasets) < 2:
                raise ValueError(
                    "At least two datasets are required for spatial plots. Please add more datasets."
                )
            data_plot = [self.datasets[0], self.datasets[1]]
            data_names = [ds.name for ds in data_plot]
        elif (
            isinstance(data_names, list)
            and len(data_names) == 2
            and all(isinstance(name, str) for name in data_names)
        ):
            existing_names = [ds.name for ds in self.datasets]
            missing_names = [name for name in data_names if name not in existing_names]
            if missing_names:
                raise ValueError(
                    f"The following dataset names were not found in the DataDiagnostics object: {missing_names}."
                )

            data_plot = [next(ds for ds in self.datasets if ds.name == name) for name in data_names]
        else:
            raise TypeError(
                "'data_names' must be a list of two strings representing dataset names."
            )

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset {dataset.name}. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )

        # Validate year range
        for dataset in data_plot:
            start_year, end_year = data_general.validate_year_range(
                dataset, start_year, end_year, process_name="absolute difference"
            )
        data_plot_filtered = copy.deepcopy(data_plot)
        for i, dataset in enumerate(data_plot):
            data_plot_filtered[i].data = dataset.data.sel(time=slice(str(start_year), str(end_year)))

        # Compute and plot absolute difference
        abs_diff = self._compute_abs_diff(var_name, data_plot_filtered)
        if cbar_limit is None:
            cbar_limit = np.max(np.abs(abs_diff.values))
        levels = np.linspace(-cbar_limit, cbar_limit, 13)
        year_range = f"{start_year}-{end_year}"
        title_plot = (
            f"Difference in {self.variables[var_name]['long_name']} "
            f"for {year_range} ({data_names[0]} - {data_names[1]})"
        )

        abs_diff_plot, _ = plots_general.plot_spatial(
            abs_diff,
            clon=clon,
            title=title_plot,
            cb_label=f"difference in {var_name} ({self.variables[var_name]['units']})",
            cmap=cmocean.cm.thermal_r,
            levels=levels,
        )

        # Save plot to path if given
        data_names_str = "_".join([name.replace(" ", "-") for name in data_names])
        plots_general.save_or_show_plot(
            abs_diff_plot,
            output_path,
            plot_filename=f"abs_diff_{var_name}_{data_names_str}_{year_range}_clon_{clon}",
            plot_name="Absolute difference plot",
        )

        return


    def eff_size_plot(self, var_name, data_names=None, output_path=None, start_year=None, end_year=None,
                      clon=0, alpha=0.05, stat=ttest_ind,):
        """
        Generate effect size plot for the given datasets and variable marking
        grid points with statistically significant differences.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_names : list[str], optional
            List of names of two simulation ensembles to compare. If None, the first two datasets
            in the diagnostics object are used.
        output_path : str, optional
            Path to save the spatial plots.
        start_year : int
            Start year to plot.
        end_year : int
            End year to plot.
        clon : int
            Central longitude for the spatial map (default: 0).
        alpha : float
            Significance level for the statistical test (default: 0.05).
        stat : Callable
            Statistical test function to use for significance testing (default: ttest_ind).
        """

        # Validate inputs
        if data_names is None:
            if len(self.datasets) < 2:
                raise ValueError(
                    "At least two datasets are required for spatial plots. Please add more datasets."
                )
            data_plot = [self.datasets[0], self.datasets[1]]
            data_names = [ds.name for ds in data_plot]
        elif (
            isinstance(data_names, list)
            and len(data_names) == 2
            and all(isinstance(name, str) for name in data_names)
        ):
            existing_names = [ds.name for ds in self.datasets]
            missing_names = [name for name in data_names if name not in existing_names]
            if missing_names:
                raise ValueError(
                    f"The following dataset names were not found in the DataDiagnostics object: {missing_names}."
                )

            data_plot = [next(ds for ds in self.datasets if ds.name == name) for name in data_names]
        else:
            raise TypeError("'data_names' must be a list of two strings representing dataset names.")

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(
                    f"Variable '{var_name}' not found in the simulated dataset {dataset.name}. "
                    f"Available variables: {list(dataset.data.data_vars.keys())}"
                )
            if "realization" not in dataset.data.coords:
                raise ValueError(
                    f"Dataset '{dataset.name}' must contain a 'realization' coordinate for ensemble computations."
                )
        if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
            raise TypeError("The significance level 'alpha' must be a numeric value between 0 and 1.")
        if not callable(stat):
            raise TypeError("'stat' must be callable.")

        # Validate year range
        for dataset in data_plot:
            start_year, end_year = data_general.validate_year_range(
                dataset, start_year, end_year, process_name="effect size"
            )
        data_plot_filtered = copy.deepcopy(data_plot)
        for i, dataset in enumerate(data_plot):
            data_plot_filtered[i].data = dataset.data.sel(time=slice(str(start_year), str(end_year)))


        # Compute and plot effect size with significant differences
        eff_size = self._compute_eff_size_ens(var_name, data_plot_filtered)
        significant = self._compute_significant_diff(var_name, data_plot_filtered, alpha, stat)
        levels = [-2,-1.2,-0.8,-0.5,-0.2,-0.01,0.01,0.2,0.5,0.8,1.2,2.0]    # Use Cohen's limits for effect size
        year_range = f"{start_year}-{end_year}"
        title_plot = (
            f"Effect size ($d$) for {self.variables[var_name]['long_name']} "
            f"for {year_range} ({data_names[0]} - {data_names[1]})"
        )

        eff_size_plot, _ = plots_general.plot_spatial(
            eff_size,
            clon=clon,
            title=title_plot,
            cb_label=f"$d$ for {var_name} (-)",
            cmap=cmocean.cm.diff,
            levels=levels,
            significant=significant,
        )

        # Save plot to path if given
        data_names_str = "_".join([name.replace(" ", "-") for name in data_names])
        plots_general.save_or_show_plot(
            eff_size_plot,
            output_path,
            plot_filename=f"eff_size_{var_name}_{data_names_str}_{year_range}_clon_{clon}",
            plot_name="Effect size plot",
        )

        return


    def bias_plot(self, var_name, data_name=None, output_path=None, obs_path=None, obs_name=None,
                  start_year=None, end_year=None, clon=0, cbar_limit=None):
        """
        Generate bias plot for the given dataset and variable comparing with observations.

        Parameters
        ----------
        var_name : str
            Climate variable name.
        data_name : str, optional
            Name of the simulation ensemble to plot. If None, the first dataset
            in the DataDiagnostics object is used.
        output_path : str, optional
            Path to save the spatial plot.
        obs_path : str
            Path to the observations database.
        obs_name : str
            Name of the observational dataset.
        start_year : int
            Start year to plot.
        end_year : int
            End year to plot.
        clon : int
            Central longitude for the spatial map (default: 0).
        cbar_limit : float, optional
            Limit for the colorbar in the spatial plot. If None, it is taken as the
            maximum absolute value in the plot.
        """

        # Validate inputs
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError(
                    "At least one dataset is required for bias plots. Please add a dataset."
                )
            data_plot = [self.datasets[0]]
            data_name = data_plot[0].name
        elif isinstance(data_name, str):
            data_plot = [ds for ds in self.datasets if ds.name == data_name]
            if not data_plot:
                raise ValueError(
                    f"Dataset with name '{data_name}' not found in the DataDiagnostics object."
                )
            if len(data_plot) > 1:
                raise ValueError(
                    f"Multiple datasets with name '{data_name}' found in the DataDiagnostics object."
                )
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")

        if var_name not in data_plot[0].data.data_vars:
            raise ValueError(
                f"Variable '{var_name}' not found in the simulated dataset '{data_plot[0].name}'. "
                f"Available variables: {list(data_plot[0].data.data_vars.keys())}"
            )

        if obs_path is None or obs_name is None:
            raise ValueError(
                "Automatic selection of observations is not implemented yet. "
                "Please provide a path and a name for the observations database."
            )
        elif not isinstance(obs_path, str) or not isinstance(obs_name, str):
            raise TypeError(
                "'obs_path' and 'obs_name' must be strings representing the observations "
                "database path and name, respectively."
            )


        # Validate year range
        for dataset in data_plot:
            start_year, end_year = data_general.validate_year_range(
                dataset, start_year, end_year, process_name="bias"
            )
        data_plot_filtered = copy.deepcopy(data_plot)
        for i, dataset in enumerate(data_plot):
            data_plot_filtered[i].data = dataset.data.sel(time=slice(str(start_year), str(end_year)))

        # Load observations for the selected year range
        data_obs = ObservationData(obs_path, data_plot_filtered[0].data[[var_name]], obs_name)
        data_plot_filtered.append(copy.deepcopy(data_obs))
        data_names = [data_name, obs_name]


        # Compute and plot bias
        bias = self._compute_bias(var_name, data_plot_filtered)
        if cbar_limit is None:
            cbar_limit = np.max(np.abs(bias.values))
        levels = np.linspace(-cbar_limit, cbar_limit, 13)
        colors = ("OrangeBlue", ["#E66101", "white", "#0072B2"])
        cmap = LinearSegmentedColormap.from_list(*colors)
        year_range = f"{start_year}-{end_year}"
        title_plot = (
            f"Bias in {self.variables[var_name]['long_name']} for {year_range} ({data_names[0]} - {data_names[1]})"
        )

        bias_plot, _ = plots_general.plot_spatial(
            bias,
            clon=clon,
            title=title_plot,
            cb_label=f"bias in {var_name} ({self.variables[var_name]['units']})",
            cmap=cmap,
            levels=levels,
        )

        # Save plot to path if given
        data_names_str = "_".join([name.replace(" ", "-") for name in data_names])
        plots_general.save_or_show_plot(
            bias_plot,
            output_path,
            plot_filename=f"bias_{var_name}_{data_names_str}_{year_range}_clon_{clon}",
            plot_name="Bias plot",
        )

        return
