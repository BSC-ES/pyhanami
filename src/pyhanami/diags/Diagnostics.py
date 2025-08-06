import cmocean
import warnings
warnings.simplefilter("always")

import numpy as np
import xarray as xr
import concurrent.futures
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from scipy.stats import ttest_ind
from collections.abc import Iterable

from pyhanami import config
from pyhanami.utils import plot, statistics
from pyhanami.diags.Simulations import SimulationData


class DataDiagnostics:
    """
    Perform diagnostic comparisons between climate simulation ensembles.

    This class provides functionality for computing and visualizing differences in climate variables
    between simulation ensembles. It includes methods for computing annual time series, absolute
    differences, effect sizes and significance differences at grid point level.
    
    Parameters
    ----------
    datasets : SimulationData or Iterable[SimulationData]
        Ensemble or list of ensembles containing simulation data and metadata.

    Attributes
    ----------
    datasets : list[SimulationData]
        List of ensembles containing simulation data and metadata.
    variables : dict
        Configuration dictionary mapping variable names to display metadata.
    max_workers_grid : int
        Number of parallel workers used for grid-level computations.

    Methods
    -------
    _compute_time_series_annual(var_name)
        Computes annual spatial mean time series for the specified variable from both ensembles.

    _compute_abs_diff(var_name)
        Computes the absolute difference between the ensembles at the grid point level.

    _compute_eff_size_ens(var_name)
        Computes the effect size (Cohen's d) between the ensembles in parallel at the grid point level.

    _compute_significant_diff(var_name, alpha, stat)
        Computes statistically significant differences between the ensembles in parallel at the grid point level.

    time_series_plots(var_name, output_path)
        Generates and saves a plot of the annual time series for the specified variable.

    spatial_plots(var_name, output_path, alpha=0.05, stat=ttest_ind)
        Generates and saves spatial plots of the absolute difference, the effect size and the significance difference for the specified variable.
    """

    def __init__(self, datasets: Iterable[SimulationData] = None):
        if datasets is None:
            self.datasets = []
        elif isinstance(datasets, SimulationData):
            self.datasets = [datasets]
        elif isinstance(datasets, Iterable) and not isinstance(datasets, (str, bytes)) \
            and all(isinstance(ds, SimulationData) for ds in datasets):
                self.datasets = list(datasets)
        else:
            raise TypeError("Input must be a SimulationData object or an iterable of SimulationData objects.")

        # Load config parameters once
        self.variables = config.VARIABLES
        self.max_workers_grid = config.MAX_WORKERS_GRID


    def _compute_time_series_annual(self, var_name, data_plot):
        """ 
        Compute time series for the given simulation ensembles and variable. 
        
        Parameters
        ----------
        var_name (str): Climate variable name.
        data_plot (list[SimulationData]): List of simulation ensembles to compute time series for.
        
        Returns
        -------
        time_series (list[xr.DataArray]): List of annual mean time series.
        """

        # Validate inputs
        if not isinstance(data_plot, list) or len(data_plot) == 0 \
            or not all(isinstance(ds, SimulationData) for ds in data_plot):
            raise TypeError("'data_plot' must be a non-empty list of SimulationData instances.")
        
        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                            f"Available variables: {list(dataset.data.data_vars.keys())}")
            

        # Compute annual mean time series for each dataset
        time_series = []
        for data_sim in [ds.data for ds in data_plot]:
            data_var = data_sim[var_name]
            weights = statistics.area_weights(data_var)
            data_weighted = data_var.weighted(weights)

            data_area_mean = data_weighted.mean(['lat', 'lon'])
            data_time_mean = data_area_mean.resample(time='1YE').mean().compute()
            time_series.append(data_time_mean)
        
        return time_series


    def _compute_abs_diff(self, var_name, data_plot):
        """ 
        Compute absolute average difference between the simulation ensembles 
        for the given variable and time range at the grid point level. 
        
        Parameters
        ----------
        var_name (str): Climate variable name.
        data_plot (list[SimulationData]): List of two simulation ensembles to compute the absolute difference for.

        Returns
        -------
        data_diff (xr.DataArray): Absolute difference between the two ensembles.
        """
        
        # Validate inputs
        if not isinstance(data_plot, list) or len(data_plot) == 0 \
            or not all(isinstance(ds, SimulationData) for ds in data_plot):
            raise TypeError("'data_plot' must be a non-empty list of SimulationData instances.")
        
        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                            f"Available variables: {list(dataset.data.data_vars.keys())}")
            
        data_sim_1 = data_plot[0].data.persist()
        data_sim_2 = data_plot[1].data.persist()
        if not data_sim_1.time.equals(data_sim_2.time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n"
                f"  {data_plot[0].name} has time from {data_sim_1.time.min().item()} to {data_sim_1.time.max().item()}\n"
                f"  {data_plot[1].name} has time from {data_sim_2.time.min().item()} to {data_sim_2.time.max().item()}"
            )


        # Compute mean absolute difference
        data_sim_mean_1 = data_sim_1[var_name].mean(['realization','time'])
        data_sim_mean_2 = data_sim_2[var_name].mean(['realization','time'])

        data_diff = (data_sim_mean_1 - data_sim_mean_2).compute()
        if var_name in ['siconc', 'sos', 'tos']:
            data_diff = xr.where((np.isnan(data_diff)) | (data_diff==0), 10**-6, data_diff)    # Values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0

        print(f"Computed absolute difference for variable '{var_name}' between '{data_plot[0].name}' and '{data_plot[1].name}'.", flush=True)
        return data_diff


    def _compute_eff_size_ens(self, var_name, data_plot):
        """ 
        Compute average effect size (Cohen's d) between the simulation ensembles 
        in parallel for the given variable at the grid point level. 
        
        Parameters
        ----------
        var_name (str): Climate variable name.
        data_plot (list[SimulationData]): List of two simulation ensembles to compute the absolute difference for.

        Returns
        -------
        data_effect_size (xr.DataArray): Effect size between the two ensembles.
        """
        
        # Validate inputs
        if not isinstance(data_plot, list) or len(data_plot) == 0 \
            or not all(isinstance(ds, SimulationData) for ds in data_plot):
            raise TypeError("'data_plot' must be a non-empty list of SimulationData instances.")
        
        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                            f"Available variables: {list(dataset.data.data_vars.keys())}")
            
        data_sim_1 = data_plot[0].data.persist()
        data_sim_2 = data_plot[1].data.persist()
        if not data_sim_1.time.equals(data_sim_2.time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n"
                f"  {data_plot[0].name} has time from {data_sim_1.time.min().item()} to {data_sim_1.time.max().item()}\n"
                f"  {data_plot[1].name} has time from {data_sim_2.time.min().item()} to {data_sim_2.time.max().item()}"
            )
        

        # Prepare data
        data_sim_flat_1 = data_sim_1[var_name].mean('time').stack(ngrid = ['lat','lon']).compute()
        data_sim_flat_2 = data_sim_2[var_name].mean('time').stack(ngrid = ['lat','lon']).compute()
        
        #  Compute effect sizes in parallel
        tasks = [(data_sim_flat_1.sel(ngrid=i).values, data_sim_flat_2.sel(ngrid=i).values) for i in data_sim_flat_1.ngrid]
        effect_size = np.empty(len(tasks))

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers_grid, mp_context=mp.get_context("spawn")) as executor:
            for idx, value in enumerate(tqdm(executor.map(statistics.cp_effect_size_bootstrap, tasks), total=len(tasks), desc=f"Computing effect sizes for variable '{var_name}'")):
                effect_size[idx] = value

        # Convert to xarray.DataArray
        effect_size_reshaped = effect_size.reshape(data_sim_1.sizes['lat'], data_sim_1.sizes['lon'])

        data_effect_size = xr.DataArray(
            effect_size_reshaped,
            dims=["lat", "lon"],
            coords={
                "lat": data_sim_1["lat"],
                "lon": data_sim_1["lon"]
            },
            name=var_name
        )

        print(f"Computed effect size for variable '{var_name}' between '{data_plot[0].name}' and '{data_plot[1].name}'.", flush=True)
        return data_effect_size


    def _compute_significant_diff(self, var_name, data_plot, alpha=0.05, stat=ttest_ind):
        """ 
        Compute significant difference between the simulation ensembles
        in parallel for the given variable at the grid point level. 
        
        Parameters
        ----------
        var_name (str): Climate variable name.
        data_plot (list[SimulationData]): List of two simulation ensembles to compute the absolute difference for.
        alpha (float): Significance level for the statistical test.
        stat (Callable): Statistical test function to use.

        Returns
        -------
        significant (np.ndarray): Boolean array indicating significant differences between the two ensembles.
        """
        
        # Validate input
        if not isinstance(data_plot, list) or len(data_plot) == 0 \
            or not all(isinstance(ds, SimulationData) for ds in data_plot):
            raise TypeError("'data_plot' must be a non-empty list of SimulationData instances.")
        
        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                            f"Available variables: {list(dataset.data.data_vars.keys())}")
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"The significance level 'alpha' must be numeric.")
        if not (0 <= alpha <= 1):
            raise ValueError(f"'alpha' must be between 0 and 1.")
        if not callable(stat):
            raise TypeError(f"'stat' must be callable.")         

        data_sim_1 = data_plot[0].data.persist()
        data_sim_2 = data_plot[1].data.persist()
        if not data_sim_1.time.equals(data_sim_2.time):
            raise ValueError(
                f"Time coordinates of the two datasets do not match:\n"
                f"  {data_plot[0].name} has time from {data_sim_1.time.min().item()} to {data_sim_1.time.max().item()}\n"
                f"  {data_plot[1].name} has time from {data_sim_2.time.min().item()} to {data_sim_2.time.max().item()}"
            )
        
        # Prepare data
        n_lats = data_sim_1.sizes['lat']
        n_lons = data_sim_1.sizes['lon']
        n_points =  n_lats*n_lons
        n_realizations = data_sim_1.sizes['realization']
        
        data_mean_1 = data_sim_1.mean('time')
        data_mean_var_1 = data_mean_1[var_name].data
        data_mean_flat_1 = data_mean_var_1.compute().reshape((n_realizations,n_points))

        data_mean_2 = data_sim_2.mean('time')
        data_mean_var_2 = data_mean_2[var_name].data
        data_mean_flat_2 = data_mean_var_2.compute().reshape((n_realizations,n_points))
        
        d1, d2 = (data_mean_flat_1, data_mean_flat_2)

        # Compute significant differences in parallel
        tasks = [(d1[:,n], d2[:,n], alpha, stat) for n in range(n_points)]
        significant = np.zeros(n_points)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers_grid, mp_context=mp.get_context("spawn")) as executor:
            for idx, value in enumerate(executor.map(statistics.significant_diff, tasks)):
                significant[idx] = value
        significant_reshaped =  significant.reshape((n_lats,n_lons))
        
        print(f"Computed significant difference for variable '{var_name}' between '{data_plot[0].name}' and '{data_plot[1].name}'.", flush=True)
        return significant_reshaped


    def add_datasets(self, datasets):
        """ 
        Add new datasets to the DataDiagnostics object.

        Parameters
        ----------
        datasets (SimulationData or Iterable[SimulationData]): Ensemble or list of ensembles containing simulation 
                                                                data and metadata to add.
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
                warnings.warn(f"Dataset with name '{dataset.name}' already exists in the DataDiagnostics object. Skipping addition.")

        return
    
    
    def time_series_plots(self, var_name, data_names=None, output_path=None):
        """ 
        Generate time series plot for the given ensembles and variable. When no ensembles
        are specified, all datasets in the diagnostics object are used.
        
        Parameters
        ----------
        var_name (str): Climate variable name.
        data_names (str or list[str]): Name or list of names of simulation ensembles to plot.
        output_path (str): Path to save the time series plot.  
        """
        
        # Validate inputs
        if data_names is None:
            data_plot = self.datasets
            data_names = [ds.name for ds in data_plot]
        elif isinstance(data_names, str):
            data_plot = [ds for ds in self.datasets if ds.name == data_names]
            if not data_plot:
                raise ValueError(f"Dataset with name '{data_names}' not found in the DataDiagnostics object.")
        elif isinstance(data_names, list) and all(isinstance(name, str) for name in data_names):
            existing_names = [ds.name for ds in self.datasets]
            missing_names = [name for name in data_names if name not in existing_names]
            if missing_names:
                raise ValueError(f"The following dataset names were not found in the DataDiagnostics object: {missing_names}.")
            
            data_plot = [ds for ds in self.datasets if ds.name in data_names]
        else:
            raise TypeError("'data_names' must be a string or a list of strings representing dataset names.")

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{dataset.name}'. "
                            f"Available variables: {list(dataset.data.data_vars.keys())}")


        # Compute and plot time series
        time_series = self._compute_time_series_annual(var_name, data_plot)
        time_series_plot, _ = plot.time_series_plot(time_series, title=f'Annual mean time series of {self.variables[var_name][0]}',
                                                 y_label=f'{var_name} ({self.variables[var_name][1]})', labels=data_names)
        
        # Save plot to path if given
        if output_path is None:
            plt.show()
            print("Time series plot created and displayed.", flush=True)
        else:
            output_path = Path(output_path)
            if not output_path.suffix:
                output_path.mkdir(parents=True, exist_ok=True)
                data_names_str = "-".join(data_names)
                time_series_path = output_path / f"time_series_{var_name}_{data_names_str}.png"
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                time_series_path = output_path
            
            time_series_plot.savefig(time_series_path, bbox_inches='tight', dpi=100)
            print(f"Time series plot created and saved to '{time_series_path}'.", flush=True)
            
        return


    def spatial_plots(self, var_name, data_names=None, output_path=None, alpha=0.05, stat=ttest_ind):
        """ 
        Generate absolute difference and effect size plots for the given
        ensembles and variable. 
        
        Parameters
        ----------
        var_name (str): Climate variable name.
        data_names (list[str]): List of names of two simulation ensembles to compare. If None, the first two datasets
                                 in the diagnostics object are used.
        output_path (str): Path to save the spatial plots.
        alpha (float): Significance level for the statistical test.
        stat (Callable): Statistical test function to use for significance testing.
        """
        
        # Validate inputs
        if data_names is None:
            if len(self.datasets) < 2:
                raise ValueError("At least two datasets are required for spatial plots. Please add more datasets.")
            data_plot = [self.datasets[0], self.datasets[1]]
            data_names = [ds.name for ds in data_plot]
        elif isinstance(data_names, list) and len(data_names) == 2 \
            and all(isinstance(name, str) for name in data_names):
            existing_names = [ds.name for ds in self.datasets]
            missing_names = [name for name in data_names if name not in existing_names]
            if missing_names:
                raise ValueError(f"The following dataset names were not found in the DataDiagnostics object: {missing_names}.")
            
            data_plot = [ds for ds in self.datasets if ds.name in data_names]
        else:
            raise TypeError("'data_names' must be a list of two strings representing dataset names.")

        for dataset in data_plot:
            if var_name not in dataset.data.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in the simulated dataset {dataset.name}. "
                                 f"Available variables: {list(dataset.data.data_vars.keys())}")
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"The significance level 'alpha' must be numeric.")
        if not (0 <= alpha <= 1):
            raise ValueError(f"'alpha' must be between 0 and 1.")
        if not callable(stat):
            raise TypeError(f"'stat' must be callable.")
        

        # Prepare output path if given
        if output_path is not None:
            output_path = Path(output_path)
            if output_path.suffix != '':  
                raise ValueError("Output path must be a directory, not a file path, as two output files will be created.")
            
            output_path.mkdir(parents=True, exist_ok=True)
            data_names_str = "-".join(data_names)
            abs_diff_path = output_path / f"abs_diff_{var_name}_{data_names_str}.png"
            eff_size_path = output_path / f"eff_size_{var_name}_{data_names_str}.png"


        # Compute and plot absolute difference
        abs_diff = self._compute_abs_diff(var_name, data_plot)
        limit = np.max(np.abs(abs_diff.values))
        levels = np.linspace(-limit, limit, 13)
        
        abs_diff_plot, _ = plot.spatial_plot(abs_diff, title=f'Difference in {self.variables[var_name][0]} ({data_plot[0].name} - {data_plot[1].name})',
                                          cb_label=f"difference in {var_name} ({self.variables[var_name][1]})", cmap=cmocean.cm.thermal, levels=levels)
        
        if output_path is None:
            plt.show()
            print("Absolute difference plot created and displayed.", flush=True)
        else:
            abs_diff_plot.savefig(abs_diff_path, bbox_inches='tight', dpi=100)
            print(f"Absolute difference plot created and saved to '{abs_diff_path}'.\n", flush=True)


        # Compute and plot effect size with significant differences
        eff_size = self._compute_eff_size_ens(var_name, data_plot)
        significant = self._compute_significant_diff(var_name, data_plot, alpha, stat)
        levels = [-2,-1.2,-0.8,-0.5,-0.2,-0.01,0.01,0.2,0.5,0.8,1.2,2.0]    # Use Cohen's limits for effect size

        eff_size_plot, _ = plot.spatial_plot(eff_size, title=f"Cohen's effect size ($d$) for {self.variables[var_name][0]} ({data_plot[0].name} - {data_plot[1].name})",
                                          cb_label=f"$d$ for {var_name} (-)", cmap=cmocean.cm.diff, levels=levels, significant=significant)

        if output_path is None:
            plt.show()
            print("Effect size plot created and displayed.", flush=True)
        else:
            eff_size_plot.savefig(eff_size_path, bbox_inches='tight', dpi=100)
            print(f"Effect size plot created and saved to '{eff_size_path}'.\n", flush=True)
            
        return