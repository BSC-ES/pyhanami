import cmocean
import numpy as np
import xarray as xr
import concurrent.futures

from tqdm import tqdm
from pathlib import Path
from typing import Callable
from scipy.stats import ttest_ind

from pyhanami import config
from pyhanami.utils import plot, statistics
from pyhanami.diags.Simulations import SimulationData


class DataDiagnostics:
    """
    Perform diagnostic comparisons between two climate simulation ensembles.

    This class provides functionality for computing and visualizing differences in climate variables
    between a reference and a test simulation ensembles. It includes methods for computing annual 
    time series, absolute differences, effect sizes and significance differences at grid point level.
    
    Parameters
    ----------
    ref : SimulationData
        Reference ensemble containing simulation data and metadata.
    test : SimulationData
        Test ensemble containing simulation data and metadata.

    Attributes
    ----------
    ref : SimulationData
        Instance containing the reference ensemble and metadata.
    test : SimulationData
        Instance containing the test ensemble and metadata.
    variables : dict
        Configuration dictionary mapping variable names to display metadata.
    max_workers_grid : int
        Number of parallel workers used for grid-level computations.

    Methods
    -------
    _compute_time_series_annual(varname)
        Computes annual spatial mean time series for the specified variable from both ensembles.

    _compute_abs_diff(varname)
        Computes the absolute difference between the ensembles at the grid point level.

    _compute_eff_size_ens(varname)
        Computes the effect size (Cohen's d) between the ensembles in parallel at the grid point level.

    _compute_significant_diff(varname, alpha, stat)
        Computes statistically significant differences between the ensembles in parallel at the grid point level.

    time_series_plots(varname, output_path)
        Generates and saves a plot of the annual time series for the specified variable.

    spatial_plots(varname, output_path, alpha=0.05, stat=ttest_ind)
        Generates and saves spatial plots of the absolute difference, the effect size and the significance difference for the specified variable.
    """

    def __init__(self, ref: SimulationData, test: SimulationData):
        self.ref = ref
        self.test = test

        # Load config parameters once
        self.variables = config.VARIABLES
        self.max_workers_grid = config.MAX_WORKERS_GRID


    def _compute_time_series_annual(self, varname):
        """ 
        Compute time series for the given simulation ensembles and variable. 
        
        Parameters
        ----------
        varname (str): Climate variable name.
        
        Returns
        -------
        time_series (list[xr.DataArray]): List of annual mean time series.
        """

        time_series = []
        for data_sim in [self.ref.data, self.test.data]:
            data_var = data_sim[varname]
            weights = statistics.area_weights(data_var)
            data_weighted = data_var.weighted(weights)

            data_area_mean = data_weighted.mean(['lat', 'lon'])
            data_time_mean = data_area_mean.resample(time='1YE').mean()
            time_series.append(data_time_mean)

        return time_series


    def _compute_abs_diff(self, varname):
        """ 
        Compute absolute average difference between the simulation ensembles 
        for the given variable at the grid point level. 
        
        Parameters
        ----------
        varname (str): Climate variable name.

        Returns
        -------
        data_diff (xr.DataArray): Absolute difference between the two ensembles.
        """

        data_sim_1 = self.ref.data[varname].mean(['realization','time'])
        data_sim_2 = self.test.data[varname].mean(['realization','time'])

        data_diff = data_sim_1 - data_sim_2
        if varname in ['siconc', 'sos', 'tos']:
            data_diff = xr.where((np.isnan(data_diff)) | (data_diff==0), 10**-6, data_diff)    # Values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0

        print(f"Computed absolute difference for variable '{varname}' between '{self.ref.name}' and '{self.test.name}'.", flush=True)
        return data_diff


    def _compute_eff_size_ens(self, varname):
        """ 
        Compute average effect size (Cohen's d) between the simulation ensembles 
        in parallel for the given variable at the grid point level. 
        
        Parameters
        ----------
        varname (str): Climate variable name.

        Returns
        -------
        data_effect_size (xr.DataArray): Effect size between the two ensembles.
        """
    
        # Prepare data
        data_sim_1 = self.ref.data.persist()
        data_sim_2 = self.test.data.persist()

        data_sim_flat_1 = data_sim_1[varname].mean('time').stack(ngrid = ['lat','lon']).load()
        data_sim_flat_2 = data_sim_2[varname].mean('time').stack(ngrid = ['lat','lon']).load()

        #  Compute effect sizes in parallel
        tasks = [(data_sim_flat_1.sel(ngrid=i).values, data_sim_flat_2.sel(ngrid=i).values,) for i in data_sim_flat_1.ngrid]
        effect_size = np.empty(len(tasks))

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers_grid) as executor:
            for idx, value in enumerate(tqdm(executor.map(statistics.cp_effect_size_bootstrap, tasks), total=len(tasks), desc=f"Computing effect sizes for variable '{varname}'")):
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
            name=varname
        )

        print(f"Computed effect size for variable '{varname}' between '{self.ref.name}' and '{self.test.name}'.", flush=True)
        return data_effect_size


    def _compute_significant_diff(self, varname, alpha, stat):
        """ 
        Compute significant difference between the simulation ensembles
        in parallel for the given variable at the grid point level. 
        
        Parameters
        ----------
        varname (str): Climate variable name.
        alpha (float): Significance level for the statistical test.
        stat (Callable): Statistical test function to use.

        Returns
        -------
        significant (np.ndarray): Boolean array indicating significant differences between the two ensembles.
        """

        data_sim_1 = self.ref.data.persist()
        data_sim_2 = self.test.data.persist()

        # Prepare data
        n_lats = data_sim_1.sizes['lat']
        n_lons = data_sim_1.sizes['lon']
        n_points =  n_lats*n_lons
        n_realizations = data_sim_1.sizes['realization']

        data_mean_1 = data_sim_1.mean('time').compute()
        data_mean_var_1 = data_mean_1[varname].data
        data_mean_flat_1 = data_mean_var_1.reshape((n_realizations,n_points))

        data_mean_2 = data_sim_2.mean('time').compute()
        data_mean_var_2 = data_mean_2[varname].data
        data_mean_flat_2 = data_mean_var_2.reshape((n_realizations,n_points))

        d1, d2 = (data_mean_flat_1, data_mean_flat_2)

        # Compute significant differences in parallel
        tasks = [(d1[:,n], d2[:,n], alpha, stat) for n in range(n_points)]
        significant = np.zeros(n_points)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers_grid) as executor:
            for idx, value in enumerate(executor.map(statistics.significant_diff, tasks)):
                significant[idx] = value
        significant_reshaped =  significant.reshape((n_lats,n_lons))

        print(f"Computed significant difference for variable '{varname}' between '{self.ref.name}' and '{self.test.name}'.", flush=True)
        return significant_reshaped


    def time_series_plots(self, varname, output_path):
        """ 
        Generate time series plot for the given ensembles and variable. 
        
        Parameters
        ----------
        varname (str): Climate variable name.
        output_path (str): Path to save the time series plot.  
        """

        # Validate inputs
        if varname not in self.ref.data.data_vars:
            raise ValueError(f"Variable '{varname}' not found in the simulated dataset {self.ref.name}. "
                        f"Available variables: {list(self.ref.data.data_vars.keys())}")
        if varname not in self.test.data.data_vars:
            raise ValueError(f"Variable '{varname}' not found in the simulated dataset {self.test.name}."
                        f"Available variables: {list(self.test.data.data_vars.keys())}")

        # Prepare output path
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            time_series_path = output_path / f"time_series_{varname}_{self.ref.name}-{self.test.name}.png"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            time_series_path = output_path

        # Compute and plot time series
        time_series = self._compute_time_series_annual(varname)
        time_series_plot, _ = plot.time_series_plot(time_series, title=f'Annual mean time series of {self.variables[varname][0]}',
                                                 y_label=f'{varname} ({self.variables[varname][1]})', labels=[self.ref.name, self.test.name])
        time_series_plot.savefig(time_series_path, bbox_inches='tight', dpi=100)

        print(f"Time series plot saved to '{time_series_path}'.", flush=True)
        return


    def spatial_plots(self, varname, output_path, alpha=0.05, stat=ttest_ind):
        """ 
        Generate absolute difference and effect size plots for the given
        ensembles and variable. 
        
        Parameters
        ----------
        varname (str): Climate variable name.
        output_path (str): Path to save the spatial plots.
        alpha (float): Significance level for the statistical test.
        stat (Callable): Statistical test function to use for significance testing.
        """

        # Validate inputs
        if varname not in self.ref.data.data_vars:
            raise ValueError(f"Variable '{varname}' not found in the simulated dataset {self.ref.name}. "
                        f"Available variables: {list(self.ref.data.data_vars.keys())}")
        if varname not in self.test.data.data_vars:
            raise ValueError(f"Variable '{varname}' not found in the simulated dataset {self.test.name}."
                        f"Available variables: {list(self.test.data.data_vars.keys())}")
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"The significance level 'alpha' must be numeric.")
        if not (0 <= alpha <= 1):
            raise ValueError(f"'alpha' must be between 0 and 1.")
        if not callable(stat):
            raise TypeError(f"'stat' must be callable.")

        # Prepare output path
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            abs_diff_path = output_path / f"abs_diff_{varname}_{self.ref.name}-{self.test.name}.png"
            eff_size_path = output_path / f"eff_size_{varname}_{self.ref.name}-{self.test.name}.png"
        else:
            raise ValueError("Output path must be a directory, not a file path, as two output files will be created.")

        # Compute and plot absolute difference
        abs_diff = self._compute_abs_diff(varname)
        limit = np.max(np.abs(abs_diff.values))
        levels = np.linspace(-limit, limit, 13)

        abs_diff_plot, _ = plot.spatial_plot(abs_diff, title=f'Difference in {self.variables[varname][0]} ({self.ref.name} - {self.test.name})',
                                          cb_label=f"difference in {varname} ({self.variables[varname][1]})", cmap=cmocean.cm.thermal, levels=levels)
        abs_diff_plot.savefig(abs_diff_path, bbox_inches='tight', dpi=100)
        print(f"Absolute difference plot saved to '{abs_diff_path}'.\n", flush=True)

        # Compute and plot effect size with significant differences
        eff_size = self._compute_eff_size_ens(varname)
        significant = self._compute_significant_diff(varname, alpha, stat)
        levels = [-2,-1.2,-0.8,-0.5,-0.2,-0.01,0.01,0.2,0.5,0.8,1.2,2.0]    # Use Cohen's limits for effect size

        eff_size_plot, _ = plot.spatial_plot(eff_size, title=f"Cohen's effect size ($d$) for {self.variables[varname][0]} ({self.ref.name} - {self.test.name})",
                                          cb_label=f"$d$ for {varname} (-)", cmap=cmocean.cm.diff, levels=levels, significant=significant)
        eff_size_plot.savefig(eff_size_path, bbox_inches='tight', dpi=100)
        print(f"Effect size plot saved to '{eff_size_path}'.", flush=True)

        return