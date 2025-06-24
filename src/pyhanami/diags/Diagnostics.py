import cmocean
import numpy as np
import xarray as xr
import concurrent.futures

from pathlib import Path
from typing import Callable
from scipy.stats import ttest_ind
from pyhanami import config
from pyhanami.utils import plot, statistics
from pyhanami.diags.Simulations import SimulationData


class DataDiagnostics:
    def __init__(self, ref: SimulationData, test: SimulationData):
        self.ref = ref
        self.test = test

        # Load config parameters once
        self.variables = config.VARIABLES


    def _compute_time_series(self, varname: str):
        """ Compute time series for the given ensembles. """
        raise NotImplementedError("This function is not implemented yet.")


    def _compute_abs_diff(self, varname: str) -> xr.DataArray:
        """ Compute absolute average difference between the simulation ensembles 
        for the given variable at the grid point level. """

        data_sim_1 = self.ref.data[varname].mean(['realization','time'])
        data_sim_2 = self.test.data[varname].mean(['realization','time'])

        data_diff = data_sim_1 - data_sim_2
        if varname in ['siconc', 'sos', 'tos']:
            data_diff = xr.where((np.isnan(data_diff)) | (data_diff==0), 10**-6, data_diff)    # Values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0

        return data_diff


    def _compute_eff_size_ens(self, varname: str) -> xr.DataArray:
        """ Compute average effect size between the simulation ensembles 
        in parallel for the given variable at the grid point level. """
    
        # Prepare data
        data_sim_1 = self.ref.data.persist()
        data_sim_2 = self.test.data.persist()

        data_sim_flat_1 = data_sim_1[varname].mean('time').stack(ngrid = ['lat','lon']).load()
        data_sim_flat_2 = data_sim_2[varname].mean('time').stack(ngrid = ['lat','lon']).load()

        #  Compute effect sizes in parallel
        tasks = [(data_sim_flat_1.sel(ngrid=i).values, data_sim_flat_2.sel(ngrid=i).values,) for i in data_sim_flat_1.ngrid]
        effect_size = np.empty(len(tasks))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for idx, value in enumerate(executor.map(statistics.cp_effect_size_bootstrap, tasks)):
                effect_size[idx] = value

        # Convert to xarrat.DataArray
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

        return data_effect_size


    def _compute_significant_diff(self, varname: str, alpha: float, stat: Callable) -> np.ndarray:
        """ Compute significant differences between the simulation ensembles
        in parallel for the given variable at the grid point level. """

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

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for idx, value in enumerate(executor.map(statistics.significant_diff, tasks)):
                significant[idx] = value
        significant_reshaped =  significant.reshape((n_lats,n_lons))

        return significant_reshaped


    def time_series_plots(self, varname: str, output_path: str):
        """ Generate time series plots for the given ensembles and variable. """

        # Prepare output folder
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        time_series_path = output_path / "time_series.png"

        # Compute and plot time series
        time_series = self._compute_time_series(varname)
        time_series_plot = plot.time_series_plot(time_series)
        time_series_plot.savefig(time_series_path, bbox_inches='tight', dpi=100)

        print(f'Time series plots saved to {output_path}.')
        return


    def spatial_plots(self, varname: str, output_path: str, alpha: float = 0.05, stat: Callable = ttest_ind):
        """ Generate absolute difference and effect size plots
        for the given ensembles and variable. """

        # prepare output folder
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        abs_diff_path = output_path / "abs_diff.png"
        eff_size_path = output_path / "eff_size.png"

        # Compute and plot absolute difference
        abs_diff = self._compute_abs_diff(varname)
        limit = np.max(np.abs(abs_diff.values))
        levels = np.linspace(-limit, limit, 13)

        abs_diff_plot, _ = plot.spatial_plot(abs_diff, title=f'Difference in {self.variables[varname][0]} ({self.ref.name} - {self.test.name})',
                                          cb_label=f"difference in {varname} ({self.variables[varname][1]})", cmap=cmocean.cm.thermal, levels=levels)
        abs_diff_plot.savefig(abs_diff_path, bbox_inches='tight', dpi=100)

        # Compute and plot effect size with significant differences
        eff_size = self._compute_eff_size_ens(varname)
        significant = self._compute_significant_diff(varname, alpha, stat)
        levels = [-2,-1.2,-0.8,-0.5,-0.2,-0.01,0.01,0.2,0.5,0.8,1.2,2.0]    # Use Cohen's limits for effect size

        eff_size_plot, _ = plot.spatial_plot(eff_size, title=f"Cohen's effect size ($d$) for {self.variables[varname][0]} ({self.ref.name} - {self.test.name})",
                                          cb_label=f"$d$ for {varname} (-)", cmap=cmocean.cm.diff, levels=levels, significant=significant)
        eff_size_plot.savefig(eff_size_path, bbox_inches='tight', dpi=100)

        print(f'Spatial plots saved to {output_path}.')
        return