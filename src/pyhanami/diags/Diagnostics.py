import cmocean
import numpy as np
import xarray as xr
import pyhanami.config as config

from pathlib import Path
from pyhanami.utils import plot
from pyhanami.diags.Simulations import SimulationData

variables = config.VARIABLES

class DataDiagnostics:
    def __init__(self, ref: SimulationData, test: SimulationData):
        self.ref = ref
        self.test = test


    def _compute_time_series(self, varname: str):
        """ Compute time series for the given ensembles. """
        raise NotImplementedError("This function is not implemented yet.")

    def _compute_abs_diff(self, varname: str) -> xr.DataArray:
        """ Compute absolute difference between the simulation ensembles 
        at the grid point level. """
        raise NotImplementedError("This function is not implemented yet.")

    def _compute_eff_size_ens(self, varname: str) -> xr.DataArray:
        """ Compute effect size between the simulation ensembles at 
        the grid point level. """
        raise NotImplementedError("This function is not implemented yet.")

    def _compute_significant(self, varname: str) -> np.ndarray:
        """ Compute significant differences between the simulation
        ensembles at the grid point level. """
        raise NotImplementedError("This function is not implemented yet.")


    def time_series_plots(self, varname: str, output_path: str):
        """ Generate time series plots for the given ensembles. """

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

    def spatial_plots(self, varname: str, output_path: str):
        """ Generate absolute difference and effect size plots
        for the given ensembles. """

        # prepare output folder
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        abs_diff_path = output_path / "abs_diff.png"
        eff_size_path = output_path / "eff_size.png"

        # Compute and plot absolute difference
        abs_diff = self._compute_abs_diff(varname)
        limit = np.max(np.abs(abs_diff.values))
        levels = np.linspace(-limit, limit, 13)

        abs_diff_plot = plot.spatial_plot(abs_diff, title=f'Difference in {variables[varname][0]} ({self.ref.name} - {self.test.name})',
                                          cb_label=f"difference in {varname} ({variables[varname][1]})", cmap=cmocean.cm.thermal, levels=levels)
        abs_diff_plot.savefig(abs_diff_path, bbox_inches='tight', dpi=100)

        # Compute and plot effect size with significant differences
        eff_size = self._compute_eff_size_ens(varname)
        significant = self._compute_significant(varname)
        levels = [-2,-1.2,-0.8,-0.5,-0.2,-0.01,0.01,0.2,0.5,0.8,1.2,2.0]    # Use Cohen's limits for effect size

        eff_size_plot = plot.spatial_plot(eff_size, title=f"Cohen's effect size ($d$) for {variables[varname][0]} ({self.ref.name} - {self.test.name})",
                                          cb_label=f"$d$ for {varname} (-)", cmap=cmocean.cm.diff, levels=levels, significant=significant)
        eff_size_plot.savefig(eff_size_path, bbox_inches='tight', dpi=100)

        print(f'Spatial plots saved to {output_path}.')
        return