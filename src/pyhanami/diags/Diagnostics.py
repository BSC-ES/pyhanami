import numpy as np
import xarray as xr
from pathlib import Path
from pyhanami.diags.Simulations import SimulationData
from pyhanami.utils import plot


class DataDiagnostics:
    def __init__(self, ref: SimulationData, test: SimulationData):
        self.ref = ref
        self.test = test


    def _compute_time_series(self):
        """ Compute time series for the given ensembles. """
        raise NotImplementedError("This function is not implemented yet.")

    def _compute_abs_diff(self) -> xr.DataArray:
        """ Compute absolute difference between the simulation ensembles 
        at the grid point level. """
        raise NotImplementedError("This function is not implemented yet.")

    def _compute_eff_size_ens(self) -> xr.DataArray:
        """ Compute effect size between the simulation ensembles at 
        the grid point level. """
        raise NotImplementedError("This function is not implemented yet.")

    def _compute_significant(self) -> np.ndarray:
        """ Compute significant differences between the simulation
        ensembles at the grid point level. """
        raise NotImplementedError("This function is not implemented yet.")


    def time_series_plots(self, output_path: str):
        """ Generate time series plots for the given ensembles. """

        # Prepare output folder
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        time_series_path = output_path / "time_series.png"

        # Compute and plot time series
        time_series = self._compute_time_series() 
        time_series_plot = plot.time_series_plot(time_series)
        time_series_plot.savefig(time_series_path)

        print(f'Time series plots saved to {output_path}.')
        return

    def spatial_plots(self, output_path: str):
        """ Generate absolute difference and effect size plots
        for the given ensembles. """

        # prepare output folder
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        abs_diff_path = output_path / "abs_diff.png"
        eff_size_path = output_path / "eff_size.png"

        # Compute and plot absolute difference
        abs_diff = self._compute_abs_diff()
        abs_diff_plot = plot.spatial_plot(abs_diff)
        abs_diff_plot.savefig(abs_diff_path)

        # Compute and plot effect size with significant differences
        eff_size = self._compute_eff_size_ens()
        significant = self._compute_significant()
        eff_size_plot = plot.spatial_plot(eff_size, significant)
        eff_size_plot.savefig(eff_size_path)

        print(f'Spatial plots saved to {output_path}.')
        return