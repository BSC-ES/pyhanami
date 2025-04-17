import numpy as np
import xarray as xr
from pathlib import Path
from pyhanami.diags.datasets import SimulationData, ObservationData
from pyhanami.diags import plots as pts


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
        time_series_plot = pts.time_series_plot(time_series)
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
        abs_diff_plot = pts.spatial_plot(abs_diff)
        abs_diff_plot.savefig(abs_diff_path)

        # Compute and plot effect size with significant differences
        eff_size = self._compute_eff_size_ens()
        significant = self._compute_significant()
        eff_size_plot = pts.spatial_plot(eff_size, significant)
        eff_size_plot.savefig(eff_size_path)

        print(f'Spatial plots saved to {output_path}.')
        return


class ReplicabilityTest:
    def __init__(self, ref: SimulationData, test: SimulationData):
        self.ref = ref
        self.test = test
        self._compare_ensembles()
        self.obs = ObservationData(ref.data)


    def _compare_ensembles(self):
        """ Check that both provided ensembles are equivalent
        (same variables, periods, ...) """
        raise NotImplementedError("This function is not implemented yet.")

    def _compute_scores(self) -> xr.Dataset:
        """ Compute scores for both simulation ensembles."""
        raise NotImplementedError("This function is not implemented yet.")

    def _compute_eff_sizes(scores: xr.Dataset) -> np.ndarray:
        """ Compute effect sizes for the distributions of scores."""
        raise NotImplementedError("This function is not implemented yet.")

    def _apply_tests(scores: xr.Dataset) -> np.ndarray:
        """ Compare scores with statistical tests. """
        raise NotImplementedError("This function is not implemented yet.")


    def matrix_plot(self, output_path: str):
        """ Perform replicability test comparing the given simulation 
        ensembles and generate matrix plot with effect sizes and test 
        results. """

        # Prepare output folder
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        matrix_path = output_path / "matrix.png"

        # Run replicability test
        scores = self._compute_scores()
        eff_sizes = self._compute_eff_sizes(scores)
        test_results = self._apply_tests(scores)

        # Plot results
        matrix = pts.matrix_plot(eff_sizes, test_results)
        matrix.savefig(matrix_path)

        print(f'Matrix plot saved to {output_path}.')
        return