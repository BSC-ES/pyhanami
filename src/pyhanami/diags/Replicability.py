import numpy as np
import xarray as xr
from pathlib import Path
from pyhanami.utils import plot
from pyhanami.utils import report
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData


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
        matrix = plot.matrix_plot(eff_sizes, test_results, title=f"Effect size replicability test ({self.ref.name} vs {self.test.name})")
        matrix.savefig(matrix_path)

        print(f'Matrix plot saved to {output_path}.')
        return
    
    def report(self, output_path:str, time_series=False, spatial=False):
        """ Generate a summary report with the results of the replicability test 
        and the selected plots. """
        generated_plots = {'time_series': False, 'spatial': False, 'matrix': False}

        # Check if the selected plots have already been generated
        # Generate missing plots and save all in generated_plots

        report.pdf_replicability(output_path, generated_plots)
        raise NotImplementedError("This function is not implemented yet.")