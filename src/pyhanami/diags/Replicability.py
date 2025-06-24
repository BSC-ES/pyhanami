import numpy as np
import xarray as xr

from typing import Dict
from pathlib import Path
from scipy.stats import bootstrap

from pyhanami import config
from pyhanami.utils import plot, report, statistics
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData


class ReplicabilityTest:
    def __init__(self, ref: SimulationData, test: SimulationData):
        self.ref = ref
        self.test = test
        self._compare_ensembles()
        self.obs = ObservationData(ref.data)

        # Load config parameters once
        self.variables = config.VARIABLES
        self.metrics = config.METRICS
        self.tests = config.TESTS
        self.seasons = config.SEASONS
        self.regions = config.REGIONS


    def _compare_ensembles(self):
        """ Check that both provided ensembles are equivalent
        (same variables, periods, ...). """
        raise NotImplementedError("This function is not implemented yet.")


    def _compute_scores_variable(self, var_name:str, long_name:str) -> xr.Dataset:
        """ Compute scores for the given variable in both simulation ensembles. """

        data_obs = self.obs[var_name].resample(time = '1MS').sum().persist()

        # Initialize scores dictionary
        length_seasons = len(self.seasons)
        length_regions = len(self.regions)
        length_realizations = self.ref[var_name].sizes['realization']
        scores_dict = {metric['name']: np.zeros((2, length_seasons, length_regions, length_realizations))
                   for metric in self.metrics}

        # Process each dataset
        datasets = [self.ref[var_name], self.test[var_name]]
        dataset_names = [self.ref.name, self.test.name]
        for dataset_idx, data_sim in enumerate(datasets):
            data_sim = data_sim.persist()

            # Process each metric
            for metric_idx, metric in enumerate(self.metrics):
                metric_label = metric['name']
                metric_funcs = metric['functions']
                obs_needed = metric['obs_needed']


                # Compute scores with annual climatology
                if metric_idx < 2:
                    data_sim_processed = data_sim.mean(dim='time')
                else:
                    data_sim_processed = data_sim

                # Global, tropical and extratropical regions
                for r,region in enumerate(self.regions.values()):
                    data_sim_region = data_sim_processed.sel(lat=region)
                    data_obs_region = data_obs.sel(lat=region) if obs_needed else None

                    scores_region = 0
                    for metric_func in metric_funcs:
                        if metric_func.__code__.co_argcount == 3:
                            scores = metric_func(data_sim_region, data_obs_region, var_name)
                        else:
                            scores = metric_func(data_sim_region, var_name)
                        scores_region += scores
                    scores_dict[metric_label][dataset_idx,0,r,:] = scores_region


                # Compute scores with seasonal climatologies
                if length_seasons > 1:
                    data_sim_seasons = data_sim_processed.groupby('time.season')
                    data_obs_seasons = data_obs.groupby('time.season') if obs_needed else None

                    for s,season in enumerate(self.seasons[1:], start=1): 
                        data_sim_season = data_sim_seasons[season] 
                        if metric_idx < 2:
                            data_sim_season = data_sim_season.mean(dim='time')
                        data_obs_season = data_obs_seasons[season] if obs_needed else None

                        # Global, tropical and extratropical regions
                        for r,region in enumerate(self.regions.values()):
                            data_sim_region = data_sim_season.sel(lat=region)
                            data_obs_region = data_obs_season.sel(lat=region) if obs_needed else None

                            scores_region = 0
                            for metric_func in metric_funcs:
                                if metric_func.__code__.co_argcount == 3:
                                    scores = metric_func(data_sim_region, data_obs_region, var_name)
                                else:
                                    scores = metric_func(data_sim_region, var_name)
                                scores_region += scores
                            scores_dict[metric_label][dataset_idx,s,r,:] = scores_region

           
        # Add combined metric
        combined_scores = np.mean(np.stack([scores_dict[metric['name']] for metric in self.metrics], axis=0), axis=0)
        scores_dict['Combined'] = combined_scores

        # Create xarray.Dataset with scores for all metrics
        coords = {
            'dataset': dataset_names,
            'season': self.seasons,
            'region': list(self.regions.keys()),
            'realization': np.arange(length_realizations)
        }

        scores_var = {}
        for metric_name, scores in scores_dict.items():
            scores_var[metric_name] = (['dataset', 'season', 'region', 'realization'], scores)
        
        scores_dataset = xr.Dataset(data_vars=scores_var, coords=coords)
        scores_dataset.attrs['variable'] = var_name
        scores_dataset.attrs['long_name'] = long_name

        return scores_dataset
    

    def _compute_scores(self) -> Dict[str, xr.Dataset]:
        """ Compute scores for all variables in both simulation ensembles. """

        scores_all = {}
        for var_name, long_name in self.variables.items():
            print(f'Computing scores for variable {var_name}...')
            scores_var = self._compute_scores_variable(var_name, long_name)
            scores_all[var_name] = scores_var

        return scores_all


    def _compute_eff_sizes(self, scores: Dict[str, xr.Dataset]) -> np.ndarray:
        """ Compute effect sizes between the pre-computed scores separating 
        by season and region, for all available variables. """
        
        # Initialize array
        length_variables = len(self.variables)
        length_sections = len(self.seasons)*len(self.regions)
        length_metrics = len(self.metrics)
        effect_sizes = np.empty((length_variables, length_sections, length_metrics))
    
        # Loop over all scores sets
        for var_idx, var in enumerate(self.variables):
            scores_all = scores[var]

            for metric_idx, metric in enumerate(self.metrics):
                for season_idx, season in enumerate(self.seasons):
                    for region_idx, region in enumerate(list(self.regions.keys())):
                        section_idx = season_idx*len(self.regions)+region_idx
                        scores = scores_all[metric['name']].sel(season=season, region=region)

                        scores_ref = scores.sel(dataset=self.ref.name).values
                        scores_test = scores.sel(dataset=self.test.name).values

                        # Compute effect size with bootstrapping
                        bootstrap_res = bootstrap((scores_ref, scores_test), statistics.cp_effect_size, confidence_level=0.95, n_resamples=10000)
                        effect_sizes[var_idx, section_idx, metric_idx] = np.mean(bootstrap_res.bootstrap_distribution)

        return effect_sizes


    def _apply_tests(self, scores: Dict[str, xr.Dataset], alpha: float) -> np.ndarray:
        """ Compare scores with statistical tests separating 
        by season and region, for all available variables. """

        # Initialize array
        length_variables = len(self.variables)
        length_sections = len(self.seasons)*len(self.regions)
        length_tests = len(self.tests)
        test_results = np.zeros((length_variables, length_sections, length_tests), dtype=bool)

        # Loop over all scores sets
        for var_idx, var in enumerate(self.variables):
            scores_all = scores[var]

            for metric in self.metrics:
                for season_idx, season in enumerate(self.seasons):
                    for region_idx, region in enumerate(list(self.regions.keys())):
                        section_idx = season_idx*len(self.regions)+region_idx
                        scores = scores_all[metric['name']].sel(season=season, region=region)

                        scores_ref = scores.sel(dataset=self.ref.name).values
                        scores_test = scores.sel(dataset=self.test.name).values

                        # Apply statistical tests
                        for test_idx, test_name in enumerate(self.tests):
                            p_value = self.tests[test_name](scores_ref, scores_test)
                            test_results[var_idx, section_idx, test_idx] |= (p_value <= alpha)

        return test_results
            

    def matrix_plot(self, output_path: str, alpha: float = 0.05):
        """ Perform replicability test comparing the given simulation ensembles
        and generate matrix plot with effect sizes and test results. """

        # Prepare output folder
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        matrix_path = output_path / "matrix.png"

        # Run replicability test
        scores = self._compute_scores()
        eff_sizes = self._compute_eff_sizes(scores)
        test_results = self._apply_tests(scores, alpha)

        # Plot results
        matrix, _ = plot.matrix_plot(eff_sizes, test_results, title=f"Effect size replicability test ({self.ref.name} vs {self.test.name})")
        matrix.savefig(matrix_path, bbox_inches='tight', dpi=100)

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