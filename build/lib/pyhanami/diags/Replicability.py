import numpy as np
import xarray as xr
import concurrent.futures

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
        self.max_workers_vars = config.MAX_WORKERS_VARS
        self.metrics = config.METRICS
        self.tests = config.TESTS
        self.seasons = config.SEASONS
        self.regions = config.REGIONS


    def _compare_ensembles(self):
        """ Check that both provided ensembles are equivalent
        (same variables, periods, ...). """
        raise NotImplementedError("This function is not implemented yet.")


    def _compute_scores_one_var(self, var_name:str) -> tuple[str, xr.Dataset]:
        """ Compute scores for the given variable in both simulation ensembles. """

        data_obs = self.obs[[var_name]].resample(time = '1MS').sum().persist()

        # Initialize scores dictionary
        length_seasons = len(self.seasons)
        length_regions = len(self.regions)
        length_realizations = self.ref[var_name].sizes['realization']
        scores_dict = {metric['name']: np.zeros((2, length_seasons, length_regions, length_realizations))
                   for metric in self.metrics}

        # Process each dataset
        datasets = [self.ref[[var_name]], self.test[[var_name]]]
        dataset_names = [self.ref.name, self.test.name]
        for dataset_idx, data_sim in enumerate(datasets):
            data_sim = data_sim.persist()

            # Process each metric
            for metric_idx, metric in enumerate(self.metrics):
                metric_label = metric['name']
                metric_funcs = metric['functions']
                obs_needed = metric['obs_needed']

                 # Compute scores with annual and seasonal climatology
                for season_idx, season in enumerate(self.seasons):
                    if season_idx == 0:
                        data_sim_season = data_sim
                        data_obs_season = data_obs if obs_needed else None
                    else:
                        data_sim_season = data_sim.groupby('time.season')[season] 
                        data_obs_season = data_obs.groupby('time.season')[season] if obs_needed else None
                    if metric_idx < 2:
                            data_sim_season = data_sim_season.mean(dim='time')

                    # Global, tropical and extratropical regions
                    for region_idx, region in enumerate(self.regions.values()):
                        data_sim_region = data_sim_season.sel(lat=region)
                        data_obs_region = data_obs_season.sel(lat=region) if obs_needed else None
                    
                        scores_region = 0
                        for metric_func in metric_funcs:
                            if metric_func.__code__.co_argcount == 3:
                                scores = metric_func(data_sim_region, data_obs_region, var_name)
                            else:
                                scores = metric_func(data_sim_region, var_name)
                            scores_region += scores
                        scores_dict[metric_label][dataset_idx, season_idx, region_idx,:] = scores_region

           
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
        scores_dataset.attrs['long_name'] = self.variables[var_name][0]

        print(f'Computed scores for variable {var_name}...', flush=True)
        return var_name, scores_dataset
    

    def _compute_scores(self) -> Dict[str, xr.Dataset]:
        """ Compute scores for all variables in both simulation ensembles
        in parallel. """

        scores_all = {}
        vars = self.variables.keys()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers_vars) as executor:
            for results in executor.map(self._compute_scores_one_var, vars):
                var_name, scores_var = results
                scores_all[var_name] = scores_var

        print('Computed scores for all variables.\n', flush=True)
        return scores_all


    def _compute_eff_sizes(self, scores_all: dict[str, xr.Dataset]) -> np.ndarray:
        """ Compute effect sizes between the pre-computed scores separating 
        by season and region, for all available variables. """
        
        # Initialize array
        length_variables = len(self.variables)
        length_sections = len(self.seasons)*len(self.regions)
        length_metrics = len(self.metrics)
        effect_sizes = np.empty((length_variables, length_sections, length_metrics+1))
    
        # Loop over all scores sets
        for var_idx, var in enumerate(self.variables):
            scores_var = scores_all[var]

            for metric_idx, metric_name in enumerate(np.append(self.metrics['name'], 'Combined')):
                for season_idx, season in enumerate(self.seasons):
                    for region_idx, region in enumerate(list(self.regions.keys())):
                        section_idx = season_idx*len(self.regions)+region_idx
                        scores = scores_var[metric_name].sel(season=season, region=region)

                        scores_ref = scores.sel(dataset=self.ref.name).values
                        scores_test = scores.sel(dataset=self.test.name).values

                        # Compute effect size with bootstrapping
                        bootstrap_res = bootstrap((scores_ref, scores_test), statistics.cp_effect_size, confidence_level=0.95, n_resamples=10000)
                        effect_sizes[var_idx, section_idx, metric_idx] = np.mean(bootstrap_res.bootstrap_distribution)

        print('Computed effect sizes between scores distributions for all variables.', flush=True)
        return effect_sizes


    def _apply_tests(self, scores_all: Dict[str, xr.Dataset], alpha: float) -> np.ndarray:
        """ Compare scores with statistical tests separating 
        by season and region, for all available variables. """

        # Initialize array
        length_variables = len(self.variables)
        length_sections = len(self.seasons)*len(self.regions)
        length_tests = len(self.tests)
        test_results = np.zeros((length_variables, length_sections, length_tests), dtype=bool)

        # Loop over all scores sets
        for var_idx, var in enumerate(self.variables):
            scores_var = scores_all[var]

            for metric_name in np.append(self.metrics['name'], 'Combined'):
                for season_idx, season in enumerate(self.seasons):
                    for region_idx, region in enumerate(list(self.regions.keys())):
                        section_idx = season_idx*len(self.regions)+region_idx
                        scores = scores_var[metric_name].sel(season=season, region=region)

                        scores_ref = scores.sel(dataset=self.ref.name).values
                        scores_test = scores.sel(dataset=self.test.name).values

                        # Apply statistical tests
                        for test_idx, test_name in enumerate(self.tests):
                            p_value = self.tests[test_name](scores_ref, scores_test)
                            test_results[var_idx, section_idx, test_idx] |= (p_value <= alpha)

        print('Performed replicability test for all variables.', flush=True)
        return test_results
            

    def matrix_plot(self, output_path: str, alpha: float = 0.05):
        """ Perform replicability test comparing the given simulation ensembles
        and generate matrix plot with effect sizes and test results. """

        print(f'Started replicability test with significance level {alpha} to compare ensembles {self.ref.name} and {self.test.name}:\n')

        # Validate inputs
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"The significance level 'alpha' must be numeric.")
        if not (0 <= alpha <= 1):
            raise ValueError(f"'alpha' must be between 0 and 1.")

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