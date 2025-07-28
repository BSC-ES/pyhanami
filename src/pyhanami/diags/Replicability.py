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
    """
    Perform replicability test between two climate simulation ensembles.

    This class compares two climate simulation ensembles (reference and test) using 
    a variety of metrics and statistical tests to assess whether both climates are
    statistically significantly different. The test is conducted over multiple 
    variables, regions, seasons, and ensemble members. It also supports plotting 
    results and generating summary reports.
    
    Parameters
    ----------
    ref : SimulationData
        Reference ensemble containing simulation data and metadata.
    test : SimulationData
        Test ensemble containing simulation data and metadata.
    obs_path : str
        Path to the observations database.

    Attributes
    ----------
    ref : SimulationData
        Instance containing the reference ensemble and metadata.
    test : SimulationData
        Instance containing the test ensemble and metadata.
    obs : ObservationData
        Instance containing observational data for comparison.
    variables : dict
        Configuration dictionary mapping variable names to display metadata.
    max_workers_grid : int
        Number of parallel workers used for variable-wise computations.
    metrics : list of dict
        List of metrics with names and corresponding functions to compute scores.
    tests : dict
        Dictionary of statistical tests for comparing score distributions.
    seasons : list of str
        List of seasons to compute scores over.
    regions : dict
        Dictionary mapping region names to latitude bounds.

    Methods
    -------
    _compare_ensembles()
        (Not implemented) Intended to validate compatibility between the reference and test ensembles.

    _compute_scores_one_var(var_name)
        Computes scores employing several metrics for a sgiven variable across both ensembles.

    _compute_scores()
        Computes scores for all variables in parallel.

    _compute_eff_sizes(scores_all)
        Computes effect sizes (Cohen's d) between ensembles for each variable, region, and season.

    _apply_tests(scores_all, alpha)
        Applies several statistical tests between ensemble score distributions.

    matrix_plot(output_path, alpha=0.05)
        Runs the full replicability test and generates a matrix plot of the results.

    report(output_path, time_series=False, spatial=False)
        Generates a report summarizing the replicability test results and optionally includes plots.
    """

    def __init__(self, ref: SimulationData, test: SimulationData, obs_path: str):
        self.ref = ref
        self.test = test
        #self._compare_ensembles()
        self.obs = ObservationData(obs_path, ref.data)

        # Load config parameters once
        self.variables = {var: info for var, info in config.VARIABLES.items() if var in ref.data.data_vars}
        self.max_workers_vars = config.MAX_WORKERS_VARS
        self.metrics = config.METRICS
        self.tests = config.TESTS
        self.seasons = config.SEASONS
        self.regions = config.REGIONS


    def _compare_ensembles(self):
        """ Check that both provided ensembles are equivalent
        (same variables, periods, ...). """
        raise NotImplementedError("This function is not implemented yet.")


    def _compute_scores_one_var(self, var_name):
        """ 
        Compute scores for the given variable in both simulation ensembles. 
        
        Parameters
        ----------
        var_name (str): Climate variable name.

        Returns
        -------
        tuple[str, xr.Dataset]: Variable name and dataset containing computed scores.
        """

        data_obs = self.obs.data[[var_name]].resample(time = '1MS').sum().persist()

        # Initialize scores dictionary
        length_seasons = len(self.seasons)
        length_regions = len(self.regions)
        length_realizations = self.ref.data[var_name].sizes['realization']
        scores_dict = {metric['name']: np.zeros((2, length_seasons, length_regions, length_realizations))
                   for metric in self.metrics}

        # Process each dataset
        datasets = [self.ref.data[[var_name]], self.test.data[[var_name]]]
        dataset_names = [self.ref.name, self.test.name]
        for dataset_idx, data_sim in enumerate(datasets):
            data_sim = data_sim.persist()
            lat = data_sim['lat']

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
                        mask = region(lat)
                        data_sim_region = data_sim_season.where(mask, drop=True)
                        data_obs_region = data_obs_season.where(mask, drop=True) if obs_needed else None
                    
                        scores_region = 0
                        for metric_func in metric_funcs:
                            if metric_func.__code__.co_argcount == 3:
                                scores = metric_func(data_sim_region, data_obs_region, var_name)
                            else:
                                scores = metric_func(data_sim_region, var_name)
                            scores_region += scores
                        scores_dict[metric_label][dataset_idx, season_idx, region_idx,:] = scores_region
                        del mask

           
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

        print(f"\tComputed scores for variable '{var_name}'...", flush=True)
        return scores_dataset
    

    def _compute_scores(self):
        """ 
        Compute scores for all variables in both simulation ensembles in parallel. 

        Returns
        -------
        dict[str, xr.Dataset]: Dictionary of scores datasets for each variable.
        """

        scores_all = {}
        vars = list(self.variables.keys())
        # for var in vars:
        #     scores_all[var] = self._compute_scores_one_var(var)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_vars) as executor:
            for idx, value in enumerate(executor.map(self._compute_scores_one_var, vars)):
                scores_all[vars[idx]] = value

        print('Computed scores for all variables...', flush=True)
        return scores_all


    def _compute_eff_sizes(self, scores_all):
        """ 
        Compute effect size (Cohen's d) between the pre-computed scores separating 
        by season and region, for all available variables. 
        
        Parameters
        ----------
        scores_all (dict[str, xr.Dataset]): Dictionary of scores datasets for each variable.

        Returns
        -------
        effect_sizes (np.ndarray): Array of effect sizes with shape (variables, sections, metrics).
        """
        
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

        print('Computed effect sizes between scores distributions for all variables...', flush=True)
        return effect_sizes


    def _apply_tests(self, scores_all, alpha):
        """ 
        Compare scores with statistical tests separating by season
        and region, for all available variables. 
        
        Parameters
        ----------
        scores_all (dict[str, xr.Dataset]): Dictionary of scores datasets for each variable.
        alpha (float): Significance level for the statistical tests.

        Returns
        -------
        test_results (np.ndarray): Array of test results with shape (variables, sections, tests).
        """

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

        print('Performed replicability test for all variables...', flush=True)
        return test_results
            

    def matrix_plot(self, output_path, alpha=0.05):
        """ 
        Perform replicability test comparing the given simulation ensembles
        and generate matrix plot with effect sizes and test results. 
        
        Parameters
        ---------- 
        output_path (str): Path to save the matrix plot.
        alpha (float): Significance level for the statistical tests.
        """

        print(f'Started replicability test with significance level {alpha} to compare ensembles {self.ref.name} and {self.test.name}:', flush=True)

        # Validate inputs
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"The significance level 'alpha' must be numeric.")
        if not (0 <= alpha <= 1):
            raise ValueError(f"'alpha' must be between 0 and 1.")

        # Prepare output path
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            matrix_path = output_path / f"matrix_{self.ref.name}-{self.test.name}.png"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            matrix_path = output_path

        # Run replicability test
        scores = self._compute_scores()
        eff_sizes = self._compute_eff_sizes(scores)
        test_results = self._apply_tests(scores, alpha)
        
        # Plot results
        matrix, _ = plot.matrix_plot(eff_sizes, test_results, title=f"Effect size replicability test ({self.ref.name} vs {self.test.name})", variables=self.variables)
        matrix.savefig(matrix_path, bbox_inches='tight', dpi=100)

        print(f"Matrix plot saved to '{matrix_path}'.", flush=True)
        return
    
    
    def report(self, output_path, time_series=False, spatial=False):
        """ 
        Generate a summary report with the results of the replicability test 
        and the selected plots. 
        
        Parameters
        ----------
        output_path (str): Path to save the report.
        time_series (bool): Whether to include time series plots in the report.
        spatial (bool): Whether to include spatial plots in the report.
        """

        generated_plots = {'time_series': False, 'spatial': False, 'matrix': False}

        # Check if the selected plots have already been generated
        # Generate missing plots and save all in generated_plots

        report.pdf_replicability(output_path, generated_plots)
        raise NotImplementedError("This function is not implemented yet.")