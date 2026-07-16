import os
import re
import shutil
import xarray as xr

from pathlib import Path

from pyhanami.config import config_params
from pyhanami.diags.Simulations import SimulationData
from pyhanami.utils import data_general, config_scores
from pyhanami.utils.plots import plots_tc, plots_scientific_evaluation_tables
from pyhanami.utils.tcs_scores import tcs_tempestextremes, tcs_ibtracs, tcs_cymep_main


class TCEvaluation:
    """
    Compute Tropical Cyclones (TCs) metrics and derived scalar scores.

    This class provides functionality for computing various TC metrics and derived scalar scores following
    (C.M. Zarzycki et al., 2021) and plotting the results comparing simulations to IBTrACS observational
    data, as well as, several reanalysis datasets.

    Parameters
    ----------
    data_sim: SimulationData
        Simulation dataset to use.
    start_year_tc, end_year_tc : int, optional
        Initial and end years to compute the TCs metrics for.
    obs : bool
        If True, also consider observational data if available (default: True).
    wind_factor : float
        Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
    min_wind : float
        Minimum 10 m wind speed in m/s for TCs detection (default: 10.0).
    basin : int
        Basin/hemisphere to consider for the analysis (default: -1). Codes are:
            - <0 → GLOB (Global domain)
            - 1  → NATL (North Atlantic)
            - 2  → EPAC (Eastern Pacific)
            - 3  → CPAC (Central Pacific)
            - 4  → WPAC (Western Pacific)
            - 5  → NIO (North Indian Ocean)
            - 6  → SIO (South Indian Ocean)
            - 7  → SPAC (South Pacific)
            - 8  → SATL (South Atlantic)
            - 9  → FLA (Florida)
            - 20 → NHEMI (Northern Hemisphere)
            - 21 → SHEMI (Southern Hemisphere)
            - otherwise → NONE (unrecognized)
    bin_size : float
        Size of the bins in degrees for computing the TCs metrics with CyMeP (default: 2.5).
    tc_config : TCConfig
        Configuration dataclass with parameters necessary for the TC evaluation. If None, default values
        from the configuration file `pyhanami.config.scientific_evaluation_parameters.yaml` will be used.

    Attributes
    ----------
    sim_name : str
        Name of the simulation dataset.
    obs_names : str
        Name of the observational dataset, if requested.
    obs : bool
        Whether to use observational data.
    start_year_tc, end_year_tc : int
        Initial and end years to compute the TCs metrics for.
    min_wind : float
        Minimum 10 m wind speed in m/s for TCs detection.
    config_cymep : dict
        Dictionary with configuration parameters for each dataset (containing traj_filename,
        short_name, unstructured, ens_members, years_per_member, wind_speed_correction),
        necessary for the CyMeP package.
    tracks_path : str
        Path to save temporary files.
    bin_size : float
        Size of the bins in degrees for computing the TCs metrics with CyMeP.
    metrics_metadata : dict
        Metadata for the TC metrics.
    data_cymep : xr.Dataset
        TCs metrics computed with CyMeP.
    model_names : list[str]
        List of model names included in the TCs metrics dataset.
    clim_bias : xr.Dataset
        Global mean climatological bias for each TC metric.
    storm_bias : xr.Dataset
        Global mean storm bias for each TC metric.
    temp_corr : xr.Dataset
        Seasonal correlation for each TC metric.
    spatial_corr : xr.Dataset
        Spatial correlation for each TC metric.
    """

    def __init__(self, data_sim, start_year_tc=None, end_year_tc=None, obs=True, wind_factor=1.0,
                 min_wind=10.0, basin=-1, bin_size=2.5, tc_config=None):

        # Validate input
        if not isinstance(data_sim, SimulationData):
            raise TypeError("'data_sim' must be an instance of SimulationData.")
        self.sim_name = data_sim.name
        self.obs_names = []
        self.obs = obs

        if not isinstance(min_wind, (int, float)) or min_wind < 10.0:
            raise ValueError(
                "The minimum 10 m wind speed for TCs detection 'min_wind' must be a numeric value of at least 10 m/s."
            )
        self.min_wind = min_wind

        if not isinstance(bin_size, (int, float)) or bin_size <= 0:
            raise ValueError(
                "The 'bin_size' for computing the TCs scores must be a positive numeric value."
            )
        self.bin_size = bin_size

        self.config_cymep = {}
        self.metrics_metadata = data_general.load_yaml_file(config_params.TCS_METRICS_PATH)

        # Load TC evaluation parameters
        if tc_config is None:
            tc_config = config_scores.TCConfig()
        elif not isinstance(tc_config, config_scores.TCConfig):
            raise TypeError(
                "'tc_config' must be an instance of the TCConfig dataclass defined in 'pyhanami.utils.config_scores'."
            )

        # Select years for TCs analysis
        self.start_year_tc, self.end_year_tc = data_general.validate_year_range(
            data_sim, start_year_tc, end_year_tc, process_name="TC"
        )
        print(
            f"\tYears selected for TCs scores computation: {self.start_year_tc}-{self.end_year_tc}.",
            flush=True,
        )
        data_sim_tcs = data_sim.data.sel(time=slice(str(self.start_year_tc), str(self.end_year_tc)))

        # Prepare output folder for temporary files
        self.tracks_path = config_params.TC_DATA_PATH / "temp_tracks"
        self.tracks_path.mkdir(parents=True, exist_ok=True)


        # # Prepare IBTrACS data, and observational data if requested
        self._prepare_IBTrACS_data()
        print(
            f"\tIBTrACS TCs data preprocessed and saved to '{self.config_cymep['IBTrACS'][0]}'.",
            flush=True,
        )

        if self.obs:
            self.obs_names = ["JRA55"]
            self._prepare_obs_data()
            print(
                f"\tObservational TCs data preprocessed and saved to '{self.tracks_path}'.",
                flush=True,
            )

        # Prepare simulation data
        print("\tStarting simulation data TCs tracking. TempestExtremes output:", flush=True)
        self._prepare_sim_data(data_sim_tcs, wind_factor=wind_factor, tc_config=tc_config)
        print(
            f"\tSimulation data TCs tracking completed and saved to '{self.config_cymep[self.sim_name][0]}'.",
            flush=True,
        )

        # Compute TCs metrics
        print("\tStarting Tropical Cyclones metrics computation. CyMeP output:", flush=True)
        self.data_cymep, self.model_names = self._compute_cymep_metrics(basin=basin, tc_config=tc_config)
        print("\tTCs metrics computed. See attribute `data_cymep`.", flush=True)

        self.clim_bias, self.storm_bias = self._retrieve_biases()
        print("\tBiases computation completed. See attributes `clim_bias` and `storm_bias`.", flush=True)

        self.temp_corr, self.spatial_corr =  self._retrieve_correlations()
        print("\tCorrelations computation completed. See attributes `temp_corr` and `spatial_corr`.", flush=True)

        # Delete intermediate files
        shutil.rmtree(self.tracks_path)

        print(f"\nTropical Cyclones scores computation completed between years {self.start_year_tc} "
              f"and {self.end_year_tc}.", flush=True)
        return


    def _prepare_IBTrACS_data(self):
        """
        Prepare IBTrACS TCs data to use as a reference for the TCs metrics computation.
        """

        # Retrieve IBTrACS data metadata
        ib_path, ib_end_year = tcs_ibtracs.check_ibtracs_file(
            self.start_year_tc,
            self.end_year_tc,
            output_path=self.tracks_path,
            min_wind=self.min_wind,
        )

        # Add IBTrACS parameters to CyMeP configuration file
        year_range = ib_end_year - config_params.IBTRACS_START_YEAR + 1
        ib_config = [str(ib_path), "IBTrACS", False, 1, year_range, 1.0]
        self.config_cymep["IBTrACS"] = ib_config

        return


    def _prepare_obs_data(self):
        """
        Prepare observational TCs data for the TCs metrics computation.
        """

        data_path = config_params.TC_DATA_PATH
        for name in self.obs_names:
            # Look for observational data
            obs_files = list(data_path.glob(f"{name.lower()}_*.txt"))
            if obs_files is None:
                raise FileNotFoundError(f"No observational TCs data found for '{name}' in '{data_path}'.")
            obs_path = obs_files[0]

            # Check whether the current version covers the selected period
            match = re.search(rf"{name.lower()}_(\d+)-(\d+)", obs_path.name)
            obs_start_year = int(match.group(1))
            obs_end_year = int(match.group(2))

            if self.start_year_tc < obs_start_year or obs_end_year < self.end_year_tc:
                data_general.warn_always(
                    f"The available observational TCs data for '{name}' ({obs_start_year}-{obs_end_year}) does not "
                    f"cover the selected period for TCs metrics computation ({self.start_year_tc}-{self.end_year_tc})."
                    f" This dataset will not be considered for the TCs metrics computation."
                )
                break

            # Apply wind threshold
            if self.min_wind > 10.0:
                new_obs_path = (
                    self.tracks_path
                    / f"{name.lower()}_{obs_start_year}_{obs_end_year}_{self.min_wind:.1f}_False_1_1.0.txt"
                )
                obs_path = tcs_tempestextremes.filter_tracks_by_wind(
                    obs_path,
                    new_obs_path,
                    cutoff_wind=self.min_wind
                )
            else:
                new_obs_path = self.tracks_path / obs_path.name
                shutil.copy2(obs_path, new_obs_path)

            year_range = obs_end_year - obs_start_year + 1
            obs_config = [str(new_obs_path), name, False, 1, year_range, 1.0]
            self.config_cymep[name] = obs_config

        return


    def _prepare_sim_data(self, data_sim, wind_factor=1.0, tc_config=None):
        """
        Prepare simulation data for the TCs metrics computation (detect
        and track TCs with TempestExtremes).

        Parameters
        ----------
        data_sim : xr.Dataset
            Simulation data.
        wind_factor : float
            Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
        tc_config : TCConfig
            Configuration dataclass with parameters necessary for the TempestExtremes functions.
        """

        # Run TempestExtremes tracking on simulated data
        tracks_sim_path = tcs_tempestextremes.run_tempestExtremes(
            data_sim,
            self.sim_name,
            self.tracks_path,
            min_wind=self.min_wind,
            psl_delta=tc_config.psl_delta,
            psl_dist=tc_config.psl_dist,
            z_delta=tc_config.z_delta,
            z_dist=tc_config.z_dist,
            z_offset=tc_config.z_offset,
            merge_dist=tc_config.merge_dist,
            traj_range=tc_config.traj_range,
            traj_min_length=tc_config.traj_min_length,
            traj_max_gap=tc_config.traj_max_gap,
            min_len=tc_config.min_len,
            max_lat=tc_config.max_lat,
        )
        new_sim_path = (
            self.tracks_path
            / f"{self.sim_name.replace(' ', '-')}_{self.start_year_tc}-{self.end_year_tc}_{self.min_wind:.1f}_"
            f"False_1_{wind_factor:.1f}.txt"
        )
        os.rename(tracks_sim_path, new_sim_path)

        # Add simulation data parameters to CyMeP configuration file
        year_range = self.end_year_tc - self.start_year_tc + 1
        self.config_cymep[self.sim_name] = [
            str(new_sim_path),
            self.sim_name,
            False,
            1,
            year_range,
            wind_factor,
        ]

        return


    def _compute_cymep_metrics(self, basin=-1, tc_config=None):
        """
        Compute TCs metrics using the CyMeP package.

        Parameters
        ----------
        basin : int
            Basin/hemisphere to consider for the analysis (default: -1). Codes are:
                - <0 → GLOB (Global domain)
                - 1  → NATL (North Atlantic)
                - 2  → EPAC (Eastern Pacific)
                - 3  → CPAC (Central Pacific)
                - 4  → WPAC (Western Pacific)
                - 5  → NIO (North Indian Ocean)
                - 6  → SIO (South Indian Ocean)
                - 7  → SPAC (South Pacific)
                - 8  → SATL (South Atlantic)
                - 9  → FLA (Florida)
                - 20 → NHEMI (Northern Hemisphere)
                - 21 → SHEMI (Southern Hemisphere)
                - otherwise → NONE (unrecognized)
        tc_config : TCConfig
            Configuration dataclass with parameters necessary for the TempestExtremes functions.

        Returns
        -------
        data_cymep : xr.Dataset
            TCs metrics computed with CyMeP.
        model_names : list[str]
            List of model names included in the TCs metrics dataset.
        """

        # Prepare CyMeP configuration file
        config_cymep_path = self.tracks_path / "cymep_config.csv"
        tcs_cymep_main.prepare_configs_file(self.config_cymep, output_path=config_cymep_path)

        # Run CyMeP TCs metrics computation
        data_cymep = tcs_cymep_main.run_cymep_pyhanami(
            self.start_year_tc,
            self.end_year_tc,
            output_path=self.tracks_path,
            gridsize=self.bin_size,
            basin=basin,
            csvfilename=config_cymep_path,
            truncate_years=tc_config.truncate_years,
            do_defineMIbypres=tc_config.do_defineMIbypres,
            do_fill_missing_pw=tc_config.do_fill_missing_pw,
            do_special_filter_obs=tc_config.do_special_filter_obs,
            THRESHOLD_ACE_WIND=tc_config.threshold_ace_wind,
            THRESHOLD_PACE_PRES=tc_config.threshold_pace_pres,
        )
        model_names = list(data_cymep.model.values)

        return data_cymep, model_names


    def _retrieve_biases(self):
        """
        Retrieve climatological and storm biases from CyMeP output data.

        Returns
        -------
        clim_bias : xr.Dataset
            Global mean climatological bias for each TC metric.
        storm_bias : xr.Dataset
            Global mean storm bias for each TC metric.
        """

        # With numpy arrays (not used anymore, kept for reference)
        # Retrieve biases
        # clim_mean = np.transpose(
        #   [self.data_cymep[f'clim_mean_{metric}'].values
        #   for metric in self.metrics_metadata
        #   if self.metrics_metadata[metric]['temporal']==True]
        # )
        # clim_bias = np.concatenate(([clim_mean[0]], clim_mean[1:] - clim_mean[0]))

        # storm_mean = np.transpose(
        #   [self.data_cymep[f'storm_mean_{metric}'].values
        #   for metric in self.metrics_metadata
        #   if self.metrics_metadata[metric]['temporal']==True and metric!='count']
        # )
        # storm_bias = np.concatenate(([storm_mean[0]], storm_mean[1:] - storm_mean[0]))


        # Retrieve biases as xarray.Datasets
        clim_metrics = [
            metric
            for metric in self.metrics_metadata
            if self.metrics_metadata[metric]["temporal"] == True
        ]
        clim_mean = self.data_cymep[[f"clim_mean_{metric}" for metric in clim_metrics]]

        clim_bias = xr.concat(
            [
                clim_mean.isel(model=0).expand_dims("model"),
                clim_mean.isel(model=slice(1, None)) - clim_mean.isel(model=0),
            ],
            dim="model",
        )
        clim_bias = clim_bias.rename(
            {f"clim_mean_{metric}": f"clim_bias_{metric}" for metric in clim_metrics}
        )


        storm_metrics = [
            metric
            for metric in self.metrics_metadata
            if self.metrics_metadata[metric]["temporal"] == True and metric != "count"
        ]
        storm_mean = self.data_cymep[[f"storm_mean_{metric}" for metric in storm_metrics]]

        storm_bias = xr.concat(
            [
                storm_mean.isel(model=0).expand_dims("model"),
                storm_mean.isel(model=slice(1, None)) - storm_mean.isel(model=0),
            ],
            dim="model",
        )
        storm_bias = storm_bias.rename(
            {f"storm_mean_{metric}": f"storm_bias_{metric}" for metric in storm_metrics}
        )

        return clim_bias, storm_bias


    def _retrieve_correlations(self):
        """
        Retrieve temporal and spatial correlations from CyMeP output data,
        and define related plotting parameters.

        Returns
        -------
        temp_corr : xr.Dataset
            Seasonal correlation for each TC metric.
        spatial_corr : xr.Dataset
            Spatial correlation for each TC metric.
        """

        # With numpy arrays (not used anymore, kept for reference)
        # Retrieve correlations
        # temp_corr = np.transpose(
        #   [self.data_cymep[var].values
        #   for var in self.data_cymep.data_vars
        #   if var.startswith('temporal_scorr_')]
        # )
        # spatial_corr = np.transpose(
        #   [self.data_cymep[var].values
        #   for var in self.data_cymep.data_vars
        #   if var.startswith('spatial_pcorr_')]
        # )


        # Retrieve correlations as xarray.Datasets
        temp_corr = self.data_cymep[
            [var for var in self.data_cymep.data_vars if var.startswith("temporal_scorr_")]
        ]
        spatial_corr = self.data_cymep[
            [var for var in self.data_cymep.data_vars if var.startswith("spatial_pcorr_")]
        ]

        return temp_corr, spatial_corr


    def _table_plots_parameters(self, reference=True):
        """
        Prepare parameters for the table plots.

        Parameters
        ----------
        reference : bool
            Whether to display reference values in the first row (not colored) and
            reanalyses in the subsequent rows (default: True).

        Returns
        -------
        data_names : list[str]
            List of dataset names.
        year_range : str
            Year range of the evaluation period.
        rows_to_remove : int
            Number of datasets to remove from the top of the table (generally, reference
            and reanalyses datasets).
        """
        data_names = self.model_names
        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        rows_to_remove = 0 if reference else len(self.obs_names) + 1

        return data_names, year_range, rows_to_remove


    def save_data(self, output_path):
        """
        Save computed EEOFs, PCs, and frequency of ISO events to NetCDF files.

        Parameters
        ----------
        output_path : str
            Path to save the data files.
        """

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"

        # Prepare dataset names for filenames
        name = f"IBTrACS_{self.sim_name.replace(' ', '-')}"
        if self.obs:
            name += "_obs"

        # Save all CyMeP output data to NetCDF file
        data_cymep_path = output_path / f"tcs_metrics_{name}_{year_range}.nc"
        self.data_cymep.to_netcdf(data_cymep_path)
        print(
            f"TCs metrics computed for '{self.sim_name}' simulations saved to '{data_cymep_path}'.",
            flush=True,
        )

        # Save biases and correlations to Numpy files
        clim_bias_path = output_path / f"tcs_clim_bias_{name}_{year_range}.nc"
        self.clim_bias.to_netcdf(clim_bias_path)
        print(f"Global climatological mean bias saved to '{clim_bias_path}'.", flush=True)

        storm_bias_path = output_path / f"tcs_storm_bias_{name}_{year_range}.nc"
        self.storm_bias.to_netcdf(storm_bias_path)
        print(f"Global storm mean bias saved to '{storm_bias_path}'.", flush=True)

        temp_corr_path = output_path / f"tcs_temp_corr_{name}_{year_range}.nc"
        self.temp_corr.to_netcdf(temp_corr_path)
        print(f"Global seasonal correlation saved to '{temp_corr_path}'.", flush=True)

        spatial_corr_path = output_path / f"tcs_spatial_corr_{name}_{year_range}.nc"
        self.spatial_corr.to_netcdf(spatial_corr_path)
        print(f"Global spatial correlation saved to '{spatial_corr_path}'.", flush=True)

        return


    def clim_bias_table(self, output_path=None, reference=True):
        """
        Generate and save/display table plot with global climatological mean bias for each TC metric.

        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) and reanalyses in 
            the subsequent rows (default: True).
        """

        # Prepare data and plotting parameters
        data = [self.clim_bias]
        data_names, year_range, rows_to_remove = self._table_plots_parameters(reference=reference)
        bias_type = "climatological"

        # Create and save/display table plot
        plots_scientific_evaluation_tables.tc_evaluation_bias_scores_table(
            data,
            data_names=data_names,
            year_range=year_range,
            bias_type=bias_type,
            bin_size=self.bin_size,
            output_path=output_path,
            rows_to_remove=rows_to_remove,
        )

        return


    def storm_bias_table(self, output_path=None, reference=True):
        """
        Generate and save/display table plot with global storm mean bias for each TC metric.

        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) and reanalyses
            in the subsequent rows (default: True).
        """

        # Prepare data and plotting parameters
        data = [self.storm_bias]
        data_names, year_range, rows_to_remove = self._table_plots_parameters(reference=reference)
        bias_type = "storm"

        # Create and save/display table plot
        plots_scientific_evaluation_tables.tc_evaluation_bias_scores_table(
            data,
            data_names=data_names,
            year_range=year_range,
            bias_type=bias_type,
            bin_size=self.bin_size,
            output_path=output_path,
            rows_to_remove=rows_to_remove,
        )

        return


    def temp_corr_table(self, output_path=None, reference=True):
        """
        Generate and save/display table plot with global seasonal correlation for each TC metric.

        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) and reanalyses in 
            the subsequent rows (default: True).
        """

        # Prepare data and plotting parameters
        data = [self.temp_corr]
        data_names, year_range, rows_to_remove = self._table_plots_parameters(reference=reference)
        correlation_type = "seasonal"

        # Create and save/display table plot
        plots_scientific_evaluation_tables.tc_evaluation_correlation_scores_table(
            data,
            data_names=data_names,
            year_range=year_range,
            correlation_type=correlation_type,
            bin_size=self.bin_size,
            output_path=output_path,
            rows_to_remove=rows_to_remove,
        )

        return


    def spatial_corr_table(self, output_path=None, reference=True):
        """
        Generate and save/display table plot with global spatial correlation for each TC metric.

        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) and reanalyses in 
            the subsequent rows (default: True).
        """

        # Prepare data and plotting parameters
        data = [self.spatial_corr]
        data_names, year_range, rows_to_remove = self._table_plots_parameters(reference=reference)
        correlation_type = "spatial"

        # Create and save/display table plot
        plots_scientific_evaluation_tables.tc_evaluation_correlation_scores_table(
            data,
            data_names=data_names,
            year_range=year_range,
            correlation_type=correlation_type,
            bin_size=self.bin_size,
            output_path=output_path,
            rows_to_remove=rows_to_remove,
        )

        return


    def linear_plots(self, output_path=None):
        """
        Generate and save/display linear plots of monthly and interannual cycles for each
        TC metric comparing all datasets.

        Parameters
        ----------
        output_path : str, optional
            Path to save the linear plots. If None, the plots are displayed but not saved.
        """

        # Prepare data and plotting parameters
        data = [self.data_cymep]
        data_names = self.model_names
        year_init = self.start_year_tc
        year_end = self.end_year_tc

        # Create and save/display linear plot
        plots_tc.plot_linear_cycles(
            data,
            data_names=data_names,
            start_year=year_init,
            end_year=year_end,
            output_path=output_path,
        )

        return


    def spatial_plots(self, output_path=None, clon=0):
        """
        Generate and save/display spatial plots comparing simulations with IBTrACS data for each TC metric.

        Parameters
        ----------
        output_path : str, optional
            Path to save the spatial plots. If None, the plots are displayed but not saved.
        clon : int
            Central longitude for the spatial maps (default: 0).
        """

        # Prepare data and plotting parameters
        data = self.data_cymep
        data_name = self.sim_name
        year_range = f"{self.start_year_tc}-{self.end_year_tc}"

        # Create and save/display spatial plots
        plots_tc.plot_spatial(
            data,
            data_name=data_name,
            year_range=year_range,
            bin_size=self.bin_size,
            output_path=output_path,
            clon=clon,
        )

        return
