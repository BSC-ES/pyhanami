import warnings
warnings.simplefilter("always")

import os
import re
import shutil
import cmocean
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from collections.abc import Iterable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from pyhanami.config import config_params
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData
from pyhanami.utils import data_general, iso_metrics, plot
from pyhanami.utils.tcs_metrics import tcs_tempestextremes, tcs_ibtracs, tcs_cymep_main

class TCMetrics:
    """
    Compute Tropical Cyclones (TCs) metrics.
    
    This class provides functionality for computing various TC metricsfollowing (C.M. Zarzycki et al., 2021) and plotting 
    the results comparing simulations to IBTrACS observational data, as well as, several reanalysis datasets.

    Parameters
    ----------
    data_sim: SimulationData
        Simulation dataset to use.
    start_year_tc, end_year_tc : int, optional
        Initial and end years to compute the TCs metrics for.
    obs : bool
        If True, also consider obsrvational data if available (default: False).
    wind_factor : float
        Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
    min_wind : float
        Minimum 10 m wind speed in m/s for TCs detection (default: 10.0).
    bin_size : float
        Size of the bins in degrees for computing the TCs metrics with CyMeP (default: 2.5).

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
    clim_bias : np.ndarray
        Global mean climatological bias for each TC metric.
    storm_bias : np.ndarray
        Global mean storm bias for each TC metric.
    temp_corr : np.ndarray
        Seasonal correlation for each TC metric.
    spatial_corr : np.ndarray
        Spatial correlation for each TC metric.
    """

    def __init__(self, data_sim : SimulationData, start_year_tc: int = None, end_year_tc: int = None, obs: bool = False, 
                 wind_factor: float = 1.0, min_wind: float = 10.0, bin_size : float = 2.5):
        
        # Validate input
        if not isinstance(data_sim, SimulationData):
            raise TypeError("'data_sim' must be an instance of SimulationData.")   
        self.sim_name = data_sim.name
        self.obs_names = None
        self.obs = obs

        if not isinstance(min_wind, (int, float)) or min_wind < 10.0:
            raise ValueError("The minimum 10 m wind speed for TCs detection 'min_wind' must be a numeric value of at least 10 m/s.")
        self.min_wind = min_wind

        self.config_cymep = {}
        self.bin_size = bin_size
        self.metrics_metadata = data_general.load_yaml_file(config_params.TCS_METRICS_PATH)


        # Select years for TCs analysis
        self.start_year_tc, self.end_year_tc = data_general.validate_year_range(data_sim, start_year_tc, end_year_tc, process_name="TC")
        print(f"\tYears selected for TCs metrics computation: {self.start_year_tc}-{self.end_year_tc}.", flush=True)
        data_sim_tcs = data_sim.data.sel(time=slice(np.datetime64(f"{self.start_year_tc}-01-01"), np.datetime64(f"{self.end_year_tc}-12-31")))

        # Prepare output folder for temporary files
        self.tracks_path = config_params.TC_DATA_PATH / "temp_tracks"
        self.tracks_path.mkdir(parents=True, exist_ok=True)


        # # Prepare IBTrACS data, and observational data if requested
        self._prepare_IBTrACS_data()
        print(f"\tIBTrACS TCs data preprocessed and saved to '{self.config_cymep['IBTrACS'][0]}'.", flush=True)

        if self.obs:
            self.obs_names = ['ERA5', 'JRA55']
            self._prepare_obs_data()
            print(f"\tObservational TCs data preprocessed and saved to '{self.tracks_path}'.", flush=True)

        # Prepare simulation data
        print("\tStarting simulation data TCs tracking. TempestExtremes output:", flush=True)
        self._prepare_sim_data(data_sim_tcs, wind_factor=wind_factor)
        print(f"\tSimulation data TCs tracking completed and saved to '{self.config_cymep[self.sim_name][0]}'.", flush=True)
        

        # Compute TCs metrics
        print("\tStarting Tropical Cyclones metrics computation. CyMeP output:", flush=True)
        self._compute_cymep_metrics()
        print(f"\tTCs metrics computed. See attribute `data_cymep`.", flush=True)

        self._retrieve_biases()
        print("\tBiases computation completed. See attributes `clim_bias` and `storm_bias`.", flush=True)

        self._retrieve_correlations()
        print("\tCorrelations computation completed. See attributes `temp_corr` and `spatial_corr`.", flush=True)

        # # Delete intermediate files
        # shutil.rmtree(self.tracks_path)

        print(f"\nTropical Cyclones metrics computation completed between years {self.start_year_tc} and {self.end_year_tc}.", flush=True)
        return


    def _prepare_IBTrACS_data(self):
        """
        Prepare IBTrACS TCs data to use as a reference for the TCs metrics computation.
        """

        # Retrieve IBTrACS data metadata
        ib_path, ib_end_year = tcs_ibtracs.check_ibtracs_file(self.start_year_tc, self.end_year_tc, output_path=self.tracks_path,
                                                              min_wind=self.min_wind)  

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
            obs_files = list(data_path.glob(f'{name}_*.txt'))
            if obs_files is None:
                raise FileNotFoundError(f"No observational TCs data found for '{name}' in '{data_path}'.")
            obs_path = obs_files[0]
            
            # Check whether the current version covers the selected period
            match = re.search(rf'{name}_(\d+)-(\d+)', obs_path.name)
            obs_start_year = int(match.group(1))
            obs_end_year = int(match.group(2))

            if obs_start_year > self.start_year_tc or obs_end_year < self.end_year_tc:
                warnings.warn(f"The available observational TCs data for '{name}' ({obs_start_year}-{obs_end_year}) does not "
                              f"cover the selected period for TCs metrics computation ({self.start_year_tc}-{self.end_year_tc})."
                              f" This dataset will not be considered for the TCs metrics computation.")
                break

            # Apply wind threshold
            if self.min_wind > 10.0:
                new_obs_path = self.tracks_path / f'{name.lower()}_{obs_start_year}_{obs_end_year}_{self.min_wind:.1f}_False_1_1.0.txt'
                obs_path = tcs_tempestextremes.filter_tracks_by_wind(obs_path, new_obs_path, cutoff_wind=self.min_wind)
            else:
                new_obs_path = self.tracks_path / obs_path.name
                shutil.copy2(obs_path, new_obs_path)

            year_range = obs_end_year - obs_start_year + 1
            obs_config = [str(new_obs_path), name, False, 1, year_range, 1.0]
            self.config_cymep[name] = obs_config

        return


    def _prepare_sim_data(self, data_sim, wind_factor=1.0):
        """
        Prepare simulation data for the TCs metrics computation (detect 
        and track TCs with TempestExtremes).

        Parameters
        ----------
        data_sim : xr.Dataset
            Simulation data.
        wind_factor : float
            Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
        """

        # Run TempestExtremes tracking on simulated data
        tracks_sim_path = tcs_tempestextremes.run_tempestExtremes(data_sim, self.sim_name, self.tracks_path, min_wind=self.min_wind)
        new_sim_path = self.tracks_path / f"{('-').join(self.sim_name.split())}_{self.start_year_tc}-{self.end_year_tc}_{self.min_wind:.1f}_False_1_{wind_factor:.1f}.txt"
        os.rename(tracks_sim_path, new_sim_path)

        # Add simulation data parameters to CyMeP configuration file
        year_range = self.end_year_tc - self.start_year_tc + 1
        self.config_cymep[self.sim_name] = [str(new_sim_path), self.sim_name, False, 1, year_range, wind_factor]

        return


    def _compute_cymep_metrics(self):
        """
        Compute TCs metrics using the CyMeP package.
        """

        # Prepare CyMeP configuration file
        config_cymep_path = self.tracks_path / "cymep_config.csv"
        tcs_cymep_main.prepare_configs_file(self.config_cymep, output_path=config_cymep_path)

        # Run CyMeP TCs metrics computation
        self.data_cymep = tcs_cymep_main.run_cymep_pyhanami(self.start_year_tc, self.end_year_tc, output_path=self.tracks_path, 
                                                            gridsize=self.bin_size, csvfilename=config_cymep_path)
        self.model_names = self.data_cymep.model.values 

        return


    def _retrieve_biases(self):
        """
        Retrieve climatological and storm biases from CyMeP output data,
        and define related plotting parameters.
        """

        # Retrieve biases
        clim_mean = np.transpose([self.data_cymep[f'clim_mean_{metric}'].values for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True])   
        self.clim_bias = np.concatenate(([clim_mean[0]], clim_mean[1:] - clim_mean[0]))

        storm_mean = np.transpose([self.data_cymep[f'storm_mean_{metric}'].values for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True 
                      and metric!='count'])
        self.storm_bias = np.concatenate(([storm_mean[0]], storm_mean[1:] - storm_mean[0]))

        # Define plotting parameters
        self.cbar_ticks_bias = ['Negative bias', 'No bias', 'Positive bias']
        self.colors_bias = ("BlueRed", ['tab:blue', 'white', 'tab:red'])

        return
    

    def _retrieve_correlations(self):
        """
        Retrieve temporal and spatial correlations from CyMeP output data,
        and define related plotting parameters.
        """

        # Retrieve correlations
        self.temp_corr = np.transpose([self.data_cymep[var].values for var in self.data_cymep.data_vars if var.startswith('temporal_scorr_')])
        self.spatial_corr = np.transpose([self.data_cymep[var].values for var in self.data_cymep.data_vars if var.startswith('spatial_pcorr_')])

        # Define plotting parameters
        self.cbar_ticks_corr = ['Negative correlation', 'No correlation', 'Positive correlation']
        self.colors_corr = ("OrangeGreen", ['tab:orange', 'white', 'tab:green'])

        return


    def save_data(self, output_path):
        """
        Save computed EEOFs, PCs, and frequency of ISO events to NetCDF and Numpy files.

        Parameters
        ----------
        output_path : str
            Path to save the data files.
        """

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare dataset names for filenames
        name = f'IBTrACS_{('-').join(self.sim_name.split())}'
        if self.obs:
            name += '_obs'


        # Save all CyMeP output data to NetCDF file
        data_cymep_path = output_path / f'tcs_metrics_{name}_{self.start_year_tc}-{self.end_year_tc}.nc'
        self.data_cymep.to_netcdf(data_cymep_path)
        print(f"TCs metrics computed for '{self.sim_name}' simulations saved to '{data_cymep_path}'.", flush=True)


        # Save biases and correlations to Numpy files
        clim_bias_path = output_path / f'tcs_clim_bias_{name}_{self.start_year_tc}-{self.end_year_tc}.npy'
        np.save(clim_bias_path, self.clim_bias)
        print(f"Global climatological mean bias saved to '{clim_bias_path}'.", flush=True)

        storm_bias_path = output_path / f'tcs_storm_bias_{name}_{self.start_year_tc}-{self.end_year_tc}.npy'
        np.save(storm_bias_path, self.storm_bias)
        print(f"Global storm mean bias saved to '{storm_bias_path}'.", flush=True)

        temp_corr_path = output_path / f'tcs_temp_corr_{name}_{self.start_year_tc}-{self.end_year_tc}.npy'
        np.save(temp_corr_path, self.temp_corr)
        print(f"Global seasonal correlation saved to '{temp_corr_path}'.", flush=True)

        spatial_corr_path = output_path / f'tcs_spatial_corr_{name}_{self.start_year_tc}-{self.end_year_tc}.npy'
        np.save(spatial_corr_path, self.spatial_corr)
        print(f"Global spatial correlation saved to '{spatial_corr_path}'.", flush=True)

        return


    def clim_bias_table(self, output_path=None):
        """
        Generate and save/display table plot with global climatological mean bias for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        cols_clim_bias = np.append([f'{self.bin_size}° x {self.bin_size}°'], [fr'$\overline{{b}}_{{clim,{metric}}}$ ({self.metrics_metadata[metric]["units"]})' 
                                                                            for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True])
        clim_bias_table_plot, _ = plot.table_plot(self.clim_bias, title=f'Global climatological mean bias ({year_range})', col_labels=cols_clim_bias, 
                                                  row_labels=self.model_names, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias)
        
        plot.save_or_show_plot(clim_bias_table_plot, output_path, plot_filename=f"tcs_climatological_bias_table_{('-').join(self.sim_name.split())}_{year_range}",
                               plot_name="Climatological bias table for TCs metrics plot")
        
        return
    

    def storm_bias_table(self, output_path=None):
        """
        Generate and save/display table plot with global storm mean bias for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        cols_storm_bias = np.append([f'{self.bin_size}° x {self.bin_size}°'], [fr'$\overline{{b}}_{{storm,{metric}}}$ ({self.metrics_metadata[metric]["units"]})' 
                                                                               for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True 
                                                                               and metric!='count'])
        storm_bias_table_plot, _ = plot.table_plot(self.storm_bias, title=f'Global storm mean bias ({year_range})', col_labels=cols_storm_bias, 
                                                   row_labels=self.model_names, cbar_ticks=self.cbar_ticks_bias, colors=self.colors_bias)
        
        plot.save_or_show_plot(storm_bias_table_plot, output_path, plot_filename=f"tcs_storm_bias_table_{('-').join(self.sim_name.split())}_{year_range}",
                               plot_name="Storm bias table for TCs metrics plot")
        
        return

    
    def temp_corr_table(self, output_path=None):
        """
        Generate and save/display table plot with global seasonal correlation for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        cols_temp_corr = np.append([f'{self.bin_size}° x {self.bin_size}°'], [fr'$\rho_{{s,{metric}}}$' for metric in self.metrics_metadata 
                                                                              if self.metrics_metadata[metric]['temporal']==True])
        temp_corr_table_plot, _ = plot.table_plot(self.temp_corr, title=f'Global seasonal correlation ({year_range})', col_labels=cols_temp_corr, 
                                                  row_labels=self.model_names, cbar_ticks=self.cbar_ticks_corr, colors=self.colors_corr)
        
        plot.save_or_show_plot(temp_corr_table_plot, output_path, plot_filename=f"tcs_seasonal_corr_table_{('-').join(self.sim_name.split())}_{year_range}",
                               plot_name="Seasonal correlation table for TCs metrics plot")

        return
    
    
    def spatial_corr_table(self, output_path=None):
        """
        Generate and save/display table plot with global spatial correlation for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """

        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        cols_spatial_corr = np.append([f'{self.bin_size}° x {self.bin_size}°'], [fr'$r_{{xy,{metric}}}$' for metric in self.metrics_metadata 
                                                                                 if self.metrics_metadata[metric]['spatial']==True])
        spatial_corr_table_plot, _ = plot.table_plot(self.spatial_corr, title=f'Global spatial correlation ({year_range})', col_labels=cols_spatial_corr,
                                                     row_labels=self.model_names, cbar_ticks=self.cbar_ticks_corr, colors=self.colors_corr)
        
        plot.save_or_show_plot(spatial_corr_table_plot, output_path, plot_filename=f"tcs_spatial_corr_table_{('-').join(self.sim_name.split())}_{year_range}",
                               plot_name="Spatial correlation table for TCs metrics plot")

        return


    def linear_plots(self, output_path=None):
        """
        Generate and save/display linear plots comparing all datasets for each TC metric.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the linear plots. If None, the plots are displayed but not saved.
        """

        # Prepare metrics metadata
        temporal_metrics = [metric for metric in self.metrics_metadata if self.metrics_metadata[metric]['temporal']==True]
        linear_metrics_names = [self.metrics_metadata[metric]['short_name'] for metric in temporal_metrics]
        linear_metrics_units = [self.metrics_metadata[metric]['units'] for metric in temporal_metrics]

        # Prepare labels and titles
        linear_ylabel = [f'{name} ({unit})' for name, unit in zip(linear_metrics_names, linear_metrics_units)]
        linear_month_titles = [f'{name} seasonal cycle' for name in linear_metrics_names] 
        linear_year_titles = [f'{name} interannual cycle' for name in linear_metrics_names]

        # Prepare invariant arrays
        months = np.arange(1, 13, dtype=int)
        years = np.arange(self.start_year_tc, self.end_year_tc+1, dtype=int)  
        year_range = f"{self.start_year_tc}-{self.end_year_tc}"


        # Create plots
        for i, name in enumerate(linear_metrics_names):
            # Create line plot for monthly cycles
            linear_month_data = self.data_cymep[f'per_month_{temporal_metrics[i]}'].rename({'month': 'time'})
            linear_month_data_list = [linear_month_data.sel(model=model) for model in self.model_names]

            plt.figure(figsize=(10, 6))
            for j, month_data in enumerate(linear_month_data_list):
                plt.plot(months, month_data, 'o-', markersize=4, linewidth=1.2, label=self.model_names[j])
            plt.xticks(months, months)
            plt.xlabel('month', fontsize=12)
            plt.ylabel(linear_ylabel[i], fontsize=12)
            plt.title(linear_month_titles[i], fontsize=16)
            plt.grid(True)
            plt.legend()

            plot.save_or_show_plot(plt.gcf(), output_path, plot_name=f"Linear monthly cycle plot for TC {name}",
                                   plot_filename=f"tcs_{name.lower()}_monthly_cycle_plot_{self.sim_name}_{year_range}")

            # Create line plot for interannual cycles
            linear_year_data = self.data_cymep[f'per_year_{temporal_metrics[i]}'].rename({'year': 'time'})
            linear_year_data_list = [linear_year_data.sel(model=model) for model in self.model_names]              

            plt.figure(figsize=(10, 6))
            for j, year_data in enumerate(linear_year_data_list):
                plt.plot(years, year_data, 'o-', markersize=4, linewidth=1.2, label=self.model_names[j])
            plt.xlabel('year', fontsize=12)
            plt.ylabel(linear_ylabel[i], fontsize=12)
            plt.title(linear_year_titles[i], fontsize=16)
            plt.grid(True)
            plt.legend()

            plot.save_or_show_plot(plt.gcf(), output_path, plot_name=f"Linear interannual cycle plot for TC {name}",
                                   plot_filename=f"tcs_{name.lower()}_interannual_cycle_plot_{('-').join(self.sim_name.split())}_{year_range}")

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

        # Prepare metrics metadata
        spatial_metrics = [metric for metric in self.metrics_metadata if self.metrics_metadata[metric]['spatial']==True]
        spatial_metrics_names = [self.metrics_metadata[metric]['short_name'] for metric in spatial_metrics]
        spatial_metrics_units = [self.metrics_metadata[metric]['units'] for metric in spatial_metrics]

        # Prepare labels and titles
        year_range = f"{self.start_year_tc}-{self.end_year_tc}"
        spatial_titles = [f'TC {name} density for {self.bin_size}°x{self.bin_size}° cells ({year_range})' for name in spatial_metrics_names]
        spatial_bias_titles = [f"TC {name} bias ('{self.sim_name}' vs 'IBTrACS') for {self.bin_size}°x{self.bin_size}° cells ({year_range})" for name in spatial_metrics_names]
        spatial_cb_labels = [f'{name} ({unit})' for name, unit in zip(spatial_metrics_names, spatial_metrics_units)]


        # Create plots
        for i, name in enumerate(spatial_metrics_names):

            # Create a modified colormap with white for NaN values
            cmap_modified = cmocean.cm.thermal_r.copy()
            # cmap_modified.set_bad('white')


            spatial_abs_data = self.data_cymep[f'spatial_abs_{spatial_metrics[i]}']
            spatial_abs_data = spatial_abs_data.where(spatial_abs_data != 0)  # Set zero values to NaN for better visualization
            spatial_abs_plot, _ = plot.two_spatial_plots(spatial_abs_data.sel(model='IBTrACS'), spatial_abs_data.sel(model=self.sim_name), clon=clon,
                                                         title_1='IBTrACS', title_2=self.sim_name, suptitle=spatial_titles[i], cb_label=spatial_cb_labels[i],
                                                         cmap=cmap_modified)
            plot.save_or_show_plot(spatial_abs_plot, output_path, plot_filename=f"tcs_{name.lower()}_spatial_abs_plot_{('-').join(self.sim_name.split())}_{year_range}_clon_{clon}",
                                    plot_name=f"Spatial plot for TC {name}")

            spatial_bias_data = self.data_cymep[f'spatial_bias_{spatial_metrics[i]}'].sel(model=self.sim_name)
            limit = np.ceil(np.nanmax(np.abs(spatial_bias_data.values)))
            levels = np.linspace(-limit, limit, 13)
            spatial_bias_plot, _ = plot.spatial_plot(spatial_bias_data, clon=clon, title=spatial_bias_titles[i], cb_label=f'bias in {spatial_cb_labels[i]}', 
                                                     cmap=LinearSegmentedColormap.from_list(*self.colors_bias), levels=levels)
                                                     #cmap=cmocean.cm.diff)
            plot.save_or_show_plot(spatial_bias_plot, output_path, plot_filename=f"tcs_{name.lower()}_spatial_bias_plot_{('-').join(self.sim_name.split())}_{year_range}_clon_{clon}",
                                   plot_name=f"Spatial bias plot for TC {name}")
            
        return


class ScientificEvaluation:
    """
    Compute and plot metrics for scientific model skill evaluation.

    This class provides functionality for computing and visualizing metrics to evaluate how well a model 
    reproduces several phenomena. Currently, it includes methods for bimodal ISO indices.
    
    Parameters
    ----------
    datasets : SimulationData or Iterable[SimulationData], optional
        Ensemble or list of ensembles containing simulation data and metadata.

    Attributes
    ----------
    datasets : list[SimulationData]
        List of ensembles containing simulation data and metadata.
    variables : dict
        Configuration dictionary mapping variable names to display metadata.
    """    

    def __init__(self, datasets: Iterable[SimulationData] = None):        
        if datasets is None:
            self.datasets = []
        else:
            if isinstance(datasets, SimulationData):
                self.datasets = [datasets]
            elif isinstance(datasets, Iterable) and not isinstance(datasets, (str, bytes)) \
                and all(isinstance(ds, SimulationData) for ds in datasets):
                self.datasets = list(datasets)
            else:
                raise TypeError("Input must be a SimulationData object or an iterable of SimulationData objects.")

        # Load config parameters once
        self.variables = data_general.load_yaml_file(config_params.VARIABLES_PATH)

        return


    def add_datasets(self, datasets):
        """ 
        Add new datasets to the ScientificEvaluation object.

        Parameters
        ----------
        datasets : SimulationData or Iterable[SimulationData])
            Ensemble or list of ensembles containing simulation data and metadata to add.
        """

        # Validate input
        if isinstance(datasets, SimulationData):
            datasets = [datasets]
        elif not isinstance(datasets, Iterable) or isinstance(datasets, (str, bytes)) \
            or not all(isinstance(ds, SimulationData) for ds in datasets):
            raise TypeError("Input must be a SimulationData object or an iterable of SimulationData objects.")
        
        # Check for duplicate datasets
        for dataset in datasets:
            if not any(ds.name == dataset.name for ds in self.datasets):
                self.datasets.append(dataset)
            else:
                warnings.warn(f"Dataset with name '{dataset.name}' already exists in the ScientificEvaluation object. Skipping addition.")

        return
    
    
    def bimodal_ISO(self, data_name=None, output_path=None, start_year_eeof=None, end_year_eeof=None, plot_eeofs=False,
                    years_pc=None, correct_pc=False, obs=False, obs_path=None, obs_name=None, clon=0, lat_range=(-30, 30), 
                    lags=[-10, -5, 0], n_modes=2, window=141, low_freq=1/90, high_freq=1/25):
        """
        Compute bimodal ISO indices following (K. Kikuchi, 2020) and plot results for the selected years.  
        Moreover, compute temporal correlation, standard deviations ratio and Taylor Skill Score between 
        observations and simulations mean monthly frequency of ISO events following (M. Nakano et al., 2019) 
        when an observations dataset is provided.

        Parameters
        ----------
        data_name : str, optional
            Name of simulation ensemble to use. If None, the first dataset in the 
            ScientificEvaluation object is used.
        output_path : str, optional
            Path to save plots.
        start_year_eeof, end_year_eeof : int
            Initial and end years to compute the TSS for.
        plot_eeofs : bool
            If True, also spatially plot EEOFs (default: False).
        years_pc : int or list[int], optional
            Years to compute the indices for.
        correct_pc : bool
            Whether to adjust simulated PCs by dividing by alpha (default: False).
        obs : bool
            If True, also plot observational data if available (default: False).
        obs_path : str or list[str], optional
            Path to the observations database.
        obs_name : str or list[str], optional
            Name of the observational dataset.
        clon : int
            Central longitude for the spatial EEOFs maps (default: 0).
        lat_range : tuple
            Geographic latitude bounds (default: (-30, 30)).
        lags : list[int]
            Lag values to consider (default: [-10, -5, 0]).
        n_modes : int)
            Number of EEOFs modes to compute (default: 2).
        window_size : int
            Length of the filter kernel (default: 141).
        low_freq : float
            Lower cutoff frequency (default: 1/90).
        high_freq : float
            Upper cutoff frequency (default: 1/25).

        Returns
        -------
        corr : float, optional
            Temporal correlation coefficient of the ISO seasonality (returned only if `obs=True`).
        sigma : float, optional
            Ratio of the standard deviations (model/obs) of the ISO seasonality (returned only if `obs=True`).
        tss : float, optional
            Taylor Skill Score of the ISO seasonality (returned only if `obs=True`).
        """

        # Validate input
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError("At least one dataset is required for the Bimodal ISO indices.")
            data_plot = self.datasets[0]
            data_name = data_plot.name
        elif isinstance(data_name, str):
            data_plot = [ds for ds in self.datasets if ds.name == data_name]
            if not data_plot:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_plot = data_plot[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")
        
        # Prepare output path if given
        if output_path is not None:
            output_path = Path(output_path)
            if output_path.suffix != '':  
                raise ValueError("Output path must be a directory, not a file path, as multiple files may be created.")
            
            output_path.mkdir(parents=True, exist_ok=True)
        

        # Prepare simulated and observed OLR data
        var_name = "rlut"
        if var_name not in data_plot.data.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{data_name}'. "
                            f"Available variables: {list(data_plot.data.data_vars.keys())}")
        olr_data_unfiltered = data_plot.data[var_name].sortby("lat").sel(lat=slice(*lat_range)).compute()
        olr_data = iso_metrics.apply_lanczos_bandpass_filter(olr_data_unfiltered, window, low_freq, high_freq)

        olr_years = olr_data.time.dt.year
        start_year = int(olr_years.min())
        end_year = int(olr_years.max())
        if start_year_eeof is None:
            start_year_eeof = start_year
        if end_year_eeof is None:
            end_year_eeof = end_year

        if years_pc is not None:
            if isinstance(years_pc, int):
                years_pc = [years_pc]
            if not any(year in olr_years.values for year in years_pc):
                raise ValueError(f"Some years are missing in the selected dataset {data_name}.")
        
        if obs:
            data_obs = ObservationData(obs_path, olr_data.to_dataset(), name=obs_name)
            olr_obs_unfiltered = data_obs.data[var_name].sortby("lat").sel(lat=slice(*lat_range)).compute()
            olr_obs = iso_metrics.apply_lanczos_bandpass_filter(olr_obs_unfiltered, window, low_freq, high_freq)


        # Conduct EEOF analysis and plot if requested
        if obs:
            name = obs_name
            data_eeof = olr_obs
        else:
            name = data_name
            data_eeof = olr_data

        eeof_winter = iso_metrics.perform_EEOF_analysis(data_eeof, start_year_eeof, end_year_eeof, 'boreal_winter', lags, n_modes)
        eeof_summer = iso_metrics.perform_EEOF_analysis(data_eeof, start_year_eeof, end_year_eeof, 'boreal_summer', lags, n_modes)

        if plot_eeofs:
            eeofw_plot, _ = plot.eeofs_plot(eeof_winter, clon=clon, title=f'MJO convective pattern {name} (DJFMA {start_year_eeof}-{end_year_eeof})', 
                                            cb_label=f'scaled EEOF ({self.variables[var_name][1]})', cmap=LinearSegmentedColormap.from_list("BlueRed", ['tab:blue', 'white', 'tab:red']))            
            
            if output_path is None:
                plt.show()
                print("MJO EEOFs plot created and displayed.", flush=True)
            else:
                eeofw_path = output_path / f"eeof_boreal_winter_{name}_{start_year_eeof}-{end_year_eeof}"

                # eeof_winter.to_netcdf(eeofw_path.with_suffix('.nc'))
                eeofw_plot.savefig(eeofw_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
                print(f"MJO EEOFs plot created and saved to '{eeofw_path.with_suffix('.png')}'.", flush=True)

            eeofs_plot, _ = plot.eeofs_plot(eeof_summer, clon=clon, title=f'BSISO convective pattern {name} (JJASO {start_year_eeof}-{end_year_eeof})',
                                            cb_label=f'scaled EEOF ({self.variables[var_name][1]})', cmap=LinearSegmentedColormap.from_list("GreenOrange", ['tab:green', 'white', 'tab:orange']))       
            if output_path is None:
                plt.show()
                print("BSISO EEOFs plot created and displayed.", flush=True)
            else:
                eeofs_path = output_path / f"eeof_boreal summer_{name}_{start_year_eeof}-{end_year_eeof}"
                
                # eeof_winter.to_netcdf(eeofw_path.with_suffix('.nc'))
                eeofs_plot.savefig(eeofs_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
                print(f"BSISO EEOFs plot created and saved to '{eeofs_path.with_suffix('.png')}'.", flush=True)
        print('EEOF analysis completed.\n', flush=True)
        

        # Compute PCs and plot if requested
        pcs_sim = iso_metrics.compute_PCs(olr_data, [eeof_winter, eeof_summer])
        alpha = None
        if obs:
            pcs_obs = iso_metrics.compute_PCs(olr_obs, [eeof_winter, eeof_summer])
            if correct_pc:
                pcs_sim, alpha = iso_metrics.adjust_PCs(pcs_sim, pcs_obs)
                print(f'Simulated PCs have been adjusted using the {obs_name} observations.', flush=True)
        else:
            if correct_pc:
                warnings.warn(f"Simulated PCs cannot be adjusted if observations are not provided. Execution will continue without modifying the PCs.")
                
        if years_pc is not None:
            for year in years_pc:
                pcs_year = pcs_sim.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
                pcs_plot, _ = plot.pcs_plot(pcs_year, title=f'Bimodal ISO indices {data_name} ({year})')
                
                if output_path is None:
                    plt.show()
                    print(f"PCs (bimodal ISO indices) for year {year} plot created and displayed.", flush=True)
                else:
                    pcs_path = output_path / f"pcs_{data_name}_{year}_projected_{start_year_eeof}-{end_year_eeof}"

                    # pcs_year.to_netcdf(pcs_path.with_suffix('.nc'))
                    pcs_plot.savefig(pcs_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
                    print(f"PCs (bimodal ISO indices) plot for year {year} created and saved to '{pcs_path.with_suffix('.png')}'.", flush=True)
        print('PCs (bimodal ISO indices) computation completed.\n', flush=True)

        
        # Compute and plot monthly frequency of ocurrence of ISO events
        events_sim = pcs_sim['label']
        freq_ISO_sim = iso_metrics.compute_freq_ISO(events_sim)
        
        if obs:
            name = f'{data_name}_{obs_name}'
            events_obs = pcs_obs['label']
            freq_ISO_obs = iso_metrics.compute_freq_ISO(events_obs)
            corr, sigma, tss = iso_metrics.compute_TSS(freq_ISO_sim, freq_ISO_obs)
        else:
            name = data_name
            freq_ISO_obs, corr, sigma, tss = None, None, None, None

        freq_plot, _ = plot.freq_ISO_plot(freq_ISO_sim, freq_ISO_obs, alpha=alpha, corr=corr, sigma=sigma, tss=tss,
                                          title=f'Mean monthly frequency of ISO events ({start_year}-{end_year})',
                                          sim_label=data_name, obs_label=obs_name)

        if output_path is None:
            plt.show()
            print(f"Mean monthly frequency of ISO events plot created and displayed.", flush=True)
        else:
            freq_path = output_path / f"freq_ISO_{name}_{start_year}-{end_year}"

            # freq_ISO_sim.to_netcdf(freq_path.with_suffix('.nc'))
            freq_plot.savefig(freq_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
            print(f"Mean monthly frequency of ISO events plot created and saved to '{freq_path.with_suffix('.png')}'.", flush=True)
        print('Mean monthly frequency computation completed.\n', flush=True)

        if obs:
            print(f"Computed Taylor Skill Score (TSS) between simulations and observations:\n" + 
            f"\tTemporal correlation (R): {corr:.2f}, Ratio standard deviations ($\\sigma$): {sigma:.2f}, TSS: {tss:.2f}\n", flush=True)
            return corr, sigma, tss
        else:
            return 
        
    
    def compute_tc_metrics(self, data_name=None, start_year_tc=None, end_year_tc=None, obs=False, wind_factor=1.0, min_wind=10, 
                           bin_size=2.5):
        """
        Compute Tropical Cyclones (TCs) metrics following (C.M. Zarzycki et al., 2021) and plot results.

        Parameters
        ----------
        data_name : str, optional
            Name of simulation ensemble to use. If None, the first dataset in the ScientificEvaluation 
            object is used.
        start_year_tc, end_year_tc : int, optional
            Initial and end years to compute the TCs metrics for.
        obs : bool
            If True, include observational data if available (default: False).
        wind_factor : float
            Wind speed correction factor (to normalize the provided wind to 10 m wind) for simulations (default: 1.0).
        min_wind : float
            Minimum 10 m wind speed in m/s for TCs detection (default: 10.0).
        bin_size : float
            Size of the bins in degrees for computing the TCs metrics with CyMeP (default: 2.5).
        """

        # Validate input
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError("At least one dataset is required for the Tropical Cyclones metrics.")
            data_TC = self.datasets[0]
            data_name = data_TC.name
        elif isinstance(data_name, str):
            data_TC = [ds for ds in self.datasets if ds.name == data_name]
            if not data_TC:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_TC = data_TC[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")

        # Create a TCMetrics object and compute metrics
        print(f"Performing TCs analysis for dataset '{data_name}':", flush=True)
        tc_metrics = TCMetrics(data_TC, start_year_tc=start_year_tc, end_year_tc=end_year_tc, obs=obs, wind_factor=wind_factor, min_wind=min_wind,
                               bin_size=bin_size)

        return tc_metrics

        # input_path = data_TC.data_path

        # if obs:
        #     if obs_path is None or obs_name is None or obs_wind_factor is None:
        #         raise NotImplementedError('Automatic selection of observations is not implemented yet. '
        #                                   'Please provide at least one path, one name and the corresponding wind factor if you want to include observations.')
            
        #     # Convert to lists if single values are provided                
        #     obs_path = [obs_path] if isinstance(obs_path, (str, Path)) else list(obs_path)
        #     obs_name = [obs_name] if isinstance(obs_name, str) else list(obs_name)
        #     obs_wind_factor = [obs_wind_factor] if isinstance(obs_wind_factor, (int, float)) else list(obs_wind_factor)

        #     # Validate lengths match
        #     if not (len(obs_path) == len(obs_name) == len(obs_wind_factor)):
        #         raise ValueError("'obs_path', 'obs_name' and 'obs_wind_factor' must have the same length.")

                




        # # Plot TC genesis and trajectory density if requested
        # if full_output:
        #     # Get simulated tracks and counts
        #     sim_tracks = tcs_tempestextremes.read_tracks_tempestExtremes(tracks_sim_path)
        #     sim_counts_gen, sim_counts_traj = tcs_tempestextremes.compute_tc_counts(sim_tracks, start_year, end_year, bin_size=bin_size, cutoff_wind=min_wind)

        #     # Get IBTrACS tracks and counts
        #     ib_tracks = tcs_tempestextremes.read_tracks_tempestExtremes(ib_path)
        #     ib_counts_gen, ib_counts_traj = tcs_tempestextremes.compute_tc_counts(ib_tracks, start_year, end_year, bin_size=bin_size, cutoff_wind=min_wind)



        # # Prepare observations TCs data if requested
        # if obs:
        #     obs_tracks_path = config_params.TC_DATA_PATH
        #     for name, path, wind in zip(obs_name, obs_path, obs_wind_factor):
        #         # Check if TCs data is already present for the selected years, minimum wind and observations dataset
        #         obs_files = list(obs_tracks_path.glob(f"{name}_*.txt"))

        #         found = False
        #         for obs_file in obs_files:
        #             parts = obs_file.stem.split('_')

        #             if len(parts) >= 2 and '-' in parts[1]:
        #                 year_range = parts[1]
        #                 try:
        #                     # Check if the file covers the selected period
        #                     file_start, file_end = map(int, year_range.split('-'))                            
        #                     if file_start <= start_year and file_end >= end_year:

        #                         # Check if the file matches the selected min_wind
        #                         obs_min_wind = float(parts[2])
        #                         if abs(obs_min_wind - min_wind) < 1e-6:
        #                             unstructured = parts[3].lower() == 'true'
        #                             ens_members = int(parts[4])
        #                             aux_wind_factor = float(parts[5])
        #                             found = True
        #                             break
        #                 except ValueError:
        #                     continue
        #         if found:
        #             configs[name] = [obs_file.name, name.lower(), unstructured, ens_members, years, aux_wind_factor]

        #         # Compute TCs data for observations if not already present
        #         else:
        #             # Load observations
        #             var_names = ["psl", "uas", "vas", "zg300", "zg500"]
        #             data_sim_selected = data_sim_all[var_names]
        #             data_obs = ObservationData(path, data_sim_selected, name=name)
                    
        #             ens_members = 1 if 'realization' not in data_obs.data.dims else data_obs.data.dims['realization']
        #             unstructured = False

        #             # Run TempestExtremes tracking on observational data
        #             print(f'Starting Tropical Cyclones tracking using TempestExtremes for {name} observations...', flush=True)
        #             tracks_obs_path = tcs_tempestextremes.run_tempestExtremes(data_obs, name, obs_tracks_path, min_wind=min_wind)
        #             tracks_obs_path = tracks_obs_path.rename(tracks_obs_path.with_name(f"{name}_{start_year}-{end_year}_{min_wind:.1f}_{unstructured}_{ens_members}_{wind:.1f}.txt"))
        #             print(f"Tropical Cyclones tracking completed for {name} observations. Output files added to '{obs_tracks_path}'.", flush=True)

        #             configs[name] = [tracks_obs_path, name, unstructured, ens_members, years, wind]