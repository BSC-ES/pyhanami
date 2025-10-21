import warnings
warnings.simplefilter("always")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from collections.abc import Iterable
from matplotlib.colors import LinearSegmentedColormap

from pyhanami.config import config_params
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData
from pyhanami.utils import data_general, iso_metrics, plot
from pyhanami.utils.tcs_metrics import tcs_tempestextremes, tcs_ibtracs, tcs_cymep_main

import time


class ScientificEvaluation:
    """
    Compute and plot metrics for scientific model skill evaluation.

    This class provides functionality for computing and visualizing metrics to evaluate how well a model 
    reproduces several phenomena. Currently, it includes methods for bimodal ISO indices.
    
    Parameters
    ----------
    datasets : SimulationData or Iterable[SimulationData]
        Ensemble or list of ensembles containing simulation data and metadata.

    Attributes
    ----------
    datasets : list[SimulationData]
        List of ensembles containing simulation data and metadata.
    variables : dict
        Configuration dictionary mapping variable names to display metadata.

    Methods
    -------
    add_datasets(datasets)
        Adds new datasets to the ScientificEvaluation object.

    bimodal_ISO(data_name=None, output_path=None, start_year_eeof=None, end_year_eeof=None, plot_eeofs=False, years_pc=None, 
                correct_pc=False, obs=False, obs_path=None, obs_name=None, clon=0, lat_range=(-30, 30), lags=[-10, -5, 0], 
                n_modes=2, window=141, low_freq=1/90, high_freq=1/25)
        Computes bimodal ISO indices following (K. Kikuchi, 2020) and plots results for the selected years.  
        Moreover, computes temporal correlation, standard deviations ratio and Taylor Skill Score between 
        observations and simulations mean monthly frequency of ISO events following (M. Nakano et al., 2019) 
        when an observations dataset is provided.
    tcs_metrics(data_name=None, wind_factor=1.0, output_path=None, start_year=None, end_year=None, min_wind=10.0, full_output=False, 
                bin_size=2.5, clon=0, obs=False, obs_path=None, obs_name=None, obs_wind_factor=None)
        Computes Tropical Cyclones (TCs) metrics following (C.M. Zarzycki et al., 2021) and plots results.
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
        datasets (SimulationData or Iterable[SimulationData]): Ensemble or list of ensembles containing simulation 
                                                                data and metadata to add.
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
        data_name (str): Name of simulation ensemble to use.
        output_path (str): Path to save plots.
        start_year_eeof, end_year_eeof (int): Initial and end years to compute the TSS for.
        plot_eeofs (bool): If True, also spatially plot EEOFs (default: False).
        years_pc (int or list[int]): Years to compute the indices for.
        correct_pc (bool): Whether to adjust simulated PCs by dividing by alpha (default: False).
        obs (bool): If True, also consider and plot observational data if available (default: False).
        obs_path (str): Path to the observations database.
        obs_name (str): Name of the observational dataset.
        clon (int): Central longitude for the spatial EEOFs maps (default: 0).
        lat_range (tuple): Geographic latitude bounds (default: (-30, 30)).
        lags (list[int]): Lag values to consider (default: [-10, -5, 0]).
        n_modes (int): Number of EEOFs modes to compute (default: 2).
        window_size (int): Length of the filter kernel (default: 141).
        low_freq (float): Lower cutoff frequency (default: 1/90).
        high_freq (float): Upper cutoff frequency (default: 1/25).

        Returns
        -------
        if `obs=True`:
            corr (float): Temporal correlation coefficient of the ISO seasonality.
            sigma (float): Ratio of the standard deviations (model/obs) of the ISO seasonality.
            tss (float): Taylor Skill Score of the ISO seasonality.
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
        var_name = "olr"
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
        
    
    def tcs_metrics(self, data_name=None, wind_factor=1.0, output_path=None, start_year=None, end_year=None, min_wind=10.0, 
                    full_output=False, bin_size=2.5, clon=0, obs=False, obs_path=None, obs_name=None, 
                    obs_wind_factor=None):
        """
        Compute Tropical Cyclones (TCs) metrics following (C.M. Zarzycki et al., 2021) and plot results.

        Parameters
        ----------
        data_name (str): Name of simulation ensemble to use.
        wind_factor (float): Wind speed correction factor (to normalize provided wind to 10 m wind) for simulations (default: 1.0).
        output_path (str): Path to save plots.
        start_year, end_year (int): Initial and end years to compute the TCs metrics for.
        min_wind (float): minimum 10 m wind speed in m/s for TCs detection (default: 10.0).
        full_output (bool): Whether to include spatial and linear plots from the CyMeP output (default: False).
        bin_size (float): Size of the bins in degrees for the spatial density plots (default: 2.5).
        clon (int): Central longitude for the spatial maps (default: 0).
        obs (bool): Whether to consider observational data if available (default: False).
        obs_path (str or list[str]): Path/s to the observations database/s.
        obs_name (str or list[str]): Name/s of the observational dataset/s.
        obs_wind_factor (float or list[float]): Wind speed correction factor/s (to normalize provided wind to 10 m wind) for observations.
        """

        # Validate input
        if data_name is None:
            if len(self.datasets) < 1:
                raise ValueError("At least one dataset is required for the Tropical Cyclones metrics.")
            data_plot = self.datasets[0]
            data_name = data_plot.name
        elif isinstance(data_name, str):
            data_plot = [ds for ds in self.datasets if ds.name == data_name]
            if not data_plot:
                raise ValueError(f"Dataset with name '{data_name}' not found in the ScientificEvaluation object.")
            data_plot = data_plot[0]
        else:
            raise TypeError("'data_name' must be a string representing a dataset name.")
        input_path = data_plot.data_path

        if obs:
            if obs_path is None or obs_name is None or obs_wind_factor is None:
                raise NotImplementedError('Automatic selection of observations is not implemented yet. '
                                          'Please provide at least one path, one name and the corresponding wind factor if you want to include observations.')
            else:                
                obs_path = [obs_path] if isinstance(obs_path, (str, Path)) else list(obs_path)
                obs_name = [obs_name] if isinstance(obs_name, str) else list(obs_name)
                obs_wind_factor = [obs_wind_factor] if isinstance(obs_wind_factor, (int, float)) else list(obs_wind_factor)
                if len(obs_path) != len(obs_name) or len(obs_path) != len(obs_wind_factor):
                    raise ValueError("'obs_path', 'obs_name' and 'obs_wind_factor' must have the same length.")

                

        # Prepare output path
        if output_path is not None:
            output_path = Path(output_path)
            if output_path.suffix != '':  
                raise ValueError("Output path must be a directory, not a file path, as multiple files may be created.")
            
            output_path.mkdir(parents=True, exist_ok=True)
            tracks_path = output_path
        else:
            tracks_path = input_path.parent / f"tropical_cyclones_metrics_{data_name}_output"


        # Prepare simulated data
        data_years = data_plot.data.time.dt.year
        start_year_data = int(data_years.min())
        end_year_data = int(data_years.max())
        if start_year is None:
            start_year = start_year_data
            print(f"As no start year was provided, the first year available in the {data_name} dataset ({start_year}) will be used.", flush=True)
        if end_year is None:
            end_year = end_year_data
            print(f"As no end year was provided, the last year available in the {data_name} dataset ({end_year}) will be used.", flush=True)

        data_sim_all = data_plot.data.sel(time=slice(np.datetime64(f"{start_year}-01-01"), np.datetime64(f"{end_year}-12-31")), method="nearest")
        if data_sim_all.time.size == 0:
            raise ValueError(f"No data available in the {data_name} dataset in the selected years {start_year}-{end_year}.")


        # Run TempestExtremes tracking on simulated data
        tracks_sim_path = tcs_tempestextremes.run_tempestExtremes(data_sim_all, data_name, tracks_path, min_wind=min_wind)
        print(f"Tropical Cyclones tracking completed for {data_name}. Output files saved to '{tracks_path}'.", flush=True)


        # Prepare IBTrACS TCs data
        configs = {}
        years = end_year - start_year + 1
        ib_path = tcs_ibtracs.check_ibtracs_file(start_year, end_year, min_wind=min_wind)  
        ib_config = [ib_path, "IBTrACS", False, 1, years, 1.0]
        configs["IBTrACS"] = ib_config


        # Plot TC genesis and trajectory density if requested
        if full_output:
            # Get simulated tracks and counts
            sim_tracks = tcs_tempestextremes.read_tracks_tempestExtremes(tracks_sim_path)
            sim_counts_gen, sim_counts_traj = tcs_tempestextremes.compute_tc_counts(sim_tracks, start_year, end_year, bin_size=bin_size, cutoff_wind=min_wind)

            # Get IBTrACS tracks and counts
            ib_tracks = tcs_tempestextremes.read_tracks_tempestExtremes(ib_path)
            ib_counts_gen, ib_counts_traj = tcs_tempestextremes.compute_tc_counts(ib_tracks, start_year, end_year, bin_size=bin_size, cutoff_wind=min_wind)


            # Generate genesis density plots
            gen_plot, _ = plot.two_spatial_plots(ib_counts_gen, sim_counts_gen, clon=clon, title_1='IBTrACS', title_2=data_name,
                                                  suptitle=f"TCs genesis density per {bin_size}°x{bin_size}° cell ({start_year}-{end_year})",
                                                  cb_label="N° of tropical cyclones formed", show_contours=False)
            plot.save_or_show_plot(gen_plot, output_path, plot_filename=f"tcs_genesis_density_ibtracs_vs_{data_name}_{start_year}-{end_year}",
                                   plot_name="\nTC genesis plot")

            # Generate trajectory density plots
            traj_plot, _ = plot.two_spatial_plots(ib_counts_traj, sim_counts_traj, clon=clon, title_1='IBTrACS', title_2=data_name,
                                                  suptitle=f"TCs trajectory density per {bin_size}°x{bin_size}° cell ({start_year}-{end_year})",
                                                  cb_label="N° of tropical cyclones passed", show_contours=False)
            plot.save_or_show_plot(traj_plot, output_path, plot_filename=f"tcs_trajectory_density_ibtracs_vs_{data_name}_{start_year}-{end_year}",
                                   plot_name="TC trajectories plot")


        # Prepare observations TCs data if requested
        if obs:
            obs_tracks_path = config_params.DATA_PATH
            for name, path, wind in zip(obs_name, obs_path, obs_wind_factor):
                # Check if TCs data is already present for the selected years, minimum wind and observations dataset
                obs_files = list(obs_tracks_path.glob(f"{name}_*.txt"))

                found = False
                for obs_file in obs_files:
                    parts = obs_file.stem.split('_')

                    if len(parts) >= 2 and '-' in parts[1]:
                        year_range = parts[1]
                        try:
                            # Check if the file covers the selected period
                            file_start, file_end = map(int, year_range.split('-'))                            
                            if file_start <= start_year and file_end >= end_year:

                                # Check if the file matches the selected min_wind
                                obs_min_wind = float(parts[2])
                                if abs(obs_min_wind - min_wind) < 1e-6:
                                    unstructured = parts[3].lower() == 'true'
                                    ens_members = int(parts[4])
                                    aux_wind_factor = float(parts[5])
                                    found = True
                                    break
                        except ValueError:
                            continue
                if found:
                    configs[name] = [obs_file.name, name.lower(), unstructured, ens_members, years, aux_wind_factor]

                # Compute TCs data for observations if not already present
                else:
                    # Load observations
                    var_names = ["psl", "uas", "vas", "zg300", "zg500"]
                    data_sim_selected = data_sim_all[var_names]
                    data_obs = ObservationData(path, data_sim_selected, name=name)
                    
                    ens_members = 1 if 'realization' not in data_obs.data.dims else data_obs.data.dims['realization']
                    unstructured = False

                    # Run TempestExtremes tracking on observational data
                    tracks_obs_path = tcs_tempestextremes.run_tempestExtremes(data_obs, name, obs_tracks_path, min_wind=min_wind)
                    tracks_obs_path = tracks_obs_path.rename(tracks_obs_path.with_name(f"{name}_{start_year}-{end_year}_{min_wind:.1f}_{unstructured}_{ens_members}_{wind:.1f}.txt"))
                    print(f"Tropical Cyclones tracking completed for {name} observations. Output files added to '{obs_tracks_path}'.", flush=True)

                    configs[name] = [tracks_obs_path, name, unstructured, ens_members, years, wind]


        # Create configuration file for CyMeP with the simulations parameters in the last row
        configs[data_name] = [tracks_sim_path, data_name, False, 1, years, wind_factor]
        tcs_cymep_main.prepare_read_configs(configs)

        # Compute TCs metrics with CyMeP
        data_cymep = tcs_cymep_main.run_cymep(start_year, end_year, output_path=tracks_path, gridsize=bin_size)
        model_names = data_cymep.model.values
        data_metrics = data_general.load_yaml_file(config_params.TCS_METRICS_PATH)

        if full_output:
            tc_metrics_names = [data_metrics[metric]['short_name'] for metric in data_metrics if data_metrics[metric]['temporal']==True]
            tc_metrics_units = [data_metrics[metric]['units'] for metric in data_metrics if data_metrics[metric]['temporal']==True]

            # Create linear plots (comparing all datasets)
            linear_ylabel = [f'{name} ({unit})' for name, unit in zip(tc_metrics_names[:5], tc_metrics_units[:5])]
            linear_month_titles = [f'{name} seasonal cycle' for name in tc_metrics_names[:5]] 
            linear_year_titles = [f'{name} interannual cycle' for name in tc_metrics_names[:5]]

            for name, i in enumerate(tc_metrics_names[:5]):
                # Create line plot for monthly cycles
                linear_month_data = data_cymep[f'per_month_{name}'].rename({'month': 'time'})
                linear_month_data_list = [linear_month_data.sel(model=model) for model in model_names]
                months = np.arange(1, 13, dtype=int)

                plt.figure(figsize=(10, 6))
                for j, month_data in enumerate(linear_month_data_list):
                    plt.plot(months, month_data, 'o-', markersize=4, linewidth=1.2, label=model_names[j])
                plt.xticks(months, months)
                plt.xlabel('month', fontsize=12)
                plt.ylabel(linear_ylabel[i], fontsize=12)
                plt.title(linear_month_titles[i], fontsize=16)
                plt.grid(True)
                plt.legend()

                plot.save_or_show_plot(plt.gcf(), output_path, plot_filename=f"tcs_{name.lower()}_monthly_cycle_plot_{data_name}_{start_year}-{end_year}",
                                        plot_name=f"Linear monthly cycle plot for TC {name}")

                # Create line plot for interannual cycles
                linear_year_data = data_cymep[f'per_year_{name}'].rename({'year': 'time'})
                linear_year_data_list = [linear_year_data.sel(model=model) for model in model_names] 
                years = np.arange(start_year, end_year+1, dtype=int)               

                plt.figure(figsize=(10, 6))
                for j, year_data in enumerate(linear_year_data_list):
                    plt.plot(years, year_data, 'o-', markersize=4, linewidth=1.2, label=model_names[j])
                plt.xlabel('year', fontsize=12)
                plt.ylabel(linear_ylabel[i], fontsize=12)
                plt.title(linear_year_titles[i], fontsize=16)
                plt.grid(True)
                plt.legend()

                plot.save_or_show_plot(plt.gcf(), output_path, plot_filename=f"tcs_{name.lower()}_interannual_cycle_plot_{data_name}_{start_year}-{end_year}",
                                        plot_name=f"Linear interannual cycle plot for TC {name}")


            # Create spatial plots (comparing simulations with IBTrACS)
            spatial_titles = [f'TC {name} density per {bin_size}°x{bin_size}° cell ({start_year}-{end_year})' for name in tc_metrics_names if name != 'lmi']
            spatial_bias_titles = [f'TC {name} bias with respect to IBTrACS per {bin_size}°x{bin_size}° cell ({start_year}-{end_year})' for name in tc_metrics_names if name != 'lmi']
            
            spatial_cb_labels = [f'{name} ({unit})' for name, unit in zip(tc_metrics_names, tc_metrics_units) if name != 'lmi']
            spatial_vars = [var.replace('spatial_abs_','') for var in data_cymep.data_vars if var.startswith('spatial_abs')]

            for i, var in enumerate(spatial_vars):
                spatial_abs_data = data_cymep[f'spatial_abs_{var}']
                spatial_abs_plot, _ = plot.two_spatial_plots(spatial_abs_data.sel(model='IBTrACS'), spatial_abs_data.sel(model=data_name), clon=clon,
                                                        title_1='IBTrACS', title_2=data_name, suptitle=spatial_titles[i], cb_label=spatial_cb_labels[i])
                plot.save_or_show_plot(spatial_abs_plot, output_path, plot_filename=f"tcs_{name.lower()}_spatial_abs_plot_{data_name}_{start_year}-{end_year}",
                                        plot_name=f"Spatial plot for TC {name}")

                spatial_bias_data = data_cymep[f'spatial_bias_{var}']
                spatial_bias_plot, _ = plot.spatial_plot(spatial_bias_data.sel(model=data_name), clon=clon, title=spatial_bias_titles[i],
                                                        cb_label=f'bias in {spatial_cb_labels[i]}')
                plot.save_or_show_plot(spatial_bias_plot, output_path, plot_filename=f"tcs_{name.lower()}_spatial_bias_plot_{data_name}_{start_year}-{end_year}",
                                       plot_name=f"Spatial bias plot for TC {name}")


        # Prepare labels for table plots with scalar statistics
        rows = model_names

        cols_clim_bias = [fr'$\overline{{b}}_{{clim,{metric}}}$ ({data_metrics[metric]["units"]})' 
                          for metric in data_metrics if data_metrics[metric]['temporal']==True]
        cols_storm_bias = [fr'$\overline{{b}}_{{storm,{metric}}}$ ({data_metrics[metric]["units"]})' 
                          for metric in data_metrics if data_metrics[metric]['temporal']==True and metric!='count']
        cbar_ticks_bias = ['Negative bias', 'No bias', 'Positive bias']

        cols_temp_corr = [fr'$\rho_{{s,{metric}}}$' for metric in data_metrics 
                          if data_metrics[metric]['temporal']==True]
        cols_spatial_corr = [fr'$r_{{xy,{metric}}}$' for metric in data_metrics 
                             if data_metrics[metric]['spatial']==True]
        cbar_ticks_corr = ['Low correlation', '', 'High correlation']

        # Generate bias tables
        data_clim_mean = [data_cymep[var].values for var in data_cymep.data_vars if var.startswith('clim_mean_')]
        data_clim_bias = np.append(data_clim_mean[0], data_clim_mean[1:] - data_clim_mean[0], axis=1)
        table_clim_bias_plot, _ = plot.table_plot(data_clim_bias, title='Global climatological mean bias', col_labels=cols_clim_bias, 
                                                  row_labels=rows, cbar_tick=cbar_ticks_bias)
        plot.save_or_show_plot(table_clim_bias_plot, output_path, plot_filename=f"tcs_climatological_bias_table_{data_name}_{start_year}-{end_year}",
                               plot_name="Climatological bias table for TCs metrics plot")
        
        data_storm_mean = [data_cymep[var].values for var in data_cymep.data_vars if var.startswith('storm_mean_')]
        data_storm_bias = np.append(data_storm_mean[0], data_storm_mean[1:] - data_storm_mean[0], axis=1)
        table_storm_bias_plot, _ = plot.table_plot(data_storm_bias, title='Global storm mean bias', col_labels=cols_storm_bias, 
                                                  row_labels=rows, cbar_tick=cbar_ticks_bias)
        plot.save_or_show_plot(table_storm_bias_plot, output_path, plot_filename=f"tcs_storm_bias_table_{data_name}_{start_year}-{end_year}",
                               plot_name="Storm bias table for TCs metrics plot")
        
        # Generate correlation tables
        data_temp_corr = [data_cymep[var].values for var in data_cymep.data_vars if var.startswith('temporal_scorr_')]
        table_temp_corr_plot, _ = plot.table_plot(data_temp_corr, title='Global seasonal correlation', col_labels=cols_temp_corr,
                                                  row_labels=rows, cbar_tick=cbar_ticks_corr)
        plot.save_or_show_plot(table_temp_corr_plot, output_path, plot_filename=f"tcs_temp_corr_table_{data_name}_{start_year}-{end_year}",
                               plot_name="Seasonal correlation table for TCs metrics plot")
        
        data_spatial_corr = [data_cymep[var].values for var in data_cymep.data_vars if var.startswith('spatial_pcorr_')]
        table_spatial_corr_plot, _ = plot.table_plot(data_spatial_corr, title='Global spatial correlation', col_labels=cols_spatial_corr,
                                                     row_labels=rows, cbar_tick=cbar_ticks_corr)
        plot.save_or_show_plot(table_spatial_corr_plot, output_path, plot_filename=f"tcs_spatial_corr_table_{data_name}_{start_year}-{end_year}",
                               plot_name="Spatial correlation table for TCs metrics plot")

        return