import warnings
warnings.simplefilter("always")

import matplotlib.pyplot as plt

from pathlib import Path
from collections.abc import Iterable
from matplotlib.colors import LinearSegmentedColormap

from pyhanami.config import config_params
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData
from pyhanami.utils import data_general, iso_metrics, plot

import time


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