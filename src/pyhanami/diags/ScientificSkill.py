import warnings
warnings.simplefilter("always")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from collections.abc import Iterable

from pyhanami import config
from pyhanami.utils import iso_metrics, plot
from pyhanami.diags.Simulations import SimulationData
from pyhanami.diags.Observations import ObservationData


class ScientificEvaluation:
    """
    Compute and plot metrics for scientific model skill evaluation.

    This class provides functionality for computing and visualizing metrics to evaluate how well a model 
    reproduces several phenomena. Currently, it includes methods for bimodal ISO indices.
    
    Parameters
    ----------
    datasets : SimulationData or Iterable[SimulationData]
        Ensemble or list of ensembles containing simulation data and metadata.
    obs_path : str
        Path to the observations database.

    Attributes
    ----------
    datasets : list[SimulationData]
        List of ensembles containing simulation data and metadata.
    obs_path : str
        Path to the observations database.
    obs : ObservationData
        Instance containing observational data for comparison.
    variables : dict
        Configuration dictionary mapping variable names to display metadata.

    Methods
    -------
    _apply_Lanczos_bandpass_filter(raw_olr_data, window, low_freq, high_freq):
        Filters Outgoing Longwave Radiation (OLR) data with a Lanczos bandpass filter to isolate the intraseasonal component for ISO evaluation.

    _perform_EEOF_analysis(olr_data, year_init, year_end, season, lags, n_modes):
        Performs Extended Empirical Orthogonal Function (EEOF) analysis to Outgoing Longwave Radiation (OLR) data to identify MJO and BSISO events.
    
    _compute_PCs(olr_data, eeofs, year):
        Computes Principal Components (PCs) of OLR data using previously computed EEOFs for each ISO mode (MJO and BSISO).

    add_datasets(datasets)
        Adds new datasets to the ScientificEvaluation object.

    bimodal_ISO(data_name, year_init_eeof, year_end_eeof, years_pc, output_path, plot_eeofs, clon, lat_range, lags, n_modes, window, low_freq, high_freq):
        Computes bimodal ISO indices following (K. Kikuchi, 2020) and plot results for the selected years.
    """

    def __init__(self, datasets: Iterable[SimulationData] = None, obs_path: str = None):        
        self.obs_path = obs_path
        if datasets is None:
            self.datasets = []
            self.obs = None
        else:
            if isinstance(datasets, SimulationData):
                self.datasets = [datasets]
            elif isinstance(datasets, Iterable) and not isinstance(datasets, (str, bytes)) \
                and all(isinstance(ds, SimulationData) for ds in datasets):
                self.datasets = list(datasets)
            else:
                raise TypeError("Input must be a SimulationData object or an iterable of SimulationData objects.")

            self.obs = ObservationData(self.obs_path, self.datasets[0].data)

        # Load config parameters once
        self.variables = config.VARIABLES

        return


    def _apply_Lanczos_bandpass_filter(self, raw_olr_data, window=141, low_freq=1/90, high_freq=1/25):
        """
        Filter Outgoing Longwave Radiation (OLR) data with a Lanczos bandpass filter to
        isolate the intraseasonal component for ISO evaluation.

        Parameters
        ----------
        raw_olr_data (xr.DataArray): Input unfiltered OLR data.
        window_size (int): Length of the filter kernel.
        low_freq (float): Lower cutoff frequency.
        high_freq (float): Upper cutoff frequency.
 
        Returns
        -------
        filered_olr (xr.DataArray): Lanczos filtered OLR data.
        """

        # Validate input
        if not isinstance(raw_olr_data, xr.DataArray):
            raise TypeError("'raw_olr_data' must be an xarray.DataArray.")
        if not isinstance(window, int):
            raise TypeError("'window' must be an integer.")
        if not isinstance(low_freq, (int, float)) or not isinstance(high_freq, (int, float)):
            raise TypeError("The frequency cutoffs 'low_freq' and 'high_freq' must be numeric.")

        # Filter data
        filtered_olr_data = iso_metrics.apply_lanczos_bandpass(raw_olr_data, window, low_freq, high_freq)
        filtered_olr_data.name = "olr"

        return filtered_olr_data
    

    def _perform_EEOF_analysis(self, olr_data, year_init, year_end, season, lags=[-10, -5, 0], n_modes=2):
        """
        Perform Extended Empirical Orthogonal Function (EEOF) analysis to Outgoing Longwave Radiation (OLR) data
        to identify MJO and BSISO events (boreal winter and boreal summer modes of ISO, respectively).

        Parameters
        ----------
        olr_data (xr.DataArray): Input OLR data.
        year_init (int): Start year for filtering.
        year_end (int): End year for filtering.
        season (str): Season to filter ('boreal winter' or 'boreal summer').
        lags (list[int]): Lag values to consider.
        n_modes (int): Number of EEOFs modes to compute.

        Returns
        -------
        eeof_analysis_data (xr.Dataset): Output of EEOF analysis (first 'n_modes' EEOFs, eigenvalues and explanined variances).
        """

        # Validate input
        if not isinstance(olr_data, xr.DataArray):
            raise TypeError("'olr_data' must be an xarray.DataArray.")
        years = olr_data.time.dt.year.values
        if (year_init not in years) or (year_end not in years):
            raise ValueError("Invalid 'year_init' and/or 'year_end', years not found in the provided dataset.")
        if not isinstance(lags, list) or not all(isinstance(lag, int) for lag in lags):
            raise TypeError("'lags' must be a list of integer lag days.")

        # Generate and vertically stack blocks for EOF analysis
        blocks = iso_metrics.extract_season_blocks(olr_data, year_init, year_end, season)
        lagged_blocks = np.vstack([iso_metrics.generate_lagged_matrix(block, lags)[0] for block in blocks if block.time.size > 0])

        # Perform EEOF analysis
        weights = iso_metrics.broadcasted_area_weights(olr_data, "lag", lags)
        eofs, eigvals, var_frac = iso_metrics.apply_EEOF_analysis(lagged_blocks, weights, n_modes)
        del blocks, lagged_blocks

        # Reshape EEOFs and adjust sign to fit to Kikuchi's paper
        n_lag, n_lat, n_lon = len(lags), len(olr_data.lat), len(olr_data.lon)
        eof_reshaped = eofs.reshape((n_modes), n_lag, n_lat, n_lon)

        if season == "boreal_winter":
            eof_reshaped[0, ...] *= -1
        elif season == "boreal_summer":
            eof_reshaped[0, ...] *= -1
            eof_reshaped[1, ...] *= -1
        else:
            raise ValueError("Invalid 'season' provided.")


        # Compile EEOF analysis output as an xr.Dataset
        eof_data = xr.DataArray(
            eof_reshaped,
            dims=["mode", "lag", "lat", "lon"],
            coords={
                "mode": np.arange(1, n_modes + 1),
                "lag": lags,
                "lat": olr_data.lat,
                "lon": olr_data.lon
            },
            name="eeof"
        )
        eig_data = xr.DataArray(
            eigvals,
            dims=["mode"],
            coords={"mode": np.arange(1, n_modes+1)},
            name="eigvals"
        )
        var_data  = xr.DataArray(
            var_frac,
            dims=["mode"],
            coords={"mode": np.arange(1, n_modes+1)},
            name="var_frac"
        )

        eeof_analysis_data = xr.Dataset({
            "eeof": eof_data,
            "eigval": eig_data,
            "var_frac": var_data

        })

        print(f"Computed EEOFs for {season} between years {year_init} and {year_end}.", flush=True)
        return eeof_analysis_data
    

    def _compute_PCs(self, olr_data, eeofs):
        """
        Compute Principal Components (PCs) of Outgoing Longwave Radiation (OLR) data using previously computed
        Extended Empirical Orthogonal Functions (EEOFs) for each ISO mode (MJO and BSISO).

        Parameters
        ----------
        olr_data (xr.DataArray): Input OLR data.
        eeofs (list[xr.Dataset]): Output of EEOF analysis (EEOFs, eigenvalues and explanined variances) for boreal winter and boreal summer.

        Returns
        -------
        pc_data (xr.Dataset): PCs and their corresponding amplitude (both raw and standarized, i.e. normalized by the eigenvalues) for each ISO mode.
        """

        # Validate input
        if not isinstance(olr_data, xr.DataArray):
            raise TypeError("'olr_data' must be an xarray.DataArray.")
        if not isinstance(eeofs, list) or len(eeofs) == 0 \
            or not all(isinstance(ds, xr.Dataset) for ds in eeofs):
            raise TypeError("'eeofs' must be a non-empty list of xarray.Datasets.")

        # Generate area-weighted lagged matrix from OLR data
        lags = eeofs[0].lag.values.astype(int).tolist()
        lagged_matrix, times = iso_metrics.generate_lagged_matrix(olr_data, lags)
        weights = iso_metrics.broadcasted_area_weights(olr_data, "lag", lags)
        lagged_wmatrix = lagged_matrix * weights[None, :]

        # Compute PCs and the corresponding amplitudes
        pc, pc_std, amp, amp_std = iso_metrics.project_PCs(lagged_wmatrix, eeofs)

        # Assign label for each time step depending on the amplitudes (1: Significant MJO, 2: Significant BSISO; 0: Insignificant)
        labels = np.where(
                (amp[0] > amp[1]) & (amp_std[0] >= 1), 
                1,
            np.where(
                (amp[1] > amp[0]) & (amp_std[1] >= 1),  
                2,
                0   
                )
            )


        # Compile PCs and amplitudes as an xr.Dataset
        modes = eeofs[0].mode.values
        pcw_da      = xr.DataArray(pc[0],      dims=("time","mode"), coords={"time":times, "mode":modes}, name="PC_MJO_raw")
        pcw_std_da   = xr.DataArray(pc_std[0],  dims=("time","mode"), coords={"time":times, "mode":modes}, name="PC_MJO_std")
        ampw_da     = xr.DataArray(amp[0],     dims=("time"),        coords={"time":times},               name="amp_MJO_raw")
        ampw_std_da = xr.DataArray(amp_std[0], dims=("time"),        coords={"time":times},               name="amp_MJO_std")

        pcs_da      = xr.DataArray(pc[1],      dims=("time","mode"), coords={"time":times, "mode":modes}, name="PC_BSISO_raw")
        pcs_std_da   = xr.DataArray(pc_std[1],  dims=("time","mode"), coords={"time":times, "mode":modes}, name="PC_BSISO_std")
        amps_da     = xr.DataArray(amp[1],     dims=("time"),        coords={"time":times},               name="amp_BSISO_raw")
        amps_std_da = xr.DataArray(amp_std[1], dims=("time"),        coords={"time":times},               name="amp_BSISO_std")

        label_data = xr.DataArray(labels, dims=("time"), coords={"time":times}, name="label")
        pcs_data = xr.Dataset({
        "PC_MJO_raw" :    pcw_da,
        "PC_MJO_std" :    pcw_std_da,
        "amp_MJO_raw":    ampw_da,
        "amp_MJO_std":    ampw_std_da,
        "PC_BSISO_raw" :    pcs_da,
        "PC_BSISO_std" :    pcs_std_da,
        "amp_BSISO_raw":    amps_da,
        "amp_BSISO_std":    amps_std_da,
            "label"    :    label_data,
        })

        print(f"Computed PCs.", flush=True)
        return pcs_data
    

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

        # Add observation data if not already present
        if self.obs is None:
            self.obs = ObservationData(self.obs_path, self.datasets[0].data)

        return
    
    
    def bimodal_ISO(self, data_name=None, year_init_eeof=None, year_end_eeof=None, years_pc=None, output_path=None, plot_eeofs=True, clon=0, 
                    lat_range=(-30,30), lags=[-10, -5, 0], n_modes=2, window=141, low_freq=1/90, high_freq=1/25):
        """
        Compute bimodal ISO indices following (K. Kikuchi, 2020) and plot results for the selected years.

        Parameters
        ----------
        data_name (str): Name of simulation ensemble to use.
        year_init_eeof, year_end_eeof (int): Initial and end years to perform the EEOF analysis.
        years_pc (int or list[int]): Years to compute the indices for.
        output_path (str): Path to save plots.
        plot_eeofs (bool): If True, alos spatially plot EEOFs.
        clon (int): Central longitude for the spatial EEOFs maps.
        lat_range (tuple): Geographic latitude bounds.
        lags (list[int]): Lag values to consider.
        n_modes (int): Number of EEOFs modes to compute.
        window_size (int): Length of the filter kernel.
        low_freq (float): Lower cutoff frequency.
        high_freq (float): Upper cutoff frequency.
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
        
        var_name = "olr"
        if var_name not in data_plot.data.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in the simulated dataset '{data_name}'. "
                            f"Available variables: {list(data_plot.data.data_vars.keys())}")
        olr_data_unfiltered = data_plot.data[var_name].sortby("lat").sel(lat=slice(*lat_range)).compute()
        olr_data = self._apply_Lanczos_bandpass_filter(olr_data_unfiltered, window, low_freq, high_freq)

        olr_years = olr_data.time.dt.year
        if year_init_eeof is None:
            year_init_eeof = int(olr_years.min())
        if year_end_eeof is None:
            year_end_eeof = int(olr_years.max())
        if isinstance(years_pc, int):
            years_pc = [years_pc]
        if not any(year in olr_years.values for year in years_pc):
            raise ValueError(f"Some years are missing in the selected dataset {data_name}.")
        
        # Prepare output path if given
        if output_path is not None:
            output_path = Path(output_path)
            if output_path.suffix != '':  
                raise ValueError("Output path must be a directory, not a file path, as multiple files may be created.")
            
            output_path.mkdir(parents=True, exist_ok=True)


        # Conduct EEOF analysis and plot if requested
        eeof_winter = self._perform_EEOF_analysis(olr_data, year_init_eeof, year_end_eeof, 'boreal_winter', lags, n_modes)
        eeof_summer = self._perform_EEOF_analysis(olr_data, year_init_eeof, year_end_eeof, 'boreal_summer', lags, n_modes)

        if plot_eeofs:
            eeofw_plot, _ = plot.eeofs_plot(eeof_winter, clon=clon, title=f'Boreal winter ISO convective pattern {data_name} (DJFMA {year_init_eeof}-{year_end_eeof})', 
                                            cb_label=f'scaled EEOF ({self.variables['olr'][1]})')            
            if output_path is None:
                plt.show()
                print("Boreal winter EEOFs plot created and displayed.", flush=True)
            else:
                eeofw_path = output_path / f"eeof_boreal_winter_{data_name}_{year_init_eeof}-{year_end_eeof}"

                # eeof_winter.to_netcdf(eeofw_path.with_suffix('.nc'))
                eeofw_plot.savefig(eeofw_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
                print(f"Boreal winter EEOFs plot created and saved to '{eeofw_path}.", flush=True)

            eeofs_plot, _ = plot.eeofs_plot(eeof_summer, clon=clon, title=f'Boreal summer ISO convective pattern {data_name} (JJASO {year_init_eeof}-{year_end_eeof})',
                                            cb_label=f'scaled EEOF ({self.variables['olr'][1]})')       
            if output_path is None:
                plt.show()
                print("Boreal summer EEOFs plot created and displayed.\n", flush=True)
            else:
                eeofs_path = output_path / f"eeof_boreal_summer_{data_name}_{year_init_eeof}-{year_end_eeof}"
                
                # eeof_winter.to_netcdf(eeofw_path.with_suffix('.nc'))
                eeofs_plot.savefig(eeofs_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
                print(f"Boreal summer EEOFs plot created and saved to '{eeofs_path}.\n", flush=True)
        

        # Compute and plot PCs
        pcs = self._compute_PCs(olr_data, [eeof_winter, eeof_summer])
        for year in years_pc:
            pcs_year = pcs.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
            pcs_plot, _ = plot.pcs_plot(pcs_year, title=f'Bimodal ISO indices {data_name} ({year})')
            
            if output_path is None:
                plt.show()
                print(f"PCs for year {year} plot created and displayed.", flush=True)
            else:
                pcs_path = output_path / f"pcs_{data_name}_{year}"

                # pcs_year.to_netcdf(pcs_path.with_suffix('.nc'))
                pcs_plot.savefig(pcs_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
                print(f"PCs plot for year {year} created and saved to '{pcs_path}'.", flush=True)

        return
        