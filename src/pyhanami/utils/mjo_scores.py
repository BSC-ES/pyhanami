import xeofs
import numpy as np
import pandas as pd
import xarray as xr

from pyhanami.utils import statistics


def remove_seasonal_cycle(data, start_year_ref, end_year_ref, n_harmonics=3):
    """
    Remove seasonal cycle at the grid point level by subtracting the time-mean 
    and first 'num_harmonics' harmonics of the annual cycle, computed from a 
    reference period.

    Parameters
    ----------
    data : xr.DataArray
        Data with a daily 'time' coordinate.
    start_year_ref, end_year_ref : int
        Initial and end years for computing the reference seasonal cycle.
    n_harmonics : int
        Number of harmonics (besides the mean) to include in the seasonal 
        cycle (default: 3).

    Returns
    -------
    anomalies : xr.DataArray
       Anomaly field with the seasonal cycle removed.
    """

    # Validate input
    if not isinstance(data, xr.DataArray):
        raise TypeError("'data' must be an xarray.DataArray.")

    # Remove leap days (February 29th) to avoid issues with data shape
    data = data.sel(time=~((data['time'].dt.month == 2) & (data['time'].dt.day == 29)))

    # Compute climatology over the reference period (assuming the data is daily and ignoring leap days)
    data_ref = data.sel(time=slice(str(start_year_ref), str(end_year_ref)))
    clim = data_ref.groupby("time.dayofyear").mean("time")
    
    
    # Compute FFT of the climatology along the day-of-year dimension 
    N = clim.sizes["dayofyear"]
    fft_clim = np.fft.fft(clim, n=N, axis=0)
    
    # Zero out all but the mean (index 0) and the first 'num_harmonics' harmonics
    # (keep the 0th coefficient and the first num_harmonics (positive frequencies) 
    # and their symmetric negative counterparts)
    fft_filtered = fft_clim.copy()
    fft_filtered[n_harmonics+1 : N - n_harmonics] = 0
    
    # Inverse FFT to reconstruct the filtered climatology
    clim_filtered = np.fft.ifft(fft_filtered, axis=0).real
    

    # Save filtered climatology to a dataset with the same dayofyear coordinate
    clim_filtered_da = xr.DataArray(clim_filtered, coords=clim.coords, dims=clim.dims)
    

    # Remove the seasonal cycle from the full dataset (for each time, subtract the 
    # filtered climatology for that day-of-year)
    anomalies = data.groupby("time.dayofyear") - clim_filtered_da
    
    return anomalies


def remove_longer_time_scale_components(data, start_year_ref, end_year_ref, lat_range=(-15, 15), rolling_window_size=120, 
                                        n_harmonics=3, normalize_std=False):
    """
    Remove longer-time-scale components (seasonal cycle and interannual variability) at the grid 
    point level, using harmonic filtering to remove the seasonal cycle.

    Parameters
    ----------
    data : xr.Dataset
        Dataset to filter.
    start_year_ref, end_year_ref : int
        Initial and end years for computing the reference seasonal cycle.
    lat_range : tuple
        Geographic latitude bounds (default: (-15, 15)).
    rolling_window_size : int
        Window size for rolling mean to remove low-frequency variability (default: 120 days).
    n_harmonics : int
        Number of harmonics to remove from the seasonal cycle (default: 3).
    normalize_std : bool
        Whether to normalize anomalies by fixed standard deviations when removing the 
        seasonal cycle (default: False).

    Returns
    -------
    filtered_data : xr.DataArray
        Filtered data for each variable.
    anom_data : xr.DataArray
        Anomalies for each variable.
    std_data : xr.DataArray
        Standard deviations for each variable.
    """

    # Validate input
    if not isinstance(data, xr.Dataset):
        raise TypeError("'data' must be an xr.Dataset.")
    

    # Reference std from (M.Wheeler et al., (2004)) for normalization (if needed)
    fixed_std = {
        'rlut': 15.1,   # OLR
        'ua850': 1.81,  # 850 hPa zonal wind
        'ua200': 4.81   # 200 hPa zonal wind
    }

    
    # Filter data
    processed_data = []
    anom_list = []
    std_list = []

    for var_name, data_var in data.data_vars.items():
        # Select time period and remove leap days to avoid issues with data shape
        # data_var = data_var.sel(time=slice(str(start_year_ref), str(end_year_ref)))
        data_var = data_var.sel(time=~((data_var['time'].dt.month == 2) & (data_var['time'].dt.day == 29)))

        # Select latitude region
        data_var = data_var.sel(lat=slice(*lat_range))


        # Compute anomalies (remove mean + first 'n_harmonics' harmonics of seasonal cycle)
        anomalies = remove_seasonal_cycle(data_var, start_year_ref, end_year_ref, n_harmonics)

        # Store anomalies
        anomalies_aux = anomalies.expand_dims('variable')
        anomalies_aux['variable'] = [var_name]
        anom_list.append(anomalies_aux)

        # Remove time mean
        anomalies = anomalies - anomalies.mean(dim='time')

        # Apply rolling mean (not centered, previous 120 days) and subtract
        rolling_mean = anomalies.rolling(time=rolling_window_size, center=False).mean() 
        anomalies = anomalies - rolling_mean
        anomalies = anomalies.isel(time=slice(rolling_window_size, -1))

        # Latitude weighting and average over latitude
        weights = statistics.area_weights(anomalies)
        anomalies_lat_avg = anomalies.weighted(weights).mean(dim='lat')


        # Compute standard deviation and normalize
        std_dev = np.sqrt(anomalies_lat_avg.var(dim='time')).mean(dim='lon')
        # std_dev = np.sqrt(anomalies_lat_avg.var(dim='time').mean(dim='lon'))  # CHANGED ACCORDING TO VENTRICE ET AL 2013

        std_aux = std_dev.expand_dims('variable')
        std_aux['variable'] = [var_name]
        std_list.append(std_aux)

        anomalies_lat_avg /= std_dev

        # Normalize by fixed std dev (M.Wheeler et al., (2004))
        if normalize_std:
            std = fixed_std.get(var_name)
            if std is None:
                raise ValueError(f"Standard deviation for variable '{var_name}' is not defined.")
            anomalies_lat_avg /= std

        # Add 'variable' dimension for concatenation
        anomalies_lat_avg = anomalies_lat_avg.expand_dims('variable')
        anomalies_lat_avg['variable'] = [var_name]

        processed_data.append(anomalies_lat_avg)


    # Save data to xr.Datasets
    filtered_data = xr.concat(processed_data, dim='variable').transpose('time', 'lon', 'variable')
    filtered_data.attrs = {'description': 'Preprocessed data with seasonal cycle removed, latitude averaged and normalized by standard deviation.'}

    anom_data = xr.concat(anom_list, dim='variable').transpose('time', 'lon', 'lat', 'variable')
    anom_data.attrs = {'description': 'Anomalies with seasonal cycle removed, before mean subtraction and normalization.'}

    std_data = xr.concat(std_list, dim='variable')
    std_data.attrs = {'description': 'Standard deviations of anomalies with seasonal cycle removed, before mean subtraction and normalization.'}


    return filtered_data, anom_data, std_data


def fit_CEOF_model_xeofs(data, n_modes=2):
    """
    Fit model for Combined Empirical Orthogonal Function (COEF) analysis using xeofs.

    Parameters
    ----------
    data : xr.DataArray
        Climate data.
    n_modes : int
        Number of CEOFs modes to compute (default: 2)

    Returns
    -------
    eof_model : xeofs.single.EOF
        Fitted CEOF model that can be used to retrieve EOFs, eigenvalues, 
        explained variance, PCs and reconstructed data.
    """

    # Create CEOF model
    eof_model = xeofs.single.EOF(
        n_modes=n_modes,
        center=False,        # We already removed the mean in preprocessing
        standardize=False,  # We already normalized by std dev in preprocessing
    )

    # Fit model
    eof_model.fit(data, dim=['time'])

    return eof_model


def retrieve_EOFs_xeofs(eof_model, vars_order=['ua850', 'ua200', 'rlut']):
    """
    Retrieve elements of Combined Empirical Orthogonal Function (COEF) analysis from 
    the provided fitted model and correct them to match the output of eofs.xarray.Eof.

    Parameters
    ----------
    eof_model : xeofs.single.EOF
        Fitted CEOF model.
    vars_order : list[str]
        Name of climate variables in the correct order to be saved in the
        reconstructed dataset (default: ['ua850', 'ua200', 'rlut']).

    Returns
    -------
    eof_xeofs : xr.DataArray
        Resulting EOFs
    eigenvalues_xeofs : xr.DataArray
        Resulting eigenvalues.
    variance_xeofs : xr.DataArray
        Resulting explained variance (normalized by eigenvalues). 
    pc_xeofs : xr.DataArray
        Resulting PCs.
    reconstructed_xeofs : xr.DataArray
        Resulting reconstructed data.
    """

    # Retrieve model's outcome
    eof_xeofs = eof_model.components()
    variance_xeofs = eof_model.explained_variance_ratio() * 100
    eigenvalues_xeofs = eof_model.explained_variance()
    pc_xeofs = eof_model.scores()
    reconstructed_xeofs = eof_model.inverse_transform(pc_xeofs)


    # Correct EOFs to match output of eofs.xarray.Eof
    eof_xeofs = eof_xeofs.assign_coords(mode=[0, 1]).sel(variable=vars_order)
    eof_xeofs.loc[dict(mode=0)] *= -1
    eof_xeofs.attrs.pop('solver_kwargs', None)

    # Correct explained variance to match output of eofs.xarray.Eof
    variance_xeofs = variance_xeofs.assign_coords(mode=[0, 1])
    variance_xeofs.attrs.pop('solver_kwargs', None)

    # Normalize and correct PCs to match output of eofs.xarray.Eof
    pc_xeofs = (pc_xeofs/np.sqrt(eigenvalues_xeofs)).assign_coords(mode=[0, 1]).transpose('time', 'mode')
    pc_xeofs.loc[dict(mode=0)] *= -1
    pc_xeofs.attrs.pop('solver_kwargs', None)

    # Correct eigenvalues to match output of eofs.xarray.Eof
    eigenvalues_xeofs = eigenvalues_xeofs.assign_coords(mode=[0, 1])
    eigenvalues_xeofs.attrs.pop('solver_kwargs', None)

    # Correct reconstructed data to match output of eofs.xarray.Eof
    reconstructed_xeofs = reconstructed_xeofs.sel(variable=vars_order)
    reconstructed_xeofs.attrs.pop('solver_kwargs', None)

    return eof_xeofs, eigenvalues_xeofs, variance_xeofs, pc_xeofs, reconstructed_xeofs


def perform_CEOF_analysis(data=None, eof_model=None, n_modes=2, vars_order=['ua850', 'ua200', 'rlut']):
    """
    Perform Combined Empirical Orthogonal Function (COEF) analysis to 
    identify MJO events.

    Parameters
    ----------
    data : xr.DataArray
        Climate data. It is not needed if a fitted `eof_model` is provided.
    eof_model : xeofs.single.EOF
        Fitted CEOF model that can be used to retrieve EOFs, eigenvalues, 
        explained variance, PCs and reconstructed data. If not provided, 
        the model will be fitted using the provided data.
    n_modes : int
        Number of CEOFs modes to compute (default: 2)
    vars_order : list[str]
        Name of climate variables in the correct order to be saved in the
        reconstructed dataset (default: ['ua850', 'ua200', 'rlut']).

    Returns
    -------
    eof_analysis_data : xr.Dataset
        Output of CEOF analysis ('eof', 'eigval', 'var_frac' and 'pc' 
        for the first 'n_modes')
    """

    # Validate input
    if data is not None and not isinstance(data, xr.DataArray):
        raise TypeError("'data' must be an xarray.DataArray.")
    if not isinstance(n_modes, int):
        raise TypeError("'n_modes' must be an integer.")
    
    # Fit CEOF model if not provided
    if data is None:
        if eof_model is None:
            raise ValueError("When 'data' is not provided, 'eof_model' must be provided.")
    else:
        eof_model = fit_CEOF_model_xeofs(data, n_modes)

    # Retrieve output of CEOF analysis
    eofs, eigvals, var_frac, pcs, _ = retrieve_EOFs_xeofs(eof_model, vars_order)

    # Compile CEOF analysis output as a xr.Dataset
    ceof_analysis_data = xr.Dataset({
        "eof": eofs,
        "eigval": eigvals,
        "var_frac": var_frac,
        "pc": pcs
    })

    return ceof_analysis_data


def correct_EOFs(eof_new, eof_ref, n_modes=2):
    """
    Correct sign and order of the first 'n_modes' EOFs, and related measures,
    based on their correlation with a reference EOF (typically following 
    (M.Wheeler et al., (2004))).
    NOTE: only working for 'n_modes=2' for now.

    Paramters
    ---------
    eof_new : xr.Dataset
        Output of CEOF analysis to be corrected (EOFs, eigenvalues, explained 
        variance and PCs).
    eof_ref : xr.Dataset
        Reference output of CEOF analysis (EOFs, eigenvalues, explained variance 
        and PCs).
    n_modes : int
        Number of modes to correct (default: 2).

    Returns
    -------
    eof_corrected : xr.Dataset
        Corrected output of CEOF analysis (EOFs, eigenvalues, explained variance 
        and PCs).
    """

    # Validate input
    if not isinstance(eof_ref, xr.Dataset) or not isinstance(eof_new, xr.Dataset):
        raise TypeError("'eof_ref' and 'eof_new' must be xarray.Datasets.")
    if not isinstance(n_modes, int):
        raise TypeError("'n_modes' must be an integer.")
    

    # Select reference variable
    ref_var = 'ua850'
    eof_ref_var = eof_ref['eof'].sel(variable=ref_var)

    eof_corrected = eof_new.copy(deep=True)
    eof_var = eof_corrected['eof'].sel(variable=ref_var)


    # Compute correlation matrix between EOFs of reference and new dataset
    corr_matrix = np.array([
        [xr.corr(eof_ref_var.sel(mode=i), eof_var.sel(mode=j)) for j in range(n_modes)] for i in range(n_modes)
    ])

    # Check whether modes should be swapped
    keep = abs(corr_matrix[0,0]) + abs(corr_matrix[1,1])  # Keep original order
    swap = abs(corr_matrix[0,1]) + abs(corr_matrix[1,0])  # Swap modes

    # Swap modes to achieve the best correlation match
    if swap > keep:
        eof_corrected['eof'].loc[dict(mode=0)] = eof_new['eof'].sel(mode=1)
        eof_corrected['eigval'].loc[dict(mode=0)] = eof_new['eigval'].sel(mode=1)
        eof_corrected['var_frac'].loc[dict(mode=0)] = eof_new['var_frac'].sel(mode=1)
        eof_corrected['pc'].loc[dict(mode=0)] = eof_new['pc'].sel(mode=1)

        eof_corrected['eof'].loc[dict(mode=1)] = eof_new['eof'].sel(mode=0)
        eof_corrected['eigval'].loc[dict(mode=1)] = eof_new['eigval'].sel(mode=0)
        eof_corrected['var_frac'].loc[dict(mode=1)] = eof_new['var_frac'].sel(mode=0)
        eof_corrected['pc'].loc[dict(mode=1)] = eof_new['pc'].sel(mode=0)

        corr_matrix = corr_matrix[:, ::-1]


    # Check whether the sign of the EOFs and PCs should be inverted
    if corr_matrix[0,0] < 0:
        eof_corrected['eof'].loc[dict(mode=0)] *= -1
        eof_corrected['pc'].loc[dict(mode=0)] *= -1
    if corr_matrix[1,1] < 0:
        eof_corrected['eof'].loc[dict(mode=1)] *= -1
        eof_corrected['pc'].loc[dict(mode=1)] *= -1

    return eof_corrected


def deteremine_phases(pcs):
    """
    Determine MJO phase for each day based on the angle in the phase
    space defined by the first two PCs.

    Parameters
    ----------
    pcs : xr.DataArray
        Principal Components (PCs) from CEOF analysis, with dimensions 
        ['time', 'mode'].

    Returns
    -------
    phase : xr.DataArray
        MJO phase for each day, with dimensions ['time'].
    """
    
    # Validate input
    if not isinstance(pcs, xr.DataArray):
        raise TypeError("'pcs' must be an xarray.DataArray.")
    
    phases = []
    for pc1, pc2 in zip(pcs.isel(mode=0).values, pcs.isel(mode=1).values):
        # Determine angle in degrees
        angle_rad = np.arctan2(pc1, pc2)
        angle_deg = np.degrees(angle_rad) % 360

        # Determine phase based on angle (8 phases, each covering 45 degrees)
        if 0 <= angle_deg < 360:
            phase_values = ((((angle_deg + 180) // 45) % 8) + 1).astype(int)
        else:
            phase_values = np.nan
        phases.append(phase_values)
        
    phases = xr.DataArray(phases, coords={'time': pcs.time}, dims=['time'])
    return phases


def compute_phase_counts(pcs, threshold=None):
    """
    Compute the number of MJO days and active MJO days per phase based 
    on the amplitude of the first two Principal Components (PCs).

    Parameters
    ----------
    pcs : xr.DataArray
        Principal Components (PCs) from CEOF analysis, with dimensions 
        ['time', 'mode'].
    threshold : float
        Threshold for the amplitude of the first two PCs to consider the  
        MJO active at a given day. If None, the mean amplitude is used 
        as a threshold.

    Returns
    -------
    phase_counts : xr.Dataset
        Mean amplitude and days per phase (total and only for active MJO days).
        It contains the following variables: 'mean_amplitude', 'mean_active_amplitude',
        'total_counts' and 'active_counts' per phase.
    """

    # Validate input
    if not isinstance(pcs, xr.DataArray):
        raise TypeError("'pcs' must be an xarray.DataArray.")
    
    # Compute and filter amplitude
    amplitude = np.sqrt(pcs.isel(mode=0)**2 + pcs.isel(mode=1)**2)
    if threshold is None:
        threshold = amplitude.mean(dim='time')
        active_days = amplitude > threshold
    else:
        active_days = amplitude > threshold


    # Determine phase for each day
    phases = deteremine_phases(pcs)

    # Compute mean amplitude per phase
    mean_amplitude_per_phase = []
    mean_active_amplitude_per_phase = []
    for phase in range(1, 9):
        phase_mask = phases == phase
        if phase_mask.sum() > 0:
            mean_amp = amplitude.where(phase_mask).mean(dim='time').values.item()
            mean_active_amp = amplitude.where(phase_mask & active_days).mean(dim='time').values.item()
        else:
            mean_amp = 0.0
            mean_active_amp = 0.0
        mean_amplitude_per_phase.append(mean_amp)
        mean_active_amplitude_per_phase.append(mean_active_amp)

    # Compute total counts and active counts per phase
    total_counts = phases.to_pandas().value_counts().sort_index()
    active_counts = phases.to_pandas()[active_days.values].value_counts().sort_index()


    # Compile results into a xr.Dataset
    phase_counts = xr.Dataset(
        {
            'mean_amplitude': ('phase', mean_amplitude_per_phase),
            'mean_active_amplitude': ('phase', mean_active_amplitude_per_phase),
            'total_counts': ('phase', total_counts.reindex(range(1, 9), fill_value=0).values),
            'active_counts': ('phase', active_counts.reindex(range(1, 9), fill_value=0).values)
        },
        coords={'phase': np.arange(1, 9)}
    )

    return phase_counts