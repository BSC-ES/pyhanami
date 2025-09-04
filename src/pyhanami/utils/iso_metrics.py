import numpy as np
import xarray as xr

from eofs.standard import Eof


def math_sinc(x):
    """ Compute mathematical sinc function, defined as sinc(x)/x."""
    return np.sinc(x/np.pi)


def lanczos_kernel(window_size, low_freq, high_freq):
    """
    Create bandpass filter kernel using Lanczos window.

    Parameters
    ----------
    window_size (int): Length of the filter kernel.
    low_freq (float): Lower cutoff frequency.
    high_freq (float): Upper cutoff frequency.

    Returns
    -------
    h (np.ndarray): Symmetric bandpass filter kernel.
    """

    # Generate symmetric time vector
    assert window_size%2 == 1
    n = window_size//2
    t = np.arange(-n, n+1, dtype=float)

    # Compute low and high pass filter kernels
    h_low = 2*low_freq*math_sinc(2*np.pi*low_freq*t)
    h_high = 2*high_freq*math_sinc(2*np.pi*high_freq*t)

    # Compute Lanczos window to smooth the kernel
    lanczos = math_sinc(t*np.pi/n)

    # Combine to compute Lanczos kernel
    h = (h_high-h_low)*lanczos
    h[t == 0] = 2*(high_freq-low_freq)

    return h


def lanczos_bandpass(data, window=141, low_freq=1/90, high_freq=1/25):
    """ 
    Apply Lanczos filtering to a 1D signal.

    Parameters
    ----------
    data (np.ndarray): 1D input signal to be filtered.
    window_size (int): Length of the filter kernel.
    low_freq (float): Lower cutoff frequency.
    high_freq (float): Upper cutoff frequency.

    Returns
    -------
    filtered_data (np.ndarray): Filtered signal.
    """

    kernel = lanczos_kernel(window, low_freq, high_freq)
    filtered_data = np.convolve(data, kernel, mode='same')

    return filtered_data


def apply_lanczos_bandpass(data, window=141, low_freq=1/90, high_freq=1/25):
    """
    Apply Lanczos filtering to data within an xarray object.

    Parameters
    ----------
    data (xarray.DataArray): Input data to be filtered.
    window_size (int): Length of the filter kernel.
    low_freq (float): Lower cutoff frequency.
    high_freq (float): Upper cutoff frequency.

    Returns
    -------
    filtered_data (xarray.DataArray): Filtered data.
    """
    
    def _filter_func(x):
        return lanczos_bandpass(x, window, low_freq, high_freq)
    
    filtered_data = xr.apply_ufunc(
        _filter_func,
        data,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
    )
    return filtered_data


def apply_lanczos_bandpass_filter(raw_olr_data, window=141, low_freq=1/90, high_freq=1/25):
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
    filtered_olr_data = apply_lanczos_bandpass(raw_olr_data, window, low_freq, high_freq)
    filtered_olr_data.name = "olr"

    return filtered_olr_data


def extract_season_blocks(data, start_year, end_year, season, cutoff_points=90):
    """
    Extract time blocks for the given season for each year, dismissing blocks with less
    than the specified number of points.

    Parameters
    ----------
    data (xarray.DataArray): Input data.
    start_year (int): Start year for filtering.
    end_year (int): End year for filtering.
    season (str): Season to filter.
    cutoff_points (int): Minimum number of points necessary to keep a block.

    Returns
    -------
    blocks (list[xarray.DataArray]): Extracted season blocks.
    """

    # Validate input
    if not isinstance(data, xr.DataArray):
        raise TypeError("'data' must be an xarray.DataArray.")
    years = data.time.dt.year
    if (start_year not in years) or (end_year not in years):
        raise ValueError("Invalid 'start_year' and/or 'end_year', years not found in the provided dataset.")
    
    # Compute blocks
    blocks = []
    for year in range(start_year, end_year+1):
        # Generate season labels for the corresponding years
        if season == "boreal_winter":
            start = np.datetime64(f"{year}-12-01")
            end = np.datetime64(f"{year+1}-04-30")
        elif season == "boreal_summer":
            start = np.datetime64(f"{year}-06-01")
            end = np.datetime64(f"{year}-10-31")
        else:
            raise ValueError("Invalid 'season' provided.")
        
        # Only keep blocks with a minimum number of points
        block = data.sel(time=slice(start, end))
        if block.time.size >= cutoff_points:
            blocks.append(block)

    return blocks


def generate_lagged_matrix(data, lags):
    """
    Create lagged versions of the data and stack (lag, lat, lon) them into
    a single feature dimension to generate a lagged matrix.

    Parameters
    ----------
    data (xarray.Dataset): Input data.
    lags (list[int]): Lag values to add.

    Returns
    -------
    lagged_matrix (np.ndarray): Stacked lagged version of the data.
    times (np.ndarray): Times included in the lagged matrix.
    """

    # Update attributes to avoid issues with time ranges
    if "actual_range" in data.coords["time"].attrs:
        del data.coords["time"].attrs["actual_range"]
    data = data.copy(deep=True)

    # Create lagged versions of the season data
    lagged_list = []
    for lag in lags:
        shifted = data.shift(time=-lag).assign_coords(time=data["time"])
        lagged_list.append(shifted)
    lagged = xr.concat(lagged_list, dim="lag").assign_coords(lag=lags)

    # Stack all lags
    lagged = lagged.transpose("time", "lag", "lat", "lon")
    valid = ~np.any(np.isnan(lagged), axis=(1, 2, 3))

    lagged_matrix = lagged.isel(time=valid).stack(feature=("lag", "lat", "lon")).values
    times = lagged.time.values[valid]

    return lagged_matrix, times


def broadcasted_area_weights(data, dim_name=None, dim_values=None):
    """
    Compute latitude-based area weights, broadcast them to match the shape of the given data 
    and add new dimension if given.

    Parameters
    ----------
    data (xr.DataArray): Input data with reference shape.
    dim_name (str): Name of the additional dimension.
    dim_values (list[float]): Values of the additional dimension.

    Returns
    -------
    weights_vector (np.dnarray): Flattened area weights matching the shape of the provided data.
    """

    # Compute standard area weights
    lat_rad = np.deg2rad(data.lat)
    weights_1D = np.cos(lat_rad)

    # Broadcast to match data shape
    weights_2D = xr.DataArray(weights_1D, coords={"lat": data.lat}, dims=["lat"])
    weights_broadcasted = weights_2D.broadcast_like(data.isel(time=0))

    # Add extra dimension if given
    if dim_name is None:
        weights_vector = weights_broadcasted.stack(feature=("lat", "lon")).values
    else:
        weights_broadcasted = weights_broadcasted.expand_dims({dim_name: dim_values}).assign_coords({dim_name: dim_values})
        weights_vector = weights_broadcasted.stack(feature=(dim_name, "lat", "lon")).values

    return weights_vector


def compute_EEOFs(data, weights, n_modes=2):
    """
    Compute Extended Empirical Orthogonal Functions (EEOFs) with area weights.
    
    Parameters
    ----------
    data (np.ndarray): Input data.
    weights (np.ndarray): Area weights with same shape as data.
    n_modes (int): Number of EEOFs to compute.

    Returns
    -------
    eofs (np.ndarray): Resulting EOFs.
    eigvals (np.ndarray): Resulting eigenvalues.
    var_frac (np.ndarray): Resulting explained variance (normalized eigenvalues).
    """

    solver = Eof(data, weights=weights)
    eofs = solver.eofs(neofs=n_modes)
    eigvals = solver.eigenvalues(neigs=n_modes)
    var_frac = solver.varianceFraction(neigs=n_modes)

    return eofs, eigvals, var_frac


def perform_EEOF_analysis(olr_data, start_year, end_year, season, lags=[-10, -5, 0], n_modes=2):
    """
    Perform Extended Empirical Orthogonal Function (EEOF) analysis to Outgoing Longwave Radiation (OLR) data
    to identify MJO and BSISO events (boreal winter and boreal summer modes of ISO, respectively).

    Parameters
    ----------
    olr_data (xr.DataArray): Input OLR data.
    start_year (int): Start year for filtering.
    end_year (int): End year for filtering.
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
    if (start_year not in years) or (end_year not in years):
        raise ValueError("Invalid 'start_year' and/or 'end_year', years not found in the provided dataset.")
    if not isinstance(lags, list) or not all(isinstance(lag, int) for lag in lags):
        raise TypeError("'lags' must be a list of integer lag days.")

    # Generate and vertically stack blocks for EOF analysis
    blocks = extract_season_blocks(olr_data, start_year, end_year, season)
    lagged_blocks = np.vstack([generate_lagged_matrix(block, lags)[0] for block in blocks if block.time.size > 0])

    # Perform EEOF analysis
    weights = broadcasted_area_weights(olr_data, "lag", lags)
    eofs, eigvals, var_frac = compute_EEOFs(lagged_blocks, weights, n_modes)
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

    print(f"Computed EEOFs for {season} between years {start_year} and {end_year}.", flush=True)
    return eeof_analysis_data


def project_PCs(data, eeofs):
    """
    Compute Principal Components (PCs) of the input data by projecting it onto the given
    Empirical Orthogonal Functions (EOFs).

    Parameters
    ----------
    data (xr.Dataset): Input data.
    eeofs (list[xr.Dataset]): Output of EEOF analyses (EEOFs and eigenvalues).

    Returns
    -------
    pc (list): raw PCs for each set of EOFs.
    pc_std (list): standarized PCs (i.e. normalized by the corresponding eigenvalues) for each set of EOFs.
    amp (list): raw amplitudes for each set of EOFs.
    amp_std (list): standarized amplitudes (i.e. normalized by the corresponding eigenvalues) for each set of EOFs.
    """

    pc = []
    pc_std = []
    amp = []
    amp_std = []

    for eeof_data in eeofs:
        eof = eeof_data["eeof"]
        eof_flat = eof.stack(feature=("lag", "lat", "lon")).values
        eig = eeof_data["eigval"].values

        pc_aux = data.dot(eof_flat.T)
        pc_std_aux = pc_aux / np.sqrt(eig)[None, :]
        pc.append(pc_aux)
        pc_std.append(pc_std_aux)

        amp_aux = np.linalg.norm(pc_aux, axis=1)
        amp_std_aux =  np.linalg.norm(pc_std_aux, axis=1)
        amp.append(amp_aux)
        amp_std.append(amp_std_aux)

    return pc, pc_std, amp, amp_std


def compute_PCs(olr_data, eeofs):
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
    lagged_matrix, times = generate_lagged_matrix(olr_data, lags)
    weights = broadcasted_area_weights(olr_data, "lag", lags)
    lagged_wmatrix = lagged_matrix * weights[None, :]

    # Compute PCs and the corresponding amplitudes
    pc, pc_std, amp, amp_std = project_PCs(lagged_wmatrix, eeofs)

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


def compute_freq_ISO(events):
    """
    Compute the mean monthly frequency of ocurrence of ISO events (distinguishing between MJO and BSISO).

    Parameters
    ----------
    events (xr.DataArray): Input labelled events data.

    Returns
    -------
    freq_ISO (xr.Dataset): Mean monthly frequency of ocurrence.
    """

    # Validate input
    if not isinstance(events, xr.DataArray):
        raise TypeError("'events' must be an xarray.DataArray.")
    
    # Compute monthly frequency
    n_mjo = (events == 1)
    n_bsiso = (events == 2)

    freq_mjo = n_mjo.groupby("time.month").mean().compute()
    freq_bsiso = n_bsiso.groupby("time.month").mean().compute()

    freq_ISO = xr.Dataset({
        "freq_MJO": freq_mjo,
        "freq_BSISO": freq_bsiso
    })

    print(f"Computed mean monthly frequency of ISO events.", flush=True)
    return freq_ISO


def compute_TSS(freq_ISO, freq_obs):
    """
    Compute the Taylor Skill Score (TSS) comparing simulated and observed 
    mean monthly frequency of ocurrence of ISO events.
    
    Parameters
    ----------
    freq_ISO (xr.Dataset): Simulated mean monthly frequency of ocurrence.
    freq_obs (xr.Dataset): Observed mean monthly frequency of ocurrence.

    Returns
    -------
    corr (float): Temporal correlation coefficient of the seasonality.
    sigma (float): Ratio of the standard deviations (model/obs) of the seasonality.
    tss (float): Taylor Skill Score.
    """

    # Validate input
    if not isinstance(freq_ISO, xr.Dataset) or not isinstance(freq_obs, xr.Dataset):
        raise TypeError("Simulated and observed frequencies must be xarray.Datasets.")
    
    # Compute frequencies
    freq_diff_sim = freq_ISO['freq_BSISO'] - freq_ISO['freq_MJO']
    freq_diff_obs = freq_obs['freq_BSISO'] - freq_obs['freq_MJO']


    # Compute statistics (Note: corr_0 is the maximum correlation attainable by the model, here assumed to be 1)
    corr_0 = 1
    corr = xr.corr(freq_diff_sim, freq_diff_obs, dim='month')
    sigma = freq_diff_sim.std(dim='month') / freq_diff_obs.std(dim='month')

    tss = (4 * (1+corr)**4) / ((sigma + (1/sigma))**2 * (1+corr_0)**2)

    print(f"Computed Taylor Skill Score (TSS) between simulations and observations:\n" + 
            f"\tTemporal correlation (R): {corr:.2f}, Ratio standard deviations ($\\sigma$): {sigma:.2f}, TSS: {tss:.2f}\n", flush=True)
    return corr, sigma, tss