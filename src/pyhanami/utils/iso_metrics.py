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


def extract_season_blocks(data, year_init, year_end, season, cutoff_points=90):
    """
    Extract time blocks for the given season for each year, dismissing blocks with less
    than the specified number of points.

    Parameters
    ----------
    data (xarray.DataArray): Input data.
    year_init (int): Start year for filtering.
    year_end (int): End year for filtering.
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
    if (year_init not in years) or (year_end not in years):
        raise ValueError("Invalid 'year_init' and/or 'year_end', years not found in the provided dataset.")
    
    # Compute blocks
    blocks = []
    for year in range(year_init, year_end+1):
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


def apply_EEOF_analysis(data, weights, n_modes=2):
    """
    Perform Extended Empirical Orthogonal Function (EEOF) analysis with area weights.
    
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