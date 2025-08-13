import numpy as np
import xarray as xr


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