"""
This module contains functions adapted from the wavenumber_frequency GitHub repository
Original source: https://github.com/brianpm/wavenumber_frequency/tree/master
Original author: Brian Medeiros
License: MIT License
Copyright (c) 2024 Brian Medeiros
"""

import numpy as np
import xarray as xr

from scipy.signal import detrend
from scipy.ndimage import convolve1d


# Functions adapted from wavenumber_frequency_functions.py

def rmv_annual_cycle(data, spd, f_crit):
    """
    Remove frequencies less than 'f_crit' from data.

    Parameters
    ----------
    data : xr.DataArray
        Data.
    spd : int
        Sampling frequency, in samples/day.
    f_crit : float
        Frequency threshold, remove frequencies < f_crit.

    Returns
    -------
    z : xr.DataArray
        DataArray with frequencies < f_crit removed.


    Note: fft/ifft preserves the mean because z = fft(x), z[0] is the mean.
          To keep the mean here, we need to keep the 0 frequency.

    Note: This function reproduces the results from the NCL version.

    Note: Two methods are available, one using fft/ifft and the other rfft/irfft.
          They both produce output that is indistinguishable from NCL's result.
    """

    dimz = data.sizes
    ntim = dimz["time"]
    time_ax = list(data.dims).index("time")

    # Method 1: Uses the complex FFT, returns the negative frequencies too, but they
    # should be redundant b/c they are conjugate of positive ones.
    cf = np.fft.fft(data.values, axis=time_ax)
    freq = np.fft.fftfreq(ntim, spd)
    cf[(freq != 0) & (np.abs(freq) < f_crit), ...] = 0.0  # keeps the mean
    z = np.fft.ifft(cf, n=ntim, axis=0)

    # Method 2: Uses the real FFT. In this case,
    # cf = np.fft.rfft(data.values, axis=time_ax)
    # freq = np.linspace(1, (ntim*spd)//2, (ntim*spd)//2) / ntim
    # fcrit_ndx = np.argwhere(freq < f_crit).max()
    # if fcrit_ndx > 1:
    #     cf[1:fcrit_ndx+1, ...] = 0.0
    # z = np.fft.irfft(cf, n=ntim, axis=0)

    z = xr.DataArray(z.real, dims=data.dims, coords=data.coords)

    return z


def decompose_to_sym_asym(arr):
    """
    Mimic NCL function to decompose into symmetric and asymmetric parts
    (it produces indistinguishable results from NCL version).

    Parameters
    ----------
    arr : xr.DataArray
        Data.

    Returns
    -------
    out : xr.DataArray
        DataArray with symmetric component in SH and asymmetric component in NH.
    """

    lat_dim = arr.dims.index("lat")
    # flag to follow NCL convention and put symmetric component in SH
    # & asymmetric in NH
    # method: use flip to reverse latitude, put in DataArray for coords, use loc/isel
    # to assign to negative/positive latitudes (exact equator is left alone)
    data_sym = 0.5 * (arr.values + np.flip(arr.values, axis=lat_dim))
    data_asy = 0.5 * (arr.values - np.flip(arr.values, axis=lat_dim))
    data_sym = xr.DataArray(data_sym, dims=arr.dims, coords=arr.coords)
    data_asy = xr.DataArray(data_asy, dims=arr.dims, coords=arr.coords)

    out = arr.copy()  # might not be best to copy, but is safe
    out.loc[{"lat": arr["lat"][arr["lat"] < 0]}] = data_sym.isel(lat=data_sym.lat < 0)
    out.loc[{"lat": arr["lat"][arr["lat"] > 0]}] = data_asy.isel(lat=data_asy.lat > 0)

    return out


def apply_lat_aggregation(d, lat_aggreg):
    if lat_aggreg == "sum":
        r = d.sum(dim="lat").squeeze()
    elif lat_aggreg == "mean":
        r = d.mean(dim="lat").squeeze()
    else:
        raise ValueError(f"lat_aggreg set to {lat_aggreg}, must be `mean` or `sum`")

    return r


def resolve_waves_Hayashi(var_fft, n_day_win, spd):
    """
    This is a direct translation from the NCL routine to python/xarray.

    Parameters
    ----------
    var_fft : xr.DataArray
        Data with  dimensions of wavenumber and frequency.
    n_day_win : int
        Length of the segments, in days.
    spd : int
        Sampling frequency, in `timesteps`/day.

    Returns
    -------
    pee : xr.DataArray
        Reordered data with correct westward and eastward propagation.
    """

    # -------------------------------------------------------------
    # Special reordering to resolve the Progressive and Retrogressive waves
    # Reference: Hayashi, Y.
    #    A Generalized Method of Resolving Disturbances into
    #    Progressive and Retrogressive Waves by Space and
    #    Fourier and Time Cross-Spectral Analysis
    #    J. Meteor. Soc. Japan, 1971, 49: 125-128.
    # -------------------------------------------------------------

    # in NCL var_fft is dimensioned (2,mlon,nSampWin), but the first dim doesn't matter b/c python supports complex numbers.
    #
    # Create array PEE(NL+1,NT+1) which contains the (real) power spectrum.
    # all the following assume indexing starting with 0
    # In this array (PEE), the negative wavenumbers will be from pn=0 to NL/2-1 (left).
    # The positive wavenumbers will be for pn=NL/2+1 to NL (right).
    # Negative frequencies will be from pt=0 to NT/2-1 (left).
    # Positive frequencies will be from pt=NT/2+1 to NT  (right).
    # Information about zonal mean will be for pn=NL/2 (middle).
    # Information about time mean will be for pt=NT/2 (middle).
    # Information about the Nyquist Frequency is at pt=0 and pt=NT
    #

    # In PEE, define the
    # WESTWARD waves to be either
    #          positive frequency and negative wavenumber
    #          OR
    #          negative freq and positive wavenumber.
    # EASTWARD waves are either positive freq and positive wavenumber
    #          OR negative freq and negative wavenumber.

    # Note that frequencies are returned from fftpack are ordered like so
    #    input_time_pos [ 0    1   2    3     4      5    6   7  ]
    #    ouput_fft_coef [mean 1/7 2/7  3/7 nyquist -3/7 -2/7 -1/7]
    #                    mean,pos freq to nyq,neg freq hi to lo
    #
    # Rearrange the coef array to give you power array of freq and wave number east/west
    # Note east/west wave number *NOT* eq to fft wavenumber see Hayashi '71
    # Hence, NCL's 'cfftf_frq_reorder' can *not* be used.
    # BPM: This goes for np.fft.fftshift
    #
    # For ffts that return the coefficients as described above, here is the algorithm
    # coeff array var_fft(2,n,t)   dimensioned (2,0:numlon-1,0:numtim-1)
    # new space/time pee(2,pn,pt) dimensioned (2,0:numlon  ,0:numtim  )
    #
    # NOTE: one larger in both freq/space dims
    # the initial index of 2 is for the real (indx 0) and imag (indx 1) parts of the array
    #
    #
    #    if  |  0 <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  0 <= pt < numtim/2-1             | numtim/2 <= t <= numtim-1
    #
    #    if  |  0         <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  numtime/2 <= pt <= numtim                | 0        <= t <= numtim/2
    #
    #    if  |  numlon/2  <= pn <= numlon    then    | 0  <= n <= numlon/2
    #        |  0         <= pt <= numtim/2          | numtim/2 <= t <= 0
    #
    #    if  |  numlon/2   <= pn <= numlon    then    | 0        <= n <= numlon/2
    #        |  numtim/2+1 <= pt <= numtim            | numtim-1 <= t <= numtim/2

    # local variables : dimvf, numlon, N, varspacetime, pee, wave, freq

    # logging.debug(f"[Hayashi] n_day_win: {n_day_win}, spd: {spd}")
    dimnames = var_fft.dims
    dimvf = var_fft.shape
    mlon = len(var_fft["wavenumber"])  # number of longitudes = numer of wavenumbers
    N = len(var_fft["frequency"])
    k_dim_index = dimnames.index("wavenumber")
    f_dim_index = dimnames.index("frequency")

    # logging.info(
    #     f"[Hayashi] input dims is {dimnames}, {dimvf} || Input dtype: {var_fft.dtype = }"
    # )
    # logging.info(f"[Hayashi] input coords is {var_fft.coords}")
    # logging.debug(
    #     f"[Hayashi] wavenumber axis is {k_dim_index}, frequency axis is {f_dim_index}"
    # )

    if len(dimnames) != len(var_fft.coords):
        # logging.error("The size of var_fft.coords is incorrect.")
        raise ValueError("STOP")

    nshape = list(dimvf)
    nshape[k_dim_index] += 1
    nshape[f_dim_index] += 1

    # logging.debug(f"[Hayashi] The nshape ends up being {nshape}")
    # this is a reordering, use Ellipsis to allow arbitrary number of dimensions,
    # but we insist that the wavenumber and frequency dims are rightmost.
    # we will fill the new array in increasing order (arbitrary choice)
    varspacetime = np.full(nshape, np.nan, dtype=type(var_fft))
    # first two are the negative wavenumbers (westward),
    # second two are the positive wavenumbers (eastward)
    # logging.debug(
    #     f"[Hayashi] Assign values into array. Notable numbers: mlon//2={mlon//2}, N//2={N//2}"
    # )

    varspacetime[..., 0 : mlon // 2, 0 : N // 2] = var_fft[
        ..., mlon // 2 : 0 : -1, N // 2 :
    ]  # neg.k, pos.w
    varspacetime[..., 0 : mlon // 2, N // 2 :] = var_fft[
        ..., mlon // 2 : 0 : -1, 0 : N // 2 + 1
    ]  # neg.k,
    varspacetime[..., mlon // 2 :, 0 : N // 2 + 1] = var_fft[
        ..., 0 : mlon // 2 + 1, N // 2 :: -1
    ]  # assign eastward & neg.freq.
    varspacetime[..., mlon // 2 :, N // 2 + 1 :] = var_fft[
        ..., 0 : mlon // 2 + 1, -1 : N // 2 - 1 : -1
    ]  # assign eastward & pos.freq.
    # logging.debug(f"[Hayashi] Shape after reordering: {varspacetime.shape}")
    # logging.debug(f"[Hayashi] Sum after reordering: {varspacetime.sum()}")
    # #  Create the real power spectrum pee = sqrt(real^2+imag^2)^2
    # logging.debug(
    #     f"[Hayashi] calculate power by absolute value (i.e. sqrt(real**2 + imag**2))and squaring."
    # )

    pee = (np.abs(varspacetime)) ** 2
    # logging.debug(
    #     f"[Hayashi] sum of pee {pee.sum()}. Type of pee: {type(pee)} Dtype: {pee.dtype}"
    # )
    # logging.debug(f"[Hayashi] put into DataArray")
    # add meta data for use upon return
    wave = np.arange(-mlon // 2, (mlon // 2) + 1, 1, dtype=int)
    freq = (
        np.linspace(-1 * n_day_win * spd / 2, n_day_win * spd / 2, (n_day_win * spd) + 1)
        / n_day_win
    )

    # logging.debug(f"[Hayashi] freq size is {freq.shape}.")
    odims = list(dimnames)
    odims[-2] = "wavenumber"
    odims[-1] = "frequency"
    ocoords = {}
    for c in var_fft.coords:
        # logging.debug(f"[hayashi] working on coordinate {c}")
        if (c != "wavenumber") and (c != "frequency"):
            ocoords[c] = var_fft[c]
        elif c == "wavenumber":
            ocoords["wavenumber"] = wave
        elif c == "frequency":
            ocoords["frequency"] = freq
    pee = xr.DataArray(pee, dims=odims, coords=ocoords)
    z = pee.copy()
    z.loc[{"frequency": 0}] = np.nan
    # logging.debug(f"[Hayashi] Sum at the end (removing zero freq): {z.sum().item()}")

    return pee


def split_hann_taper(series_length, fraction):
    """
    Implements `split cosine bell` taper of length `series_length` where only a
    fraction of points are tapered (combined on both ends).

    Parameters
    ----------
    series_length : int
        Length of the series to be tapered.
    fraction : float
        Fraction of the series to be tapered (combined on both ends).

    Returns
    -------
    series_taper : np.ndarray
            Array that tapers to zero on the ends.

    Note: to taper to the mean of a series X:
        XTAPER = (X - X.mean())*series_taper + X.mean()
    """

    npts = int(np.rint(fraction * series_length))  # total size of taper
    taper = np.hanning(npts)
    series_taper = np.ones(series_length)
    series_taper[0 : npts // 2 + 1] = taper[0 : npts // 2 + 1]
    series_taper[-npts // 2 + 1 :] = taper[npts // 2 + 1 :]

    return series_taper


def compute_spacetime_power(data, seg_size=96, n_overlap=60, spd=1, lat_bounds=None,
                            do_symmetries=False, rmv_low_freq=False, lat_aggreg="sum"):
    """
    Perform space-time spectral decomposition and return power spectrum following
    (M.C. Wheeler & G.N. Kiladis, 1999).

    Parameters
    ----------
    data : xr.DataArray
        Data to compute the spectrum of (with dimensions time, lat, lon).
    seg_size : int
        Size of the time segments to be decomposed, in days (default: 96).
    n_overlap : int
        Number of overlapping points between segments, in days (default: 60).
    spd : int
        Sampling rate, in samples/day (default: 1).
    lat_range : tuple
        Geographic latitude bounds (default: (-15, 15)).
    do_symmetries : bool
        Whether to follow NCL convention of putting symmetric component in SH,
        antisymmetric in NH. If True, the returned DataArray will have a `component`
        dimension (default: False).
    rmv_low_freq : bool
        Whether to remove low frequencies, below 1/seg_size (default: False).
    lat_aggreg : str
        Latitude aggregation method, either 'sum' or 'mean' (default: 'sum').

    Returns
    -------
    z_final : xr.DataArray
        Space-time power spectrum.

    Method
    ------
        1. Subsample in latitude if latitude_bounds is specified.
        2. Detrend the data (but keeps the mean value, as in NCL)
        3. High-pass filter if rmvLowFrq is True
        4. Construct symmetric/antisymmetric array if dosymmetries is True.
        5. Construct overlapping window view of data.
        6. Detrend the segments (strange enough, removing mean).
        7. Apply taper in time dimension of windows (aka segments).
        8. Fourier transform
        9. Apply Hayashi reordering to get propagation direction & convert to power.
       10. Return DataArray with power.
    """

    # Convert from days to time steps
    seg_size = spd * seg_size
    n_overlap = spd * n_overlap

    if lat_bounds is not None:
        assert isinstance(lat_bounds, tuple)
        data = data.sel(lat=slice(*lat_bounds))  # CAUTION: is this a mutable argument?
        # logging.info(f"Data reduced by latitude bounds. Size is {data.sizes}")
        slat = lat_bounds[0]
        nlat = lat_bounds[1]
    else:
        slat = data["lat"].min().item()
        nlat = data["lat"].max().item()

    # "Remove dominant signals"

    # "detrend" the data, including removing the mean (uses scipy.signal.detrend):
    #  --> ncl version keeps the mean:
    xmean = data.mean(dim="time")
    xdetr = detrend(data.values, axis=0, type="linear")
    xdetr = xr.DataArray(xdetr, dims=data.dims, coords=data.coords)
    xdetr += xmean  # put the mean back in
    # --> Tested and confirmed that this approach gives same answer as NCL

    # field testing it: pass
    # if not(hasattr(xdetr, "name")) or (not isinstance(xdetr.name, str) ):
    #     xdetr.name = "detrended"
    # xdetr.to_netcdf("/Users/brianpm/Documents/pout_0_detrend.nc")

    # filter low-frequencies
    if rmv_low_freq:
        data = rmv_annual_cycle(xdetr, spd, 1 / seg_size)
    # --> Tested and confirmed that this function gives same answer as NCL
    # testing: pass -- indistinguisable from file produced by NCL
    # data.name = "filtered"
    # data.to_netcdf("/Users/brianpm/Documents/pout_1_filtered.nc")

    # NOTE: we have altered "data" to be detrended & filtered at this point

    dimsizes = data.sizes  # dict
    lon_size = dimsizes["lon"]
    lat_size = dimsizes["lat"]
    lat_dim = data.dims.index("lat")
    if do_symmetries:
        data = decompose_to_sym_asym(data)
    # testing: pass -- Gets the same result as NCL.

    # logging.debug(
    #     f"[spacetime_power] data shape after removing low frequencies: {data.shape}"
    # )
    # logging.debug(
    #     f"[spacetime_power] variance of data before windowing: {np.var(data).item()}"
    # )

    # 2. Windowing with the xarray "rolling" operation, and then limit overlap with `construct` to produce a new dataArray.
    # WK99 recommend "2-month" overlap
    # Shape of x_win: (_, lat, lon, segments: spd*seg_size)
    x_roll = data.rolling(time=seg_size, min_periods=seg_size)  # WK99 use 96-day window
    assert seg_size - n_overlap > 0, (
        "Error, inconsistent specification of 'seg_size' and 'n_overlap' results in "
        f"stride of {seg_size - n_overlap}, but must be > 0."
    )
    x_win = x_roll.construct("segments")
    x_win = x_win.isel(time=slice(seg_size - 1, None, seg_size - n_overlap))

    # logging.debug(f"[spacetime_power] x_win shape is {x_win.shape}")
    # Additional detrend for each segment:
    if np.logical_not(np.any(np.isnan(x_win))):
        # logging.info("No missing, so use simplest segment detrend.")
        x_win_detr = detrend(
            x_win.values, axis=-1, type="linear"
        )  # <-- missing data makes this not work
        x_win = xr.DataArray(x_win_detr, dims=x_win.dims, coords=x_win.coords)
    else:
        # logging.warning(
        #     "EXTREME WARNING -- This method to detrend with missing values present does not quite work, probably need to do interpolation instead."
        # )
        # logging.warning(
        #     "There are missing data in x_win, so have to try to detrend around them."
        # )
        x_win_cp = x_win.values.copy()
        # logging.info(
        #     f"[spacetime_power] x_win_cp windowed data has shape {x_win_cp.shape} \n \t It is a numpy array, copied from x_win which has dims: {x_win.sizes} \n \t ** about to detrend this in the rightmost dimension."
        # )
        x_win_cp[np.logical_not(np.isnan(x_win_cp))] = detrend(
            x_win_cp[np.logical_not(np.isnan(x_win_cp))]
        )
        x_win = xr.DataArray(x_win_cp, dims=x_win.dims, coords=x_win.coords)
    # logging.debug(
    #     f"[spacetime_power] x_win variance of segments: {np.var(x_win, axis=(1,2,3)).values}"
    # )

    # 3. Taper in time to make the signal periodic, as required for FFT.
    # taper = np.hanning(seg_size)  # WK seem to use some kind of stretched out hanning window; unclear if it matters
    taper = split_hann_taper(seg_size, 0.1)  # try to replicate NCL's
    x_wintap = x_win * taper  # would do XTAPER = (X - X.mean())*series_taper + X.mean()
    # But since we have removed the mean, taper going to 0 is equivalent to taper going to the mean.
    # logging.debug(
    #     f"[spacetime_power] x_wintap variance of segments: {np.var(x_wintap, axis=(1,2,3)).values}"
    # )

    # Do the transform using 2D FFT
    # - normalize by dimension sizes
    z = np.fft.fft2(x_wintap, axes=(2, 3)) / (lon_size * seg_size)

    # NOTE: with this normalization, the power spectral density should
    #       be calculated as np.abs(z)**2 * dlon * dt * lon_size * seg_size
    #       where dt = 1/spd, so dt*seg_size=[length of segment in time]
    #       and dlon = lon[1]-lon[0] (= size of longitude dimension in degrees)
    #       _When only the positive frequencies are used, also multiply by 2._
    # AND: the integral of the power spectral density is then equal to the variance
    #      In this case, gets the variance of x_wintap; for suitably large seg_size,
    #      the tapering shouldn't matter much, so VAR[x_wintap] ≃ VAR[x_win]
    # TEST OF INT[psd] = VAR[x_wintap]:
    # dlon = x_wintap['lon'][1].item()-x_wintap['lon'][0].item()
    # dt = 1/spd
    # Nlon = lon_size
    # Nt = seg_size
    # print(f"{dlon = }, {dt = }, {Nlon = }, {Nt = }")
    # psd = (np.abs(z)**2) * dlon * dt * (Nlon * Nt)
    # logging.debug(f"{psd.shape = }")
    # variance = np.var(x_wintap, axis=(2,3))
    # logging.debug(f"VARIANCE ARRAY IS LEFT AS: {variance.shape = }")
    # kx = np.fft.fftfreq(Nlon, dlon)
    # ky = np.fft.fftfreq(Nt, dt)
    # psd_integral = np.sum(psd, axis=(2,3))* (kx[1]-kx[0]) * (ky[1]-ky[0])
    # logging.debug(f"PSD_INTEGRAL SHAPE: {psd_integral.shape}")
    # print(f"Variance of data (0,0): {variance[0,0]} -- Integral of spectrum: {psd_integral[0,0]}")

    # z has both positive & negative frequencies : usually you'd take the positive in each dimension and double it

    # Or do the transform with 2 steps (equivalent!)
    # z = np.fft.fft(x_wintap, axis=2) / lon_size  # note that np.fft.fft() produces same answers as NCL cfftf
    # z = np.fft.fft(z, axis=3) / seg_size

    z = xr.DataArray(
        z,
        dims=("time", "lat", "wavenumber", "frequency"),
        coords={
            "time": x_wintap["time"],
            "lat": x_wintap["lat"],
            "wavenumber": np.fft.fftfreq(lon_size, 1 / lon_size),
            "frequency": np.fft.fftfreq(seg_size, 1 / spd),
        },
    )

    # The FFT is returned following ``standard order`` which has negative frequencies in second half of array.
    #
    # IMPORTANT:
    # If this were typical 2D FFT, we would do the following to get the frequencies and reorder:
    #         z_k = np.fft.fftfreq(x_wintap.shape[-2], 1/lon_size)
    #         z_v = np.fft.fftfreq(x_wintap.shape[-1], 1)  # Assumes 1/(1-day) timestep
    # reshape to get the frequencies centered
    #         z_centered = np.fft.fftshift(z, axes=(2,3))
    #         z_k_c = np.fft.fftshift(z_k)
    #         z_v_c = np.fft.fftshift(z_v)
    # and convert to DataArray as this:
    #         d1 = list(x_win.dims)
    #         d1[-2] = "wavenumber"
    #         d1[-1] = "frequency"
    #         c1 = {}
    #         for d in d1:
    #             if d in x_win.coords:
    #                 c1[d] = x_win[d]
    #             elif d == "wavenumber":
    #                 c1[d] = z_k_c
    #             elif d == "frequency":
    #                 c1[d] = z_v_c
    #         z_centered = xr.DataArray(z_centered, dims=d1, coords=c1)
    # BUT THAT IS INCORRECT TO GET THE PROPAGATION DIRECTION OF ZONAL WAVES
    # (in testing, it seems to end up opposite in wavenumber)
    # Apply reordering per Hayashi to get correct wave propagation convention
    #     this function is customized to expect z to be a DataArray
    z_pee = resolve_waves_Hayashi(z, seg_size // spd, spd)
    # z_pee is spectral power already.
    # z_pee is a DataArray w/ coordinate vars for wavenumber & frequency

    # average over all available segments and sum over latitude
    # OUTPUT DEPENDS ON SYMMETRIES
    if do_symmetries:
        # multipy by 2 b/c we only used one hemisphere
        z_symmetric = 2.0 * z_pee.isel(lat=z_pee.lat < 0).mean(dim="time")
        z_symmetric = apply_lat_aggregation(z_symmetric, lat_aggreg)
        z_symmetric.name = "power"
        z_antisymmetric = 2.0 * z_pee.isel(lat=z_pee.lat > 0).mean(dim="time")
        z_antisymmetric = apply_lat_aggregation(z_antisymmetric, lat_aggreg)
        z_antisymmetric.name = "power"
        z_final = xr.concat([z_symmetric, z_antisymmetric], "component")
        z_final = z_final.assign_coords({"component": ["symmetric", "antisymmetric"]})
    else:
        lat = z_pee["lat"]
        lat_inds = np.argwhere(((lat <= nlat) & (lat >= slat)).values).squeeze()
        z_final = z_pee.isel(lat=lat_inds).mean(dim="time")
        z_final = apply_lat_aggregation(z_final, lat_aggreg)

    return z_final


def gen_dispersion_curves(n_wave_type=6, n_planetary_wave=50, rlat=0.0, ahe=[50.0, 25.0, 12.0]):
    """
    Derive the shallow water dispersion curves.

    Parameters
    ----------
    n_wave_type : int
        Number of wave types (both symmetric and antisymmetric) to compute (default: 6).
        In the default case, the wave types are:
            - 0,1,2 (ASYMMETRIC): "MRG", "IG", "EIG" (mixed rossby gravity, inertial
            gravity, equatorial inertial gravity)
            - 3,4,5 (SYMMETRIC): "Kelvin", "ER", "IG" (Kelvin, equatorial rossby,
            inertial gravity)
    n_planetary_wave : int
        Number of planetary waves to compute (default: 50).
    rlat : float
        Latitude in radians (just one latitude) (default: 0.0).
    ahe : list[float]
        Equivalent depths to compute (default: [50., 25., 12.]).

    Returns
    -------
    afreq : np.ndarray
        Frequency, shape is (n_wave_type, n_equiv_depth, n_planetary_wave).
    apzwn : np.ndarray
        Zonal savenumber, shape is (n_wave_type, n_equiv_depth, n_planetary_wave).
    """

    n_equiv_depth = len(ahe)  # this was an input originally, but I don't know why.
    pi = np.pi
    radius = 6.37122e06  # [m]   average radius of earth
    g = 9.80665  # [m/s] gravity at 45 deg lat used by the WMO
    omega = 7.292e-05  # [1/s] earth's angular vel
    # U     = 0.0   # NOT USED, so Commented
    # Un    = 0.0   # since Un = U*T/L  # NOT USED, so Commented
    ll = 2.0 * pi * radius * np.cos(np.abs(rlat))
    beta = 2.0 * omega * np.cos(np.abs(rlat)) / radius
    fillval = 1e20

    # NOTE: original code used a variable called del,
    #       I just replace that with `dell` because `del` is a python keyword.

    # Initialize the output arrays
    afreq = np.empty((n_wave_type, n_equiv_depth, n_planetary_wave))
    apzwn = np.empty((n_wave_type, n_equiv_depth, n_planetary_wave))

    for ww in range(1, n_wave_type + 1):
        for ed, he in enumerate(ahe):
            # This loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(beta)*(g*he)**(0.25) This is close to pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed
            L = np.sqrt(
                c / beta
            )  # was: (g*he)**(0.25)/np.sqrt(beta), this is Rossby radius of deformation

            for wn in range(1, n_planetary_wave + 1):
                s = -20.0 * (wn - 1) * 2.0 / (n_planetary_wave - 1) + 20.0
                k = 2.0 * np.pi * s / ll
                kn = k * L

                # Anti-symmetric curves
                if ww == 1:  # MRG wave
                    if k < 0:
                        dell = np.sqrt(1.0 + (4.0 * beta) / (k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)

                    if k == 0:
                        deif = np.sqrt(c * beta)

                    if k > 0:
                        deif = fillval

                if ww == 2:  # n=0 IG wave
                    if k < 0:
                        deif = fillval

                    if k == 0:
                        deif = np.sqrt(c * beta)

                    if k > 0:
                        dell = np.sqrt(1.0 + (4.0 * beta) / (k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)

                if ww == 3:  # n=2 IG wave
                    n = 2.0
                    dell = beta * c
                    deif = np.sqrt((2.0 * n + 1.0) * dell + (g * he) * k**2)

                    # Apply some corrections to the above calculated frequency.......
                    for i in range(1, 5 + 1):
                        deif = np.sqrt(
                            (2.0 * n + 1.0) * dell + (g * he) * k**2 + g * he * beta * k / deif
                        )

                # Symmetric curves
                if ww == 4:  # n=1 ER wave
                    n = 1.0
                    if k < 0.0:
                        dell = (beta / c) * (2.0 * n + 1.0)
                        deif = -beta * k / (k**2 + dell)
                    else:
                        deif = fillval

                if ww == 5:  # Kelvin wave
                    deif = k * c

                if ww == 6:  # n=1 IG wave
                    n = 1.0
                    dell = beta * c
                    deif = np.sqrt((2.0 * n + 1.0) * dell + (g * he) * k**2)

                    # Apply some corrections to the above calculated frequency
                    for i in range(1, 5 + 1):
                        deif = np.sqrt(
                            (2.0 * n + 1.0) * dell + (g * he) * k**2 + g * he * beta * k / deif
                        )

                eif = deif  # + k*U since  U=0.0
                P = 2.0 * np.pi / (eif * 24.0 * 60.0 * 60.0)  #  => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.

                apzwn[ww - 1, ed - 1, wn - 1] = s
                if deif != fillval:
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would re-calculate now
                    afreq[ww - 1, ed - 1, wn - 1] = 1.0 / P
                else:
                    afreq[ww - 1, ed - 1, wn - 1] = fillval

    return afreq, apzwn



# Original functions to compute the power spectra (not adapted from the wavenumber_frequency repository)

def variable_smooth_wavefreq(data, freq_dim="frequency", wavenum_dim="wavenumber"):
    """
    Compute background spectrum by smoothing with a filter in frequency and wavenumber.
    Following (M.C. Wheeler & G.N. Kiladis, 1999), the smoothing is done by using a
    1-2-1 filter and performing 10 passes in frequency to all frequencies and, then,
    10/20/30/40 passes in wavenumber depending on the frequency range (in increasing
    frequency).

    Parameters
    ----------
    data : xr.DataArray
        Data array to be smoothed.
    freq_dim : str
        Name of the frequency dimension (default: 'frequency').
    wavenum_dim : str
        Name of the wavenumber dimension (default: 'wavenumber').

    Returns
    -------
    smoothed : xr.DataArray
        Smoothed data array.
    """

    # Load coordinates and data
    freq = data[freq_dim]
    wnum = data[wavenum_dim]
    data_smoothed = data.data.copy()

    # Define 1-2-1 kernel
    kernel = np.array([1, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()

    # Smooth 10 times in frequency
    for _ in range(10):
        data_smoothed = convolve1d(
            data_smoothed,
            weights=kernel,
            axis=data.get_axis_num(freq_dim),
            mode="nearest"
        )

    # Smooth in wavenumber with stepped passes
    n_freq = freq.size
    freq_axis = data.get_axis_num(freq_dim)
    wnum_axis = data.get_axis_num(wavenum_dim)

    # Define ranges
    split = n_freq // 4

    # Ensure all bins are covered
    pass_map = [10] * split + [20] * split + [30] * split + [40] * (n_freq - 3 * split)

    # Uncomment to do it in just 2 steps rather than 3
    # split = n_freq // 3
    # pass_map = [10] * split + [25] * split + [40] * (n_freq - 2*split)  # Ensure all bins are covered

    # Apply smoothing in wavenumber for each frequency bin
    for i, n_passes in enumerate(pass_map):
        indexer = [slice(None)] * data.ndim
        indexer[freq_axis] = i

        for _ in range(n_passes):
            data_smoothed[tuple(indexer)] = convolve1d(
                data_smoothed[tuple(indexer)],
                weights=kernel,
                axis=wnum_axis,
                mode="nearest"
            )

    # Save resulsts to xr.DataArray
    smoothed = xr.DataArray(
        data_smoothed,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs,
        name="background_spectrum",
    )

    return smoothed


def wavenum_freq_analysis(data, seg_size=96, n_overlap=60, lat_range=(-15, 15)):
    """
    Perform wavenumber-frequency analysis and return the normalized spectral symmetric
    and antisymmetric components obtained dividing by a smoothed background following
    (M.C. Wheeler & G.N. Kiladis, 1999).

    Parameters
    ----------
    data : xr.DataArray
        Data to compute the spectrum of.
    seg_size : int
        Size of the segments to perform the spectral analysis on, in days (default: 96).
    n_overlap : int
        Number of overlapping points between segments, in days (default: 60).
    lat_range : tuple
        Geographic latitude bounds (default: (-15, 15)).

    Returns
    -------
    nspec_sym : xr.DataArray
        Normalized spectrum of the symmetric component.
    nspec_asy : xr.DataArray
        Normalized spectrum of the antisymmetric component.
    z2_sym : xr.DataArray
        Raw spectral power of the symmetric component.
    z2_asy : xr.DataArray
        Raw spectral power of the antisymmetric component.
    background : xr.DataArray
        Smoothed background spectrum used for normalization.
    """

    # Validate input
    if not isinstance(data, xr.DataArray):
        raise ValueError("Input data must be an xarray DataArray.")

    # Get the "raw" spectral power (original function from the wavenumber_frequency repository)
    z2 = compute_spacetime_power(
        data,
        seg_size,
        n_overlap,
        spd=1,
        lat_bounds=lat_range,
        do_symmetries=True,
        rmv_low_freq=True,
    )
    z2avg = z2.mean(dim="component")

    # Get rid of spurious power at \nu = 0
    z2.loc[{"frequency": 0}] = np.nan

    # Compute background (derived from both symmetric and antisymmetric)
    background = variable_smooth_wavefreq(z2avg, freq_dim="frequency", wavenum_dim="wavenumber")

    # Separate components
    z2_sym = z2[0, ...]
    z2_asy = z2[1, ...]

    # Normalize by background
    nspec_sym = z2_sym / background
    nspec_asy = z2_asy / background

    # Add names to the data arrays
    nspec_sym.name = "sym_spec"
    nspec_asy.name = "asym_spec"
    background.name = "background"

    return nspec_sym, nspec_asy, z2_sym, z2_asy, background


def wavenum_freq_analysis_wrapper(args):
    """
    Wrapper function to perform wavenumber-frequency analysis with a single argument,
    for use with multiprocessing.

    Parameters
    ----------
    args : tuple
        Tuple containing (data, seg_size, n_overlap, lat_range).

    Returns
    -------
    result : tuple
        Result of wavenumber_freq_analysis.
    """

    data, seg_size, n_overlap, lat_range = args
    result = wavenum_freq_analysis(data, seg_size, n_overlap, lat_range)

    return result



# Original functions to postprocess the power spectra (not adapted from the wavenumber_frequency repository)

def sum_power_over_area( power, freq_bounds=None, wavenum_bounds=None, freq_dim="frequency",
                         wavenum_dim="wavenumber"):
    """
    Sum power over a specified area in wavenumber-frequency space.

    Parameters
    ----------
    power : xr.DataArray
        Power spectrum data.
    freq_bounds : tuple
        Frequency bounds for summation. If None, all values
        are included.
    wavenum_bounds : tuple
        Wavenumber bounds for summation. If None, all values
        are included.
    freq_dim : str
        Name of the frequency dimension (default: 'frequency').
    wavenum_dim : str
        Name of the wavenumber dimension (default: 'wavenumber').

    Returns
    -------
    power_sum : float
        Sum of power over the specified area.
    """

    # Validate input
    if not isinstance(power, xr.DataArray):
        raise ValueError("Input power must be an xr.DataArray.")
    if freq_dim not in power.dims or wavenum_dim not in power.dims:
        raise ValueError(
            f"Specified dimensions '{freq_dim}' and/or '{wavenum_dim}' not found in the power xr.DataArray."
        )

    # Select the area in wavenumber-frequency space
    if freq_bounds is not None:
        if freq_bounds[0] > freq_bounds[1]:
            raise ValueError(
                "Invalid frequency bounds: lower bound must be smaller than upper bound."
            )
        power_freq_filtered = power.sel({freq_dim: slice(freq_bounds[0], freq_bounds[1])})
    else:
        power_freq_filtered = power
    if wavenum_bounds is not None:
        if wavenum_bounds[0] > wavenum_bounds[1]:
            raise ValueError(
                "Invalid wavenumber bounds: lower bound must be smaller than upper bound."
            )
        power_filtered = power_freq_filtered.sel(
            {wavenum_dim: slice(wavenum_bounds[0], wavenum_bounds[1])}
        )
    else:
        power_filtered = power_freq_filtered


    # Sum over the specified area (both over wavenumber and frequency)
    power_sum = power_filtered.sum(skipna=True).item()

    return power_sum


def compute_eastward_westward_ratio(power, freq_bounds=None, wavenum_bounds=None,
                                    freq_dim="frequency", wavenum_dim="wavenumber"):
    """
    Compute the ratio of eastward to westward power in a specified area of
    wavenumber-frequency space.

    Parameters
    ----------
    power : xr.DataArray
        Power spectrum data.
    freq_bounds : tuple
        Frequency bounds for summation.
    wavenum_bounds : tuple
        Positive wavenumber bounds for summation.
    freq_dim : str
        Name of the frequency dimension (default: 'frequency').
    wavenum_dim : str
        Name of the wavenumber dimension (default: 'wavenumber').

    Returns
    -------
    ratio : float
        Ratio of eastward to westward power in the specified area.
    eastward_sum : float
        Sum of eastward power in the specified area.
    westward_sum : float
        Sum of westward power in the specified area.
    """

    # Validate input
    if not isinstance(power, xr.DataArray):
        raise ValueError("Input power must be an xr.DataArray.")
    if freq_bounds is None or freq_bounds[0] > freq_bounds[1]:
        raise ValueError(
            "Frequency bounds 'freq_bounds' must be specified and valid "
            "(lower bound must be smaller than upper bound)."
        )
    if (
        wavenum_bounds is None
        or wavenum_bounds[0] < 0
        or wavenum_bounds[1] < 0
        or wavenum_bounds[0] > wavenum_bounds[1]
    ):
        raise ValueError(
            "Wavenumber bounds 'wavenum_bounds' must be specified, positive and valid "
            "(lower bound must be smaller than upper bound)."
        )
    if freq_dim not in power.dims or wavenum_dim not in power.dims:
        raise ValueError(
            f"Specified dimensions '{freq_dim}' and/or '{wavenum_dim}' not found in the power xr.DataArray."
        )

    # Compute eastward and westward sums
    eastward_sum = sum_power_over_area(power, freq_bounds, wavenum_bounds, freq_dim, wavenum_dim)
    westward_sum = sum_power_over_area(
        power, freq_bounds, (-wavenum_bounds[1], -wavenum_bounds[0]), freq_dim, wavenum_dim
    )

    # Compute ratio
    if westward_sum == 0:
        raise ValueError("Westward power sum is zero, cannot compute eastward/westward ratio.")
    else:
        ratio = eastward_sum / westward_sum

    return ratio, eastward_sum, westward_sum


def compute_power_periodicity(power, freq_bounds=None, wavenum_bounds=None, freq_dim="frequency",
                              wavenum_dim="wavenumber"):
    """
    Compute the power-weighted mean period from the wavenumber-frequency power spectra (P_WFPS)
    in a specified area.

    Parameters:
    power : xr.DataArray
        Power spectrum data.
    freq_bounds : tuple
        Frequency bounds for analysis. If None, all frequencies
        are included.
    wavenum_bounds : tuple
        Wavenumber bounds for analysis. If None, all wavenumbers
        are included.
    freq_dim : str
        Name of the frequency dimension (default: 'frequency').
    wavenum_dim : str
        Name of the wavenumber dimension (default: 'wavenumber').

    Returns:
    -------
    pwfps : float
        Power-weighted mean period in the specified area.
    """

    # Validate input
    if not isinstance(power, xr.DataArray):
        raise ValueError("Input power must be an xr.DataArray.")
    if freq_dim not in power.dims or wavenum_dim not in power.dims:
        raise ValueError(
            f"Specified dimensions '{freq_dim}' and/or '{wavenum_dim}' not found in the power xr.DataArray."
        )

    # Select the area in wavenumber-frequency space
    if freq_bounds is not None:
        if freq_bounds[0] > freq_bounds[1]:
            raise ValueError(
                "Invalid frequency bounds: lower bound must be smaller than upper bound."
            )
        power_freq_filtered = power.sel({freq_dim: slice(freq_bounds[0], freq_bounds[1])})
    else:
        power_freq_filtered = power
    if wavenum_bounds is not None:
        if wavenum_bounds[0] > wavenum_bounds[1]:
            raise ValueError(
                "Invalid wavenumber bounds: lower bound must be smaller than upper bound."
            )
        power_filtered = power_freq_filtered.sel(
            {wavenum_dim: slice(wavenum_bounds[0], wavenum_bounds[1])}
        )
    else:
        power_filtered = power_freq_filtered

    # Compute power-weighted sum of periods
    freq_values = power_filtered[freq_dim].values
    if np.any(freq_values == 0):
        raise ValueError("Frequency values include zero, cannot compute periods (1/freq).")

    period_values = 1 / freq_values
    power_weighted_periods = (power_filtered * period_values).sum(skipna=True).item()

    # Compute total power in the area
    total_power = power_filtered.sum(skipna=True).item()
    if total_power == 0:
        raise ValueError("Total power in the specified area is zero, cannot compute periodicity.")

    # Compute period from the wavenumber-frequency power spectra (P_WFPS)
    pwfps = power_weighted_periods / total_power

    return pwfps
