"""
This module contains functions adapted from the CyMeP package
Original source: https://github.com/zarzycki/cymep
Original author: Colin Zarzycki
License: MIT License
Copyright (c) 2021 Colin Zarzycki
"""

import os
import re
import numpy as np
import xarray as xr
import netCDF4 as nc

from datetime import datetime


# Functions adapted from cymep/functions/getTrajectories.py

def getTrajectories(filename, nVars, headerDelimStr, isUnstruc):
    """
    Read trajectory data from a TempestExtremes file format.

    Parameters:
    ----------
    filename : str
        Path to input trajectory file.
    nVars : int
        Number of variables per trajectory point (-1 for auto-detection).
    headerDelimStr : str
        String that marks header lines (e.g. "start").
    isUnstruc : bool
        If True, adds an extra column for unstructured grid data.

    Returns:
    ----------
    numtraj : int
        Number of trajectories found.
    maxNumPts : int
        Maximum length of any trajectory.
    ncols : int
        Number of data columns per point.
    prodata : np.ndarray
        Array of shape (nvars, ntraj, maxpts) containing trajectory data.
    """

    print("Getting trajectories from TempestExtremes file...")
    print("Running getTrajectories on '%s' with unstruc set to '%s'" % (filename, isUnstruc))
    print("nVars set to %d and headerDelimStr set to '%s'" % (nVars, headerDelimStr))

    # Using the newer with construct to close the file automatically.
    with open(filename) as f:
        data = f.readlines()

    # Find total number of trajectories and maximum length of trajectories
    numtraj = 0
    numPts = []

    for line in data:
        if headerDelimStr in line:
            # if header line, store number of points in given traj in numPts
            headArr = line.split()
            numtraj += 1
            numPts.append(int(headArr[1]))
        else:
            # if not a header line, and nVars = -1, find number of columns in data point
            if nVars < 0:
                nVars = len(line.split())

    maxNumPts = max(numPts)  # Maximum length of ANY trajectory
    print("Found %d columns" % nVars)
    print("Found %d trajectories" % numtraj)

    # Initialize storm and line counter
    stormID = -1
    lineOfTraj = -1

    # Create array for data
    if isUnstruc:
        prodata = np.empty((nVars + 1, numtraj, maxNumPts))
    else:
        prodata = np.empty((nVars, numtraj, maxNumPts))
    prodata[:] = np.nan

    for i, line in enumerate(data):
        if headerDelimStr in line:  # check if header string is satisfied
            stormID += 1  # increment storm
            lineOfTraj = 0  # reset trajectory line to zero
        else:
            ptArr = line.split()
            for jj in range(nVars):
                if isUnstruc:
                    prodata[jj + 1, stormID, lineOfTraj] = ptArr[jj]
                else:
                    prodata[jj, stormID, lineOfTraj] = ptArr[jj]
            lineOfTraj += 1  # increment line

    # Make sure we return the correct size of the array
    if isUnstruc:
        ncols = nVars + 1
    else:
        ncols = nVars

    print("... done reading data")
    return numtraj, maxNumPts, ncols, prodata


def writeTrajectories(filename, data, ntraj, npts, headerDelimStr="start"):
    """
    Write trajectory data to a file in TempestExtremes format.

    Parameters:
    ----------
    filename : str
        Output file path.
    data : np.ndarray
        Data array of shape (nvars, ntraj, maxpts).
    ntraj : int
        Number of trajectories.
    npts : int
        Max number of points.
    headerDelimStr : str
        Header delimiter string (default: 'start').
    """

    print(f"Writing TempestExtremes file to {filename}...")
    with open(filename, "w") as f:
        for storm in range(ntraj):
            # Find valid points for this storm
            valid_points = ~np.isnan(data[0, storm, :])
            num_valid = int(np.sum(valid_points))

            # Get time values from last 4 columns for header
            yyyy = int(data[-4, storm, 0])
            mm = int(data[-3, storm, 0])
            dd = int(data[-2, storm, 0])
            hh = int(data[-1, storm, 0])

            # Write header
            f.write(f"{headerDelimStr}\t{num_valid}\t{yyyy}\t{mm}\t{dd}\t{hh}\n")

            # Write data points
            for pt in range(num_valid):
                line_data = []
                # Process all columns except time columns
                for col in range(data.shape[0] - 4):
                    val = data[col, storm, pt]
                    if col <= 1:  # First two columns are integers
                        line_data.append(f"{int(val)}")
                    else:
                        line_data.append(f"{val:e}")

                # Add time columns
                for timevar in [-4, -3, -2, -1]:
                    line_data.append(f"{int(data[timevar, storm, pt])}")

                f.write("\t" + "\t".join(line_data) + "\n")


def getNodes(filename, nVars, isUnstruc):
    """
    Read node data from a TempestExtremes nodes file format.

    Parameters:
    ----------
    filename : str
        Path to input nodes file.
    nVars : int
        Number of variables per node (-1 for auto-detection).
    isUnstruc : bool
        If True, adds extra columns for unstructured grid data.

    Returns:
    ----------
    numnodetimes : int
        Number of timesteps with nodes.
    maxNumPts : int
        Maximum number of nodes at any timestep.
    prodata : np.ndarray
        Array of shape (nvars+4/5, numnodetimes, maxpts) containing node data.
    """

    print("Getting nodes from TempestExtremes file...")

    # Using the newer with construct to close the file automatically.
    with open(filename) as f:
        data = f.readlines()

    # Find total number of trajectories and maximum length of trajectories
    numnodetimes = 0
    numPts = []

    for line in data:
        if re.match(r"\w", line):
            # if header line, store number of points in given traj in numPts
            headArr = line.split()
            numnodetimes += 1
            numPts.append(int(headArr[3]))
        else:
            # if not a header line, and nVars = -1, find number of columns in data point
            if nVars < 0:
                nVars = len(line.split())

    maxNumPts = max(numPts)  # Maximum length of ANY trajectory
    print("Found %d columns" % nVars)
    print("Found %d trajectories" % numnodetimes)
    print("Found %d maxNumPts" % maxNumPts)

    # Initialize storm and line counter
    stormID = -1
    lineOfTraj = -1

    # Create array for data
    if isUnstruc:
        prodata = np.empty((nVars + 5, numnodetimes, maxNumPts))
    else:
        prodata = np.empty((nVars + 4, numnodetimes, maxNumPts))

    prodata[:] = np.nan
    nextHeadLine = 0

    for i, line in enumerate(data):
        if re.match(r"\w", line):  # check if header string is satisfied
            stormID += 1  # increment storm
            lineOfTraj = 0  # reset trajectory line to zero
            headArr = line.split()
            YYYY = int(headArr[0])
            MM = int(headArr[1])
            DD = int(headArr[2])
            HH = int(headArr[4])
        else:
            ptArr = line.split()
            for jj in range(nVars - 1):
                if isUnstruc:
                    prodata[jj + 1, stormID, lineOfTraj] = ptArr[jj]
                else:
                    prodata[jj, stormID, lineOfTraj] = ptArr[jj]
            if isUnstruc:
                prodata[nVars + 1, stormID, lineOfTraj] = YYYY
                prodata[nVars + 2, stormID, lineOfTraj] = MM
                prodata[nVars + 3, stormID, lineOfTraj] = DD
                prodata[nVars + 4, stormID, lineOfTraj] = HH
            else:
                prodata[nVars, stormID, lineOfTraj] = YYYY
                prodata[nVars + 1, stormID, lineOfTraj] = MM
                prodata[nVars + 2, stormID, lineOfTraj] = DD
                prodata[nVars + 3, stormID, lineOfTraj] = HH
            lineOfTraj += 1  # increment line

    print("... done reading data")
    return numnodetimes, maxNumPts, prodata



# Functions adapted from cymep/functions/mask_tc.py

def maskTC(lat, lon, dohemi=False):
    """
    Classify a tropical cyclone (TC) into a basin or hemisphere based on latitude and longitude.

    Parameters
    ----------
    lat : float
    lon : float
        Longitude of the TC (degrees).
    dohemi : bool
        Whether to only classify by hemisphere (if True) or basin-by-basin using
        latitude/longitude boundaries (if False) (default: False).

    Returns
    -------
    basin : int
        Integer code for the basin/hemisphere:
            - 0 → Unclassified / outside defined basins
            - 1 → North Atlantic (NATL)
            - 2 → Eastern Pacific (EPAC)
            - 3 → Central Pacific (CPAC)
            - 4 → Western Pacific (WPAC)
            - 5 → North Indian Ocean (NIO)
            - 6 → South Indian Ocean (SIO)
            - 7 → South Pacific (SPAC)
            - 20 → Northern Hemisphere (if `dohemi=True`) (NHEMI)
            - 21 → Southern Hemisphere (if `dohemi=True`) (SHEMI)
    """

    # If lon is negative, switch to [0, 360] convention
    if lon < 0.0:
        lon = lon + 360.0

    if dohemi == True:
        if lat >= 0.0:
            basin = 20
        else:
            basin = 21
    else:
        # Coefficients for calculating ATL/EPAC sloped line
        m = -0.58
        b = 0.0 - m * 295.0
        maxlat = 50.0

        if lat >= 0.0 and lat <= maxlat and lon > 257.0 and lon <= 359.0:
            funcval = m * lon + b
            if lat > funcval:
                basin = 1
            else:
                basin = 2
        elif lat >= 0.0 and lat <= maxlat and lon > 220.0 and lon <= 257.0:
            basin = 2
        elif lat >= 0.0 and lat <= maxlat and lon > 180.0 and lon <= 220.0:
            basin = 3
        elif lat >= 0.0 and lat <= maxlat and lon > 100.0 and lon <= 180.0:
            basin = 4
        elif lat >= 0.0 and lat <= maxlat and lon > 30.0 and lon <= 100.0:
            basin = 5
        elif lat < 0.0 and lat >= -maxlat and lon > 30.0 and lon <= 135.0:
            basin = 6
        elif lat < 0.0 and lat >= -maxlat and lon > 135.0 and lon <= 290.0:
            basin = 7
        else:
            basin = 0

    return basin


def getbasinmaskstr(gridchoice):
    """
    Map a basin or hemisphere integer code (or list) to a string label.

    Parameters
    ----------
    gridchoice : int or list[int]):
        Basin/hemisphere code(s). If a sequence is provided, only the first element
        is considered. Codes are:
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

    Returns
    -------
    strbasin : str
        String label corresponding to the input basin/hemisphere code.
    """

    if hasattr(gridchoice, "__len__"):
        if gridchoice[0] == 1:
            strbasin = "NHEMI"
        else:
            strbasin = "SHEMI"
    else:
        if gridchoice < 0:
            strbasin = "GLOB"
        else:
            if gridchoice == 1:
                strbasin = "NATL"
            elif gridchoice == 2:
                strbasin = "EPAC"
            elif gridchoice == 3:
                strbasin = "CPAC"
            elif gridchoice == 4:
                strbasin = "WPAC"
            elif gridchoice == 5:
                strbasin = "NIO"
            elif gridchoice == 6:
                strbasin = "SIO"
            elif gridchoice == 7:
                strbasin = "SPAC"
            elif gridchoice == 8:
                strbasin = "SATL"
            elif gridchoice == 9:
                strbasin = "FLA"
            elif gridchoice == 20:
                strbasin = "NHEMI"
            elif gridchoice == 21:
                strbasin = "SHEMI"
            else:
                strbasin = "NONE"

    return strbasin



# Functions adapted from cymep/functions/pattern_corr.py

def pattern_cor(x, y, w, opt):
    """
    Compute the weighted pattern correlation between two spatial fields.

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
        Second spatial field (lat x lon).
    w : np.ndarray
        Weights to apply.
    opt : int
        Correlation option (0 for anomaly correlation (mean removed before correlation),
        otherwise, direct correlation without mean removal).

    Returns
    -------
    r : float
        Weighted correlation coefficient between the spatial fields.
    """

    if x.shape != y.shape:
        print("Shapes of x and y do not match!")
        quit()

    xdims = x.shape
    nlat = xdims[0]
    nlon = xdims[1]

    if np.isscalar(w):
        WGT = np.empty((nlat, nlon))
        WGT[:] = w
    else:
        if w.ndim == 1:
            w = np.expand_dims(w, axis=1)
            WGT = np.repeat(w, nlon, axis=1)
        elif w.ndim == 2:
            WGT = w
        else:
            quit()

    # Set weight to 0 where either x or y is nan/missing
    WGT = np.where(np.isnan(x), 0.0, WGT)
    WGT = np.where(np.isnan(y), 0.0, WGT)

    if opt == 0:
        sumWGT = np.nansum(WGT)
        xAvgArea = np.nansum(x * WGT) / sumWGT
        yAvgArea = np.nansum(y * WGT) / sumWGT

        xAnom = x - xAvgArea
        yAnom = y - yAvgArea

        xyCov = np.nansum(WGT * xAnom * yAnom)
        xAnom2 = np.nansum(WGT * xAnom**2)
        yAnom2 = np.nansum(WGT * yAnom**2)
    else:
        xyCov = np.nansum(WGT * x * y)
        xAnom2 = np.nansum(WGT * x**2)
        yAnom2 = np.nansum(WGT * y**2)

    # Calculate coefficient
    r = xyCov / (np.sqrt(xAnom2) * np.sqrt(yAnom2))

    return r


def wgt_arearmse2(x, y, w, opt):
    """
    Compute the weighted area Root Mean Square Error (RMSE) between two spatial fields.

    Parameters
    ----------
    x : np.ndarray
        First spatial field (lat x lon).
    y : np.ndarray
        Second spatial field (lat x lon).
    w : np.ndarray
        Weights to apply.
    opt : int
        Option for handling zeros (0 to use raw values, and 1 to set matching 0s
        to NaNs (to avoid getting 0 when obs 0)).

    Returns
    -------
    rmse : float
        Weighted area RMSE between the spatial fields.
    """

    if x.shape != y.shape:
        print("Shapes of x and y do not match!")
        quit()

    sumd = 0.0
    sumw = 0.0

    xdims = x.shape
    nlat = xdims[0]
    nlon = xdims[1]

    if np.isscalar(w):
        WGT = np.empty((nlat, nlon))
        WGT[:] = w
    else:
        if w.ndim == 1:
            w = np.expand_dims(w, axis=1)
            WGT = np.repeat(w, nlon, axis=1)
        elif w.ndim == 2:
            WGT = w
        else:
            quit()

    # Set weight to 0 where either x or y is nan/missing
    WGT = np.where(np.isnan(x), 0.0, WGT)
    WGT = np.where(np.isnan(y), 0.0, WGT)

    if opt == 1:
        xtmp = np.where(np.logical_and(x == 0, y == 0), float("NaN"), x)
        ytmp = np.where(np.logical_and(x == 0, y == 0), float("NaN"), y)
    else:
        xtmp = x
        ytmp = y

    for ii in range(nlat):
        for jj in range(nlon):
            if ~np.isnan(xtmp[ii, jj]) and ~np.isnan(ytmp[ii, jj]):
                sumd = sumd + WGT[ii, jj] * (xtmp[ii, jj] - ytmp[ii, jj]) ** 2
                sumw = sumw + WGT[ii, jj]
                # SUMD = SUMD + WGT(ML,NL)* (T(ML,NL)-Q(ML,NL))**2
                # SUMW = SUMW + WGT(ML,NL)

    if sumw != 0.0:
        rmse = np.sqrt(sumd / sumw)

    return rmse


def wgt_areaave2(x, w, opt):
    """
    Compute the weighted area average of a spatial field.

    Parameters
    ----------
    x : np.ndarray
        Input spatial field (lat x lon).
    w : np.ndarray
        Weights to apply.
    opt : int
        Option for handling zeros (0 to use raw values, and 1 to set matching 0s
        to NaNs (to avoid getting 0 when obs 0)).

    Returns
    -------
    ave : float
        Weighted area average of the spatial field.
    """

    sumt = 0.0
    sumw = 0.0

    xdims = x.shape
    nlat = xdims[0]
    nlon = xdims[1]

    if np.isscalar(w):
        WGT = np.empty((nlat, nlon))
        WGT[:] = w
    else:
        if w.ndim == 1:
            w = np.expand_dims(w, axis=1)
            WGT = np.repeat(w, nlon, axis=1)
        elif w.ndim == 2:
            WGT = w
        else:
            quit()

    # Set weight to 0 where either x or y is nan/missing
    WGT = np.where(np.isnan(x), 0.0, WGT)

    if opt == 1:
        xtmp = np.where(x == 0, float("NaN"), x)
    else:
        xtmp = x

    for ii in range(nlat):
        for jj in range(nlon):
            if ~np.isnan(xtmp[ii, jj]):
                sumt = sumt + WGT[ii, jj] * xtmp[ii, jj]
                sumw = sumw + WGT[ii, jj]

    if sumw != 0.0:
        ave = sumt / sumw

    return ave


def taylor_stats(x, y, w, opt):
    """
    Compute Taylor diagram statistics between a test and a reference variables.

    Parameters
    ----------
    x : np.ndarray
        Test variable (lat x lon).
    y : np.ndarray
        Reference variable (lat x lon), truth or control.
    w : np.ndarray
        Weights to apply.
    opt : int
        Currently unused.

    Returns
    -------
    pc : float
        Pearson correlation between the variables.
    ratio : float
        Ratio of standard deviations (test/reference).
    bias : float
        Relative bias with respect to the reference.
    xmean : float
        Weighted mean of `x`.
    ymean : float
        Weighted mean of `y`.
    xvar : float
        Weighted variance of `x`.
    yvar : float
        Weighted variance of `y`.
    rmse : float
        Normalized RMSE between the variables.
    """

    if x.shape != y.shape:
        print("Shapes of x and y do not match!")
        quit()

    # Figure out rank of x and y arrs
    if np.isscalar(x):
        xrank = 0
    else:
        xrank = x.ndim

    xdims = x.shape
    nlat = xdims[0]
    nlon = xdims[1]

    if np.isscalar(w):
        WGT = np.empty((nlat, nlon))
        WGT[:] = w
    else:
        if w.ndim == 1:
            w = np.expand_dims(w, axis=1)
            WGT = np.repeat(w, nlon, axis=1)
        elif w.ndim == 2:
            WGT = w
        else:
            quit()

    # Set weight to 0 where either x or y is nan/missing
    WGT = np.where(np.isnan(x), 0.0, WGT)
    WGT = np.where(np.isnan(y), 0.0, WGT)

    # Calculate pattern correlation
    pc = pattern_cor(x, y, WGT, 0)

    # Calculate averages, variance, and RMSE
    if xrank == 1:
        print("xrank 1 not supported currently")
        quit()
    else:
        xmean = wgt_areaave2(x, WGT, 0)
        ymean = wgt_areaave2(y, WGT, 0)
        # TODO, allow WGT to be 4D conforming
        wsum = np.sum(WGT)
        xdiff = x - xmean
        ydiff = y - ymean
        xvar = np.nansum(WGT * xdiff**2) / wsum
        yvar = np.nansum(WGT * ydiff**2) / wsum
        rmse = wgt_arearmse2(x, y, WGT, 0)

    # Calculate bias
    bias = xmean - ymean
    if ymean != 0.0:
        bias = (bias / ymean) * 100
    else:
        bias = float("NaN")

    # Calculate ratio and update RMSE
    if yvar != 0:
        ratio = np.sqrt(xvar / yvar)
        rmse = rmse / np.sqrt(yvar)
    else:
        ratio = float("NaN")

    return pc, ratio, bias, xmean, ymean, xvar, yvar, rmse



# Functions adapted from cymep/functions/track_density.py

def track_density(gridsize, lonstart, clat, clon, setzeros):
    """
    Compute track density (number of occurrences) on a latitude-longitude grid.

    Parameters
    ----------
    gridsize : float
        Grid spacing (in degrees) for both latitude and longitude.
    lonstart : float
        Starting longitude (degrees), which defines the western boundary of the grid.
    clat : np.ndarray
        Latitudes of the track points.
    clon : np.ndarray
        Longitudes of the track points.
    setzeros : bool
        Whether to set grid cells with zero counts to NaN (if True) or retain them (if False).

    Returns
    -------
    countarr : np.ndarray
        Number of track points per grid cell (lat x lon).
    lat : np.ndarray
        Latitude coordinates of the grid centers.
    lon : np.ndarray
        Longitude coordinates of the grid centers.
    """

    # Error checking
    if clat.size != clon.size:
        print("ERROR in track_density")
        print("clat size is " + str(clat.size) + " but clon size is " + str(clon.size))
        quit()

    # Create grid
    latS = -90.0
    latN = 90.0
    lonW = lonstart
    lonE = lonstart + 360.0

    dlat = gridsize
    dlon = gridsize

    nlat = int((latN - latS) / dlat) + 1
    mlon = int((lonE - lonW) / dlon)

    lat = np.linspace(latS, latN, num=nlat)
    lon = np.linspace(lonW, lonE - dlon, num=mlon)

    countarr = np.empty((nlat, mlon))
    # countarr[:] = np.nan

    # Count data
    countarr[:] = 0
    jl = 0
    il = 0

    npts = clat.size

    for nn, zz in enumerate(range(npts)):
        if ~np.isnan(clon[nn]):
            jl = int((clat[nn] - latS) / dlat)
            il = int((clon[nn] - lonW) / dlon)
            if il > (mlon - 1):
                print("mlon needs correcting at: " + str(il))
                il = 0
            countarr[jl, il] = countarr[jl, il] + 1

    print(
        "count: min=" + str(int(np.nanmin(countarr))) + "   max=" + str(int(np.nanmax(countarr)))
    )
    print("count: sum=" + str(int(np.nansum(countarr))))

    if setzeros:
        countarr = np.where(countarr == 0, float("NaN"), countarr)

    return countarr, lat, lon


def track_mean(gridsize, lonstart, clat, clon, cvar, meanornot, minhits):
    """
    Compute gridded mean or cumulative values of a variable along tracks.

    Parameters
    ----------
    gridsize : float
        Grid spacing (in degrees) for both latitude and longitude.
    lonstart : float
        Starting longitude (degrees), which defines the western boundary of the grid.
    clat : np.ndarray
        Latitudes of the track points.
    clon : np.ndarray
        Longitudes of the track points.
    cvar : np.ndarray
        Variable values associated with each track point (e.g., intensity, pressure).
    meanornot : bool
        Whether to compute the mean (if True) or just the cumulative sums (if False)
        of `cvar` per grid cell.
    minhits : int
        Minimum number of hits per grid cell required to include the result (cells with
        fewer hits are set to NaN).

    Returns
    -------
    cumulative : np.ndarray
        Mean or cumulative sum of `cvar` values per grid cell (lat x lon).
    lat : np.ndarray
        Latitude coordinates of the grid centers.
    lon : np.ndarray
        Longitude coordinates of the grid centers.
    """

    # Create grid
    latS = -90.0
    latN = 90.0
    lonW = lonstart
    lonE = lonstart + 360.0

    dlat = gridsize
    dlon = gridsize

    nlat = int((latN - latS) / dlat) + 1
    mlon = int((lonE - lonW) / dlon)

    lat = np.linspace(latS, latN, num=nlat)
    lon = np.linspace(lonW, lonE - dlon, num=mlon)

    countarr = np.empty((nlat, mlon))
    cumulative = np.empty((nlat, mlon))

    # Count data
    countarr[:] = 0
    cumulative[:] = 0
    jl = 0
    il = 0

    npts = clat.size

    for nn, zz in enumerate(range(npts)):
        if ~np.isnan(clon[nn]):
            jl = int((clat[nn] - latS) / dlat)
            il = int((clon[nn] - lonW) / dlon)
            if il > (mlon - 1):
                print("mlon needs correcting at: " + str(il))
                il = 0
            countarr[jl, il] = countarr[jl, il] + 1
            cumulative[jl, il] = cumulative[jl, il] + cvar[nn]

    # set to nan if cumulative less than the specified number of min hits
    cumulative = np.where(countarr < minhits, float("NaN"), cumulative)

    if meanornot:
        # Normalize by dividing by count
        countarr = np.where(countarr == 0, float("NaN"), countarr)
        cumulative = cumulative / countarr

    # print("count: min="+str(int(np.nanmin(countarr)))+"   max="+str(int(np.nanmax(countarr))))
    # print("count: sum="+str(int(np.nansum(countarr))))
    print("cumulative: min=" + str(np.nanmin(cumulative)) + "   max=" + str(np.nanmax(cumulative)))
    print("cumulative: sum=" + str(np.nansum(cumulative)))

    return cumulative, lat, lon


def track_minmax(gridsize, lonstart, clat, clon, cvar, minmax, minhits):
    """
    Compute gridded minimum or maximum values of a variable along tracks.

    Parameters
    ----------
    gridsize : float
        Grid spacing (in degrees) for both latitude and longitude.
    lonstart : float
        Starting longitude (degrees), which defines the western boundary of the grid.
    clat : np.ndarray
        Latitudes of the track points.
    clon : np.ndarray
        Longitudes of the track points.
    cvar : np.ndarray
        Variable values associated with each track point (e.g., intensity, pressure).
    minmax : {"min", "max"}
        Whether to compute minimum or maximum values per grid cell.
    minhits : int
        Minimum number of hits per grid cell required to include the result (cells with
        fewer hits are set to NaN).

    Returns
    -------
    countarr : np.ndarray
        Minimum or maximum `cvar` values per grid cell (lat x lon).
    lat : np.ndarray
        Latitude coordinates of the grid centers.
    lon : np.ndarray
        Longitude coordinates of the grid centers.
    """

    # Create grid
    latS = -90.0
    latN = 90.0
    lonW = lonstart
    lonE = lonstart + 360.0

    dlat = gridsize
    dlon = gridsize

    nlat = int((latN - latS) / dlat) + 1
    mlon = int((lonE - lonW) / dlon)

    lat = np.linspace(latS, latN, num=nlat)
    lon = np.linspace(lonW, lonE - dlon, num=mlon)

    countarr = np.empty((nlat, mlon))

    # Count data
    countarr[:] = np.nan
    jl = 0
    il = 0

    npts = clat.size

    for nn, zz in enumerate(range(npts)):
        if ~np.isnan(clon[nn]):
            jl = int((clat[nn] - latS) / dlat)
            il = int((clon[nn] - lonW) / dlon)
            if il > (mlon - 1):
                print("mlon needs correcting at: " + str(il))
                il = 0

            if ~np.isnan(cvar[nn]):
                if np.isnan(countarr[jl, il]):
                    countarr[jl, il] = cvar[nn]
                else:
                    if cvar[nn] > countarr[jl, il] and minmax == "max":
                        countarr[jl, il] = cvar[nn]
                    elif cvar[nn] < countarr[jl, il] and minmax == "min":
                        countarr[jl, il] = cvar[nn]
                    else:
                        # This means we have a valid cvar but a countarr value exists that is more extreme
                        pass

    print("count: min=" + str(np.nanmin(countarr)) + "   max=" + str(np.nanmax(countarr)))

    return countarr, lat, lon



# Functions adapted from cymep/functions/write_spatial.py

def write_spatial_netcdf(spatialdict, permondict, peryrdict, taydict, modelsin, nyears,
                         nmonths, latin, lonin, globaldict):
    """
    Write spatial, temporal, and summary climate metrics to a NetCDF file.

    Parameters
    ----------
    spatialdict : dict[np.ndarray]
        Dictionary of 3D arrays (model x lat x lon) with spatial metrics.
    permondict : dict[np.ndarray]
        Dictionary of 2D arrays (model x months) with monthly metrics.
    peryrdict : dict[np.ndarray]
        Dictionary of 2D arrays (model x years) with yearly metrics.
    taydict : dict[np.ndarray]
        Dictionary of 1D arrays (model) with metrics for Taylor diagrams.
    modelsin : list[str]
        Model names.
    nyears : int
        Number of years in the dataset.
    nmonths : int
        Number of months in the dataset.
    latin : np.ndarray
        Latitude values.
    lonin : np.ndarray
        Longitude values.
    globaldict : dict
        Global metadata (`strbasin`, `csvfilename`, ...).

    Returns
    -------
    netcdf_path : str
        Path to the created NetCDF file.
    """

    # Convert modelsin from pandas to list
    modelsin = modelsin.tolist()

    # Set up dimensions
    nmodels = len(modelsin)
    nlats = latin.size
    nlons = lonin.size
    nchar = 16

    netcdfdir = "./netcdf-files/"
    os.makedirs(os.path.dirname(netcdfdir), exist_ok=True)
    netcdfile = (
        netcdfdir
        + "/netcdf_"
        + globaldict["strbasin"]
        + "_"
        + os.path.splitext(globaldict["csvfilename"])[0]
    )

    # Open a netCDF file to write
    netcdf_path = netcdfile + ".nc"
    ncout = nc.Dataset(netcdf_path, "w", format="NETCDF4")

    # Dfine axis size
    ncout.createDimension("model", nmodels)  # unlimited
    ncout.createDimension("lat", nlats)
    ncout.createDimension("lon", nlons)
    ncout.createDimension("characters", nchar)
    ncout.createDimension("months", nmonths)
    ncout.createDimension("years", nyears)

    # Create latitude axis
    lat = ncout.createVariable("lat", "f", ("lat"))
    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"
    lat.axis = "Y"

    # Create longitude axis
    lon = ncout.createVariable("lon", "f", ("lon"))
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"
    lon.axis = "X"

    # Write lon + lat
    lon[:] = lonin[:]
    lat[:] = latin[:]

    # Create variable arrays
    for ii in spatialdict:
        vout = ncout.createVariable(ii, "f", ("model", "lat", "lon"), fill_value=1e20)
        # vout.long_name = 'density'
        # vout.units = '1/year'
        vout[:] = np.ma.masked_invalid(spatialdict[ii][:, :, :])

    for ii in permondict:
        vout = ncout.createVariable(ii, "f", ("model", "months"), fill_value=1e20)
        vout[:] = np.ma.masked_invalid(permondict[ii][:, :])

    for ii in peryrdict:
        vout = ncout.createVariable(ii, "f", ("model", "years"), fill_value=1e20)
        vout[:] = np.ma.masked_invalid(peryrdict[ii][:, :])

    for ii in taydict:
        vout = ncout.createVariable(ii, "f", ("model"), fill_value=1e20)
        vout[:] = np.ma.masked_invalid(taydict[ii][:])

    # Write model names to char
    model_names = ncout.createVariable("model_names", "c", ("model", "characters"))
    model_names[:] = nc.stringtochar(np.array(modelsin).astype("S16"))

    # today = datetime.today()
    ncout.description = "Tropical Cyclones metrics processed data"
    ncout.history = "Created " + datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    for ii in globaldict:
        ncout.setncattr(ii, str(globaldict[ii]))

    # close files
    ncout.close()

    return netcdf_path


def write_cymep_output_pyhanami(per_month_dict, per_year_dict, clim_mean_dict, storm_mean_dict,
                                temp_scorr_dict, spatial_dict, spatial_pcorr_dict, model_names,
                                nyears, nmonths, lat_idxs, lon_idxs, attrs_dict, descript_dict):
    """
    Write spatial, temporal, and summary climate metrics to xarray.Dataset.

    Parameters
    ----------
    per_month_dict : dict[np.ndarray]
        Dictionary of 2D arrays (model x months) with monthly metrics.
    per_year_dict : dict[np.ndarray]
        Dictionary of 2D arrays (model x years) with yearly metrics.
    clim_mean_dict : dict[np.ndarray]
        Dictionary of 1D arrays (model) with climatological mean metrics.
    storm_mean_dict : dict[np.ndarray]
        Dictionary of 1D arrays (model) with mean storm metrics.
    temp_scorr_dict : dict[np.ndarray]
        Dictionary of 1D arrays (model) with temporal Spearman rank correlation metrics.
    spatial_dict : dict[np.ndarray]
        Dictionary of 3D arrays (model x lat x lon) with spatial metrics.
    spatial_pcorr_dict : dict[np.ndarray]
        Dictionary of 3D arrays (model x lat x lon) with spatial Pearson correlation metrics.
    model_names : list[str]
        Model names.
    nyears : int
        Number of years in the dataset.
    nmonths : int
        Number of months in the dataset.
    lat_idxs : np.ndarray
        Latitude values.
    lon_idxs : np.ndarray
        Longitude values.
    attrs_dict : dict
        Global metadata (`strbasin`, `csvfilename`, ...).
    descript_dict : dict
        Descriptions for each metric.

    Returns
    -------
    data_cymep : xr.Dataset
        Dataset containing all the metrics.
    """

    # Create coordinates dictionary
    coords = {
        "model": model_names.tolist(),
        "lat": (
            "lat",
            lat_idxs,
            {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
                "axis": "Y",
            },
        ),
        "lon": (
            "lon",
            lon_idxs,
            {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
                "axis": "X",
            },
        ),
        "month": np.arange(nmonths),
        "year": np.arange(nyears),
    }


    # Initialize data variables dictionary
    data_vars = {}

    # Add temporal variables (2D: model x months/years)
    for name, data in per_month_dict.items():
        data_vars[name] = xr.DataArray(
            data,
            dims=("model", "month"),
            coords={"model": coords["model"], "month": coords["month"]},
            attrs=descript_dict[name],
        )

    for name, data in per_year_dict.items():
        data_vars[name] = xr.DataArray(
            data,
            dims=("model", "year"),
            coords={"model": coords["model"], "year": coords["year"]},
            attrs=descript_dict[name],
        )

    # Add mean temporal variables (1D: model)
    for name, data in clim_mean_dict.items():
        data_vars[name] = xr.DataArray(
            data,
            dims=("model",),
            coords={"model": coords["model"]},
            attrs=descript_dict[name]
        )

    for name, data in storm_mean_dict.items():
        data_vars[name] = xr.DataArray(
            data, dims=("model",),
            coords={"model": coords["model"]},
            attrs=descript_dict[name]
        )

    # Add temporal correlation variables (1D: model)
    for name, data in temp_scorr_dict.items():
        data_vars[name] = xr.DataArray(
            data, dims=("model",),
            coords={"model": coords["model"]},
            attrs=descript_dict[name]
        )

    # Add spatial variables (3D: model x lat x lon)
    for name, data in spatial_dict.items():
        data_vars[name] = xr.DataArray(
            data,
            dims=("model", "lat", "lon"),
            coords={"model": coords["model"], "lat": coords["lat"], "lon": coords["lon"]},
            attrs=descript_dict[name],
        )

    # Add spatial correlation variables (1D: model)
    for name, data in spatial_pcorr_dict.items():
        data_vars[name] = xr.DataArray(
            data, dims=("model",),
            coords={"model": coords["model"]},
            attrs=descript_dict[name]
        )


    # Create xarray Dataset with all variables
    data_cymep = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "description": "Tropical Cyclones metrics processed data.",
            "history": "Created " + datetime.today().strftime("%Y-%m-%d-%H:%M:%S"),
            **{key: str(value) for key, value in attrs_dict.items()},
        },
    )

    return data_cymep


def write_dict_csv(vardict, modelsin):
    """
    Write dictionary of variables to individual csv files (one per variable).

    Parameters
    ----------
    vardict : dict
        Variable names and with the correpsonding values.
    modelsin : list[str]
        Model names.
    """

    # Create variable array
    csvdir = "./csv-files/"
    os.makedirs(os.path.dirname(csvdir), exist_ok=True)
    for ii in vardict:
        csvfilename = csvdir + "/" + str(ii) + ".csv"
        if vardict[ii].shape == modelsin.shape:
            tmp = np.concatenate(
                (np.expand_dims(modelsin, axis=1), np.expand_dims(vardict[ii], axis=1)), axis=1
            )
        else:
            tmp = np.concatenate((np.expand_dims(modelsin, axis=1), vardict[ii]), axis=1)
        np.savetxt(csvfilename, tmp, delimiter=",", fmt="%s")

    return


def write_single_csv(vardict, modelsin, csvdir, csvname):
    """
    Write dictionary of variables to a single csv file.

    Parameters
    ----------
    vardict : dict
        Variable names and with the correpsonding values.
    modelsin : (str or list[str])
        Model names. If scalar, function writes a single-row CSV.
        If array, must align with variable arrays in `vardict`.
    csvdir : str
        Path to directory where csv file will be written.
    csvname : str
        Name of the output csv file.
    """

    # Create variable array
    os.makedirs(os.path.dirname(csvdir), exist_ok=True)
    csvfilename = csvdir + "/" + csvname

    # If a single line csv with one model
    if np.isscalar(modelsin):
        tmp = np.empty((1, len(vardict)))
        headerstr = "Model"
        iterix = 0
        for ii in vardict:
            headerstr = headerstr + "," + ii
            tmp[0, iterix] = vardict[ii]
            iterix += 1

        # Create a dummy numpy string array of "labels" with the control name to append as column #1
        labels = np.empty((1, 1), dtype="<U10")
        labels[:] = modelsin
        # Stack labels and numpy dict arrays horizontally as non-header data
        tmp = np.hstack((labels, tmp))

    # Else, the more common outcome; 2-D arrays
    else:
        # Concat models to first axis
        firstdict = list(vardict.keys())[0]
        headerstr = "Model," + firstdict

        if vardict[firstdict].shape == modelsin.shape:
            tmp = np.concatenate(
                (np.expand_dims(modelsin, axis=1), np.expand_dims(vardict[firstdict], axis=1)),
                axis=1,
            )
        else:
            tmp = np.concatenate((np.expand_dims(modelsin, axis=1), vardict[firstdict]), axis=1)

        for ii in vardict:
            if ii != firstdict:
                tmp = np.concatenate((tmp, np.expand_dims(vardict[ii], axis=1)), axis=1)
                headerstr = headerstr + "," + ii

    # Write header + data array
    np.savetxt(csvfilename, tmp, delimiter=",", fmt="%s", header=headerstr, comments="")

    return
