"""
This script contains functions adapted from the CyMeP package
Original source: https://github.com/zarzycki/cymep
Original author: Colin Zarzycki
License: MIT License
Copyright (c) 2021 Colin Zarzycki
"""

import os
import re
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc

from pathlib import Path
from datetime import datetime


# cymep/functions/getTrajectories.py functions

def getTrajectories(filename, nVars, headerDelimStr, isUnstruc):
    """
    Read trajectory data from a TempestExtremes file format.

    Parameters:
    ----------
    filename (str): Path to input trajectory file.
    nVars (int): Number of variables per trajectory point (-1 for auto-detection).
    headerDelimStr (str): String that marks header lines (e.g. "start").
    isUnstruc (bool): If True, adds an extra column for unstructured grid data.

    Returns:
    ----------
    numtraj (int): Number of trajectories found.
    maxNumPts (int): Maximum length of any trajectory.
    ncols (int): Number of data columns per point.
    prodata (np.ndarray): Array of shape (nvars, ntraj, maxpts) containing trajectory data.
    """

    print("Getting trajectories from TempestExtremes file...")
    print("Running getTrajectories on '%s' with unstruc set to '%s'" % (filename, isUnstruc))
    print("nVars set to %d and headerDelimStr set to '%s'" % (nVars, headerDelimStr))

    # Using the newer with construct to close the file automatically.
    with open(filename) as f:
        data = f.readlines()

    # Find total number of trajectories and maximum length of trajectories
    numtraj=0
    numPts=[]

    for line in data:
        if headerDelimStr in line:
            # if header line, store number of points in given traj in numPts
            headArr = line.split()
            numtraj += 1
            numPts.append(int(headArr[1]))
        else:
            # if not a header line, and nVars = -1, find number of columns in data point
            if nVars < 0:
                nVars=len(line.split())

    maxNumPts = max(numPts) # Maximum length of ANY trajectory
    print("Found %d columns" % nVars)
    print("Found %d trajectories" % numtraj)

    # Initialize storm and line counter
    stormID=-1
    lineOfTraj=-1

    # Create array for data
    if isUnstruc:
        prodata = np.empty((nVars+1,numtraj,maxNumPts))
    else:
        prodata = np.empty((nVars,numtraj,maxNumPts))
    prodata[:] = np.nan

    for i, line in enumerate(data):
        if headerDelimStr in line:  # check if header string is satisfied
            stormID += 1      # increment storm
            lineOfTraj = 0    # reset trajectory line to zero
        else:
            ptArr = line.split()
            for jj in range(nVars):
                if isUnstruc:
                    prodata[jj+1,stormID,lineOfTraj]=ptArr[jj]
                else:
                    prodata[jj,stormID,lineOfTraj]=ptArr[jj]
            lineOfTraj += 1   # increment line

    # Make sure we return the correct size of the array
    if isUnstruc:
        ncols=nVars+1
    else:
        ncols=nVars

    print("... done reading data")
    return numtraj, maxNumPts, ncols, prodata


def writeTrajectories(filename, data, ntraj, npts, headerDelimStr="start"):
    """
    Write trajectory data to a file in TempestExtremes format.

    Parameters:
    ----------
    filename (str): Output file path.
    data (np.ndarray): Data array of shape (nvars, ntraj, maxpts).
    ntraj (int): Number of trajectories.
    npts (int): Max number of points.
    headerDelimStr (str): Header delimiter string (default: 'start').
    """

    print(f"Writing TempestExtremes file to {filename}...")
    with open(filename, 'w') as f:
        for storm in range(ntraj):
            # Find valid points for this storm
            valid_points = ~np.isnan(data[0,storm,:])
            num_valid = int(np.sum(valid_points))

            # Get time values from last 4 columns for header
            yyyy = int(data[-4,storm,0])
            mm = int(data[-3,storm,0])
            dd = int(data[-2,storm,0])
            hh = int(data[-1,storm,0])

            # Write header
            f.write(f"{headerDelimStr}\t{num_valid}\t{yyyy}\t{mm}\t{dd}\t{hh}\n")

            # Write data points
            for pt in range(num_valid):
                line_data = []
                # Process all columns except time columns
                for col in range(data.shape[0]-4):
                    val = data[col,storm,pt]
                    if col <= 1:  # First two columns are integers
                        line_data.append(f"{int(val)}")
                    else:
                        line_data.append(f"{val:e}")

                # Add time columns
                for timevar in [-4, -3, -2, -1]:
                    line_data.append(f"{int(data[timevar,storm,pt])}")

                f.write("\t" + "\t".join(line_data) + "\n")


def getNodes(filename, nVars, isUnstruc):
    """
    Read node data from a TempestExtremes nodes file format.

    Parameters:
    ----------
    filename (str): Path to input nodes file.
    nVars (int): Number of variables per node (-1 for auto-detection).
    isUnstruc (bool): If True, adds extra columns for unstructured grid data.

    Returns:
    ----------
    numnodetimes (int): Number of timesteps with nodes.
    maxNumPts (int): Maximum number of nodes at any timestep.
    prodata (np.ndarray): Array of shape (nvars+4/5, numnodetimes, maxpts) containing node data.
    """

    print("Getting nodes from TempestExtremes file...")

    # Using the newer with construct to close the file automatically.
    with open(filename) as f:
        data = f.readlines()

    # Find total number of trajectories and maximum length of trajectories
    numnodetimes=0
    numPts=[]

    for line in data:
        if re.match(r'\w', line):
            # if header line, store number of points in given traj in numPts
            headArr = line.split()
            numnodetimes += 1
            numPts.append(int(headArr[3]))
        else:
            # if not a header line, and nVars = -1, find number of columns in data point
            if nVars < 0:
                nVars=len(line.split())

    maxNumPts = max(numPts) # Maximum length of ANY trajectory
    print("Found %d columns" % nVars)
    print("Found %d trajectories" % numnodetimes)
    print("Found %d maxNumPts" % maxNumPts)

    # Initialize storm and line counter
    stormID=-1
    lineOfTraj=-1

    # Create array for data
    if isUnstruc:
        prodata = np.empty((nVars+5,numnodetimes,maxNumPts))
    else:
        prodata = np.empty((nVars+4,numnodetimes,maxNumPts))

    prodata[:] = np.nan
    nextHeadLine=0

    for i, line in enumerate(data):
        if re.match(r'\w', line):  # check if header string is satisfied
            stormID += 1      # increment storm
            lineOfTraj = 0    # reset trajectory line to zero
            headArr = line.split()
            YYYY = int(headArr[0])
            MM = int(headArr[1])
            DD = int(headArr[2])
            HH = int(headArr[4])
        else:
            ptArr = line.split()
            for jj in range(nVars-1):
                if isUnstruc:
                    prodata[jj+1,stormID,lineOfTraj]=ptArr[jj]
                else:
                    prodata[jj,stormID,lineOfTraj]=ptArr[jj]
            if isUnstruc:
                prodata[nVars+1,stormID,lineOfTraj]=YYYY
                prodata[nVars+2,stormID,lineOfTraj]=MM
                prodata[nVars+3,stormID,lineOfTraj]=DD
                prodata[nVars+4,stormID,lineOfTraj]=HH
            else:
                prodata[nVars  ,stormID,lineOfTraj]=YYYY
                prodata[nVars+1,stormID,lineOfTraj]=MM
                prodata[nVars+2,stormID,lineOfTraj]=DD
                prodata[nVars+3,stormID,lineOfTraj]=HH
            lineOfTraj += 1   # increment line

    print("... done reading data")
    return numnodetimes, maxNumPts, prodata



# cymep/functions/mask_tc.py functions

def maskTC(lat, lon, dohemi=False):
    """
    Classify a tropical cyclone (TC) into a basin or hemisphere based on latitude and longitude.

    Parameters
    ----------
    lat (float): Latitude of the TC (degrees).
    lon (float): Longitude of the TC (degrees).
    dohemi (bool): Whether to only classify by hemisphere (if True) or basin-by-basin using 
        latitude/longitude boundaries (if False) (default: False).

    Returns
    -------
    basin (int): Integer code for the basin/hemisphere:
        - 0 → Unclassified / outside defined basins
        - 1 → North Atlantic (NATL)
        - 2 → Eastern Pacific (EPAC)
        - 3 → Central Pacific (CPAC)
        - 4 → Western Pacific (WPAC)
        - 5 → North Indian Ocean (NIO)
        - 6 → South Indian Ocean (SIO)
        - 7 → South Pacific (SPAC)
        - 20 → Northern Hemisphere (if `dohemi=True`)
        - 21 → Southern Hemisphere (if `dohemi=True`)
    """
   
    # If lon is negative, switch to [0, 360] convention
    if lon < 0.0:
        lon = lon + 360.

    if dohemi == True:
        if lat >= 0.:
            basin = 20
        else:
            basin = 21
    else:
        # Coefficients for calculating ATL/EPAC sloped line
        m = -0.58
        b = 0. -m*295.
        maxlat = 50.0

        if lat >= 0. and lat <= maxlat and lon > 257. and lon <= 359.:
            funcval = m*lon + b
            if lat > funcval:
                basin = 1
            else:
                basin = 2
        elif lat >= 0. and lat <= maxlat  and lon > 220. and lon <= 257.:
            basin = 2
        elif lat >= 0. and lat <= maxlat  and lon > 180. and lon <= 220.:
            basin = 3
        elif lat >= 0. and lat <= maxlat  and lon > 100. and lon <= 180.:
            basin = 4
        elif lat >= 0. and lat <= maxlat  and lon > 30.  and lon <= 100.:
            basin = 5
        elif lat  < 0. and lat >= -maxlat and lon > 30.  and lon <= 135.:
            basin = 6
        elif lat  < 0. and lat >= -maxlat and lon > 135. and lon <= 290.:
            basin = 7
        else:
            basin = 0

    return basin
  

def getbasinmaskstr(gridchoice):
    """
    Map a basin or hemisphere integer code (or list) to a string label.

    Parameters
    ----------
    gridchoice (int or list[int]): Basin/hemisphere code(s). If a sequence is provided, 
        only the first element is considered. Codes are:
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
            - <0 → GLOB (Global domain)
            - otherwise → NONE (unrecognized)

    Returns
    -------
    strbasin (str): String label corresponding to the input basin/hemisphere code.
    """

    if hasattr(gridchoice, "__len__"):
        if gridchoice[0] == 1:
            strbasin="NHEMI"
        else:
            strbasin="SHEMI"
    else:
        if gridchoice < 0:
            strbasin="GLOB"
        else:
            if gridchoice == 1:
                strbasin="NATL"
            elif gridchoice == 2:
                strbasin="EPAC"
            elif gridchoice == 3:
                strbasin="CPAC"
            elif gridchoice == 4:
                strbasin="WPAC"
            elif gridchoice == 5:
                strbasin="NIO"
            elif gridchoice == 6:
                strbasin="SIO"
            elif gridchoice == 7:
                strbasin="SPAC"
            elif gridchoice == 8:
                strbasin="SATL"
            elif gridchoice == 9:
                strbasin="FLA"
            elif gridchoice == 20:
                strbasin="NHEMI"
            elif gridchoice == 21:
                strbasin="SHEMI"
            else:
                strbasin="NONE"

    return strbasin



# cymep/functions/pattern_corr.py functions

def pattern_cor(x, y, w, opt):
    """
    Compute the weighted pattern correlation between two spatial fields.

    Parameters
    ----------
    x (np.ndarray): First spatial field (lat x lon).
    y (np.ndarray): Second spatial field (lat x lon).
    w (np.ndarray): Weights to apply.
    opt (int): Correlation option (0 for anomaly correlation (mean removed before correlation),
        otherwise, direct correlation without mean removal).

    Returns
    -------
    r (float): Weighted correlation coefficient between the spatial fields.
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
            w=np.expand_dims(w, axis=1)
            WGT=np.repeat(w,nlon,axis=1)
        elif w.ndim == 2:
            WGT=w
        else:
            quit()
    
    # Set weight to 0 where either x or y is nan/missing
    WGT   = np.where(np.isnan(x),0.,WGT)
    WGT   = np.where(np.isnan(y),0.,WGT)
    
    if opt == 0:
        sumWGT   = np.nansum(WGT)
        xAvgArea = np.nansum(x*WGT)/sumWGT
        yAvgArea = np.nansum(y*WGT)/sumWGT

        xAnom    = x - xAvgArea
        yAnom    = y - yAvgArea

        xyCov    = np.nansum(WGT*xAnom*yAnom)
        xAnom2   = np.nansum(WGT*xAnom**2)
        yAnom2   = np.nansum(WGT*yAnom**2)
    else:
        xyCov    = np.nansum(WGT*x*y)
        xAnom2   = np.nansum(WGT*x**2)
        yAnom2   = np.nansum(WGT*y**2)  
    
    # Calculate coefficient
    r   = xyCov/(np.sqrt(xAnom2)*np.sqrt(yAnom2))

    return r
  

def wgt_arearmse2(x, y, w, opt):
    """
    Compute the weighted area Root Mean Square Error (RMSE) between two spatial fields.

    Parameters
    ----------
    x (np.ndarray): First spatial field (lat x lon).
    y (np.ndarray): Second spatial field (lat x lon).
    w (np.ndarray): Weights to apply.
    opt (int): Option for handling zeros (0 to use raw values, and 1 to set matching 0s 
        to NaNs (to avoid getting 0 when obs 0)).

    Returns
    -------
    rmse (float): Weighted area RMSE between the spatial fields.
    """

    if x.shape != y.shape:
        print("Shapes of x and y do not match!")
        quit()
        
    sumd = 0.
    sumw = 0.
    
    xdims = x.shape
    nlat = xdims[0]
    nlon = xdims[1]
    
    if np.isscalar(w):
        WGT = np.empty((nlat, nlon))
        WGT[:] = w
    else:
        if w.ndim == 1:
            w=np.expand_dims(w, axis=1)
            WGT=np.repeat(w,nlon,axis=1)
        elif w.ndim == 2:
            WGT=w
        else:
            quit()
    
    # Set weight to 0 where either x or y is nan/missing
    WGT   = np.where(np.isnan(x),0.,WGT)
    WGT   = np.where(np.isnan(y),0.,WGT)
    
    if opt == 1:
        xtmp = np.where(np.logical_and(x==0,y==0),float('NaN'),x)
        ytmp = np.where(np.logical_and(x==0,y==0),float('NaN'),y)
    else:
        xtmp = x
        ytmp = y

    for ii in range(nlat):
        for jj in range(nlon):
            if ~np.isnan(xtmp[ii,jj]) and ~np.isnan(ytmp[ii,jj]):
                sumd = sumd + WGT[ii,jj] * ( xtmp[ii,jj] - ytmp[ii,jj] )**2
                sumw = sumw + WGT[ii,jj]
                #SUMD = SUMD + WGT(ML,NL)* (T(ML,NL)-Q(ML,NL))**2
                #SUMW = SUMW + WGT(ML,NL)
        
    if sumw != 0.0:
        rmse = np.sqrt(sumd/sumw)

    return rmse
  

def wgt_areaave2(x, w, opt):
    """
    Compute the weighted area average of a spatial field.

    Parameters
    ----------
    x (np.ndarray): Input spatial field (lat x lon).
    w (np.ndarray): Weights to apply.
    opt (int): Option for handling zeros (0 to use raw values, and 1 to set matching 0s 
        to NaNs (to avoid getting 0 when obs 0)).

    Returns
    -------
    ave (float): Weighted area average of the spatial field.
    """
    
    sumt = 0.
    sumw = 0.
    
    xdims = x.shape
    nlat = xdims[0]
    nlon = xdims[1]
    
    if np.isscalar(w):
        WGT = np.empty((nlat, nlon))
        WGT[:] = w
    else:
        if w.ndim == 1:
            w=np.expand_dims(w, axis=1)
            WGT=np.repeat(w,nlon,axis=1)
        elif w.ndim == 2:
            WGT=w
        else:
            quit()
    
    # Set weight to 0 where either x or y is nan/missing
    WGT = np.where(np.isnan(x),0.,WGT)
    
    if opt == 1:
        xtmp = np.where(x==0,float('NaN'),x)
    else:
        xtmp = x

    for ii in range(nlat):
        for jj in range(nlon):
            if ~np.isnan(xtmp[ii,jj]):
                sumt = sumt + WGT[ii,jj] * xtmp[ii,jj] 
                sumw = sumw + WGT[ii,jj]
        
    if sumw != 0.0:
        ave = sumt/sumw

    return ave
  

def taylor_stats(x, y, w, opt):
    """
    Compute Taylor diagram statistics between a test and a reference variables.

    Parameters
    ----------
    x (np.ndarray): Test variable (lat x lon).
    y (np.ndarray): Reference variable (lat x lon), truth or control.
    w (np.ndarray): Weights to apply.
    opt (int): Currently unused.

    Returns
    -------
    pc (float): Pattern correlation between the variables.
    ratio (float): Ratio of standard deviations (test/reference).
    bias (float): Relative bias with respect to the reference.
    xmean (float): Weighted mean of `x`.
    ymean (float): Weighted mean of `y`.
    xvar (float): Weighted variance of `x`.
    yvar (float): Weighted variance of `y`.
    rmse (float): Normalized RMSE between the variables.
    """

    if x.shape != y.shape:
        print("Shapes of x and y do not match!")
        quit()
        
    # Figure out rank of x and y arrs
    if np.isscalar(x):
        xrank=0
    else:
        xrank=x.ndim

    xdims = x.shape
    nlat = xdims[0]
    nlon = xdims[1]
    
    if np.isscalar(w):
        WGT = np.empty((nlat, nlon))
        WGT[:] = w
    else:
        if w.ndim == 1:
            w=np.expand_dims(w, axis=1)
            WGT=np.repeat(w,nlon,axis=1)
        elif w.ndim == 2:
            WGT=w
        else:
            quit()
  
    # Set weight to 0 where either x or y is nan/missing
    WGT   = np.where(np.isnan(x),0.,WGT)
    WGT   = np.where(np.isnan(y),0.,WGT)
    
    # Calculate pattern correlation
    pc = pattern_cor(x,y,WGT,0)
    
    # Calculate averages, variance, and RMSE
    if xrank == 1:
        print("xrank 1 not supported currently")
        quit()
    else:                                            
        xmean   = wgt_areaave2(x, WGT, 0)
        ymean   = wgt_areaave2(y, WGT, 0)
        # TODO, allow WGT to be 4D conforming
        wsum    = np.sum(WGT)
        xdiff   = x-xmean 
        ydiff   = y-ymean
        xvar    = np.nansum(WGT*xdiff**2)/wsum 
        yvar    = np.nansum(WGT*ydiff**2)/wsum             
        rmse    = wgt_arearmse2(x,y, WGT, 0)

    # Calculate bias
    bias = xmean-ymean
    if ymean != 0.:
        bias = (bias/ymean)*100
    else:
        bias = float('NaN')

    # Calculate ratio and update RMSE
    if yvar != 0:
        ratio  = np.sqrt(xvar/yvar)
        rmse = rmse/np.sqrt(yvar)
    else:
        ratio = float('NaN')

    return pc, ratio, bias, xmean, ymean, xvar, yvar, rmse



# cymep/functions/track_density.py functions

def track_density(gridsize, lonstart, clat, clon, setzeros):
    """
    Compute track density (number of occurrences) on a latitude-longitude grid.

    Parameters
    ----------
    gridsize (float): Grid spacing (in degrees) for both latitude and longitude.
    lonstart (float): Starting longitude (degrees), which defines the western boundary of the grid.
    clat (np.ndarray): Latitudes of the track points.
    clon (np.ndarray): Longitudes of the track points.
    setzeros (bool): Whether to set grid cells with zero counts to NaN (if True) or retain them (if False).

    Returns
    -------
    countarr (np.ndarray): Number of track points per grid cell (lat x lon).
    lat (np.ndarray): Latitude coordinates of the grid centers.
    lon (np.ndarray): Longitude coordinates of the grid centers.
    """

    # Error checking 
    if clat.size != clon.size:
        print("ERROR in track_density")
        print("clat size is "+str(clat.size)+" but clon size is "+str(clon.size))
        quit()

    # Create grid 
    latS = -90.
    latN = 90.
    lonW = lonstart
    lonE = lonstart + 360.

    dlat =  gridsize
    dlon =  gridsize

    nlat = int((latN-latS)/dlat) + 1
    mlon = int((lonE-lonW)/dlon)
    
    lat = np.linspace(latS, latN,      num=nlat)
    lon = np.linspace(lonW, lonE-dlon, num=mlon)

    countarr = np.empty((nlat, mlon))
    #countarr[:] = np.nan
    
    
    # Count data 
    countarr[:] = 0
    jl = 0
    il = 0
    
    npts = clat.size
    
    for nn, zz in enumerate(range(npts)):
        if ~np.isnan(clon[nn]):
            jl = int( (clat[nn]-latS) / dlat )
            il = int( (clon[nn]-lonW) / dlon )
            if il > (mlon-1):
                print("mlon needs correcting at: "+str(il))
                il = 0
            countarr[jl,il] = countarr[jl,il] + 1
    
    print("count: min="+str(int(np.nanmin(countarr)))+"   max="+str(int(np.nanmax(countarr))))
    print("count: sum="+str(int(np.nansum(countarr))))
    
    if setzeros:
        countarr = np.where(countarr == 0, float('NaN'), countarr)
    
    return countarr, lat, lon


def track_mean(gridsize, lonstart, clat, clon, cvar, meanornot, minhits):
    """
    Compute gridded mean or cumulative values of a variable along tracks.

    Parameters
    ----------
    gridsize (float): Grid spacing (in degrees) for both latitude and longitude.
    lonstart (float): Starting longitude (degrees), which defines the western boundary of the grid.
    clat (np.ndarray): Latitudes of the track points.
    clon (np.ndarray): Longitudes of the track points.
    cvar (np.ndarray): Variable values associated with each track point (e.g., intensity, pressure).
    meanornot (bool): Whether to compute the mean (if True) or just the cumulative sums (if False) 
        of `cvar` per grid cell.
    minhits (int): Minimum number of hits per grid cell required to include the result (cells with 
        fewer hits are set to NaN).

    Returns
    -------
    cumulative (np.ndarray): Mean or cumulative sum of `cvar` values per grid cell (lat x lon).
    lat (np.ndarray): Latitude coordinates of the grid centers.
    lon (np.ndarray): Longitude coordinates of the grid centers.
    """

    # Create grid 
    latS = -90.
    latN = 90.
    lonW = lonstart
    lonE = lonstart + 360.

    dlat =  gridsize
    dlon =  gridsize

    nlat = int((latN-latS)/dlat) + 1
    mlon = int((lonE-lonW)/dlon)
    
    lat = np.linspace(latS, latN,      num=nlat)
    lon = np.linspace(lonW, lonE-dlon, num=mlon)

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
            jl = int( (clat[nn]-latS) / dlat )
            il = int( (clon[nn]-lonW) / dlon )
            if il > (mlon-1):
                print("mlon needs correcting at: "+str(il))
                il = 0
            countarr[jl,il] = countarr[jl,il] + 1
            cumulative[jl,il] = cumulative[jl,il] + cvar[nn]

    # set to nan if cumulative less than the specified number of min hits
    cumulative = np.where(countarr < minhits, float('NaN'), cumulative)

    if meanornot:
        #Normalize by dividing by count
        countarr = np.where(countarr == 0, float('NaN'), countarr)
        cumulative = cumulative / countarr

    #print("count: min="+str(int(np.nanmin(countarr)))+"   max="+str(int(np.nanmax(countarr))))
    #print("count: sum="+str(int(np.nansum(countarr))))
    print("cumulative: min="+str(np.nanmin(cumulative))+"   max="+str(np.nanmax(cumulative)))
    print("cumulative: sum="+str(np.nansum(cumulative)))  

    return cumulative, lat, lon
  

def track_minmax(gridsize, lonstart, clat, clon, cvar, minmax, minhits):
    """
    Compute gridded minimum or maximum values of a variable along tracks.

    Parameters
    ----------
    gridsize (float): Grid spacing (in degrees) for both latitude and longitude.
    lonstart (float): Starting longitude (degrees), which defines the western boundary of the grid.
    clat (np.ndarray): Latitudes of the track points.
    clon (np.ndarray): Longitudes of the track points.
    cvar (np.ndarray): Variable values associated with each track point (e.g., intensity, pressure).
    minmax ({"min", "max"}): Whether to compute minimum or maximum values per grid cell.
    minhits (int): Placeholder for minimum number of hits per grid cell..

    Returns
    -------
    countarr (np.ndarray): Minimum or maximum `cvar` values per grid cell (lat x lon).
    lat (np.ndarray): Latitude coordinates of the grid centers.
    lon (np.ndarray): Longitude coordinates of the grid centers.
    """

    # Create grid 
    latS = -90.
    latN = 90.
    lonW = lonstart
    lonE = lonstart + 360.

    dlat =  gridsize
    dlon =  gridsize

    nlat = int((latN-latS)/dlat) + 1
    mlon = int((lonE-lonW)/dlon)
    
    lat = np.linspace(latS, latN,      num=nlat)
    lon = np.linspace(lonW, lonE-dlon, num=mlon)

    countarr = np.empty((nlat, mlon))
  
    # Count data 
    countarr[:] = np.nan
    jl = 0
    il = 0
    
    npts = clat.size
    
    for nn, zz in enumerate(range(npts)):
        if ~np.isnan(clon[nn]):
            jl = int( (clat[nn]-latS) / dlat )
            il = int( (clon[nn]-lonW) / dlon )
            if il > (mlon-1):
                print("mlon needs correcting at: "+str(il))
                il = 0

            if ~np.isnan(cvar[nn]):
                if np.isnan(countarr[jl,il]):
                    countarr[jl,il] = cvar[nn]
                else:
                    if cvar[nn] > countarr[jl,il] and minmax == "max":
                        countarr[jl,il] = cvar[nn]          
                    elif cvar[nn] < countarr[jl,il] and minmax == "min":
                        countarr[jl,il] = cvar[nn]                    
                    else:
                        # This means we have a valid cvar but a countarr value exists that is more extreme
                        pass
    
    print("count: min="+str(np.nanmin(countarr))+"   max="+str(np.nanmax(countarr)))

    return countarr, lat, lon
  
  

# cymep/functions/write_spatial.py functions

def write_spatial_netcdf(spatialdict, permondict, peryrdict, taydict, modelsin, nyears, nmonths, latin, lonin, globaldict):
    """
    Write spatial, temporal, and summary climate metrics to a NetCDF file.

    Parameters
    ----------
    spatialdict (dict[np.ndarray]): Dictionary of 3D arrays (model x lat x lon) with spatial variables.
    permondict (dict[np.ndarray]): Dictionary of 2D arrays (model x months) with monthly metrics.
    peryrdict (dict[np.ndarray]): Dictionary of 2D arrays (model x years) with yearly metrics.
    taydict (dict[np.ndarray]): Dictionary of 1D arrays (model) with summary metrics.
    modelsin (list[str]): Model names.
    nyears (int) Number of years in the dataset.
    nmonths (int): Number of months in the dataset.
    latin (np.ndarray): Latitude values.
    lonin (np.ndarray): Longitude values.
    globaldict (dict): lobal metadata (`strbasin`, `csvfilename`, ...).
    """

    # Convert modelsin from pandas to list
    modelsin=modelsin.tolist()
    
    # Set up dimensions
    nmodels=len(modelsin)
    nlats=latin.size
    nlons=lonin.size
    nchar=16
    
    netcdfdir="./netcdf-files/"
    os.makedirs(os.path.dirname(netcdfdir), exist_ok=True)
    netcdfile=netcdfdir+"/netcdf_"+globaldict['strbasin']+"_"+os.path.splitext(globaldict['csvfilename'])[0]
    
    # Open a netCDF file to write
    ncout = nc.Dataset(netcdfile+".nc", 'w', format='NETCDF4')

    # Dfine axis size
    ncout.createDimension('model', nmodels)  # unlimited
    ncout.createDimension('lat', nlats)
    ncout.createDimension('lon', nlons)
    ncout.createDimension('characters', nchar)
    ncout.createDimension('months', nmonths)
    ncout.createDimension('years', nyears)

    # Create latitude axis
    lat = ncout.createVariable('lat', 'f', ('lat'))
    lat.standard_name = 'latitude'
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'

    # Create longitude axis
    lon = ncout.createVariable('lon', 'f', ('lon'))
    lon.standard_name = 'longitude'
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'

    # Write lon + lat
    lon[:] = lonin[:]
    lat[:] = latin[:]

    # Create variable arrays
    for ii in spatialdict:
        vout = ncout.createVariable(ii, 'f', ('model', 'lat', 'lon'), fill_value=1e+20)
        # vout.long_name = 'density'
        # vout.units = '1/year'
        vout[:] = np.ma.masked_invalid(spatialdict[ii][:,:,:])

    for ii in permondict:
        vout = ncout.createVariable(ii, 'f', ('model', 'months'), fill_value=1e+20)
        vout[:] = np.ma.masked_invalid(permondict[ii][:,:])
        
    for ii in peryrdict:
        vout = ncout.createVariable(ii, 'f', ('model', 'years'), fill_value=1e+20)
        vout[:] = np.ma.masked_invalid(peryrdict[ii][:,:])
        
    for ii in taydict:
        vout = ncout.createVariable(ii, 'f', ('model'), fill_value=1e+20)
        vout[:] = np.ma.masked_invalid(taydict[ii][:])

    # Write model names to char
    model_names = ncout.createVariable('model_names', 'c', ('model', 'characters'))
    model_names[:] = nc.stringtochar(np.array(modelsin).astype('S16'))
    
    # today = datetime.today()
    ncout.description = "Coastal metrics processed data"
    ncout.history = "Created " + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    for ii in globaldict:
        ncout.setncattr(ii, str(globaldict[ii]))
    
    # close files
    ncout.close()

    return
    

def write_dict_csv(vardict, modelsin):
    """
    Write dictionary of variables to individual csv files (one per variable).

    Parameters
    ----------
    vardict (dict): Variable names and with the correpsonding values.
    modelsin (list[str]): Model names.
    """

    # Create variable array
    csvdir="./csv-files/"
    os.makedirs(os.path.dirname(csvdir), exist_ok=True)
    for ii in vardict:
        csvfilename = csvdir+"/"+str(ii)+".csv"
        if vardict[ii].shape == modelsin.shape:
            tmp = np.concatenate((np.expand_dims(modelsin, axis=1),np.expand_dims(vardict[ii], axis=1)), axis=1)
        else:
            tmp = np.concatenate((np.expand_dims(modelsin, axis=1), vardict[ii]), axis=1)
        np.savetxt(csvfilename, tmp, delimiter=",", fmt="%s")

    return


def write_single_csv(vardict, modelsin, csvdir, csvname):
    """
    Write dictionary of variables to a single csv file.

    Parameters
    ----------
    vardict (dict): Variable names and with the correpsonding values.
    modelsin (str or list[str]): Model names. If scalar, function writes a single-row CSV.
        If array, must align with variable arrays in `vardict`.
    csvdir (str): Path to directory where csv file will be written.
    csvname (str): Name of the output csv file.
    """

    # Create variable array
    os.makedirs(os.path.dirname(csvdir), exist_ok=True)
    csvfilename = csvdir+"/"+csvname
    
    # If a single line csv with one model
    if np.isscalar(modelsin):
        tmp = np.empty((1,len(vardict)))
        headerstr="Model"
        iterix = 0
        for ii in vardict:
            headerstr=headerstr+","+ii
            tmp[0,iterix]=vardict[ii]
            iterix += 1
        
        # Create a dummy numpy string array of "labels" with the control name to append as column #1
        labels = np.empty((1,1),dtype="<U10")
        labels[:] = modelsin
        # Stack labels and numpy dict arrays horizontally as non-header data
        tmp = np.hstack((labels, tmp))
    
    # Else, the more common outcome; 2-D arrays
    else:
        # Concat models to first axis
        firstdict=list(vardict.keys())[0]
        headerstr="Model,"+firstdict
    
        if vardict[firstdict].shape == modelsin.shape:
            tmp = np.concatenate((np.expand_dims(modelsin, axis=1),np.expand_dims(vardict[firstdict], axis=1)), axis=1)
        else:
            tmp = np.concatenate((np.expand_dims(modelsin, axis=1), vardict[firstdict]), axis=1)
    
        for ii in vardict:
            if ii != firstdict:
                tmp = np.concatenate((tmp, np.expand_dims(vardict[ii], axis=1)), axis=1)
                headerstr=headerstr+","+ii
    
    # Write header + data array
    np.savetxt(csvfilename, tmp, delimiter=",", fmt="%s", header=headerstr, comments="")

    return


# cymep/conver-traj/ibtracs-to-tempest.ncl functions (translated to Python)
def load_ibtracs_data(ibfile, stix, enix, ibversion, ms_to_kts, flip_grid_180=True):
    """
    Extract and preprocess IBTrACS data. 

    Parameters
    ----------
    ibfile (xarray.Dataset): IBTrACS dataset.
    stix (int): Starting index of the storm to extract.
    enix (int): Ending index of the storm to extract.
    ibversion (str): Version of IBTrACS dataset.
    ms_to_kts (float): Conversion factor from m/s to knots.
    flip_grid_180 (bool): Whether to convert longitudes from [-180, 180] to [0, 360] (default: True).

    Returns
    -------
    iblat (xarray.DataArray): Latitudes of the storm track.
    iblon (xarray.DataArray): Longitudes of the storm track.
    ibtype (xarray.DataArray): Storm types along the track.
    ibwind (xarray.DataArray): Wind speeds along the track (in knots).
    ibpres (xarray.DataArray): Pressures along the track (in hPa).
    ibtime (xarray.DataArray): Time points along the track.
    ibname (xarray.DataArray): Storm names.
    ibbasin (xarray.DataArray): Basin codes of the storm.
    """

    if ibversion == "v3":
        iblat = ibfile.lat_wmo.isel(storm=slice(stix, enix+1)) * 0.01
        iblon = ibfile.lon_wmo.isel(storm=slice(stix, enix+1)) * 0.01
        ibtype = ibfile.nature_wmo.isel(storm=slice(stix, enix+1)).astype(int)
        ibwind = ibfile.wind_wmo.isel(storm=slice(stix, enix+1)) * 0.1 / ms_to_kts
        ibpres = ibfile.pres_wmo.isel(storm=slice(stix, enix+1))
        ibtime = ibfile.time_wmo.isel(storm=slice(stix, enix+1))
    else:
        iblat = ibfile.lat.isel(storm=slice(stix, enix+1))
        iblon = ibfile.lon.isel(storm=slice(stix, enix+1))
        ibtype = ibfile.nature.isel(storm=slice(stix, enix+1))
        ibwind = ibfile.wmo_wind.isel(storm=slice(stix, enix+1)) / ms_to_kts
        ibpres = ibfile.wmo_pres.isel(storm=slice(stix, enix+1)) * 100
        ibtime = ibfile.time.isel(storm=slice(stix, enix+1))
    
    ibname = ibfile.name.isel(storm=slice(stix, enix+1))
    ibbasin = ibfile.basin.isel(storm=slice(stix, enix+1)).astype(str)

    if flip_grid_180:
        iblon = xr.where(iblon < 0, iblon + 360, iblon)
    
    return iblat, iblon, ibtype, ibwind, ibpres, ibtime, ibname, ibbasin


def great_circle_distance(lat1, lon1, lat2, lon2, npts=2, iu=4):
    """
    Calculate great circle distance between points and interpolate points along the path
    following 'gc_latlon' function in NCL.
    
    Parameters
    ----------
    lat1 (float or np.ndarray): Latitude(s) of first point(s).
    lon1 (float or np.ndarray): Longitude(s) of first point(s).
    lat2 (float or np.ndarray): Latitude(s) of second point(s).
    lon2 (float or np.ndarray): Longitude(s) of second point(s).
    npts (int): Number of points to interpolate (default: 2).
    iu (int): Unit flag (default: 4)
        |iu| = 1: radians
        |iu| = 2: degrees
        |iu| = 3: meters
        |iu| = 4: kilometers
        sign(iu) > 0: longitudes in [0,360]
        sign(iu) < 0: longitudes in [-180,180]
    
    Returns
    -------
    distance (float or np.ndarray): Great circle distance in requested units.
    gclat (np.ndarray): Latitudes along great circle path
    gclon (np.ndarray): Longitudes along great circle path
    spacing (float): Distance between interpolated points
    """

    R = 6371.0  # Earth's radius in km
    
    # Convert to numpy arrays if needed
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)
    
    # Convert to radians
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    
    # Calculate great circle distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))


    # Convert to requested units
    if abs(iu) == 1:  # radians
        distance = c
    elif abs(iu) == 2:  # degrees
        distance = np.degrees(c)
    elif abs(iu) == 3:  # meters
        distance = R * c * 1000
    else:  # kilometers
        distance = R * c

    # Interpolate points if requested
    if npts > 2:
        f = np.linspace(0, 1, npts)
        
        # Calculate intermediate points
        A = np.sin((1-f)*c) / np.sin(c)
        B = np.sin(f*c) / np.sin(c)
        
        x = A[...,np.newaxis] * np.cos(lat1) * np.cos(lon1) + \
            B[...,np.newaxis] * np.cos(lat2) * np.cos(lon2)
        y = A[...,np.newaxis] * np.cos(lat1) * np.sin(lon1) + \
            B[...,np.newaxis] * np.cos(lat2) * np.sin(lon2)
        z = A[...,np.newaxis] * np.sin(lat1) + B[...,np.newaxis] * np.sin(lat2)
        
        gclat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        gclon = np.degrees(np.arctan2(y, x))
        
        # Adjust longitude range based on iu sign
        if iu > 0:  # [0,360]
            gclon = (gclon + 360) % 360
        else:  # [-180,180]
            gclon = ((gclon + 180) % 360) - 180
            
        spacing = distance / (npts - 1)
    else:
        gclat = np.array([lat1, lat2])
        gclon = np.array([lon1, lon2])
        spacing = distance
    
    return distance, gclat, gclon, spacing


def correct_pres_data(wind, pres):
    """
    Apply K&Z 07 relationship to fill missing pressure-wind data.
    
    Parameters
    ----------
    wind (float): Wind speed in m/s.
    pres (float): Pressure in Pa.
    
    Returns
    -------
    wind (float): Corrected wind speed in m/s.
    pres (float): Corrected pressure in Pa.
    """

    a, b, c = 2.3, 1010.0, 0.76
    if np.isnan(wind) and not np.isnan(pres):
        wind = a * (b - pres/100.0)**c
    elif not np.isnan(wind) and np.isnan(pres):
        pres = 100.0 * (b - (wind/a)**(1.0/c))
    elif np.isnan(wind) and np.isnan(pres):
        wind, pres = 15.0, 100800.0

    return wind, pres


def process_ibtracs(ibdir="../data", ibfilename="IBTrACS.since1980.v04r01.nc", ibversion="v4", ibst_yr=1980, iben_yr=datetime.now().year, 
                    gridfile="../data/topog.nc", is_grid_2d=False, flip_grid_180=True, cut_regional=False, cut_regional_ring_width=8,
                    correct_pres_wind=True, dur_thresh=3, print_to_screen=True, print_names=False):
    """
    Convert IBTrACS data to TempestExtremes format.

    Parameters
    ----------
    ibdir (str): Directory containing the IBTrACS data file.
    ibfilename (str): Name of the IBTrACS data file (default: "IBTrACS.since1980.v04r01.nc").
    ibversion (str): Version of the IBTrACS data (default: "v4").
    ibst_yr (int): Start year for IBTrACS data (default: 1990).
    iben_yr (int): End year for IBTrACS data (default: datetime.now().year).
    gridfile (str): Path to the grid NetCDF file containing topography data.
    is_grid_2d (bool): Whether the grid is 2D (default: True).
    flip_grid_180 (bool): Whether to flip longitudes from [-180, 180] to [0, 360] (default: True).
    cut_regional (bool): Whether to cut the grid to a regional domain (default: False).
    cut_regional_ring_width (int): Width of the ring to add around the regional domain (default: 8).
    correct_pres_wind (bool): Whether to apply pressure-wind correction to fill in missing P/W with K&Z 07 (default: True).
    dur_thresh (int): Minimum duration threshold for storms (default: 3).
    print_to_screen (bool): Whether to print progress to the screen (default: True).
    print_names (bool): Whether to print storm names (default: False).
    """

    # Define constants
    g = 9.80665
    ms_to_kts = 1.94384449

    # Load IBTrACS data
    print(f"Processing IBTrACS data from {ibst_yr} to {iben_yr}, this may take a while...")
    ibfile = xr.open_dataset(Path(ibdir) / ibfilename)
    

    # Find storm bounds
    ibyear = ibfile.season.values.astype(int)
    valid_years = (ibyear >= ibst_yr-1) & (ibyear <= iben_yr+1)
    stix = np.where(valid_years)[0][0]
    enix = np.where(valid_years)[0][-1]

    # Load IBTrACS data
    iblat, iblon, ibtype, ibwind, ibpres, ibtime, ibname, ibbasin = load_ibtracs_data(
        ibfile, stix, enix, ibversion, ms_to_kts, flip_grid_180
    )

    
    # Convert start and end dates to time units used in IBTrACS
    start_date = pd.Timestamp(f"{ibst_yr}-01-01 00:00:00")
    end_date = pd.Timestamp(f"{iben_yr}-12-31 23:00:00")
    
    ib_units = ibtime.attrs['units']  
    start_date_units = (start_date - pd.Timestamp(ib_units.split("since ")[1])).total_seconds()/3600
    end_date_units = (end_date - pd.Timestamp(ib_units.split("since ")[1])).total_seconds()/3600


    # Get dimensions of the storm data
    ibstormcount = len(ibfile.season.isel(storm=slice(stix, enix+1)))
    ibntimes = iblat.shape[1]  # Number of time points per storm

    # Process storm names (convert from char to str)
    ibnames = []
    for i in range(ibstormcount):
        name = str(ibname[i].values).replace(',', '')
        ibnames.append(name)


    # Correct IBTrACS time precision issues
    ibtime = xr.where(ibtime.notnull(), np.round(ibtime, decimals=3), ibtime)

    # Mask data outside the time range
    valid_time = (ibtime >= start_date_units) & (ibtime <= end_date_units)
    ibwind = xr.where(valid_time, ibwind, np.nan)
    ibpres = xr.where(valid_time, ibpres, np.nan)
    iblat = xr.where(valid_time, iblat, np.nan)
    iblon = xr.where(valid_time, iblon, np.nan)
    ibtime = xr.where(valid_time, ibtime, np.nan)
    ibnames = xr.where(valid_time, ibnames, np.nan)

    # Mask data at non-standard time steps (not every 6 hours)
    eps = 0.00001
    nonstandard_time = (np.mod(ibtime, 0.25) >= eps) | (np.mod(ibtime, 0.25) <= -eps)
    ibwind = xr.where(nonstandard_time, np.nan, ibwind)
    ibpres = xr.where(nonstandard_time, np.nan, ibpres)
    iblat = xr.where(nonstandard_time, np.nan, iblat)
    iblon = xr.where(nonstandard_time, np.nan, iblon)
    ibtime = xr.where(nonstandard_time, np.nan, ibtime)


    # Load PHIS data
    topog = xr.open_dataset(gridfile)
    surf_geopotential = topog['topog'] * g
    phis = surf_geopotential.to_dataset().rename({'topog': 'PHIS'})


    # Process each storm and save IBTrACS data with TempestExtremes format
    ibtempest_filename = f"ibtracs_{ibst_yr}-{iben_yr}_GLOB.{ibversion}.txt"
    output_path = Path("../data/") / ibtempest_filename
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ii in range(ibstormcount):
            # Initialize arrays for grid indices
            latix = np.zeros(ibntimes, dtype=int)
            lonix = np.zeros(ibntimes, dtype=int)
        
            for jj in range(ibntimes):
                if not np.isnan(iblat[ii,jj].values):
                    # Apply pressure-wind relationship if requested
                    if correct_pres_wind:
                        wind, pres = correct_pres_data(ibwind[ii,jj].values, ibpres[ii,jj].values)
                        ibwind[ii,jj] = wind
                        ibpres[ii,jj] = pres


                # Process grid information if provided
                if gridfile:
                    if is_grid_2d:
                        # Load 2D grid data if not already loaded
                        if 'gridlat' not in locals():
                            gridf = topog
                            gridlat = gridf.XLAT
                            gridlon = gridf.XLONG
                            num2dlat, num2dlon = gridlat.shape
                        
                        # Find nearest grid point using great circle distance
                        gcdist, _, _, _ = great_circle_distance(iblat[ii,jj], iblon[ii,jj], gridlat, gridlon)
                        idx = np.unravel_index(np.argmin(gcdist), gcdist.shape)
                        latix[jj], lonix[jj] = idx

                        # Apply regional domain cutting if requested
                        if cut_regional:
                            if (latix[jj] <= (cut_regional_ring_width-1) or
                                latix[jj] >= (num2dlat-cut_regional_ring_width) or
                                lonix[jj] <= (cut_regional_ring_width-1) or
                                lonix[jj] >= (num2dlon-cut_regional_ring_width)):
                                iblat[ii,jj] = np.nan
                                iblon[ii,jj] = np.nan
                    else:
                        # Load 1D grid data if not already loaded
                        if 'gridlat' not in locals():
                            gridf = topog
                            gridlat = gridf.lat
                            gridlon = gridf.lon

                        # Find nearest grid points
                        latix[jj] = np.abs(gridlat - iblat[ii,jj]).argmin()
                        lonix[jj] = np.abs(gridlon - iblon[ii,jj]).argmin()
                else:
                    latix[jj] = -999
                    lonix[jj] = -999

            
            # Count number of valid entries for this storm
            numentries = np.sum(~np.isnan(iblat[ii,:].values))

            # Check if storm meets duration threshold and has valid name
            if numentries > dur_thresh:
                # Find first non-missing index
                valid_points = ~np.isnan(iblat[ii,:].values)
                if not any(valid_points):
                    continue
                ibstix = np.where(valid_points)[0][0]

                # Get date components from first valid time point
                thisdate = pd.Timestamp(ib_units.split("since ")[1]) + pd.Timedelta(hours=float(ibtime[ii,ibstix].values))
                
                # Create header string
                if print_names:
                    header = ibnames[ii]
                else:
                    header = "start"
                    
                headstr = f"{header}\t{numentries}\t{thisdate.year}\t{thisdate.month}\t{thisdate.day}\t{thisdate.hour}"
                
                if print_to_screen:
                    print()
                    print(headstr)
                
                # Check for missing pressure and wind data
                missing_both = np.logical_and(
                    np.logical_and(
                        np.isnan(ibpres[ii,:].values), 
                        np.isnan(ibwind[ii,:].values)
                    ),
                    ~np.isnan(iblat[ii,:].values)
                )
                
                # If all valid points are missing both pressure and wind
                if np.array_equal(~np.isnan(iblat[ii,:].values), missing_both):
                    print(f"********** {ibnames[ii]} in {ibbasin[ii,0].values} is missing ALL pres and wind data " 
                        f"at all times {thisdate.year}\t{thisdate.month}\t{thisdate.day}\t{thisdate.hour}")
                elif np.any(missing_both):
                    print(f"{ibnames[ii]} is missing some pres and wind data at same time "
                          f"{thisdate.year}\t{thisdate.month}\t{thisdate.day}\t{thisdate.hour}")
                    pass

                # Write header to file
                f.write(f"{headstr}\n")

            
            # Process trajectory points
            for jj in range(ibstix, ibntimes):
                if not np.isnan(iblat[ii,jj].values):
                    # Get nearest grid points from previously calculated indices
                    thisLat = latix[jj]
                    thisLon = lonix[jj]

                    # Get surface geopotential at storm location
                    if (iblon[ii,jj] <= phis.lon.max() and iblon[ii,jj] >= phis.lon.min()):
                        thisPHIS = float(phis.PHIS.sel(
                            lat=iblat[ii,jj].values,
                            lon=iblon[ii,jj].values,
                            method='nearest'
                        ))
                    else:
                        thisPHIS = float(phis.PHIS.sel(
                            lat=iblat[ii,jj].values,
                            lon=phis.lon.max(),
                            method='nearest'
                        ))
                    
                    # Get datetime components for this point
                    thisdate = pd.Timestamp(ib_units.split("since ")[1]) + pd.Timedelta(hours=float(ibtime[ii,jj].values))
                    
                    # Create string with storm data
                    stormstr = (f"\t{thisLon}\t{thisLat}"
                            f"\t{iblon[ii,jj].values:6.2f}\t{iblat[ii,jj].values:6.2f}"
                            f"\t{ibpres[ii,jj].values:6.0f}\t{ibwind[ii,jj].values:8.2f}"
                            f"\t{thisPHIS:7.3e}"
                            f"\t{thisdate.year}\t{thisdate.month}\t{thisdate.day}\t{thisdate.hour}")
                    
                    if print_to_screen:
                        print(stormstr)
                    
                    # Write storm data to file
                    f.write(f"{stormstr}\n")


    print(f"IBTrACS data with TempestExtremes format saved to {output_path}.")
    return