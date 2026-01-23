"""
This script contains functions adapted from the CyMeP package
Original source: https://github.com/zarzycki/cymep
Original author: Colin Zarzycki
License: MIT License
Copyright (c) 2021 Colin Zarzycki
"""

import re
import shutil
import requests
import numpy as np
import xarray as xr
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime

from pyhanami.config import config_params
from pyhanami.utils.tcs_scores import tcs_tempestextremes


# Functions adapted from cymep/conver-traj/ibtracs-to-tempest.ncl (translated to Python)

def prepare_ibtracs_data(ib_file, start_idx, end_idx, ms_to_kts=1.94384449, flip_grid_180=True):
    """
    Extract and preprocess IBTrACS data. 

    Parameters
    ----------
    ib_file : xarray.Dataset
        IBTrACS dataset.
    start_idx : int
        Starting index of the storm to extract.
    end_idx : int
        Ending index of the storm to extract.
    ms_to_kts : float
        Conversion factor from m/s to knots.
    flip_grid_180 : bool
        Whether to convert longitudes from [-180, 180] to [0, 360] (default: True).

    Returns
    -------
    ib_lat : xarray.DataArray
        Latitudes of the storm track.
    ib_lon : xarray.DataArray
        Longitudes of the storm track.
    ib_type : xarray.DataArray
        Storm types along the track.
    ib_wind : xarray.DataArray
        Wind speeds along the track (in knots).
    ib_pres : xarray.DataArray
        Pressures along the track (in hPa).
    ib_time : xarray.DataArray
        Time points along the track.
    ib_name : xarray.DataArray
        Storm names.
    ib_basin : xarray.DataArray
        Basin codes of the storm.
    """

    ib_dataset=config_params.IBTRACS_DATASET.lower()
    if config_params.IBTRACS_VERSION == "v3":
        ib_lat = ib_file[f'lat_{ib_dataset}'].isel(storm=slice(start_idx, end_idx+1)) * 0.01
        ib_lon = ib_file[f'lon_{ib_dataset}'].isel(storm=slice(start_idx, end_idx+1)) * 0.01
        ib_type = ib_file[f'nature_{ib_dataset}'].isel(storm=slice(start_idx, end_idx+1)).astype(int)
        ib_wind = ib_file[f'wind_{ib_dataset}'].isel(storm=slice(start_idx, end_idx+1)) * 0.1 / ms_to_kts
        ib_pres = ib_file[f'pres_{ib_dataset}'].isel(storm=slice(start_idx, end_idx+1))
        ib_time = ib_file[f'time_{ib_dataset}'].isel(storm=slice(start_idx, end_idx+1))
    else:
        ib_lat = ib_file.lat.isel(storm=slice(start_idx, end_idx+1))
        ib_lon = ib_file.lon.isel(storm=slice(start_idx, end_idx+1))
        ib_type = ib_file.nature.isel(storm=slice(start_idx, end_idx+1)).astype(str)
        ib_wind = ib_file[f'{ib_dataset}_wind'].isel(storm=slice(start_idx, end_idx+1)) / ms_to_kts
        ib_pres = ib_file[f'{ib_dataset}_pres'].isel(storm=slice(start_idx, end_idx+1)) * 100
        ib_time = ib_file.time.isel(storm=slice(start_idx, end_idx+1))

    ib_name = ib_file.name.isel(storm=slice(start_idx, end_idx+1))
    ib_basin = ib_file.basin.isel(storm=slice(start_idx, end_idx+1)).astype(str)

    if flip_grid_180:
        ib_lon = xr.where(ib_lon < 0, ib_lon + 360, ib_lon)

    return ib_lat, ib_lon, ib_type, ib_wind, ib_pres, ib_time, ib_name, ib_basin


def correct_time_data(ib_time, ib_wind, ib_pres, ib_lat, ib_lon, ib_names, valid_time):
    """
    Correct and mask IBTrACS data based on time validity and standard time steps.

    Parameters
    ----------
    ib_time : xarray.DataArray
        Time points along the storm track.
    ib_wind : xarray.DataArray
        Wind speeds along the storm track.
    ib_pres : xarray.DataArray
        Pressures along the storm track.
    ib_lat : xarray.DataArray
        Latitudes along the storm track.
    ib_lon : xarray.DataArray
        Longitudes along the storm track.
    ib_names : xarray.DataArray
        Storm names.
    valid_time : xarray.DataArray
        Boolean array indicating valid time points.

    Returns
    -------
    ib_time : xarray.DataArray
        Corrected time points.
    ib_wind : xarray.DataArray
        Masked wind speeds.
    ib_pres : xarray.DataArray
        Masked pressures.
    ib_lat : xarray.DataArray
        Masked latitudes.
    ib_lon : xarray.DataArray
        Masked longitudes.
    ib_names : xarray.DataArray
        Masked storm names.
    """

    # Correct null values
    ib_time = xr.where(ib_time.notnull(), ib_time, np.datetime64('NaT'))

    # Mask data outside the time range
    ib_wind = xr.where(valid_time, ib_wind, np.nan)
    ib_pres = xr.where(valid_time, ib_pres, np.nan)
    ib_lat = xr.where(valid_time, ib_lat, np.nan)
    ib_lon = xr.where(valid_time, ib_lon, np.nan)
    ib_time = xr.where(valid_time, ib_time, np.datetime64('NaT'))
    ib_names = xr.where(valid_time, ib_names, '')

    # Mask data at non-standard time steps (not every 6 hours)
    eps = 0.00001
    hours_since_midnight = ib_time.dt.hour + ib_time.dt.minute/60.0
    nonstandard_time = (np.mod(hours_since_midnight, 6) >= eps) | (np.mod(hours_since_midnight, 6) <= -eps)

    ib_wind = xr.where(nonstandard_time, np.nan, ib_wind)
    ib_pres = xr.where(nonstandard_time, np.nan, ib_pres)
    ib_lat = xr.where(nonstandard_time, np.nan, ib_lat)
    ib_lon = xr.where(nonstandard_time, np.nan, ib_lon)
    ib_time = xr.where(nonstandard_time, np.datetime64('NaT'), ib_time)
    ib_names = xr.where(nonstandard_time, '', ib_names)

    return ib_time, ib_wind, ib_pres, ib_lat, ib_lon, ib_names


def correct_wind_pres_data(wind, pres):
    """
    Apply K&Z 07 relationship to fill missing pressure-wind data
    when possible, keep NaN values otherwise.

    Not used anymore, replaced by vectorized version.
    
    Parameters
    ----------
    wind : float
        Wind speed in m/s.
    pres : float
        Pressure in Pa.

    Returns
    -------
    wind : float
        Corrected wind speed in m/s.
    pres : float
        Corrected pressure in Pa.
    """

    a, b, c = 2.3, 1010.0, 0.76
    with np.errstate(invalid='ignore', divide='ignore'):
        if np.isnan(wind) and not np.isnan(pres):
                wind = a * np.power((b - pres/100.0), c)
        elif not np.isnan(wind) and np.isnan(pres):
                pres = 100.0 * (b - np.power((wind/a), (1.0/c)))
        elif np.isnan(wind) and np.isnan(pres):
            wind, pres = 15.0, 100800.0

    return wind, pres


def correct_wind_pres_data_vectorized(wind, pres):
    """
    Apply K&Z 07 relationship to fill missing pressure-wind data
    when possible, keep NaN values otherwise.
    
    Parameters
    ----------
    wind : np.ndarray
        Wind speed in m/s.
    pres : np.ndarray
        Pressure in Pa.

    Returns
    -------
    wind : np.ndarray
        Corrected wind speed in m/s.
    pres : np.ndarray
        Corrected pressure in Pa.
    """

    wind_corr = wind.copy()
    pres_corr = pres.copy()

    a, b, c = 2.3, 1010.0, 0.76
    with np.errstate(invalid='ignore', divide='ignore'):
        mask1 = np.isnan(wind) & ~np.isnan(pres)
        wind_corr[mask1] = a * np.power((b - pres[mask1]/100.0), c)

        mask2 = ~np.isnan(wind) & np.isnan(pres)
        pres_corr[mask2] = 100.0 * (b - np.power((wind[mask2]/a), (1.0/c)))

        mask3 = np.isnan(wind) & np.isnan(pres)
        wind_corr[mask3] = 15.0
        pres_corr[mask3] = 100800.0

    return wind_corr, pres_corr


def great_circle_distance(lat1, lon1, lat2, lon2, npts=2, iu=4):
    """
    Calculate great circle distance between points and interpolate points along the path
    following 'gc_latlon' function in NCL.
    
    Parameters
    ----------
    lat1 : float or np.ndarray
        Latitude(s) of first point(s).
    lon1 : float or np.ndarray
        Longitude(s) of first point(s).
    lat2 : float or np.ndarray
        Latitude(s) of second point(s).
    lon2 : float or np.ndarray
        Longitude(s) of second point(s).
    npts : int
        Number of points to interpolate (default: 2).
    iu : int
        Unit flag (default: 4)
            |iu| = 1: radians
            |iu| = 2: degrees
            |iu| = 3: meters
            |iu| = 4: kilometers
            sign(iu) > 0: longitudes in [0,360]
            sign(iu) < 0: longitudes in [-180,180]
    
    Returns
    -------
    distance : float or np.ndarray
        Great circle distance in requested units.
    gclat : np.ndarray
        Latitudes along great circle path
    gclon : np.ndarray
        Longitudes along great circle path
    spacing : float
        Distance between interpolated points
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


def convert_ibtracs_to_tempest(start_year=config_params.IBTRACS_START_YEAR, end_year=datetime.now().year, min_wind=10.0,
                               flip_grid_180=True, is_grid_2d=False, cut_regional=False, cut_regional_ring_width=8, 
                               correct_pres_wind=True, dur_thresh=3, print_names=False):
    """
    Convert IBTrACS data to TempestExtremes format and save to a .txt file.

    Parameters
    ----------
    start_year : int
        Start year for IBTrACS data (default: config_params.IBTRACS_START_YEAR).
    end_year : int
        End year for IBTrACS data (default: datetime.now().year).
    min_wind : float
        minimum 10 m wind speed in m/s for TCs detection (default: 10.0).
    flip_grid_180 : bool
        Whether to flip longitudes from [-180, 180] to [0, 360] (default: True).
    is_grid_2d : bool
        Whether the grid is 2D (default: False).
    cut_regional : bool
        Whether to cut the grid to a regional domain (default: False).
    cut_regional_ring_width : int
        Width of the ring to add around the regional domain (default: 8).
    correct_pres_wind : bool
        Whether to apply pressure-wind correction to fill in missing P/W with K&Z 07 (default: True).
    dur_thresh : int
        Minimum duration threshold for storms (default: 3).
    print_names : bool
        Whether to print storm names in the output file (default: False).
    """

    # Define constants
    ms_to_kts = 1.94384449

    # Load IBTrACS data
    print(f"Processing IBTrACS data from {start_year} to {end_year} with TempestExtremes...", flush=True)
    ib_file = xr.open_dataset(config_params.IBTRACS_PATH)


    # Extract storms data for the specified years
    # Find storm bounds for the specified years (first and last instances of valid years)
    ib_year = ib_file.season.values.astype(int)
    valid_years = (ib_year >= start_year-1) & (ib_year <= end_year+1)
    start_idx = np.where(valid_years)[0][0]
    end_idx = np.where(valid_years)[0][-1]

    # Split IBTrACS data for the specified storms
    ib_lat, ib_lon, ib_type, ib_wind, ib_pres, ib_time, ib_name, ib_basin = prepare_ibtracs_data(
        ib_file, start_idx, end_idx, ms_to_kts, flip_grid_180
    )


    # Preprocess storm parameters
    # Convert start and end dates to time units used in IBTrACS
    start_date = np.datetime64(f"{start_year:04d}-01-01T00:00:00")
    end_date = np.datetime64(f"{end_year:04d}-12-31T23:00:00")
    valid_time = (ib_time >= start_date) & (ib_time <= end_date)

    # Get dimensions of the storm data (Nº storms + Nº time points per storm)
    ib_stormcount = len(ib_file.season.isel(storm=slice(start_idx, end_idx+1)))
    ib_ntimes = ib_lat.shape[1]  

    # Process storm names (convert from char to str)
    ib_names_list = [str(ib_name[i].values).replace(',', '') for i in range(ib_stormcount)]
    ib_names = xr.DataArray(
        ib_names_list,
        dims=ib_name.dims,
        coords=ib_name.coords,
        name='name'
    )

    # Correct time values
    ib_time, ib_wind, ib_pres, ib_lat, ib_lon, ib_names = correct_time_data(
        ib_time, ib_wind, ib_pres, ib_lat, ib_lon, ib_names, valid_time
    )

    # Preload and process grid data if requested
    topog = xr.open_dataset(config_params.TOPOG_PATH) 
    if is_grid_2d:
        gridlat = topog.XLAT
        gridlon = topog.XLONG
        num2dlat, num2dlon = gridlat.shape
    else:
        gridlat = topog.lat
        gridlon = topog.lon
                            
    # Prepare PHIS data
    topog_varname = config_params.TOPOG_VARNAME
    surf_geopotential = topog[topog_varname] * config_params.G
    phis = surf_geopotential.to_dataset().rename({topog_varname: 'PHIS'}) 


    # Process each storm and save IBTrACS data in TempestExtremes format
    ib_tempest_filename = f"ibtracs_{config_params.IBTRACS_VERSION}_{start_year}-{end_year}_{min_wind:.1f}_False_1_1.0.txt"
    output_dir = config_params.TC_DATA_PATH
    output_path = output_dir / ib_tempest_filename

    with open(output_path, 'w') as f:
        for ii in tqdm(range(ib_stormcount), total=ib_stormcount, desc='\tProcessing storms', unit=' storms'):
            # Check for at least one not NaN point
            valid_points = ~np.isnan(ib_lat[ii,:].values)
            if not np.any(valid_points):
                continue

            # Initialize lat-lon arrays
            latix = np.repeat(np.nan, ib_ntimes)
            lonix = np.repeat(np.nan, ib_ntimes)

            # Apply pressure-wind relationship to correct for missing values if requested
            valid_indices = np.where(valid_points)[0]
            if correct_pres_wind:
                wind, pres = correct_wind_pres_data_vectorized(ib_wind[ii,valid_indices].values, ib_pres[ii,valid_indices].values)
                ib_wind[ii,valid_indices] = wind
                ib_pres[ii,valid_indices] = pres

            # Apply minimum wind speed threshold
            max_wind = np.nanmax(np.abs(ib_wind[ii,:].values))
            if max_wind < min_wind:
                continue


            # Process storm points
            if topog:
                if is_grid_2d:
                    for jj in valid_indices:                           
                        # Find nearest grid point using great circle distance
                        gcdist, _, _, _ = great_circle_distance(ib_lat[ii,jj], ib_lon[ii,jj], gridlat, gridlon)
                        idx = np.unravel_index(np.argmin(gcdist), gcdist.shape)
                        latix[jj], lonix[jj] = idx

                        # Apply regional domain cutting if requested
                        if cut_regional:
                            if (latix[jj] <= (cut_regional_ring_width-1) or
                                latix[jj] >= (num2dlat-cut_regional_ring_width) or
                                lonix[jj] <= (cut_regional_ring_width-1) or
                                lonix[jj] >= (num2dlon-cut_regional_ring_width)):
                                ib_lat[ii,jj] = np.nan
                                ib_lon[ii,jj] = np.nan
                else:
                    for jj in valid_indices:   
                        # Find nearest grid points
                        latix[jj] = np.abs((gridlat - ib_lat[ii,jj]).values).argmin()
                        lonix[jj] = np.abs((gridlon - ib_lon[ii,jj]).values).argmin()
            else:
                for jj in valid_indices:   
                    latix[jj] = -999
                    lonix[jj] = -999

            
            # Count number of valid entries for this storm
            numentries = np.sum(~np.isnan(ib_lat[ii,:].values))

            # Check if storm meets duration threshold and has valid name
            if numentries > dur_thresh:
                # Find first non-missing index
                ib_start_idx = valid_indices[0]

                # Get date components from first valid time point
                dt = pd.Timestamp(ib_time[ii,ib_start_idx].values)
                thisdate = (dt.year, dt.month, dt.day, dt.hour)

                # Create and write header string
                header = ib_names[ii] if print_names else "start"
                headstr = f"{header}\t{numentries}\t{thisdate[0]}\t{thisdate[1]}\t{thisdate[2]}\t{thisdate[3]}"
                f.write(f"{headstr}\n")
                

                # Check for missing pressure and wind data
                missing_both = np.logical_and(
                    np.logical_and(
                        np.isnan(ib_pres[ii,:].values), 
                        np.isnan(ib_wind[ii,:].values)
                    ),
                    ~np.isnan(ib_lat[ii,:].values)
                )
                
                # Handle points missing both pressure and wind
                if np.array_equal(~np.isnan(ib_lat[ii,:].values), missing_both):
                    print(f"{ib_names[ii]} in {ib_basin[ii,0].values} is missing all pressure and wind data " 
                        f"at all times {thisdate[0]}\t{thisdate[1]}\t{thisdate[2]}\t{thisdate[3]}.", flush=True)
                elif np.any(missing_both):
                    print(f"{ib_names[ii]} is missing some pressure and wind data at same time "
                          f"{thisdate[0]}\t{thisdate[1]}\t{thisdate[2]}\t{thisdate[3]}.", flush=True)
                    

                # Process and write trajectory points
                for jj in range(ib_start_idx, ib_ntimes):
                    if not np.isnan(ib_lat[ii,jj].values):
                        # Get nearest grid points from previously calculated indices
                        thisLat = latix[jj].astype(int)
                        thisLon = lonix[jj].astype(int)

                        # Get surface geopotential at storm location
                        lon_val = ib_lon[ii,jj].values
                        if (lon_val <= phis.lon.max() and lon_val >= phis.lon.min()):
                            thisPHIS = phis.PHIS.sel(lat=ib_lat[ii,jj].values, lon=lon_val, method='nearest').item()
                        else:
                            thisPHIS = phis.PHIS.sel(lat=ib_lat[ii,jj].values, lon=phis.lon.max(), method='nearest').item()

                        # Get datetime components for this point
                        dt = pd.Timestamp(ib_time[ii,jj].values)
                        thisdate = (dt.year, dt.month, dt.day, dt.hour)
                        
                        # Create string with storm data and write to file
                        stormstr = (f"\t{thisLon}\t{thisLat}"
                                f"\t{ib_lon[ii,jj].values:6.2f}\t{ib_lat[ii,jj].values:6.2f}"
                                f"\t{ib_pres[ii,jj].values:6.0f}\t{ib_wind[ii,jj].values:8.2f}"
                                f"\t{thisPHIS:7.3e}"
                                f"\t{thisdate[0]}\t{thisdate[1]}\t{thisdate[2]}\t{thisdate[3]}")
                        f.write(f"{stormstr}\n")


    print(f"IBTrACS data in TempestExtremes format saved to {output_path}.", flush=True)
    return



# Original functions (not adapted from CyMeP)

def check_ibtracs_date():
    """
    Check the last modification date of the selected IBTrACS data file on the NOAA website.

    Returns
    -------
    ib_date : datetime
        Last modification date.
    """

    # Get web parameters
    ib_url = config_params.IBTRACS_URL
    ib_base_url, ib_filename = ib_url.rsplit('/', 1)
    ib_date = None

    try:
        # Get the directory listing page
        response = requests.get(ib_base_url)
        response.raise_for_status()

        # Parse HTML content to find the last modification date
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if cols and ib_filename in cols[0].text:
                date_str = cols[1].text.strip()
                print(date_str, flush=True)
                ib_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                break

        # Raise error if file not found
        if ib_date is None:
            raise ValueError("Issue encountered when preparing IBTrACS data for Tropical Cyclones:"
                              f"\n\tCould not find the IBTrACS file '{ib_filename}' on the webpage '{ib_url}'."
                              "\n\tPlease, check the IBTRACS_URL parameter in 'config/config_params.py'")

    except requests.exceptions.RequestException as e:
        raise ValueError("Issue encountered when preparing IBTrACS data for Tropical Cyclones:"
                         f"\n\tError accessing IBTrACS webpage '{ib_base_url}': {e}."
                         "\n\tPlease, check your Internet connection and the IBTRACS_URL parameter in 'config/config_params.py'.")

    return ib_date


def download_ibtracs():
    """
    Download and save IBTrACS data file using IBTRACS_URL and IBTRACS_PATH from 'config/config_params.py'.
    """

    ib_file_path = config_params.IBTRACS_PATH
    ib_file_path.parent.mkdir(parents=True, exist_ok=True)
    if ib_file_path.exists():
        ib_file_path.unlink()
    ib_file_path.touch()

    ib_url = config_params.IBTRACS_URL
    try:
        response = requests.get(ib_url, stream=True)
        response.raise_for_status()

        with open(ib_file_path, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
        print(f"Updated IBTrACS data file and saved to {ib_file_path}.", flush=True)
    
    except requests.exceptions.RequestException as e:
        raise ValueError("Issue encountered when preparing IBTrACS data for Tropical Cyclones:"
                         f"\n\tError downloading IBTrACS data file from {ib_url}: {e}."
                         "\n\tPlease, check your Internet connection and the IBTRACS_URL parameter in 'config/config_params.py'.")

    return


def check_ibtracs_file(start_year, end_year, output_path, min_wind=10.0):
    """
    Check, and create if not existing, .txt file with IBTrACS data for the specified 
    time period in TempestExtremes format, downloading updated IBTrACS data if necessary.
    Then, apply minimum wind speed threshold if needed, and copy the final file to the 
    temporary output_path.

    Parameters
    ----------
    start_year : int
        Start year for IBTrACS data.
    end_year : int
        End year for IBTrACS data.
    output_path : str
        Path to save temporary files.
    min_wind : float
        Minimum 10 m wind speed in m/s for TCs detection (default: 10.0).

    Returns
    -------
    new_ib_file_path : Path
        Path to the processed IBTrACS data file.
    file_end_year : int
        End year of the processed IBTrACS data file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate input start year
    if start_year < config_params.IBTRACS_START_YEAR:
        raise ValueError("Issue encountered when preparing IBTrACS data for Tropical Cyclones:"
                         f"\n\tIBTrACS data only goes back to {config_params.IBTRACS_START_YEAR}. "
                         f"\n\tHence, the requested start year {start_year} is not covered."
                          "\n\tPlease, choose a later start year.")

    # Check years in existing IBTrACS files
    search_path = config_params.TC_DATA_PATH
    current_version = config_params.IBTRACS_VERSION
    ib_files = list(search_path.glob(f"ibtracs_{current_version}_*.txt"))

    ib_file = None
    ib_file_path = None
    if ib_files:
        # Get the existing file for the current version
        ib_file = ib_files[0]
        match = re.search(rf'ibtracs_{current_version}_(\d+)-(\d+)', ib_file.name)
        if match:
            file_start_year = int(match.group(1))
            file_end_year = int(match.group(2))

            # Check whether the current version covers the given period
            if file_start_year <= start_year and end_year <= file_end_year:
                ib_file_path = ib_file
                
        else:
            raise ValueError("Issue encountered when preparing IBTrACS data for Tropical Cyclones:"
                             f"\n\tCould not parse years from existing IBTrACS file name '{ib_file}' in "
                             f"the package's data directory '{config_params.TC_DATA_PATH}'.")


    # Download new IBTrACS data if no file exists for the requested version and period
    if ib_file_path is None:
        # Check if the selected year is available on the IBTrACS website
        web_date = check_ibtracs_date()
        if web_date.year < end_year:
            raise ValueError("Issue encountered when preparing IBTrACS data for Tropical Cyclones:"
                             f"\n\tThe requested end year {end_year} is not covered by the IBTrACS data available "
                             f"on '{config_params.IBTRACS_URL}'. \n\tThe latest available year is {web_date.year}."
                             "\n\tPlease, choose an earlier end year or update the IBTRACS_URL in 'config/config_params.py'"
                             " if newer data is available on another website")
        # Remove outdated IBTrACS file
        if ib_file is not None:
            ib_file.unlink()

        # Download and process new IBTrACS data
        download_ibtracs()
        file_start_year = config_params.IBTRACS_START_YEAR
        file_end_year = web_date.year
        ib_file_path = convert_ibtracs_to_tempest(end_year=file_end_year, min_wind=10.0)

    # Apply wind threshold
    if min_wind > 10.0:
        new_ib_file_path = output_path / f'ibtracs_{current_version}_{file_start_year}-{file_end_year}_{min_wind:.1f}_False_1_1.0.txt'
        tcs_tempestextremes.filter_tracks_by_wind(ib_file_path, new_ib_file_path, cutoff_wind=min_wind)
    else:
        new_ib_file_path = output_path / ib_file_path.name
        shutil.copy2(ib_file_path, new_ib_file_path)

    return new_ib_file_path, file_end_year