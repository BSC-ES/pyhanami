"""
This script calls functions from the TempestExtremes package
Original source: https://github.com/ClimateGlobalChange/tempestextremes
Original author: Paul Ullrich
License: BSD 2-Clause License
Copyright (c) 2025, Paul Ullrich

The criteria for Tropical Cyclones (TCs) detection is taken from (C.M. Zarzycki & P.A. Ullrich, 2017; https://doi.org/10.1002/2016GL071606).
From Section '3.4 Sample Optimization' in the paper: pslFOmag = 2 hPa, wcOffset = 1°, mergeDist = 6°, trajRange = 8°, trajMaxGap = 18 h,
maxTopo = 1500 m, maxLat = 50°, minWind = 10 m s−1, pslFOdist = 5.5°, wcFOmag =− 6 m, wcFOdist = 6.5°, trajMinLen = 60 h
NOTE: based on the criteria used at CEMA (JAMSTEC), the wind threshold is set higher (to 17 m/s).
"""

import shutil
import subprocess
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path


def check_tempestExtremes_installed():
    """
    Check if TempestExtremes is installed and available in the system PATH.
    """
    required_bins = ['DetectNodes', 'StitchNodes', 'HistogramNodes']
    check_binaries = all(shutil.which(binary) is not None for binary in required_bins)

    if not check_binaries:
        raise EnvironmentError("TempestExtremes binaries are not installed or not found in the system PATH. "
                               "Please install TempestExtremes: https://github.com/ClimateGlobalChange/tempestextremes")


def detect_nodes(input_file, output_file="detected_nodes.txt", psl_delta=200.0, psl_dist=5.5, z_delta=-6.0, 
                 z_dist=6.5, z_offset=1.0, merge_dist=6.0):
    """
    Wrapper for DetectNodes function from TempestExtremes to identify nodal features related to tropical cyclones.

    Parameters
    ----------
    input_file (str): input .nc file containing the necessary variables.
    output_file (str): output .txt file to write down the detected nodal features (default: "detected_nodes.txt").
    psl_delta (float): strength of local psl minimum in hPa (default: 200.0).
    psl_dist (float): allowable distance from local psl minimum for psl closed contour in degrees (default: 5.5).
    z_delta (float): strength of warm core anomaly given by zg maximum in m (default: -6.0).
    z_dist (float): allowable distance from warm core anomaly for z closed contour in degrees (default: 6.5).
    z_offset (float): maximum separation between psl minimum and zg maximum in degrees (default: 1.0).
    merge_dist (float): minimum allowable distance between two candidates in degrees (default: 6.0).
    """

    cmd = [
        "DetectNodes",
        "--verbosity", "0",
        "--timestride", "1",
        "--closedcontourcmd", f"PSL,{psl_delta},{psl_dist},0;_DIFF(Z300,Z500),{z_delta},{z_dist},{z_offset}",
        "--mergedist", f"{merge_dist}",
        "--searchbymin", "PSL",
        "--outputcmd", "PSL,min,0;_VECMAG(UBOT,VBOT),max,2;PHIS,max,0",
        "--in_data", f"{input_file}",
        "--out", f"{output_file}"
    ]

    subprocess.run(cmd, check=True)
    print(f"Nodes detection completed. DetectNodes output saved to '{output_file}'", flush=True)


def stitch_nodes(input_file, output_file="cyclones_trajectories.txt", traj_range=8.0, traj_min_length=10, traj_max_gap=3, 
                 min_wind=17.0, min_len=10, max_lat=50.0, format_str="i,j,lon,lat,slp,wind,phis"):
    """
    Wrapper for StitchNodes function from TempestExtremes to identify actual TC tracks from the detected nodes.

    Parameters
    ----------
    input_file (str): input .txt nodefile with the detected nodal features.
    output_file (str): output .txt file to write down a filtered list of TC candidates (default: "cyclones_trajectories.txt").
    traj_range (float): maximum travel distance for a cyclone in 6h in degrees (default: 8.0).
    traj_min_length (int): minimum cyclone lifetime in 6h (default: 10).
    traj_max_gap (int): maximum allowable gap in cyclone trajectory in 6h (default: 3).
    min_wind (float): minimum lowest model level wind speed in m/s (default: 17.0).
    min_len (int): minimum track length in 6h (default: 10).
    max_lat (float): maximum latitude of psl minimum in degrees (default: 50.0).
    format_str (str): format of columns to be added in the output_file, note the following are 
        automatically added as final columns in the output: year (yyyy), month (mm), day (dd), 
        hour (hh) (default: "i,j,lon,lat,slp,wind,phis").
    """

    cmd = [
        "StitchNodes",
        "--format", f"{format_str}", 
        "--range", f"{traj_range}",
        "--mintime", f"{traj_min_length}",
        "--maxgap", f"{traj_max_gap}",
        "--in", f"{input_file}",
        "--out", f"{output_file}",
        "--threshold", f"wind,>=,{min_wind},{min_len};lat,<=,{max_lat},{min_len};lat,>=,-{max_lat},{min_len}"
    ]

    subprocess.run(cmd, check=True)


def histogram_nodes(input_file, output_file="cyclones_trajectories.nc", ilon_col=3, ilatcol=4):
    """
    Wrapper for HistogramNodes function from TempestExtremes to generate histogram of TC detections (easily 
    displayed with 'ncview output_file &').
    
    Parameters
    ----------
    input_file (str): input .txt file containing TC tracks.
    output_file (str): output .nc file to include histogram of TC detections (default: "cyclones_trajectories.nc").
    ilon_col (int): column index for longitude in input file (default: 3).
    ilatcol (int): column index for latitude in input file (default: 4).
    """

    cmd = [
        "HistogramNodes",
        "--in", f"{input_file}",
        "--iloncol", f"{ilon_col}",
        "--ilatcol", f"{ilatcol}",
        "--out", f"{output_file}"
    ]

    subprocess.run(cmd, check=True)


def track_tcs(data_name, input_path, output_path, hist=False):
    """
    Identify tracks of Tropical Cyclones (TCs) in the input data using TempestExtremes.

    Parameters
    ----------
    data_name (str): Name of the dataset.
    input_path (str): Input .nc file path.
    output_path (str): Output path.
    hist (bool): If True, generate a histogram of TC detections as a .nc file (default: False).
    """

    # Check input and output paths
    input_path = Path(input_path)
    if input_path.suffix != '.nc':
        raise ValueError("Input must be a path to a .nc file.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")
    
    output_path = Path(output_path)
    if output_path.suffix != '':
        raise ValueError("Output path must be a directory, not a file path, as multiple files may be created.")
    output_path.mkdir(parents=True, exist_ok=True)


    # Check availability of TempestExtremes binaries
    check_tempestExtremes_installed()

    # Run DetectNodes from TempestExtremes
    nodes_path = output_path / f"detected_nodes_{data_name}.txt"
    detect_nodes(input_path, nodes_path)

    # Run StitchNodes from TempestExtremes
    tracks_path = output_path / f"cyclones_trajectories_{data_name}.txt"
    stitch_nodes(nodes_path, tracks_path)

    # Run HistogramNodes from TempestExtremes if requested
    if hist:
        hist_path = output_path / f"cyclones_trajectories_{data_name}.nc"
        histogram_nodes(tracks_path, hist_path)


    # Delete intermediate files
    nodes_path.unlink()
    for log_file in output_path.glob("log*.txt"):
        log_file.unlink()

    return tracks_path


def read_tracks_tempestExtremes(tracks_file):
    """
    Read TC trajectories from TempestExtremes output file.
    
    Parameters
    ----------
    tracks_file (str): Path to the TempestExtremes output .txt file.

    Returns
    -------
    tracks (list[list[dict]]): List of trajectories.
    """

    if not Path(tracks_file).exists():
        raise FileNotFoundError(f"Tracks file '{tracks_file}' does not exist.")

    tracks = []
    with open(tracks_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('start'):
                parts = line.strip().split()
                num_points = int(parts[1])
                trajectory = []

                for _ in range(num_points):
                    fields = f.readline().strip().split()
                    point = {
                        'lon': float(fields[2]),
                        'lat': float(fields[3]),
                        'psl': float(fields[4]),
                        'wind': float(fields[5]),
                        'phis': float(fields[6]),
                        'year': int(fields[7]),
                        'month': int(fields[8]),
                        'day': int(fields[9]),
                        'hour': int(fields[10])
                    }
                    trajectory.append(point)
                tracks.append(trajectory)

    return tracks


def compute_tc_counts(tracks, start_year, end_year, bin_size=2.5, cutoff_wind=17.0):
    """
    Compute TC genesis and tracks density for each grid box (bin_size x bin_size)
    from a list of TC trajectories.

    Parameters
    ----------
    tracks (list[list[dict]]): List of trajectories.
    start_year, end_year (int): Initial and final years to compute the densities.
    bin_size (float): Size of the bins in degrees (default: 2.5).
    cutoff_wind (float): Minimum wind speed in m/s to consider a TC genesis (default: 17.0).

    Returns
    -------
    counts_gen (xarray.DataArray): TC genesis density.
    counts_traj (xarray.DataArray): TC tracks density.
    """

    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)

    # Create (lat, lon) list of TC positions
    genesis_pts = []
    traj_pts = []

    for track in tracks:
        # Check time range
        first_point = track[0]
        t = pd.Timestamp(year=first_point['year'], month=first_point['month'], day=first_point['day'], hour=first_point['hour'])
        
        # Save genesis and tracks locations
        if start_date <= t <= end_date:
            genesis_found = False
            for point in track:
                lat = point['lat']
                lon = point['lon'] 
                traj_pts.append([lat, lon])

                if not genesis_found and point['wind'] > cutoff_wind:
                    genesis_pts.append([lat, lon])
                    genesis_found = True


    # Group TC positions in bins and save to xarray.DataArray
    lat_bins = np.arange(-90, 90+bin_size, bin_size)  # from -60 to 60 (inclusive)
    lon_bins = np.arange(0, 360+bin_size, bin_size)   # 0 to 360 (if using 0-360 format)

    # Count genesis points
    hist_gen, lat_edges_gen, lon_edges_gen = np.histogram2d(
        [pt[0] for pt in genesis_pts],  
        [pt[1] for pt in genesis_pts], 
        bins=[lat_bins, lon_bins]
    )
    lat_centers_gen = (lat_edges_gen[:-1] + lat_edges_gen[1:]) / 2
    lon_centers_gen = (lon_edges_gen[:-1] + lon_edges_gen[1:]) / 2

    counts_gen = xr.DataArray(
        hist_gen,
        coords={"lat": lat_centers_gen, "lon": lon_centers_gen},
        dims=["lat", "lon"],
        name="point_density",
        attrs={"units": "count"}
    )

    # Count trajectory points
    hist_traj, lat_edges_traj, lon_edges_traj = np.histogram2d(
        [pt[0] for pt in traj_pts],  
        [pt[1] for pt in traj_pts],   
        bins=[lat_bins, lon_bins]
    )
    lat_centers_traj = (lat_edges_traj[:-1] + lat_edges_traj[1:]) / 2
    lon_centers_traj = (lon_edges_traj[:-1] + lon_edges_traj[1:]) / 2

    counts_traj = xr.DataArray(
        hist_traj,
        coords={"lat": lat_centers_traj, "lon": lon_centers_traj},
        dims=["lat", "lon"],
        name="point_density",
        attrs={"units": "count"}
    )

    return counts_gen, counts_traj