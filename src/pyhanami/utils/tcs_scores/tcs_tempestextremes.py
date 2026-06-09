"""
This script calls functions from the TempestExtremes package
Original source: https://github.com/ClimateGlobalChange/tempestextremes
Original author: Paul Ullrich
License: BSD 2-Clause License
Copyright (c) 2025, Paul Ullrich

The criteria for Tropical Cyclones (TCs) detection and tracking is taken from
(C.M. Zarzycki & P.A. Ullrich, 2017). From Section '3.4 Sample Optimization'
in the paper (param_name_here (param_name_in_paper) = default_value):
- psl_delta (pslFOmag) = 2 hPa
- z_offset (wcMaxOffset) = 1°
- merge_dist (mergeDist) = 6°
- traj_range (trajRange) = 8°
- traj_max_gap (trajMaxGap) = 18 h
- maxTopo = 1500 m
- max_lat (maxLat) = 50°
- min_wind (minWind) = 10 m s**-1
- psl_dist (pslFOdist) = 5.5°
- z_delta (scFOmax) = −6 m
- z_dist (wcFOdist) = 6.5°
- traj_min_length (trajMinLen) = 60 h
The default values are defined in the configuration file
`pyhanami.config.scientific_evaluation_parameters.yaml`.

NOTE: We recommend adjusting the min_wind parameter (i.e. 10 m wind speed threshold)
according to the model's (or reanalysis') horizontal resolution following the
criteria established in (K.J.E. Walsh et al., 2007).
"""

import shutil
import subprocess
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from pyhanami.utils import data_general
from pyhanami.config import config_params


def prepare_data_tempestExtremes(data, data_name):
    """
    Prepare input data for TempestExtremes.

    Parameters
    ----------
    data : xarray.Dataset
        Input dataset containing the necessary variables.
    data_name : str
        Name of the dataset.

    Returns
    -------
    data_tempestExtremes : xarray.Dataset
        Dataset with variables renamed and surface geopotential added.
    """

    # Check required variables and rename them
    data_vars = []
    var_names = ["psl", "uas", "vas", "zg300", "zg500"]
    new_names = ["PSL", "UBOT", "VBOT", "Z300", "Z500"]
    for var_name, new_name in zip(var_names, new_names):
        if var_name not in data.data_vars:
            raise ValueError(
                f"Variable '{var_name}' not found in the simulated dataset '{data_name}'. "
                f"Available variables: {list(data.data_vars.keys())}"
            )

        data_vars.append(data.rename({var_name: new_name})[new_name])

    # Add surface geopotential
    if "phis" in data.data_vars:
        data_vars.append(data.rename({"phis": "PHIS"})["PHIS"])
    else:
        topog_varname = config_params.TOPOG_VARNAME
        topog_original = xr.open_dataset(config_params.TOPOG_PATH)
        topog_regridded = data_general.regrid_data(
            topog_original, data[["psl"]].isel(time=0), var=topog_varname, method="bilinear"
        )

        surf_geopotential = topog_regridded[topog_varname] * config_params.G
        phis = surf_geopotential.to_dataset().rename({topog_varname: "PHIS"})
        data_vars.append(phis["PHIS"])

    data_tempestExtremes = xr.merge(
        data_vars, join="inner"
    )  # join='inner' keeps only common coordinates
    return data_tempestExtremes


def check_tempestExtremes_installed():
    """
    Check if TempestExtremes is installed and available in the system PATH.
    """
    required_bins = ["DetectNodes", "StitchNodes", "HistogramNodes"]
    check_binaries = all(shutil.which(binary) is not None for binary in required_bins)

    if not check_binaries:
        raise EnvironmentError(
            "TempestExtremes binaries are not installed or not found in the system PATH. "
            "Please install TempestExtremes: https://github.com/ClimateGlobalChange/tempestextremes"
        )


def detect_nodes(input_file, output_file="detected_nodes.txt", psl_delta=200.0, psl_dist=5.5,
                 z_delta=-6.0, z_dist=6.5, z_offset=1.0, merge_dist=6.0):
    """
    Wrapper for DetectNodes function from TempestExtremes to identify nodal features related to
    tropical cyclones.

    Parameters
    ----------
    input_file : str
        Input .nc file containing the necessary variables.
    output_file : str
        Output .txt file to write down the detected nodal features (default: "detected_nodes.txt").
    psl_delta : float
        Strength of local psl minimum, in Pa (default: 200.0).
    psl_dist : float
        Allowable distance from local psl minimum for psl closed contour, in degrees (default: 5.5).
    z_delta : float
        Strength of warm core anomaly given by zg maximum, in m (default: -6.0).
    z_dist : float
        Allowable distance from warm core anomaly for z closed contour, in degrees (default: 6.5).
    z_offset : float
        Maximum separation between psl minimum and zg maximum, in degrees (default: 1.0).
    merge_dist : float
        Minimum allowable distance between two candidates, in degrees (default: 6.0).
    """

    cmd = [
        "DetectNodes",
        "--verbosity",
        "0",
        # "--timestride", "1",  # Deprecated, it only examines discrete times at the given stride, replaced by --timefilter
        "--timefilter",
        "6hr",
        "--closedcontourcmd",
        f"PSL,{psl_delta},{psl_dist},0;_DIFF(Z300,Z500),{z_delta},{z_dist},{z_offset}",
        "--mergedist",
        f"{merge_dist}",
        "--searchbymin",
        "PSL",
        "--outputcmd",
        "PSL,min,0;_VECMAG(UBOT,VBOT),max,2;PHIS,max,0",
        "--in_data",
        f"{input_file}",
        "--out",
        f"{output_file}",
    ]

    subprocess.run(cmd, check=True)
    print(f"Nodes detection completed. DetectNodes output saved to '{output_file}'", flush=True)


def stitch_nodes(input_file, output_file="cyclones_trajectories.txt", traj_range=8.0,
                 traj_min_length=10, traj_max_gap=3, min_wind=10.0, min_len=10,
                 max_lat=50.0, format_str="i,j,lon,lat,slp,wind,phis"):
    """
    Wrapper for StitchNodes function from TempestExtremes to identify actual TC tracks
    from the detected nodes.

    Parameters
    ----------
    input_file : str
        Input .txt nodefile with the detected nodal features.
    output_file : str
        Output .txt file to write down a filtered list of TC candidates (default:
        "cyclones_trajectories.txt").
    traj_range : float
        Maximum travel distance for a cyclone in 6h, in degrees (default: 8.0).
    traj_min_length : int
        Minimum cyclone lifetime, in 6h-intervals (default: 10).
    traj_max_gap : int
        Maximum allowable gap in cyclone trajectory, in 6h-intervals (default: 3).
    min_wind : float
        Minimum 10 m wind speed, in m/s (default: 10.0).
    min_len : int
        Minimum track length, in 6h-intervals (default: 10).
    max_lat : float
        Maximum latitude of psl minimum, in degrees (default: 50.0).
    format_str : str
        Format of columns to be added in the output_file, note the following are
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
        "--threshold", 
        f"wind,>=,{min_wind},{min_len};lat,<=,{max_lat},{min_len};lat,>=,-{max_lat},{min_len}",
    ]

    subprocess.run(cmd, check=True)


def histogram_nodes(input_file, output_file="cyclones_trajectories.nc", ilon_col=3, ilatcol=4):
    """
    Wrapper for HistogramNodes function from TempestExtremes to generate histogram of TC detections
    (easily displayed with 'ncview output_file &').

    Parameters
    ----------
    input_file : str
        Input .txt file containing TC tracks.
    output_file : str
        Output .nc file to include histogram of TC detections (default: "cyclones_trajectories.nc").
    ilon_col : int
        Column index for longitude in input file (default: 3).
    ilatcol : int
        Column index for latitude in input file (default: 4).
    """

    cmd = [
        "HistogramNodes",
        "--in", f"{input_file}",
        "--iloncol", f"{ilon_col}",
        "--ilatcol", f"{ilatcol}",
        "--out", f"{output_file}",
    ]

    subprocess.run(cmd, check=True)


def track_tcs(data_name, input_path, output_path, hist=False, psl_delta=200.0, psl_dist=5.5,
              z_delta=-6.0, z_dist=6.5, z_offset=1.0, merge_dist=6.0, traj_range=8.0,
              traj_min_length=10, traj_max_gap=3, min_wind=10.0, min_len=10, max_lat=50.0):
    """
    Identify tracks of Tropical Cyclones (TCs) in the input data using TempestExtremes.

    Parameters
    ----------
    data_name : str
        Name of the dataset.
    input_path : str
        Input .nc file path.
    output_path : str
        Output path.
    hist : bool
        If True, generate a histogram of TC detections as a .nc file (default: False).
    psl_delta : float
        Strength of local psl minimum, in Pa (default: 200.0).
    psl_dist : float
        Allowable distance from local psl minimum for psl closed contour, in degrees (default: 5.5).
    z_delta : float
        Strength of warm core anomaly given by zg maximum, in m (default: -6.0).
    z_dist : float
        Allowable distance from warm core anomaly for z closed contour, in degrees (default: 6.5).
    z_offset : float
        Maximum separation between psl minimum and zg maximum, in degrees (default: 1.0).
    merge_dist : float
        Minimum allowable distance between two candidates, in degrees (default: 6.0).
    traj_range : float
        Maximum travel distance for a cyclone in 6h, in degrees (default: 8.0).
    traj_min_length : int
        Minimum cyclone lifetime, in 6h-intervals (default: 10).
    traj_max_gap : int
        Maximum allowable gap in cyclone trajectory, in 6h-intervals (default: 3).
    min_wind : float
        Minimum 10 m wind speed for TCs detection, in m/s (default: 10.0).
    min_len : int
        Minimum track length, in 6h-intervals (default: 10).
    max_lat : float
        Maximum latitude of psl minimum, in degrees (default: 50.0).
    """

    # Check input and output paths
    input_path = Path(input_path)
    if input_path.suffix != ".nc":
        raise ValueError("Input must be a path to a .nc file.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

    output_path = Path(output_path)
    if output_path.suffix != "":
        raise ValueError(
            "Output path must be a directory, not a file path, as multiple files may be created."
        )
    output_path.mkdir(parents=True, exist_ok=True)


    # Check availability of TempestExtremes binaries
    check_tempestExtremes_installed()

    # Run DetectNodes from TempestExtremes
    nodes_path = output_path / f"detected_nodes_{data_name}.txt"
    detect_nodes(
        input_path,
        nodes_path,
        psl_delta=psl_delta,
        psl_dist=psl_dist,
        z_delta=z_delta,
        z_dist=z_dist,
        z_offset=z_offset,
        merge_dist=merge_dist,
    )

    # Run StitchNodes from TempestExtremes
    tracks_path = output_path / f"cyclones_trajectories_{data_name}.txt"
    stitch_nodes(
        nodes_path,
        tracks_path,
        traj_range=traj_range,
        traj_min_length=traj_min_length,
        traj_max_gap=traj_max_gap,
        min_wind=min_wind,
        min_len=min_len,
        max_lat=max_lat,
    )

    # Run HistogramNodes from TempestExtremes if requested
    if hist:
        hist_path = output_path / f"cyclones_trajectories_{data_name}.nc"
        histogram_nodes(tracks_path, hist_path)


    # Delete intermediate files
    nodes_path.unlink()
    for log_file in output_path.glob("log*.txt"):
        log_file.unlink()

    return tracks_path


def run_tempestExtremes(data, data_name, output_path, tracks_hist=False, psl_delta=200.0,
                        psl_dist=5.5, z_delta=-6.0, z_dist=6.5, z_offset=1.0, merge_dist=6.0,
                        traj_range=8.0, traj_min_length=10, traj_max_gap=3, min_wind=10.0,
                        min_len=10, max_lat=50.0):
    """
    Main function to run TempestExtremes for identifying Tropical Cyclones (TCs) tracks.

    Parameters
    ----------
    data : xarray.Dataset
        Input dataset.
    data_name : str
        Name of the dataset.
    output_path : str
        Output path.
    tracks_hist : bool
        If True, generate a histogram of TC detections as a .nc file (default: False).
    psl_delta : float
        Strength of local psl minimum, in Pa (default: 200.0).
    psl_dist : float
        Allowable distance from local psl minimum for psl closed contour, in degrees (default: 5.5).
    z_delta : float
        Strength of warm core anomaly given by zg maximum, in m (default: -6.0).
    z_dist : float
        Allowable distance from warm core anomaly for z closed contour, in degrees (default: 6.5).
    z_offset : float
        Maximum separation between psl minimum and zg maximum, in degrees (default: 1.0).
    merge_dist : float
        Minimum allowable distance between two candidates, in degrees (default: 6.0).
    traj_range : float
        Maximum travel distance for a cyclone in 6h, in degrees (default: 8.0).
    traj_min_length : int
        Minimum cyclone lifetime, in 6h-intervals (default: 10).
    traj_max_gap : int
        Maximum allowable gap in cyclone trajectory, in 6h-intervals (default: 3).
    min_wind : float
        Minimum 10 m wind speed for TCs detection, in m/s (default: 10.0).
    min_len : int
        Minimum track length, in 6h-intervals (default: 10).
    max_lat : float
        Maximum latitude of psl minimum, in degrees (default: 50.0).

    Returns
    -------
    tracks_path : str
        Path to the TempestExtremes output .txt file with TC tracks.
    """

    # Prepare data to be used as input for TempestExtremes
    data_name = ("-").join(data_name.split())
    output_path = Path(output_path)

    data_tempestExtremes = prepare_data_tempestExtremes(data, data_name)
    data_tempestExtremes_path = output_path / f"{data_name}_tcs_tempestExtremes_input.nc"
    data_tempestExtremes.to_netcdf(data_tempestExtremes_path)

    # Run TempestExtremes to identify TCs tracks
    tracks_path = track_tcs(
        data_name,
        data_tempestExtremes_path,
        output_path,
        hist=tracks_hist,
        psl_delta=psl_delta,
        psl_dist=psl_dist,
        z_delta=z_delta,
        z_dist=z_dist,
        z_offset=z_offset,
        merge_dist=merge_dist,
        traj_range=traj_range,
        traj_min_length=traj_min_length,
        traj_max_gap=traj_max_gap,
        min_wind=min_wind,
        min_len=min_len,
        max_lat=max_lat,
    )
    data_tempestExtremes_path.unlink(missing_ok=True)

    return tracks_path


def read_tracks_tempestExtremes(tracks_path):
    """
    Read TC trajectories from TempestExtremes output file.

    Parameters
    ----------
    tracks_path : str
        Path to the TempestExtremes output .txt file.

    Returns
    -------
    tracks : list[list[dict]]
        List of trajectories.
    """

    if not Path(tracks_path).exists():
        raise FileNotFoundError(f"Tracks file '{tracks_path}' does not exist.")

    tracks = []
    with open(tracks_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("start"):
                parts = line.strip().split()
                num_points = int(parts[1])
                trajectory = []

                for _ in range(num_points):
                    fields = f.readline().strip().split()
                    point = {
                        "lon": float(fields[2]),
                        "lat": float(fields[3]),
                        "psl": float(fields[4]),
                        "wind": float(fields[5]),
                        "phis": float(fields[6]),
                        "year": int(fields[7]),
                        "month": int(fields[8]),
                        "day": int(fields[9]),
                        "hour": int(fields[10]),
                    }
                    trajectory.append(point)
                tracks.append(trajectory)

    return tracks


def compute_density_histogram(points, lat_bins, lon_bins, name="point_density"):
    """
    Compute density histogram for given points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) with (lat, lon) points.
    lat_bins : np.ndarray
        Latitude bin edges.
    lon_bins : np.ndarray
        Longitude bin edges.
    name : str
        Name of the resulting DataArray (default: "point_density").

    Returns
    -------
    hist_dens : xarray.DataArray
        2D histogram of point densities.
    """

    hist, lat_edges, lon_edges = np.histogram2d(
        [pt[0] for pt in points], [pt[1] for pt in points], bins=[lat_bins, lon_bins]
    )
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    hist_dens = xr.DataArray(
        hist,
        coords={"lat": lat_centers, "lon": lon_centers},
        dims=["lat", "lon"],
        name=name,
        attrs={"units": "count"},
    )

    return hist_dens


def compute_tc_counts(tracks, start_year, end_year, bin_size=2.5, cutoff_wind=10.0):
    """
    Compute TC genesis and tracks density for each grid box (bin_size x bin_size)
    from a list of TC trajectories.

    Parameters
    ----------
    tracks : list[list[dict]]
        List of trajectories.
    start_year, end_year : int
        Initial and final years to compute the densities.
    bin_size : float
        Size of the bins in degrees (default: 2.5).
    cutoff_wind : float
        Minimum wind speed in m/s to consider a TC genesis (default: 10.0).

    Returns
    -------
    counts_gen : xarray.DataArray
        TC genesis density.
    counts_traj : xarray.DataArray
        TC tracks density.
    """

    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)

    # Create (lat, lon) list of TC positions
    genesis_pts = []
    traj_pts = []

    for track in tracks:
        # Check time range
        first_point = track[0]
        t = pd.Timestamp(
            year=first_point["year"],
            month=first_point["month"],
            day=first_point["day"],
            hour=first_point["hour"],
        )

        # Save genesis and tracks locations
        if start_date <= t <= end_date:
            genesis_found = False
            for point in track:
                lat = point["lat"]
                lon = point["lon"]
                traj_pts.append([lat, lon])

                if not genesis_found and point["wind"] > cutoff_wind:
                    genesis_pts.append([lat, lon])
                    genesis_found = True


    # Group TC positions in bins and save to xarray.DataArray
    lat_bins = np.arange(-90, 90 + bin_size, bin_size)  # from -60 to 60 (inclusive)
    lon_bins = np.arange(0, 360 + bin_size, bin_size)  # 0 to 360 (if using 0-360 format)

    counts_gen = compute_density_histogram(genesis_pts, lat_bins, lon_bins, "genesis_density")
    counts_traj = compute_density_histogram(traj_pts, lat_bins, lon_bins, "track_density")

    return counts_gen, counts_traj


def filter_tracks_by_wind(tracks_path, filtered_tracks_path, cutoff_wind=10.0):
    """
    Filter TC trajectories from TempestExtremes output file, keeping only those with
    at least one point exceeding the specified 10 m wind speed.

    Parameters
    ----------
    tracks_file : str
        Path to the TempestExtremes output .txt file.
    filtered_tracks_path : str
        Path to save the filtered tracks .txt file.
    cutoff_wind : float
        Minimum 10 m wind speed in m/s to consider a TC track (default: 10.0).
    """

    # Prepare files
    if not Path(tracks_path).exists():
        raise FileNotFoundError(f"Tracks file '{tracks_path}' does not exist.")
    filtered_tracks_path = Path(filtered_tracks_path)
    filtered_tracks_path.parent.mkdir(parents=True, exist_ok=True)

    # Read and filter tracks
    filtered_tracks = []
    with open(tracks_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("start"):
                parts = line.strip().split()
                num_points = int(parts[1])
                track_lines = [line]
                wind_exceeds_cutoff = False

                # Read and store all points
                for _ in range(num_points):
                    point_line = f.readline()
                    track_lines.append(point_line)
                    fields = point_line.strip().split()
                    wind = float(fields[5])
                    if wind >= cutoff_wind:
                        wind_exceeds_cutoff = True

                # Write track if it contains at least one point above wind threshold
                if wind_exceeds_cutoff:
                    filtered_tracks.extend(track_lines)

    # Write filtered tracks to output file
    with open(filtered_tracks_path, "w") as f:
        for line in filtered_tracks:
            f.write(line)

    return
