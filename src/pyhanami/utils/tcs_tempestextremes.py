import shutil
import subprocess

# The criteria for detection is taken from (C.M. Zarzycki & P.A. Ullrich, 2017; https://doi.org/10.1002/2016GL071606):
# From Section 3.4 Sample Optimization: pslFOmag = 2 hPa, wcOffset = 1°, mergeDist = 6°, trajRange = 8°, trajMaxGap = 18 h,
# maxTopo = 1500 m, maxLat = 50°, minWind = 10 m s−1, pslFOdist = 5.5°, wcFOmag =− 6 m, wcFOdist = 6.5°, trajMinLen = 60 h
# NOTE: based on the criteria that they use at CEMA, the wind threshold is set higher (to 70 m/s).


def check_tempestextremes_installed():
    """
    Check if TempestExtremes is installed and available in the system PATH.
    Raises an error if not found.
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


def stitch_nodes(input_file, output_file="cyclones_trajectores.txt", traj_range=8.0, traj_min_length=10, traj_max_gap=3, 
                 min_wind=17.0, min_len=10, max_lat=50.0, format_str="i,j,lon,lat,slp,wind,phis"):
    """
    Wrapper for StitchNodes function from TempestExtremes to identify actual TC tracks from the detected nodes.

    Parameters
    ----------
    input_file (str): input .txt nodefile with the detected nodal features.
    output_file (str): output .txt file to write down a filtered list of TC candidates.
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


def histogram_nodes(input_file, output_file="cyclones_trajectores.nc", ilon_col=3, ilatcol=4):
    """
    Wrapper for HistogramNodes function from TempestExtremes to generate histogram of TC detections (easily 
    displayed with 'ncview output_file &).
    
    Parameters
    ----------
    input_file (str): input .txt file containing TC tracks.
    output_file (str): output .nc file to include histogram of TC detections (default: "cyclones_trajectores.nc").
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


def track_tcs(input_path, output_path, start_year=None, end_year=None):
    
    check_tempestextremes_installed()
    return NotImplementedError("This function is not yet implemented.")