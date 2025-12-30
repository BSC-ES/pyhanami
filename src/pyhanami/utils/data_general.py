import warnings
warnings.simplefilter("always")

import yaml
import xesmf as xe
import xarray as xr

from pathlib import Path
from cartopy.util import add_cyclic_point


def load_yaml_file(yaml_path):
    """
    Load data from a .yaml file into a dictionary.

    Parameters
    ----------
    yaml_path : str
        Path to the file.

    Returns
    -------
    data : dict
        Content of the file in a dictionary form.
    """

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Data path {yaml_path} not found.")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return data


def prepare_data(data_path, **xr_kwargs):
    """ 
    Load climate data from netcdf file or catalogue interface and
    return it as a dask array.
    
    Parameters
    ----------
    data_path : str
        Path to the data file or catalogue interface
    **xr_kwargs : dict
        Additional keyword arguments to pass to `xarray.open_dataset`.

    Returns
    -------
    data : xr.Dataset
        Loaded dataset.
    """

    # Validate input
    if not isinstance(data_path, (str, Path)):
        raise TypeError("The provided path must be a string or Path object.")
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} not found.")


    # Handle netCDF files
    if data_path.suffix in ['.nc', '.nc4', '.netcdf']:
        data = xr.open_dataset(data_path, chunks="auto", **xr_kwargs)
    else:
        raise NotImplementedError("This function is not implemented for intake catalogues yet.")

    return data


def cyclic_wrapper(data, dim="lon"):
    """
    Add a cyclic (wrap-around) point along the specified dimension.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array.
    dim : str
        Dimension along which to add the cyclic point (default: "lon").

    Returns
    -------
    wrapped_data : xarray.DataArray
        Data array with cyclic point added.
    """

    # Validate inputs
    if not isinstance(data, xr.DataArray):
        raise TypeError("Input must be an xarray.DataArray.")
    if dim not in data.dims:
        raise ValueError(f"Dimension '{dim}' not found in data dimensions: {list(data.dims)}.")


    # Apply cartopy's cyclic point function
    axis = data.get_axis_num(dim)
    wrap_data, wrap_coord = add_cyclic_point(data.values, coord=data[dim].values, axis=axis)
    
    # Rebuild the DataArray with the new coordinates and dimensions
    coords = {k: (wrap_coord if k == dim else v) for k, v in data.coords.items()}
    wrapped_data = xr.DataArray(
        wrap_data,
        coords=coords,
        dims=data.dims,
        attrs=data.attrs,
        name=data.name
    )

    return wrapped_data


def regrid_data(source_ds, target_ds, var=None, method='conservative', cyclic_point=False, time_dim='time' ):
        """
        Regrid one or all variables from the source dataset to the target dataset.

        Parameters
        ----------
        source_ds : xarray.Dataset
            Source dataset.
        target_ds : xarray.Dataset
            Target dataset.
        var : str
            Variable to regrid.
        method : str
            Regridding method (default: 'conservative').
        cyclic_point : bool
            Whether to handle cyclic points (default: False).
        time_dim : str
            Name of the time dimension (default: 'time').

        Returns
        -------
        regridded_ds : xarray.Dataset
            Regridded dataset.
        """

        # Validate inputs
        if not isinstance(source_ds, xr.Dataset):
            raise TypeError("The source dataset must be an xarray.Dataset.")
        if not isinstance(target_ds, xr.Dataset):
            raise TypeError("The target dataset must be an xarray.Dataset.")
        
        if var is None:
            vars_to_regrid = list(source_ds.data_vars.keys())
            if not vars_to_regrid:
                raise ValueError("No data variables found in source dataset.")
        else:
            if var not in source_ds.data_vars:
                raise ValueError(f"Variable '{var}' not found in source dataset.")
            vars_to_regrid = [var]

        if not isinstance(method, str):
            raise TypeError("Regridding method must be a string.")
        if not isinstance(cyclic_point, bool):
            raise TypeError("'cyclic_point' must be a boolean.")  
        if not isinstance(time_dim, str):
            raise TypeError("'time_dim' must be a string representing the time dimension name.")
        

        source_ds_copy = source_ds.copy()

        # Select first timestep (we are only performing spatial regridding) and handle cyclic longitudes if needed
        if cyclic_point:
            source_ds_copy = source_ds_copy.map(cyclic_wrapper, keep_attrs=True)
        source_ds_t0 = source_ds_copy.isel({time_dim: 0}, drop=True) if time_dim in source_ds_copy.dims else source_ds_copy
        target_ds_t0 = target_ds.isel({time_dim: 0}, drop=True) if time_dim in target_ds.dims else target_ds

        # Drop time coordinate to avoid conflicts during regridding
        source_ds_t0 = source_ds_t0.drop_vars(time_dim, errors='ignore')
        target_ds_t0 = target_ds_t0.drop_vars(time_dim, errors='ignore')
        
        # Build regridder and perform interpolation for all time steps
        regridder = xe.Regridder(source_ds_t0, target_ds_t0, method)
        regridded_vars = {}
        for var_name in vars_to_regrid:
            regridded_var = regridder(source_ds_copy[var_name])
            regridded_var.attrs = source_ds[var_name].attrs
            regridded_vars[var_name] = regridded_var
        regridded_ds = xr.Dataset(regridded_vars)
        
        # Preserve both global and coordinate attributes
        for coord_name in source_ds.coords:
            if coord_name in regridded_ds.coords:
                regridded_ds[coord_name].attrs = source_ds[coord_name].attrs
        regridded_ds.attrs = source_ds.attrs


        del source_ds_copy

        return regridded_ds


def validate_year_range(data_sim, start_year=None, end_year=None, process_name=None):
    """
    Validate and adjust the provided year range against the dataset's time dimension.

    Parameters
    ----------
    data_sim : SimulationData
        Simulation dataset containing a time dimension.
    start_year : int, optional
        Start year for the analysis. If None, uses the dataset's start year.
    end_year : int, optional
        End year for the analysis. If None, uses the dataset's end year.
    process_name : str, optional
        Name of the process/analysis being performed.

    Returns
    -------
    start_year : int
        Validated start year.
    end_year : int
        Validated end year.
    """

    # Get available years in the dataset
    sim_name = data_sim.name
    if 'time' not in data_sim.data.dims:
        raise ValueError(f"In the {process_name} analysis, the simulation dataset does not contain a 'time' dimension.")
    
    years = data_sim.data.time.dt.year
    start_year_data = int(years.min())
    end_year_data = int(years.max())


    # Set default years if not provided
    if start_year is None:
        start_year = start_year_data
        print(f"\tAs no start year was provided for the {process_name} analysis, the first year available" +
              f" in the '{sim_name}' dataset ({start_year_data}) will be used.", flush=True)
    if end_year is None:
        end_year = end_year_data
        print(f"\tAs no end year was provided for the {process_name} analysis, the last year available" +
              f" in the '{sim_name}' dataset ({end_year_data}) will be used.", flush=True)
    
    # Validate year range
    if start_year > end_year:
        raise ValueError(f"For the {process_name} analysis, the start year ({start_year}) must be less than" +
                         f" the end year ({end_year}).")
    if start_year < start_year_data or end_year_data < end_year:
        raise ValueError(f"The year range for the {process_name} analysis ({start_year}-{end_year}) must be" + 
                         f" within the available simulation data range ({start_year_data}-{end_year_data}).")

    return start_year, end_year