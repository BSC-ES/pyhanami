import pint
import numpy as np
import xesmf as xe
import xarray as xr

from pathlib import Path
from pyhanami import config
from cartopy.util import add_cyclic_point


def prepare_data(data_path, **xr_kwargs):
    """ 
    Load climate data from netcdf file or catalogue interface.
    
    Parameters
    ----------
    data_path (str): Path to the data file or catalogue interface
    **xr_kwargs (dict): Additional keyword arguments to pass to `xarray.open_dataset`

    Returns
    -------
    data (xr.Dataset): Loaded dataset.
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


def normalize_units(unit_str):
    """ 
    Normalize format of units to be compatible with pint.UnitRegistry. 

    Parameters
    ----------
    unit_str (str): Input units.

    Returns
    -------
    new_units (str): Units with corrected format (e.g. from 'm s-1' to 'm s**-1').
    """

    # Validate input
    if not isinstance(unit_str, str):    
        raise TypeError("Input must be a string.")


    # Look for incompatible exponent format and correct it
    parts = unit_str.split()
    new_parts = []
    for part in parts:
        new_part = []
        for i, c in enumerate(part):
            if ((c == '-' and part[i-1] not in {'^', '*'}) or (c.isdigit() and part[i-1] not in {'-', '^', '*'})):
                new_part.append(f'**{c}')
            else:
                new_part.append(c)
        new_parts.append(''.join(new_part))

    new_units = ' '.join(new_parts)
    return new_units


def check_data(data):
    """ 
    Check provided data (available variables, units, coordinates names, ...). 

    Parameters
    ----------
    data (xr.Dataset): Input dataset to check.
    """

    # Validate input
    if not isinstance(data, xr.Dataset):    
        raise TypeError("Input must be an xarray.Dataset.")


    # Check coordinates
    required_coords = ['time', 'lat', 'lon']
    missing_coords = [c for c in required_coords if c not in data.coords]
    if missing_coords:
        raise ValueError(f"The dataset is missing the following coordinates: {', '.join(missing_coords)}.")
    
    for coord in ['lat', 'lon']:
        coord_values = data[coord].values
        if coord_values.ndim != 1:
            raise ValueError(f"'{coord}' coordinate must be a 1D array.")
        coord_diffs = np.diff(coord_values)
        if not (np.all(coord_diffs > 0) or np.all(coord_diffs < 0)):
            raise ValueError(f"'{coord}' coordinate must be strictly increasing or decreasing.")


    # Check each variable and its units    
    variables = config.VARIABLES
    ureg = pint.UnitRegistry()
    for var, da in data.data_vars.items():
        if var not in variables:
            raise ValueError(f"Variable '{var}' not found in 'config.VARIABLES'.")
        
        expected_long_name, expected_units = variables[var]
        var_attrs = da.attrs

        if 'long_name' not in var_attrs or var_attrs['long_name'] != expected_long_name:
            var_attrs['long_name'] = expected_long_name

        if 'units' not in var_attrs:
                raise ValueError(f"Variable '{var}' is missing a 'units' attribute.")
        
        try:
            var_unit = ureg.Unit(normalize_units(var_attrs['units']))
            expected_unit = ureg.Unit(expected_units)
        except Exception as e:
            raise ValueError(f"Variable '{var}' has unrecognized units: '{var_attrs['units']}'. Error: {e}")
        if var_unit != expected_unit:
            raise ValueError(f"Variable '{var}' has incorrect units: '{var_attrs['units']}' "
                             f"(expected '{expected_units}').")
        else:
            var_attrs['units'] = expected_units
            
    print("Data check passed: all variables and coordinates are valid.", flush=True)
    return


def cyclic_wrapper(data, dim="lon"):
    """
    Add a cyclic (wrap-around) point along the specified dimension.

    Parameters
    ----------
    data (xarray.DataArray): Input data array.
    dim (str): Dimension along which to add the cyclic point.

    Returns
    -------
    wrapped_data (xarray.DataArray): Data array with cyclic point added.
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
        source_ds (xarray.Dataset): Source dataset.
        target_ds (xarray.Dataset): Target dataset.
        var (str): Variable to regrid.
        method (str): Regridding method.
        cyclic_point (bool): Whether to handle cyclic points.
        time_dim (str): Name of the time dimension.

        Returns
        -------
        regridded_ds (xarray.Dataset): Regridded dataset.
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