import xesmf as xe
import xarray as xr

from cartopy.util import add_cyclic_point


def prepare_data(catalogue_path):
    """ Load data from catalogue interface."""
    raise NotImplementedError("This function is not implemented yet.")


def check_data(data):
    """ Check provided data (available variables, units, ...). """
    raise NotImplementedError("This function is not implemented yet.")


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


        source_ds_copy = source_ds.copy()

        # Select first timestep (we are only performing spatial regridding) and handle cyclic longitudes if needed
        if cyclic_point:
            source_ds_copy = source_ds_copy.map(cyclic_wrapper, keep_attrs=True)
        source_ds_t0 = source_ds_copy.isel({time_dim: 0}) if time_dim in source_ds_copy.dims else source_ds_copy
        target_ds_t0 = target_ds.isel({time_dim: 0}) if time_dim in target_ds.dims else target_ds
        
        # Build regridder and perform interpolation for all time steps
        with xe.Regridder(source_ds_t0, target_ds_t0, method) as regridder:
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