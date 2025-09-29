import numpy as np
import xarray as xr
import pandas as pd

from pyhanami.utils import data_general
from pyhanami.config import config_params


class DataChecker:
    """
    Validates and corrects datasets with climate variables.
    
    This class checks the standard compliance, spatial and temporal completeness, physical
    plausibility, and consistency of variables and coordinates in a dataset with climate 
    variables. It collects all encountered errors and warnings during the checking process 
    and prints a summary at the end. 

    Attributes
    ----------
    error_msg : list[str]
        List of error messages encountered during data checking.
    warning_msg : list[str]
        List of warning messages encountered during data checking.
    variables : dict
        Dictionary of variables metadata loaded from the 'variables.yaml' configuration file.

    Methods
    -------
    normalize_units(unit_str)
        Normalize format of units to be compatible with pint.UnitRegistry. Not used anymore, but kept for reference.
    normalize_time_format(data)
        Checks datetime format of the provided data and converts to 'np.datetime64[ns]' if needed.
    check_standard_compliance(data)
        Checks availability and format of coordinates and metadata of the given dataset.
    check_spatial_completeness(data)
        Checks for the lack (or presence) of NaN values for atmosphere (or ocean) variables.
    check_spatial_consistency(data)
        Checks grid features of the provided data. Not implemented yet, but left for reference.
    check_temporal_completeness(data)
        Checks presence of all timesteps between the minimum and maximum time values in the given dataset.
    check_physical_plausibility(data)
        Checks that each variable in the provided dataset is within an established physically reasonable range of values.
    check_data(data)
        Runs all checks and, if necessary, correct provided data (coordinates names and format, available variables, units, ...). 
    """

    def __init__(self):
        self.error_msg = []
        self.warning_msg = []
        self.variables = data_general.load_yaml_file(config_params.VARIABLES_PATH)


    @staticmethod
    def normalize_units(unit_str):
        """ 
        Normalize format of units to be compatible with pint.UnitRegistry. 
        Not used anymore, but kept for reference.

        Parameters
        ----------
        unit_str (str): Input units.

        Returns
        -------
        new_units (str): Units with corrected format (e.g. from 'm s-1' to 'm s**-1').
        """

        # Validate input
        if not isinstance(unit_str, str):    
            raise TypeError("Input must be a string for units.")

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


    @staticmethod
    def normalize_time_format(data):
        """ 
        Check datetime format of the provided data and convert to 'np.datetime64[ns]' if needed. 

        Parameters
        ----------
        data (xr.Dataset): Input dataset to check.

        Returns
        -------
        data (xr.Dataset): Checked dataset.
        error (lsit[str]): List of error messages encountered during datetime format normalization.
        warning (lsit[str]): List of warning messages encountered during datetime format normalization.
        """

        time = data['time']
        time_type = type(time.values[0])

        errors = []
        warnings = []

        try:
            # Check calendar type
            if not np.issubdtype(time_type, np.datetime64):
                datetimeindex = data.indexes['time'].to_datetimeindex(time_unit='ns')
                data = data.assign_coords(time=("time", datetimeindex.values))
                warnings.append(
                    f"Data 'time' coordinate was not in 'np.datetime64' format but '{time_type}' instead. " 
                    f" It has been converted automatically but better to provide it in the correct format from the beginning."
                )
            else:
                # Check if the data frequency is daily or coarser
                idxs = data.indexes['time'].values
                diffs = np.diff(idxs).astype('timedelta64[m]')
                daily_or_coarser = np.all((diffs.astype(int) % (24*60)) == 0)

                if daily_or_coarser: 
                    # Change 'hh:mm:ss' to midnight if needed           
                    idxs_floor = idxs.astype('datetime64[D]').astype('datetime64[ns]')
                    already_midnight = np.all(idxs == idxs_floor)

                    if not already_midnight:
                        data = data.assign_coords(time=idxs_floor)
                        warnings.append(
                            f"Data 'time' coordinate was not in 'YYYY-MM-DDT00:00:00' format (hours were not set to midnight). " 
                            f" It has been changed automatically but better to provide it in the correct format from the beginning."
                        )
        except Exception as e:
            errors.append(f'Error normalizing time format: {e}.')

        return data, errors, warnings


    def check_standard_compliance(self, data):
        """ 
        Check availability and format of coordinates and metadata of the given dataset. 

        Parameters
        ----------
        data (xr.Dataset): Input dataset to check.

        Returns
        ----------
        data (xr.Dataset): Checked and corrected dataset.
        """

        # Check coordinates
        required_coords = ['time', 'lat', 'lon']
        missing_coords = [c for c in required_coords if c not in data.coords]
        if missing_coords:
            self.error_msg.append(
                f"The dataset is missing the following coordinates: {', '.join(missing_coords)}. "
            )

            # Critical error: print and raise immediately
            error_message = f"{len(self.error_msg)} errors encountered while checking the provided dataset:\n"
            error_message += "\n".join(f'\t- {error}' for error in self.error_msg)
            error_message += "\nData check failed due to errors listed above. Please, correct the dataset before proceeding."
            raise RuntimeError(error_message)


        # Continue checking if all coordinates are available
        for coord in ['lat', 'lon']:
            coord_values = data[coord].values
            if coord_values.ndim != 1:
                self.error_msg.append(
                    f"'{coord}' coordinate must be a 1D array. " 
                )
            coord_diffs = np.diff(coord_values)
            if not (np.all(coord_diffs > 0) or np.all(coord_diffs < 0)):
                self.error_msg.append(
                    f"'{coord}' coordinate must be strictly increasing or decreasing. " 
                )

        data, errors, warnings = self.normalize_time_format(data)
        self.error_msg = self.error_msg + errors
        self.warning_msg = self.warning_msg + warnings
        
        # Check each variable and its units   
        # ureg = pint.UnitRegistry()
        for var in data.data_vars:
            if var not in self.variables:
                self.error_msg.append(
                    f"Variable '{var}' not found in  'variables.yaml'. " 
                )

            else: 
                expected_long_name = self.variables[var]['long_name']
                expected_units = self.variables[var]['units']
                var_attrs = data[var].attrs

                if 'long_name' not in var_attrs or var_attrs['long_name'] != expected_long_name:
                    data[var].attrs['long_name'] = expected_long_name
                    self.warning_msg.append(
                        f"Variable '{var}' was missing the corresponding 'long_name' attribute: '{expected_long_name}'. " 
                        f" It has been added automatically but better to provide it in the correct format from the beginning."
                    )
                
                # Check units and only accept if they are the same as the expected_units (also same format)
                if 'units' not in var_attrs:
                    self.error_msg.append(
                        f"Variable '{var}' is missing a 'units' attribute. "
                    )

                elif var_attrs['units'] != expected_units:
                    self.error_msg.append(
                        f"Variable '{var}' has incorrect or incompatible units: '{var_attrs['units']}' (expected '{expected_units}'). "
                    )

                # Check units and convert them to the expected_units if possible
                # try:
                #     quantity = ureg.Quantity(data[var].values, var_attrs['units'])
                #     converted_values = quantity.to(expected_units).magnitude
                #     data[var].values = converted_values
                #     data[var].attrs['units'] = expected_units
                # except Exception as e:
                #     self.error_msg.append(
                #          f"Variable '{var}' has incorrect or incompatible units: '{var_attrs['units']}' " 
                #          f"(expected '{expected_units}'). Error: {e}"
                #     )

        return data


    def check_spatial_completeness(self, data, land_mask=None):
        """ 
        Check for the lack (or presence) of NaN values for atmosphere
        (or ocean) variables. 

        Parameters
        ----------
        data (xr.Dataset): Input dataset to check.
        land_mask (xr.Dataset): Land mask to check ocean variables.

        Returns
        ----------
        data (xr.Dataset): Checked and corrected dataset.
        """

        all_nan = False
        for var in data.data_vars:
            if var in self.variables:

                data_values = data[var].values
                if np.isnan(data_values).all():
                    self.error_msg.append(
                        f"Variable '{var}' only contains NaN values." 
                    )
                    all_nan = True

                else:
                    mask = self.variables[var]['mask']
                    if mask == 'atm':
                        if np.isnan(data_values).any():
                            self.error_msg.append(
                                f"Variable '{var}' contains NaN values. " 
                                "Not acceptable for an atmosphere variable." 
                            )
                    elif mask == 'oce':
                        if not np.isnan(data_values).any():
                            self.error_msg.append(
                                f"Variable '{var}' does not contain any NaN values. " 
                                "Not acceptable for an ocean variable." 
                            )
                    else:
                        self.error_msg.append(
                            f"Unrecognizable mask for variable '{var}' in 'variables.yaml'. "
                            "Please, check the variables metadata."
                        )

        if all_nan:
            # Critical error: print and raise immediately
            error_message = f"{len(self.error_msg)} errors encountered while checking the provided dataset:\n"
            error_message += "\n".join(f'\t- {error}' for error in self.error_msg)
            error_message += "\nData check failed due to errors listed above. Please, correct the dataset before proceeding."
            raise RuntimeError(error_message)

        return data


    def check_spatial_consistency(self, data):
        """ 
        Check grid features of the provided data. 
        Not implemented yet, but left for reference.

        Parameters
        ----------
        data (xr.Dataset): Input dataset to check.

        Returns
        ----------
        data (xr.Dataset): Checked and corrected dataset.
        """

        return data


    def check_temporal_completeness(self, data):
        """ 
        Check presence of all timesteps between the minimum and maximum 
        time values in the given dataset.

        Parameters
        ----------
        data (xr.Dataset): Input dataset to check.
        """

        time = data['time'].values
        time_sorted = np.sort(time)

        # Infer time frequency
        time_index = pd.to_datetime(time_sorted)
        inferred_freq = pd.infer_freq(time_index[:3])
        if inferred_freq is None:
            self.error_msg.append(
                "Could not infer the frequency of the dataset from the first 3 timesteps. Please, check the time coordinate."
            )

        else:
            expected_times = pd.date_range(start=time_index[0], end=time_index[-1], freq=inferred_freq)
            missing_times = expected_times.difference(time_index)
            if len(missing_times) != 0:
                self.error_msg.append(
                    f"Missing {len(missing_times)} timesteps between {time_index[0]} and {time_index[-1]}. Missing values: "
                    f"{','.join(missing_times.strftime('%Y-%m-%d %H:%M:%S').tolist())}"
                )

        return data


    def check_physical_plausibility(self, data):
        """ 
        Check that each variable in the provided dataset is within an established 
        physically reasonable range of values.

        Parameters
        ----------
        data (xr.Dataset): Input dataset to check.

        Returns
        ----------
        data (xr.Dataset): Checked and corrected dataset.
        """

        for var in data.data_vars:
            if var in self.variables:
                expected_var = self.variables[var]
                if expected_var['check_enabled']:
                    data_values = data[var].values

                    if 'max' in expected_var:
                        max_ref = expected_var['max']
                        max_value = np.nanmax(data_values)

                        if max_ref < max_value:
                            self.error_msg.append(
                                f"Physically unlikely value for variable '{var}': {max_value}. "
                                f"Greater than upper bound {max_ref}."
                            )

                    if 'min' in expected_var:
                        min_ref = expected_var['min']
                        min_value = np.nanmin(data_values)

                        if min_ref > min_value:
                            self.error_msg.append(
                                f"Physically unlikely value for variable '{var}': {min_value}.  "
                                f"Smaller than lower bound {min_ref}."
                            ) 

                    if 'boundaries' in expected_var:
                        if expected_var['boundaries'] is not None:
                            self.error_msg.append(
                                f"Variable '{var}' has non-null 'boundaries' in 'variables.yaml'. "
                                f"No physical plausibility check has been implemented for this case yet."
                            )
                        
                else:
                    self.warning_msg.append(
                        f"Skipping physical plausibility check on variable '{var}' (check_enabled: False)."
                    )

        return data


    def check_data(self, data):
        """ 
        Run all checks and, if necessary, correct provided data (coordinates names and format, 
        available variables, units, ...). 

        Parameters
        ----------
        data (xr.Dataset): Input dataset to check.

        Returns
        ----------
        data (xr.Dataset): Checked and corrected dataset.
        """

        # Validate input
        if not isinstance(data, xr.Dataset):    
            raise TypeError("Input must be an xarray.Dataiker.gonzalez@bsc.esset.")


        # Check data
        data = self.check_standard_compliance(data)
        data = self.check_spatial_completeness(data)
        data = self.check_spatial_consistency(data) # Not implemented yet
        data = self.check_temporal_completeness(data)
        data = self.check_physical_plausibility(data)
        
        # Summarize outcome of data check
        if len(self.warning_msg) != 0:
            print(f"{len(self.warning_msg)} warnings encountered while checking the provided dataset:", flush=True)
            for warning in self.warning_msg:
                print(f'\t- {warning}', flush=True)

        if len(self.error_msg) == 0:
            print("Data check passed: all variables and coordinates are valid.", flush=True)
        else:        
            error_message = f"{len(self.error_msg)} errors encountered while checking the provided dataset:\n"
            error_message += "\n".join(f'\t- {error}' for error in self.error_msg)
            error_message += "\nData check failed due to errors listed above. Please, correct the dataset before proceeding."
            raise RuntimeError(error_message)

        return data


