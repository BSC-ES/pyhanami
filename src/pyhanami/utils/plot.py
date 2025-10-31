import cmocean
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.mpl.ticker as cticker

from scipy.stats import bootstrap
from pyhanami.utils import data_general
from pyhanami.config import config_params
from cartopy.util import add_cyclic_point
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm


def save_or_show_plot(plot_obj, output_path, plot_filename, plot_name):
    """
    Either saves a plot to file or displays it based on output path.
    
    Parameters
    ----------
    plot_obj (matplotlib.figure.Figure): The plot object to save or show.
    output_path (pathlib.Path or None): Directory to save the plot; if None, the plot is displayed.
    plot_filename (str): Base name for the plot file.
    plot_name (str): Name of the plot for display messages.
    """

    if output_path is None:
        plt.show()
        print(f"{plot_name} created and displayed.", flush=True)
    else:
        plot_path = output_path / f"{plot_filename}.png"
        plot_obj.savefig(plot_path, bbox_inches='tight', dpi=150)
        print(f"{plot_name} created and saved to '{plot_path}'.", flush=True)

    plt.close(plot_obj)
    return


def time_series_plot(time_series, title='Mean time series', y_label='', x_label='time', labels=None, 
                     time_freq='annual', start_year=None, end_year=None, plot_ens=False):
    """ 
    Generate time series plot of one or more ensembles, including the 2.5th, 5th, 75th and 97.5th percentiles.

    Parameters
    ----------
    time_series : xarray.DataArray or list[xr.DataArray]
        Time series data.
    title : str
        Title of the plot (default: 'Mean time series').
    y_label : str
        Label for the y-axis (default: '').
    labels : list[str]
        Labels for each time series.
    time_freq : str
        Time frequency (default: 'annual').
    start_year : int
        Start year for filtering.
    end_year : int
        End year for filtering.
    plot_ens : bool
        Whether to plot individual ensemble members trajectories (default: False).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        Plot axis.
    """

    # Validate input
    if isinstance(time_series, xr.DataArray):
        time_series = [time_series]
    elif isinstance(time_series, list):
        if len(time_series) == 0:
            raise ValueError("The time series cannot be empty.")
        for i, series in enumerate(time_series):
            if not isinstance(series, xr.DataArray):
                raise TypeError(f"Element {i} in time_series must be a xarray.DataArray.")
            if "time" not in series.dims:
                raise ValueError(f"Element {i} in time_series is missing 'time' dimension.")
    else:
        raise TypeError("The data must be either a xarray.DataArray or a list of xarray.DataArray.")
    
    if labels is not None:
        if not isinstance(labels, list) or len(labels) != len(time_series) \
            or not all(isinstance(label, str) for label in labels):
            raise ValueError("If provided, 'labels' must be a list of strings with the same length as time_series.")

    if start_year is None or end_year is None:
        raise TypeError("'start_year' and 'end_year' must be non-empty.")
   

    # Filter each time series to the requested dates range
    filtered_timeseries = []
    for series in time_series:
        series_filtered = series.sel(time=slice(str(start_year), str(end_year)))
        if series_filtered is not None:
            filtered_timeseries.append(series_filtered)
        else:
            raise ValueError(f"No data available in the selected range {start_year}-{end_year} for some datasets.")

    # Find common dates across all filtered time series
    # date_sets = [set(series["time"].dt.date.values) for series in filtered_timeseries]
    # common_dates = sorted(set.intersection(*date_sets))
    # if not common_dates:
    #     raise ValueError("No overlapping dates in the selected range across all time series.")
    # final_timeseries = [series.sel(time=series["time"].dt.date.isin(common_dates)) for series in filtered_timeseries]


    # Create plot
    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=150)
    labels = labels or [f'Series {i+1}' for i in range(len(filtered_timeseries))]

    for series, label in zip(filtered_timeseries, labels):
        dates = series['time'].values

        if 'realization' in series.dims and series.sizes['realization'] > 1:
            # Plot mean
            mean_series = series.mean('realization')
            line, = ax.plot(dates, mean_series.values, lw=2, ls='-.', label=label, zorder=4)
            color = line.get_color()

            # Plot individual ensemble members
            if plot_ens:
                for i in range(series.sizes['realization']):
                    ax.plot(dates, series.isel(realization=i).values, color=color, alpha=0.2, zorder=2)

            # Compute confidence intervals (bootstrap of mean)
            q025, q975 = [], []
            for i in range(len(dates)):
                data = series.isel(time=i).values
                bs_res = bootstrap((data,), np.mean, confidence_level=0.95)
                q025.append(bs_res.confidence_interval.low)
                q975.append(bs_res.confidence_interval.high)
            ax.fill_between(dates, q025, q975, facecolor=color, alpha=0.6, zorder=3)

            # Plot ensemble spread (5th–95th percentile)
            series = series.chunk({"realization": -1})
            quantiles = series.quantile([0.05, 0.95], dim='realization', skipna=True)
            ax.fill_between(dates, quantiles[0], quantiles[1], facecolor=color, alpha=0.2, zorder=1)

        else:
            ax.plot(dates, series.values, lw=2, ls='-.', label=label)


    # Set x-ticks and labels
    if time_freq == 'annual':
        unit = 'Y'
        step = 1
    elif time_freq == 'monthly':
        unit = 'M'
        step = 2
    elif time_freq == 'daily':
        unit = 'D'
        step = 2
    else:
        raise ValueError("Invalid 'time_freq'. Must be 'annual', 'monthly' or 'daily'.")
    
    x_ticks = [np.datetime64(str(year), unit) for year in np.arange(start_year, end_year+step, 1)]
    x_labels= [str(date) for date in x_ticks]
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # Plot formatting
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=14)
    ax.grid()

    plt.tight_layout()

    return fig, ax


def style_cartopy_axis(ax, show_gridlines=True, ocean_data=False, lw_coast=0.8, lw_borders=0.5, 
                       gl_lw=0.6, gl_fontsize=15):
    """
    Add standard geographic features and optional gridlines to a Cartopy axis.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Axis with a Cartopy geographic projection.
    show_gridlines : bool
        Whether to add gridlines with latitude and longitude labels (default: True).
    ocean_data : bool
        Whether only ocean data is provided and the land should be masked in white (default: False).

    """

    # Add land overlay (white fill, no edges) for ocean data
    if ocean_data:  
        ax.add_feature(cf.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=3)

    # General features
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=lw_coast, zorder=4)
    ax.add_feature(cf.BORDERS.with_scale("50m"), lw=lw_borders, zorder=4)

    # Gridlines
    if show_gridlines:
        gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=gl_lw, color='black', alpha=0.8, linestyle='-.')
        gl.xlabel_style = {"size": gl_fontsize}
        gl.ylabel_style = {"size": gl_fontsize}

    return


def add_colorbar(fig, mappable, ax_l, ax_r, ax_b, label='', fontsize=15, levels=None, dist=0.07, width=0.02, **colorbar_kwargs):
    """
    Create a horizontal colorbar below the given axes.

    Parameters
    ----------
    fig : (matplotlib.figure.Figure)
        Figure to add the colorbar to.
    mappable : (matplotlib artist)
        Object for colorbar.
    ax_l, ax_r, ax_b : (matplotlib.axes.Axes)
        Leftmost, rightmost, and bottom axes used to determine the bounds.
    label : (str)
        Colorbar label (default: '').
    fontsize : (int)
        Font size for label and tick labels (default: 15).
    levels : (np.ndarray)
        Contour levels for the colorbar ticks.
    dist : (float)
        Vertical distance below ax_b for placing the colorbar (default: 0.07).
    width : (float)
        Thickness of the colorbar axis (default: 0.02).
    **colorbar_kwargs : Additional arguments passed to fig.colorbar().

    Returns
    -------
    cb_ax : (matplotlib.colorbar.Colorbar)
        Colorbar.
    """

    left = ax_l.get_position().extents[0]
    right =  ax_r.get_position().extents[2]
    bottom = ax_b.get_position().extents[1]
            
    cb_ax = fig.add_axes([
        left,  # x position
        bottom-dist,  # y position
        right-left,  # x width
        width  # y width
    ])

    cbar = fig.colorbar(mappable, cax=cb_ax, orientation='horizontal', shrink=0.5, pad=0.05, aspect=40, **colorbar_kwargs)
    cb_ax.set_xlabel(label, fontsize=fontsize)

    if levels is not None:
        cbar.set_ticks(levels)
    cbar.ax.tick_params(labelsize=fontsize)

    return cbar


def spatial_plot(data, clon=0, title='Spatial plot', cb_label='', cmap=cmocean.cm.thermal, levels=None, significant=None,
                 vmin=None, vmax=None, show_contours=True, contour_fontsize=12, gridlines=True, **plot_kwargs):
    """ 
    Generate a spatial plot using Cartopy with a significance mask if selected.

    Parameters
    ----------
    data : xarray.DataArray
        2D dataset to plot with dimensions (lat, lon).
    clon : int
        Central longitude for the spatial map.
    title : str
        Title of the plot (default: 'Spatial plot').
    cb_label : str
        Label to display below the colorbar (default: ''). 
    cmap : matplotlib colormap
        Colormap (default: cmocean.cm.thermal).
    levels : np.ndarray
        Contour levels.
    significant : np.ndarray
        Mask for significance hatching.
    vmin, vmax : float
        Min. and max. values for the colormap.
    show_contours : bool
        Whether to overlay contour lines (default: True).
    contour_fontsize : int
        Font size for contour labels (default: 12).
    gridlines : bool
        Whether to show gridlines (default: True).
    **plot_kwargs : Additional arguments passed to contourf.

    Returns
    -------
    fig : (matplotlib.figure.Figure)
        Generated plot.
    ax : (matplotlib.axes._subplots.AxesSubplot)
        Plot axis (an array of two axes for 'siconc' and just one axis otherwise).
    """

    # Validate inputs
    if not isinstance(data, xr.DataArray):
        raise TypeError("The data must be a xarray.DataArray.")
    if 'lat' not in data.coords or 'lon' not in data.coords:
        raise ValueError("Could not identify latitude and longitude coordinates.")
    if not isinstance(clon, (int, float)) or not (0 <= clon <= 360):
        raise TypeError("The central longitude 'clon' must be a numeric value between 0º and 360º.")
    
    # Correct 0 and NaN values (values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0)
    var_name = data.name
    if var_name in {'siconc', 'sos', 'tos'}:
        data = xr.where((np.isnan(data)) | (data==0), 10**-10, data)
        ocean_data = True
    else:
        ocean_data = False   

    
    # Add cyclic point
    aux, lon = add_cyclic_point(data, coord=data.lon.values)
    data_cyclic = xr.DataArray(data=aux, dims=['lat', 'lon'], coords={'lat': data.lat.values, 'lon': lon}, name=data.name, attrs=data.attrs)


    if var_name != 'siconc':
        # Create figure
        fig, ax = plt.subplots(1, figsize=(20, 10), dpi=150, subplot_kw={'projection': ccrs.Robinson(central_longitude=clon), "aspect": 'auto'}, gridspec_kw = {'wspace':0.01, 'hspace':0.02})
        style_cartopy_axis(ax, show_gridlines=gridlines, ocean_data=ocean_data)
        cb = data_cyclic.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False, **plot_kwargs)

        # Add contour lines if requested
        if show_contours:
            contour_lines = data_cyclic.plot.contour(ax=ax, transform=ccrs.PlateCarree(), levels=levels, colors='black', linewidths=0.5)
            ax.clabel(contour_lines, fontsize=contour_fontsize, colors='black') 

        # Significance hatching
        if significant is not None:
            if significant.shape != data.shape:
                raise ValueError(f"Mask shape {significant.shape} does not match data shape {data.values.shape}.")
            else:
                aux, lon = add_cyclic_point(significant, coord=data.lon.values)
                mask = np.ma.masked_where(aux == 0, data_cyclic.values)  
                ax.pcolor(data_cyclic.lon.values, data_cyclic.lat.values, mask, transform=ccrs.PlateCarree(), hatch='..', zorder=1, alpha=0.)

        # Add title if given
        if title:
            ax.set_title(title, fontsize=20, pad=20)

        # Add colorbar if cb_label is given
        if cb_label:
            _ = add_colorbar(fig=fig, mappable=cb, ax_l=ax, ax_r=ax, ax_b=ax, label=cb_label, levels=levels)


    # Special Stereographic projection plot for sea ice concentration
    else:
        lat_limit = 40

        # Compute a circle in axes coordinates, to be used as a boundary for the map
        # (it allows to pan/zoom as much as needed, the boundary will be permanently circular)
        theta = np.linspace(0, 2*np.pi, 100)
        center =  [0.5, 0.5] 
        radius = 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)


        ### North pole ###
        proj = ccrs.Stereographic(central_latitude=90, central_longitude=0)
        crs = ccrs.PlateCarree()

        # Create figure
        fig, ax = plt.subplots(1,2, figsize=(10,5.5), dpi=150, subplot_kw={'projection': proj})

        # Customize gridlines
        ax1 = ax[0]
        ax1.set_extent([-180, 180, lat_limit, 90], crs=crs)

        gl1 = ax1.gridlines(draw_labels=True, linewidth=.4, color='gray', linestyle='-.')
        gl1.xlabel_style = {'size':8}
        gl1.ylabel_style = {'size':8}
        gl1.ylocator = plt.MultipleLocator(10)

        # Specify borders and coastlines
        ax1.add_feature(cf.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=3)
        ax1.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.4, zorder=4)
        ax1.add_feature(cf.BORDERS.with_scale("50m"), lw=0.2, zorder=4)

        ax1.set_boundary(circle, transform=ax1.transAxes)

        # Add contour lines if requested
        if show_contours:
            contour_lines = data_cyclic.plot.contour(ax=ax1, transform=ccrs.PlateCarree(), levels=levels, colors='black', linewidths=0.5)
            ax1.clabel(contour_lines, fontsize=contour_fontsize-5, colors='black')

        # Add data 
        cb = data_cyclic.plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False, **plot_kwargs)

        # Significance hatching
        if significant is not None:
            if significant.shape != data.shape:
                raise ValueError(f"Mask shape {significant.shape} does not match data shape {data.values.shape}.")
            else:
                aux, lon = add_cyclic_point(significant, coord=data.lon.values)
                mask = np.ma.masked_where(aux == 0, data_cyclic.values)  
                ax1.pcolor(data_cyclic.lon.values, data_cyclic.lat.values, mask, transform=ccrs.PlateCarree(), hatch='..', zorder=1, alpha=0.)


        ### South pole ###
        proj = ccrs.Stereographic(central_latitude=-90, central_longitude=0)   

        # Customize gridlines
        ax2 = ax[1]
        ax2.projection = proj
        ax2.set_extent([-180, 180, -90, -lat_limit], crs=crs)

        gl2 = ax2.gridlines(draw_labels=True, linewidth=.5, color='gray', linestyle='-.')
        gl2.xlabel_style = {'size':8}
        gl2.ylabel_style = {'size':8}
        gl2.ylocator = plt.MultipleLocator(10)

        # Specify borders and coastlines
        ax2.add_feature(cf.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=3)
        ax2.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.4, zorder=4)
        ax2.add_feature(cf.BORDERS.with_scale("50m"), lw=0.2, zorder=4)

        ax2.set_boundary(circle, transform=ax2.transAxes)

        # Add contour lines if requested
        if show_contours:
            contour_lines = data_cyclic.plot.contour(ax=ax2, transform=ccrs.PlateCarree(), levels=levels, colors='black', linewidths=0.5)
            ax2.clabel(contour_lines, fontsize=contour_fontsize-5, colors='black')

        # Add data 
        cb = data_cyclic.plot.contourf(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False, **plot_kwargs)

        # Significance hatching
        if significant is not None:
            if significant.shape != data.shape:
                raise ValueError(f"Mask shape {significant.shape} does not match data shape {data.values.shape}.")
            else:
                aux, lon = add_cyclic_point(significant, coord=data.lon.values)
                mask = np.ma.masked_where(aux == 0, data_cyclic.values)  
                ax2.pcolor(data_cyclic.lon.values, data_cyclic.lat.values, mask, transform=ccrs.PlateCarree(), hatch='..', zorder=1, alpha=0.)
                

        # Add titles
        plt.subplots_adjust(top=0.9) 
        ax1.set_title("North Pole", fontsize=10)
        ax2.set_title("South Pole", fontsize=10, pad=15)
        if title:
            fig.suptitle(title)

        # Add colorbar if cb_label is given
        if cb_label:
            _ = add_colorbar(fig=fig, mappable=cb, ax_l=ax[0], ax_r=ax[1], ax_b=ax[1], label=cb_label, fontsize=8, levels=levels, dist=0.09)

    return fig, ax


def two_spatial_plots(data_1, data_2, clon=0, title_1='Spatial plot 1', title_2='Spatial plot 2', suptitle='Spatial plots', 
                        cb_label='', cmap=cmocean.cm.thermal, levels=12, significant_1=None, significant_2=None, 
                        vmin=None, vmax=None, show_contours=True, contour_fontsize=6, gridlines=True, **plot_kwargs):
    """
    Generate two spatial plots side by side using Cartopy with significance masks if selected.

    Parameters
    ----------
    data_1 (xarray.DataArray): First 2D dataset to plot with dimensions (lat, lon).
    data_2 (xarray.DataArray): Second 2D dataset to plot with dimensions (lat, lon).
    clon (int): Central longitude for the spatial maps.
    title_1 (str): Title of the first plot (default: 'Spatial plot 1').
    title_2 (str): Title of the second plot (default: 'Spatial plot 2').
    suptitle (str): Common title for the two plots (default: 'Spatial plots').
    cb_label (str): Label to display below the common colorbar; set to False to not add a colorbar (default: ''). 
    cmap (matplotlib colormap): Colormap (default: cmocean.cm.thermal).
    levels (np.ndarray): Contour levels (default: 12).
    significant_1 (np.ndarray): Mask for significance hatching in the first plot.
    significant_2 (np.ndarray): Mask for significance hatching in the second plot.
    vmin, vmax (float): Common min. and max. values for the colormap.
    show_contours (bool): Whether to overlay contour lines (default: True).
    contour_fontsize (int): Font size for contour labels (default: 12).
    gridlines (bool): Whether to show gridlines (default: True).
    **plot_kwargs: Additional arguments passed to contourf.

    Returns
    -------
    new_fig (matplotlib.figure.Figure): Generated plot.
    ax (matplotlib.axes._subplots.AxesSubplot): Plot axis.
    """

    # Validate inputs
    if not isinstance(data_1, xr.DataArray) or not isinstance(data_2, xr.DataArray):
        raise TypeError("The data must be a xarray.DataArray.")
    if 'lat' not in data_1.coords or 'lon' not in data_1.coords or 'lat' not in data_2.coords or 'lon' not in data_2.coords:
        raise ValueError("Could not identify latitude and longitude coordinates.")
    if not isinstance(clon, (int, float)) or not (0 <= clon <= 360):
        raise TypeError("The central longitude 'clon' must be a numeric value between 0º and 360º.")


    # Tried reusing spatial_plot function but not working
    # # Create separate spatial plots
    # fig_1, ax_1 = spatial_plot(data_1, clon=clon, title=title_1, cb_label=False, cmap=cmap, levels=levels, significant=significant_1,
    #                            vmin=vmin, vmax=vmax, show_contours=show_contours, contour_fontsize=contour_fontsize, gridlines=gridlines, **plot_kwargs)
    # fig_2, ax_2 = spatial_plot(data_2, clon=clon, title=title_2, cb_label=False, cmap=cmap, levels=levels, significant=significant_2,
    #                            vmin=vmin, vmax=vmax, show_contours=show_contours, contour_fontsize=contour_fontsize, gridlines=gridlines, **plot_kwargs)

    # # Combine both plots into a single figure
    # old_figs = [fig_1, fig_2]
    # old_axs = [ax_1, ax_2]
    # for fig, ax in zip(old_figs, old_axs):
    #     fig.delaxes(ax)
    #     plt.close(fig)

    # new_fig = plt.figure(figsize=(12, 5))
    # axs = new_fig.axes
    # for i, ax in enumerate(old_axs):
    #     ax.set_figure(new_fig)
    #     new_fig.add_axes(ax)
    #     ax.change_geometry(1, 2, i+1)

    # new_fig.suptitle(suptitle, fontsize=14)

    # # Add common colorbar if cb_label is given
    # if cb_label:
    #     mappable = old_axs[0].collections[0]
    #     cbar = new_fig.colorbar(mappable, ax=axs, orientation='horizontal', pad=0.17)
    #     cbar.ax.tick_params(labelsize=8)
    #     cbar.set_label(cb_label, fontsize=9)


    # Create figure
    fig, axs = plt.subplots(1,2, figsize=(12, 4.5), dpi=150, subplot_kw={'projection': ccrs.Robinson(central_longitude=clon), "aspect": 'auto'})#, gridspec_kw = {'wspace':0.01, 'hspace':0.02})

    # Add first plot
    # Correct 0 and NaN values (values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0)
    var_name = data_1.name
    if var_name in {'siconc', 'sos', 'tos'}:
        data_1 = xr.where((np.isnan(data_1)) | (data_1==0), 10**-10, data_1)
        ocean_data = True
    else:
        ocean_data = False   

    # Add cyclic point
    aux, lon = add_cyclic_point(data_1, coord=data_1.lon.values)
    data_cyclic = xr.DataArray(data=aux, dims=['lat', 'lon'], coords={'lat': data_1.lat.values, 'lon': lon}, name=data_1.name, attrs=data_1.attrs)

    # Create figure 
    ax_1 = axs[0]
    style_cartopy_axis(ax_1, show_gridlines=gridlines, ocean_data=ocean_data, lw_coast=0.4, lw_borders=0.2, gl_lw=0.3, gl_fontsize=7)
    cb = data_cyclic.plot.contourf(ax=ax_1, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False, **plot_kwargs)

    # Add contour lines if requested
    if show_contours:
        contour_lines = data_cyclic.plot.contour(ax=ax_1, transform=ccrs.PlateCarree(), levels=levels, vmin=vmin, vmax=vmax, colors='black', linewidths=0.2)
        ax_1.clabel(contour_lines, fontsize=contour_fontsize, colors='black') 

    # Significance hatching
    if significant_1 is not None:
        if significant_1.shape != data_1.shape:
            raise ValueError(f"Mask shape {significant_1.shape} does not match data shape {data_1.values.shape}.")
        else:
            aux, lon = add_cyclic_point(significant_1, coord=data_1.lon.values)
            mask = np.ma.masked_where(aux == 0, data_cyclic.values)
            ax_1.pcolor(data_cyclic.lon.values, data_cyclic.lat.values, mask, transform=ccrs.PlateCarree(), hatch='..', zorder=1, alpha=0.)

    # Add title if given
    if title_1:
        ax_1.set_title(title_1, fontsize=12, pad=8)


    # Add second plot
    # Correct 0 and NaN values (values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0)
    var_name = data_2.name
    if var_name in {'siconc', 'sos', 'tos'}:
        data_2 = xr.where((np.isnan(data_2)) | (data_2==0), 10**-10, data_2)
        ocean_data = True
    else:
        ocean_data = False   

    # Add cyclic point
    aux, lon = add_cyclic_point(data_2, coord=data_2.lon.values)
    data_cyclic = xr.DataArray(data=aux, dims=['lat', 'lon'], coords={'lat': data_2.lat.values, 'lon': lon}, name=data_2.name, attrs=data_2.attrs)

    # Create figure
    ax_2 = axs[1]
    style_cartopy_axis(ax_2, show_gridlines=gridlines, ocean_data=ocean_data, lw_coast=0.4, lw_borders=0.2, gl_lw=0.3, gl_fontsize=7)
    cb = data_cyclic.plot.contourf(ax=ax_2, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False, **plot_kwargs)

    # Add contour lines if requested
    if show_contours:
        contour_lines = data_cyclic.plot.contour(ax=ax_2, transform=ccrs.PlateCarree(), levels=levels, vmin=vmin, vmax=vmax, colors='black', linewidths=0.2)
        ax_2.clabel(contour_lines, fontsize=contour_fontsize, colors='black')

    # Significance hatching
    if significant_2 is not None:
        if significant_2.shape != data_2.shape:
            raise ValueError(f"Mask shape {significant_2.shape} does not match data shape {data_2.values.shape}.")
        else:
            aux, lon = add_cyclic_point(significant_2, coord=data_2.lon.values)
            mask = np.ma.masked_where(aux == 0, data_cyclic.values)
            ax_2.pcolor(data_cyclic.lon.values, data_cyclic.lat.values, mask, transform=ccrs.PlateCarree(), hatch='..', zorder=1, alpha=0.)

    # Add title if given
    if title_2:
        ax_2.set_title(title_2, fontsize=12, pad=8)

    # Add shared colorbar if cb_label is given
    if cb_label:
        cbar = fig.colorbar(cb, ax=axs, orientation="horizontal", fraction=0.03, pad=0.1, aspect=60)
        cbar.set_ticks(cb.levels)
        
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(cb_label, fontsize=8)
        plt.tight_layout(rect=[0,0.15,1,0.96])
    else:
        plt.tight_layout(rect=[0,0.03,1,1])

    fig.suptitle(suptitle, fontsize=14)

    return fig, axs


def matrix_plot(eff_sizes, test_results, test=4, title='Effect sizes replicability test', variables=None, 
                seasons=None, regions=None):
    """ 
    Generate a matrix plot with effect sizes and results of the replicability test for the selected statistical test/s.

    Parameters
    ----------
    eff_sizes : np.ndarray
        Effect sizes between the scores with shape (n_rows, n_cols, n_indices). 
    test_results : np.ndarray
        Results of the statistical tests with shape (n_rows, n_cols, 4).
    test : int
        Statistical test to use, corresponding to last dimension of test_results (0: KS-test, 
        1: T-test, 2: U-test, 3: B-test, 4: All) (default: 4).
    title : str
        Title of the plot (default: 'Effect sizes replicability test').
    variables : dict
        Dictionary with variables to be included in the plot and their descriptions.
    seasons : list
        List of seasons to include in the plot.
    regions : list
        List of regions to include in the plot.

    Returns
    -------
    fig : (matplotlib.figure.Figure)
        Generated matrix plot.
    ax : (matplotlib.axes._subplots.AxesSubplot)
        Plot axis.
    """

    # Validate inputs
    if not isinstance(eff_sizes, np.ndarray) or not isinstance(test_results, np.ndarray):
        raise TypeError("The effect sizes and the test results must be numpy arrays.")
    if eff_sizes.shape[:2] != test_results.shape[:2]:
        raise ValueError("Mismatched spatial dimensions between effect sizes and test results.")
    if test not in range(5):
        raise ValueError("Invalid test index. Must be: 0 (KS-test), 1 (T-test), 2 (U-test), 3 (B-test), 4 (All).")

    # Prepare parameters
    if variables is None:
        variables = list(data_general.load_yaml_file(config_params.VARIABLES_PATH).keys())
    if seasons is None:
        seasons = config_params.SEASONS
    if regions is None:
        regions = list(config_params.REGIONS.keys())

    abs_eff_sizes = np.abs(eff_sizes)
    if test == 4:
        test_outcome = np.any(test_results, axis=2)
    else:
        test_outcome = test_results

    # Define custom colors
    green = 'tab:green' 
    red = 'tab:red' 

    limits_colorbar = np.array([0, 0.01, 0.2, 0.5, 0.8, 1.2, 2])
    blue_colors = LinearSegmentedColormap.from_list('', ['#ffffff', 'yellow'])
    orange_colors = LinearSegmentedColormap.from_list('', ['#ffffff', 'tab:blue'])
    cmap_aux = np.append(blue_colors([0, 0.2]),orange_colors(np.linspace(0.2, 1, 4)), axis=0)
    cmap = ListedColormap(cmap_aux)
    norm = BoundaryNorm(limits_colorbar, cmap.N)

    # Define the grid
    n_rows, n_cols, n_indices = abs_eff_sizes.shape
    fig_size = (22, 16)
    fig, ax = plt.subplots(figsize=fig_size)

    x_labels = [f'{s1} {s2}' for s1 in seasons for s2 in regions]
    y_labels = variables


    # Loop over each cell in the grid
    circle_radius = 0.13
    for i in range(n_rows):
        for j in range(n_cols):
            values = abs_eff_sizes[i, j, :]

            # Define the vertices in the cell
            x, y = j, i
            corners = np.array([[x, y], [x+1, y], [x+1, y+1], [x, y+1]])
            center = [(x + x+1)/2, (y + y+1)/2]
            
            # Create four triangles
            triangles = [
                [corners[3], center, corners[0]],  # Left
                [corners[0], center, corners[1]],  # Bottom
                [corners[1], center, corners[2]],  # Right
                [corners[2], center, corners[3]],  # Top
                ]

            # Plot each triangle with the corresponding p-value
            for k, triangle in enumerate(triangles):
                number = values[k]
                color = cmap(norm(number)) if ~np.isnan(number) else 'white'
                polygon = Polygon(triangle, color=color)
                ax.add_patch(polygon)

                # Add numeric p-value at the center of the triangle
                centroid = np.mean(triangle, axis=0)
                ax.text(centroid[0], centroid[1], f"{values[k]:.2f}", 
                        ha="center", va="center", fontsize=7, color="#283747")
                
            # Add inner edges of the triangles
            for corner in corners:
                ax.plot([center[0], corner[0]], [center[1], corner[1]], color="black", lw=0.2)


            # Check if any triangle value is less than alpha
            if test_outcome[i, j]==1:
                circle_color = red
            elif test_outcome[i, j]==0 and ~np.isnan(number):
                circle_color = green
            else:
                circle_color = 'white'

            # Add circle in the center of the cell
            circle = Circle(center, radius=circle_radius, facecolor=circle_color, edgecolor="black", lw=0.3, zorder=3)
            ax.add_patch(circle)


    # Add gridlines (cells' borders)
    for x in range(n_cols + 1):  
        ax.plot([x, x], [0, n_rows], color='black', linewidth=0.9)
    for y in range(n_rows + 1): 
        ax.plot([0, n_cols], [y, y], color='black', linewidth=0.9)

    # Customize axes
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')

    # Only ticks at the center of each column/row
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    plt.xticks(rotation=35, ha="right")
    # plt.yticks(rotation=30)

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()

    plt.xlabel('Period')
    plt.ylabel('Variables')
    plt.title(title, y=1.01)


    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, aspect=50, pad=0.02, ticks=limits_colorbar)
    cbar.set_label('Effect size')

    # Add label to each segment
    ranges = ['Very small', 'Small', 'Medium', 'Large', 'Very large', 'Huge']
    for i, loc in enumerate(limits_colorbar[:-1]):
        mid_loc = (limits_colorbar[i] + limits_colorbar[i + 1]) / 2
        cbar.ax.text(0.6, mid_loc, ranges[i], ha='center', va='center', color='black', rotation=90, fontsize=8)


    # Add box with Rejection/No rejection legend
    legend_ax = fig.add_axes([0.552, 0.02, 0.1, 0.01]) # [left, bottom, width, height]
    legend_ax.axis('off')  

    handles = [
        plt.Line2D([0], [0], color=green, lw=5, label='No rejection'),
        plt.Line2D([0], [0], color=red, lw=5, label='Rejection')
    ]
    legend_ax.legend(handles=handles, loc='center', ncol=2)


    # Add square with legend for scores
    square_size = 0.055
    inset_ax = fig.add_axes([0.856, 0.025, square_size*(fig_size[1]/fig_size[0]), square_size])  # [left, bottom, width, height] in figure coordinates
    inset_ax.set_xlim(0, 1)
    inset_ax.set_ylim(0, 1)
    inset_ax.axis("off")

    triangles = [
        [[0, 0], [0.5, 0.5], [1, 0]],  # Bottom 
        [[0, 1], [0.5, 0.5], [1, 1]],  # Top 
        [[0, 0], [0.5, 0.5], [0, 1]],  # Left 
        [[1, 0], [0.5, 0.5], [1, 1]]   # Right 
    ]
    for tri in triangles:
        polygon = Polygon(tri, edgecolor='black', facecolor='none', linewidth=0.9)
        inset_ax.add_patch(polygon)

    # Add text to each triangle
    texts = ["RK08", "Bias", "RMSE", "All"]
    text_positions = [(0.22, 0.5), (0.5, 0.81), (0.78, 0.5), (0.5, 0.19)]
    for pos, txt in zip(text_positions, texts):
        inset_ax.text(pos[0], pos[1], txt, ha="center", va="center", fontsize=9)

    return fig, ax


def eeofs_plot(eeof, clon=0, title='ISO convection patterns', cb_label='scaled EEOF', cmap='RdBu_r', levels=13, vmin=None, vmax=None):
    """ 
    Generate plot of Empirical Orthogonal Functions (EOFs) for each ISO mode (MJO and BSISO)
    during boreal winter and boreal summer separately.

    Parameters
    ----------
    eeof : xarray.Dataset
        EEOFs data.
    clon : int
        Central longitude for the spatial maps.
    title : str
        Title of the plot.
    cb_label : str
        Label to display below the colorbar. 
    cmap : matplotlib colormap
        Colormap.
    levels : np.ndarray
        Contour levels.
    vmin, vmax : float
        Min. and max. values for the colormap.

    Returns
    -------
    fig : (matplotlib.figure.Figure)
        Generated plot.
    ax : (matplotlib.axes._subplots.AxesSubplot)
        Plot axis.
    """

    # Validate input
    if not isinstance(eeof, xr.Dataset):
        raise TypeError("The EOFss data must be xarray.Datasets.")    
    if not isinstance(clon, (int, float)) or not (0 <= clon <= 360):
        raise TypeError("The central longitude 'clon' must be a numeric value between 0º and 360º.")
    
    lags = eeof.lag.values
    modes = eeof.mode.values

    # Compute scaling factor to convert EEOFs to physical units
    eigen = eeof['eigval']
    sigma = 1.0
    scale = np.sqrt(eigen) * sigma


    # Prepare tick for axes        
    lon_ticks = np.arange(-180, 181, 60)
    lat_ticks = [-20, 0, 20]


    # Create plot
    fig, axs = plt.subplots(len(lags), len(modes), figsize=(10, 4), dpi=150,
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=clon)}, 
                            constrained_layout=False)
    fig.subplots_adjust(wspace=0, hspace=0, top=0.86)

    for j, lag in enumerate(lags):
        eeof_lag = eeof['eeof'].sel(lag=lag)
        eeof_phys = eeof_lag * scale

        for l, mode in enumerate(modes):
            ax = axs[j, l]
            style_cartopy_axis(ax, show_gridlines=False)

            eeof_aux = eeof_phys.sel(mode=mode)
            cb = eeof_aux.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels,
                                       vmin=vmin, vmax=vmax, add_colorbar=False)
            ax.text(
                0.97, 0.9, f"lag = {lag} days",
                transform=ax.transAxes, 
                ha="right", va="top", 
                fontsize=6,
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.4", alpha=0.8)
            )

    
            # Add titles and axes labels
            if j == 0:
                ax.set_title(f"EEOF{mode} ({100*eeof['var_frac'].sel(mode=mode).values :.2f}%)", fontsize=12)
            else:
                ax.set_title("")
            if j == len(lags)-1:
                ax.set_xlabel("longitude", fontsize=8)
                if l == 0:
                    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree(central_longitude=clon))
                elif l == len(modes)-1:
                    ax.set_xticks(lon_ticks[1:], crs=ccrs.PlateCarree(central_longitude=clon))
                else:
                    ax.set_xticks(lon_ticks[1:-1], crs=ccrs.PlateCarree(central_longitude=clon))
                ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
            else:
                ax.set_xlabel("")
            if l == 0:
                ax.set_ylabel("latitude", fontsize=8)
                ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
            else:
                ax.set_ylabel("")
            ax.tick_params(axis='both', labelsize=6)

    # Add shared colorbar        
    cbar = fig.colorbar(cb, ax=axs, orientation="horizontal", shrink=0.5, pad=0.17, aspect=40)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cb_label, fontsize=9)

    fig.suptitle(title, fontsize=14)

    return fig, axs


def pcs_plot(pcs, title='Bimodal ISO indices', normalized=True):
    """ 
    Generate plot of the Bimodal ISO indices, i.e. the Principal Components (PCs) for each ISO mode (MJO and BSISO).

    Parameters
    ----------
    pcs : xarray.Dataset
        PCs data.
    title : str
        Title of the plot.
    normalized : bool
        Wether to plot raw or normalized (by the corresponding eigenvalues) PCs.

    Returns
    -------
    fig : (matplotlib.figure.Figure)
        Generated plot.
    ax : (matplotlib.axes._subplots.AxesSubplot)
        Plot axis.
    """

    # Validate input
    if not isinstance(pcs, xr.Dataset):
        raise TypeError("The PCs data must be an xarray.Dataset.")
    if normalized:
        pcs_MJO = pcs['PC_MJO_std']
        pcs_BSISO = pcs['PC_BSISO_std']
        amp_MJO = pcs['amp_MJO_std']
        amp_BSISO = pcs['amp_BSISO_std']
    else:
        pcs_MJO = pcs['PC_MJO_raw']
        pcs_BSISO = pcs['PC_BSISO_raw']
        amp_MJO = pcs['amp_MJO_raw']
        amp_BSISO = pcs['amp_BSISO_raw']
    modes = pcs_MJO.mode.values
    
    # Prepare time labels
    time = pcs_MJO.time.values
    start = np.datetime64(time.min(), "M")
    end = np.datetime64(time.max(), "M")

    months = np.arange(start, end + np.timedelta64(2, "M"), np.timedelta64(1, "M"))
    tick_labels = [str(m)[:7] for m in months]

    # Create plot
    fig, axs = plt.subplots(3,1, figsize=(12, 8), sharex=True, dpi=150)
    colors = [['tab:blue', 'tab:red', 'tab:brown', 'tab:pink'], ['tab:orange', 'tab:green',' tab:purple', 'tab:gray']]
    labels = ['MJO','BSISO']
    y_lim = 4
    
    # Plot PCs
    for i, data in enumerate([pcs_MJO, pcs_BSISO]):
        for j in modes:
            axs[i].plot(data.time, data.sel(mode=j), lw=1.2, ls='-', label=f'{labels[i]} PC{j}', color=colors[i][j-1])
        
        # Plot formatting
        axs[i].set_xticks(months)
        axs[i].tick_params(axis='both', labelsize=8)
        axs[i].set_ylim(-y_lim, y_lim)
        axs[i].set_ylabel(f'Normalized PC', fontsize=10)
        axs[i].set_title(labels[i], fontsize=12)
        axs[i].legend(fontsize=8, loc='upper right')
        axs[i].grid()

    # Plot amplitudes
    for i, data in enumerate([amp_MJO, amp_BSISO]):
        axs[2].plot(data.time, data, lw=1.2, ls='-', label=f'{labels[i]}', color=colors[i][0])
        axs[2].fill_between(data.time, data, where=data>=0, color=colors[i][0], alpha=0.3)

    axs[2].set_xticks(months)
    axs[2].tick_params(axis='both', labelsize=8)
    axs[2].set_ylim(0, y_lim)
    axs[2].set_ylabel(f'|Normalized PCs|', fontsize=10)
    axs[2].set_title('Amplitude', fontsize=12)
    axs[2].legend(fontsize=8, loc='upper right')
    axs[2].grid()

    axs[2].set_xticklabels(tick_labels, rotation=45, ha='right')
    axs[2].set_xlabel('time', fontsize=10)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig, axs


def freq_ISO_plot(freq_ISO_sim, freq_ISO_obs=None, alpha=None, corr=None, sigma=None, tss=None, title='Mean monthly frequency of ISO events', 
                  sim_label='simulations', obs_label='observations'):
    """ 
    Generate plot of the mean monthly frequency of occurrence of ISO events (ISO seasonality) 
    distinguishing between MJO and BSISO. If observations are provided, also include Taylor 
    Skill Score (TSS) statistics below the plot.

    Parameters
    ----------
    freq_ISO_sim : xarray.Dataset
        Mean monthly frequency of occurrence data for simulated data.
    freq_ISO_obs : xarray.Dataset
        Mean monthly frequency of occurrence data for observations.
    alpha : float
        Ratio between simulated and observed standarized PCs' amplitudes.
    corr : float
        Temporal correlation coefficient of the seasonality.
    sigma : float
        Ratio of the standard deviations (model/obs) of the seasonality.
    tss : float
        Taylor Skill Score of the seasonality.
    title : str
        Title of the plot.
    sim_label, obs_label : str
        Labels for simulated and observed data.

    Returns
    -------
    fig : (matplotlib.figure.Figure)
        Generated plot.
    ax : (matplotlib.axes._subplots.AxesSubplot)
        Plot axis.
    """

    # Validate input
    if not isinstance(freq_ISO_sim, xr.Dataset) or (freq_ISO_obs is not None and not isinstance(freq_ISO_obs, (xr.Dataset))):
        raise TypeError("The frequency of occurrence data must be xarray.Datasets.")
    

    # Create plot
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    plt.tight_layout()

    bar_width = 0.4
    x = np.arange(1,13)
    ax.axhline(0, color='black', lw=0.8)
    
    if freq_ISO_obs is None:
        ax.bar(x, freq_ISO_sim['freq_MJO'], color='tab:blue', label=f'MJO {sim_label}', align='center')
        ax.bar(x, -freq_ISO_sim['freq_BSISO'], color='tab:orange', label=f'BSISO {sim_label}', align='center') 

        # MJO legend (upper right)
        mjo_handles = [plt.Rectangle((0,0),1,1, color='tab:blue', label=f'MJO {sim_label}')]
        legend_mjo = ax.legend(handles=mjo_handles, loc='upper right', fontsize=8)
        ax.add_artist(legend_mjo)

        # BSISO legend (lower right)
        bsiso_handles = [plt.Rectangle((0,0),1,1, color='tab:orange', label=f'BSISO {sim_label}')]
        legend_bsiso = ax.legend(handles=bsiso_handles, loc='lower right', fontsize=8)
        ax.add_artist(legend_bsiso)
    else:
        # MJO
        ax.bar(x, freq_ISO_sim['freq_MJO'], color='tab:blue', label=f'MJO {sim_label}', width=-bar_width, align='edge')
        ax.bar(x, freq_ISO_obs['freq_MJO'], color='white', edgecolor='tab:blue', hatch='////', linewidth=0.8, label=f'MJO {obs_label}', width=bar_width, align='edge')

        # BSISO
        ax.bar(x, -freq_ISO_sim['freq_BSISO'], color='tab:orange', label=f'BSISO {sim_label}', width=-bar_width, align='edge')
        ax.bar(x, -freq_ISO_obs['freq_BSISO'], color='white', edgecolor='tab:orange', hatch='////', linewidth=0.5, label=f'BSISO {obs_label}', width=bar_width, align='edge')
        
        # MJO legend (upper right)
        mjo_handles = [
            plt.Rectangle((0,0),1,1, color='tab:blue', label=f'MJO {sim_label}'),
            plt.Rectangle((0,0),1,1, facecolor="white", edgecolor='tab:blue', hatch='////', label=f'MJO {obs_label}')
        ]
        legend_mjo = ax.legend(handles=mjo_handles, loc='upper right', fontsize=8)
        ax.add_artist(legend_mjo)

        # BSISO legend (lower right)
        bsiso_handles = [
            plt.Rectangle((0,0),1,1, color='tab:orange', label=f'BSISO {sim_label}'),
            plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='tab:orange', hatch='////', label=f'BSISO {obs_label}')
        ]
        legend_bsiso = ax.legend(handles=bsiso_handles, loc='lower right', fontsize=8)
        ax.add_artist(legend_bsiso)

        # Add Taylor Skill Score (TSS) statistics
        if alpha is None:
            stats_text = (
                f"Statistics: R={f'{corr:.2f}' if corr is not None else 'N/A'}, "
                f"$\\sigma$={f'{sigma:.2f}' if sigma is not None else 'N/A'}, "
                f"TSS={f'{tss:.2f}'if tss is not None else 'N/A'}"
            )
        else:
            stats_text = (
                f"Statistics: $\\alpha$={f'{alpha:.2f}' if alpha is not None else 'N/A'}, "
                f"R={f'{corr:.2f}' if corr is not None else 'N/A'}, "
                f"$\\sigma$={f'{sigma:.2f}' if sigma is not None else 'N/A'}, "
                f"TSS={f'{tss:.2f}'if tss is not None else 'N/A'}"
            )
        fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))
        fig.subplots_adjust(top=0.78, bottom=0.16)


    # Plot formatting
    ax.set_xticks(x)
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_xlabel('month', fontsize=10)

    y_ticks = np.linspace(-1, 1, 11)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{abs(y):.1f}' for y in y_ticks])
    ax.set_ylabel('frequency of occurrence', fontsize=10)
    
    ax.set_title(title, fontsize=12)

    return fig, ax


def table_plot(data, title='Climate variables', col_labels='', row_labels='', cbar_ticks=['Low', '0', 'High'], colors=('RdBu_r')):
    """ 
    Generate a table plot with climate data.

    Parameters
    ----------
    data (np.ndarray): 2D array with the data to display in the table.
    title (str): Title of the table.
    col_labels (list): Column labels.
    row_labels (list): Row labels.
    cbar_ticks (list): Labels for the colorbar ticks.
    colors (tuple): Colormap.

    Returns
    -------
    fig (matplotlib.figure.Figure): Generated table plot.
    ax (matplotlib.axes._subplots.AxesSubplot): Plot axis.
    """

    # Validate input
    if data is None or not isinstance(data, np.ndarray):
        raise TypeError("The data must be provided as a numpy ndarray.")
    if data.ndim != 2:
        raise ValueError("The data array must be 2-dimensional.")
    if len(row_labels) != data.shape[0]:
        raise ValueError("The number of row labels must match the number of rows in the data.")
    if len(col_labels) != data.shape[1]:
        raise ValueError("The number of column labels must match the number of columns in the data.")


    # Create figure with adjusted height based on number of rows
    n_rows = len(row_labels)
    height = max(3, n_rows * 0.6)
    fig, ax = plt.subplots(figsize=(5, height), dpi=200)
    
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)

    table = plt.table(cellText=np.round(data,2), rowLabels=row_labels, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.1, 1.4)


    # Fix cell height in points and convert to fraction of axes height (1 point = 1/72 inch)
    cell_height_pt = 30
    axes_height_inches = fig.get_size_inches()[1] * ax.get_position().height
    cell_height = (cell_height_pt / 72.0) / axes_height_inches

    # Customize first column (row labels)
    for row in range(1,n_rows+1):
        table.get_celld()[(row, -1)].set_height(cell_height)
        table.auto_set_column_width(-1)

    # Customize other columns
    for col in range(len(col_labels)):
        table.auto_set_column_width(col)

        # Set appearance for header cells
        table.get_celld()[(0,col)].set_height(cell_height)

        # Paint the cells in the first row (corresponding to IBTrACS) with light gray
        table.get_celld()[(1,col)].set_height(cell_height)
        cell = table[(1, col)]
        cell.set_facecolor('lightgray')


        # Color the cells in rows 2-on
        values = data[1:, col]
        vmin, vmax = values.min(), values.max()

        abs_max = max(abs(vmin), abs(vmax))
        norm = plt.Normalize(-abs_max, abs_max)
        cmap = LinearSegmentedColormap.from_list(**colors)

        for row in range(2, n_rows+1):
            table.get_celld()[(row, col)].set_height(cell_height)
            val = data[row-1, col]
            color = cmap(norm(val))
            table[(row, col)].set_facecolor(color)

        
    # Add a horizontal colorbar below the table
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                        orientation='horizontal', pad=0.1, shrink=1.7, aspect=35)

    # Add ticks to bar
    left_tick = norm.vmin + (norm.vmax - norm.vmin) * 0.08  # 8% from left
    middle_tick = 0
    right_tick = norm.vmax - (norm.vmax - norm.vmin) * 0.08  # 8% from right
    cbar.set_ticks([left_tick, middle_tick, right_tick])
    cbar.set_ticklabels(cbar_ticks)
    cbar.ax.tick_params(labelsize=12, length=0)
    
        
    return fig, ax