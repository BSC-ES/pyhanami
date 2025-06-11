import cmocean
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt

from cartopy.util import add_cyclic_point


def time_series_plot(time_series):
    """ Generate time series plot. """
    raise NotImplementedError("This function is not implemented yet.")


def style_cartopy_axis(ax, show_gridlines=True):
    """
    Add standard geographic features and optional gridlines to a Cartopy axis.

    Parameters
    ----------
    ax (cartopy.mpl.geoaxes.GeoAxesSubplot):  Axis with a Cartopy geographic projection.
    show_gridlines (bool): Whether to add gridlines with latitude and longitude labels.

    Returns
    -------
    None

    """

    # Features
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.8)
    ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.5)

    # Gridlines
    if show_gridlines:
        gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.6, color='black', alpha=0.8, linestyle='-.')
        gl.xlabel_style = {"size": 15}
        gl.ylabel_style = {"size": 15}


def add_colorbar(fig, mappable, ax_l, ax_r, ax_b, label='', fontsize=15, levels=None, dist=0.07, width=0.02, **colorbar_kwargs):
    """
    Create a horizontal colorbar below the given axes.

    Parameters
    ----------
    fig (matplotlib.figure.Figure): Figure to add the colorbar to.
    mappable (matplotlib artist): Object for colorbar.
    ax_l, ax_r, ax_b (matplotlib.axes.Axes): Leftmost, rightmost, and bottom axes used to determine the bounds.
    label (str): Colorbar label.
    fontsize (int): Font size for label and tick labels.
    levels (np.ndarray): Contour levels for the colorbar ticks.
    dist (float): Vertical distance below ax_b for placing the colorbar.
    width (float): Thickness of the colorbar axis.
    **colorbar_kwargs: Additional arguments passed to fig.colorbar().

    Returns
    -------
    cb_ax (matplotlib.colorbar.Colorbar): Colorbar.
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


def spatial_plot(data, title='Spatial plot', cb_label='', cmap=cmocean.cm.thermal, levels=None, significant=None,
                 vmin=None, vmax=None, show_contours=True, contour_fontsize=12, gridlines=True, **plot_kwargs):
    """ 
    Generate a spatial plot using Cartopy with a significance mask if selected.

    Parameters
    ----------
    data (xarray.DataArray): 2D dataset to plot with dimensions (lat, lon).
    title (str): Title of the plot.
    cb_label (str): Label to display below the colorbar. 
    cmap (matplotlib colormap): Colormap.
    levels (np.ndarray): Contour levels.
    significant (np.ndarray): Boolean mask for significance hatching.
    vmin, vmax (float): Min. and max. values for the colormap.
    show_contours (bool): Whether to overlay contour lines.
    contour_fontsize (int): Font size for contour labels.
    gridlines (bool): Whether to show gridlines.
    **plot_kwargs: Additional arguments passed to contourf.

    Returns
    -------
    fig (matplotlib.figure.Figure): Generated plot.
    ax (matplotlib.axes._subplots.AxesSubplot): Plot axis.
    """

    # Validate inputs
    if not isinstance(data, xr.DataArray):
        raise TypeError("The data must be a xarray.DataArray.")
    if 'lat' not in data.coords or 'lon' not in data.coords:
        raise ValueError("Could not identify latitude and longitude coordinates.")
    
    # Add cyclic point
    aux, lon = add_cyclic_point(data, coord=data.lon.values)
    data_cyclic = xr.DataArray(data=aux, dims=['lat', 'lon'], coords={'lat': data.lat.values, 'lon': lon}, name=data.name, attrs=data.attrs)

    # Create figure
    fig, ax = plt.subplots(1, figsize=(20, 10), dpi=150, subplot_kw={'projection': ccrs.Robinson(), "aspect": 'auto'}, gridspec_kw = {'wspace':0.01, 'hspace':0.02})
    style_cartopy_axis(ax, show_gridlines=gridlines)
    cb = data_cyclic.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False, **plot_kwargs)

    # Add contour lines if requested
    if show_contours:
        contour_lines = data_cyclic.plot.contour(ax=ax, transform=ccrs.PlateCarree(), levels=levels, colors='black', linewidths=0.5)
        ax.clabel(contour_lines, fontsize=contour_fontsize, colors='black')  # For siconc better use: fontsize=8

    # Significance hatching
    if significant is not None:
        if significant.shape != data_cyclic.shape:
            raise ValueError(f"Mask shape {significant.shape} does not match data shape {data_cyclic.values.shape}.")
        else:
            aux, lon = add_cyclic_point(significant, coord=data.lon.values)
            mask = np.ma.masked_where(aux == 0, data_cyclic.values)  
            ax.pcolor(data_cyclic.lon.values, data_cyclic.lat.values, mask, transform=ccrs.PlateCarree(), hatch='..', zorder=1, alpha=0.)

    # Add title if given
    if title:
        ax.set_title(title, fontsize=20, pad=20)

    # Add colorbar
    colorbar = add_colorbar(fig=fig, mappable=cb, ax_r=ax, ax_l=ax, ax_b=ax, label=cb_label, levels=levels)


    # NOTE: missing special Stereographic projection plot for sea ice concentration

    return fig


def matrix_plot(eff_sizes, test_results):
    """ Generate matrix plot with effect sizes and statistical tests results. """
    raise NotImplementedError("This function is not implemented yet.")

