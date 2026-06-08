import cmocean
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.path as mpath

from pathlib import Path
from matplotlib import colors
from scipy.stats import bootstrap
from cartopy.util import add_cyclic_point
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

from pyhanami.utils import data_general
from pyhanami.config import config_params


def save_or_show_plot(plot_obj, output_path, plot_filename, plot_name, custom_name=True):
    """
    Either saves a plot to file or displays it based on output path.

    Parameters
    ----------
    plot_obj : matplotlib.figure.Figure
        The plot object to save or show.
    output_path : str, optional
        Directory to save the plot; if None, the plot is displayed.
    plot_filename : str
        Base name for the plot file.
    plot_name : str
        Name of the plot for display messages.
    custom_name : bool
        Whether a custom name can be provided for the output file or a
        directory should be provided instead (default: True).
    """

    if output_path is None:
        print(f"{plot_name} created and displayed:", flush=True)
        plt.show()
    else:
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            plot_path = output_path / f"{plot_filename}.png"
        else:
            if not custom_name:
                raise ValueError(
                    f"More than one plot cannot be saved to the same file '{output_path}'. "
                    "Please provide a directory path instead."
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path = output_path

        plot_obj.savefig(plot_path, bbox_inches="tight", dpi=150)
        print(f"{plot_name} created and saved to '{plot_path}'.", flush=True)

    plt.close(plot_obj)
    return


def plot_time_series(time_series, title="Mean time series", y_label="", labels=None,
                     time_freq="annual", start_year=None, end_year=None,  plot_ens=False):
    """
    Generate time series plot of one or more ensembles, including the 2.5th, 5th, 75th 
    and 97.5th percentiles.

    Parameters
    ----------
    time_series : xr.DataArray or list[xr.DataArray]
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
        raise TypeError(
            "The data must be either a xarray.DataArray or a list of xarray.DataArray."
        )

    if labels is not None:
        if (
            not isinstance(labels, list)
            or len(labels) != len(time_series)
            or not all(isinstance(label, str) for label in labels)
        ):
            raise ValueError(
                "If provided, 'labels' must be a list of strings with the same length as time_series."
            )

    if start_year is None or end_year is None:
        raise TypeError("'start_year' and 'end_year' must be non-empty.")


    # Filter each time series to the requested dates range
    filtered_timeseries = []
    for series in time_series:
        series_filtered = series.sel(time=slice(str(start_year), str(end_year)))
        if series_filtered is not None:
            filtered_timeseries.append(series_filtered)
        else:
            raise ValueError(
                f"No data available in the selected range {start_year}-{end_year} for some datasets."
            )

    # Find common dates across all filtered time series
    # date_sets = [set(series["time"].dt.date.values) for series in filtered_timeseries]
    # common_dates = sorted(set.intersection(*date_sets))
    # if not common_dates:
    #     raise ValueError("No overlapping dates in the selected range across all time series.")
    # final_timeseries = [series.sel(time=series["time"].dt.date.isin(common_dates)) for series in filtered_timeseries]


    # Create plot
    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=150)
    labels = labels or [f"Series {i + 1}" for i in range(len(filtered_timeseries))]

    for series, label in zip(filtered_timeseries, labels):
        dates = series["time"].values

        if "realization" in series.dims and series.sizes["realization"] > 1:
            # Plot mean
            mean_series = series.mean("realization")
            (line,) = ax.plot(dates, mean_series.values, lw=2, ls="-.", label=label, zorder=4)
            color = line.get_color()

            # Plot individual ensemble members
            if plot_ens:
                for i in range(series.sizes["realization"]):
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
            quantiles = series.quantile([0.05, 0.95], dim="realization", skipna=True)
            ax.fill_between(dates, quantiles[0], quantiles[1], facecolor=color, alpha=0.2, zorder=1)

        else:
            ax.plot(dates, series.values, lw=2, ls="-.", label=label)

    # Set x-ticks and labels
    if time_freq == "annual":
        unit = "Y"
        step = 1
    elif time_freq == "monthly":
        unit = "M"
        step = 2
    elif time_freq == "daily":
        unit = "D"
        step = 2
    else:
        raise ValueError("Invalid 'time_freq'. Must be 'annual', 'monthly' or 'daily'.")

    x_ticks = [np.datetime64(str(year), unit) for year in np.arange(start_year, end_year + step, 1)]
    x_labels = [str(date) for date in x_ticks]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Plot formatting
    ax.set_xlabel("time", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(linestyle=":")

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
        gl = ax.gridlines(
            draw_labels=True,
            crs=ccrs.PlateCarree(),
            linewidth=gl_lw,
            color="black",
            alpha=0.8,
            linestyle="-.",
        )
        gl.xlabel_style = {"size": gl_fontsize}
        gl.ylabel_style = {"size": gl_fontsize}

    return


def add_colorbar(fig, mappable, ax_l, ax_r, ax_b, label="", fontsize=15, levels=None, dist=0.07,
                 width=0.02, **colorbar_kwargs):
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
    right = ax_r.get_position().extents[2]
    bottom = ax_b.get_position().extents[1]

    cb_ax = fig.add_axes(
        [
            left,  # x position
            bottom - dist,  # y position
            right - left,  # x width
            width,  # y width
        ]
    )

    cbar = fig.colorbar(
        mappable,
        cax=cb_ax,
        orientation="horizontal",
        shrink=0.5,
        pad=0.05,
        aspect=40,
        **colorbar_kwargs,
    )
    cb_ax.set_xlabel(label, fontsize=fontsize)

    if levels is not None:
        cbar.set_ticks(levels)
    cbar.ax.tick_params(labelsize=fontsize)

    return cbar


def plot_spatial( data, clon=0, title="Spatial plot", cb_label="", cmap=cmocean.cm.thermal,
                 levels=None, significant=None, vmin=None, vmax=None, show_contours=True,
                 contour_fontsize=12, gridlines=True, **plot_kwargs):
    """
    Generate a spatial plot using Cartopy with a significance mask if selected.

    Parameters
    ----------
    data : xr.DataArray
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
    if "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError("Could not identify latitude and longitude coordinates.")
    if not isinstance(clon, (int, float)) or not (0 <= clon <= 360):
        raise TypeError(
            "The central longitude 'clon' must be a numeric value between 0º and 360º."
        )

    # Correct 0 and NaN values (values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0)
    var_name = data.name
    if var_name in {"siconc", "sos", "tos"}:
        data = xr.where((np.isnan(data)) | (data == 0), 10**-10, data)
        ocean_data = True
    else:
        ocean_data = False

    # Add cyclic point
    aux, lon = add_cyclic_point(data, coord=data.lon.values)
    data_cyclic = xr.DataArray(
        data=aux,
        dims=["lat", "lon"],
        coords={"lat": data.lat.values, "lon": lon},
        name=data.name,
        attrs=data.attrs,
    )

    if var_name != "siconc":
        # Create figure
        fig, ax = plt.subplots(
            1,
            figsize=(20, 10),
            dpi=150,
            subplot_kw={"projection": ccrs.Robinson(central_longitude=clon), "aspect": "auto"},
            gridspec_kw={"wspace": 0.01, "hspace": 0.02},
        )
        style_cartopy_axis(ax, show_gridlines=gridlines, ocean_data=ocean_data)
        cb = data_cyclic.plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            **plot_kwargs,
        )

        # Add contour lines if requested
        if show_contours:
            contour_lines = data_cyclic.plot.contour(
                ax=ax,
                transform=ccrs.PlateCarree(),
                levels=levels,
                colors="black",
                linewidths=0.5
            )
            ax.clabel(contour_lines, fontsize=contour_fontsize, colors="black")

        # Significance hatching
        if significant is not None:
            if significant.shape != data.shape:
                raise ValueError(
                    f"Mask shape {significant.shape} does not match data shape {data.values.shape}."
                )

            aux, lon = add_cyclic_point(significant, coord=data.lon.values)
            mask = np.ma.masked_where(aux == 0, data_cyclic.values)
            ax.pcolor(
                data_cyclic.lon.values,
                data_cyclic.lat.values,
                mask,
                transform=ccrs.PlateCarree(),
                hatch="..",
                zorder=1,
                alpha=0.0,
            )

        # Add title if given
        if title:
            ax.set_title(title, fontsize=20, pad=20)

        # Add colorbar if cb_label is given
        if cb_label:
            _ = add_colorbar(
                fig=fig,
                mappable=cb,
                ax_l=ax,
                ax_r=ax,
                ax_b=ax,
                label=cb_label,
                levels=levels
            )

    # Special Stereographic projection plot for sea ice concentration
    else:
        lat_limit = 40

        # Compute a circle in axes coordinates, to be used as a boundary for the map
        # (it allows to pan/zoom as much as needed, the boundary will be permanently circular)
        theta = np.linspace(0, 2 * np.pi, 100)
        center = [0.5, 0.5]
        radius = 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)


        ### North pole ###
        proj = ccrs.Stereographic(central_latitude=90, central_longitude=0)
        crs = ccrs.PlateCarree()

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(10, 5.5), dpi=150, subplot_kw={"projection": proj})

        # Customize gridlines
        ax1 = ax[0]
        ax1.set_extent([-180, 180, lat_limit, 90], crs=crs)

        gl1 = ax1.gridlines(draw_labels=True, linewidth=0.4, color="gray", linestyle="-.")
        gl1.xlabel_style = {"size": 8}
        gl1.ylabel_style = {"size": 8}
        gl1.ylocator = plt.MultipleLocator(10)

        # Specify borders and coastlines
        ax1.add_feature(cf.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=3)
        ax1.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.4, zorder=4)
        ax1.add_feature(cf.BORDERS.with_scale("50m"), lw=0.2, zorder=4)

        ax1.set_boundary(circle, transform=ax1.transAxes)

        # Add contour lines if requested
        if show_contours:
            contour_lines = data_cyclic.plot.contour(
                ax=ax1,
                transform=ccrs.PlateCarree(),
                levels=levels,
                colors="black",
                linewidths=0.5
            )
            ax1.clabel(contour_lines, fontsize=contour_fontsize - 5, colors="black")

        # Add data
        cb = data_cyclic.plot.contourf(
            ax=ax1,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            **plot_kwargs,
        )

        # Significance hatching
        if significant is not None:
            if significant.shape != data.shape:
                raise ValueError(
                    f"Mask shape {significant.shape} does not match data shape {data.values.shape}."
                )

            aux, lon = add_cyclic_point(significant, coord=data.lon.values)
            mask = np.ma.masked_where(aux == 0, data_cyclic.values)
            ax1.pcolor(
                data_cyclic.lon.values,
                data_cyclic.lat.values,
                mask,
                transform=ccrs.PlateCarree(),
                hatch="..",
                zorder=1,
                alpha=0.0,
            )

        ### South pole ###
        proj = ccrs.Stereographic(central_latitude=-90, central_longitude=0)

        # Customize gridlines
        ax2 = ax[1]
        ax2.projection = proj
        ax2.set_extent([-180, 180, -90, -lat_limit], crs=crs)

        gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="-.")
        gl2.xlabel_style = {"size": 8}
        gl2.ylabel_style = {"size": 8}
        gl2.ylocator = plt.MultipleLocator(10)

        # Specify borders and coastlines
        ax2.add_feature(cf.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=3)
        ax2.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.4, zorder=4)
        ax2.add_feature(cf.BORDERS.with_scale("50m"), lw=0.2, zorder=4)

        ax2.set_boundary(circle, transform=ax2.transAxes)

        # Add contour lines if requested
        if show_contours:
            contour_lines = data_cyclic.plot.contour(
                ax=ax2,
                transform=ccrs.PlateCarree(),
                levels=levels,
                colors="black",
                linewidths=0.5
            )
            ax2.clabel(contour_lines, fontsize=contour_fontsize - 5, colors="black")

        # Add data
        cb = data_cyclic.plot.contourf(
            ax=ax2,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            **plot_kwargs,
        )

        # Significance hatching
        if significant is not None:
            if significant.shape != data.shape:
                raise ValueError(
                    f"Mask shape {significant.shape} does not match data shape {data.values.shape}."
                )

            aux, lon = add_cyclic_point(significant, coord=data.lon.values)
            mask = np.ma.masked_where(aux == 0, data_cyclic.values)
            ax2.pcolor(
                data_cyclic.lon.values,
                data_cyclic.lat.values,
                mask,
                transform=ccrs.PlateCarree(),
                hatch="..",
                zorder=1,
                alpha=0.0,
            )

        # Add titles
        plt.subplots_adjust(top=0.9)
        ax1.set_title("North Pole", fontsize=10)
        ax2.set_title("South Pole", fontsize=10, pad=15)
        if title:
            fig.suptitle(title)

        # Add colorbar if cb_label is given
        if cb_label:
            _ = add_colorbar(
                fig=fig,
                mappable=cb,
                ax_l=ax[0],
                ax_r=ax[1],
                ax_b=ax[1],
                label=cb_label,
                fontsize=8,
                levels=levels,
                dist=0.09,
            )

    return fig, ax


def two_spatial_plots(data_1, data_2, clon=0, title_1="Spatial plot 1", title_2="Spatial plot 2",
                      suptitle="Spatial plots", cb_label="", cmap=cmocean.cm.thermal, levels=12,
                      significant_1=None, significant_2=None, vmin=None, vmax=None,
                      show_contours=True, contour_fontsize=6, gridlines=True, **plot_kwargs):
    """
    Generate two spatial plots side by side using Cartopy with significance masks if selected.

    Parameters
    ----------
    data_1 : xr.DataArray
        First 2D dataset to plot with dimensions (lat, lon).
    data_2 : xr.DataArray
        Second 2D dataset to plot with dimensions (lat, lon).
    clon : int
        Central longitude for the spatial maps.
    title_1 : str
        Title of the first plot (default: 'Spatial plot 1').
    title_2 : str
        Title of the second plot (default: 'Spatial plot 2').
    suptitle : str
        Common title for the two plots (default: 'Spatial plots').
    cb_label : str
        Label to display below the common colorbar; set to False to not add a colorbar (default: '').
    cmap : matplotlib colormap
        Colormap (default: cmocean.cm.thermal).
    levels : np.ndarray
        Contour levels (default: 12).
    significant_1 : np.ndarray
        Mask for significance hatching in the first plot.
    significant_2 : np.ndarray
        Mask for significance hatching in the second plot.
    vmin, vmax : float
        Common min. and max. values for the colormap.
    show_contours : bool
        Whether to overlay contour lines (default: True).
    contour_fontsize : int
        Font size for contour labels (default: 12).
    gridlines : bool
        Whether to show gridlines (default: True).
    **plot_kwargs :
        Additional arguments passed to contourf.

    Returns
    -------
    new_fig : matplotlib.figure.Figure
        Generated plot.
    axs : list[matplotlib.axes._subplots.AxesSubplot]
        Plot axes.
    """

    # Validate inputs
    if not isinstance(data_1, xr.DataArray) or not isinstance(data_2, xr.DataArray):
        raise TypeError("The data must be a xarray.DataArray.")
    if (
        "lat" not in data_1.coords
        or "lon" not in data_1.coords
        or "lat" not in data_2.coords
        or "lon" not in data_2.coords
    ):
        raise ValueError("Could not identify latitude and longitude coordinates.")
    if not isinstance(clon, (int, float)) or not (0 <= clon <= 360):
        raise TypeError(
            "The central longitude 'clon' must be a numeric value between 0º and 360º."
        )

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


    # Calculate common vmin and vmax if not provided
    if vmin is None or vmax is None:
        combined_min = min(float(data_1.min()), float(data_2.min()))
        combined_max = max(float(data_1.max()), float(data_2.max()))
        if vmin is None:
            vmin = combined_min
        if vmax is None:
            vmax = combined_max


    # Create figure
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        dpi=150,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=clon), "aspect": "auto"},
    )  # , gridspec_kw = {'wspace':0.01, 'hspace':0.02})

    # Add first plot
    # Correct 0 and NaN values (values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0)
    var_name = data_1.name
    if var_name in {"siconc", "sos", "tos"}:
        data_1 = xr.where((np.isnan(data_1)) | (data_1 == 0), 10**-10, data_1)
        ocean_data = True
    else:
        ocean_data = False

    # Add cyclic point
    aux, lon = add_cyclic_point(data_1, coord=data_1.lon.values)
    data_cyclic = xr.DataArray(
        data=aux,
        dims=["lat", "lon"],
        coords={"lat": data_1.lat.values, "lon": lon},
        name=data_1.name,
        attrs=data_1.attrs,
    )

    # Create figure
    ax_1 = axs[0]
    style_cartopy_axis(
        ax_1,
        show_gridlines=gridlines,
        ocean_data=ocean_data,
        lw_coast=0.4,
        lw_borders=0.2,
        gl_lw=0.3,
        gl_fontsize=7,
    )
    cb = data_cyclic.plot.contourf(
        ax=ax_1,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        **plot_kwargs,
    )

    # Add contour lines if requested
    if show_contours:
        contour_lines = data_cyclic.plot.contour(
            ax=ax_1,
            transform=ccrs.PlateCarree(),
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            colors="black",
            linewidths=0.2,
        )
        ax_1.clabel(contour_lines, fontsize=contour_fontsize, colors="black")

    # Significance hatching
    if significant_1 is not None:
        if significant_1.shape != data_1.shape:
            raise ValueError(
                f"Mask shape {significant_1.shape} does not match data shape {data_1.values.shape}."
            )

        aux, lon = add_cyclic_point(significant_1, coord=data_1.lon.values)
        mask = np.ma.masked_where(aux == 0, data_cyclic.values)
        ax_1.pcolor(
            data_cyclic.lon.values,
            data_cyclic.lat.values,
            mask,
            transform=ccrs.PlateCarree(),
            hatch="..",
            zorder=1,
            alpha=0.0,
        )

    # Add title if given
    if title_1:
        ax_1.set_title(title_1, fontsize=12, pad=8)


    # Add second plot
    # Correct 0 and NaN values (values which are exactly 0 are painted in white, not with the corresponding colorbar color for 0)
    var_name = data_2.name
    if var_name in {"siconc", "sos", "tos"}:
        data_2 = xr.where((np.isnan(data_2)) | (data_2 == 0), 10**-10, data_2)
        ocean_data = True
    else:
        ocean_data = False

    # Add cyclic point
    aux, lon = add_cyclic_point(data_2, coord=data_2.lon.values)
    data_cyclic = xr.DataArray(
        data=aux,
        dims=["lat", "lon"],
        coords={"lat": data_2.lat.values, "lon": lon},
        name=data_2.name,
        attrs=data_2.attrs,
    )

    # Create figure
    ax_2 = axs[1]
    style_cartopy_axis(
        ax_2,
        show_gridlines=gridlines,
        ocean_data=ocean_data,
        lw_coast=0.4,
        lw_borders=0.2,
        gl_lw=0.3,
        gl_fontsize=7,
    )
    cb = data_cyclic.plot.contourf(
        ax=ax_2,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        **plot_kwargs,
    )

    # Add contour lines if requested
    if show_contours:
        contour_lines = data_cyclic.plot.contour(
            ax=ax_2,
            transform=ccrs.PlateCarree(),
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            colors="black",
            linewidths=0.2,
        )
        ax_2.clabel(contour_lines, fontsize=contour_fontsize, colors="black")

    # Significance hatching
    if significant_2 is not None:
        if significant_2.shape != data_2.shape:
            raise ValueError(
                f"Mask shape {significant_2.shape} does not match data shape {data_2.values.shape}."
            )

        aux, lon = add_cyclic_point(significant_2, coord=data_2.lon.values)
        mask = np.ma.masked_where(aux == 0, data_cyclic.values)
        ax_2.pcolor(
            data_cyclic.lon.values,
            data_cyclic.lat.values,
            mask,
            transform=ccrs.PlateCarree(),
            hatch="..",
            zorder=1,
            alpha=0.0,
        )

    # Add title if given
    if title_2:
        ax_2.set_title(title_2, fontsize=12, pad=8)

    # Add shared colorbar if cb_label is given
    if cb_label:
        cbar = fig.colorbar(
            cb,
            ax=axs,
            orientation="horizontal",
            fraction=0.03,
            pad=0.1,
            aspect=60,
            extend="neither",
        )
        cbar.set_ticks(cb.levels)

        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(cb_label, fontsize=8)
        plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    else:
        plt.tight_layout(rect=[0, 0.03, 1, 1])

    fig.suptitle(suptitle, fontsize=14)

    return fig, axs


def plot_matrix(eff_sizes, test_results, test=4, title="Effect sizes replicability test",
                variables=None, seasons=None, regions=None):
    """
    Generate a matrix plot with effect sizes and results of the replicability test for 
    the selected statistical test/s.

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
        raise ValueError(
            "Invalid test index. Must be: 0 (KS-test), 1 (T-test), 2 (U-test), 3 (B-test), 4 (All)."
        )

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
    green = "tab:green"
    red = "tab:red"

    limits_colorbar = np.array([0, 0.01, 0.2, 0.5, 0.8, 1.2, 2])
    blue_colors = LinearSegmentedColormap.from_list("", ["#ffffff", "yellow"])
    orange_colors = LinearSegmentedColormap.from_list("", ["#ffffff", "tab:blue"])
    cmap_aux = np.append(blue_colors([0, 0.2]), orange_colors(np.linspace(0.2, 1, 4)), axis=0)
    cmap = ListedColormap(cmap_aux)
    norm = BoundaryNorm(limits_colorbar, cmap.N)

    # Define the grid
    n_rows, n_cols, n_indices = abs_eff_sizes.shape
    fig_size = (22, 16)
    fig, ax = plt.subplots(figsize=fig_size, dpi=150)

    x_labels = [f"{s1} {s2}" for s1 in seasons for s2 in regions]
    y_labels = variables


    # Loop over each cell in the grid
    circle_radius = 0.13
    for i in range(n_rows):
        for j in range(n_cols):
            values = abs_eff_sizes[i, j, :]

            # Define the vertices in the cell
            x, y = j, i
            corners = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]])
            center = [(x + x + 1) / 2, (y + y + 1) / 2]

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
                color = cmap(norm(number)) if ~np.isnan(number) else "white"
                polygon = Polygon(triangle, color=color)
                ax.add_patch(polygon)

                # Add numeric p-value at the center of the triangle
                centroid = np.mean(triangle, axis=0)
                ax.text(
                    centroid[0],
                    centroid[1],
                    f"{values[k]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#283747",
                )

            # Add inner edges of the triangles
            for corner in corners:
                ax.plot([center[0], corner[0]], [center[1], corner[1]], color="black", lw=0.2)


            # Check if any triangle value is less than alpha
            if test_outcome[i, j] == 1:
                circle_color = red
            elif test_outcome[i, j] == 0 and ~np.isnan(number):
                circle_color = green
            else:
                circle_color = "white"

            # Add circle in the center of the cell
            circle = Circle(
                center,
                radius=circle_radius,
                facecolor=circle_color,
                edgecolor="black",
                lw=0.3,
                zorder=3,
            )
            ax.add_patch(circle)


    # Add gridlines (cells' borders)
    for x in range(n_cols + 1):
        ax.plot([x, x], [0, n_rows], color="black", linewidth=0.9)
    for y in range(n_rows + 1):
        ax.plot([0, n_cols], [y, y], color="black", linewidth=0.9)

    # Customize axes
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")

    # Only ticks at the center of each column/row
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    plt.xticks(rotation=35, ha="right")
    # plt.yticks(rotation=30)

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()

    plt.xlabel("Period")
    plt.ylabel("Variables")
    plt.title(title, y=1.01)


    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        aspect=50,
        pad=0.02,
        ticks=limits_colorbar,
    )
    cbar.set_label("Effect size")

    # Add label to each segment
    ranges = ["Very small", "Small", "Medium", "Large", "Very large", "Huge"]
    for i, loc in enumerate(limits_colorbar[:-1]):
        mid_loc = (limits_colorbar[i] + limits_colorbar[i + 1]) / 2
        cbar.ax.text(
            0.6,
            mid_loc,
            ranges[i],
            ha="center",
            va="center",
            color="black",
            rotation=90,
            fontsize=8,
        )

    # Add box with Rejection/No rejection legend
    legend_ax = fig.add_axes([0.552, 0.02, 0.1, 0.01])  # [left, bottom, width, height]
    legend_ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=green, lw=5, label="No rejection"),
        plt.Line2D([0], [0], color=red, lw=5, label="Rejection"),
    ]
    legend_ax.legend(handles=handles, loc="center", ncol=2)

    # Add square with legend for scores
    square_size = 0.055
    inset_ax = fig.add_axes([0.856, 0.025, square_size * (fig_size[1] / fig_size[0]), square_size])  # [left, bottom, width, height] in figure coordinates
    inset_ax.set_xlim(0, 1)
    inset_ax.set_ylim(0, 1)
    inset_ax.axis("off")

    triangles = [
        [[0, 0], [0.5, 0.5], [1, 0]],  # Bottom
        [[0, 1], [0.5, 0.5], [1, 1]],  # Top
        [[0, 0], [0.5, 0.5], [0, 1]],  # Left
        [[1, 0], [0.5, 0.5], [1, 1]],  # Right
    ]
    for tri in triangles:
        polygon = Polygon(tri, edgecolor="black", facecolor="none", linewidth=0.9)
        inset_ax.add_patch(polygon)

    # Add text to each triangle
    texts = ["RK08", "Bias", "RMSE", "All"]
    text_positions = [(0.22, 0.5), (0.5, 0.81), (0.78, 0.5), (0.5, 0.19)]
    for pos, txt in zip(text_positions, texts):
        inset_ax.text(pos[0], pos[1], txt, ha="center", va="center", fontsize=9)

    return fig, ax


def plot_table(data, title="Climate variables", col_labels="", row_labels="", cbar_ticks=["Low", "", "High"],
               cbar_colors=("RedGreen", ["tab:red", "white", "tab:green"]), limits=None, reference=True,
               decimals=1):
    """
    Generate a table plot with climate data.

    Parameters
    ----------
    data : np.ndarray
        2D array with the data to display in the table.
    title : str
        Title of the table.
    col_labels : list
        Column labels. Note, the row labels will be added as the first column, hence,
        the length of col_labels must be equal to number of columns in data + 1.
    row_labels : list
        Row labels.
    cbar_ticks : list
        Labels for the colorbar ticks (default: ['Low', '0', 'High']).
    cbar_colors : tuple
        Colormap (default: ("RedGreen", ['tab:red', 'white', 'tab:green'])).
    limits : np.ndarray
        Colormap limits (min, max) for each column in the table. If None, the limits will
        be automatically set as the maximum and minimum values in each column.
    reference : bool
        Whether to use the first row of data as reference (not colored) (default: True).
    decimals : int
        Number of decimals to round the data values (default: 1).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated table plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        Plot axis.
    """

    # Validate input
    if data is None or not isinstance(data, np.ndarray):
        raise TypeError("The data must be a np.ndarray.")
    if data.ndim != 2:
        raise ValueError("The data array must be 2-dimensional.")
    if len(row_labels) != data.shape[0]:
        raise ValueError(
            f"The number of row labels ({len(row_labels)}) must match the number of rows in the data ({data.shape[0]})."
        )
    if len(col_labels) - 1 != data.shape[1]:
        raise ValueError(
            f"The number of column labels minus one ({len(col_labels) - 1}) must match the number of columns in the data ({data.shape[1]})."
        )
    if limits is not None and (limits.shape[0] != data.shape[1] or limits.shape[1] != 2):
        raise ValueError("Limits must be a 2D array with shape (n_columns, 2).")


    # Create figure with adjusted height based on number of rows
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    height = max(3, n_rows * 0.6)
    width = 4
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)

    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=20)

    formatted_data = np.array([[f"{val:.{decimals}f}" for val in row] for row in data])
    cell_text = np.column_stack((np.reshape(row_labels, (-1, 1)), formatted_data))
    table = plt.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    # table.scale(1.1, 1.4)


    # Automatically adjust width of first column (row labels)
    table.auto_set_column_width(0)
    row_label_width_inch = table.get_celld()[(0, 0)].get_width()

    axes_width_inches = width * ax.get_position().width
    row_label_length = row_label_width_inch / axes_width_inches

    # Calculate proportional widths for other columns based on label lengths
    col_label_lengths = [len(str(label)) for label in col_labels[1:]]
    total_col_length = np.sum(col_label_lengths) + row_label_length

    col_widths = [(length / total_col_length) for length in col_label_lengths]
    col_widths_inch = [w * axes_width_inches for w in col_widths]

    # Fix cell height in points and convert to fraction of axes height (1 point = 1/72 inch)
    cell_height_pt = 30
    axes_height_inches = height * ax.get_position().height
    cell_height_inch = (cell_height_pt / 72.0) / axes_height_inches


    # Customize first column (row labels)
    table[(0, 0)].set_facecolor("whitesmoke")
    table.get_celld()[(0, 0)].get_text().set_fontsize(12)
    for row in range(1, n_rows + 1):
        table[(row, 0)].get_text().set_ha("left")
    for row in range(n_rows + 1):
        table.get_celld()[(row, 0)].set_width(row_label_width_inch)
        table.get_celld()[(row, 0)].set_height(cell_height_inch)

    # Customize other columns (data cells)
    start_color_cell = 1 if reference else 0
    custom_norm = limits is not None
    cmap = LinearSegmentedColormap.from_list(*cbar_colors)
    for col in range(1, n_cols):
        for row in range(n_rows + 1):
            table.get_celld()[(row, col)].set_width(col_widths_inch[col - 1])
            table.get_celld()[(row, col)].set_height(cell_height_inch)

        if reference:
            # Paint the cells in the first row (corresponding to the reference data) with light gray
            table[(1, col)].set_facecolor("lightgray")

        # Color the remaining cells (rows 2-on if `reference` is True, else all rows)
        if custom_norm:
            vmin, vmax = limits[col - 1]
        else:
            values = data[start_color_cell:, col - 1]
            vmin, vmax = values.min(), values.max()
        if vmin == vmax:
            vmin = vmax - 1e-6
        norm = plt.Normalize(vmin, vmax)

        # if custom_norm:
        #     norm = plt.Normalize(vmin, vmax)
        # else:
        #     values = data[start_color_cell:, col-1]
        #     vmin, vmax = values.min(), values.max()

        #     abs_max = max(abs(vmin), abs(vmax))
        #     norm = plt.Normalize(-abs_max, abs_max)

        for row in range(start_color_cell + 1, n_rows + 1):
            val = data[row - 1, col - 1]
            color = cmap(norm(val))
            table[(row, col)].set_facecolor(color)

    # Add a horizontal colorbar below the table
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        pad=0.1,
        shrink=1.7,
        aspect=35,
    )

    # Add ticks to bar
    left_tick = norm.vmin + (norm.vmax - norm.vmin) * 0.07  # 7% from left
    middle_tick = norm.vmin + (norm.vmax - norm.vmin) * 0.5  # middle
    right_tick = norm.vmax - (norm.vmax - norm.vmin) * 0.07  # 7% from right
    cbar.set_ticks([left_tick, middle_tick, right_tick])
    cbar.set_ticklabels(cbar_ticks)
    cbar.ax.tick_params(labelsize=12, length=0)

    return fig, ax


def plot_two_tables(data_1, data_2, title="Climate variables", col_labels=["", ""], row_labels=["", ""],
                    common_bar=True, cbar_ticks=[["Low", "", "High"], ["Low", "", "High"]],
                    cbar_colors=("RedGreen", ["tab:red", "white", "tab:green"]), limits=[None, None],
                    references=[True, True],  decimals=[1, 1]):
    """
    Generate a table plot with climate data.

    Parameters
    ----------
    data_1 : np.ndarray
        2D array with the data to display in the first table.
    data_2 : np.ndarray
        2D array with the data to display in the second table.
    title : str
        Title of the plot.
    col_labels : list[list]
        Column labels for each table. Note, the row labels will be added as the first column,
        hence, the length of each col_labels must be equal to number of columns in the
        corresponding data + 1.
    row_labels : list[list]
        Row labels for each table.
    common_bar : bool
        Whether to use a common colorbar for both tables (default: True).
    cbar_ticks : list[list]
        Labels for the colorbar ticks for each table (default: [['Low', '', 'High'],
        ['Low', '', 'High']]).
    cbar_colors : tuple
        Colormap (default: ("RedGreen", ['tab:red', 'white', 'tab:green'])).
    limits : list[np.ndarray]
        Colormap limits (min, max) for each column in each table. If None, the limits will
        be automatically set as the maximum and minimum values in each column.
    references : list[bool]
        Whether to use the first row of data as reference (not colored) for each table
        (default: [True, True]).
    decimals : list[int]
        Number of decimals to round the data values for each table (default: [1, 1]).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated table plot.
    axs : list[matplotlib.axes._subplots.AxesSubplot]
        Plot axes.
    """

    # Validate input
    if (
        data_1 is None
        or not isinstance(data_1, np.ndarray)
        or data_2 is None
        or not isinstance(data_2, np.ndarray)
    ):
        raise TypeError("Both data must be a np.ndarray.")
    if data_1.ndim != 2 or data_2.ndim != 2:
        raise ValueError("Both data arrays must be 2-dimensional.")
    if len(row_labels[0]) != data_1.shape[0] or len(row_labels[1]) != data_2.shape[0]:
        raise ValueError(
            f"The number of row labels ({len(row_labels[0])}, {len(row_labels[1])}) must match the number of rows "
            f"in both data arrays ({data_1.shape[0]}, {data_2.shape[0]})."
        )
    if len(col_labels[0]) - 1 != data_1.shape[1] or len(col_labels[1]) - 1 != data_2.shape[1]:
        raise ValueError(
            f"The number of column labels minus one ({len(col_labels[0]) - 1}, {len(col_labels[1]) - 1}) must match "
            f"the number of columns in both data arrays ({data_1.shape[1]}, {data_2.shape[1]})."
        )
    if limits is not None and (
        limits[0].shape[0] != data_1.shape[1]
        or limits[1].shape[0] != data_2.shape[1]
        or limits[0].shape[1] != 2
        or limits[1].shape[1] != 2
    ):
        raise ValueError(
            "Limits must be a list of 2D arrays with shape (n_columns, 2) for both data arrays."
        )

    # Create figure with adjusted height based on number of rows
    n_rows = [len(rows) for rows in row_labels]
    n_cols = [len(cols) for cols in col_labels]
    height = max(5, sum(n_rows) * 0.6)
    width = 4
    fig, axs = plt.subplots(2, 1, figsize=(width, height), dpi=200)
    fig.suptitle(title, fontsize=16)


    # Fix cell height in points and convert to fraction of axes height (1 point = 1/72 inch)
    cell_height_pt = 30
    axes_height_inches = height * axs[1].get_position().height
    cell_height_inch = (cell_height_pt / 72.0) / axes_height_inches


    # Add one table to each subplot
    for ax, data, decimal, row_label, col_label, n_row, n_col, limit, reference in zip(
        axs, [data_1, data_2], decimals, row_labels, col_labels, n_rows, n_cols, limits, references
    ):
        # Add data
        formatted_data = np.array([[f"{val:.{decimal}f}" for val in row] for row in data])
        cell_text = np.column_stack((np.reshape(row_label, (-1, 1)), formatted_data))
        table = ax.table(cellText=cell_text, colLabels=col_label, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        # table.scale(1.1, 1.4)
        ax.axis("off")

        # Automatically adjust width of first column (row labels)
        table.auto_set_column_width(0)
        row_label_width_inch = table.get_celld()[(0, 0)].get_width()

        axes_width_inches = width * ax.get_position().width
        row_label_length = row_label_width_inch / axes_width_inches

        # Calculate proportional widths for other columns based on label lengths
        col_label_lengths = [len(str(label)) for label in col_label[1:]]
        total_col_length = np.sum(col_label_lengths) + row_label_length

        col_widths = [(length / total_col_length) for length in col_label_lengths]
        col_widths_inch = [w * axes_width_inches for w in col_widths]


        # Customize first column (row labels)
        table[(0, 0)].set_facecolor("whitesmoke")
        table.get_celld()[(0, 0)].get_text().set_fontsize(12)
        for row in range(1, n_row + 1):
            table[(row, 0)].get_text().set_ha("left")
        for row in range(n_row + 1):
            table.get_celld()[(row, 0)].set_width(row_label_width_inch)
            table.get_celld()[(row, 0)].set_height(cell_height_inch)

        # Customize other columns (data cells)
        start_color_cell = 1 if reference else 0
        custom_norm = limit is not None
        cmap = LinearSegmentedColormap.from_list(*cbar_colors)
        for col in range(1, n_col):
            for row in range(n_row + 1):
                table.get_celld()[(row, col)].set_width(col_widths_inch[col - 1])
                table.get_celld()[(row, col)].set_height(cell_height_inch)

            if reference:
                # Paint the cells in the first row (corresponding to the reference data) with light gray
                table[(1, col)].set_facecolor("lightgray")

            # Color the remaining cells (rows 2-on if `reference` is True, else all rows)
            if custom_norm:
                vmin, vmax = limit[col - 1]
            else:
                values = data[start_color_cell:, col - 1]
                vmin, vmax = values.min(), values.max()
            if vmin == vmax:
                vmin = vmax - 1e-6
            norm = plt.Normalize(vmin, vmax)

            # if custom_norm:
            #     norm = plt.Normalize(vmin, vmax)
            # else:
            #     values = data[start_color_cell:, col-1]
            #     vmin, vmax = values.min(), values.max()

            #     abs_max = max(abs(vmin), abs(vmax))
            #     norm = plt.Normalize(-abs_max, abs_max)

            for row in range(start_color_cell + 1, n_row + 1):
                val = data[row - 1, col - 1]
                color = cmap(norm(val))
                table[(row, col)].set_facecolor(color)

    if common_bar:
        # Save space for colorbar
        fig.subplots_adjust(bottom=0.15)

        # Create fixed cax (colorbar axes); if not done, the tables will be resized when adding the colorbar
        left = axs[0].get_position().extents[0]
        right = axs[-1].get_position().extents[2]
        bottom = axs[-1].get_position().extents[1]

        x_offset = 0.35
        cb_ax = fig.add_axes(
            [
                left - x_offset,  # x position
                bottom - 0.1,  # y position
                right + 2 * x_offset - left,  # x width
                0.04,  # y width
            ]
        )

        # Add horizontal colorbar in the fixed cax
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cb_ax,
            orientation="horizontal",
            pad=0.05,
            shrink=0.5,
            aspect=40,
        )

        # Create mappable for colorbar
        # mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        # mappable.set_array([])  # avoid warnings for empty mappable

        # # Create fix cax (colorbar axes) positioned below both tables' axes
        # cbar = add_colorbar(fig=fig, mappable=mappable, ax_l=axs[0], ax_r=axs[-1], ax_b=axs[-1], label='',
        #                     fontsize=12, levels=None, dist=0.1, width=0.03, shrink=0.5, aspect=40)

        # Add ticks to bar
        left_tick = norm.vmin + (norm.vmax - norm.vmin) * 0.07  # 7% from left
        middle_tick = norm.vmin + (norm.vmax - norm.vmin) * 0.5  # middle
        right_tick = norm.vmax - (norm.vmax - norm.vmin) * 0.07  # 7% from right
        cbar.set_ticks([left_tick, middle_tick, right_tick])
        cbar.set_ticklabels(cbar_ticks)
        cbar.ax.tick_params(labelsize=12, length=0)
    else:
        raise NotImplementedError("One colorbar per table is not implemented yet.")

    return fig, axs


def plot_grouped_bars(data, x_values=None, title="Grouped bar plot", x_label="", y_label="",
                      labels=None):
    """
    Generate a grouped bar plot.

    Parameters
    ----------
    data : np.ndarray
        2D array with the data to display in the bar plot.
    x_values : list
        Values for the x-axis. If None, default integer
        values will be used.
    title : str
        Title of plot (default: 'Grouped bar plot').
    x_label : str
        Label for the x-axis (default: '').
    y_label : str
        Label for the y-axis (default: '').
    labels : list
        Labels for each group in the bar plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated table plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        Plot axis.
    """

    # Validate input
    if not isinstance(data, np.ndarray):
        raise TypeError("The data must be a np.ndarray.")

    # Prepare plotting parameters
    if x_values is None:
        x_values = np.arange(data.shape[1])

    n_bars = data.shape[0]
    total_width = 0.75
    bar_width = total_width / n_bars


    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Plot bars
    for i, dataset in enumerate(data):
        ax.bar(
            x_values - (total_width / 2) + (i + 0.5) * bar_width,
            dataset,
            width=bar_width,
            label=labels[i] if labels else None,
            zorder=2,
        )

    # Plot formatting
    ax.set_xticks(x_values)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=13, pad=15)

    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(zorder=0, linestyle=":")
    plt.tight_layout()

    return fig, ax


def plot_two_grouped_bars(data_1, data_2, x1_values=None, x2_values=None, x1_values_minor=None,
                          x2_values_minor=None, suptitle="Grouped bar plot", title_1="First bar plot",
                          title_2="Second bar plot", x1_label="", x2_label="", y1_label="",
                          y2_label="", labels=None):
    """
    Generate two grouped bar plots side by side.

    Parameters
    ----------
    data_1, data_2 : np.ndarray
        2D arrays with the data to display in the bar plots.
    x1_values, x2_values : list
        Values for the x-axes of the individual bar plots. If None,
        default integer values will be used.
    x1_values_minor, x2_values_minor : list
        Values for the minor ticks on the x-axes used for the grids. If
        None, the grid will use the corresponding major x-axis ticks.
    suptitle : str
        Title of the entire figure (default: 'Grouped bar plot').
    title_1, title_2 : str
        Titles for the individual bar plots (default: 'First bar plot',
        'Second bar plot').
    x1_label, x2_label : str
        Labels for the x-axes of the individual bar plots (default: '').
    y1_label, y2_label : str
        Labels for the y-axes of the individual bar plots (default: '').
    labels : list
        Labels for each group in the bar plots (assuming same for both plots).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated table plot.
    axs : matplotlib.axes._subplots.AxesSubplot
        Plot axes.
    """

    # Validate input
    if (
        not isinstance(data_1, np.ndarray)
        or not isinstance(data_2, np.ndarray)
        or data_1.shape[0] != data_2.shape[0]
    ):
        raise TypeError("The data must be np.ndarrays with the same number of rows.")

    # Prepare plotting parameters
    if x1_values is None:
        x1_values = np.arange(data_1.shape[1])
    if x2_values is None:
        x2_values = np.arange(data_2.shape[1])


    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # Plot each dataset
    total_width = 0.75
    for i, (data, x_values, x_values_minor, x_label, y_label, title) in enumerate(
        zip(
            [data_1, data_2],
            [x1_values, x2_values],
            [x1_values_minor, x2_values_minor],
            [x1_label, x2_label],
            [y1_label, y2_label],
            [title_1, title_2],
        )
    ):
        if data.shape[1] != len(x_values):
            raise ValueError(
                "The number of columns in the data must match the length of x_values."
            )

        n_bars = data.shape[0]
        bar_width = total_width / n_bars

        # Plot bars
        for j, dataset in enumerate(data):
            axs[i].bar(
                x_values - (total_width / 2) + (j + 0.5) * bar_width,
                dataset,
                width=bar_width,
                label=labels[j] if labels else None,
                zorder=2,
            )

        # Plot formatting
        axs[i].set_xticks(x_values)
        axs[i].tick_params(axis="both", labelsize=8)

        axs[i].set_xlabel(x_label, fontsize=10)
        axs[i].set_ylabel(y_label, fontsize=10)
        axs[i].set_title(title, fontsize=12, pad=15)

        axs[i].legend(loc="lower right", fontsize=7, framealpha=0.9)

        if x_values_minor is not None:
            axs[i].set_xticks(x_values_minor, minor=True)
            axs[i].grid(which="minor", axis="x", zorder=0, linestyle=":")
        else:
            axs[i].grid(axis="x", zorder=0, linestyle=":")
        axs[i].grid(axis="y", zorder=0, linestyle=":")

    # Add shared title and adjust layout
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    return fig, axs


def plot_grouped_bars_two_axes(data_1, data_2, x_values=None, x_values_minor=None,
                               title="Two y-axes grouped bar plot", x_label="", y1_label="",
                               y2_label="", y1_lim=None, y2_lim=None, labels=None):
    """
    Generate a grouped bar plot with a left y-axis with solid bars plotted to
    the left of the corresponding x-axis values and a right y-axis with hatched
    bars plotted to the right of the corresponding x-axis values.

    Parameters
    ----------
    data_1 : np.ndarray
        2D array with the data to display in the solid bar plot on the left y-axis.
    data_2 : np.ndarray
        2D array with the data to display in the hatched bar plot on the right y-axis.
    x_values : list
        Values for the x-axis. If None, default values will
        be used (e.g., ['Group 1', 'Group 2', ...]).
    x_values_minor : list
        Values for the minor ticks on the x-axis used for the grid.
        If None, the grid will use the major x-axis ticks.
    title : str
        Title of plot (default: 'Two y-axes grouped bar plot').
    x_label : str
        Label for the x-axis (default: '').
    y1_label, y2_label : str
        Labels for the left and right y-axes (default: '').
    y1_lim, y2_lim : tuple
        Limits for the left and right y-axes.
    labels : list
        Labels for each group in the bar plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated table plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        Plot axis.
    """

    # Validate input
    if not isinstance(data_1, np.ndarray) or not isinstance(data_2, np.ndarray):
        raise TypeError("The data must be np.ndarrays.")
    if data_1.shape != data_2.shape:
        raise ValueError("The two data arrays must have the same shape.")

    # Prepare plotting parameters
    if x_values is None:
        x_values = np.arange(data_1.shape[1])
    n_bars = data_1.shape[0]
    total_width = 0.4
    epsilon = 0.02
    bar_width = total_width / n_bars


    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=150)

    # Plot solid left bars on the left y-axis
    bars_colors = []
    for i, dataset in enumerate(data_1):
        bar1 = ax1.bar(
            x_values - (total_width + epsilon) + (i + 0.5) * bar_width,
            dataset,
            width=bar_width,
            label=labels[i] if labels else None,
            zorder=2,
        )
        bars_colors.append(bar1.patches[0].get_facecolor())

    # Plot hatched right bars on the right y-axis
    ax2 = ax1.twinx()
    plt.rcParams["hatch.linewidth"] = 1.7
    for i, dataset in enumerate(data_2):
        ax2.bar(
            x_values + epsilon + (i + 0.5) * bar_width,
            dataset,
            width=bar_width * 0.8,
            hatch="//",
            facecolor="white",
            edgecolor=bars_colors[i],
            linewidth=1.7,
            label=labels[i] if labels else None,
            zorder=2,
        )

    # Plot formatting
    ax1.set_xticks(x_values)
    ax1.set_xlabel(x_label, fontsize=10)

    ax1.set_ylabel(y1_label + " (solid bars)", fontsize=10)
    if y1_lim is not None:
        ax1.set_ylim(y1_lim)
    ax2.set_ylabel(y2_label + " (hatched bars)", fontsize=10)
    if y2_lim is not None:
        ax2.set_ylim(y2_lim)

    if x_values_minor is not None:
        ax1.set_xticks(x_values_minor, minor=True)
        ax1.grid(which="minor", axis="x", zorder=0, linestyle=":")
    else:
        ax1.grid(axis="x", zorder=0, linestyle=":")
    ax1.grid(axis="y", zorder=0, linestyle=":")

    ax1.set_title(title, fontsize=14, pad=15)

    # Create custom legend with colored rectangles
    if labels:
        legend_elements = []
        for i, label in enumerate(labels):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=bars_colors[i], label=label))
        ax2.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)

    plt.tight_layout()

    return fig, ax1


def plot_dots_two_axes(data_1, data_2, x_values=None, x_values_minor=None, title="Two y-axes dot plot",
                       x_label="", y1_label="", y2_label="", y1_lim=None, y2_lim=None, labels=None):
    """
    Generate two dots plots together one on the left y-axis and
    the other on the right y-axis with different symbols.

    Parameters
    ----------
    data_1, data_2 : np.ndarray
        2D array with the data to display in the dot plot on the
        left y-axis and right y-axis, respectively.
    x_values : list
        Values for the x-axis. If None, default integer values
        will be used.
    x_values_minor : list
        Values for the minor ticks on the x-axis used for the grid.
        If None, the grid will use the major x-axis ticks.
    title : str
        Title of plot (default: 'Two y-axes dot plot').
    x_label : str
        Label for the x-axis (default: '').
    y1_label, y2_label : str
        Labels for the left and right y-axes (default: '').
    y1_lim, y2_lim : tuple
        Limits for the left and right y-axes.
    labels : list
        Labels for each group in the bar plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated table plot.
    axs  : tuple[matplotlib.axes._subplots.AxesSubplot]
        Plot axes.
    """

    # Validate input
    if (
        not isinstance(data_1, np.ndarray)
        or not isinstance(data_2, np.ndarray)
        or data_1.shape != data_2.shape
    ):
        raise TypeError("The data must be np.ndarrays with the same shape.")

    # Prepare plotting parameters
    if x_values is None:
        x_values = np.arange(data_1.shape[1])
    n_datasets = data_1.shape[0]
    total_span = 0.75
    dot_spacing = total_span / n_datasets

    # Create figure
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    ax2 = ax1.twinx()

    # Plot data
    for i, (dataset_1, dataset_2) in enumerate(zip(data_1, data_2)):
        sc1 = ax1.scatter(
            x_values - (total_span / 2) + (i + 0.5) * dot_spacing,
            dataset_1,
            label=labels[i] if labels else None,
            marker="o",
            linewidth=1,
            zorder=2,
        )
        sc2 = ax2.scatter(
            x_values - (total_span / 2) + (i + 0.5) * dot_spacing,
            dataset_2,
            label=labels[i] if labels else None,
            marker="s",
            linewidth=1,
            zorder=2,
        )

        # Adjust markers' colors
        edge_color = sc1.get_facecolor()[0]
        edge_color_rgb = colors.to_rgb(edge_color)
        face_color = tuple(min(1, max(0, c * 1.3)) for c in edge_color_rgb)

        sc1.set_edgecolor(edge_color)
        sc1.set_facecolor(face_color)

        sc2.set_edgecolor(edge_color)
        sc2.set_facecolor(face_color)


    # Plot formatting
    ax1.set_xticks(x_values)
    ax1.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax1.set_xlabel(x_label, fontsize=11)

    ax1.set_ylabel(y1_label + " (circles)", fontsize=11)
    if y1_lim is not None:
        ax1.set_ylim(y1_lim)
    ax2.set_ylabel(y2_label + " (squares)", fontsize=11)
    if y2_lim is not None:
        ax2.set_ylim(y2_lim)

    if x_values_minor is not None:
        ax1.set_xticks(x_values_minor, minor=True)
        ax1.grid(which="minor", axis="x", zorder=0, linestyle=":")
    else:
        ax1.grid(axis="x", zorder=0, linestyle=":")

    # Create custom legend with colored rectangles
    if labels:
        legend_elements = []
        for i, label in enumerate(labels):
            # Get the color from the first scatter plot
            color = ax1.collections[i].get_facecolors()[0]
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label))
        ax1.legend(handles=legend_elements, fontsize=8)

    ax1.set_title(title, fontsize=13, pad=15)
    plt.tight_layout()

    return fig, (ax1, ax2)
