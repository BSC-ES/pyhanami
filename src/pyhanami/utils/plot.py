import cmocean
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt

from pyhanami import config
from scipy.stats import bootstrap
from cartopy.util import add_cyclic_point
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm


def time_series_plot(time_series, title='Mean time series', y_label='', labels=None, time_freq='annual', start_year=None, end_year=None):
    """ 
    Generate time series plot of one or more ensembles, including the 2.5th, 5th, 75th and 97.5th percentiles.

    Parameters
    ----------
    time_series (xarray.DataArray or list of xr.DataArray): Time series data.
    title (str): Title of the plot.
    y_label (str): Label for the y-axis.
    labels (list[str]): Labels for each time series.
    time_freq (str): Time frequency.
    start_year, end_year (int): Years for filtering.

    Returns
    -------
    fig (matplotlib.figure.Figure): Generated plot.
    ax (matplotlib.axes._subplots.AxesSubplot): Plot axis.
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
    
    if not isinstance(title, str) or not isinstance(y_label, str):
        raise TypeError("Both 'title' and 'y_label' must be strings.")
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
            line, = ax.plot(dates, mean_series.values, lw=2, ls='-.', label=label)
            color = line.get_color()

            # Plot individual ensemble members
            # for i in range(series.sizes['realization']):
            #     ax.plot(common_years, series.isel(realization=i).values, color=color, alpha=0.2)

            # Compute confidence intervals (bootstrap of mean)
            q025, q975 = [], []
            for i in range(len(dates)):
                data = series.isel(time=i).values
                bs_res = bootstrap((data,), np.mean, confidence_level=0.95)
                q025.append(bs_res.confidence_interval.low)
                q975.append(bs_res.confidence_interval.high)
            ax.fill_between(dates, q025, q975, facecolor=color, alpha=0.6)

            # Plot ensemble spread (5th–95th percentile)
            series = series.chunk({"realization": -1})
            quantiles = series.quantile([0.05, 0.95], dim='realization', skipna=True)
            ax.fill_between(dates, quantiles[0], quantiles[1], facecolor=color, alpha=0.2)

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
    ax.set_xlabel('time', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(title, fontsize=18)
    ax.legend(fontsize=14)
    ax.grid()

    plt.tight_layout()

    return fig, ax


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
    significant (np.ndarray): Mask for significance hatching.
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
    
    if not isinstance(title, str) or not isinstance(cb_label, str):
        raise TypeError("Both 'title' and 'cb_label' must be strings.")
    
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
        if significant.shape != data.shape:
            raise ValueError(f"Mask shape {significant.shape} does not match data shape {data.values.shape}.")
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

    return fig, ax


def matrix_plot(eff_sizes, test_results, test=4, title='Effect sizes replicability test', variables=None, 
                seasons=None, regions=None):
    """ 
    Generate a matrix plot with effect sizes and results of the replicability test for the selected statistical test/s.

    Parameters
    ----------
    eff_sizes (np.ndarray): Effect sizes between the scores with shape (n_rows, n_cols, n_indices). 
    test_results (np.ndarray): Results of the statistical tests with shape (n_rows, n_cols, 4).
    test (int): Statistical test to use, corresponding to last dimension of test_results (0: KS-test, 1: T-test, 2: U-test, 3: B-test, 4: All).
    title (str): Title of the plot.
    variables (dict): Dictionary with variables to be included in the plot and their descriptions.
    seasons (list): List of seasons to include in the plot.
    regions (list): List of regions to include in the plot.

    Returns
    -------
    fig (matplotlib.figure.Figure): Generated matrix plot.
    ax (matplotlib.axes._subplots.AxesSubplot): Plot axis.
    """

    # Validate inputs
    if not isinstance(eff_sizes, np.ndarray) or not isinstance(test_results, np.ndarray):
        raise TypeError("The effect sizes and the test results must be numpy arrays.")
    if eff_sizes.shape[:2] != test_results.shape[:2]:
        raise ValueError("Mismatched spatial dimensions between effect sizes and test results.")
    if test not in range(5):
        raise ValueError("Invalid test index. Must be: 0 (KS-test), 1 (T-test), 2 (U-test), 3 (B-test), 4 (All).")
    if not isinstance(title, str):
        raise TypeError("'title' must be a string.")

    # Prepare parameters
    if variables is None:
        variables = list(config.VARIABLES.keys())
    if seasons is None:
        seasons = config.SEASONS
    if regions is None:
        regions = list(config.REGIONS.keys())

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