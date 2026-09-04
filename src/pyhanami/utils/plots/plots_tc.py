import cmocean
import numpy as np
import matplotlib.pyplot as plt

from pyhanami.utils import data_general
from pyhanami.config import config_params
from pyhanami.utils.plots import plots_general
from matplotlib.colors import LinearSegmentedColormap


def plot_linear_cycle_one(data, x_values, x_ticks, x_label, y_label, data_labels, title, output_path=None,
                           plot_name="Linear cycle plot", plot_filename="linear_cycle_plot"):
    """
    Generate and save/display a linear plot of monthly/interannual cycles.

    Parameters
    ----------
    data : list[np.ndarray]
        Datasets to plot (with `model` coordinate).
    x_values : np.ndarray
        Values for the x-axis, also used for ticks positions.
    x_ticks : list[str]
        Labels for the x-axis ticks.
    x_label, y_label : str
        Axes labels.
    data_labels : list[str]
        Labels for the data series to plot.
    title : str
        Plot title.
    output_path : str, optional
        Path to save the linear plot. If None, the plot is displayed but not saved.
    plot_name : str
        Name of the plot for display messages (default: "Linear cycle plot").
    plot_filename : str
        Base name for the plot file (default: "linear_cycle_plot").
    """

    # Create plot
    fig, axs = plt.subplots(1,1, figsize=(10, 6), dpi=150)
    for dataset, label in zip(data, data_labels):
        axs.plot(x_values, dataset, "o-", markersize=4, linewidth=1.2, label=label)

    # Customize plot
    axs.set_xticks(x_values)
    axs.set_xticklabels(x_ticks, rotation=45, ha="right")
    axs.set_xlabel(x_label, fontsize=12)
    axs.set_ylabel(y_label, fontsize=12)

    axs.set_title(title, fontsize=16)
    axs.grid(linestyle=":")
    axs.legend()

    # Save/display plot
    plots_general.save_or_show_plot(
        fig,
        output_path,
        plot_name=plot_name,
        plot_filename=plot_filename,
        custom_name=False,
    )

    return


def plot_linear_cycles(data, data_names, start_year, end_year, output_path=None):
    """
    Generate and save/display linear plots of monthly and interannual cycles for multiple TC metrics
    comparing multiple datasets.

    Parameters
    ----------
    data : list[xr.Dataset]
        List of datasets to plot.
    data_names : list[str]
        List of dataset names (assuming observations/reanalyses as the first datasets).
    start_year, end_year : int
        Start and end years of the evaluation period.
    output_path : str, optional
        Path to save the linear plots. If None, the plots are displayed but not saved.
    """

    # Prepare invariant arrays/strings
    months = np.arange(1, 13, dtype=int)
    months_ticks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = np.arange(start_year, end_year + 1, dtype=int)

    name_file = '_'.join([name.replace(" ", "-") for name in data_names])
    year_range = f"{start_year}-{end_year}"

    # Prepare metrics metadata
    metrics_metadata = data_general.load_yaml_file(config_params.TCS_METRICS_PATH)
    temporal_metrics = [
        metric
        for metric in metrics_metadata
        if metrics_metadata[metric]["temporal"] == True
    ]


    # Create two plots for each metric
    for metric in temporal_metrics:
        name = metrics_metadata[metric]["short_name"]
        units = metrics_metadata[metric]["units"]

        # Prepare plotting parameters
        y_label = f"{name} ({units})"
        month_title = f"{name} seasonal cycle ({year_range})"
        year_title = f"{name} interannual cycle"

        # Prepare data for plotting
        data_ref_month = [
            data[0].isel(model=i)[f"per_month_{metric}"].values
            for i in range(data[0].model.shape[0]-1)
        ]
        data_sim_month = [ds.isel(model=-1)[f"per_month_{metric}"].values for ds in data]
        data_plot_month = np.stack([*data_ref_month, *data_sim_month])  # Ensure that all arrays have the same shape

        data_ref_year = [
            data[0].isel(model=i)[f"per_year_{metric}"].values
            for i in range(data[0].model.shape[0]-1)
        ]
        data_sim_year = [ds.isel(model=-1)[f"per_year_{metric}"].values for ds in data]
        data_plot_year = np.stack([*data_ref_year, *data_sim_year])  # Ensure that all arrays have the same shape

        # Create line plot for monthly cycles
        plot_linear_cycle_one(
            data=data_plot_month,
            x_values=months,
            x_ticks=months_ticks,
            x_label="month",
            y_label=y_label,
            data_labels=data_names,
            title=month_title,
            output_path=output_path,
            plot_name=f"Linear seasonal cycle plot for TC {name}",
            plot_filename=f"tcs_{name.lower()}_seasonal_cycle_plot_{name_file}_{year_range}",
        )

        # Create line plot for interannual cycles
        plot_linear_cycle_one(
            data=data_plot_year,
            x_values=years,
            x_ticks=years,
            x_label="year",
            y_label=y_label,
            data_labels=data_names,
            title=year_title,
            output_path=output_path,
            plot_name=f"Linear interannual cycle plot for TC {name}",
            plot_filename=f"tcs_{name.lower()}_interannual_cycle_plot_{name_file}_{year_range}",
        )

    return


def plot_spatial(data, data_name, year_range, bin_size, output_path=None, clon=0):
    """
    Generate and save/display spatial plots comparing simulations with IBTrACS data for multiple TC metrics.

    Parameters
    ----------
    data : xr.Dataset
        Dataset to plot.
    data_name : str
        Name of the dataset.
    year_range : str
        Year range of the evaluation period.
    bin_size : float
        Size of the bins in degrees used for computing the TCs metrics.
    output_path : str, optional
        Path to save the spatial plots. If None, the plots are displayed but not saved.
    clon : int
        Central longitude for the spatial maps (default: 0).
    """

    # Prepare metrics metadata
    metrics_metadata = data_general.load_yaml_file(config_params.TCS_METRICS_PATH)
    spatial_metrics = [
        metric
        for metric in metrics_metadata
        if metrics_metadata[metric]["spatial"] == True
    ]

    # Create a modified colormap with white for NaN values
    cmap_modified = cmocean.cm.thermal_r.copy()
    # cmap_modified.set_bad('white')


    # Create two figures for each metric
    name_file = data_name.replace(" ", "-")
    for metric in spatial_metrics:
        metric_name = metrics_metadata[metric]["short_name"]
        metric_units = metrics_metadata[metric]["units"]
        colorbar_label = f"{metric_name} ({metric_units})"


        # Generate figure with two spatial plots
        two_plots_title = f"TC {metric_name} density for {bin_size}°x{bin_size}° cells ({year_range})"

        # Prepare data (set 0 to NaN for better visualization)
        data_two_plots = data.sel(model=["IBTrACS", data_name])[f"spatial_abs_{metric}"]
        data_two_plots = data_two_plots.where(data_two_plots != 0)

        # Create and save/display figure
        two_spatial_plots, _ = plots_general.two_spatial_plots(
            data_two_plots.sel(model="IBTrACS"),
            data_two_plots.sel(model=data_name),
            clon=clon,
            title_1="IBTrACS",
            title_2=data_name,
            suptitle=two_plots_title,
            cb_label=colorbar_label,
            cmap=cmap_modified,
        )

        plots_general.save_or_show_plot(
            two_spatial_plots,
            output_path,
            plot_filename=f"tcs_{metric_name.lower()}_spatial_abs_plot_{name_file}_{year_range}_clon_{clon}",
            plot_name=f"Spatial plot for TC {metric_name} for '{data_name}' and 'IBTrACS'",
            custom_name=False,
        )


        # Generat figure with bias spatial plot
        bias_title = (f"TC {metric_name} bias ('{data_name}' - 'IBTrACS') "
                      f"for {bin_size}°x{bin_size}° cells ({year_range})")

        # Prepare data
        data_plot_bias = data[f'spatial_bias_{metric}'].sel(model=data_name)
        limit = np.ceil(np.nanmax(np.abs(data_plot_bias.values)))
        levels = np.linspace(-limit, limit, 13)

        # Create and save/display figure
        spatial_bias_plot, _ = plots_general.plot_spatial(
            data_plot_bias,
            clon=clon,
            title=bias_title,
            cb_label=f"bias in {metric_name}",
            cmap=LinearSegmentedColormap.from_list("RedBlue", ["#B2182B", "white", "#0072B2"],),
            levels=levels,
        )

        plots_general.save_or_show_plot(
            spatial_bias_plot,
            output_path,
            plot_filename=f"tcs_{metric_name.lower()}_spatial_bias_plot_{name_file}_{year_range}_clon_{clon}",
            plot_name=f"Spatial bias plot for TC {metric_name} for '{data_name}' and 'IBTrACS'",
            custom_name=False,
        )

    return
