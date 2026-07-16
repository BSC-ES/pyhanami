import numpy as np

from pyhanami.utils import data_general
from pyhanami.config import config_params
from pyhanami.utils.plots import plots_general

VARIABLES = data_general.load_yaml_file(config_params.VARIABLES_PATH)


def evaluation_table_metadata(data_names, rows_to_remove=0):
    """
    Prepare strings for file names and rows in scalar scores tables.

    Parameters
    ----------
    data_names : list[str]
        List of dataset names (assuming observations as the first dataset).
    rows_to_remove : int
        Number of datasets to remove from the top of the table (generally, reference
        and reanalyses datasets) (default: 0).
    
    Returns
    -------
    rows : list[str]
        List of dataset names to be used as row labels in the table.
    name_file : str
        Name of the file to save the table plot.
    reference : bool
        Whether to display reference values in the first row (not colored) and
        reanalyses values (if any) in the subsequent rows.
    """
    data_names_converted = [name.replace(" ", "-") for name in data_names]

    if rows_to_remove == 0:
        rows = data_names
        name_file = f"{'_'.join(data_names_converted[1:])}_ref_{data_names_converted[0]}"
        reference = True
    else:
        rows = data_names[rows_to_remove:]
        name_file = f"{'_'.join(data_names_converted[rows_to_remove:])}_no_ref_{data_names_converted[0]}"
        reference = False

    return rows, name_file, reference


def evaluation_symmetric_colorbar_limits(data, start_row=1):
    """
    Compute symmetric colorbar limits for scalar scores tables.

    Parameters
    ----------
    data : list[xr.DataArray]
        List of datasets to plot.
    start_row : int
        Row index to start computing the limits (default: 1, to skip the reference row).

    Returns
    -------
    limits : np.ndarray
        Array of symmetric colorbar limits for each column in the table.
    """

    maxs = np.max(np.abs(data[start_row:, :]), axis=0)
    limits = np.stack([-maxs, maxs], axis=1)

    return limits


def general_evaluation_scores_table(data, data_names, year_range, var_names=None, ensemble=False,
                                    output_path=None, reference=True):
    """
    Generate and save/display table plot with general scalar scores for the given dataset(s) 
    and variable(s).

    Parameters
    ----------
    data : list[xr.Dataset]
        List of datasets to plot.
    data_names : list[str]
        List of dataset names (assuming observations as the first dataset).
    year_range : str
        Year range of the evaluation period.
    var_names : str or list[str], optional
        Climate variable(s) name(s). If None, all variables in the analysis will be used.
    ensemble : bool
        Whether the simulation datasets are single members or ensembles (default: False).
    output_path : str, optional
        Path to save the table plot. If None, the table is displayed but not saved.
    reference : bool
        Whether to display reference values in the first row (not colored) (default: True).
    """

    # Prepare plotting parameters
    rows, name_file, _ = evaluation_table_metadata(data_names, rows_to_remove=0 if reference else 1)
    if reference:
        data_ref = np.array([1, 1, 1])

    cbar_ticks = ["Worst performance (0)", " ", "Best performance (1)"]
    colors = ("RedGreen", ["tab:red", "white", "tab:green"])


    # Plot each variable separately
    for var_name in var_names:
        var_name_title = VARIABLES[var_name]["long_name"]
        if ensemble:
            # cols = [
            #   ' ',
            #   r'$\overline{\text{BIAS}}$',
            #   r'$\overline{\text{eBIAS}}$',
            #   r'$\overline{\text{RMSE}}$',
            #   r'$\overline{\text{eRMSE}}$',
            #   r'$\overline{r}_{xy}$'
            # ]
            cols = [
                " ",
                r"$\overline{\text{eBIAS}}$",
                r"$\overline{\text{eRMSE}}$",
                r"$\overline{r}_{xy}$",
            ]
            title = f"Scalar scores for {var_name_title} (ensemble mean) ({year_range})"
        else:
            # cols = [' ', 'BIAS', 'eBIAS', 'RMSE', 'eRMSE', r'$r_{xy}$']
            cols = [" ", "eBIAS", "eRMSE", r"$r_{xy}$"]
            title = f"Scalar scores for {var_name_title} ({year_range})"
        limits = np.repeat([[0, 1]], len(cols) - 1, axis=0)

        # Prepare plot data
        data_sim = [
            dataset[["bias_rel", "rmse_rel", "pcorr"]]
            .sel(variable=var_name)
            .to_array()
            .values
            for dataset in data
        ]

        # Ensure that all arrays have the same shape
        data_plot = np.stack([data_ref, *data_sim]) if reference else np.stack(data_sim)


        # Generate and save/display plot
        general_scores_plot, _ = plots_general.plot_table(
            data_plot,
            title=title,
            col_labels=cols,
            row_labels=rows,
            cbar_ticks=cbar_ticks,
            cbar_colors=colors,
            limits=limits,
            reference=reference,
            decimals=3,
        )

        plots_general.save_or_show_plot(
            general_scores_plot,
            output_path,
            plot_filename=f"general_scalar_scores_table_{var_name}_{name_file}_{year_range.replace(' ', '-')}",
            plot_name="General scalar scores table plot",
        )

    return


def iso_evaluation_scores_table(data, data_names, year_range, correct_pc=False, output_path=None,
                                reference=True):
    """
    Generate and save/display table plot with ISO scalar scores for the given dataset(s).

    Parameters
    ----------
    data : list[dict]
        List of datasets to plot.
    data_names : list[str]
        List of dataset names (assuming observations as the first dataset).
    year_range : str
        Year range of the evaluation period.
    correct_pc : bool
        Whether simulated PCs have been adjusted with observational data (default: False).
    output_path : str, optional
        Path to save the table plot. If None, the table is displayed but not saved.
    reference : bool
        Whether to display reference values in the first row (not colored) (default: True).
    """

    # Prepare plotting parameters
    rows, name_file, _ = evaluation_table_metadata(data_names, rows_to_remove=0 if reference else 1)
    if reference:
        data_ref = np.array([1.0, 1.0, 1.0, 1.0])

    cols_scalar_scores = [
        "",
        r"  $\alpha$  ",
        r"    $R$    ",
        r"  $\sigma$  ",
        r"$\text{TSS}$",
    ]  # Same number of characters needed to get same column width
    cbar_ticks = ["Worst performance", " ", "Best performance"]
    colors = ("RedGreen", ["tab:red", "white", "tab:green"])

    plot_title = f"ISO scalar scores ({year_range})"
    if correct_pc:
        plot_title += " (corrected PCs)"
        name_file += "_corrected_PCs"

    # Prepare plot data
    data_sim = [
        [
            dataset["alpha"],
            dataset["R"],
            dataset["sigma"],
            dataset["TSS"],
        ]
        for dataset in data
    ]

    # Ensure that all arrays have the same shape
    data_plot = np.stack([data_ref, *data_sim]) if reference else np.stack(data_sim)


    # Generate and save/display plot
    iso_scores_plot, _ = plots_general.plot_table(
        data_plot,
        title=plot_title,
        col_labels=cols_scalar_scores,
        row_labels=rows,
        cbar_ticks=cbar_ticks,
        cbar_colors=colors,
        reference=reference,
        decimals=2,
    )

    plots_general.save_or_show_plot(
        iso_scores_plot,
        output_path,
        plot_filename=f"iso_scalar_scores_table_{name_file}_{year_range.replace(' ', '-')}",
        plot_name="ISO scalar scores table plot",
    )

    return


def mjo_evaluation_table_metadata_general(data_names, on=True, reference=True):
    """
    Prepare strings for file names and rows, and data for MJO scalar scores tables.

    Parameters
    ----------
    data_names : list[str]
        List of dataset names (assuming observations as the first dataset).
    on : bool
        Whether to include "on the observations" in the row labels (default: True).
    reference : bool
        Whether to display reference values in the first row (not colored) (default: True).
    
    Returns
    -------
    rows : list[str]
        List of dataset names to be used as row labels in the table.
    name_file : str
        Name of the file to save the table plot.
    """

    # Prepare strings
    data_names_converted = [name.replace(" ", "-") for name in data_names]
    obs_name = data_names_converted[0]

    if on:
        rows = (
            [f"{sim_name} on {obs_name}" for sim_name in data_names[1:]]
            + [f"{sim_name} on {sim_name}" for sim_name in data_names[1:]]
        )
    else:
        rows = data_names[1:]

    # Add references if needed
    if reference:
        rows.insert(0, obs_name)
        name_file = f"{'_'.join(data_names_converted[1:])}_ref_{data_names_converted[0]}"
    else:
        name_file = f"{'_'.join(data_names_converted[1:])}_no_ref_{data_names_converted[0]}"

    return rows, name_file


def mjo_evaluation_table_metadata_per_method(method_name, data, reference=True, mjo_vars=['ua850', 'ua200', 'rlut']):
    """
    Prepare data, column names, title, and plot name for MJO scalar scores tables based on the evaluation method.

    Parameters
    ----------
    method_name : str
        Name of the MJO evaluation method.
    data : list[xr.DataArray]
        List of datasets to plot.
    reference : bool
        Whether to display reference values in the first row (not colored) (default: True).
    mjo_vars : list[str]
        Climate variables used for the MJO analysis (default: ['ua850', 'ua200', 'rlut']).
    
    Returns
    -------
    data_plot : np.ndarray
        Array of data to be used in the table plot with proper shape.
    col_names : list[str]
        List of column names to be used in the table.
    title : str
        Title of the table plot.
    plot_name : str
        Name of the plot to be used for saving/displaying the table plot.
    """

    # Prepare data with original shape
    if method_name in ["ceof_corr_table", "power_bias_table"]:
        data_original = np.array([ds.isel(dataset=-1).values for ds in data])
    else:
        data_original = np.array([ds.isel(dataset=i+1).values for i in range(2) for ds in data])

    if reference:
        data_original = np.insert(data_original, 0, data[0].isel(dataset=0).values, axis=0)


    # Reshape data for plotting and define strings based on method
    if method_name == "ceof_corr_table":
        data_plot = data_original.reshape(data_original.shape[0], -1)
        col_names = [
            rf"$r_{{\text{{{var}}}, {mode+1}}}$"
            for var in data[0]["variable"].values
            for mode in data[0]["mode"].values
        ]

        title = "Correlation between Combined EOFs"
        plot_name = "Correlation in Combined EOFs table plot"

    elif method_name == "ceof_bias_table":
        data_plot = data_original.reshape(data_original.shape[0], -1)
        col_names = [
            rf"$b_{{\text{{expl var}}, 1}}$ (%)",
            rf"$b_{{\text{{expl var}}, 2}}$ (%)",
            # *[
            #     rf"$b_{{\text{{expl var}}, {mode}}}$ (%)"
            #     for mode in data[0]["mode"].values
            # ],
            rf"$b_{{\text{{max lead-lag corr}}}}$ (-)",
            rf"$b_{{\text{{MJO period}}}}$ (days)",
        ]

        title = "Bias derived from Combined EOF analysis"
        plot_name = "Bias derived from Combined EOF analysis table plot"

    elif method_name == "mean_amplitude_bias_table":
        data_plot = data_original
        col_names = [rf"$\overline{{b}}_{{ph\, {phase}}}$" for phase in data[0]["phase"].values]

        title = "Bias in climatological mean MJO amplitude per phase"
        plot_name = "Bias in climatological mean MJO amplitude per phase table plot"

    elif method_name == "active_days_bias_table":
        data_plot = data_original
        col_names = [rf"$\overline{{b}}_{{ph\, {phase}}}$ (days)" for phase in data[0]["phase"].values]
        # (['Dataset \ Phase'], [str(phase) for phase in sdata[0].phase.values])

        title = "Bias in climatological active MJO days per phase"
        plot_name = "Bias in climatological active MJO days per phase table plot"

    elif method_name == "power_bias_table":
        data_plot = data_original.reshape(data_original.shape[0], -1)
        col_names = [
            *[rf"$b^{{\text{{E/W}}}}_{{\text{{{var_name}}}}}$" for var_name in mjo_vars],
            *[rf"$b_{{\text{{{var_name}}}}}^{{\text{{E/O}}}}$" for var_name in mjo_vars],
            *[
                rf"$b_{{\text{{{var_name}}}}}^{{\text{{period}}}}$ (days)"
                for var_name in mjo_vars
            ],
        ]

        title = "Bias derived from power spectra"
        plot_name = "Bias derived from power spectra table plot"

    else:
        raise ValueError(f"Method '{method_name}' is not recognized for MJO evaluation.")

    return data_plot, col_names, title, plot_name


def mjo_evaluation_scores_table(data, data_names, year_range, method_name, data_res,
                                mjo_vars=['ua850', 'ua200', 'rlut'], output_path=None, reference=True):
    """
    Generate and save/display table plot with MJO bias scores for the given dataset(s).

    Parameters
    ----------
    data : list[xr.DataArray]
        List of datasets to plot.
    data_names : list[str]
        List of dataset names (assuming observations/reanalyses as the first datasets).
    year_range : str
        Year range of the evaluation period.
    method_name : str
        Name of the MJO evaluation method.
    data_res : float
        Resolution of the data used for the MJO analysis.
    mjo_vars : list[str]
        Climate variables used for the MJO analysis (default: ['ua850', 'ua200', 'rlut']).
    output_path : str, optional
        Path to save the table plot. If None, the table is displayed but not saved.
    reference : bool
        Whether to display reference values in the first row (not colored) (default: True).
    """

    # Prepare plotting general parameters
    projected_on = False if method_name in ["ceof_corr_table", "power_bias_table"] else True
    rows, name_file = mjo_evaluation_table_metadata_general(data_names, on=projected_on, reference=reference)


    # Prepare data for plotting and method-specific parameters
    data_plot, col_names, plot_title, plot_name = mjo_evaluation_table_metadata_per_method(
        method_name, data, reference=reference, mjo_vars=mjo_vars
    )

    col_names = [f"{data_res}° x {data_res}°"] + col_names
    plot_title += f" ({year_range})"
    plot_filename = f"{method_name}_{name_file}_{year_range.replace(' ', '-')}"

    if method_name == "ceof_corr_table":
        cbar_ticks = ["Negative correlation (-1)", "No correlation (0)", "Positive correlation (1)"]
        cbar_colors = ("RedGreen", ["tab:red", "white", "tab:green"])
        cbar_limits = np.repeat([[-1, 1]], len(col_names) - 1, axis=0)
    else:
        cbar_ticks = ["Negative bias", "No bias", "Positive bias"]
        cbar_colors = ("RedGreen", ["tab:red", "white", "tab:green"])  # ("BlueRed", ['tab:blue', 'white', 'tab:red'])
        cbar_limits = evaluation_symmetric_colorbar_limits(data_plot, start_row=1 if reference else 0)

    decimals = 0 if method_name == "active_days_bias_table" else 2


    # Generate and save/display plot
    mjo_table_plot, _ = plots_general.plot_table(
        data_plot,
        title=plot_title,
        col_labels=col_names,
        row_labels=rows,
        cbar_ticks=cbar_ticks,
        cbar_colors=cbar_colors,
        limits = cbar_limits,
        reference=reference,
        decimals=decimals
    )

    plots_general.save_or_show_plot(
        mjo_table_plot,
        output_path,
        plot_filename=plot_filename,
        plot_name=plot_name,
    )

    return


def mjo_evaluation_scores_two_tables(data_1, data_2, data_names, year_range, output_path=None, reference=True):
    """
    Generate and save/display two table plots with climatological (annually averaged) mean
    MJO amplitude and active MJO days per phase bias.

    Parameters
    ----------
    data_1 : xr.DataArray
        List of datasets for the first table to plot (climatological mean MJO amplitude bias).
    data_2 : xr.DataArray
        List of datasets for the second table to plot (active MJO days per phase bias).
    data_names : list[str]
        List of dataset names (assuming observations/reanalyses as the first datasets), same
        for both tables.
    year_range : str
        Year range of the evaluation period.
    output_path : str, optional
        Path to save the table plot. If None, the table is displayed but not saved.
    reference : bool
        Whether to display reference values in the first row (not colored) (default: True).
    """

    # Prepare plotting general parameters
    rows, name_file = mjo_evaluation_table_metadata_general(data_names, on=True, reference=reference)
    rows = [rows, rows]

    # Prepare data for plotting and method-specific parameters
    data_plot_1, col_names_1, _, _ = mjo_evaluation_table_metadata_per_method(
        'mean_amplitude_bias_table', data_1, reference=reference
    )
    col_names_1 = ["Bias in Climatological mean amplitude"] + col_names_1

    data_plot_2, col_names_2, _, _ = mjo_evaluation_table_metadata_per_method(
        'active_days_bias_table', data_2, reference=reference
    )
    col_names_2 = ["Bias in Climatological active days"] + col_names_2

    col_names_all = [col_names_1, col_names_2]
    plot_title = f"Bias in climatological MJO activity per phase ({year_range})"
    plot_filename = f"activity_per_phase_bias_tables_{name_file}_{year_range.replace(' ', '-')}"
    plot_name = "Bias in climatological MJO activity per phase table plot"

    cbar_ticks = ["Negative bias", "No bias", "Positive bias"]
    cbar_colors = ("RedGreen", ["tab:red", "white", "tab:green"])  # ("BlueRed", ['tab:blue', 'white', 'tab:red'])

    limits_1 = evaluation_symmetric_colorbar_limits(data_plot_1, start_row=1 if reference else 0)
    limits_2 = evaluation_symmetric_colorbar_limits(data_plot_2, start_row=1 if reference else 0)
    cbar_limits = [limits_1, limits_2]

    references = [reference, reference]
    decimals = [2, 2]

    # Generate and save/display plot
    mjo_two_tables_plot, _ = plots_general.plot_two_tables(
        data_plot_1,
        data_plot_2,
        title=plot_title,
        col_labels=col_names_all,
        row_labels=rows,
        cbar_ticks=cbar_ticks,
        cbar_colors=cbar_colors,
        limits = cbar_limits,
        references=references,
        decimals=decimals
    )

    plots_general.save_or_show_plot(
        mjo_two_tables_plot,
        output_path,
        plot_filename=plot_filename,
        plot_name=plot_name,
    )

    return


def tc_evaluation_table_data(data, rows_to_remove=0):
    """
    Prepare data for TCs scalar scores tables.

    Parameters
    ----------
    data : list[xr.DataArray]
        List of datasets to plot.
    rows_to_remove : int
        Number of datasets to remove from the top of the table (generally, reference
        and reanalyses datasets) (default: 0).

    Returns
    -------
    data_plot : np.ndarray
        Array of data to be used in the table plot, with shape (n_datasets, n_scores).
    """

    # Retrieve scores for simulation datasets (last model in each dataset)
    data_sim = [
        dataset.isel(model=-1).to_array().values
        for dataset in data
    ]

    # Ensure that all arrays have the same shape
    if rows_to_remove == 0:
        data_ref = data[0].isel(model=slice(None, -1)).to_array().values.transpose()
        data_plot = np.stack([*data_ref, *data_sim])
    else:
        data_plot = np.stack(data_sim)

    return data_plot


def tc_evaluation_bias_scores_table(data, data_names, year_range, bias_type, bin_size, output_path=None,
                                    rows_to_remove=0):
    """
    Generate and save/display table plot with TCs bias scores for the given dataset(s).

    Parameters
    ----------
    data : list[xr.DataArray]
        List of datasets to plot.
    data_names : list[str]
        List of dataset names (assuming observations/reanalyses as the first datasets).
    year_range : str
        Year range of the evaluation period.
    bias_type : str
        Type of bias to evaluate ("climatological" or "storm").
    bin_size : float
        Size of the bins in degrees used computing the TCs metrics.
    output_path : str, optional
        Path to save the table plot. If None, the table is displayed but not saved.
    rows_to_remove : int
        Number of datasets to remove from the top of the table (generally, reference
        and reanalyses datasets) (default: 0).
    """

    # Prepare strings for rows, file names and titles
    rows, name_file, reference = evaluation_table_metadata(data_names, rows_to_remove)
    plot_title = f"Global {bias_type} mean bias ({year_range})"

    # Prepare column labels based on bias type
    metrics_metadata = data_general.load_yaml_file(config_params.TCS_METRICS_PATH)
    if bias_type == "climatological":
        cols_bias = np.append(
            [f"{bin_size}° x {bin_size}°"],
            [
                rf"$\overline{{b}}_{{clim,{metric}}}$ ({metrics_metadata[metric]['units']})"
                for metric in metrics_metadata
                if metrics_metadata[metric]["temporal"] == True
            ],
        )
    elif bias_type == "storm":
        cols_bias = np.append(
            [f"{bin_size}° x {bin_size}°"],
            [
                rf"$\overline{{b}}_{{storm,{metric}}}$ ({metrics_metadata[metric]['units']})"
                for metric in metrics_metadata
                if metrics_metadata[metric]["temporal"] == True and metric != "count"
            ],
        )
    else:
        raise ValueError("Invalid bias type. Must be 'climatological' or 'storm'.")

    # Prepare plot data
    data_plot = tc_evaluation_table_data(data, rows_to_remove)

    # Prepare colorbar parameters
    cbar_ticks_bias = ["Negative bias", "No bias", "Positive bias"]
    colors_bias = ("RedGreen", ["tab:red", "white", "tab:green"],)
    # Alternative: ("BlueRed", ['tab:blue', 'white', 'tab:red'])

    limits_bias = evaluation_symmetric_colorbar_limits(data_plot, start_row=1 if reference else 0)


    # Generate and save/display plot
    tc_bias_plot, _ = plots_general.plot_table(
        data_plot,
        title=plot_title,
        col_labels=cols_bias,
        row_labels=rows,
        cbar_ticks=cbar_ticks_bias,
        cbar_colors=colors_bias,
        limits = limits_bias,
        reference=reference,
    )

    plots_general.save_or_show_plot(
        tc_bias_plot,
        output_path,
        plot_filename=f"tc_{bias_type}_bias_scalar_scores_table_{name_file}_{year_range.replace(' ', '-')}",
        plot_name=f"TC {bias_type} bias scalar scores table plot",
    )

    return


def tc_evaluation_correlation_scores_table(data, data_names, year_range, correlation_type, bin_size, output_path=None,
                                           rows_to_remove=0):
    """
    Generate and save/display table plot with TCs correlation scores for the given dataset(s).

    Parameters
    ----------
    data : list[xr.DataArray]
        List of datasets to plot.
    data_names : list[str]
        List of dataset names (assuming observations/reanalyses as the first datasets).
    year_range : str
        Year range of the evaluation period.
    correlation_type : str
        Type of correlation to evaluate ("seasonal" or "spatial").
    bin_size : float
        Size of the bins in degrees used computing the TCs metrics.
    output_path : str, optional
        Path to save the table plot. If None, the table is displayed but not saved.
    rows_to_remove : int
        Number of datasets to remove from the top of the table (generally, reference
        and reanalyses datasets) (default: 0).
    """

    # Prepare strings for rows, file names and titles
    rows, name_file, reference = evaluation_table_metadata(data_names, rows_to_remove)
    plot_title = f"Global {correlation_type} correlation ({year_range})"

    # Prepare column labels based on correlation type
    metrics_metadata = data_general.load_yaml_file(config_params.TCS_METRICS_PATH)
    if correlation_type == "seasonal":
        cols_corr = np.append(
            [f"{bin_size}° x {bin_size}°"],
            [
                rf"$\rho_{{s,{metric}}}$"
            for metric in metrics_metadata
            if metrics_metadata[metric]["temporal"] == True
            ],
        )
    elif correlation_type == "spatial":
        cols_corr = np.append(
            [f"{bin_size}° x {bin_size}°"],
            [
                rf"$r_{{xy,{metric}}}$"
                for metric in metrics_metadata
                if metrics_metadata[metric]["spatial"] == True
            ],
        )
    else:
        raise ValueError("Invalid correlation type. Must be 'seasonal' or 'spatial'.")

    # Prepare plot data
    data_plot = tc_evaluation_table_data(data, rows_to_remove)

    # Prepare colorbar parameters
    cbar_ticks_corr = ['Negative correlation (-1)', 'No correlation (0)', 'Positive correlation (1)']
    colors_corr = ("RedGreen", ['tab:red', 'white', 'tab:green'])
    # Alternative: ("OrangeGreen", ['tab:orange', 'white', 'tab:green'])
    limits_corr = np.repeat([[-1, 1]], len(cols_corr) - 1, axis=0)


    # Generate and save/display plot
    tc_corr_plot, _ = plots_general.plot_table(
        data_plot,
        title=plot_title,
        col_labels=cols_corr,
        row_labels=rows,
        cbar_ticks=cbar_ticks_corr,
        cbar_colors=colors_corr,
        limits=limits_corr,
        reference=reference,
        decimals=2,
    )

    plots_general.save_or_show_plot(
        tc_corr_plot,
        output_path,
        plot_filename=f"tcs_{correlation_type}_corr_table_{name_file}_{year_range.replace(' ', '-')}",
        plot_name=f"{correlation_type.capitalize()} correlation table for TCs metrics plot",
    )

    return
