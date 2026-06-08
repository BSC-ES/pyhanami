import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import BoundaryNorm

from pyhanami.utils.mjo_scores import mjo_spectrum_funcs


def plot_ceofs(ceofs, title="Combined EOFs",
               vars_colors={"ua850": "#1f77b4", "ua200": "#ff7f0e", "rlut": "#2ca02c"},
               labels_linestyles={"Dataset 1": "-", "Dataset 2": "--", "Dataset 3": ":"}):
    """
    Generate plot of the first two Combined Empirical Orthogonal Functions (CEOFs)
    including all three variables (ua850, ua200 and rlut) in the same plot, for
    up to three different datasets.

    Parameters
    ----------
    ceofs : list[xr.DataArray]
        List of CEOFs for each variable.
    title : str
        Title of the plot.
    vars_colors : dict
        Dictionary mapping variable names to colors.
    labels_linestyles : dict
        Dictionary mapping dataset names to line styles.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated plot.
    axs : matplotlib.axes._subplots.AxesSubplot
        Plot axes.
    """

    # Validate input
    if not isinstance(ceofs, list) or not all(isinstance(ds, xr.DataArray) for ds in ceofs):
        raise TypeError("The CEOFs data must be provided as a list of xr.DataArrays.")


    # Define plotting style parameters
    shade_factors = [0.8, 1, 1.2]  # [0.6, 0.85, 1.25]

    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=150)

    # Bring subplots closer together
    plt.subplots_adjust(wspace=0.15)

    # Generate plots
    for i, ds in enumerate(ceofs):
        ds_name = list(labels_linestyles.keys())[i]
        for var in ds["variable"].values:
            subset = ds.sel(variable=var)

            # Determine color and style
            base = vars_colors.get(var, "black")
            base_rgb = colors.to_rgb(base)
            color = tuple(min(1, max(0, c * shade_factors[i])) for c in base_rgb)

            # Loop over modes
            for i_mode in range(2):
                ax = axs[i_mode]
                subset_one_mode = subset.sel(mode=i_mode)

                ax.plot(
                    subset_one_mode.lon,
                    subset_one_mode,
                    color=color,
                    linestyle=labels_linestyles[ds_name],
                    label=f"{ds_name} - {var}",
                )

    # Formatting
    for j, ax in enumerate(axs):
        ax.set_xlabel("longitude (°E)", fontsize=12)
        if j == 0:
            ax.set_ylabel("normalized amplitude", fontsize=12)
        ax.set_title(f"Combined EOF{j + 1}", fontsize=13)
        ax.margins(x=0)  # remove whitespace before first data point
        ax.grid(linestyle=":")  # alpha=0.8
        ax.axhline(0, color="black", linestyle="-")
        ax.tick_params(axis="both", which="major", labelsize=11)

    # Build custom legend entries for datasets (linestyles)
    dataset_handles = [
        Line2D([0], [0], color="black", linestyle=labels_linestyles[ds], linewidth=2, label=ds)
        for ds in labels_linestyles.keys()
    ]

    # Build legend entries for variables (colors)
    variable_handles = [
        Line2D([0], [0], color=vars_colors[var], linestyle="-", linewidth=2, label=var)
        for var in vars_colors.keys()
    ]

    # Add both legends to the figure
    dataset_legend = fig.legend(
        handles=dataset_handles,
        loc="lower center",
        bbox_to_anchor=(0.35, -0.12),  # Position left
        ncol=len(dataset_handles),
        frameon=True,
        fontsize=11,
        title="Dataset:",
    )

    variable_legend = fig.legend(
        handles=variable_handles,
        loc="lower center",
        bbox_to_anchor=(0.65, -0.12),  # Position right
        ncol=len(variable_handles),
        frameon=True,
        fontsize=11,
        title="Variable:",
    )

    # Add the first legend back (matplotlib removes it when adding the second)
    fig.add_artist(dataset_legend)

    fig.suptitle(title, fontsize=14, y=0.99)
    plt.tight_layout()
    return fig, axs


def plot_power_spectrum(spectrum, component="symmetric", x_lim=[-10, 10], y_lim=[0.01, 0.25],
                        title="Symmetric power spectrum", cmap=None, levels=None, vmin=None,
                        vmax=None, mjo_box=True, mjo_freq_bounds=(1 / 96, 1 / 30),
                        mjo_wavenum_bounds=(0, 4)):
    """
    Generate symmetric power spectrum plot with theoretical dispersion curves
    overlaid.

    Parameters
    ---------
    spectrum: xr.DataArray
        Power spectrum data (with 'wavenumber' and 'frequency' coordinates).
    component : str
        Component and dispersion curves to plot, either 'symmetric' or
        'antisymmetric' (default: 'symmetric').
    x_lim : list[float]
        Limits for the x-axis (default: [-10, 10]).
    y_lim : list[float]
        Limits for the y-axis (default: [0.01, 0.25]).
    title : str
        Title of the plot (default: 'Symmetric power spectrum').
    cmap : matplotlib colormap
        Colormap.
    levels : np.ndarray
        Contour levels.
    vmin, vmax : float
        Min. and max. values for the colormap.
    mjo_box : bool
        Whether to draw a dashed box around the MJO region (default: True).
    mjo_freq_bounds : tuple
        Frequency bounds corresponding to the MJO band in the wavenumber-frequency space (default: (1/96, 1/30)).
    mjo_wavenum_bounds : tuple
        Wavenumber bounds corresponding to the MJO band in the wavenumber-frequency space (default: (0, 4)).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        Plot axis.
    """

    # Validate input
    if not isinstance(spectrum, xr.DataArray):
        raise ValueError("The spectrum must be an xr.DataArray")

    if component == "symmetric":
        ii_range = [3, 4, 5]
    elif component == "antisymmetric":
        ii_range = [0, 1, 2]
    else:
        raise ValueError(
            f"Invalid component option '{component}'. Choose either 'symmetric' or 'antisymmetric'."
        )

    # Get data for the theoretical dispersion curves:
    swfreq, swwn = mjo_spectrum_funcs.gen_dispersion_curves()

    # Prepare data
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    z = spectrum.transpose().sel(frequency=slice(0, 0.5), wavenumber=slice(-15, 15))
    z.loc[{"frequency": 0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z["wavenumber"], z["frequency"])

    # Prepare plotting parameters
    if cmap is None:
        colors_custom = [
            # "#FFFFFF",   # White
            "#C6DBEF",  # Light blue
            "#6BAED6",  # Medium blue
            "#2171B5",  # Dark blue
            "#41AB5D",  # Light green
            "#9dd561",  # Dark green
            "#facc0a",  # Light yellow
            "#f4a604",  # Darker yellow
            "#C70039",  # Light red
            "#900C3F",  # Medium red
            "#581845",  # Dark red
        ]
        cmap = colors.ListedColormap(colors_custom)

    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)
    if levels is None:
        levels = np.linspace(vmin, vmax, 12)
    norm = BoundaryNorm(levels, len(levels))


    # Create plot
    fig, ax = plt.subplots(dpi=150)

    # Plot spectrum
    img = ax.contourf(
        kmesh0,
        vmesh0,
        z,
        cmap=cmap,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extend="max",
        zorder=0,
    )

    # Add dispersion curves
    color_disp_curves = "black"
    lw_disp_curves = 0.5
    for ii in ii_range:
        ax.plot(swk[ii, 0, :], swf[ii, 0, :], color=color_disp_curves, lw=lw_disp_curves, zorder=1)
        ax.plot(swk[ii, 1, :], swf[ii, 1, :], color=color_disp_curves, lw=lw_disp_curves, zorder=1)
        ax.plot(swk[ii, 2, :], swf[ii, 2, :], color=color_disp_curves, lw=lw_disp_curves, zorder=1)
    ax.axvline(0, linestyle="dashed", color="lightgray", zorder=1)

    # Add dashed box around MJO region if requested
    if mjo_box:
        mjo_box = Rectangle(
            (mjo_wavenum_bounds[0], mjo_freq_bounds[0]),
            width=mjo_wavenum_bounds[1] - mjo_wavenum_bounds[0],
            height=mjo_freq_bounds[1] - mjo_freq_bounds[0],
            edgecolor="black",
            facecolor="none",
            linestyle="dashed",
            lw=1.5,
            zorder=2,
        )
        ax.add_patch(mjo_box)

    # Plot formatting
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("zonal wavenumber")
    ax.set_ylabel("frequency (1/day)")
    ax.set_title(title)

    fig.colorbar(img, ax=ax, label="normalized power")

    return fig, ax


def plot_power_spectrum_two(spectrum_1, spectrum_2, component="symmetric", x_lim=[-10, 10],
                            y_lim=[0.01, 0.25], title_1="Spectrum 1", title_2="Spectrum 2",
                            suptitle="Symmetric power spectrum", cmap=None,  levels=None,
                            vmin=None, vmax=None, mjo_box=True, mjo_freq_bounds=(1 / 96, 1 / 30),
                            mjo_wavenum_bounds=(0, 4)):
    """
    Generate symmetric power spectrum plot with theoretical dispersion curves
    overlaid.

    Parameters
    ---------
    spectrum_1, spectrum_2: xr.DataArray
        Power spectrum data (with 'wavenumber' and 'frequency' coordinates).
    component : str
        Component and dispersion curves to plot, either 'symmetric' or
        'antisymmetric' (default: 'symmetric').
    x_lim : list[float]
        Limits for the x-axis (default: [-10, 10]).
    y_lim : list[float]
        Limits for the y-axis (default: [0.01, 0.25]).
    title : str
        Title of the plot (default: 'Symmetric power spectrum').
    cmap : matplotlib colormap
        Colormap.
    levels : np.ndarray
        Contour levels.
    vmin, vmax : float
        Min. and max. values for the colormap.
    mjo_box : bool
        Whether to draw a dashed box around the MJO region (default: True).
    mjo_freq_bounds : tuple
        Frequency bounds corresponding to the MJO band in the wavenumber-frequency space (default: (1/96, 1/30)).
    mjo_wavenum_bounds : tuple
        Wavenumber bounds corresponding to the MJO band in the wavenumber-frequency space (default: (0, 4)).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        Plot axis.
    """

    # Validate input
    if not isinstance(spectrum_1, xr.DataArray) or not isinstance(spectrum_2, xr.DataArray):
        raise ValueError("Both spectra must be xr.DataArray")

    if component == "symmetric":
        ii_range = [3, 4, 5]
    elif component == "antisymmetric":
        ii_range = [0, 1, 2]
    else:
        raise ValueError(
            f"Invalid component option '{component}'. Choose either 'symmetric' or 'antisymmetric'."
        )

    # Get data for the theoretical dispersion curves:
    swfreq, swwn = mjo_spectrum_funcs.gen_dispersion_curves()

    # Prepare data
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    z_1 = spectrum_1.transpose().sel(frequency=slice(0, 0.5), wavenumber=slice(-15, 15))
    z_1.loc[{"frequency": 0}] = np.nan
    kmesh0_1, vmesh0_1 = np.meshgrid(z_1["wavenumber"], z_1["frequency"])

    z_2 = spectrum_2.transpose().sel(frequency=slice(0, 0.5), wavenumber=slice(-15, 15))
    z_2.loc[{"frequency": 0}] = np.nan
    kmesh0_2, vmesh0_2 = np.meshgrid(z_2["wavenumber"], z_2["frequency"])

    # Prepare plotting parameters
    if cmap is None:
        colors_custom = [
            # "#FFFFFF",   # White
            "#C6DBEF",  # Light blue
            "#6BAED6",  # Medium blue
            "#2171B5",  # Dark blue
            "#41AB5D",  # Light green
            "#9dd561",  # Dark green
            "#facc0a",  # Light yellow
            "#f4a604",  # Darker yellow
            "#C70039",  # Light red
            "#900C3F",  # Medium red
            "#581845",  # Dark red
        ]
        cmap = colors.ListedColormap(colors_custom)

    if vmin is None:
        vmin = np.nanmin([z_1, z_2])
    if vmax is None:
        vmax = np.nanmax([z_1, z_2])
    if levels is None:
        levels = np.linspace(vmin, vmax, 12)
    norm = BoundaryNorm(levels, len(levels))


    # Create plot
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, dpi=150)

    # Separate subplots
    # plt.subplots_adjust(wspace=0.2)

    # Plot spectrum
    img = axs[0].contourf(
        kmesh0_1,
        vmesh0_1,
        z_1,
        cmap=cmap,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extend="max",
        zorder=0,
    )
    img = axs[1].contourf(
        kmesh0_2,
        vmesh0_2,
        z_2,
        cmap=cmap,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extend="max",
        zorder=0,
    )

    # Add dispersion curves
    color_disp_curves = "black"
    lw_disp_curves = 0.5
    for ax, title in zip(axs, [title_1, title_2]):
        for ii in ii_range:
            ax.plot(
                swk[ii, 0, :], swf[ii, 0, :], color=color_disp_curves, lw=lw_disp_curves, zorder=1
            )
            ax.plot(
                swk[ii, 1, :], swf[ii, 1, :], color=color_disp_curves, lw=lw_disp_curves, zorder=1
            )
            ax.plot(
                swk[ii, 2, :], swf[ii, 2, :], color=color_disp_curves, lw=lw_disp_curves, zorder=1
            )
        ax.axvline(0, linestyle="dashed", color="lightgray", zorder=1)

        # Add dashed box around MJO region if requested
        if mjo_box:
            mjo_box = Rectangle(
                (mjo_wavenum_bounds[0], mjo_freq_bounds[0]),
                width=mjo_wavenum_bounds[1] - mjo_wavenum_bounds[0],
                height=mjo_freq_bounds[1] - mjo_freq_bounds[0],
                edgecolor="black",
                facecolor="none",
                linestyle="dashed",
                lw=1.5,
                zorder=2,
            )
            ax.add_patch(mjo_box)

        # Plot formatting
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel("zonal wavenumber", fontsize=12)
        ax.set_ylabel("frequency (1/day)", fontsize=12)
        ax.set_title(title, fontsize=13)

    fig.suptitle(suptitle, fontsize=14)
    fig.colorbar(img, ax=axs, label="normalized power")

    return fig, axs
