import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as cticker

from matplotlib import colors

from pyhanami.utils.plots import plots_general


def plot_eeofs(eeof, clon=0, title="ISO convection patterns", cb_label="scaled EEOF",
               cmap="RdBu_r", levels=13,  vmin=None, vmax=None):
    """
    Generate plot of Empirical Orthogonal Functions (EOFs) for each ISO mode (MJO and
    BSISO) during boreal winter and boreal summer separately.

    Parameters
    ----------
    eeof : xr.Dataset
        EEOFs data (including variables `eeof`, `eigval` and `var_frac`, containing
        the EEOFs, eigenvalues and variance fractions, respectively).
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
        raise TypeError(
            "The central longitude 'clon' must be a numeric value between 0º and 360º."
        )

    lags = eeof.lag.values
    modes = eeof.mode.values

    # Compute scaling factor to convert EEOFs to physical units
    eigen = eeof["eigval"]
    sigma = 1.0
    scale = np.sqrt(eigen) * sigma


    # Prepare tick for axes
    lon_ticks = np.arange(-180, 181, 60)
    lat_ticks = [-20, 0, 20]

    # Create plot
    fig, axs = plt.subplots(
        len(lags),
        len(modes),
        figsize=(10, 4),
        dpi=150,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=clon)},
        constrained_layout=False,
    )
    fig.subplots_adjust(wspace=0, hspace=0, top=0.86)

    for j, lag in enumerate(np.flip(lags)):
        eeof_lag = eeof["eeof"].isel(lag=j)
        eeof_phys = eeof_lag * scale

        for k, mode in enumerate(modes):
            ax = axs[j, k]
            plots_general.style_cartopy_axis(ax, show_gridlines=False)

            eeof_aux = eeof_phys.sel(mode=mode)
            cb = eeof_aux.plot.contourf(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                levels=levels,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            ax.text(
                0.97,
                0.9,
                f"lag = {-lag} days",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                bbox={
                    "facecolor": colors.to_rgba('white', alpha=1), 
                    "edgecolor": 'black',
                    "boxstyle": 'square,pad=0.4',
                }
            )

            # Add titles and axes labels
            if j == 0:
                ax.set_title(
                    f"EEOF{mode} ({100 * eeof['var_frac'].sel(mode=mode).values:.2f}%)",
                    fontsize=12,
                )
            else:
                ax.set_title("")
            if j == len(lags) - 1:
                ax.set_xlabel("longitude", fontsize=8)
                if k == 0:
                    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree(central_longitude=clon))
                elif k == len(modes) - 1:
                    ax.set_xticks(lon_ticks[1:], crs=ccrs.PlateCarree(central_longitude=clon))
                else:
                    ax.set_xticks(lon_ticks[1:-1], crs=ccrs.PlateCarree(central_longitude=clon))
                ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
            else:
                ax.set_xlabel("")
            if k == 0:
                ax.set_ylabel("latitude", fontsize=8)
                ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
            else:
                ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=6)

    # Add shared colorbar
    cbar = fig.colorbar(cb, ax=axs, orientation="horizontal", shrink=0.5, pad=0.17, aspect=40)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cb_label, fontsize=9)

    fig.suptitle(title, fontsize=14)

    return fig, axs


def plot_pcs(pcs, title="Bimodal ISO indices", normalized=True):
    """
    Generate plot of the Bimodal ISO indices, i.e. the Principal Components (PCs) for each 
    ISO mode (MJO and BSISO).

    Parameters
    ----------
    pcs : xr.Dataset
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
        pcs_MJO = pcs["PC_MJO_std"]
        pcs_BSISO = pcs["PC_BSISO_std"]
        amp_MJO = pcs["amp_MJO_std"]
        amp_BSISO = pcs["amp_BSISO_std"]
    else:
        pcs_MJO = pcs["PC_MJO_raw"]
        pcs_BSISO = pcs["PC_BSISO_raw"]
        amp_MJO = pcs["amp_MJO_raw"]
        amp_BSISO = pcs["amp_BSISO_raw"]
    modes = pcs_MJO.mode.values

    # Prepare time labels
    time = pcs_MJO.time.values
    start = np.datetime64(time.min(), "M")
    end = np.datetime64(time.max(), "M")

    months = np.arange(start, end + np.timedelta64(2, "M"), np.timedelta64(1, "M"))
    tick_labels = [str(m)[:7] for m in months]

    # Create plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, dpi=150)
    plot_colors = [
        ["#0072B2", "#B2182B", "tab:brown", "tab:pink"],
        ["#E66101", "#7B3294", "tab:green", "tab:gray"],
    ]
    labels = ["MJO", "BSISO"]

    # Compute the maximum absolute value from all PCs and amplitudes
    y_lim = max(
        int(np.max(np.abs(pcs_MJO.values))) + 1,
        int(np.max(np.abs(pcs_BSISO.values))) + 1,
        int(np.max(np.abs(amp_MJO.values))) + 1,
        int(np.max(np.abs(amp_BSISO.values))) + 1,
    )

    # Plot PCs
    for i, data in enumerate([pcs_MJO, pcs_BSISO]):
        for j in modes:
            axs[i].plot(
                data.time,
                data.sel(mode=j),
                lw=1.2,
                ls="-",
                label=f"{labels[i]} PC{j}",
                color=plot_colors[i][j - 1],
            )

        # Plot formatting
        axs[i].set_xticks(months)
        axs[i].tick_params(axis="both", labelsize=8)
        axs[i].set_ylim(-y_lim, y_lim)
        axs[i].set_ylabel("Normalized PC", fontsize=10)
        axs[i].set_title(labels[i], fontsize=12)
        axs[i].legend(fontsize=8, loc="upper right")
        axs[i].grid(linestyle=":")

    # Plot amplitudes
    for i, data in enumerate([amp_MJO, amp_BSISO]):
        axs[2].plot(data.time, data, lw=1.2, ls="-", label=f"{labels[i]}", color=plot_colors[i][0])
        axs[2].fill_between(data.time, data, where=data >= 0, color=plot_colors[i][0], alpha=0.3)

    axs[2].set_xticks(months)
    axs[2].tick_params(axis="both", labelsize=8)
    axs[2].set_ylim(0, y_lim)
    axs[2].set_ylabel("|Normalized PCs|", fontsize=10)
    axs[2].set_title("Amplitude", fontsize=12)
    axs[2].legend(fontsize=8, loc="upper right")
    axs[2].grid(linestyle=":")

    axs[2].set_xticklabels(tick_labels, rotation=45, ha="right")
    axs[2].set_xlabel("time", fontsize=10)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig, axs


def plot_freq_ISO(freq_ISO_sim, freq_ISO_obs=None, alpha=None, corr=None, sigma=None, tss=None,
                  title="Mean monthly frequency of ISO events", sim_label="simulations",
                  obs_label="observations"):
    """
    Generate plot of the mean monthly frequency of occurrence of ISO events (ISO seasonality)
    distinguishing between MJO and BSISO. If observations are provided, also include Taylor
    Skill Score (TSS) statistics below the plot.

    Parameters
    ----------
    freq_ISO_sim : xr.Dataset
        Mean monthly frequency of occurrence data for simulated data.
    freq_ISO_obs : xr.Dataset
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
    if not isinstance(freq_ISO_sim, xr.Dataset) or (
        freq_ISO_obs is not None and not isinstance(freq_ISO_obs, (xr.Dataset))
    ):
        raise TypeError("The frequency of occurrence data must be xarray.Datasets.")


    # Create plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    plt.tight_layout()

    bar_width = 0.4
    x = np.arange(1, 13)
    ax.axhline(0, color="black", lw=0.8)

    if freq_ISO_obs is None:
        ax.bar(
            x,
            freq_ISO_sim["freq_MJO"],
            color="#0072B2",
            label=f"MJO {sim_label}",
            align="center",
            zorder=2,
        )
        ax.bar(
            x,
            -freq_ISO_sim["freq_BSISO"],
            color="#E66101",
            label=f"BSISO {sim_label}",
            align="center",
            zorder=2,
        )

        # MJO legend (upper right)
        mjo_handles = [plt.Rectangle((0, 0), 1, 1, color="#0072B2", label=f"MJO {sim_label}")]
        legend_mjo = ax.legend(handles=mjo_handles, loc="upper right", fontsize=8)
        ax.add_artist(legend_mjo)

        # BSISO legend (lower right)
        bsiso_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#E66101", label=f"BSISO {sim_label}")
        ]
        legend_bsiso = ax.legend(handles=bsiso_handles, loc="lower right", fontsize=8)
        ax.add_artist(legend_bsiso)
    else:
        # MJO
        ax.bar(
            x,
            freq_ISO_sim["freq_MJO"],
            color="#0072B2",
            label=f"MJO {sim_label}",
            width=-bar_width,
            align="edge",
            zorder=2,
        )
        ax.bar(
            x,
            freq_ISO_obs["freq_MJO"],
            color="white",
            edgecolor="#0072B2",
            hatch="////",
            linewidth=0.8,
            label=f"MJO {obs_label}",
            width=bar_width,
            align="edge",
            zorder=2,
        )

        # BSISO
        ax.bar(
            x,
            -freq_ISO_sim["freq_BSISO"],
            color="#E66101",
            label=f"BSISO {sim_label}",
            width=-bar_width,
            align="edge",
            zorder=2,
        )
        ax.bar(
            x,
            -freq_ISO_obs["freq_BSISO"],
            color="white",
            edgecolor="#E66101",
            hatch="////",
            linewidth=0.5,
            label=f"BSISO {obs_label}",
            width=bar_width,
            align="edge",
            zorder=2,
        )

        # MJO legend (upper right)
        mjo_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#0072B2", label=f"MJO {sim_label}"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="#0072B2",
                hatch="////",
                label=f"MJO {obs_label}",
            ),
        ]
        legend_mjo = ax.legend(handles=mjo_handles, loc="upper right", fontsize=8)
        ax.add_artist(legend_mjo)

        # BSISO legend (lower right)
        bsiso_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#E66101", label=f"BSISO {sim_label}"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="#E66101",
                hatch="////",
                label=f"BSISO {obs_label}",
            ),
        ]
        legend_bsiso = ax.legend(handles=bsiso_handles, loc="lower right", fontsize=8)
        ax.add_artist(legend_bsiso)

        # Add scalar scores
        stats_text = (
            f"Scores: $\\alpha$={f'{alpha:.2f}' if alpha is not None else 'N/A'}, "
            f"R={f'{corr:.2f}' if corr is not None else 'N/A'}, "
            f"$\\sigma$={f'{sigma:.2f}' if sigma is not None else 'N/A'}, "
            f"TSS={f'{tss:.2f}' if tss is not None else 'N/A'}"
        )
        fig.text(
            0.5,
            0.02,
            stats_text,
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": 'white', "edgecolor": 'black'},
        )
        fig.subplots_adjust(top=0.78, bottom=0.16)


    # Plot formatting
    ax.set_xticks(x)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlabel("month", fontsize=10)

    y_ticks = np.linspace(-1, 1, 11)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{abs(y):.1f}" for y in y_ticks])
    ax.set_ylabel("frequency of occurrence", fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.grid(zorder=0, linestyle=":")

    return fig, ax
