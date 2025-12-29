# Methodology

This page describes the methodology used in _pyhanami_ for evaluating model replicability and assessing ESMs' ability to reproduce real-world climate phenomena.


## Data preprocessing

<!-- TO DO: add explanation of the intake catalog when implemented.

Moreover, -->
The package includes a `DataChecker` class that verifies the integrity and consistency of the input datasets (both simulations and observations) before any analysis is performed. This ensures that the data meets the required standards and formats, reducing the likelihood of errors during processing. It checks for:

- **Standard compliance**: names and format of coordinates (`time`, `lat`, `lon`), and variables’ attributes (`long_name` and `units`).
- **Spatial completeness**: no NaN values present in atmosphere variables, and at least one (but not all) NaN value in the ocean variables.
- **Temporal completeness**: all timesteps present between the minimum and maximum time values.
- **Physical plausibility**: all variables lie within the established ranges and boundaries defined in `src/pyhanami/config/variables.yaml`.



## Diagnostic visualizations

This section provides an overview of how the data is prepared and processed for the diagnostic visualizations implemented within the `DataDiagnostics` class.

First of all, the package includes methods to create **time series** plots for a given climate variable. For these, the data is spatially averaged with area weights (based on latitude) to obtain global mean time series. The time series can be computed as annual, monthly or daily means. Given a simulation ensemble, the time series are first calculated for each ensemble member, and then the ensemble mean is plotted together with the 2.5%, 5%, 95% and 97.5% percentiles. Besides, the trajectories of individual members can be included in the plot.

Additionally, the package allows to generate spatial comparisons between different simulation datasets for a given climate variable. These include:
- **Absolute difference**: absolute average difference between the datasets at the grid point level. Before computing the difference, the mean over time (for the whole period covered by the datasets) and the mean over ensemble members (if the datasets contain multiple ensemble members) are calculated.
- **Effect size (Cohen's $d$)**: effect size between the datasets at the grid point level. This is taken as Cohen's effect size ($d$), computed as
    
    $$
    d = \frac{\mu_1 - \mu_2}{\sigma},
    $$
    
    where μ₁ and μ₂ are the means of the two datasets taken over the ensemble members (after averaging over time), and σ is the pooled standard deviation. The pooled standard deviation is calculated as
    
    $$
    \sigma = \sqrt{\frac{(n_1 - 1)\sigma_1^2 + (n_2 - 1)\sigma_2^2}{n_1 + n_2 - 2}},
    $$

    where $\sigma_1$ and $\sigma_2$ are the standard deviations of the two datasets taken over the ensemble members (after averaging over time), and $n_1$ and $n_2$ are the number of samples (ensemble members) in each dataset.

    Note that the effect size is not computed just once for all the ensembles members. Instead,  bootstrapping is used to compute the effect size multiple times, generating a distribution of effect sizes. The mean of this distribution is taken as the final effect size value. Moreover, a _t_-test is performed to assess whether the differences between the two datasets are statistically significant at the grid point level.



## Replicability

This section describes the statistical approach implemented within the `ReplicabilityTest` class to assess the replicability of ESMs. The aim of the test is to evaluate whether two sets of simulated ensembles are statistically indistinguishable with a given significance level (see ([[Preprint] K. Keller et al., 2025](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1367/)) for a more detailed description of the methodology).

The structure of the test is the following: given two ensembles generated in two different computing environments (e.g., different hardware or software stack), we assign a score to each ensemble member, resulting in two distributions of scores which are then compared by combining several statistical tests, such as the Kolmogorov-Smirnov test, with the null hypothesis that both samples are drawn from the same underlying distribution.

When running an ESM, the surface of the Earth is represented as a grid composed of multiple cells. We use the climatologies (mean values over a span of time) of climate variables at each of these cells to compute the weights for the test. Some of the variables used are the temperature of the air in the atmosphere, the precipitation rate, the pressure at mean sea level, or the salinity of the sea surface. Directly averaging the values of these variables spatially would result in the loss of important information regarding regional differences among the ensemble members. 

Due to this, we first apply various metrics at the cell level, which contrast the value of one member to either the mean of the entire simulated ensemble or the mean of a common reference set of observations. Particularly, we combine the outcome of three different metrics and evaluate them with four distinct statistical tests, rejecting the null hypothesis whenever at least one of the tests results in a rejection.

As the test is implemented within _pyhanami_, it allows users to test for replicability across a set of climate variables, distinguishing by region:
- Global
- Tropics (latitude below 30º)
- Extratropics (latitude above 30º)

and season:
- All (January–December)
- DJF (December–February)
- MAM (March–May)
- JJA (June–August)
- SON (September–November)

This separation facilitates the identification of more specific issues that may be obscured when averaging the simulation results globally or over the entire year.

Finally, given two ensembles, the results of the replicability tests are summarized in a **matrix plot**, in which each matrix cell represents the comparison between the ensembles for a specific variable, season and region. Particularly, each cell is divided into four triangular sections representing the effect size obtained using various metrics, together with a circle in the center indicating the statistical tests outcome:
- Green: no rejection of the null hypothesis (replicable)
- Red: rejection of the null hypothesis (not replicable)

This visualization allows for a quick assessment of the replicability, helping identify variables showing significant differences and seasonal/regional patterns.


## Scientific skill

The `ScientificEvaluation` class implements various plots and scalar metrics to analyze how well ESMs reproduce key climate phenomena. These metrics compare model output against observational data to quantify the models' skill in capturing specific features of the Earth's climate system.

This section explains the approach used to evaluate several climate phenomena. Currently, the package includes the following scientific skill metrics:

### Tropical IntraSeasonal Oscillation (ISO): Bimodal ISO indices

Two separate indices are defined for the Madden-Julian Oscillation (MJO) and the Boreal Summer ISO (BSISO), following [(K. Kikuchi, 2020)](https://link.springer.com/article/10.1007/s00382-019-05037-z). These indices capture the ISO behavior during boreal winter and boreal summer, respectively. They are computed by performing an Extended Empirical Orthogonal Function (EEOF) analysis using TOA Outgoing Longwave Radiation (OLR) data, and then projecting the OLR data onto the first two EEOFs. This results in two Principal Components (PCs) for MJO and two for BSISO, which together represent the **bimodal ISO indices**. These indices are normalized by dividing by one standard deviation during the period taken for the EEOF analysis, i.e. the squared root of the corresponding eigenvalue.

<!-- EOFs are spatial patterns showing where things tend to vary together, and they look for the simplest explanation of the most variance (if a dataset could be described with just one pattern, what pattern would capture the most?). The 2nd EOF explains the second-most, and so on, and they are mathematically independent (orthogonal). The whole dataset can be reconstructed by adding a weighted combination of a few EOF pattern. Moreover, each one has an associated time series (PC) which shows when the pattern was active and how strongly. When a dataset contains an oscillatory phenomenon, EOF1 + EOF2 together represent a physical mode, being EOF1 like 'phase 1' and EOF2 like 'phase 2' (90º out of phase); hence, combining them gives a rotating or propagating structure. In this case, the physical meaning is in the pair EOF1 + EOF2, not in each EOF individually. This is very common when both eigenvalues are nearly equal and EOF1 and EOF2 look like the same map but shifted in space, however, for other phenomena, they can also represent two different independent physical modes (ex. global temperature, where EOF1 is the overall warming pattern and EOF2 is the ENSO pattern?). 

On another note, given that if \lambda is an eigenvalue, then -\lambda is also an eigenvalue, the sign of an EOF does not have a physical meaning by itself. It might change depending on the algorithm used. Nevertheless, it is important to note that there is meaningful information in the relative sign struture (i.e. which regions vary together or oppositely). -->

In order to obtain scalar metrics, we also compute the **temporal correlation (R)**, **standard deviation ratio (σ)**, and **Taylor Skill Score (TSS)** between simulations and observations using the PCs' amplitude, following [(M. Nakano et al., 2019)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL082443). Specifically, the mean monthly frequency of MJO and BSISO events (ISO seasonality) is calculated using the amplitude of the corresponding PCs. The MJO frequency is then subtracted from the BSISO frequency, and this difference is compared between simulations and observations.

The temporal correlation indicates how well the phase of the ISO seasonality is reproduced by a model. While the ratio of standard deviations (model/observations) provides information about the amplitude (MJO/BSISO contrast) of the seasonality. Finally, the $\text{TSS}$ combines both the correlation and the standard deviation, allowing to assess how well a model matches the ISO seasonality of the observations with a single score defined as

$$
\text{TSS} = \frac{4(1+R)^4}{(\sigma + 1/\sigma)^2(1+R_0)^2},
$$

where $R_0$ is the maximum correlation that can be achieved by the model, taken as $R_0=1$.

Since models usually underestimate the amplitude of the ISO, the PCs can be adjusted before computing the above quantities by scaling them with the **PCs' amplitude ratio ($\alpha$)** between simulations and observations, defined as

$$
\alpha = \frac{\overline{\lVert\text{PC}_{\text{MJO}}^{\text{sim}}\rVert} + \overline{\lVert\text{PC}_{\text{BSISO}}^{\text{sim}}\rVert}}{\overline{\lVert\text{PC}_{\text{MJO}}^{\text{obs}}\rVert} + \overline{\lVert\text{PC}_{\text{BSISO}}^{\text{obs}}\rVert}}.
$$

After applying this correction, the average number of ISO events per year becomes comparable between simulations and observations. This adjustment ensures that the computed statistics better reflect how well the model reproduces the ISO seasonality pattern (i.e., the relative occurrence of MJO vs BSISO). However, because the simulated PCs have been normalized, these statistics cannot be used to assess absolute amplitude differences of MJO and BSISO between simulations and observations.

The implementation of the analysis described above produces three types of diagnostic plots:
- **EEOFs**: multiple spatial plots showing the first two EEOFs for boreal winter (during DJFMA) and for boreal summer (during JJASO). The EEOFs are scaled before plotting using the corresponding eigenvalues, and each of them is plotted separately for three different time lags (-10, -5 and 0 days).
- **PCs**: two time series plots displaying the temporal evolution of the first two normalized PCs for MJO and BSISO, along with a third plot showing the evolution of the amplitude ( $\scriptsize{\sqrt{\text{PC}_1^2 + \text{PC}_2^2}}$ ) of each set of PCs.
- **ISO seasonality**: mean monthly distribution of ISO events separating MJO and BSISO, with comparison between simulations and observations available (including the values for the scalar metrics $R$, $\sigma$ and $\text{TSS}$).

In all cases, a colormap with a blue to red gradient is used for boreal winter (or MJO), while a green to orange gradient is used for boreal summer (or BSISO).

### Tropical Cyclones (TCs): TC metrics

TC trajectories are detected and tracked using the [TempestExtremes package](https://github.com/ClimateGlobalChange/tempestextremes). The default criteria for TC detection is taken from [(C.M. Zarzycki & P.A. Ullrich, 2017)](https://doi.org/10.1002/2016GL071606). However, we recommend adjusting the `min_wind` parameter (i.e. 10 m wind speed detection threshold) according to the model's (or reanalysis') horizontal resolution following the criteria established in [(K.J.E. Walsh et al., 2007)](https://doi.org/10.1175/JCLI4074.1) (see Fig. 2 in the paper for guidance). Moreover, the TempestExtremes package requires **surface geopotential** (`phis`) data to track TCs. In here, this is computed from topography data taken from the [GEBCO_2024 Grid](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2024-grid), a global terrain model for ocean and land which provides elevation data with a horizontal resolution of 15 arc-seconds (~ 0.5 km). Using this data, `phis` is computed by multiplying the topography (in meters) by the standard gravity (9.80665 m/s²). Then, before using it, `phis` is regridded to match the horizontal resolution of the input dataset. 

Following [(C.M. Zarzycki et al., 2021)](https://journals.ametsoc.org/view/journals/apme/60/5/JAMC-D-20-0149.1.xml), the obtained trajectories are then used to compute various temporal and spatial statistics for each of the following TC metrics (we refer to the paper for the exact formulas):

- **Annual/monthly frequency (counts)**: number of discrete storm events.
- **Tropical Cyclone days (TCD)**: computed counting each occurrence of a 6-hourly tracked point during a storm's lifetime.
- **Accumulated cyclone energy (ACE)**: calculated using the 6-hourly maximum 10 m wind speed at each trajectory point.
- **Pressure ACE (PACE)**: similar to ACE, but using the 6-hourly minimum sea level pressure instead of the wind at each trajectory point.
- **Latitude of lifetime-maximum intensity (LMI)**: defined as the absolute value of the latitude where a TC reaches its maximum intensity (defined by maximum 10 m wind).

From these, we generate monthly/yearly global time series that allow computing the following scalar temporal statistics:
- **Global climatological mean bias** with respect to a reference observational dataset over a given period ($\bar{b}_{clim}$).
- **Global storm mean values** (dividing by the counts) over a given period ($\bar{b}_{storm}$).
- **Global Spearman rank correlation** coefficient ($\rho_s$) over a given period.

Moreover, we generate spatial plots of the absolute values and biases with respect to an observational reference for the aforementioned metrics (except for LMI) together with the minimum sea level pressure, the maximum 10 m wind and the TC genesis. From these, we compute the following scalar spatial statistics:
- **Global Pearson correlation** coefficient ($r_{xy}$).

Note that the TC genesis is defined as the first tracked point of each storm's lifetime.

Finally, the [International Best Track Archive for Climate Stewardship (IBTrACS)](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01552) is used as the default observational TCs dataset when computing these statistics.




<!--TO DO: add examples of the plots.

TO ADD:
- TCs: 'per_month', 'per_year', 'climo_mean', 'storm_mean', 'temp_scorr', 'spatial', 'spatial_pcorr' for each metric (counts, tcd, ace, pace, lmi). -->