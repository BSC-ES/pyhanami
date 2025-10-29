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

First of all, the package includes methods to create time series plots for a given climate variable. For these, the data is spatially averaged with area weights (based on latitude) to obtain global mean time series. The time series can be computed as annual, monthly or daily means. Given a simulation ensemble, the time series are first calculated for each ensemble member, and then the ensemble mean is plotted together with the 2.5%, 5%, 95% and 97.5% percentiles. Besides, the trajectories of individual members can be included in the plot.

Additionally, the package allows to generate spatial comparisons between different simulation datasets for a given climate variable. These include:
- **Absolute difference**: absolute average difference between the datasets at the grid point level. Before computing the difference, the mean over time (for the whole period covered by the datasets) and the mean over ensemble members (if the datasets contain multiple ensemble members) are calculated.
- **Effect size (Cohen's _d_)**: effect size between the datasets at the grid point level. This is taken as Cohen's effect size (_d_), computed as:
    $$
    d = \frac{\mu_1 - \mu_2}{\sigma},
    $$
    where μ₁ and μ₂ are the means of the two datasets taken over the ensemble members (after averaging over time), and σ is the pooled standard deviation. The pooled standard deviation is calculated as:
    $$
    \sigma = \sqrt{\frac{(n_1 - 1)\sigma_1² + (n_2 - 1)\sigma_2²}{n_1 + n_2 - 2}},
    $$
    where σ₁ and σ₂ are the standard deviations of the two datasets taken over the ensemble members (after averaging over time), and n₁ and n₂ are the number of samples (ensemble members) in each dataset.

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

The `ScientificEvaluation` class implements various scalar metrics to analyze how well ESMs reproduce key climate phenomena. These metrics compare model output against observational data to quantify the models' skill in capturing specific features of the Earth's climate system.

Currently, the package includes the following scientific skill metrics:
- **Bimodal ISO indices:** two separate indices are defined for the MJO and the BSISO, following [(K. Kikuchi, 2020)](https://link.springer.com/article/10.1007/s00382-019-05037-z). These indices capture the ISO behavior during boreal winter and boreal summer, respectively. They are computed by performing an Extended Empirical Orthogonal Function (EEOF) analysis using TOA Outgoing Longwave Radiation (OLR) data, and then projecting the OLR data onto the first two EEOFs. This results in two Principal Components (PCs) for MJO and two for BSISO, which together represent the bimodal ISO indices. Note that the indices are normalized by dividing by one standard deviation during the period taken for the EEOF analysis, i.e. the squared root of the corresponding eigenvalue.


    In order to obtain scalar metrics, we also compute the **temporal correlation (R)**, **standard deviation ratio (σ)**, and **Taylor Skill Score (TSS)** between simulations and observations using the PCs' amplitude, following [(M. Nakano et al., 2019)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL082443). Specifically, the mean monthly frequency of MJO and BSISO events (ISO seasonality) is calculated using the amplitude of the corresponding PCs. The MJO frequency is then subtracted from the BSISO frequency, and this difference is compared between simulations and observations.

    The temporal correlation indicates how well the phase of the ISO seasonality is reproduced by a model. While the ratio of standard deviations (model/observations) provides information about the amplitude (MJO/BSISO contrast) of the seasonality. Finally, the TSS combines both the correlation and the standard deviation, allowing to assess how well a model matches the ISO seasonality of the observations with a single score defined as:
    $$
    TSS = \frac{4(1+R)⁴}{(\sigma + 1/\sigma)²(1+R_0)²},
    $$
    where R₀ is the maximum correlation that can be achieved by the model, taken as R₀=1.

    The implementation of the analysis described above produces three types of diagnostic plots:
    - **EEOFs**: multiple spatial plots showing the first two EEOFs for boreal winter (during DJFMA) and for boreal summer (during JJASO). The EEOFs are scaled before plotting using the corresponding eigenvalues, and each of them is plotted separately for three different time lags (-10, -5 and 0 days).
    - **PCs**: two time series plots displaying the temporal evolution of the first two normalized PCs for MJO and BSISO, along with a third plot showing the evolution of the amplitude (sqrt(PC1² + PC2²)) of each set of PCs.
    - **ISO seasonality**: mean monthly distribution of ISO events separating MJO and BSISO, with comparison between simulations and observations available (including the values for the scalar metrics R, σ and TSS).

    In all cases, a colormap with a blue to red gradient is used for boreal winter (or MJO), while a green to orange gradient is used for boreal summer (or BSISO).

- **TCs metrics:** following [(C.M. Zarzycki et al., 2021)](https://journals.ametsoc.org/view/journals/apme/60/5/JAMC-D-20-0149.1.xml) various temporal and spatial statistics are computed for each of the following TCs metrics:

    - **Annual/monthly frequency (counts)**: number of discrete storm events.
    - **Tropical Cyclone days (TCD)**: computed counting each occurrence of a 6-hourly tracked point during a storm's lifetime. 
    - **Accumulated cyclone energy (ACE)**: calculated using the 6-hourly maximum 10 m wind speed at each trajectory point.
    - **Pressure ACE (PACE)**: similar to ACE, but using the 6-hourly minimum sea level pressure instead of the wind at each trajectory point.
    - **Latitude of lifetime-maximum intensity (LMI)**: defined as the absolute value of the latitude where a TC reaches its maximum intensity (defined by maximum 10 m wind).

    From these, we generate monthly/yearly global time series that allow computing the following scalar temporal statistics:
    - **Global climatological mean bias** with respect to a reference observational dataset over a given period ($\bar{b}_{clim}$).
    - **Global storm mean values** (dividing by the counts) over a given period ($\bar{b}_{storm}$).
    - **Global Spearman rank correlation** coefficient ($\rho_s$) over a given period.

    Moreover, we generate spatial plots of the absolute values and biases with respect to a observational reference for the aforementioned metrics (except for LMI) together with the minimum sea level pressure, the maximum 10 m wind and the TC genesis. From these, we compute the following scalar spatial statistics:
    - **Global Pearson correlation** coefficient ($r_{xy}$).

    Note that [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) is used as the default observational dataset.




<!--TO DO: add examples of the plots.

TO ADD:
- TCs: 'per_month', 'per_year', 'climo_mean', 'storm_mean', 'temp_scorr', 'spatial', 'spatial_pcorr' for each metric (counts, tcd, ace, pace, lmi). NOTE: no spatial metrics for lmi as it is a latitude. -->