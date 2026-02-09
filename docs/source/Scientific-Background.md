# Scientific Background

This page provides the scientific background necessary to understand the evaluations performed by _pyhanami_. It covers key concepts related to Earth System Models (ESMs), replicability, and the specific climate phenomena analyzed by the tool.

## Replicability

An ESM is replicable if performing the same experiment (with the same model and forcings) using different computing environments or compilers leads to _identical_ results representing the same climate. In practice, bit-for-bit replicability is not feasible due to the chaotic nature of this type of models. However, we can aim to achieve statistically indistinguishable results. _pyhanami_ provides a replicability test to assess whether this indistinguishability holds between two given ensembles of simulated data, following the methodology presented in ([[Preprint] K. Keller et al., 2025](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1367/)).


## Scientific skill

Scientific model skill refers to the ability of an ESM to accurately represent and predict various aspects of the climate system, including its capacity to forecast future climate changes or to capture complex patterns and relationships within the system. 

Currently, evaluation of the following phenomena is available within _pyhanami_:
- **Tropical IntraSeasonal Oscillation (ISO):** characterized by large-scale convective anomalies that modulate tropical atmospheric circulation on 30-90 day timescales. This is the predominant phenomenon in the tropics throughout the year. Following [(K. Kikuchi et al., 2012)](https://link.springer.com/article/10.1007/s00382-011-1159-1), we differentiate between two modes of ISO:     
    - The **Madden-Julian Oscillation (MJO)** mode during boreal winter with predominant eastward propagation along the equator.
    - The **Boreal Summer ISO (BSISO)** mode during boreal summer with not only eastward but also northward/northwestward propagation over the northern Indian Ocean and the western North Pacific.

- **Tropical Cyclones (TCs):** warm-core, cyclonic storms characterized by heavy precipitation and strong winds that begin over tropical oceans. They can vary in speed, size, and intensity.

<!-- TO ADD:
- Storm genesis: first entry for each individual storm's lifetime (C.M. Zarzycki et al. 2021). Our definition (CEMA's): first point in the storm's lifetime with a wind above the cut-off wind.
- Define: TCD, ACE, PACE, LMI -->