# pyhanami

pyhanami is a tool designed to evaluate the replicability and scientific skill of Earth System Models (ESMs).


## Replicability

An ESM is replicable if performing the same experiment (with the same model and forcing) using different computing environments or compilers leads to _identical_ results representing the same climate. In practice, bit-for-bit replicability is not feasible due to the chaotic nature of this type of models. However, we can aim to achieve statistically indistinguishable results. pyhanami provides a replicability test to assess whether this indistinguishability holds between two given ensembles of simulated data, following the methodology presented in (add paper).


## Scientific skill

Scientific model skill refers to the ability of an ESM to accurately represent and predict various aspects of the climate system, including its capacity to forecast future climate changes or to capture complex patterns and relationships within the system. (add what exactly the package offers related to this)


## Features


## Example: Performing a replicability test
```python
import pyhanami as hnmi

# Retrieve data from catalogue interface
ref = hnmi.SimulationData('interface_path_ref', name='ref')
test = hnmi.SimulationData('interface_path_test', name='test')

# Create plots
diags = hnmi.DataDiagnostics(ref, test)
diags.time_series_plots('output_path')
diags.spatial_plots('output_path')

# Run test and report
tester = hnmi.ReplicabilityTest(ref, test)
tester.matrix_plot('output_path')
tester.report('output_path', time_series=True, spatial=True)
```