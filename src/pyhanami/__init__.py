from importlib.metadata import version

# Load main public classes in a specific order to avoid conflicts between xarray and netCDF4
from .diags.ScientificSkill import ScientificEvaluation
from .diags.Diagnostics import DataDiagnostics
from .diags.Observations import ObservationData
from .diags.Replicability import ReplicabilityTest
from .diags.Simulations import SimulationData
from .utils.config_scores import ISOConfig, MJOConfig, TCConfig

try:
    __version__ = version("pyhanami")
except Exception:
    # Package metadata is unavailable (e.g., running from an uninstalled source)
    __version__ = "unknown"

__all__ = [
    "DataDiagnostics", 
    "ObservationData", 
    "ReplicabilityTest", 
    "ScientificEvaluation", 
    "SimulationData", 
    "ISOConfig", 
    "MJOConfig", 
    "TCConfig",
]
