"""Physical constants and shared dataset names (no heavy dependencies).

Import this module first from ``mcflow_plotting`` so notebooks can use ``NU_AIR``
without pulling in HDF5 / spectrum code.
"""

# Kinematic viscosity of air at ~room temperature (m² s⁻¹)
NU_AIR = 1.5e-5

# Dataset leaf name written by the hot-wire calibration pipeline
CALIBRATED_VELOCITY_DATASET = "Velocity_calibrated_hotwire_ms"

__all__ = ["NU_AIR", "CALIBRATED_VELOCITY_DATASET"]
