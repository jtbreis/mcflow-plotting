"""Hot-wire / pitot HDF5 IO, calibration, spectra, and Taylor-scale utilities."""

from ..constants import CALIBRATED_VELOCITY_DATASET, NU_AIR
from .calibration import CalibrationFit, fit_voltage_to_velocity_poly
from .io import (
    CalibratedVelocity,
    collect_calibration_voltage_pitot_means,
    find_dataset_by_leaf,
    load_calibrated_hotwire_velocity,
    load_first_group_pitot_and_hotwire_voltage,
    write_pitot_calibrated_hotwire_h5,
)
from .reports import markdown_row, print_spectrum_markdown_tables
from .spectrum import velocity_series_to_e11_spectrum
from .taylor import (
    dissipation_from_time_derivatives,
    epsilon_from_welch_spectrum,
    integral_length_scale_longitudinal,
    taylor_microscale_from_epsilon,
    taylor_microscale_from_series,
    welch_spectrum_for_epsilon,
)
from .tke import (
    compute_normalized_spectrum_run,
    fit_kolmogorov_reference_line,
    resolve_spectrum_legend_label,
    spectrum_estimator_label,
)
from .turbulence import (
    dissipation_epsilon,
    kolmogorov_length,
    kolmogorov_velocity,
)

__all__ = [
    "NU_AIR",
    "CALIBRATED_VELOCITY_DATASET",
    "CalibratedVelocity",
    "CalibrationFit",
    "collect_calibration_voltage_pitot_means",
    "compute_normalized_spectrum_run",
    "dissipation_epsilon",
    "dissipation_from_time_derivatives",
    "epsilon_from_welch_spectrum",
    "find_dataset_by_leaf",
    "fit_kolmogorov_reference_line",
    "fit_voltage_to_velocity_poly",
    "integral_length_scale_longitudinal",
    "kolmogorov_length",
    "kolmogorov_velocity",
    "load_calibrated_hotwire_velocity",
    "load_first_group_pitot_and_hotwire_voltage",
    "markdown_row",
    "print_spectrum_markdown_tables",
    "resolve_spectrum_legend_label",
    "spectrum_estimator_label",
    "taylor_microscale_from_epsilon",
    "taylor_microscale_from_series",
    "velocity_series_to_e11_spectrum",
    "welch_spectrum_for_epsilon",
    "write_pitot_calibrated_hotwire_h5",
]
