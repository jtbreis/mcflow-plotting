"""Polynomial U(V): pitot mean velocity vs mean raw hot-wire voltage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationFit:
    poly_degree: int
    coeffs_high_first: np.ndarray
    rms_residual_at_means: float

    def as_poly1d(self) -> np.poly1d:
        return np.poly1d(self.coeffs_high_first)

    def apply_voltages(
        self,
        v_raw: np.ndarray,
        *,
        clip_low: float | None = None,
        clip_high: float | None = None,
    ) -> np.ndarray:
        v = np.asarray(v_raw, dtype=float)
        if clip_low is not None and clip_high is not None:
            v = np.clip(v, clip_low, clip_high)
        return np.asarray(self.as_poly1d()(v), dtype=float)


def fit_voltage_to_velocity_poly(
    v_mean: np.ndarray,
    u_mean: np.ndarray,
    *,
    max_degree: int = 3,
) -> CalibrationFit:
    """Least-squares polynomial with degree ``min(max_degree, n_points-1)``."""
    v_mean = np.asarray(v_mean, dtype=float)
    u_mean = np.asarray(u_mean, dtype=float)
    if v_mean.size < 2:
        raise ValueError("Need at least two calibration points.")
    deg = min(int(max_degree), v_mean.size - 1)
    coeffs = np.polyfit(v_mean, u_mean, deg=deg)
    poly = np.poly1d(coeffs)
    rms = float(np.sqrt(np.mean((poly(v_mean) - u_mean) ** 2)))
    return CalibrationFit(
        poly_degree=int(deg),
        coeffs_high_first=np.asarray(coeffs, dtype=float),
        rms_residual_at_means=rms,
    )
