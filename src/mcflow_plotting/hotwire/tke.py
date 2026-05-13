"""Helpers for TKE / Kolmogorov-normalized spectrum workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_calibrated_hotwire_velocity
from .spectrum import velocity_series_to_e11_spectrum
from .turbulence import (
    NU_AIR,
    dissipation_epsilon,
    kolmogorov_length,
    kolmogorov_velocity,
)


def spectrum_estimator_label(method_used: str) -> str:
    if method_used == "welch":
        return "Welch PSD (Hann, averaged segments)"
    if method_used == "fft":
        return "Full-record FFT (boxcar periodogram)"
    return method_used


def resolve_spectrum_legend_label(path: Path, user_label: str, *, n_datasets: int) -> str:
    t = user_label.strip()
    if t:
        return t
    if n_datasets == 1:
        return r"$E_{11}(k) / (u_\eta^2 \eta)$"
    return path.stem


def compute_normalized_spectrum_run(
    path: Path | str,
    user_legend: str,
    *,
    spectrum_method: str,
    nu: float = NU_AIR,
    n_datasets: int,
) -> dict[str, object]:
    """Load HDF5, PSD, ε, η, u_η, and Kolmogorov-normalized spectrum arrays."""
    p = Path(path)
    cv = load_calibrated_hotwire_velocity(p)
    k, e11, u_mean, spec_var, method_used, meta = velocity_series_to_e11_spectrum(
        cv.u, cv.fs_hz, method=spectrum_method
    )
    eps = dissipation_epsilon(k, e11, nu=nu)
    eta = kolmogorov_length(eps, nu=nu)
    u_eta = kolmogorov_velocity(eps, nu=nu)
    uprime2 = float(np.var(cv.u - u_mean))
    k_eta = k * eta
    e11_norm = e11 / (u_eta**2 * eta)
    legend = resolve_spectrum_legend_label(p, user_legend, n_datasets=n_datasets)
    return {
        "legend": legend,
        "path": str(p),
        "dpath": cv.dataset_path,
        "n_u": int(cv.u.size),
        "fs_hz": float(cv.fs_hz),
        "u_mean": float(u_mean),
        "uprime2": uprime2,
        "spec_var": float(spec_var),
        "eps": float(eps),
        "eta": float(eta),
        "u_eta": float(u_eta),
        "k_eta": k_eta,
        "e11_norm": e11_norm,
        "spectrum_meta": meta,
        "spectrum_method": method_used,
    }


def fit_kolmogorov_reference_line(
    k_eta: np.ndarray,
    e11_norm: np.ndarray,
    *,
    k_lo: float = 0.02,
    k_hi: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return ``(k_eta_ref, e_ref, c_ref)`` with e_ref = c_ref * k_eta_ref**(-5/3).

    Amplitude ``c_ref`` is set from the median of kη·E_norm^(3/5) in the band, else a fallback bin.
    """
    lo, hi = k_lo, k_hi
    band = (k_eta >= lo) & (k_eta <= hi) & np.isfinite(e11_norm) & (e11_norm > 0)
    if np.count_nonzero(band) >= 5:
        km = float(np.median(k_eta[band]))
        em = float(np.median(e11_norm[band]))
        c_ref = em * km ** (5.0 / 3.0)
    else:
        i = len(k_eta) // 3
        km = float(k_eta[i])
        em = float(e11_norm[i])
        c_ref = em * km ** (5.0 / 3.0)

    k_eta_ref = np.logspace(
        np.log10(lo),
        np.log10(min(hi, float(np.max(k_eta)) * 0.8)),
        50,
    )
    e_ref = c_ref * k_eta_ref ** (-5.0 / 3.0)
    return k_eta_ref, e_ref, c_ref
