"""Longitudinal spectrum E11(k) from streamwise velocity (Taylor hypothesis)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def velocity_series_to_e11_spectrum(
    u: np.ndarray,
    fs_hz: float,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, str, dict[str, object]]:
    """
    One-sided PSD → E11 with k1 = 2π f / U_mean and E11 = P_uu * U_mean / (2π).

    ``method``: ``"welch"`` (Hann segments) or ``"fft"`` (full-record boxcar periodogram).

    Returns ``k1_positive, e11, u_mean, variance_puu_integral, method_used, meta``.
    """
    u = np.asarray(u, dtype=float)
    u_mean = float(np.mean(u))
    u_fluc = u - u_mean

    m = method.strip().lower()
    if m not in ("welch", "fft"):
        raise ValueError(f"method must be 'welch' or 'fft', got {m!r}.")

    if m == "welch":
        if nperseg is None:
            nperseg = min(65_536, max(8192, u.size // 4))
        if noverlap is None:
            noverlap = nperseg // 2

        f_hz, p_uu = signal.welch(
            u_fluc,
            fs=fs_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            average="mean",
        )
        meta: dict[str, object] = {
            "psd_backend": "scipy.signal.welch",
            "window": "hann",
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "scaling": "density",
            "detrend": False,
            "average": "mean",
        }
    else:
        f_hz, p_uu = signal.periodogram(
            u_fluc,
            fs=fs_hz,
            window="boxcar",
            scaling="density",
            return_onesided=True,
            detrend=False,
        )
        meta = {
            "psd_backend": "scipy.signal.periodogram",
            "window": "boxcar",
            "scaling": "density",
            "return_onesided": True,
            "detrend": False,
        }

    k = (2.0 * np.pi * f_hz) / u_mean
    e11 = p_uu * u_mean / (2.0 * np.pi)
    variance_from_psd = float(np.trapezoid(p_uu, f_hz))
    mask = k > 0.0
    return k[mask], e11[mask], u_mean, variance_from_psd, m, meta
