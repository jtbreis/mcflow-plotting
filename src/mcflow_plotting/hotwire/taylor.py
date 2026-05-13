"""Taylor microscale, dissipation from time derivatives, integral length scale."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..constants import NU_AIR
from .spectrum import velocity_series_to_e11_spectrum
from .turbulence import dissipation_epsilon


def taylor_microscale_from_series(
    u: np.ndarray,
    fs: float,
) -> tuple[float, float, float, float]:
    """
    Return (lambda_m, u_rms, U_mean, mean_du_dt_sq).

    lambda_m = U * u_rms / sqrt(mean((du'/dt)^2))  with u' = u - U.
    """
    u = np.asarray(u, dtype=float)
    u_mean = float(np.mean(u))
    u_fluc = u - u_mean
    u_rms = float(np.sqrt(np.mean(u_fluc**2)))
    dt = 1.0 / fs
    du_dt = np.gradient(u_fluc, dt)
    mean_du_dt_sq = float(np.mean(du_dt**2))
    if mean_du_dt_sq <= 0.0:
        raise ValueError("mean (du'/dt)^2 is non-positive")
    lambda_m = float(u_mean * u_rms / np.sqrt(mean_du_dt_sq))
    return lambda_m, u_rms, u_mean, mean_du_dt_sq


def taylor_microscale_from_epsilon(
    u_rms: float,
    epsilon: float,
    *,
    nu: float = NU_AIR,
) -> float:
    """λ = (15 ν u_rms² / ε)^(1/2) for isotropic turbulence."""
    return float(np.sqrt(15.0 * nu * u_rms**2 / epsilon))


def dissipation_from_time_derivatives(
    u_mean: float,
    mean_du_dt_sq: float,
    *,
    nu: float = NU_AIR,
) -> float:
    """ε = 15 ν ⟨(du'/dt)²⟩ / Ū² (Taylor hypothesis)."""
    return float(15.0 * nu * mean_du_dt_sq / (u_mean**2))


def integral_length_scale_longitudinal(
    u: np.ndarray,
    fs: float,
    u_mean: float | None = None,
    *,
    max_lag_samples: int | None = None,
) -> tuple[float, float, bool]:
    """
    L11 = Ū T_L; T_L = ∫ R(τ)/⟨u′²⟩ dτ to first zero crossing of normalized autocorrelation.
    """
    u = np.asarray(u, dtype=float)
    if u_mean is None:
        u_mean = float(np.mean(u))
    u_fluc = u - u_mean
    n = int(u_fluc.size)
    if n < 4:
        raise ValueError("Need at least 4 samples for integral length scale.")

    ac_full = signal.correlate(u_fluc, u_fluc, mode="full", method="fft")
    mid = n - 1
    cov = ac_full[mid : mid + n].astype(float)
    cov /= float(n)
    var = float(cov[0])
    if var <= 0.0:
        raise ValueError("Zero variance in velocity fluctuations.")
    rho = cov / var
    tau = np.arange(n, dtype=float) / float(fs)

    m_max = n if max_lag_samples is None else min(max_lag_samples, n)
    rho = rho[:m_max]
    tau = tau[:m_max]

    neg = np.where(rho[1:] <= 0.0)[0]
    used_first_zero = bool(neg.size > 0)

    if used_first_zero:
        k1 = int(neg[0]) + 1
        t0 = float(tau[k1 - 1])
        t1 = float(tau[k1])
        r0 = float(rho[k1 - 1])
        r1 = float(rho[k1])
        if r0 <= 0.0:
            tz = t0
        else:
            tz = t0 - r0 * (t1 - t0) / (r1 - r0)
        if k1 >= 2:
            t_l = float(np.trapezoid(rho[:k1], tau[:k1]))
            t_l += 0.5 * r0 * (tz - t0)
        else:
            t_l = 0.5 * float(rho[0]) * tz
    else:
        t_l = float(np.trapezoid(rho, tau))

    l_11 = float(u_mean * t_l)
    return l_11, t_l, used_first_zero


def welch_spectrum_for_epsilon(
    u: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Welch E11 and variance integral (same defaults as ``velocity_series_to_e11_spectrum``)."""
    k, e11, u_mean, var_puu, _m, _meta = velocity_series_to_e11_spectrum(
        u, fs, method="welch", nperseg=nperseg, noverlap=noverlap
    )
    return k, e11, u_mean, var_puu


def epsilon_from_welch_spectrum(
    u: np.ndarray,
    fs: float,
    *,
    nu: float = NU_AIR,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> float:
    k, e11, _, _ = welch_spectrum_for_epsilon(
        u, fs, nperseg=nperseg, noverlap=noverlap
    )
    return dissipation_epsilon(k, e11, nu=nu)
