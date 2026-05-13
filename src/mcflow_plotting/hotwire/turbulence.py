"""Kolmogorov scales and longitudinal dissipation from E11(k)."""

from __future__ import annotations

import numpy as np

from ..constants import NU_AIR


def dissipation_epsilon(
    k: np.ndarray,
    e11: np.ndarray,
    *,
    nu: float = NU_AIR,
) -> float:
    """ε = 15 ν ∫ k² E11(k) dk (longitudinal spectrum, isotropic relation)."""
    k = np.asarray(k, dtype=float)
    e11 = np.asarray(e11, dtype=float)
    return float(15.0 * nu * np.trapezoid((k**2) * e11, k))


def kolmogorov_length(epsilon: float, *, nu: float = NU_AIR) -> float:
    return float((nu**3 / epsilon) ** 0.25)


def kolmogorov_velocity(epsilon: float, *, nu: float = NU_AIR) -> float:
    """u_η = (ν ε)^(1/4)."""
    return float((nu * epsilon) ** 0.25)
