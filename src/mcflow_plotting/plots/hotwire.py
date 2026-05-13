"""Matplotlib helpers for hot-wire calibration, PDF, and TKE spectrum figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from mcflow_plotting.style.colors import FLOW_COLORS
from mcflow_plotting.style.figure import finalize_figure, use_lab_matplotlib_style


def plot_calibration_scatter_and_fit(
    *,
    v_mean: np.ndarray,
    u_mean: np.ndarray,
    v_curve: np.ndarray,
    u_curve: np.ndarray,
    poly_degree: int,
    rms_cal: float,
    v_min: float,
    v_max: float,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> tuple[Any, Any]:
    """Scatter of calibration means and polynomial U(V) with clip guides."""
    use_lab_matplotlib_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        v_mean,
        u_mean,
        s=48,
        zorder=3,
        color=FLOW_COLORS[0],
        edgecolors="0.2",
        linewidths=0.6,
        label="Calibration means (Ū_pitot vs V̄_raw)",
    )
    ax.plot(
        v_curve,
        u_curve,
        lw=1.6,
        color=FLOW_COLORS[1],
        alpha=0.9,
        label=f"U(V) poly fit, deg {poly_degree}",
    )
    ax.set_xlabel("Hot-wire mean raw voltage (V)")
    ax.set_ylabel("Pitot mean velocity (m s⁻¹)")
    ax.set_title(
        f"Calibration mapping (-5 V < V̄ < 5 V) — RMS at means: {rms_cal:.4f} m s⁻¹",
        fontsize=12,
    )
    ax.axvline(v_min, color="0.5", ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.axvline(v_max, color="0.5", ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    finalize_figure(fig)
    return fig, ax


def plot_pitot_vs_calibrated_hotwire(
    *,
    t_sub: np.ndarray,
    u_pitot_sub: np.ndarray,
    u_hotwire_sub: np.ndarray,
    poly_degree: int,
    v_clip_low: float,
    v_clip_high: float,
    run_name: str,
    figsize: tuple[float, float] = (10.0, 4.5),
) -> tuple[Any, Any]:
    use_lab_matplotlib_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        t_sub,
        u_pitot_sub,
        lw=1.6,
        color=FLOW_COLORS[0],
        alpha=0.9,
        zorder=2,
        label="Pitot velocity (measured)",
    )
    ax.plot(
        t_sub,
        u_hotwire_sub,
        lw=1.6,
        ls="--",
        color=FLOW_COLORS[1],
        alpha=0.9,
        zorder=3,
        label=(
            f"Hot-wire → velocity (deg-{poly_degree} poly, V clipped to "
            f"[{v_clip_low:g}, {v_clip_high:g}] V)"
        ),
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m s⁻¹)")
    ax.set_title(f"Pitot vs calibrated hot-wire — {run_name}", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    finalize_figure(fig)
    return fig, ax


def plot_hotwire_velocity_pdf(
    *,
    u: np.ndarray,
    fs_hz: float,
    h5_name: str,
    dataset_path: str,
    hist_bins: int = 120,
    use_kde: bool = False,
    kde_max_samples: int = 200_000,
    rng_seed: int = 0,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> tuple[Any, Any]:
    use_lab_matplotlib_style()
    u_mean = float(np.mean(u))
    u_std = float(np.std(u))
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        u,
        bins=hist_bins,
        density=True,
        color=FLOW_COLORS[0],
        alpha=0.35,
        edgecolor="0.25",
        linewidth=0.4,
        label="Histogram (density)",
    )
    if use_kde:
        rng = np.random.default_rng(rng_seed)
        n = u.size
        if n > kde_max_samples:
            idx = rng.choice(n, size=kde_max_samples, replace=False)
            u_kde = u[idx]
        else:
            u_kde = u
        kde = gaussian_kde(u_kde)
        x = np.linspace(float(np.min(u)), float(np.max(u)), 400)
        ax.plot(x, kde(x), color=FLOW_COLORS[1], lw=1.6, label="Gaussian KDE")

    ax.set_xlabel(r"Velocity $u$ (m s$^{-1}$)")
    ax.set_ylabel(r"PDF($u$)")
    ax.set_title(f"Hot-wire velocity PDF — {h5_name}\n{dataset_path}", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.02,
        0.98,
        rf"$\langle u\rangle = {u_mean:.4f}$ m s$^{{-1}}$, $\sigma_u = {u_std:.4f}$ m s$^{{-1}}$, $N={u.size}$, $f_s={fs_hz:.0f}$ Hz",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    finalize_figure(fig)
    return fig, ax


def _annotate_kolmogorov_slope(
    ax,
    k_eta_ref: np.ndarray,
    e_ref: np.ndarray,
    *,
    frac_along: float = 0.55,
) -> None:
    i_ann = int(round(frac_along * (len(k_eta_ref) - 1)))
    kx = float(k_eta_ref[i_ann])
    ky = float(e_ref[i_ann])
    ann = ax.annotate(
        r"$-5/3$",
        xy=(kx, ky),
        xytext=(14, 12),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="k",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.2",
            linewidth=1.2,
        ),
        zorder=6,
    )
    ann.set_path_effects(
        [pe.withStroke(linewidth=3.5, foreground="white"), pe.Normal()]
    )


def plot_tke_spectrum_kolmogorov_normalized(
    runs: Sequence[dict[str, Any]],
    *,
    k_eta_ref: np.ndarray,
    e_ref: np.ndarray,
    figsize: tuple[float, float] = (7.0, 5.0),
    annotate_slope: bool = True,
) -> tuple[Any, Any]:
    """
    Overlay ``runs[*]['k_eta']`` vs ``runs[*]['e11_norm']`` and a black dashed −5/3 reference.

    ``runs`` entries are dicts from ``mcflow_plotting.hotwire.compute_normalized_spectrum_run``.
    """
    use_lab_matplotlib_style()
    fig, ax = plt.subplots(figsize=figsize)
    for idx, r in enumerate(runs):
        ax.loglog(
            r["k_eta"],
            r["e11_norm"],
            color=FLOW_COLORS[idx % len(FLOW_COLORS)],
            lw=1.6,
            alpha=0.9,
            zorder=3 + idx,
            label=str(r["legend"]),
        )
    ax.loglog(
        k_eta_ref,
        e_ref,
        color="k",
        ls="--",
        lw=1.6,
        alpha=0.9,
        zorder=2,
    )
    if annotate_slope:
        _annotate_kolmogorov_slope(ax, k_eta_ref, e_ref)
    ax.set_xlabel(r"$k \eta$")
    ax.set_ylabel(r"$E_{11}(k) / (u_\eta^2 \eta)$")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", ls="-", alpha=0.25)
    finalize_figure(fig)
    return fig, ax
