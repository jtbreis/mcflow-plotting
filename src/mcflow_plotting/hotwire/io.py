"""HDF5 IO for calibration files and calibrated hot-wire velocity."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

from ..constants import CALIBRATED_VELOCITY_DATASET


class CalibratedVelocity(NamedTuple):
    """Mean-removed analysis uses ``u``; ``fs`` from file attrs (Hz)."""

    u: np.ndarray
    fs_hz: float
    dataset_path: str


def find_dataset_by_leaf(h5: h5py.File, leaf_name: str) -> str:
    """Return first HDF5 path whose final component equals ``leaf_name``."""

    candidates: list[str] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and name.rsplit("/", 1)[-1] == leaf_name:
            candidates.append(name)

    h5.visititems(visitor)
    if not candidates:
        raise KeyError(f"No dataset with leaf name {leaf_name!r} in {h5.filename!r}.")
    return candidates[0]


def load_calibrated_hotwire_velocity(
    path: Path | str,
    *,
    dataset_leaf: str = CALIBRATED_VELOCITY_DATASET,
) -> CalibratedVelocity:
    """Load calibrated streamwise velocity (m s⁻¹) and sample rate."""
    p = Path(path)
    with h5py.File(p, "r") as f:
        fs = float(f.attrs.get("sample_rate_Hz", 5000.0))
        dpath = find_dataset_by_leaf(f, dataset_leaf)
        u = f[dpath][:].astype(float)
    return CalibratedVelocity(u=u, fs_hz=fs, dataset_path=dpath)


def collect_calibration_voltage_pitot_means(
    calibration_h5: Path | str,
    *,
    v_min_strict: float,
    v_max_strict: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return sorted (mean raw V, mean pitot U) per run group; drop means outside (v_min, v_max) open interval.

    Returns ``(v_mean, u_mean, n_skipped)``.
    """
    p = Path(calibration_h5)
    v_means: list[float] = []
    u_means: list[float] = []
    with h5py.File(p, "r") as f:
        for run_name in f:
            g = f[run_name]
            u = g["Velocity in ms-1"]["Velocity in ms-1"][:]
            v = g["hot-wire"]["raw_data"][:]
            u_means.append(float(np.mean(u)))
            v_means.append(float(np.mean(v)))
    order = np.argsort(v_means)
    v_arr = np.asarray(v_means, dtype=float)[order]
    u_arr = np.asarray(u_means, dtype=float)[order]
    ok = (v_arr > v_min_strict) & (v_arr < v_max_strict)
    n_skipped = int(np.size(ok) - np.count_nonzero(ok))
    return v_arr[ok], u_arr[ok], n_skipped


def load_first_group_pitot_and_hotwire_voltage(
    path: Path | str,
) -> tuple[str, np.ndarray, np.ndarray]:
    """First top-level group: pitot velocity (m/s) and raw hot-wire voltage arrays."""
    p = Path(path)
    with h5py.File(p, "r") as f:
        run_name = next(iter(f.keys()))
        g = f[run_name]
        u = g["Velocity in ms-1"]["Velocity in ms-1"][:].astype(float)
        v = g["hot-wire"]["raw_data"][:].astype(float)
    return str(run_name), u, v


def write_pitot_calibrated_hotwire_h5(
    output_h5: Path | str,
    *,
    run_name: str,
    u_pitot: np.ndarray,
    u_hotwire_calibrated: np.ndarray,
    fs_hz: float,
    source_h5: Path | str,
    calibration_h5: Path | str,
    poly_degree: int,
    poly_coeffs_high_first: np.ndarray,
    v_clip_inclusive: tuple[float, float],
    calibration_v_mean_strict_range: str,
) -> Path:
    """Write pitot + calibrated hot-wire velocities (gzip datasets)."""
    out = Path(output_h5)
    with h5py.File(out, "w") as f:
        grp = f.create_group(run_name)
        vel = grp.create_group("Pitot")
        vel.create_dataset(
            "Velocity Pitot in ms-1", data=u_pitot, compression="gzip"
        )
        grp.create_dataset(
            "Velocity_calibrated_hotwire_ms",
            data=u_hotwire_calibrated,
            compression="gzip",
        )
        f.attrs["sample_rate_Hz"] = fs_hz
        f.attrs["source_h5"] = str(Path(source_h5))
        f.attrs["calibration_h5"] = str(Path(calibration_h5))
        f.attrs["calibration_poly_degree"] = poly_degree
        f.attrs["calibration_poly_coeffs_high_first"] = poly_coeffs_high_first
        f.attrs["hotwire_raw_V_clip_inclusive"] = v_clip_inclusive
        f.attrs["hotwire_calibration_mean_V_strict_range"] = calibration_v_mean_strict_range
    return out
