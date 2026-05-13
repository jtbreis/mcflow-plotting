"""Markdown reporting for spectrum / configuration tables."""

from __future__ import annotations

import os


def markdown_row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def print_spectrum_markdown_tables(
    *,
    spectrum_method: str,
    spectrum_description: str,
    spectrum_meta: dict[str, object],
    nu_air: float,
    k_ref_lo: float,
    k_ref_hi: float,
    runs: list[dict[str, object]],
) -> None:
    """Print shared PSD configuration and one Markdown row per dataset."""
    print()
    print("<!-- copy/paste into Markdown -->")
    print()
    print("### Spectrum run — configuration (shared)")
    print()
    print(markdown_row(("Setting", "Value")))
    print(markdown_row(("---", "---")))
    rows_cfg: list[tuple[str, str]] = [
        (
            "Env `TKE_SPECTRUM_METHOD`",
            os.environ.get("TKE_SPECTRUM_METHOD", "(unset)"),
        ),
        ("Effective spectrum method", spectrum_method),
        ("Spectrum estimator", spectrum_description),
        ("PSD backend", str(spectrum_meta.get("psd_backend", ""))),
        ("PSD `scaling`", str(spectrum_meta.get("scaling", ""))),
        ("PSD window", str(spectrum_meta.get("window", ""))),
    ]
    if spectrum_method == "welch":
        rows_cfg.extend(
            [
                (
                    "`nperseg` (first run; depends on N if default)",
                    str(spectrum_meta.get("nperseg", "")),
                ),
                ("`noverlap` (first run)", str(spectrum_meta.get("noverlap", ""))),
                ("`average`", str(spectrum_meta.get("average", ""))),
            ]
        )
    else:
        rows_cfg.append(
            ("`return_onesided`", str(spectrum_meta.get("return_onesided", "")))
        )
    rows_cfg.extend(
        [
            ("`detrend` (PSD)", str(spectrum_meta.get("detrend", ""))),
            ("ν (air, ε integral)", f"{nu_air:g} m²/s"),
            (
                "Reference slope fit band $k\\eta$ (from 1st dataset)",
                f"[{k_ref_lo:g}, {k_ref_hi:g}]",
            ),
            (
                "Taylor hypothesis",
                r"$k_1 = 2\pi f / \bar{U}$, $\bar{U}=\langle u\rangle$",
            ),
            (
                "$E_{11}$ from one-sided $P_{uu}$",
                r"$E_{11}(k_1)=P_{uu}(f)\,\bar{U}/(2\pi)$",
            ),
            (
                "Dissipation model",
                r"$\varepsilon = 15\nu \int k_1^2 E_{11}\,\mathrm{d}k_1$",
            ),
            ("Number of datasets plotted", str(len(runs))),
        ]
    )
    for label, val in rows_cfg:
        print(markdown_row((label, val)))

    print()
    print("### Spectrum run — per dataset")
    print()
    hdr = (
        "Legend",
        "HDF5",
        "Dataset in file",
        "$N$",
        "$f_s$ (Hz)",
        r"$\bar{U}$ (m/s)",
        r"$\langle u'^2\rangle$",
        r"$\int P_{uu}\mathrm{d}f$",
        r"$\varepsilon$",
        r"$\eta$ (m)",
        r"$u_\eta$ (m/s)",
        "`nperseg`",
        "`noverlap`",
    )
    print(markdown_row(hdr))
    print(markdown_row(("---",) * len(hdr)))
    for r in runs:
        meta_r = r.get("spectrum_meta") or {}
        nps_v = meta_r.get("nperseg")
        nov_v = meta_r.get("noverlap")
        nps = str(nps_v) if nps_v is not None else "—"
        nov = str(nov_v) if nov_v is not None else "—"
        print(
            markdown_row(
                (
                    str(r["legend"]),
                    str(r["path"]),
                    str(r["dpath"]),
                    str(r["n_u"]),
                    f"{float(r['fs_hz']):g}",
                    f"{float(r['u_mean']):.6f}",
                    f"{float(r['uprime2']):.6e}",
                    f"{float(r['spec_var']):.6e}",
                    f"{float(r['eps']):.6e}",
                    f"{float(r['eta']):.6e}",
                    f"{float(r['u_eta']):.6e}",
                    nps,
                    nov,
                )
            )
        )
    print()
