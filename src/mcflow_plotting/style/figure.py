"""Figure helpers: lab font defaults, layout, save."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mcflow_plotting.settings.standard import set_font


def use_lab_matplotlib_style(*, set_fonts: bool = True) -> None:
    """Apply serif/mathtext rcParams via ``set_font()``."""
    if set_fonts:
        set_font()


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] = (7.0, 5.0),
    set_fonts: bool = True,
    **kwargs,
):
    """``plt.subplots`` after optional ``set_font()``."""
    use_lab_matplotlib_style(set_fonts=set_fonts)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def finalize_figure(fig, *, pad: float = 0.6) -> None:
    fig.tight_layout(pad=pad)


def save_figure(
    fig,
    path: str | Path,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
