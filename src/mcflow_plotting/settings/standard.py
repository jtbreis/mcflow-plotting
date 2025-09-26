from matplotlib.ticker import ScalarFormatter
import enum as enum
import matplotlib.pyplot as plt


class Enum():
    linewidth = 2
    colormap = 'viridis'


def set_font(base_size=12):
    """
    Configure Matplotlib to use LaTeX-like fonts (Computer Modern)
    without requiring a LaTeX installation.

    Parameters:
        base_size (int): Base font size for axes labels and ticks.
    """
    plt.rcParams.update({
        # Use built-in mathtext (no LaTeX needed)
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",      # Computer Modern math font
        "axes.labelsize": base_size + 2,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "legend.fontsize": base_size,
        "figure.titlesize": base_size + 4,
        "lines.linewidth": 2,
    })


def set_ticks_sig(ax=None, ndigits=2):
    """
    Set x and y axis ticks to display 3 significant digits.
    """
    if ax is None:
        ax = plt.gca()

    formatter = ScalarFormatter()
    formatter.set_useOffset(False)
    formatter.set_useMathText(True)
    # Optional, when to switch to scientific
    formatter.set_powerlimits((-3, 4))

    # Format tick labels to 3 significant digits
    if ndigits == 1:
        def sig(x, pos):
            return f"{x:.1f}"
    elif ndigits == 2:
        def sig(x, pos):
            return f"{x:.2f}"
    elif ndigits == 3:
        def sig(x, pos):
            return f"{x:.3f}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(sig))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(sig))
