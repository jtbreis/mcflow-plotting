import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from ..settings.standard import set_font, set_ticks_sig
from scipy.stats import norm


def plot_pdf(data, scale=1, unit='m/s', variable=['V'], labels=None, xlim=[-4, 4], title=None, output=None, name=None, figsize=(5, 4), log=False, fileformat='svg'):
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600', '#5E3C99']

    if not isinstance(data, list):
        data = [data]

    plt.figure(figsize=figsize)
    set_font()
    set_ticks_sig(ndigits=1)
    plt.xlabel(f'${variable}$ in $\mathrm{{{unit}}}$')
    plt.ylabel(rf'$\mathrm{{PDF}} \left( {variable} \right)$')
    plt.xlim(xlim)
    plt.title(title)
    if log is True:
        plt.yscale('log')

    for idx, dataset in enumerate(data):
        if not isinstance(dataset, np.ndarray):
            dataset = np.array(dataset)
            dataset /= scale
            kde = gaussian_kde(dataset)
            x_vals = np.linspace(dataset.min(),
                                 dataset.max(), 200)
            pdf_vals = kde(x_vals)
            plt.plot(x_vals, pdf_vals,
                     label=f'{labels[idx]}', color=colors[idx], alpha=0.8)

            mean = dataset.mean()
            std = dataset.std()
            # Prepare annotation text for multiple datasets
            annotation = f"{labels[idx]}" "\n" fr"$\langle {variable} \rangle: {mean: .3f} \, \mathrm{{{unit}}}$" "\n" fr"$\sigma_{{{variable}}}: {std: .3f} \, \mathrm{{{unit}}}$"
            if idx == 0:
                annotations = [annotation]
            else:
                annotations.append(annotation)

    # After plotting all datasets, add the annotation box to the right
    if 'annotations' in locals():
        plt.gcf().text(
            1.2, 0.95,
            "\n\n".join(annotations),
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    if labels is not None:
        plt.legend(loc='best', fontsize=10)

    plt.tight_layout()
    if output is not None:
        plt.savefig(output+f'/plots/PDF_{name}.{fileformat}',
                    format=fileformat, bbox_inches='tight')
    plt.show()


def plot_normalized_pdf(data, scale=1, variable='v', unit='m/s', title=None, output=None, name=None, figsize=(5, 4), log=False, fileformat='svg'):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data /= scale
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(),
                         data.max(), 200)
    pdf_vals = kde(x_vals)

    plt.figure(figsize=figsize)
    set_font()
    set_ticks_sig(ndigits=1)
    plt.plot(x_vals, pdf_vals,
             label=fr'$\mathrm{{PDF}}(\hat{{{variable}}})$', color='#003f5c')
    x_norm = np.linspace(-4, 4, 200)
    plt.plot(x_norm, norm.pdf(x_norm),
             label='$N(0,1)$', linestyle='--', color='#ffa600')
    plt.legend(loc='upper right')
    plt.xlabel(
        fr'$\left( {variable} - \langle {variable} \rangle \right) / \sigma_{{{variable}}}$')
    plt.ylabel(f'$\mathrm{{PDF}}(\hat{{{variable}}})$')
    plt.xlim(-4, 4)
    if log is True:
        plt.yscale('log')
    plt.title(title)

    plt.text(0.05, 0.95, fr"$\langle {variable} \rangle: {mean: .3f} \, \mathrm{{{unit}}}$" "\n" fr"$\sigma_{{{variable}}}: {std: .3f} \, \mathrm{{{unit}}}$", transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.tight_layout()
    if output is not None:
        plt.savefig(
            output+f'/plots/PDF_norm_{name}.{fileformat}', format=fileformat, bbox_inches='tight')
    plt.show()


def plot_conditional_pdf(data_x, data_y, condition_bins, scale=1, unit='m/s',
                         variable='V', condition_label='X',
                         labels=None, xlim=[-4, 4], title=None,
                         output=None, name=None, figsize=(5, 4), log=False):
    """
    Plot conditional PDFs p(variable | condition_bin) from two datasets.

    Parameters
    ----------
    data_x : array-like
        Conditioning variable (e.g., position, time).
    data_y : array-like
        Target variable whose PDF is plotted (e.g., velocity).
    condition_bins : array-like
        Bin edges for the conditioning variable.
    scale : float
        Scale factor for data_y.
    unit : str
        Unit of the target variable.
    variable : str
        Variable name for y-axis labeling.
    condition_label : str
        Label for the conditioning variable (e.g., 'x', 'time').
    labels : list of str, optional
        Labels for each conditional PDF bin.
    xlim : list
        Limits for the x-axis.
    title : str
        Plot title.
    output : str
        Folder to save the figure to (if provided).
    name : str
        File name for saving the figure (without extension).
    figsize : tuple
        Figure size.
    log : bool
        Whether to use log scale for y-axis.
    """
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600', '#5E3C99']

    plt.figure(figsize=figsize)
    set_font()
    set_ticks_sig(ndigits=1)
    plt.xlabel(f'${variable}$ in $\mathrm{{{unit}}}$')
    plt.ylabel(
        rf'$\mathrm{{PDF}} \left( {variable} \mid {condition_label} \right)$')
    plt.xlim(xlim)
    if title:
        plt.title(title)
    if log:
        plt.yscale('log')

    annotations = []
    for i in range(len(condition_bins) - 1):
        mask = (data_x >= condition_bins[i]) & (data_x < condition_bins[i+1])
        subset = np.array(data_y[mask]) / scale
        if subset.size < 5:  # skip sparse bins
            continue

        kde = gaussian_kde(subset)
        x_vals = np.linspace(subset.min(), subset.max(), 200)
        pdf_vals = kde(x_vals)

        label = labels[i] if labels and i < len(labels) else (
            f'{condition_bins[i]:.2f} ≤ {condition_label} < {condition_bins[i+1]:.2f}'
        )

        plt.plot(x_vals, pdf_vals,
                 label=label,
                 color=colors[i % len(colors)],
                 alpha=0.8)

        mean = subset.mean()
        std = subset.std()
        annotations.append(
            f"{label}\n"
            fr"$\langle {variable} \rangle: {mean:.3f}\,\mathrm{{{unit}}}$"
            "\n" fr"$\sigma_{{{variable}}}: {std:.3f}\,\mathrm{{{unit}}}$"
        )

    # if annotations:
    #     plt.gcf().text(
    #         1.2, 0.95, "\n\n".join(annotations),
    #         ha='right', va='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    #     )

    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    if output and name:
        plt.savefig(f'{output}/plots/ConditionalPDF_{name}.svg',
                    format='svg', bbox_inches='tight')

    plt.show()


def plot_conditional_normalized_pdf(data_x, data_y, condition_bins, scale=1,
                                    variable='v', condition_label='x', unit='m/s',
                                    labels=None, title=None, output=None, name=None,
                                    figsize=(5, 4), log=False):
    """
    Plot normalized conditional PDFs p( (v - <v>) / sigma_v | x ) across condition bins.

    Parameters
    ----------
    data_x : array-like
        Conditioning variable (e.g., position, time).
    data_y : array-like
        Variable to plot PDFs of (e.g., velocity).
    condition_bins : array-like
        Bin edges for conditioning variable.
    scale : float, optional
        Scale factor for data_y.
    variable : str, optional
        Variable name for axis labels.
    condition_label : str, optional
        Label for conditioning variable.
    unit : str, optional
        Unit of the variable.
    labels : list of str, optional
        Labels for each conditional subset.
    title : str, optional
        Title of the plot.
    output : str, optional
        Folder path to save output plots.
    name : str, optional
        Base name for saved plot file.
    figsize : tuple, optional
        Figure size.
    log : bool, optional
        Whether to use log scale on the y-axis.
    """
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600', '#5E3C99']

    plt.figure(figsize=figsize)
    set_font()
    set_ticks_sig(ndigits=1)
    plt.xlabel(
        fr'$\left( {variable} - \langle {variable} \rangle \right) / \sigma_{{{variable}}}$')
    plt.ylabel(
        rf'$\mathrm{{PDF}} \left( \hat{{{variable}}} \mid {condition_label} \right)$')
    plt.xlim(-4, 4)
    if title:
        plt.title(title)
    if log:
        plt.yscale('log')

    annotations = []
    for i in range(len(condition_bins) - 1):
        mask = (data_x >= condition_bins[i]) & (data_x < condition_bins[i+1])
        subset = np.array(data_y[mask]) / scale
        if subset.size < 5:
            continue

        mean = subset.mean()
        std = subset.std()
        subset_norm = (subset - mean) / std

        kde = gaussian_kde(subset_norm)
        x_vals = np.linspace(subset_norm.min(), subset_norm.max(), 200)
        pdf_vals = kde(x_vals)

        label = labels[i] if labels and i < len(labels) else (
            f'{condition_bins[i]:.2f} ≤ {condition_label} < {condition_bins[i+1]:.2f}'
        )

        plt.plot(x_vals, pdf_vals, label=label,
                 color=colors[i % len(colors)], alpha=0.8)

        annotations.append(
            f"{label}\n"
            fr"$\langle {variable} \rangle: {mean:.3f}\,\mathrm{{{unit}}}$"
            "\n" fr"$\sigma_{{{variable}}}: {std:.3f}\,\mathrm{{{unit}}}$"
        )

    # Overlay standard normal for reference
    x_norm = np.linspace(-4, 4, 200)
    plt.plot(x_norm, norm.pdf(x_norm),
             label='$N(0,1)$', linestyle='--', color='#ffa600')

    if annotations:
        plt.gcf().text(
            1.2, 0.95, "\n\n".join(annotations),
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    if output and name:
        plt.savefig(
            f'{output}/plots/ConditionalPDF_norm_{name}.svg',
            format='svg', bbox_inches='tight'
        )

    plt.show()
