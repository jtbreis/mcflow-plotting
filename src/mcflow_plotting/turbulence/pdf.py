import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from ..settings.standard import set_font, set_ticks_sig
from scipy.stats import norm


def plot_pdf(data, scale=1, unit='m/s', variable=['V'], labels=None, xlim=[-4, 4], title=None, output=None, name=None, figsize=(5, 4), log=False):
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
        plt.savefig(output+f'/plots/PDF_{name}.svg',
                    format='svg', bbox_inches='tight')
    plt.show()


def plot_normalized_pdf(data, scale=1, variable='v', unit='m/s', title=None, output=None, name=None, figsize=(5, 4), log=False):
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
            output+f'/plots/PDF_norm_{name}.svg', format='svg', bbox_inches='tight')
    plt.show()
