import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from ..settings.standard import set_font, set_ticks_sig


def plot_pdf(data, xlabel='velocity', ylabel='PDF', scale=1, unit='m/s', variable='v', title=None, output=None):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data /= scale
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(),
                         data.max(), 200)
    pdf_vals = kde(x_vals)

    plt.figure()
    set_font()
    set_ticks_sig(ndigits=1)
    plt.plot(x_vals, pdf_vals, label='PDF')
    plt.xlabel(xlabel + f' in {unit}')
    plt.ylabel(ylabel + f'$({variable})$')
    plt.title(title)
    mean = data.mean()
    std = data.std()
    plt.text(0.05, 0.95, fr"$\langle {variable} \rangle$: {mean: .3f} {unit}" "\n" fr"$\sigma_{{{variable}}}$: {std: .3f} {unit}", transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    if output is not None:
        plt.savefig(output+f'/plots/PDF_{variable}.pdf', format='pdf')
    plt.show()


def plot_pdf_log(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(),
                         data.max(), 200)
    pdf_vals = kde(x_vals)

    plt.figure()
    plt.plot(x_vals, pdf_vals, label='PDF')
    plt.yscale('log')
    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Probability Density')
    plt.title('PDF of Velocity Magnitudes')
    plt.legend()
    plt.show()
