import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_pdf(data):
    kde = gaussian_kde(velocity_magnitudes_array)
    x_vals = np.linspace(velocity_magnitudes_array.min(),
                         velocity_magnitudes_array.max(), 200)
    pdf_vals = kde(x_vals)

    plt.figure()
    plt.plot(x_vals, pdf_vals, label='PDF')
    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Probability Density')
    plt.title('PDF of Velocity Magnitudes')
    plt.legend()
    plt.show()
