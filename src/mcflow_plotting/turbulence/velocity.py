import numpy as np

import matplotlib.pyplot as plt
from ..settings.standard import set_font, set_ticks_sig
from ..style.colors import FLOW_COLORS


def plot_rms_velocity(velocity, xlabel='Position', ylabel='RMS Velocity', title='RMS Velocity Profile'):
    """
    Plot the root mean square (RMS) velocity along a specified axis.

    Parameters:
        velocity (np.ndarray): Velocity array, shape (..., N, ...).
        axis (int): Axis along which to compute RMS.
        x (np.ndarray or None): Optional x-axis values. If None, use indices.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Plot title.
    """
    if not isinstance(velocity, np.ndarray):
        velocity = np.array(velocity)
    mean_velocity = np.mean(velocity)
    rms = np.sqrt(np.square(velocity - mean_velocity))
    set_font()
    plt.figure()
    plt.hist(
        rms.flatten(),
        bins=30,
        alpha=0.35,
        color=FLOW_COLORS[0],
        edgecolor="0.25",
        linewidth=0.4,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.6)
    plt.show()


def plot_mean_vel_time(velocity, time, scale=1, unit='m/s', variable='V', output=None):
    """
    Plot the mean velocity as a function of time.

    Parameters:
        velocity (np.ndarray): Velocity array, shape (T, ...), where T is the number of time steps.
        time (np.ndarray): Array of time values, shape (T,).
    """
    if not isinstance(velocity, np.ndarray):
        velocity = np.array(velocity)
    if not isinstance(time, np.ndarray):
        time = np.array(time)
    velocity /= scale

    sorted_indices = np.argsort(time)
    time = time[sorted_indices]
    velocity = velocity[sorted_indices]

    unique_times, inverse_indices = np.unique(time, return_inverse=True)
    mean_velocity = np.array(
        [np.mean(velocity[inverse_indices == i]) for i in range(len(unique_times))])
    std_velocity = np.array(
        [np.std(velocity[inverse_indices == i]) for i in range(len(unique_times))])
    time = unique_times
    set_font()
    plt.figure()
    set_ticks_sig(ndigits=1)
    plt.plot(
        time,
        mean_velocity,
        marker="o",
        markersize=4,
        lw=1.6,
        color=FLOW_COLORS[0],
        label=fr"$\langle {variable} \rangle$",
    )
    plt.fill_between(
        time,
        mean_velocity - std_velocity,
        mean_velocity + std_velocity,
        facecolor=FLOW_COLORS[0],
        alpha=0.22,
        interpolate=True,
        label=fr'$\sigma_{{{variable}}}$',
    )
    plt.legend(loc='upper right', fontsize=8)
    plt.xlabel('time in sec')
    plt.ylabel(f'velocity in {unit}')
    plt.title('Velocity Over Time', fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.6)
    if output is not None:
        plt.savefig(
            output + f'/plots/{variable}_over_time.pdf',
            format='pdf',
            bbox_inches='tight',
            dpi=300,
        )
    plt.show()
