"""Online pulse check: IQ-plane (pc1) and waveform (pc2) plots of a few events.

Same quantities and figures as ``main_singlefile`` of ``analysis/ana_forJPS.py``,
but only the plotted events are processed (fast enough for online use) and
nothing is written next to the data.

Per event: the pedestal is the mean I and Q over the pre-trigger samples
(``--alpha``: the first 100 ns); the pedestal-subtracted I + iQ is rotated onto
the phase of its largest excursion, which gives the "Proj" waveform.
"""
import math
from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from kidpack.rawdata import time_axis_s

NROWS = NCOLS = 4
N_EVENTS = NROWS * NCOLS
FIGSIZE = (20, 17)  # inches; the saved PNGs are 2000 x 1700 px at SAVE_DPI
SAVE_DPI = 100
SHOW_FIGSIZE = (14, 12)  # window size on screen (what the old 2x2 macro used)
DEFAULT_REBIN = 5  # default of the kidpack-monitor command (make_pulse_view itself defaults to 1)
IQ_SCATTER_REBIN = 5  # every point shown in the IQ plane averages at least this many samples
ALPHA_PEDESTAL_S = 100e-9

_STYLE = {
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 10,
    'axes.formatter.useoffset': True,
    'axes.unicode_minus': False,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'legend.fontsize': 9,
    'lines.linewidth': 1.2,
}


@dataclass
class PulseView:
    """Everything the two plots need, for ``n`` events (voltages in V, time in us)."""
    event_id: np.ndarray  # (n,)
    tbin_us: np.ndarray  # (m,) time axis of the (re)binned waveforms
    i: np.ndarray  # (n, m) raw I, rebinned, pedestal NOT subtracted
    q: np.ndarray  # (n, m) raw Q, rebinned, pedestal NOT subtracted
    proj: np.ndarray  # (n, m) waveform rotated onto the peak phase, pedestal subtracted
    ped_i: np.ndarray  # (n,)
    ped_q: np.ndarray  # (n,)
    proj_max: np.ndarray  # (n,) largest |I + iQ - pedestal|
    proj_theta: np.ndarray  # (n,) phase of that largest excursion [rad]
    peak_index: np.ndarray  # (n,) index into the (re)binned waveforms where it occurs
    rebin_factor: int  # samples averaged into each bin of i, q, proj

    @property
    def n(self):
        return len(self.event_id)


def rebin(arr, factor):
    """Average groups of ``factor`` samples along the last axis (remainder dropped)."""
    factor = int(factor)
    if factor <= 1:
        return arr
    n = (arr.shape[-1] // factor) * factor
    return arr[..., :n].reshape(*arr.shape[:-1], n // factor, factor).mean(axis=-1)


def select_events(nevents, stride=1, n=N_EVENTS):
    """Indices of the events to plot: the first ``n`` events, every ``stride``-th."""
    if stride < 1:
        raise ValueError('stride must be >= 1')
    return np.arange(0, nevents, stride)[:n]


def make_pulse_view(raw, indices, rebin_factor=1, alpha=False):
    """Pedestal, peak phase and projected waveform of the events ``indices``.

    ``rebin_factor`` samples are averaged before the peak is located (1 = none).
    """
    rebin_factor = int(rebin_factor)
    if rebin_factor < 1:
        raise ValueError('rebin factor must be >= 1')
    sample_rate, npts = raw.sample_rate, raw.npts
    pedestal_end = int(ALPHA_PEDESTAL_S * sample_rate) if alpha else int(npts * raw.ref_position / 100)
    if pedestal_end < 1:
        raise ValueError('no samples available for the pedestal (trigger position 0 %?)')

    indices = np.asarray(indices)
    i_raw, q_raw = raw.ch0[indices], raw.ch1[indices]
    ped_i = i_raw[:, :pedestal_end].mean(axis=1)
    ped_q = q_raw[:, :pedestal_end].mean(axis=1)

    vc_corr = (i_raw + 1j * q_raw) - ped_i[:, None] - 1j * ped_q[:, None]
    vc_meas = rebin(vc_corr, rebin_factor)
    magnitude = np.abs(vc_meas)
    peak = np.argmax(magnitude, axis=1)
    theta = np.angle(vc_meas[np.arange(len(indices)), peak])
    proj = np.real(vc_meas * np.exp(-1j * theta)[:, None])

    nbins = npts // rebin_factor
    tbin_us = time_axis_s(nbins, sample_rate / rebin_factor, raw.ref_position) * 1e6
    return PulseView(event_id=raw.event_id[indices], tbin_us=tbin_us,
                     i=rebin(i_raw, rebin_factor), q=rebin(q_raw, rebin_factor), proj=proj,
                     ped_i=ped_i, ped_q=ped_q,
                     proj_max=magnitude.max(axis=1), proj_theta=theta, peak_index=peak,
                     rebin_factor=rebin_factor)


def _grid(view, make_figure):
    if view.n > N_EVENTS:
        raise ValueError(f'at most {N_EVENTS} events fit in the {NROWS}x{NCOLS} grid, got {view.n}')
    fig = make_figure(figsize=FIGSIZE, layout='constrained')
    axs = fig.subplots(NROWS, NCOLS, sharex=True, sharey=True, squeeze=False)
    for k, ax in enumerate(axs.flat):
        if k >= view.n:
            ax.axis('off')  # fewer events than panels
    return fig, axs


def _is_bottom(k, n):
    return k + NCOLS >= n  # nothing plotted below this panel


def plot_iq_plane(view, make_figure=Figure):
    """pc1: I/Q of each event coloured by time, with the pedestal (x) and the peak (*).

    The points are the rebinned samples in which the peak was searched, averaged
    further only if that leaves fewer than IQ_SCATTER_REBIN samples per point (so
    with the default rebin the star is one of the drawn points, and with rebin 1
    the points are 5-sample averages as in the old macro).

    ``make_figure`` creates the figure: ``Figure`` (default) for a plain figure to
    save, ``matplotlib.pyplot.figure`` for one that can be shown in a window.
    """
    with mpl.rc_context(_STYLE):
        fig, axs = _grid(view, make_figure)
        step = math.ceil(IQ_SCATTER_REBIN / view.rebin_factor)  # extra averaging, 1 = none
        vt = rebin(view.tbin_us, step)
        image = None
        for k in range(view.n):
            ax = axs.flat[k]
            v0 = rebin(view.i[k], step)
            v1 = rebin(view.q[k], step)
            image = ax.scatter(v0 * 1e3, v1 * 1e3, c=vt, cmap='viridis', s=8)
            ax.plot(view.ped_i[k] * 1e3, view.ped_q[k] * 1e3, 'x', ms=9, color='r', label='ped')
            ax.plot((view.proj_max[k] * np.cos(view.proj_theta[k]) + view.ped_i[k]) * 1e3,
                    (view.proj_max[k] * np.sin(view.proj_theta[k]) + view.ped_q[k]) * 1e3,
                    '*', ms=10, color='r', label='proj max')
            if _is_bottom(k, view.n):
                ax.set_xlabel('I [mV]')
            if k % NCOLS == 0:
                ax.set_ylabel('Q [mV]')
            ax.set_title(f'Event #{view.event_id[k]}')
            ax.grid(True)
        axs.flat[0].legend(loc='best')
        fig.colorbar(image, ax=axs, label='time [µs]', shrink=0.6)
    return fig


def plot_waveforms(view, make_figure=Figure):
    """pc2: I, Q (raw) and the projected waveform of each event.

    The red stars mark the I and Q values at the sample where the pedestal-subtracted
    |I + iQ| is largest (``proj max``): the same point as the red star in pc1.
    """
    with mpl.rc_context(_STYLE):
        fig, axs = _grid(view, make_figure)
        for k in range(view.n):
            ax = axs.flat[k]
            ax.plot(view.tbin_us, view.i[k] * 1e3, label='I')
            ax.plot(view.tbin_us, view.q[k] * 1e3, label='Q')
            ax.plot(view.tbin_us, view.proj[k] * 1e3, label='Proj')
            peak = view.peak_index[k]
            ax.plot(view.tbin_us[peak], view.i[k, peak] * 1e3, '*', ms=10, color='r',
                    label='proj max')
            ax.plot(view.tbin_us[peak], view.q[k, peak] * 1e3, '*', ms=10, color='r',
                    label='_nolegend_')
            if _is_bottom(k, view.n):
                ax.set_xlabel('time [µs]')
            if k % NCOLS == 0:
                ax.set_ylabel('voltage [mV]')
            ax.set_title(f'Event #{view.event_id[k]}')
            ax.grid(True)
        axs.flat[0].legend(loc='best')
    return fig
