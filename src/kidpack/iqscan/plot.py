"""Quick look at an IQ scan: ch0 vs ch1, frequency vs |S21|, frequency vs arg S21.

``kidpack-iqplot`` / ``python -m kidpack.iqscan.plot``. The data is only read.

Without a reference the values are those stored (ch0 + i ch1 in mV): proportional
to S21 but including the unknown gain and cable phase, i.e. uncalibrated. With
``--ref`` (a scan of the same frequencies through the reference path) S21 is
``(ch0 + i ch1) / (ref ch0 + i ref ch1)``, as ``plot_iq_scan.py`` does.
"""
import argparse
import glob
import os
import sys

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from kidpack.iqscan.config import default_scan_dir
from kidpack.iqscan.reader import IqScanError, load_iqscan

FIGSIZE = (15, 4.8)  # inches, fits a laptop screen
SAVE_DPI = 100

_STYLE = {
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'axes.formatter.useoffset': True, 'axes.unicode_minus': False,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'lines.linewidth': 1.4,
}


def s21(scan, ref=None):
    """Complex S21 of ``scan``: the raw ch0 + i ch1 [V], divided by ``ref`` if given."""
    if ref is None:
        return scan.iq
    if len(ref.frequency) != len(scan.frequency) or not np.allclose(
            ref.frequency, scan.frequency, rtol=1e-9, atol=0):
        raise IqScanError(f'the reference {ref.path} was not measured at the same frequencies '
                          f'as {scan.path}')
    return scan.iq / ref.iq


def plot_scan(scan, ref=None, db=False, wrap=False, make_figure=Figure):
    """Figure with the IQ plane, |S21| and arg S21 of ``scan``.

    ``db``: magnitude in dB (20 log10) instead of linear. ``wrap``: phase folded
    into (-pi, pi] instead of continuous. ``make_figure``: ``Figure`` (default) or
    ``matplotlib.pyplot.figure`` for a figure that can be shown in a window.
    """
    values = s21(scan, ref)
    freq_ghz = scan.frequency / 1e9
    magnitude = np.abs(values)
    phase = np.angle(values) if wrap else np.unwrap(np.angle(values))
    calibrated = ref is not None

    with mpl.rc_context(_STYLE):
        fig = make_figure(figsize=FIGSIZE, layout='constrained')
        ax_iq, ax_mag, ax_phase = fig.subplots(1, 3)

        scale, unit = (1.0, '') if calibrated else (1e3, ' [mV]')
        ax_iq.plot(values.real * scale, values.imag * scale, 'o-', ms=3)
        ax_iq.plot(values.real[0] * scale, values.imag[0] * scale, 'o', ms=7, color='r',
                   label='first point')
        ax_iq.set_xlabel(('Re S21' if calibrated else 'ch0 (I)') + unit)
        ax_iq.set_ylabel(('Im S21' if calibrated else 'ch1 (Q)') + unit)
        ax_iq.set_aspect('equal', adjustable='datalim')
        ax_iq.legend(loc='best')

        if db:
            ax_mag.plot(freq_ghz, 20 * np.log10(magnitude), 'o-', ms=3)
            ax_mag.set_ylabel('|S21| [dB]' if calibrated else '20 log10 |IQ| [dB re 1 V]')
        else:
            ax_mag.plot(freq_ghz, magnitude * scale, 'o-', ms=3)
            ax_mag.set_ylabel('|S21|' if calibrated else '|S21| (uncalibrated)' + unit)
        ax_phase.plot(freq_ghz, phase, 'o-', ms=3)
        ax_phase.set_ylabel('arg S21 [rad]' + ('' if calibrated else ' (uncalibrated)'))
        for ax in (ax_mag, ax_phase):
            ax.set_xlabel('Frequency [GHz]')
        for ax in (ax_iq, ax_mag, ax_phase):
            ax.grid(True)

        title = f'{scan.name}: {len(freq_ghz)} points, {freq_ghz[0]:.4f}-{freq_ghz[-1]:.4f} GHz'
        if scan.power_dbm is not None:
            title += f', {scan.power_dbm:g} dBm'
        title += f' / ref {ref.name}' if calibrated else ' (uncalibrated: S21 x gain x cable phase)'
        fig.suptitle(title)
    return fig


def find_latest_scan(data_dir):
    """Newest ``*.npz`` directly under ``data_dir`` (None if there is none)."""
    files = glob.glob(os.path.join(data_dir, '*.npz'))
    return max(files, key=lambda p: (os.path.getmtime(p), p)) if files else None


def build_parser():
    p = argparse.ArgumentParser(
        prog='kidpack-iqplot',
        description='Quick look at an IQ scan: ch0 vs ch1, frequency vs |S21|, frequency vs '
                    'arg S21, shown in a window (close it to exit). The data is only read.')
    p.add_argument('file', nargs='?',
                   help='IQ-scan .npz file (default: the newest one under --data-dir)')
    p.add_argument('--data-dir', default=default_scan_dir(),
                   help='where to look for the newest file (default: %s)'
                        % default_scan_dir().replace('%', '%%'))
    p.add_argument('--ref', metavar='FILE',
                   help='reference scan at the same frequencies: S21 = scan / ref (as '
                        'plot_iq_scan.py); without it the stored values are shown uncalibrated')
    p.add_argument('--db', action='store_true', help='magnitude in dB instead of linear')
    p.add_argument('--wrap', action='store_true',
                   help='fold the phase into (-pi, pi] instead of showing it continuously')
    p.add_argument('--save', metavar='PNG', help='also write the figure to this PNG file')
    p.add_argument('--no-show', action='store_true',
                   help='do not open a window (use with --save, e.g. without a display)')
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_show and not args.save:
        parser.error('--no-show without --save would do nothing')

    path = args.file or find_latest_scan(args.data_dir)
    if path is None:
        print(f'error: no IQ-scan file found under {args.data_dir}', file=sys.stderr)
        return 1
    try:
        scan = load_iqscan(path)
        ref = load_iqscan(args.ref) if args.ref else None
        s21(scan, ref)  # fail early on mismatching frequencies
    except (OSError, IqScanError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 1

    show = not args.no_show
    if show:
        import matplotlib.pyplot as plt  # only when a window is wanted
        make_figure = plt.figure
    else:
        make_figure = Figure
    fig = plot_scan(scan, ref, db=args.db, wrap=args.wrap, make_figure=make_figure)
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fig.savefig(args.save, dpi=SAVE_DPI)
        print(f'{path} -> {args.save}')
    else:
        print(path)

    if show:
        try:
            plt.show()  # blocks until the window is closed
        finally:
            plt.close('all')
    return 0


if __name__ == '__main__':
    sys.exit(main())
