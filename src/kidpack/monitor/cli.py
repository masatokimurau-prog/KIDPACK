"""Command-line entry point: ``kidpack-monitor`` / ``python -m kidpack.monitor``."""
import argparse
import glob
import os
import sys

from matplotlib.figure import Figure

from kidpack.daq.config import default_output_dir
from kidpack.monitor.pulses import (DEFAULT_REBIN, N_EVENTS, SAVE_DPI, SHOW_FIGSIZE,
                                    make_pulse_view, plot_iq_plane, plot_waveforms,
                                    select_events)
from kidpack.rawdata import RawDataError, load_raw
from kidpack.runs import file_number_arg, find_run_file, run_label_arg


def find_latest_raw_file(data_dir):
    """Newest complete raw file under ``data_dir/run_*/data/`` (None if there is none).

    The DAQ writes files atomically, so a half-written file is never matched.
    """
    files = glob.glob(os.path.join(data_dir, 'run_*', 'data', 'run*-*.npz'))
    return max(files, key=lambda p: (os.path.getmtime(p), p)) if files else None


def build_parser():
    p = argparse.ArgumentParser(
        prog='kidpack-monitor',
        description=f'Online check of a raw NPZ file: pc1 (IQ plane) and pc2 (waveforms) of '
                    f'{N_EVENTS} events in a 4x4 grid, saved as pc1.png / pc2.png and shown in '
                    f'windows (close them to exit). The data is only read.')
    p.add_argument('file', nargs='?',
                   help='raw .npz file (default: the newest one under --data-dir)')
    p.add_argument('--run-number', type=run_label_arg, default=None, metavar='N|test|MMDD_HHMMSS',
                   help='instead of FILE: the file of this run under --data-dir (run_XX/data/runXX-YY.npz); '
                        'with --file-number that file, otherwise the one with the highest file number. '
                        'Besides a number and "test", a run can be named after its start (0831_155251, '
                        'as made by kidpack.legacy from files of the old DAQ macro)')
    p.add_argument('--file-number', type=file_number_arg, default=None, metavar='M',
                   help='file number YY within the run given by --run-number')
    p.add_argument('--data-dir', default=default_output_dir(),
                   help="where to look for the newest file (default: %s)"
                        % default_output_dir().replace('%', '%%'))
    p.add_argument('--output-dir', default='.',
                   help='where pc1.png and pc2.png are written (default: current directory)')
    p.add_argument('--rebin', type=int, default=DEFAULT_REBIN, metavar='N',
                   help='average N consecutive samples before locating the peak; the points '
                        'in pc1 are these averages (default: %(default)s)')
    p.add_argument('--no-show', action='store_true',
                   help='only write the PNG files, do not open windows (e.g. on a machine '
                        'without a display)')
    p.add_argument('--alpha', action='store_true',
                   help='pedestal from the first 100 ns instead of the pre-trigger region')
    p.add_argument('--stride', type=int, default=1, metavar='K',
                   help='plot every K-th event, starting from the first (default: 1; '
                        'the old macro used 2)')
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rebin < 1 or args.stride < 1:
        parser.error('--rebin and --stride must be >= 1')

    if args.file and (args.run_number is not None or args.file_number is not None):
        parser.error('give either FILE or --run-number / --file-number, not both')
    if args.file_number is not None and args.run_number is None:
        parser.error('--file-number needs --run-number')

    if args.run_number is not None:
        try:
            path = find_run_file(args.data_dir, args.run_number, args.file_number)
        except FileNotFoundError as e:
            print(f'error: {e}', file=sys.stderr)
            return 1
    else:
        path = args.file or find_latest_raw_file(args.data_dir)
    if path is None:
        print(f'error: no raw file found under {args.data_dir}', file=sys.stderr)
        return 1
    try:
        raw = load_raw(path)
        indices = select_events(raw.nevents, args.stride)
        view = make_pulse_view(raw, indices, rebin_factor=args.rebin, alpha=args.alpha)
    except (OSError, RawDataError, ValueError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 1

    show = not args.no_show
    if show:
        import matplotlib.pyplot as plt  # only when windows are wanted
        make_figure = plt.figure
    else:
        make_figure = Figure

    os.makedirs(args.output_dir, exist_ok=True)
    figures = []
    for name, plot in (('pc1.png', plot_iq_plane), ('pc2.png', plot_waveforms)):
        fig = plot(view, make_figure)
        fig.savefig(os.path.join(args.output_dir, name), dpi=SAVE_DPI)
        figures.append(fig)
    print(f'{path}: events {view.event_id[0]}..{view.event_id[-1]} ({view.n}) '
          f'-> {os.path.join(args.output_dir, "pc1.png")}, pc2.png')

    if show:
        try:
            for fig in figures:
                fig.set_size_inches(*SHOW_FIGSIZE)  # fit the screen; the PNGs are already saved
            plt.show()  # blocks until the windows are closed
        finally:
            plt.close('all')
    return 0


if __name__ == '__main__':
    sys.exit(main())
