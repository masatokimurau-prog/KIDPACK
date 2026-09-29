"""Command-line entry point: ``kidpack-monitor`` / ``python -m kidpack.monitor``."""
import argparse
import glob
import os
import sys

from kidpack.daq.config import default_output_dir
from kidpack.monitor.pulses import (N_EVENTS, make_pulse_view, plot_iq_plane,
                                    plot_waveforms, select_events)
from kidpack.rawdata import RawDataError, load_raw


def find_latest_raw_file(data_dir):
    """Newest complete raw file under ``data_dir/run_*/data/`` (None if there is none).

    The DAQ writes files atomically, so a half-written file is never matched.
    """
    files = glob.glob(os.path.join(data_dir, 'run_*', 'data', 'run*-*.npz'))
    return max(files, key=lambda p: (os.path.getmtime(p), p)) if files else None


def build_parser():
    p = argparse.ArgumentParser(
        prog='kidpack-monitor',
        description=f'Online check of a raw NPZ file: pc1.png (IQ plane) and pc2.png '
                    f'(waveforms) of {N_EVENTS} events in a 4x4 grid. The data is only read.')
    p.add_argument('file', nargs='?',
                   help='raw .npz file (default: the newest one under --data-dir)')
    p.add_argument('--data-dir', default=default_output_dir(),
                   help="where to look for the newest file (default: %s)"
                        % default_output_dir().replace('%', '%%'))
    p.add_argument('--output-dir', default='.',
                   help='where pc1.png and pc2.png are written (default: current directory)')
    p.add_argument('--rebin', type=int, default=1, metavar='N',
                   help='average N consecutive samples before locating the peak (default: 1)')
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

    os.makedirs(args.output_dir, exist_ok=True)
    for name, make_figure in (('pc1.png', plot_iq_plane), ('pc2.png', plot_waveforms)):
        make_figure(view).savefig(os.path.join(args.output_dir, name))
    print(f'{path}: events {view.event_id[0]}..{view.event_id[-1]} ({view.n}) '
          f'-> {os.path.join(args.output_dir, "pc1.png")}, pc2.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
