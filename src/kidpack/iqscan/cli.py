"""Command-line entry point: ``kidpack-iqscan`` / ``python -m kidpack.iqscan``."""
import sys

from kidpack.iqscan.backends import create_backends
from kidpack.iqscan.config import build_parser, config_from_args
from kidpack.iqscan.runner import run_iqscan
from kidpack.iqscan.writer import IqScanWriter


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args, parser)
    command_line = sys.argv if argv is None else ['kidpack-iqscan', *argv]

    writer = IqScanWriter(cfg, command_line=command_line)
    try:
        writer.check_new()  # before any instrument is opened
        scope, sg = create_backends(cfg)
    except FileExistsError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    except Exception as e:  # driver missing, device not found, ...
        print(f'error: could not set up the {cfg.backend} backend: {type(e).__name__}: {e}',
              file=sys.stderr)
        return 1

    return run_iqscan(cfg, scope, sg, writer).exit_code


if __name__ == '__main__':
    sys.exit(main())
