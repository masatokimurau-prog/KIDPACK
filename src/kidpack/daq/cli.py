"""Command-line entry point: ``kidpack-daq`` / ``python -m kidpack.daq``."""
import sys

from kidpack.daq.backends import create_scope, create_sg
from kidpack.daq.config import build_parser, config_from_args
from kidpack.daq.runner import run_daq
from kidpack.daq.writer import RunWriter


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args, parser)
    command_line = sys.argv if argv is None else ['kidpack-daq', *argv]

    writer = RunWriter(cfg, command_line=command_line)
    try:
        writer.check_new()  # before any device is opened
        scope = create_scope(cfg)
        try:
            sg = create_sg(cfg)
        except BaseException:
            scope.close()
            raise
    except FileExistsError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    except Exception as e:  # driver missing, device not found, ...
        print(f'error: could not set up the {cfg.backend} backend: {type(e).__name__}: {e}',
              file=sys.stderr)
        return 1

    result = run_daq(cfg, scope, sg, writer)
    return result.exit_code


if __name__ == '__main__':
    sys.exit(main())
