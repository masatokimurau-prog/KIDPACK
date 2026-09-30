"""IQ-scan configuration and command-line parsing.

The defaults are the values of the macro ``iq_scan_kimura20260703.py``, which is
known to work; the frequency range has no default (it depends on the resonator)
and is required.
"""
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from kidpack.daq.config import (BACKENDS, DEFAULT_SG_RESOURCE, VERTICAL_COUPLINGS,
                                ChannelConfig, KidParser, default_output_dir)

DEFAULT_TIME_WINDOW = 1.0  # s
SUMMARY_FILE_NAME = 'iqscan_summary.txt'


def default_scan_dir():
    """``KIDPACK/data/iqscan`` (see daq.config.default_output_dir)."""
    return os.path.join(default_output_dir(), 'iqscan')


@dataclass
class IqScanConfig:
    # scan
    f_start: float  # Hz
    f_stop: float  # Hz
    num_points: int  # frequencies, both ends included
    power: float = -2.0  # dBm at the signal generator
    settle_time: float = 0.0  # s between starting the SG and acquiring
    # output
    output_dir: str = field(default_factory=default_scan_dir)
    name: Optional[str] = None  # file stem; default iqscan_YYYYMMDDHHMM
    summary_file: Optional[str] = None
    condition: str = ''
    backend: str = 'niscope'
    # instruments
    resource: str = 'PXI2Slot2'  # digitizer
    sg_resource: str = DEFAULT_SG_RESOURCE
    sample_rate: float = 1e4  # Hz
    npts: int = 10000  # samples per record (1 s at 10 kS/s)
    num_records: int = 2
    ref_position: float = 50.0  # %
    impedance: float = 1e6  # ohm
    fetch_timeout: float = 17.0  # s
    ch0: ChannelConfig = field(default_factory=lambda: ChannelConfig(vertical_range=1.0))
    ch1: ChannelConfig = field(default_factory=lambda: ChannelConfig(vertical_range=1.0))

    @property
    def summary_path(self):
        return self.summary_file or os.path.join(self.output_dir, SUMMARY_FILE_NAME)

    def frequencies(self):
        return np.linspace(self.f_start, self.f_stop, self.num_points)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        """Raise ValueError describing the first invalid setting."""
        def need(ok, message):
            if not ok:
                raise ValueError(message)

        need(self.f_start > 0 and self.f_stop > 0, '--f-start and --f-stop must be > 0')
        need(self.num_points >= 1, '--num-points must be >= 1')
        need(np.isfinite(self.power), '--power must be a finite number')
        need(self.settle_time >= 0, '--settle-time must be >= 0')
        need(self.name is None or (self.name and os.path.basename(self.name) == self.name
                                   and self.name not in ('.', '..')),
             '--name must be a plain file name without directories')
        need(self.backend in BACKENDS, f'--backend must be one of {BACKENDS}')
        need(self.sample_rate > 0, '--sample-rate must be > 0')
        need(self.npts >= 1, 'number of points per record must be >= 1')
        need(self.num_records >= 1, '--num-records must be >= 1')
        need(0 <= self.ref_position <= 100, '--ref-position must be within 0-100 %')
        need(self.impedance > 0, '--impedance must be > 0')
        need(self.fetch_timeout > 0, '--fetch-timeout must be > 0')
        for name, ch in (('ch0', self.ch0), ('ch1', self.ch1)):
            need(ch.vertical_range > 0, f'--{name}-range must be > 0')
            need(ch.coupling in VERTICAL_COUPLINGS,
                 f'--{name}-coupling must be one of {VERTICAL_COUPLINGS}')


_D = IqScanConfig(f_start=1.0, f_stop=1.0, num_points=1)  # single source of the defaults


def build_parser():
    p = KidParser(
        prog='kidpack-iqscan',
        description='Frequency scan of the transmission (IQ scan): for each frequency the signal '
                    'generator is set and started, the digitizer acquires, and the mean I (ch0) '
                    'and Q (ch1) are stored '
                    '(exit code: 0 completed, 130 stopped with Ctrl-C, 1 error).')

    g = p.add_argument_group('scan')
    g.add_argument('--f-start', type=float, required=True, metavar='HZ', help='first frequency [Hz]')
    g.add_argument('--f-stop', type=float, required=True, metavar='HZ', help='last frequency [Hz]')
    g.add_argument('--num-points', type=int, required=True,
                   help='number of frequencies, evenly spaced, both ends included')
    g.add_argument('--power', type=float, default=_D.power, metavar='DBM',
                   help='signal generator power [dBm] (default: %(default)g)')
    g.add_argument('--settle-time', type=float, default=_D.settle_time, metavar='S',
                   help='wait between starting the generator and acquiring, per point '
                        '(default: %(default)g)')

    g = p.add_argument_group('output')
    g.add_argument('--condition', default=_D.condition,
                   help='free-text measurement condition (temperature, ...), recorded in the '
                        'summary and the YAML')
    g.add_argument('--output-dir', default=_D.output_dir,
                   help=f"directory of the scan files (default: {_D.output_dir.replace('%', '%%')})")
    g.add_argument('--name', default=None, metavar='STEM',
                   help='file name without extension (default: iqscan_YYYYMMDDHHMM); an '
                        'existing file is never overwritten')
    g.add_argument('--summary-file', default=None, metavar='PATH',
                   help=f'scan summary text file, one line per scan '
                        f'(default: OUTPUT_DIR/{SUMMARY_FILE_NAME})')
    g.add_argument('--backend', choices=BACKENDS, default=_D.backend,
                   help="'simulator' needs no hardware (for testing)")

    g = p.add_argument_group('digitizer')
    g.add_argument('--resource', default=_D.resource, help='NI-SCOPE resource name')
    g.add_argument('--sample-rate', type=float, default=_D.sample_rate, metavar='HZ',
                   help='minimum sample rate [Hz] (default: %(default)g)')
    size = g.add_mutually_exclusive_group()
    size.add_argument('--time-window', type=float, default=None, metavar='S',
                      help=f'record length [s] (default: {DEFAULT_TIME_WINDOW:g}); '
                           'npts = sample-rate x time-window')
    size.add_argument('--npts', type=int, default=None,
                      help='samples per record (alternative to --time-window)')
    g.add_argument('--num-records', type=int, default=_D.num_records,
                   help='records acquired and averaged per frequency (default: %(default)s)')
    g.add_argument('--ref-position', type=float, default=_D.ref_position, metavar='PERCENT',
                   help='reference position in the record [%%] (default: %(default)g)')
    for i in (0, 1):
        ch = getattr(_D, f'ch{i}')
        g.add_argument(f'--ch{i}-range', type=float, default=ch.vertical_range, metavar='V',
                       help=f'ch{i} vertical range [V] (default: %(default)g)')
        g.add_argument(f'--ch{i}-coupling', choices=VERTICAL_COUPLINGS, default=ch.coupling,
                       help=f'ch{i} coupling (default: %(default)s)')
        g.add_argument(f'--ch{i}-offset', type=float, default=ch.offset, metavar='V',
                       help=f'ch{i} vertical offset [V] (default: %(default)g)')
    g.add_argument('--impedance', type=float, default=_D.impedance, metavar='OHM',
                   help='input impedance of both channels (default: %(default)g)')
    g.add_argument('--fetch-timeout', type=float, default=_D.fetch_timeout, metavar='S',
                   help='max wait for the acquisition of one point [s] (default: %(default)g)')

    g = p.add_argument_group('signal generator')
    g.add_argument('--sg-resource', default=_D.sg_resource,
                   help='signal generator resource name (default: %(default)s)')
    return p


def config_from_args(args, parser=None):
    """Build a validated IqScanConfig from parsed arguments (exits via parser.error)."""
    parser = parser or build_parser()
    if args.npts is not None:
        npts = args.npts
    else:
        window = DEFAULT_TIME_WINDOW if args.time_window is None else args.time_window
        npts = int(round(args.sample_rate * window))

    cfg = IqScanConfig(
        f_start=args.f_start, f_stop=args.f_stop, num_points=args.num_points,
        power=args.power, settle_time=args.settle_time,
        output_dir=args.output_dir, name=args.name, summary_file=args.summary_file,
        condition=args.condition, backend=args.backend,
        resource=args.resource, sg_resource=args.sg_resource,
        sample_rate=args.sample_rate, npts=npts, num_records=args.num_records,
        ref_position=args.ref_position, impedance=args.impedance,
        fetch_timeout=args.fetch_timeout,
        ch0=ChannelConfig(args.ch0_range, args.ch0_coupling, args.ch0_offset),
        ch1=ChannelConfig(args.ch1_range, args.ch1_coupling, args.ch1_offset),
    )
    try:
        cfg.validate()
    except ValueError as e:
        parser.error(str(e))
    return cfg
