"""DAQ configuration and command-line parsing.

The defaults reproduce the values that were hard-coded in the original
``kid.py``; every one of them can be overridden from the command line.
"""
import argparse
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

VERTICAL_COUPLINGS = ('dc', 'ac', 'gnd')
TRIGGER_COUPLINGS = ('lf_reject', 'hf_reject', 'dc', 'ac', 'ac_plus_hf_reject')
TRIGGER_SLOPES = ('positive', 'negative')
TRIGGER_MODES = ('edge', 'random')
BACKENDS = ('niscope', 'simulator')

DEFAULT_TIME_WINDOW = 2e-6  # s
DEFAULT_SG_RESOURCE = 'PXI1Slot3'
SUMMARY_FILE_NAME = 'run_summary.txt'


def default_output_dir():
    """``KIDPACK/data`` when running from a source checkout, else ``./data``.

    Anchoring to the checkout keeps new runs out of unrelated directories
    (e.g. an old ``KID/data``) whatever the current directory is.
    """
    root = Path(__file__).resolve().parents[3]
    if (root / 'pyproject.toml').is_file() and (root / 'src' / 'kidpack').is_dir():
        return str(root / 'data')
    return 'data'


@dataclass
class ChannelConfig:
    vertical_range: float = 0.01  # V
    coupling: str = 'dc'
    offset: float = 0.0  # V


@dataclass
class TriggerConfig:
    source: str = 'VAL_EXTERNAL'
    level: float = 2.2  # V
    slope: str = 'positive'
    coupling: str = 'lf_reject'
    # 'edge': wait for the trigger described by the fields above.
    # 'random': the DAQ itself issues a software trigger every `interval`
    # seconds (unbiased noise/baseline samples); the edge settings are unused.
    mode: str = 'edge'
    interval: float = 1.0  # s between random triggers


@dataclass
class SgConfig:
    frequency: float  # Hz
    power: float  # dBm
    resource: str = DEFAULT_SG_RESOURCE


@dataclass
class DaqConfig:
    # run control
    run_number: int
    events_per_file: int
    num_files: int
    output_dir: str = field(default_factory=default_output_dir)
    summary_file: Optional[str] = None
    condition: str = ''
    backend: str = 'niscope'
    # digitizer
    resource: str = 'PXI2Slot2'
    sample_rate: float = 2.5e9  # Hz (max 2.5 GS/s for PXIe-5160)
    npts: int = 5000
    ref_position: float = 20.0  # % of record
    impedance: float = 50.0  # ohm
    bandwidth: float = -1.0  # Hz, -1 = full bandwidth
    fetch_timeout: float = 100.0  # s
    ch0: ChannelConfig = field(default_factory=ChannelConfig)
    ch1: ChannelConfig = field(default_factory=ChannelConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    # signal generator (optional; None = not controlled by the DAQ)
    sg: Optional[SgConfig] = None

    @property
    def summary_path(self):
        return self.summary_file or os.path.join(self.output_dir, SUMMARY_FILE_NAME)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        """Raise ValueError describing the first invalid setting."""
        def need(ok, message):
            if not ok:
                raise ValueError(message)

        need(self.run_number >= 0, '--run-number must be >= 0')
        need(self.events_per_file >= 1, '--events-per-file must be >= 1')
        need(self.num_files >= 1, '--num-files must be >= 1')
        need(self.backend in BACKENDS, f'--backend must be one of {BACKENDS}')
        need(self.sample_rate > 0, '--sample-rate must be > 0')
        need(self.npts >= 1, 'number of points per record must be >= 1')
        need(0 <= self.ref_position <= 100, '--ref-position must be within 0-100 %')
        need(self.impedance > 0, '--impedance must be > 0')
        need(self.fetch_timeout > 0, '--fetch-timeout must be > 0')
        for name, ch in (('ch0', self.ch0), ('ch1', self.ch1)):
            need(ch.vertical_range > 0, f'--{name}-range must be > 0')
            need(ch.coupling in VERTICAL_COUPLINGS,
                 f'--{name}-coupling must be one of {VERTICAL_COUPLINGS}')
        need(self.trigger.mode in TRIGGER_MODES,
             f'trigger mode must be one of {TRIGGER_MODES}')
        need(self.trigger.interval > 0, '--random-trigger-interval must be > 0')
        need(self.trigger.slope in TRIGGER_SLOPES,
             f'--trigger-slope must be one of {TRIGGER_SLOPES}')
        need(self.trigger.coupling in TRIGGER_COUPLINGS,
             f'--trigger-coupling must be one of {TRIGGER_COUPLINGS}')


# Defaults for argparse are read from a throw-away instance so that they are
# defined in exactly one place (the dataclasses above).
_D = DaqConfig(run_number=0, events_per_file=1, num_files=1)


class _Parser(argparse.ArgumentParser):
    """argparse accepts -1 and -0.014 as values but mistakes -8e-5 for an option.

    Voltages such as ``--trigger-level -8e-5`` are common here, so a negative
    number in exponent notation that follows an option is joined to it
    (``--trigger-level=-8e-5``) before parsing.
    """
    _EXPONENT_NUMBER = re.compile(r'^-(\d+\.?\d*|\.\d+)[eE][+-]?\d+$')

    def parse_known_args(self, args=None, namespace=None):
        args = list(sys.argv[1:] if args is None else args)
        joined = []
        for token in args:
            if (joined and self._EXPONENT_NUMBER.match(token)
                    and joined[-1].startswith('--') and '=' not in joined[-1]):
                joined[-1] = f'{joined[-1]}={token}'
            else:
                joined.append(token)
        return super().parse_known_args(joined, namespace)


def build_parser():
    p = _Parser(
        prog='kidpack-daq',
        description='Acquire KID waveforms with an NI-SCOPE digitizer '
                    '(exit code: 0 completed, 130 stopped with Ctrl-C, 1 error).')

    g = p.add_argument_group('run control')
    g.add_argument('--run-number', type=int, required=True,
                   help='run number XX (output goes to run_XX/); refuses to reuse an existing one')
    g.add_argument('--events-per-file', type=int, required=True,
                   help='events (waveforms) per output file')
    g.add_argument('--num-files', type=int, required=True,
                   help='number of files to acquire; Ctrl-C stops earlier and keeps the data so far')
    g.add_argument('--condition', default=_D.condition,
                   help='free-text measurement condition (temperature, bias, ...), '
                        'recorded in the run summary and the YAML')
    g.add_argument('--output-dir', default=_D.output_dir,
                   help='directory in which run_XX/ is created '
                        f"(default: {_D.output_dir.replace('%', '%%')})")
    g.add_argument('--summary-file', default=None, metavar='PATH',
                   help=f'run summary text file, one line appended per run '
                        f'(default: OUTPUT_DIR/{SUMMARY_FILE_NAME})')
    g.add_argument('--backend', choices=BACKENDS, default=_D.backend,
                   help="'simulator' needs no hardware (for testing)")

    g = p.add_argument_group('digitizer / horizontal')
    g.add_argument('--resource', default=_D.resource, help='NI-SCOPE resource name')
    g.add_argument('--sample-rate', type=float, default=_D.sample_rate, metavar='HZ',
                   help='minimum sample rate [Hz] (default: %(default)g)')
    size = g.add_mutually_exclusive_group()
    size.add_argument('--time-window', type=float, default=None, metavar='S',
                      help=f'record length [s] (default: {DEFAULT_TIME_WINDOW:g}); '
                           'npts = sample-rate x time-window')
    size.add_argument('--npts', type=int, default=None,
                      help='samples per record (alternative to --time-window)')
    g.add_argument('--ref-position', type=float, default=_D.ref_position, metavar='PERCENT',
                   help='trigger position in the record [%%] (default: %(default)g)')
    g.add_argument('--fetch-timeout', type=float, default=_D.fetch_timeout, metavar='S',
                   help='max wait for one trigger [s]; Ctrl-C is only handled after this '
                        'wait returns (default: %(default)g)')

    g = p.add_argument_group('digitizer / vertical')
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
    g.add_argument('--bandwidth', type=float, default=_D.bandwidth, metavar='HZ',
                   help='max input frequency of both channels, -1 = full (default: %(default)g)')

    g = p.add_argument_group('trigger',
                             'Default: wait for an edge trigger (--trigger-source/level/slope/'
                             'coupling). With --random-trigger the DAQ instead issues a '
                             'software trigger itself, at a fixed interval.')
    g.add_argument('--random-trigger', action='store_true',
                   help='issue a software trigger every --random-trigger-interval seconds '
                        'instead of waiting for an edge trigger')
    g.add_argument('--random-trigger-interval', type=float, default=None, metavar='S',
                   help=f'seconds between random triggers (default: {_D.trigger.interval:g})')
    g.add_argument('--trigger-source', default=_D.trigger.source,
                   help="trigger source, e.g. VAL_EXTERNAL or a channel name (default: %(default)s)")
    g.add_argument('--trigger-level', type=float, default=_D.trigger.level, metavar='V',
                   help='trigger level [V] (default: %(default)g)')
    g.add_argument('--trigger-slope', choices=TRIGGER_SLOPES, default=_D.trigger.slope,
                   help='(default: %(default)s)')
    g.add_argument('--trigger-coupling', choices=TRIGGER_COUPLINGS, default=_D.trigger.coupling,
                   help='(default: %(default)s)')

    g = p.add_argument_group('signal generator (optional)',
                             'Give --sg-frequency and --sg-power to have the DAQ set and start '
                             'the SG for the run (stopped at the end). Without them the SG is '
                             'not touched and can be configured externally.')
    g.add_argument('--sg-frequency', type=float, default=None, metavar='HZ')
    g.add_argument('--sg-power', type=float, default=None, metavar='DBM')
    g.add_argument('--sg-resource', default=None,
                   help=f'SG resource name (default: {DEFAULT_SG_RESOURCE})')
    return p


def config_from_args(args, parser=None):
    """Build a validated DaqConfig from parsed arguments (exits via parser.error)."""
    parser = parser or build_parser()

    if args.npts is not None:
        npts = args.npts
    else:
        window = DEFAULT_TIME_WINDOW if args.time_window is None else args.time_window
        npts = int(round(args.sample_rate * window))

    edge_options = ((args.trigger_source, _D.trigger.source), (args.trigger_level, _D.trigger.level),
                    (args.trigger_slope, _D.trigger.slope),
                    (args.trigger_coupling, _D.trigger.coupling))
    if args.random_trigger:
        if any(value != default for value, default in edge_options):
            parser.error('--trigger-source/level/slope/coupling cannot be combined with '
                         '--random-trigger')
    elif args.random_trigger_interval is not None:
        parser.error('--random-trigger-interval requires --random-trigger')
    trigger_interval = (_D.trigger.interval if args.random_trigger_interval is None
                        else args.random_trigger_interval)

    sg = None
    if args.sg_frequency is not None or args.sg_power is not None:
        if args.sg_frequency is None or args.sg_power is None:
            parser.error('--sg-frequency and --sg-power must be given together')
        sg = SgConfig(frequency=args.sg_frequency, power=args.sg_power,
                      resource=args.sg_resource or DEFAULT_SG_RESOURCE)
    elif args.sg_resource is not None:
        parser.error('--sg-resource requires --sg-frequency and --sg-power')

    cfg = DaqConfig(
        run_number=args.run_number,
        events_per_file=args.events_per_file,
        num_files=args.num_files,
        output_dir=args.output_dir,
        summary_file=args.summary_file,
        condition=args.condition,
        backend=args.backend,
        resource=args.resource,
        sample_rate=args.sample_rate,
        npts=npts,
        ref_position=args.ref_position,
        impedance=args.impedance,
        bandwidth=args.bandwidth,
        fetch_timeout=args.fetch_timeout,
        ch0=ChannelConfig(args.ch0_range, args.ch0_coupling, args.ch0_offset),
        ch1=ChannelConfig(args.ch1_range, args.ch1_coupling, args.ch1_offset),
        trigger=TriggerConfig(
            mode='random' if args.random_trigger else 'edge',
            source=args.trigger_source, level=args.trigger_level,
            slope=args.trigger_slope, coupling=args.trigger_coupling,
            interval=trigger_interval),
        sg=sg,
    )
    try:
        cfg.validate()
    except ValueError as e:
        parser.error(str(e))
    return cfg
