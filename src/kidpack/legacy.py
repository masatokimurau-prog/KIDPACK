"""Convert files of the old DAQ macro (kid.py) to the current raw-data format.

Old file (wf_YYMMDD_HHMMSS_xxHz.npz)       current file (run_XX/data/runXX-YY.npz)
    ch0, ch1  float32 (n, npts)       ->     ch0, ch1                 unchanged
    deltat    timedelta per event     ->     timestamp_unix_ns        start + deltat [ns]
    (file name: start of the DAQ)     ->     run_start_unix_ns        start [ns]
    -                                 ->     event_id                 0 ... n-1
    npts, sample_rate                 ->     npts, sample_rate        unchanged
    ref_position (int32)              ->     ref_position             float64
    daq_rate                          ->     only in the YAML and the run summary

Each old file becomes one run (file number 00), numbered in the order of the start
times. Two things are not in the old files and have to be assumed:

* The start of the DAQ is only known from the file name, to the second (kid.py cut
  the fraction off). The times of the events relative to each other are exact (deltat
  has microseconds); their absolute value is good to +-1 s.
* The file name is the local time of the DAQ PC: UTC+9 (Japan) unless told otherwise.

``timestamp_unix_ns`` of the old files is the PC time right after the waveform was
fetched, which is also what the current DAQ records for an edge trigger.

The old files hold a pickled array (``deltat``): only convert files you trust.
"""
import os
import re
from calendar import timegm
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import yaml

from kidpack import __version__
from kidpack.daq.summary import append_run_summary
from kidpack.daq.timeutil import iso_utc
from kidpack.daq.writer import atomic_write, raw_arrays
from kidpack.runs import file_stem, next_run_number, run_dir_name

JST = timezone(timedelta(hours=9))
SUMMARY_NAME = 'run_summary.txt'

_FILE_NAME = re.compile(r'wf_(\d{6})_(\d{6})_')
_OLD_KEYS = ('ch0', 'ch1', 'npts', 'sample_rate', 'ref_position', 'deltat')


class LegacyFileError(ValueError):
    """The file is not a file of the old DAQ macro."""


@dataclass
class OldRun:
    path: str
    start_ns: int  # start of the DAQ (from the file name)
    ch0: np.ndarray
    ch1: np.ndarray
    timestamp_unix_ns: np.ndarray  # start + deltat
    npts: int
    sample_rate: float
    ref_position: float
    daq_rate_hz: float


@dataclass
class ConvertedRun:
    run_number: int
    run_dir: str
    npz_path: str
    source: str
    start_ns: int
    stop_ns: int
    nevents: int
    condition: str


def start_ns_from_name(path, tz=JST):
    """Unix time [ns] of the YYMMDD_HHMMSS in a file name wf_YYMMDD_HHMMSS_xxHz.npz, in ``tz``."""
    match = _FILE_NAME.search(os.path.basename(str(path)))
    if match is None:
        raise LegacyFileError(f'{path}: the file name does not contain wf_YYMMDD_HHMMSS_ (start of the DAQ)')
    naive = datetime.strptime(match.group(1) + match.group(2), '%y%m%d%H%M%S')
    return timegm(naive.replace(tzinfo=tz).utctimetuple()) * 10 ** 9  # whole seconds: exact


def _timedelta_ns(delta):
    return (delta.days * 86400 + delta.seconds) * 10 ** 9 + delta.microseconds * 1000


def load_old_file(path, tz=JST):
    """Read one file of the old DAQ macro (never modifies it)."""
    path = str(path)
    start_ns = start_ns_from_name(path, tz)
    with np.load(path, allow_pickle=True) as d:  # deltat is a pickled array of timedeltas
        missing = [key for key in _OLD_KEYS if key not in d.files]
        if missing:
            raise LegacyFileError(f'{path}: missing key(s) {missing}: not a file of the old DAQ macro')
        ch0, ch1 = d['ch0'], d['ch1']
        deltat = list(d['deltat'])
        npts, sample_rate, ref_position = int(d['npts']), float(d['sample_rate']), float(d['ref_position'])
        daq_rate = float(d['daq_rate']) if 'daq_rate' in d.files else None

    if ch0.ndim != 2 or ch0.shape != ch1.shape or ch0.shape[1] != npts or len(deltat) != len(ch0):
        raise LegacyFileError(f'{path}: inconsistent shapes (ch0 {ch0.shape}, ch1 {ch1.shape}, '
                              f'npts {npts}, {len(deltat)} times)')
    try:
        offsets = np.array([_timedelta_ns(delta) for delta in deltat], dtype=np.int64)
    except AttributeError as e:
        raise LegacyFileError(f'{path}: deltat does not hold timedeltas') from e
    if np.any(np.diff(offsets) < 0) or offsets[0] < 0:
        raise LegacyFileError(f'{path}: the event times in deltat are not increasing')
    if daq_rate is None:  # as kid.py computed it: events per second of the whole acquisition
        daq_rate = len(deltat) / (offsets[-1] / 1e9)
    return OldRun(path, start_ns, ch0, ch1, start_ns + offsets, npts, sample_rate, ref_position, daq_rate)


def convert_old_files(items, output_dir, tz=JST, first_run_number=None, source_root=None):
    """Write each old file as a run of ``output_dir``; returns the ConvertedRun list.

    items: (path of an old file, condition text) pairs. Runs are numbered in the order of
    the start times, from ``first_run_number`` (default: the next free number). Existing
    runs are never overwritten. ``output_dir`` also gets the run summary.
    """
    runs = sorted(((load_old_file(path, tz), condition) for path, condition in items),
                  key=lambda pair: pair[0].start_ns)
    summary = os.path.join(output_dir, SUMMARY_NAME)
    number = first_run_number if first_run_number is not None else next_run_number(output_dir, summary)
    for offset in range(len(runs)):  # check everything before writing anything
        directory = os.path.join(output_dir, run_dir_name(number + offset))
        if os.path.exists(directory):
            raise FileExistsError(f'run directory already exists: {directory}')

    converted = []
    for offset, (old, condition) in enumerate(runs):
        converted.append(_write_run(old, condition, number + offset, output_dir, summary, source_root))
    return converted


def _write_run(old, condition, run_number, output_dir, summary, source_root):
    run_dir = os.path.join(output_dir, run_dir_name(run_number))
    os.makedirs(os.path.join(run_dir, 'data'))
    os.makedirs(os.path.join(run_dir, 'config'))
    stem = file_stem(run_number, 0)
    npz_path = os.path.join(run_dir, 'data', stem + '.npz')

    arrays = raw_arrays(old.ch0, old.ch1, old.timestamp_unix_ns, old.npts, old.sample_rate,
                        old.ref_position, old.start_ns)
    atomic_write(npz_path, lambda f: np.savez(f, **arrays), 'wb')

    n = len(old.ch0)
    stop_ns = int(old.timestamp_unix_ns[-1])
    source = os.path.relpath(old.path, source_root) if source_root else os.path.basename(old.path)
    document = {
        'run_number': run_number,
        'file_number': 0,
        'events_per_file': n,
        'nevents': n,
        'start_time_utc': iso_utc(old.start_ns),
        'stop_time_utc': iso_utc(stop_ns),
        'run_start_time_utc': iso_utc(old.start_ns),
        'daq_rate_hz': old.daq_rate_hz,
        'condition': condition,
        'daq_software_version': f'old DAQ macro (kid.py), converted by kidpack {__version__}',
        'git_hash': None,
        'npts': old.npts,
        'sample_rate': old.sample_rate,
        'ref_position': old.ref_position,
        'trigger': 'unknown (taken with the old DAQ macro)',
        'channels': 'unknown (taken with the old DAQ macro)',
        'converted_from': {'file': source,
                           'keys': 'ch0, ch1, npts, sample_rate, ref_position, daq_rate, deltat'},
        'timestamp_unix_ns_source': 'PC time (datetime.now()) right after the waveform was fetched, '
                                    'as start + deltat of the old file',
        'device_timestamp': 'not recorded',
        'status': 'complete',
        'errors': [],
        'warnings': [
            'converted from the old DAQ format',
            'the start of the DAQ is only known from the file name (to the second, taken as UTC+9): '
            'absolute times are good to +-1 s, times relative to each other are exact',
            'the trigger and channel settings of the old DAQ were not recorded',
        ],
    }
    atomic_write(os.path.join(run_dir, 'config', stem + '.yaml'),
                 lambda f: yaml.safe_dump(document, f, sort_keys=False, allow_unicode=True), 'w')
    append_run_summary(summary, run_number, old.start_ns, stop_ns, old.daq_rate_hz, condition)
    return ConvertedRun(run_number, run_dir, npz_path, source, old.start_ns, stop_ns, n, condition)
