"""Reader for the raw NPZ files written by the DAQ (see the I/O format specification).

Read-only: files are never modified. Files from the old ``kid.py`` (no
``event_id`` / ``timestamp_unix_ns``) can be read too.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

REQUIRED_KEYS = ('ch0', 'ch1', 'npts', 'sample_rate', 'ref_position')


class RawDataError(ValueError):
    """The file does not follow the raw-data contract."""


@dataclass
class RawData:
    path: str
    ch0: np.ndarray  # (nevents, npts) float32, volts
    ch1: np.ndarray  # (nevents, npts) float32, volts
    event_id: np.ndarray  # (nevents,)
    npts: int
    sample_rate: float  # Hz
    ref_position: float  # percent of the record (trigger position)
    timestamp_unix_ns: Optional[np.ndarray] = None  # (nevents,) int64, None in old files

    @property
    def nevents(self):
        return self.ch0.shape[0]


def time_axis_s(npts, sample_rate, ref_position):
    """Per-sample time [s] relative to the trigger (analysis convention of the spec)."""
    return (np.arange(npts) - npts * ref_position / 100) / sample_rate


def load_raw(path):
    """Load one raw NPZ file; raise RawDataError if the contract is not met."""
    path = str(path)
    with np.load(path) as d:
        missing = [key for key in REQUIRED_KEYS if key not in d.files]
        if missing:
            raise RawDataError(f'{path}: missing key(s) {missing}')
        ch0, ch1 = d['ch0'], d['ch1']
        npts = int(d['npts'])
        sample_rate = float(d['sample_rate'])
        ref_position = float(d['ref_position'])
        event_id = d['event_id'] if 'event_id' in d.files else None
        timestamps = d['timestamp_unix_ns'] if 'timestamp_unix_ns' in d.files else None

    if ch0.ndim != 2 or ch0.shape != ch1.shape:
        raise RawDataError(f'{path}: ch0/ch1 must be 2-D arrays of the same shape, '
                           f'got {ch0.shape} and {ch1.shape}')
    if ch0.shape[1] != npts:
        raise RawDataError(f'{path}: npts={npts} but the waveforms have {ch0.shape[1]} samples')
    if event_id is None:
        event_id = np.arange(ch0.shape[0])
    return RawData(path, ch0, ch1, event_id, npts, sample_rate, ref_position, timestamps)
