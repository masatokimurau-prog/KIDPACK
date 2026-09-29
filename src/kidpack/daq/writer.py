"""On-disk layout and file formats (see the I/O format specification).

    run_XX/
      data/runXX-YY.npz      raw waveforms + minimal timing/identity metadata
      config/runXX-YY.yaml   settings, times and status of that file

No log file is written; progress and errors go to the console (errors are
also recorded in the YAML of the affected file).
"""
import os

import numpy as np
import yaml

from kidpack.daq.timeutil import iso_utc
from kidpack.daq.version import software_version


class RunWriter:
    def __init__(self, cfg, timestamp_source='unknown', command_line=None):
        self.cfg = cfg
        self.timestamp_source = timestamp_source
        self.command_line = list(command_line) if command_line is not None else None
        self.run_start_ns = None  # set by the run control before the first write
        self.run_dir = os.path.join(cfg.output_dir, f'run_{cfg.run_number:02d}')
        self.data_dir = os.path.join(self.run_dir, 'data')
        self.config_dir = os.path.join(self.run_dir, 'config')
        self._version = software_version()

    def check_new(self):
        """Refuse to reuse a run number (raw data must never be overwritten)."""
        if os.path.exists(self.run_dir):
            raise FileExistsError(f'run directory already exists: {self.run_dir}')

    def create(self):
        self.check_new()
        for directory in (self.data_dir, self.config_dir):
            os.makedirs(directory)

    def stem(self, file_number):
        return f'run{self.cfg.run_number:02d}-{file_number:02d}'

    def write_file(self, file_number, events, *, file_start_ns, file_stop_ns,
                   daq_rate_hz, status, actual, errors=()):
        """Write runXX-YY.npz and runXX-YY.yaml; return the npz path."""
        cfg = self.cfg
        n = len(events)
        actual = actual or {}
        sample_rate = actual.get('sample_rate') or cfg.sample_rate
        ref_position = actual.get('ref_position') or cfg.ref_position

        arrays = dict(
            ch0=np.stack([e.ch0 for e in events]).astype(np.float32, copy=False),
            ch1=np.stack([e.ch1 for e in events]).astype(np.float32, copy=False),
            event_id=np.arange(n, dtype=np.uint32),
            timestamp_unix_ns=np.array([e.timestamp_unix_ns for e in events], dtype=np.int64),
            npts=np.int32(cfg.npts),
            sample_rate=np.float64(sample_rate),
            ref_position=np.float64(ref_position),
            run_start_unix_ns=np.int64(self.run_start_ns),
        )
        stem = self.stem(file_number)
        npz_path = os.path.join(self.data_dir, f'{stem}.npz')
        _atomic_write(npz_path, lambda f: np.savez(f, **arrays), 'wb')

        document = {
            'run_number': cfg.run_number,
            'file_number': file_number,
            'events_per_file': cfg.events_per_file,
            'nevents': n,
            'start_time_utc': iso_utc(file_start_ns),
            'stop_time_utc': iso_utc(file_stop_ns),
            'run_start_time_utc': iso_utc(self.run_start_ns),
            'daq_rate_hz': daq_rate_hz,
            'condition': cfg.condition,
            'daq_software_version': self._version['kidpack'],
            'git_hash': self._version['git'],
            'npts': cfg.npts,
            'sample_rate': float(sample_rate),
            'ref_position': float(ref_position),
            'trigger': _trigger(cfg.trigger),
            'channels': {
                'ch0': {**_channel(cfg.ch0), 'impedance': cfg.impedance, 'bandwidth': cfg.bandwidth},
                'ch1': {**_channel(cfg.ch1), 'impedance': cfg.impedance, 'bandwidth': cfg.bandwidth},
            },
            'actual_settings': actual,
            'backend': cfg.backend,
            'resource': cfg.resource,
            'fetch_timeout_s': cfg.fetch_timeout,
            'sg': ({'controlled_by_daq': True, 'frequency_hz': cfg.sg.frequency,
                    'power_dbm': cfg.sg.power, 'resource': cfg.sg.resource}
                   if cfg.sg else {'controlled_by_daq': False}),
            'timestamp_unix_ns_source': self.timestamp_source,
            'device_timestamp': 'not recorded (timestamp_device_* keys are not written)',
            'status': status,
            'errors': list(errors),
            'warnings': [],
            'command_line': self.command_line,
        }
        yaml_path = os.path.join(self.config_dir, f'{stem}.yaml')
        _atomic_write(yaml_path,
                      lambda f: yaml.safe_dump(document, f, sort_keys=False, allow_unicode=True),
                      'w')
        return npz_path


def _trigger(t):
    if t.mode == 'random':
        return {'mode': 'random', 'source': 'software', 'interval_s': t.interval}
    return {'mode': 'edge', 'source': t.source, 'level': t.level,
            'slope': t.slope, 'coupling': t.coupling}


def _channel(ch):
    return {'vertical_range': ch.vertical_range, 'coupling': ch.coupling, 'offset': ch.offset}


def _atomic_write(path, write, mode):
    """Write to a temporary file and rename, so a crash never leaves a half-written file."""
    tmp = path + '.tmp'
    kwargs = {} if 'b' in mode else {'encoding': 'utf-8'}
    with open(tmp, mode, **kwargs) as f:
        write(f)
    os.replace(tmp, path)
