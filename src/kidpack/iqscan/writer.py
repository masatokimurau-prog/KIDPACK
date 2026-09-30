"""Scan files: ``<name>.npz`` (data) and ``<name>.yaml`` (settings, times, status).

``dd`` has the layout of the old ``iq_scan.py`` output, so ``plot_iq_scan.py`` and
the scripts in ``analysis/`` read it unchanged:

    dd[:, 0]  frequency [Hz]
    dd[:, 1]  mean of ch0 (I) over all samples of all records [V]
    dd[:, 2]  mean of ch1 (Q) [V]

Extra keys (ignored by those scripts): ``ch0_std``/``ch1_std`` (sample standard
deviation), ``ch0_stderr``/``ch1_stderr`` (= std / sqrt(n_samples), assuming
independent samples), ``n_samples``, ``timestamp_unix_ns`` (per point),
``scan_start_unix_ns``, ``power_dbm``, ``sample_rate``, ``npts``, ``num_records``.
"""
import os
import time
from datetime import datetime

import numpy as np
import yaml

from kidpack.daq.timeutil import iso_utc
from kidpack.daq.version import software_version
from kidpack.daq.writer import atomic_write


class IqScanWriter:
    def __init__(self, cfg, timestamp_source='unknown', command_line=None):
        self.cfg = cfg
        self.timestamp_source = timestamp_source
        self.command_line = list(command_line) if command_line is not None else None
        stem = cfg.name or 'iqscan_' + datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
        self.stem = stem
        self.npz_path = os.path.join(cfg.output_dir, stem + '.npz')
        self.yaml_path = os.path.join(cfg.output_dir, stem + '.yaml')
        self._version = software_version()

    def check_new(self):
        """Refuse to overwrite an existing scan (raw data must never be overwritten)."""
        for path in (self.npz_path, self.yaml_path):
            if os.path.exists(path):
                raise FileExistsError(f'scan file already exists: {path}')

    def write(self, frequencies, points, *, start_ns, stop_ns, status, actual, errors=()):
        """Write the npz and the yaml of the ``len(points)`` measured frequencies."""
        cfg = self.cfg
        self.check_new()
        os.makedirs(cfg.output_dir, exist_ok=True)
        n = len(points)
        frequencies = np.asarray(frequencies, dtype=np.float64)[:n]
        actual = actual or {}
        sample_rate = actual.get('sample_rate') or cfg.sample_rate

        def column(name):
            return np.array([getattr(p, name) for p in points], dtype=np.float64)

        n_samples = np.array([p.n_samples for p in points], dtype=np.int64)
        arrays = dict(
            dd=np.column_stack([frequencies, column('mean0'), column('mean1')]),
            ch0_std=column('std0'), ch1_std=column('std1'),
            ch0_stderr=column('std0') / np.sqrt(n_samples),
            ch1_stderr=column('std1') / np.sqrt(n_samples),
            n_samples=n_samples,
            timestamp_unix_ns=np.array([p.timestamp_ns for p in points], dtype=np.int64),
            scan_start_unix_ns=np.int64(start_ns),
            power_dbm=np.float64(cfg.power),
            sample_rate=np.float64(sample_rate),
            npts=np.int32(cfg.npts),
            num_records=np.int32(cfg.num_records),
        )
        atomic_write(self.npz_path, lambda f: np.savez(f, **arrays), 'wb')

        document = {
            'name': self.stem,
            'status': status,
            'errors': list(errors),
            'start_time_utc': iso_utc(start_ns),
            'stop_time_utc': iso_utc(stop_ns),
            'condition': cfg.condition,
            'scan': {'f_start_hz': cfg.f_start, 'f_stop_hz': cfg.f_stop,
                     'num_points_requested': cfg.num_points, 'num_points_measured': n,
                     'power_dbm': cfg.power, 'settle_time_s': cfg.settle_time},
            'digitizer': {
                'resource': cfg.resource, 'sample_rate': float(sample_rate), 'npts': cfg.npts,
                'num_records': cfg.num_records, 'ref_position': cfg.ref_position,
                'trigger': 'immediate', 'impedance': cfg.impedance,
                'fetch_timeout_s': cfg.fetch_timeout,
                'channels': {name: {'vertical_range': ch.vertical_range, 'coupling': ch.coupling,
                                    'offset': ch.offset}
                             for name, ch in (('ch0', cfg.ch0), ('ch1', cfg.ch1))},
            },
            'signal_generator': {'resource': cfg.sg_resource},
            'actual_settings': actual,
            'backend': cfg.backend,
            'timestamp_unix_ns_source': self.timestamp_source,
            'daq_software_version': self._version['kidpack'],
            'git_hash': self._version['git'],
            'command_line': self.command_line,
        }
        atomic_write(self.yaml_path,
                     lambda f: yaml.safe_dump(document, f, sort_keys=False, allow_unicode=True),
                     'w')
        return self.npz_path
