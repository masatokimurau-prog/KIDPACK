"""NI-SCOPE (PXIe-5160) backend.

The driver calls, their order and their arguments follow the original
``kid.py``; only the values come from the configuration instead of being
hard-coded.
"""
import logging
import time

import numpy as np

from kidpack.daq.backends.base import DaqError, Event, ScopeBackend

log = logging.getLogger('kidpack.daq')


def _enum(enum_class, name):
    return getattr(enum_class, name.upper())


class NiScopeBackend(ScopeBackend):
    timestamp_source = ('host clock (time.time_ns) read immediately after fetch() returns; '
                        'approximates the trigger time, device timestamp not used')

    def __init__(self, cfg):
        import niscope  # imported here so that the package works without the driver
        self._niscope = niscope
        self._cfg = cfg
        self._session = niscope.Session(cfg.resource)

    def configure(self):
        cfg, s, ni = self._cfg, self._session, self._niscope
        channels = (cfg.ch0, cfg.ch1)

        for index, ch in enumerate(channels):
            s.channels[index].configure_vertical(
                range=ch.vertical_range, coupling=_enum(ni.VerticalCoupling, ch.coupling))
            s.channels[index].vertical_offset = ch.offset
        for index in range(len(channels)):
            s.channels[index].input_impedance = cfg.impedance
        for index in range(len(channels)):
            s.channels[index].max_input_frequency = cfg.bandwidth

        s.configure_horizontal_timing(
            min_sample_rate=cfg.sample_rate, min_num_pts=cfg.npts,
            ref_position=cfg.ref_position, num_records=1, enforce_realtime=True)

        s.configure_trigger_edge(
            trigger_source=cfg.trigger.source, level=cfg.trigger.level,
            trigger_coupling=_enum(ni.TriggerCoupling, cfg.trigger.coupling),
            slope=_enum(ni.TriggerSlope, cfg.trigger.slope))

        return self._read_back()

    def _read_back(self):
        """Values the driver actually applies (it may coerce the requested ones)."""
        s = self._session

        def read(obj, attribute):
            try:
                return float(getattr(obj, attribute))
            except Exception as e:  # metadata only: never abort a run because of it
                log.warning('could not read back %s: %s', attribute, e)
                return None

        actual = {'sample_rate': read(s, 'horz_sample_rate'),
                  'ref_position': read(s, 'horz_record_ref_position')}
        for index in (0, 1):
            ch = s.channels[index]
            actual[f'ch{index}'] = {name: read(ch, name) for name in
                                    ('vertical_range', 'vertical_offset',
                                     'input_impedance', 'max_input_frequency')}
        return actual

    def acquire(self):
        npts = self._cfg.npts
        with self._session.initiate():
            waveforms = self._session.channels[0, 1].fetch(timeout=self._cfg.fetch_timeout)
            timestamp_ns = time.time_ns()
        ch0, ch1 = waveforms[0].samples, waveforms[1].samples
        if len(ch0) < npts or len(ch1) < npts:
            raise DaqError(f'fetched {len(ch0)}/{len(ch1)} samples, expected at least {npts}')
        return Event(np.asarray(ch0[:npts], dtype=np.float32),
                     np.asarray(ch1[:npts], dtype=np.float32), timestamp_ns)

    def close(self):
        if self._session is not None:
            session, self._session = self._session, None
            session.close()
