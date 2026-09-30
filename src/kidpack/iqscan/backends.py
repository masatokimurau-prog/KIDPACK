"""Digitizer backends of the IQ scan (the SG backends are shared with the pulse DAQ).

The NI-SCOPE calls, their order and their arguments follow the seed
``iq_scan.py``: one ``initiate()`` per frequency acquires ``num_records`` records
of both channels immediately (no trigger), and the mean over all samples of all
records is the I (ch0) / Q (ch1) value of that frequency.
"""
import abc
import logging
import time
from dataclasses import dataclass

import numpy as np

from kidpack.daq.backends.base import DaqError, SgBackend

log = logging.getLogger('kidpack.iqscan')


@dataclass
class Measurement:
    """Mean and spread of ch0 (I) / ch1 (Q) over all samples of all records, in volts."""
    mean0: float
    mean1: float
    std0: float  # sample standard deviation
    std1: float
    n_samples: int  # samples per channel that entered the mean
    timestamp_ns: int  # time.time_ns() right after the acquisition returned


def summarize(records0, records1, timestamp_ns):
    """Statistics over the concatenated records of each channel."""
    x0 = np.concatenate([np.asarray(r, dtype=np.float64) for r in records0])
    x1 = np.concatenate([np.asarray(r, dtype=np.float64) for r in records1])
    ddof = 1 if x0.size > 1 else 0
    return Measurement(float(x0.mean()), float(x1.mean()),
                       float(x0.std(ddof=ddof)), float(x1.std(ddof=ddof)),
                       int(x0.size), timestamp_ns)


class IqScopeBackend(abc.ABC):
    #: how ``Measurement.timestamp_ns`` was obtained (recorded in the YAML)
    timestamp_source = 'host clock (time.time_ns) read immediately after fetch() returns'

    @abc.abstractmethod
    def configure(self):
        """Apply the configuration; return the settings the device actually uses (YAML-able)."""

    @abc.abstractmethod
    def measure(self):
        """Acquire the records at the current generator setting and return a Measurement."""

    def close(self):
        pass


def _enum(enum_class, name):
    return getattr(enum_class, name.upper())


class NiScopeIqBackend(IqScopeBackend):
    def __init__(self, cfg):
        import niscope  # imported here so that the package works without the driver
        self._niscope = niscope
        self._cfg = cfg
        self._session = niscope.Session(cfg.resource)

    def configure(self):
        cfg, s, ni = self._cfg, self._session, self._niscope
        s.configure_horizontal_timing(
            min_sample_rate=cfg.sample_rate, min_num_pts=cfg.npts,
            ref_position=cfg.ref_position, num_records=cfg.num_records, enforce_realtime=True)
        for index, ch in enumerate((cfg.ch0, cfg.ch1)):
            s.channels[index].configure_vertical(
                range=ch.vertical_range, coupling=_enum(ni.VerticalCoupling, ch.coupling),
                offset=ch.offset)
        for index in (0, 1):
            s.channels[index].input_impedance = cfg.impedance
        # iq_scan.py leaves the driver's default (immediate) trigger; say so explicitly
        s.configure_trigger_immediate()
        return self._read_back()

    def _read_back(self):
        s = self._session

        def read(obj, attribute):
            try:
                return float(getattr(obj, attribute))
            except Exception as e:  # metadata only: never abort a scan because of it
                log.warning('could not read back %s: %s', attribute, e)
                return None

        actual = {'sample_rate': read(s, 'horz_sample_rate')}
        for index in (0, 1):
            ch = s.channels[index]
            actual[f'ch{index}'] = {name: read(ch, name) for name in
                                    ('vertical_range', 'vertical_offset', 'input_impedance')}
        return actual

    def measure(self):
        cfg = self._cfg
        with self._session.initiate():
            waveforms = self._session.channels[0, 1].fetch(timeout=cfg.fetch_timeout)
            timestamp_ns = time.time_ns()
        if len(waveforms) != 2 * cfg.num_records:
            raise DaqError(f'expected {2 * cfg.num_records} waveforms '
                           f'({cfg.num_records} records x 2 channels), got {len(waveforms)}')
        # the driver returns them ordered record by record: ch0, ch1, ch0, ch1, ...
        return summarize([w.samples for w in waveforms[0::2]],
                         [w.samples for w in waveforms[1::2]], timestamp_ns)

    def close(self):
        if self._session is not None:
            session, self._session = self._session, None
            session.close()


# --- simulator ---------------------------------------------------------------------

class SimulatedSweepSg(SgBackend):
    """Signal generator that just remembers its setting (read by SimulatedIqScope)."""

    def __init__(self):
        self.frequency = None
        self.power = None
        self.on = False
        self.calls = []

    def start(self, frequency, power):
        self.frequency, self.power, self.on = frequency, power, True
        self.calls.append(('start', frequency, power))

    def stop(self):
        self.on = False
        self.calls.append(('stop',))


class SimulatedIqScope(IqScopeBackend):
    """Notch resonator in the middle of the scan range, seen through a cable delay.

    ch0 = Re(v), ch1 = Im(v); v is zero while the generator is off.
    """
    timestamp_source = 'simulated (host clock)'
    SIGMA = 2e-3  # V, noise per sample
    AMPLITUDE = 0.02  # V at -10 dBm; scales with the generator power
    DELAY = 50e-9  # s

    def __init__(self, cfg, sg, seed=None):
        self._cfg, self._sg = cfg, sg
        self._rng = np.random.default_rng(seed)
        self._center = 0.5 * (cfg.f_start + cfg.f_stop)
        span = abs(cfg.f_stop - cfg.f_start) or 1e6
        self._qr = self._center / (0.15 * span)  # linewidth = 15 % of the scan range

    def configure(self):
        return {'sample_rate': float(self._cfg.sample_rate), 'simulated': True}

    def transmission(self, frequency):
        """Complex S21 of the model (before amplitude and noise)."""
        dip = 0.6 / (1 + 2j * self._qr * (frequency - self._center) / self._center)
        return (1 - dip) * np.exp(-2j * np.pi * frequency * self.DELAY + 0.3j)

    def measure(self):
        if self._sg.on:
            amplitude = self.AMPLITUDE * 10 ** ((self._sg.power + 10) / 20)
            v = amplitude * self.transmission(self._sg.frequency)
        else:
            v = 0j
        n = self._cfg.npts * self._cfg.num_records
        noise = self._rng.normal(0.0, self.SIGMA / np.sqrt(n), 2)  # noise of the means
        return Measurement(float(v.real + noise[0]), float(v.imag + noise[1]),
                           self.SIGMA, self.SIGMA, n, time.time_ns())


def create_backends(cfg):
    """Return (scope, sg) for ``cfg.backend``; nothing stays open if creating the SG fails."""
    if cfg.backend == 'niscope':
        from kidpack.daq.backends.nirfsg_backend import NiRfsgBackend
        scope = NiScopeIqBackend(cfg)
        try:
            return scope, NiRfsgBackend(cfg.sg_resource)
        except BaseException:
            scope.close()
            raise
    sg = SimulatedSweepSg()
    return SimulatedIqScope(cfg, sg), sg
