"""Hardware-free backends for testing the run control and file writing."""
import time

import numpy as np

from kidpack.daq.backends.base import Event, ScopeBackend, SgBackend
from kidpack.daq.backends.pacing import IntervalPacer


class SimulatedScope(ScopeBackend):
    """Noise plus one decaying pulse at the trigger position."""
    timestamp_source = 'simulated (host clock)'

    def __init__(self, cfg, seed=None):
        self._cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._pacer = (IntervalPacer(cfg.trigger.interval)
                       if cfg.trigger.mode == 'random' else None)

    def configure(self):
        return {'sample_rate': float(self._cfg.sample_rate),
                'ref_position': float(self._cfg.ref_position),
                'simulated': True}

    def acquire(self):
        if self._pacer is not None:
            self._pacer.wait()  # like the hardware backend: one trigger per interval
        n = self._cfg.npts
        x = np.arange(n) - int(n * self._cfg.ref_position / 100)
        pulse = np.exp(-np.clip(x, 0, None) / 500.0) * (x >= 0) * 5e-3
        ch0 = 1e-3 * self._rng.standard_normal(n) + pulse
        ch1 = 1e-3 * self._rng.standard_normal(n) + 0.5 * pulse
        return Event(ch0.astype(np.float32), ch1.astype(np.float32), time.time_ns())


class SimulatedSg(SgBackend):
    def __init__(self):
        self.calls = []

    def start(self, frequency, power):
        self.calls.append(('start', frequency, power))

    def stop(self):
        self.calls.append(('stop',))
