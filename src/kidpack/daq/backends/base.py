"""Hardware-independent interfaces of the DAQ backends.

Everything that touches a device lives behind these two classes; the run
control and the file writer only ever see ``Event`` objects.
"""
import abc
from dataclasses import dataclass

import numpy as np


class DaqError(RuntimeError):
    """A recoverable-by-the-user DAQ failure (bad record length, timeout, ...)."""


@dataclass
class Event:
    """One triggered record, exactly as acquired (scaled volts, no processing)."""
    ch0: np.ndarray  # float32, shape (npts,)
    ch1: np.ndarray  # float32, shape (npts,)
    timestamp_unix_ns: int  # see ScopeBackend.timestamp_source


class ScopeBackend(abc.ABC):
    #: how ``Event.timestamp_unix_ns`` was obtained (recorded in the YAML)
    timestamp_source = 'unknown'

    @abc.abstractmethod
    def configure(self):
        """Apply the configuration; return the settings the device actually uses.

        The returned dict must be YAML-serialisable. It may contain
        ``sample_rate`` and ``ref_position``, which then replace the requested
        values in the stored files.
        """

    @abc.abstractmethod
    def acquire(self):
        """Arm, wait for one trigger and return it as an ``Event``."""

    def close(self):
        pass


class SgBackend(abc.ABC):
    @abc.abstractmethod
    def start(self, frequency, power):
        """Set frequency [Hz] and power [dBm] and start the output."""

    @abc.abstractmethod
    def stop(self):
        """Stop the output."""

    def close(self):
        pass
