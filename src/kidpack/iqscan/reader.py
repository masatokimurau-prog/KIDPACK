"""Reader for IQ-scan files (``dd`` = frequency [Hz], mean ch0 [V], mean ch1 [V]).

Reads the files of ``kidpack-iqscan`` and those of the old ``iq_scan.py`` (which
only have ``dd``). Read-only: files are never modified.
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


class IqScanError(ValueError):
    """The file is not an IQ-scan file."""


@dataclass
class IqScan:
    path: str
    frequency: np.ndarray  # (n,) Hz
    i: np.ndarray  # (n,) mean ch0 [V]
    q: np.ndarray  # (n,) mean ch1 [V]
    power_dbm: Optional[float] = None  # None in files of the old iq_scan.py

    @property
    def iq(self):
        """ch0 + i ch1 [V], as the analysis scripts build it."""
        return self.i + 1j * self.q

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.path))[0]


def load_iqscan(path):
    """Load one IQ-scan file; raise IqScanError if it does not have the expected layout."""
    path = str(path)
    with np.load(path) as d:
        if 'dd' not in d.files:
            raise IqScanError(f'{path}: no "dd" array (not an IQ-scan file?)')
        dd = np.asarray(d['dd'], dtype=np.float64)
        power = float(d['power_dbm']) if 'power_dbm' in d.files else None
    if dd.ndim != 2 or dd.shape[1] != 3 or dd.shape[0] < 1:
        raise IqScanError(f'{path}: "dd" must have shape (n, 3) = frequency, ch0, ch1; '
                          f'got {dd.shape}')
    return IqScan(path, dd[:, 0], dd[:, 1], dd[:, 2], power)
