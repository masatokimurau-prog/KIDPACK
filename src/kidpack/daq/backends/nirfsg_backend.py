"""Signal generator backend (PXIe-5654), following ``iq_scan.py``.

``PXIe_5654`` with ``rf_frequency`` / ``rf_power`` / ``initiate`` / ``abort`` is
the API of the ``nirfsg`` module installed on the DAQ PC. It is NOT the API of
the official ``nirfsg`` package on PyPI (which has ``Session`` instead), so do
not ``pip install nirfsg`` over it.
"""
from kidpack.daq.backends.base import SgBackend


class NiRfsgBackend(SgBackend):
    def __init__(self, resource):
        from nirfsg import PXIe_5654
        self._rfsg = PXIe_5654(resource)

    def start(self, frequency, power):
        self._rfsg.rf_frequency = frequency
        self._rfsg.rf_power = power
        self._rfsg.initiate()

    def stop(self):
        self._rfsg.abort()

    def close(self):
        # iq_scan.py never closes the device, so close() may not exist.
        close = getattr(self._rfsg, 'close', None)
        if close is not None:
            close()
