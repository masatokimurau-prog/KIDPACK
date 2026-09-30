"""Signal generator backend (PXIe-5654) through NI's official ``nirfsg`` package.

The calls, their order and the quirks follow the macro ``iq_scan_kimura20260703.py``,
which is known to work on the DAQ PC. For every ``start`` / ``stop`` pair (that is:
for every frequency of an IQ scan, once per run of the pulse DAQ):

    start:  open  nirfsg.Session(resource, id_query=True, reset_device=True)
            replace the session's ``lock`` by a no-op        (as the macro does)
            generation_mode = CW; configure_rf(frequency, power); output_enabled = True
            initiate()                        (returns when the RF output has settled)
    stop:   abort(); close the session

The ``lock`` replacement is kept because the macro needs it; the reason is not
known. NI's ``nirfsg`` package (and the NI-RFSG driver) must already be installed in
the Python environment: kidpack does not install it, so that the installed versions
are never touched.
"""
import contextlib
import logging

from kidpack.daq.backends.base import SgBackend

log = logging.getLogger('kidpack.daq')


class NiRfsgBackend(SgBackend):
    api = 'nirfsg.Session (CW, session reset and re-opened per start)'

    def __init__(self, resource):
        try:
            import nirfsg
        except ImportError as e:
            raise ImportError(
                "the 'nirfsg' module is not installed in this Python environment, so the "
                "signal generator cannot be controlled. kidpack does not install it: install "
                "NI's package yourself (e.g. 'pip install nirfsg', plus the NI-RFSG driver)") from e
        self._nirfsg = nirfsg
        self._resource = resource
        self._session = None  # open only between start() and stop()
        # Open the device once, exactly as a start does, so that a missing or busy
        # generator is reported now (before a run or scan is created), and note what it is.
        session = self._open()
        try:
            self.info = self._read_info(session)
        finally:
            session.close()

    def _open(self):
        session = self._nirfsg.Session(self._resource, id_query=True, reset_device=True)
        try:
            session.lock = lambda: contextlib.nullcontext()
        except BaseException:
            _close_quietly(session)
            raise
        return session

    @staticmethod
    def _read_info(session):
        """Instrument model and driver revision (the macro prints them; they may be unavailable)."""
        info = {}
        for attribute in ('instrument_model', 'specific_driver_revision'):
            try:
                info[attribute] = str(getattr(session, attribute))
            except Exception as e:
                log.warning('could not read %s of the signal generator: %r', attribute, e)
        return info

    def start(self, frequency, power):
        self.stop()  # never leave a previous session open
        session = self._open()
        try:
            session.generation_mode = self._nirfsg.GenerationMode.CW
            session.configure_rf(frequency, power)
            session.output_enabled = True
            session.initiate()
        except BaseException:
            _close_quietly(session)
            raise
        self._session = session

    def stop(self):
        session, self._session = self._session, None
        if session is None:
            return
        try:
            session.abort()
        finally:
            session.close()

    def close(self):
        self.stop()


def _close_quietly(session):
    """Close a session on an error path without hiding the error that got us here."""
    try:
        session.close()
    except Exception as e:
        log.warning('failed to close the signal generator session: %r', e)
