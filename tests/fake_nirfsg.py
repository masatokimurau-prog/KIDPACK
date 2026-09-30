"""A fake of NI's official ``nirfsg`` package (only what kidpack uses).

There is no NI driver here, so tests can only check that kidpack makes the calls of the
macro ``iq_scan_kimura20260703.py`` in the same order; they cannot check that the real
driver accepts them. The driver session's ``lock`` is a trap: using it fails, which shows
that the no-op replacement of the macro is installed.
"""
import enum
import types


def make_fake_nirfsg(fail=None, unreadable=()):
    """``fail``: name of a Session method that raises RuntimeError.
    ``unreadable``: property names of the session that raise when read."""
    mod = types.ModuleType('nirfsg')
    mod.GenerationMode = enum.Enum('GenerationMode', 'CW ARB_WAVEFORM SCRIPT')
    mod.calls = []
    mod.sessions = []

    class Session:
        def __init__(self, resource_name, id_query=False, reset_device=False, options={}):
            mod.calls.append(('open', resource_name, id_query, reset_device))
            if fail == 'open':
                raise RuntimeError('device not found')
            mod.sessions.append(self)

        def __getattribute__(self, name):
            if name in unreadable:
                raise RuntimeError(f'cannot read {name}')
            if name == 'instrument_model':
                return 'PXIe-5654'
            if name == 'specific_driver_revision':
                return 'NI-RFSG 24.5'
            return object.__getattribute__(self, name)

        def __setattr__(self, name, value):
            if name != 'lock':
                mod.calls.append(('set', name, value))
            object.__setattr__(self, name, value)

        def lock(self):  # replaced by the no-op of the macro: using it is an error
            raise AssertionError('the driver session lock must not be used')

        def _do(self, name, *args):
            mod.calls.append((name, *args))
            if fail == name:
                raise RuntimeError(f'{name} failed')

        def configure_rf(self, frequency, power_level):
            self._do('configure_rf', frequency, power_level)

        def initiate(self):
            self._do('initiate')

        def abort(self):
            self._do('abort')

        def close(self):
            self._do('close')

    mod.Session = Session
    return mod
