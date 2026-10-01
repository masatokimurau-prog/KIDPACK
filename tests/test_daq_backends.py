"""The NI backends against fake driver modules.

There is no NI hardware/driver here, so these tests only check that the
configuration is translated into the same driver calls (same order, same
arguments) as the original hard-coded kid.py / iq_scan.py. They cannot check
that the real drivers accept them.
"""
import contextlib
import enum
import sys
import time
import types

import numpy as np
import pytest

from kidpack.daq.backends.base import DaqError
from kidpack.daq.backends.nirfsg_backend import NiRfsgBackend
from kidpack.daq.backends.niscope_backend import NiScopeBackend
from kidpack.daq.config import ChannelConfig, DaqConfig, TriggerConfig
from fake_nirfsg import make_fake_nirfsg


def make_fake_niscope(record_len=6000):
    ni = types.ModuleType('niscope')
    ni.VerticalCoupling = enum.Enum('VerticalCoupling', 'AC DC GND')
    ni.TriggerCoupling = enum.Enum('TriggerCoupling', 'AC DC HF_REJECT LF_REJECT AC_PLUS_HF_REJECT')
    ni.TriggerSlope = enum.Enum('TriggerSlope', 'NEGATIVE POSITIVE SLOPE_EITHER')
    ni.WhichTrigger = enum.Enum('WhichTrigger', 'START ARM_REFERENCE REFERENCE ADVANCE')
    ni.calls = []
    ni.times = {}  # monotonic time of the first initiate / software trigger

    class Wave:
        def __init__(self, samples):
            self.samples = samples

    class Channel:
        def __init__(self, session, index):
            object.__setattr__(self, '_s', session)
            object.__setattr__(self, '_i', index)

        def configure_vertical(self, range, coupling, offset=0.0, probe_attenuation=1.0, enabled=True):
            ni.calls.append(('configure_vertical', self._i, range, coupling))
            self._s.values[self._i]['vertical_range'] = range + 0.06  # the driver coerces ranges

        def __setattr__(self, name, value):
            ni.calls.append(('set', self._i, name, value))
            self._s.values[self._i][name] = value

        def __getattr__(self, name):
            return self._s.values[self._i][name]

    class BothChannels:
        def fetch(self, timeout):
            ni.calls.append(('fetch', timeout))
            ramp = np.arange(record_len, dtype=np.float64)
            return [Wave(ramp), Wave(-ramp)]

    class Channels:
        def __init__(self, session):
            self._s = session

        def __getitem__(self, key):
            return BothChannels() if isinstance(key, tuple) else Channel(self._s, key)

    class Session:
        def __init__(self, resource):
            ni.calls.append(('open', resource))
            self.values = {0: {}, 1: {}}
            self.channels = Channels(self)
            self.horz_sample_rate = 2.5e9
            self.horz_record_ref_position = 20.0

        def configure_horizontal_timing(self, **kwargs):
            ni.calls.append(('horizontal', kwargs))

        def configure_trigger_edge(self, **kwargs):
            ni.calls.append(('trigger', kwargs))

        def configure_trigger_software(self, **kwargs):
            ni.calls.append(('trigger_software', kwargs))

        def send_software_trigger_edge(self, which_trigger):
            ni.calls.append(('send_software_trigger', which_trigger))
            ni.times.setdefault('send', []).append(time.monotonic())

        @contextlib.contextmanager
        def initiate(self):
            ni.calls.append(('initiate',))
            ni.times.setdefault('initiate', []).append(time.monotonic())
            yield
            ni.calls.append(('abort',))

        def close(self):
            ni.calls.append(('close',))

    ni.Session = Session
    return ni


@pytest.fixture
def ni(monkeypatch):
    fake = make_fake_niscope()
    monkeypatch.setitem(sys.modules, 'niscope', fake)
    return fake


def default_cfg(**kw):
    return DaqConfig(run_number=1, events_per_file=1, num_files=1, **kw)


def test_default_configuration_issues_the_calls_of_kid_py_except_the_trigger_coupling_is_dc(ni):
    backend = NiScopeBackend(default_cfg())
    backend.configure()
    dc, trigger_dc, pos = ni.VerticalCoupling.DC, ni.TriggerCoupling.DC, ni.TriggerSlope.POSITIVE
    assert ni.calls == [
        ('open', 'PXI2Slot2'),
        ('configure_vertical', 0, 0.01, dc), ('set', 0, 'vertical_offset', 0.0),
        ('configure_vertical', 1, 0.01, dc), ('set', 1, 'vertical_offset', 0.0),
        ('set', 0, 'input_impedance', 50.0), ('set', 1, 'input_impedance', 50.0),
        ('set', 0, 'max_input_frequency', -1.0), ('set', 1, 'max_input_frequency', -1.0),
        ('horizontal', dict(min_sample_rate=2.5e9, min_num_pts=5000, ref_position=20.0,
                            num_records=1, enforce_realtime=True)),
        ('trigger', dict(trigger_source='VAL_EXTERNAL', level=2.2,
                         trigger_coupling=trigger_dc, slope=pos)),
    ]


def test_non_default_settings_reach_the_driver(ni):
    cfg = default_cfg(resource='PXI1Slot9', npts=12500, ref_position=50.0, impedance=1e6,
                      bandwidth=175e6,
                      ch0=ChannelConfig(0.2, 'ac', -0.014), ch1=ChannelConfig(0.05, 'gnd', 0.001),
                      trigger=TriggerConfig('1', -8e-5, 'negative', 'dc'))
    NiScopeBackend(cfg).configure()
    calls = ni.calls
    assert ('open', 'PXI1Slot9') in calls
    assert ('configure_vertical', 0, 0.2, ni.VerticalCoupling.AC) in calls
    assert ('configure_vertical', 1, 0.05, ni.VerticalCoupling.GND) in calls
    assert ('set', 0, 'vertical_offset', -0.014) in calls
    assert ('set', 1, 'vertical_offset', 0.001) in calls
    assert ('set', 1, 'input_impedance', 1e6) in calls
    assert ('set', 0, 'max_input_frequency', 175e6) in calls
    assert ('horizontal', dict(min_sample_rate=2.5e9, min_num_pts=12500, ref_position=50.0,
                               num_records=1, enforce_realtime=True)) in calls
    assert ('trigger', dict(trigger_source='1', level=-8e-5,
                            trigger_coupling=ni.TriggerCoupling.DC,
                            slope=ni.TriggerSlope.NEGATIVE)) in calls


def test_actual_settings_are_read_back(ni):
    actual = NiScopeBackend(default_cfg()).configure()
    assert actual['sample_rate'] == 2.5e9 and actual['ref_position'] == 20.0
    assert actual['ch0']['vertical_range'] == pytest.approx(0.07)  # coerced by the "driver"
    assert actual['ch1']['input_impedance'] == 50.0
    assert actual['ch0']['max_input_frequency'] == -1.0


def test_acquire_truncates_to_npts_and_keeps_channel_order(ni):
    backend = NiScopeBackend(default_cfg(fetch_timeout=42.0))
    backend.configure()
    ni.calls.clear()
    event = backend.acquire()
    assert ni.calls == [('initiate',), ('fetch', 42.0), ('abort',)]
    assert event.ch0.dtype == event.ch1.dtype == np.float32
    assert event.ch0.shape == event.ch1.shape == (5000,)
    np.testing.assert_array_equal(event.ch0, np.arange(5000))  # ch0 first ...
    np.testing.assert_array_equal(event.ch1, -np.arange(5000))  # ... then ch1
    assert isinstance(event.timestamp_unix_ns, int) and event.timestamp_unix_ns > 1.7e18


def random_cfg(interval=0.05, **kw):
    return default_cfg(trigger=TriggerConfig(mode='random', interval=interval), **kw)


def test_edge_mode_never_touches_the_software_trigger(ni):
    backend = NiScopeBackend(default_cfg())
    backend.configure()
    backend.acquire()
    assert not [c for c in ni.calls if c[0] in ('trigger_software', 'send_software_trigger')]


def test_random_mode_configures_a_software_trigger_instead_of_an_edge_trigger(ni):
    NiScopeBackend(random_cfg()).configure()
    names = [c[0] for c in ni.calls]
    assert 'trigger_software' in names and 'trigger' not in names
    assert ('horizontal', dict(min_sample_rate=2.5e9, min_num_pts=5000, ref_position=20.0,
                               num_records=1, enforce_realtime=True)) in ni.calls


def test_random_mode_sends_the_trigger_while_armed_and_before_fetching(ni):
    backend = NiScopeBackend(random_cfg(fetch_timeout=42.0))
    backend.configure()
    ni.calls.clear()
    event = backend.acquire()
    assert ni.calls == [('initiate',), ('send_software_trigger', ni.WhichTrigger.REFERENCE),
                        ('fetch', 42.0), ('abort',)]
    assert event.ch0.shape == (5000,) and event.timestamp_unix_ns > 1.7e18
    assert 'software trigger' in backend.timestamp_source


def test_random_mode_waits_for_the_pretrigger_samples_before_triggering(monkeypatch):
    # 100 ms record with the trigger at 50 %: 50 ms of pre-trigger data must be taken first
    ni = make_fake_niscope(record_len=100_000)
    monkeypatch.setitem(sys.modules, 'niscope', ni)
    backend = NiScopeBackend(random_cfg(interval=0.001, sample_rate=1e6, npts=100_000,
                                        ref_position=50.0))
    backend.configure()
    backend.acquire()
    assert ni.times['send'][0] - ni.times['initiate'][0] >= 0.05


def test_random_mode_issues_one_trigger_per_interval(ni):
    backend = NiScopeBackend(random_cfg(interval=0.05))
    backend.configure()
    for _ in range(4):
        backend.acquire()
    sends = ni.times['send']
    assert len(sends) == 4
    # fixed grid: single gaps jitter with the OS timer, the total follows the grid
    assert all(b - a > 0.02 for a, b in zip(sends, sends[1:]))
    assert sends[-1] - sends[0] == pytest.approx(3 * 0.05, abs=0.03)


def test_a_record_shorter_than_npts_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'niscope', make_fake_niscope(record_len=100))
    backend = NiScopeBackend(default_cfg())
    backend.configure()
    with pytest.raises(DaqError):
        backend.acquire()


def test_close_is_idempotent(ni):
    backend = NiScopeBackend(default_cfg())
    backend.close()
    backend.close()
    assert ni.calls.count(('close',)) == 1


# --- signal generator: the calls of the verified macro iq_scan_kimura20260703.py ----------

@pytest.fixture
def rfsg(monkeypatch):
    fake = make_fake_nirfsg()
    monkeypatch.setitem(sys.modules, 'nirfsg', fake)
    return fake


def test_creating_the_backend_opens_the_device_once_to_check_it_and_reads_its_model(rfsg):
    sg = NiRfsgBackend('PXI2Slot3')
    assert rfsg.calls == [('open', 'PXI2Slot3', True, True), ('close',)]
    assert sg.info == {'instrument_model': 'PXIe-5654', 'specific_driver_revision': 'NI-RFSG 24.5'}
    assert 'nirfsg.Session' in sg.api


def test_start_and_stop_make_the_calls_of_the_macro_in_its_order(rfsg):
    sg = NiRfsgBackend('PXI2Slot3')
    rfsg.calls.clear()

    sg.start(5.49e9, -2.0)
    assert rfsg.calls == [
        ('open', 'PXI2Slot3', True, True),  # Session(resource, id_query=True, reset_device=True)
        ('set', 'generation_mode', rfsg.GenerationMode.CW),
        ('configure_rf', 5.49e9, -2.0),
        ('set', 'output_enabled', True),
        ('initiate',),
    ]
    rfsg.calls.clear()
    sg.stop()
    assert rfsg.calls == [('abort',), ('close',)]  # rfsg.abort(), then leaving the with-block


def test_the_driver_lock_is_replaced_by_a_no_op_like_in_the_macro(rfsg):
    sg = NiRfsgBackend('PXI2Slot3')
    sg.start(5.49e9, -2.0)
    for session in rfsg.sessions:  # the one of the check and the one of the start
        with session.lock():  # would raise AssertionError if it were still the driver's lock
            pass
    sg.stop()


def test_every_frequency_gets_a_freshly_reset_session(rfsg):
    sg = NiRfsgBackend('PXI2Slot3')
    rfsg.calls.clear()
    for frequency in (5.30e9, 5.31e9, 5.32e9):
        sg.start(frequency, -2.0)
        sg.stop()
    assert rfsg.calls.count(('open', 'PXI2Slot3', True, True)) == 3
    assert rfsg.calls.count(('abort',)) == rfsg.calls.count(('close',)) == 3
    assert [c[1:] for c in rfsg.calls if c[0] == 'configure_rf'] == [
        (5.30e9, -2.0), (5.31e9, -2.0), (5.32e9, -2.0)]


def test_starting_again_first_stops_the_previous_session(rfsg):
    sg = NiRfsgBackend('PXI2Slot3')
    sg.start(5.30e9, -2.0)
    rfsg.calls.clear()
    sg.start(5.31e9, -2.0)
    assert rfsg.calls[:3] == [('abort',), ('close',), ('open', 'PXI2Slot3', True, True)]


@pytest.mark.parametrize('failing', ['configure_rf', 'initiate'])
def test_a_failed_start_closes_the_session_and_raises(monkeypatch, failing):
    fake = make_fake_nirfsg(fail=failing)
    monkeypatch.setitem(sys.modules, 'nirfsg', fake)
    sg = NiRfsgBackend('PXI2Slot3')
    fake.calls.clear()
    with pytest.raises(RuntimeError, match=failing):
        sg.start(5.49e9, -2.0)
    assert fake.calls[-1] == ('close',)  # no session is left open
    fake.calls.clear()
    sg.stop()
    assert fake.calls == []  # nothing is running, nothing to abort


def test_stop_closes_the_session_even_if_abort_fails(monkeypatch):
    fake = make_fake_nirfsg(fail='abort')
    monkeypatch.setitem(sys.modules, 'nirfsg', fake)
    sg = NiRfsgBackend('PXI2Slot3')
    sg.start(5.49e9, -2.0)
    with pytest.raises(RuntimeError, match='abort'):
        sg.stop()
    assert fake.calls[-1] == ('close',)


def test_close_stops_a_running_generator_and_is_idempotent(rfsg):
    sg = NiRfsgBackend('PXI2Slot3')
    sg.start(5.49e9, -2.0)
    rfsg.calls.clear()
    sg.close()
    sg.close()
    assert rfsg.calls == [('abort',), ('close',)]


def test_unreadable_model_information_does_not_stop_anything(monkeypatch):
    fake = make_fake_nirfsg(unreadable=('specific_driver_revision',))
    monkeypatch.setitem(sys.modules, 'nirfsg', fake)
    sg = NiRfsgBackend('PXI2Slot3')
    assert sg.info == {'instrument_model': 'PXIe-5654'}
    sg.start(5.49e9, -2.0)


def test_a_device_that_cannot_be_opened_is_reported_when_the_backend_is_created(monkeypatch):
    fake = make_fake_nirfsg(fail='open')
    monkeypatch.setitem(sys.modules, 'nirfsg', fake)
    with pytest.raises(RuntimeError, match='device not found'):
        NiRfsgBackend('PXI2Slot3')
    assert fake.calls == [('open', 'PXI2Slot3', True, True)]


def test_a_missing_nirfsg_package_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, 'nirfsg', None)  # import raises ImportError
    with pytest.raises(ImportError, match='pip install nirfsg'):
        NiRfsgBackend('PXI2Slot3')
