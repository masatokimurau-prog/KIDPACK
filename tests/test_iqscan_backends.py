"""The NI backend of the IQ scan against a fake driver module.

There is no NI hardware/driver here, so these tests only check that the
configuration becomes the same driver calls (same order, same arguments) as the
verified macro iq_scan_kimura20260703.py and that the returned records are
grouped and averaged the way the macro does. They cannot check that the real
driver accepts the calls.
"""
import contextlib
import enum
import sys
import types

import numpy as np
import pytest

from kidpack.daq.backends.base import DaqError
from kidpack.iqscan.backends import NiScopeIqBackend, create_backends, summarize
from kidpack.iqscan.config import IqScanConfig
from fake_nirfsg import make_fake_nirfsg


class Wave:
    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=np.float64)


def records(nrec=2, nsamp=6):
    """Distinct, recognisable data: ch0 of record r is 100*r + 0..n, ch1 is -(1000*r + 0..n)."""
    ramp = np.arange(nsamp, dtype=np.float64)
    ch0 = [100.0 * r + ramp for r in range(nrec)]
    ch1 = [-(1000.0 * r + ramp) for r in range(nrec)]
    return ch0, ch1


def make_fake_niscope(nrec=2, nsamp=6, order='record-major'):
    ni = types.ModuleType('niscope')
    ni.VerticalCoupling = enum.Enum('VerticalCoupling', 'AC DC GND')
    ni.calls = []
    ch0, ch1 = records(nrec, nsamp)

    class Channel:
        def __init__(self, session, index):
            object.__setattr__(self, '_s', session)
            object.__setattr__(self, '_i', index)

        def configure_vertical(self, range, coupling, offset=0.0, probe_attenuation=1.0,
                               enabled=True):
            ni.calls.append(('configure_vertical', self._i, range, coupling, offset))
            self._s.values[self._i]['vertical_range'] = range
            self._s.values[self._i]['vertical_offset'] = offset

        def __setattr__(self, name, value):
            ni.calls.append(('set', self._i, name, value))
            self._s.values[self._i][name] = value

        def __getattr__(self, name):
            return self._s.values[self._i][name]

    class BothChannels:
        def fetch(self, timeout):
            ni.calls.append(('fetch', timeout))
            waves = []
            for r in range(nrec):  # ch0, ch1, ch0, ch1, ... as iq_scan.py assumes
                waves += [Wave(ch0[r]), Wave(ch1[r])]
            return waves

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
            self.horz_sample_rate = 1e4

        def configure_horizontal_timing(self, **kwargs):
            ni.calls.append(('horizontal', kwargs))

        def configure_trigger_immediate(self):
            ni.calls.append(('trigger_immediate',))

        @contextlib.contextmanager
        def initiate(self):
            ni.calls.append(('initiate',))
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


def cfg(**kw):
    return IqScanConfig(f_start=5.324e9, f_stop=5.328e9, num_points=3, **kw)


def test_default_configuration_issues_the_calls_of_the_verified_macro(ni):
    NiScopeIqBackend(cfg()).configure()
    dc = ni.VerticalCoupling.DC
    assert ni.calls == [
        ('open', 'PXI2Slot2'),
        ('horizontal', dict(min_sample_rate=1e4, min_num_pts=10000, ref_position=50.0,
                            num_records=2, enforce_realtime=True)),
        ('configure_vertical', 0, 1.0, dc, 0.0),
        ('configure_vertical', 1, 1.0, dc, 0.0),
        ('set', 0, 'input_impedance', 1e6),
        ('set', 1, 'input_impedance', 1e6),
    ]
    # no trigger is configured: the driver's default (immediate) is what the macro relies on
    assert ('trigger_immediate',) not in ni.calls


def test_non_default_settings_reach_the_driver(ni):
    from kidpack.daq.config import ChannelConfig
    NiScopeIqBackend(cfg(resource='PXI1Slot9', sample_rate=2e4, npts=12345, num_records=4,
                         ref_position=10.0, impedance=1e6,
                         ch0=ChannelConfig(0.5, 'ac', -0.01),
                         ch1=ChannelConfig(0.2, 'gnd', 0.02))).configure()
    assert ('open', 'PXI1Slot9') in ni.calls
    assert ('horizontal', dict(min_sample_rate=2e4, min_num_pts=12345, ref_position=10.0,
                               num_records=4, enforce_realtime=True)) in ni.calls
    assert ('configure_vertical', 0, 0.5, ni.VerticalCoupling.AC, -0.01) in ni.calls
    assert ('configure_vertical', 1, 0.2, ni.VerticalCoupling.GND, 0.02) in ni.calls
    assert ('set', 1, 'input_impedance', 1e6) in ni.calls


def test_actual_settings_are_read_back(ni):
    actual = NiScopeIqBackend(cfg()).configure()
    assert actual['sample_rate'] == 1e4
    assert actual['ch0'] == {'vertical_range': 1.0, 'vertical_offset': 0.0, 'input_impedance': 1e6}


def test_measure_acquires_immediately_inside_initiate_with_the_configured_timeout(ni):
    backend = NiScopeIqBackend(cfg(fetch_timeout=42.0))
    backend.configure()
    ni.calls.clear()
    m = backend.measure()
    assert ni.calls == [('initiate',), ('fetch', 42.0), ('abort',)]
    assert m.timestamp_ns > 1.7e18


def test_measure_averages_like_the_macro(ni):
    backend = NiScopeIqBackend(cfg())
    backend.configure()
    m = backend.measure()

    # the oracle: the verified macro, line by line
    waveforms = []
    ch0, ch1 = records()
    for r in range(2):
        waveforms += [Wave(ch0[r]), Wave(ch1[r])]
    n_record = 2
    map_func = lambda x: x.samples
    s0 = np.array(list(map(map_func, waveforms[0::2]))).sum(axis=0)
    s1 = np.array(list(map(map_func, waveforms[1::2]))).sum(axis=0)
    mean0, mean1 = s0.mean() / n_record, s1.mean() / n_record

    assert m.mean0 == pytest.approx(mean0) and m.mean1 == pytest.approx(mean1)
    assert (mean0, mean1) == pytest.approx((52.5, -(500.0 + 2.5)))  # sanity of the oracle itself
    assert m.n_samples == 2 * 6  # samples per channel over both records
    both0 = np.concatenate(ch0)
    assert m.std0 == pytest.approx(both0.std(ddof=1)) and m.std1 == pytest.approx(np.concatenate(ch1).std(ddof=1))


def test_ch0_and_ch1_are_not_swapped(ni):
    m = NiScopeIqBackend(cfg()).measure()
    assert m.mean0 > 0 > m.mean1  # ch0 data are positive, ch1 data negative


def test_the_number_of_records_is_taken_from_the_configuration(monkeypatch):
    monkeypatch.setitem(sys.modules, 'niscope', make_fake_niscope(nrec=3, nsamp=4))
    m = NiScopeIqBackend(cfg(num_records=3)).measure()
    assert m.n_samples == 12
    assert m.mean0 == pytest.approx(np.concatenate(records(3, 4)[0]).mean())


def test_a_wrong_number_of_waveforms_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'niscope', make_fake_niscope(nrec=1))
    with pytest.raises(DaqError, match='expected 4 waveforms'):
        NiScopeIqBackend(cfg(num_records=2)).measure()


def test_close_is_idempotent(ni):
    backend = NiScopeIqBackend(cfg())
    backend.close()
    backend.close()
    assert ni.calls.count(('close',)) == 1


def test_summarize_statistics():
    m = summarize([np.array([1.0, 3.0]), np.array([5.0])], [np.array([2.0, 2.0, 2.0])], 7)
    assert (m.mean0, m.mean1, m.n_samples, m.timestamp_ns) == (3.0, 2.0, 3, 7)
    assert m.std0 == pytest.approx(2.0) and m.std1 == 0.0
    assert summarize([np.array([4.0])], [np.array([1.0])], 0).std0 == 0.0  # one sample: no NaN


def test_create_backends_opens_the_scope_and_the_generator_through_nirfsg(monkeypatch):
    rfsg = make_fake_nirfsg()
    monkeypatch.setitem(sys.modules, 'nirfsg', rfsg)
    monkeypatch.setitem(sys.modules, 'niscope', make_fake_niscope())

    scope, sg = create_backends(cfg(sg_resource='PXI1Slot5'))
    assert isinstance(scope, NiScopeIqBackend)
    assert rfsg.calls[0] == ('open', 'PXI1Slot5', True, True)  # the generator was checked at creation
    assert sg.info['instrument_model'] == 'PXIe-5654'


def test_the_scope_is_closed_again_if_the_generator_cannot_be_opened(monkeypatch):
    ni_fake = make_fake_niscope()
    monkeypatch.setitem(sys.modules, 'niscope', ni_fake)
    monkeypatch.setitem(sys.modules, 'nirfsg', make_fake_nirfsg(fail='open'))
    with pytest.raises(RuntimeError, match='device not found'):
        create_backends(cfg())
    assert ni_fake.calls[-1] == ('close',)

    ni_fake.calls.clear()
    monkeypatch.setitem(sys.modules, 'nirfsg', None)  # package not installed
    with pytest.raises(ImportError, match='pip install nirfsg'):
        create_backends(cfg())
    assert ni_fake.calls[-1] == ('close',)


def test_the_simulator_backends_need_no_drivers():
    scope, sg = create_backends(cfg(backend='simulator'))
    sg.start(5.326e9, -10.0)
    assert scope.measure().n_samples == 20000


class Tap(list):
    """A call log that also appends every entry, tagged with its instrument, to a shared log."""

    def __init__(self, tag, merged):
        super().__init__()
        self.tag, self.merged = tag, merged

    def append(self, item):
        self.merged.append((self.tag, *item))
        super().append(item)


def test_a_whole_scan_follows_the_sequence_of_the_verified_macro(monkeypatch, tmp_path):
    from kidpack.iqscan.runner import run_iqscan

    merged = []
    ni, rf = make_fake_niscope(), make_fake_nirfsg()
    ni.calls, rf.calls = Tap('scope', merged), Tap('sg', merged)
    monkeypatch.setitem(sys.modules, 'niscope', ni)
    monkeypatch.setitem(sys.modules, 'nirfsg', rf)

    scan_cfg = IqScanConfig(f_start=5.2e9, f_stop=5.3e9, num_points=3, power=-2.0,
                            output_dir=str(tmp_path), name='macro')
    scope, sg = create_backends(scan_cfg)
    result = run_iqscan(scan_cfg, scope, sg)
    assert result.status == 'completed' and result.points_measured == 3

    # what happens per frequency, once the digitizer is configured
    configured = max(i for i, e in enumerate(merged) if e[:3] == ('scope', 'set', 1))
    per_point, tail = merged[configured + 1:-1], merged[-1]
    cw = rf.GenerationMode.CW
    expected = []
    for frequency in (5.2e9, 5.25e9, 5.3e9):
        expected += [
            ('sg', 'open', 'PXI2Slot3', True, True),   # nirfsg.Session(..., id_query=True, reset_device=True)
            ('sg', 'set', 'generation_mode', cw),
            ('sg', 'configure_rf', frequency, -2.0),
            ('sg', 'set', 'output_enabled', True),
            ('sg', 'initiate'),                        # rfsg.initiate()
            ('scope', 'initiate'),                     # with session.initiate():
            ('scope', 'fetch', 17.0),                  #     session.channels[0,1].fetch(timeout=17)
            ('scope', 'abort'),                        # (leaving the with-block)
            ('sg', 'abort'),                           # rfsg.abort()
            ('sg', 'close'),                           # (leaving the SG's with-block)
        ]
    assert per_point == expected
    assert tail == ('scope', 'close')

    # and what it stored
    arrays = np.load(tmp_path / 'macro.npz')
    assert arrays['dd'].shape == (3, 3)
    import yaml
    meta = yaml.safe_load((tmp_path / 'macro.yaml').read_text())
    assert meta['signal_generator']['resource'] == 'PXI2Slot3'
    assert 'nirfsg.Session' in meta['signal_generator']['api']
    assert meta['signal_generator']['instrument_model'] == 'PXIe-5654'
