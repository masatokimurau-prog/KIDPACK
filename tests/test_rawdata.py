import hashlib

import numpy as np
import pytest

from kidpack.daq.backends.simulator import SimulatedScope
from kidpack.daq.config import DaqConfig
from kidpack.daq.runner import run_daq
from kidpack.rawdata import RawDataError, load_raw, time_axis_s


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_npz(path, **overrides):
    arrays = dict(ch0=np.zeros((3, 100), np.float32), ch1=np.ones((3, 100), np.float32),
                  npts=np.int32(100), sample_rate=np.float64(2.5e9), ref_position=np.float64(20))
    arrays.update(overrides)
    arrays = {k: v for k, v in arrays.items() if v is not None}
    np.savez(path, **arrays)
    return path


def test_reads_a_file_written_by_the_daq(tmp_path):
    cfg = DaqConfig(run_number=1, events_per_file=6, num_files=1, output_dir=str(tmp_path), npts=80)
    run_daq(cfg, SimulatedScope(cfg, seed=1))
    raw = load_raw(tmp_path / 'run_01' / 'data' / 'run01-00.npz')

    assert raw.nevents == 6 and raw.npts == 80
    assert raw.ch0.shape == raw.ch1.shape == (6, 80) and raw.ch0.dtype == np.float32
    assert raw.sample_rate == 2.5e9 and raw.ref_position == 20.0
    np.testing.assert_array_equal(raw.event_id, np.arange(6))
    assert raw.timestamp_unix_ns.dtype == np.int64 and len(raw.timestamp_unix_ns) == 6


def test_reading_never_modifies_the_file(tmp_path):
    path = write_npz(tmp_path / 'a.npz')
    before = sha256(path)
    load_raw(path)
    assert sha256(path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.npz']


def test_files_from_the_old_kid_py_can_be_read(tmp_path):
    # no event_id / timestamp_unix_ns, but the old daq_rate and a pickled deltat array
    path = write_npz(tmp_path / 'old.npz', daq_rate=np.float64(49.7),
                     deltat=np.array([object(), object(), object()], dtype=object))
    raw = load_raw(path)
    np.testing.assert_array_equal(raw.event_id, [0, 1, 2])
    assert raw.timestamp_unix_ns is None


@pytest.mark.parametrize('missing', ['ch0', 'ch1', 'npts', 'sample_rate', 'ref_position'])
def test_missing_key_is_reported(tmp_path, missing):
    path = write_npz(tmp_path / 'x.npz', **{missing: None})
    with pytest.raises(RawDataError, match=missing):
        load_raw(path)


def test_inconsistent_shapes_are_rejected(tmp_path):
    with pytest.raises(RawDataError, match='same shape'):
        load_raw(write_npz(tmp_path / 'a.npz', ch1=np.zeros((3, 99), np.float32)))
    with pytest.raises(RawDataError, match='npts'):
        load_raw(write_npz(tmp_path / 'b.npz', npts=np.int32(50)))


def test_time_axis_follows_the_specification():
    t = time_axis_s(5000, 2.5e9, 20.0)
    assert t[1000] == 0.0  # index npts*ref/100 is the trigger
    assert t[0] == pytest.approx(-1000 / 2.5e9) and t[-1] == pytest.approx(3999 / 2.5e9)
    assert np.diff(t) == pytest.approx(1 / 2.5e9)
