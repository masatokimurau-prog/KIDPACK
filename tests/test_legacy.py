"""Conversion of the files of the old DAQ macro (kid.py) to the current raw-data format."""
import hashlib
import importlib.util
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from kidpack.daq.backends.simulator import SimulatedScope
from kidpack.daq.config import DaqConfig
from kidpack.daq.runner import run_daq
from kidpack.legacy import LegacyFileError, convert_old_files, load_old_file, start_ns_from_name
from kidpack.monitor.cli import main as monitor_main
from kidpack.rawdata import load_raw

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'make_sample_data.py'
ANALYSIS = ROOT / 'examples' / '02_pulse_analysis.py'

NPTS = 5000
SAMPLE_RATE = 2.5e9
# the DAQ writes the YAML keys of a hardware run too; of these a converted run knows nothing
ONLY_IN_DAQ_YAML = {'actual_settings', 'backend', 'resource', 'fetch_timeout_s', 'sg', 'command_line'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def write_old(path, nevents=3, npts=NPTS, seed=0, rate=49.7, microseconds=(123456, 7, 999999)):
    """A file written like the old kid.py: float32 waveforms, deltat as timedeltas, int32 ref_position."""
    rng = np.random.default_rng(seed)
    t = np.arange(npts)
    pulse = np.where(t >= 1000, np.exp(-(t - 1000) / 400.0), 0.0)
    ch0 = (0.01 + 0.002 * rng.standard_normal((nevents, npts)) + 0.005 * pulse).astype(np.float32)
    ch1 = (-0.02 + 0.002 * rng.standard_normal((nevents, npts)) + 0.002 * pulse).astype(np.float32)
    deltat = [timedelta(seconds=i, microseconds=microseconds[i % len(microseconds)])
              for i in range(nevents)]
    np.savez(path, ch0=ch0, ch1=ch1, npts=np.int32(npts), sample_rate=np.float64(SAMPLE_RATE),
             ref_position=np.int32(20), daq_rate=np.float64(rate),
             deltat=np.array(deltat, dtype=object))
    return ch0, ch1, deltat


def delta_ns(delta):
    return (delta.days * 86400 + delta.seconds) * 10 ** 9 + delta.microseconds * 1000


@pytest.fixture
def old(tmp_path):
    """Two old files of one day, written out of order, in a directory of their own."""
    directory = tmp_path / 'old'
    directory.mkdir()
    late = directory / 'wf_260831_162120_49.81Hz.npz'
    early = directory / 'wf_260831_154133_49.39Hz.npz'
    return dict(late=(late, write_old(late, seed=1)), early=(early, write_old(early, seed=2)),
                directory=directory)


def daq_file(tmp_path):
    cfg = DaqConfig(run_number=1, events_per_file=4, num_files=1, output_dir=str(tmp_path / 'daq'),
                    npts=NPTS)
    run_daq(cfg, SimulatedScope(cfg, seed=1))
    return tmp_path / 'daq' / 'run_01'


# --- the file name -------------------------------------------------------------------

def test_the_start_is_the_file_name_read_as_japan_time():
    expected = datetime(2026, 8, 31, 6, 55, 41, tzinfo=timezone.utc)  # 15:55:41 at UTC+9
    assert start_ns_from_name('/x/wf_260831_155541_49.84Hz.npz') == int(expected.timestamp()) * 10 ** 9


def test_another_time_zone_can_be_given():
    utc = timezone.utc
    assert (start_ns_from_name('wf_260831_155541_1Hz.npz', utc)
            - start_ns_from_name('wf_260831_155541_1Hz.npz')) == 9 * 3600 * 10 ** 9


def test_a_file_name_without_the_start_time_is_refused(tmp_path):
    path = tmp_path / 'scan.npz'
    write_old(path)
    with pytest.raises(LegacyFileError, match='wf_YYMMDD_HHMMSS_'):
        load_old_file(path)


# --- the converted data ----------------------------------------------------------------

def test_the_waveforms_are_copied_bit_for_bit(old, tmp_path):
    ch0, ch1, _ = old['early'][1]
    convert_old_files([(old['early'][0], 'c')], tmp_path / 'out')
    with np.load(tmp_path / 'out' / 'run_01' / 'data' / 'run01-00.npz') as d:
        assert d['ch0'].dtype == np.float32 and d['ch0'].tobytes() == ch0.tobytes()
        assert d['ch1'].dtype == np.float32 and d['ch1'].tobytes() == ch1.tobytes()


def test_keys_and_dtypes_are_those_of_a_file_written_by_the_daq(old, tmp_path):
    convert_old_files([(old['early'][0], 'c')], tmp_path / 'out')
    with np.load(tmp_path / 'out' / 'run_01' / 'data' / 'run01-00.npz') as mine, \
            np.load(daq_file(tmp_path) / 'data' / 'run01-00.npz') as daq:
        assert sorted(mine.files) == sorted(daq.files)
        for key in daq.files:
            assert mine[key].dtype == daq[key].dtype, key
            assert mine[key].ndim == daq[key].ndim, key


def test_the_event_times_are_the_start_plus_deltat_to_the_nanosecond(old, tmp_path):
    path, (_, _, deltat) = old['early']
    convert_old_files([(path, 'c')], tmp_path / 'out')
    start = start_ns_from_name(path)
    with np.load(tmp_path / 'out' / 'run_01' / 'data' / 'run01-00.npz') as d:
        assert d['run_start_unix_ns'] == start
        assert d['timestamp_unix_ns'].tolist() == [start + delta_ns(x) for x in deltat]
        np.testing.assert_array_equal(d['event_id'], [0, 1, 2])
        assert d['npts'] == NPTS and d['sample_rate'] == SAMPLE_RATE and d['ref_position'] == 20.0


def test_the_converted_file_is_read_by_the_package_and_the_monitor(old, tmp_path, capsys):
    convert_old_files([(old['early'][0], 'c')], tmp_path / 'out')
    raw = load_raw(tmp_path / 'out' / 'run_01' / 'data' / 'run01-00.npz')
    assert raw.nevents == 3 and raw.npts == NPTS and raw.timestamp_unix_ns is not None

    assert monitor_main(['--data-dir', str(tmp_path / 'out'), '--run-number', '1', '--no-show',
                         '--output-dir', str(tmp_path / 'png')]) == 0
    assert (tmp_path / 'png' / 'pc1.png').is_file() and (tmp_path / 'png' / 'pc2.png').is_file()


def test_the_pulse_analysis_example_runs_on_the_converted_file(old, tmp_path, monkeypatch):
    convert_old_files([(old['early'][0], 'c')], tmp_path / 'out')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['02_pulse_analysis.py',
                                      str(tmp_path / 'out' / 'run_01' / 'data' / 'run01-00.npz')])
    runpy.run_path(str(ANALYSIS), run_name='__main__')
    table = np.genfromtxt(tmp_path / 'run01-00_ana.csv', delimiter=',', names=True)
    assert len(table) == 3 and np.all(table['proj_max'] > 0.004)


# --- runs, YAML, summary -----------------------------------------------------------------

def test_runs_are_numbered_in_the_order_of_the_start_times(old, tmp_path):
    converted = convert_old_files([(old['late'][0], 'late'), (old['early'][0], 'early')],
                                  tmp_path / 'out')
    assert [(r.run_number, r.condition, r.source) for r in converted] == [
        (1, 'early', 'wf_260831_154133_49.39Hz.npz'), (2, 'late', 'wf_260831_162120_49.81Hz.npz')]
    assert sorted(p.name for p in (tmp_path / 'out').iterdir()) == ['run_01', 'run_02', 'run_summary.txt']
    assert [r.nevents for r in converted] == [3, 3]


def test_the_yaml_has_the_keys_of_the_daq_yaml_and_only_what_is_known(old, tmp_path):
    convert_old_files([(old['early'][0], 'z-scan, z = 6.15 mm')], tmp_path / 'out')
    mine = yaml.safe_load((tmp_path / 'out' / 'run_01' / 'config' / 'run01-00.yaml').read_text(encoding='utf-8'))
    daq = yaml.safe_load((daq_file(tmp_path) / 'config' / 'run01-00.yaml').read_text(encoding='utf-8'))
    assert set(daq) - set(mine) == ONLY_IN_DAQ_YAML
    assert set(mine) - set(daq) == {'converted_from'}

    start = start_ns_from_name(old['early'][0])
    assert mine['run_number'] == 1 and mine['file_number'] == 0 and mine['nevents'] == 3
    assert mine['condition'] == 'z-scan, z = 6.15 mm' and mine['status'] == 'complete'
    assert mine['daq_rate_hz'] == pytest.approx(49.7) and mine['ref_position'] == 20.0
    assert mine['run_start_time_utc'] == mine['start_time_utc'] == '2026-08-31T06:41:33.000+00:00'  # 15:41:33 JST
    assert start == int(datetime(2026, 8, 31, 6, 41, 33, tzinfo=timezone.utc).timestamp()) * 10 ** 9
    assert mine['converted_from']['file'] == 'wf_260831_154133_49.39Hz.npz'
    assert 'unknown' in str(mine['trigger']) and 'unknown' in str(mine['channels'])
    assert any('+-1 s' in w for w in mine['warnings'])
    assert mine['errors'] == []


def test_the_run_summary_has_one_line_per_run(old, tmp_path):
    convert_old_files([(old['late'][0], 'late'), (old['early'][0], 'early')], tmp_path / 'out')
    lines = (tmp_path / 'out' / 'run_summary.txt').read_text(encoding='utf-8').splitlines()
    assert lines[0].split('\t') == ['run_number', 'start_time', 'stop_time', 'daq_rate_hz', 'condition']
    rows = [line.split('\t') for line in lines[1:]]
    assert [(r[0], r[3], r[4]) for r in rows] == [('1', '49.70', 'early'), ('2', '49.70', 'late')]
    assert rows[0][1].startswith('2026-08-31T') and rows[0][1] < rows[0][2]


def test_the_stop_time_is_the_last_event(old, tmp_path):
    path, (_, _, deltat) = old['early']
    (run,) = convert_old_files([(path, 'c')], tmp_path / 'out')
    assert run.stop_ns == start_ns_from_name(path) + delta_ns(deltat[-1])
    assert run.start_ns == start_ns_from_name(path)


# --- safety ----------------------------------------------------------------------------------

def test_the_old_files_are_never_modified(old, tmp_path):
    before = tree_hashes(old['directory'])
    convert_old_files([(old['late'][0], 'a'), (old['early'][0], 'b')], tmp_path / 'out')
    assert tree_hashes(old['directory']) == before


def test_existing_runs_are_never_overwritten(old, tmp_path):
    items = [(old['late'][0], 'a'), (old['early'][0], 'b')]
    convert_old_files(items, tmp_path / 'out')
    before = tree_hashes(tmp_path / 'out')

    with pytest.raises(FileExistsError, match='run_01'):
        convert_old_files(items, tmp_path / 'out', first_run_number=1)
    with pytest.raises(FileExistsError, match='run_02'):  # the second run collides: nothing is written
        convert_old_files(items, tmp_path / 'out', first_run_number=2)
    assert tree_hashes(tmp_path / 'out') == before


def test_converting_again_continues_the_numbering(old, tmp_path):
    convert_old_files([(old['early'][0], 'a')], tmp_path / 'out')
    (run,) = convert_old_files([(old['late'][0], 'b')], tmp_path / 'out')
    assert run.run_number == 2


def test_a_file_that_is_not_from_the_old_daq_is_refused_and_nothing_is_written(old, tmp_path):
    bad = old['directory'] / 'wf_260831_160000_50.00Hz.npz'
    np.savez(bad, ch0=np.zeros((2, 10), np.float32))
    with pytest.raises(LegacyFileError, match='missing key'):
        convert_old_files([(old['early'][0], 'a'), (bad, 'b')], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_event_times_that_go_backwards_are_refused(tmp_path):
    path = tmp_path / 'wf_260831_160000_50.00Hz.npz'
    ch = np.zeros((2, 10), np.float32)
    np.savez(path, ch0=ch, ch1=ch, npts=np.int32(10), sample_rate=np.float64(1e9),
             ref_position=np.int32(20), daq_rate=np.float64(1.0),
             deltat=np.array([timedelta(seconds=2), timedelta(seconds=1)], dtype=object))
    with pytest.raises(LegacyFileError, match='not increasing'):
        load_old_file(path)


# --- scripts/make_sample_data.py ---------------------------------------------------------------

@pytest.fixture(scope='module')
def make_sample_data():
    spec = importlib.util.spec_from_file_location('make_sample_data', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_old_directory(module, root, skip=None, extra=None):
    """Tiny old files named like the nine z-scan files (a different second in each pattern)."""
    directory = root / 'Aug31st'
    directory.mkdir(parents=True)
    for i, (pattern, _) in enumerate(module.SCANS):
        if pattern == skip:
            continue
        name = pattern.split('/')[1].replace('?', '7').replace('*', '49.50Hz')
        write_old(directory / name, nevents=2, npts=100, seed=i, microseconds=(5, 250000))
    if extra:
        write_old(directory / extra, nevents=2, npts=100)
    return root


def test_the_script_makes_nine_runs_in_time_order_with_the_z_positions(make_sample_data, tmp_path):
    old_dir = fake_old_directory(make_sample_data, tmp_path / 'data')
    out = tmp_path / 'sample'
    make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])

    names = sorted(p.name for p in out.iterdir())
    assert names == ['README.md', *(f'run_0{n}' for n in range(1, 10)), 'run_summary.txt']
    rows = [line.split('\t') for line in (out / 'run_summary.txt').read_text(encoding='utf-8').splitlines()[1:]]
    assert [(r[0], r[4]) for r in rows] == [
        ('1', 'z-scan at x = 4.0 mm, z = 6.15 mm'), ('2', 'z-scan at x = 4.0 mm, z = 6.00 mm'),
        ('3', 'z-scan at x = 4.0 mm, z = 5.85 mm'), ('4', 'z-scan at x = 4.0 mm, z = 5.70 mm'),
        ('5', 'z-scan at x = 4.0 mm, z = 6.90 mm'), ('6', 'z-scan at x = 4.0 mm, z = 6.70 mm'),
        ('7', 'z-scan at x = 4.0 mm, z = 6.50 mm'), ('8', 'z-scan at x = 4.0 mm, z = 6.30 mm'),
        ('9', 'z-scan at x = 4.0 mm, z = 5.50 mm')]
    for n in range(1, 10):
        assert load_raw(out / f'run_0{n}' / 'data' / f'run0{n}-00.npz').nevents == 2

    readme = (out / 'README.md').read_text(encoding='utf-8')
    assert '| 1 | 6.15 |' in readme and '| 9 | 5.50 |' in readme and 'UTC+9' in readme and '±1 秒' in readme
    assert 'kidpack-monitor --data-dir sample_data --run-number 1' in readme


def test_the_script_does_not_overwrite_an_existing_sample_data(make_sample_data, tmp_path):
    old_dir = fake_old_directory(make_sample_data, tmp_path / 'data')
    out = tmp_path / 'sample'
    make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])
    before = tree_hashes(out)
    with pytest.raises(FileExistsError):
        make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])
    assert tree_hashes(out) == before


def test_the_script_wants_exactly_one_file_per_pattern(make_sample_data, tmp_path):
    missing = fake_old_directory(make_sample_data, tmp_path / 'a', skip=make_sample_data.SCANS[0][0])
    with pytest.raises(SystemExit, match=r'matches 0 files'):
        make_sample_data.main(['--old-data-dir', str(missing), '--output-dir', str(tmp_path / 'o1')])

    twice = fake_old_directory(make_sample_data, tmp_path / 'b', extra='wf_260831_155543_49.00Hz.npz')
    with pytest.raises(SystemExit, match=r'matches 2 files'):
        make_sample_data.main(['--old-data-dir', str(twice), '--output-dir', str(tmp_path / 'o2')])
    assert not (tmp_path / 'o1').exists() and not (tmp_path / 'o2').exists()
