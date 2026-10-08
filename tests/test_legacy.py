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
from kidpack.legacy import (LegacyFileError, convert_old_files, label_from_name, load_old_file,
                            start_ns_from_name)
from kidpack.monitor.cli import main as monitor_main
from kidpack.rawdata import load_raw

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'make_sample_data.py'
ANALYSIS = ROOT / 'examples' / '02_pulse_analysis.py'

NPTS = 5000
SAMPLE_RATE = 2.5e9
# the DAQ writes the YAML keys of a hardware run too; of these a converted run knows nothing
ONLY_IN_DAQ_YAML = {'actual_settings', 'backend', 'resource', 'fetch_timeout_s', 'sg', 'command_line'}


EARLY, LATE = '0831_154133', '0831_162120'  # the runs of the two files of the `old` fixture


def npz(out, label=EARLY):
    return Path(out) / f'run_{label}' / 'data' / f'run{label}-00.npz'


def yaml_of(out, label=EARLY):
    return Path(out) / f'run_{label}' / 'config' / f'run{label}-00.yaml'


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


def test_the_run_is_named_after_the_start_in_the_file_name():
    assert label_from_name('/x/wf_260831_155251_49.55Hz.npz') == '0831_155251'
    assert label_from_name('wf_251231_235959_1.0Hz.npz') == '1231_235959'  # the year is not in the name
    assert label_from_name('wf_260101_000000_1.0Hz.npz') == '0101_000000'


@pytest.mark.parametrize('name', ['wf_260231_120000_1Hz.npz',  # no 31st of February
                                  'wf_261301_120000_1Hz.npz',  # no 13th month
                                  'wf_260831_246000_1Hz.npz',  # no 24:60:00
                                  'wf_260831_120060_1Hz.npz',  # no second 60
                                  'scan.npz'])
def test_a_file_name_that_is_not_a_start_time_gives_no_run_name(name):
    with pytest.raises(LegacyFileError):
        label_from_name(name)


# --- the converted data ----------------------------------------------------------------

def test_the_waveforms_are_copied_bit_for_bit(old, tmp_path):
    ch0, ch1, _ = old['early'][1]
    convert_old_files([(old['early'][0], 'c')], tmp_path / 'out')
    with np.load(npz(tmp_path / 'out')) as d:
        assert d['ch0'].dtype == np.float32 and d['ch0'].tobytes() == ch0.tobytes()
        assert d['ch1'].dtype == np.float32 and d['ch1'].tobytes() == ch1.tobytes()


def test_keys_and_dtypes_are_those_of_a_file_written_by_the_daq(old, tmp_path):
    convert_old_files([(old['early'][0], 'c')], tmp_path / 'out')
    with np.load(npz(tmp_path / 'out')) as mine, \
            np.load(daq_file(tmp_path) / 'data' / 'run01-00.npz') as daq:
        assert sorted(mine.files) == sorted(daq.files)
        for key in daq.files:
            assert mine[key].dtype == daq[key].dtype, key
            assert mine[key].ndim == daq[key].ndim, key


def test_the_event_times_are_the_start_plus_deltat_to_the_nanosecond(old, tmp_path):
    path, (_, _, deltat) = old['early']
    convert_old_files([(path, 'c')], tmp_path / 'out')
    start = start_ns_from_name(path)
    with np.load(npz(tmp_path / 'out')) as d:
        assert d['run_start_unix_ns'] == start
        assert d['timestamp_unix_ns'].tolist() == [start + delta_ns(x) for x in deltat]
        np.testing.assert_array_equal(d['event_id'], [0, 1, 2])
        assert d['npts'] == NPTS and d['sample_rate'] == SAMPLE_RATE and d['ref_position'] == 20.0


def test_the_converted_file_is_read_by_the_package_and_the_monitor(old, tmp_path, capsys):
    convert_old_files([(old['early'][0], 'c')], tmp_path / 'out')
    raw = load_raw(npz(tmp_path / 'out'))
    assert raw.nevents == 3 and raw.npts == NPTS and raw.timestamp_unix_ns is not None

    assert monitor_main(['--data-dir', str(tmp_path / 'out'), '--run-number', EARLY, '--no-show',
                         '--output-dir', str(tmp_path / 'png')]) == 0
    assert (tmp_path / 'png' / 'pc1.png').is_file() and (tmp_path / 'png' / 'pc2.png').is_file()


def test_the_pulse_analysis_example_runs_on_the_converted_file(old, tmp_path, monkeypatch):
    convert_old_files([(old['early'][0], 'c')], tmp_path / 'out')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['02_pulse_analysis.py',
                                      str(npz(tmp_path / 'out'))])
    runpy.run_path(str(ANALYSIS), run_name='__main__')
    table = np.genfromtxt(tmp_path / f'run{EARLY}-00_ana.csv', delimiter=',', names=True)
    assert len(table) == 3 and np.all(table['proj_max'] > 0.004)


# --- runs, YAML, summary -----------------------------------------------------------------

def test_runs_are_named_after_the_start_and_written_in_the_order_of_the_start_times(old, tmp_path):
    converted = convert_old_files([(old['late'][0], 'late'), (old['early'][0], 'early')],
                                  tmp_path / 'out')
    assert [(r.label, r.condition, r.source) for r in converted] == [
        (EARLY, 'early', 'wf_260831_154133_49.39Hz.npz'), (LATE, 'late', 'wf_260831_162120_49.81Hz.npz')]
    assert sorted(p.name for p in (tmp_path / 'out').iterdir()) == [
        f'run_{EARLY}', f'run_{LATE}', 'run_summary.txt']
    assert npz(tmp_path / 'out', LATE).is_file() and yaml_of(tmp_path / 'out', LATE).is_file()
    assert [r.nevents for r in converted] == [3, 3]


def test_the_yaml_has_the_keys_of_the_daq_yaml_and_only_what_is_known(old, tmp_path):
    convert_old_files([(old['early'][0], 'z-scan, z = 6.15 mm')], tmp_path / 'out')
    mine = yaml.safe_load(yaml_of(tmp_path / 'out').read_text(encoding='utf-8'))
    daq = yaml.safe_load((daq_file(tmp_path) / 'config' / 'run01-00.yaml').read_text(encoding='utf-8'))
    assert set(daq) - set(mine) == ONLY_IN_DAQ_YAML
    assert set(mine) - set(daq) == {'converted_from'}

    start = start_ns_from_name(old['early'][0])
    assert mine['run_number'] == EARLY and mine['file_number'] == 0 and mine['nevents'] == 3
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
    assert [(r[0], r[3], r[4]) for r in rows] == [(EARLY, '49.70', 'early'), (LATE, '49.70', 'late')]
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

    with pytest.raises(FileExistsError, match=f'run_{EARLY}'):
        convert_old_files(items, tmp_path / 'out')
    assert tree_hashes(tmp_path / 'out') == before


def test_a_run_that_exists_stops_the_conversion_before_anything_is_written(old, tmp_path):
    convert_old_files([(old['early'][0], 'a')], tmp_path / 'out')
    before = tree_hashes(tmp_path / 'out')
    with pytest.raises(FileExistsError, match=f'run_{EARLY}'):  # the later file would be new, the earlier one not
        convert_old_files([(old['late'][0], 'b'), (old['early'][0], 'c')], tmp_path / 'out')
    assert tree_hashes(tmp_path / 'out') == before


def test_the_name_of_a_run_does_not_depend_on_what_else_is_converted(old, tmp_path):
    (late,) = convert_old_files([(old['late'][0], 'b')], tmp_path / 'out')
    (early,) = convert_old_files([(old['early'][0], 'a')], tmp_path / 'out')  # later, but an earlier start
    assert (late.label, early.label) == (LATE, EARLY)
    rows = [line.split('\t') for line in (tmp_path / 'out' / 'run_summary.txt').read_text(encoding='utf-8').splitlines()[1:]]
    assert [r[0] for r in rows] == [LATE, EARLY]  # the summary is in the order of conversion


def test_two_files_that_start_in_the_same_second_are_refused(old, tmp_path):
    twin = old['directory'] / 'wf_260831_154133_50.00Hz.npz'  # same second, another rate in the name
    write_old(twin, seed=3)
    with pytest.raises(LegacyFileError, match=f'{EARLY}: more than one file'):
        convert_old_files([(old['early'][0], 'a'), (twin, 'b')], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


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
    """Tiny old files named like the 20 z-scan / x-scan files (a different second in each pattern)."""
    directory = root / 'Aug31st'
    directory.mkdir(parents=True)
    patterns = {pattern for _, _, entries in module.SCANS for pattern, _ in entries}  # a file is in 1-2 scans
    for i, pattern in enumerate(sorted(patterns)):
        if pattern == skip:
            continue
        name = pattern.split('/')[1].replace('?', '7').replace('*', '49.50Hz')
        write_old(directory / name, nevents=2, npts=100, seed=i, microseconds=(5, 250000))
    if extra:
        write_old(directory / extra, nevents=2, npts=100)
    return root


# (name of the run, condition) in the order of the start times: the specification of sample_data/, for
# the fake files above (their seconds are the digit 7 where the pattern has a ?). The 6.30 mm file of
# the z-scan at x = 3.5 mm is also the 3.50 mm file of the x-scan, and is one run.
EXPECTED_RUNS = [
    ('0831_153857', 'x-scan at z = 6.30 mm, x = 4.00 mm'),
    ('0831_154137', 'z-scan at x = 4.0 mm, z = 6.15 mm'),
    ('0831_154217', 'z-scan at x = 4.0 mm, z = 6.00 mm'),
    ('0831_154257', 'z-scan at x = 4.0 mm, z = 5.85 mm'),
    ('0831_154347', 'z-scan at x = 4.0 mm, z = 5.70 mm'),
    ('0831_154547', 'x-scan at z = 6.30 mm, x = 3.65 mm'),
    ('0831_154747', 'z-scan at x = 3.5 mm, z = 6.90 mm'),
    ('0831_154827', 'z-scan at x = 3.5 mm, z = 6.70 mm'),
    ('0831_154907', 'z-scan at x = 3.5 mm, z = 6.50 mm'),
    ('0831_154947', 'z-scan at x = 3.5 mm, z = 6.30 mm; x-scan at z = 6.30 mm, x = 3.50 mm'),
    ('0831_155027', 'z-scan at x = 3.5 mm, z = 6.00 mm'),
    ('0831_155147', 'z-scan at x = 3.5 mm, z = 5.70 mm'),
    ('0831_155257', 'x-scan at z = 6.30 mm, x = 3.20 mm'),
    ('0831_155427', 'x-scan at z = 6.30 mm, x = 4.50 mm'),
    ('0831_155547', 'z-scan at x = 4.0 mm, z = 6.90 mm'),
    ('0831_155627', 'z-scan at x = 4.0 mm, z = 6.70 mm'),
    ('0831_155707', 'z-scan at x = 4.0 mm, z = 6.50 mm'),
    ('0831_155757', 'z-scan at x = 4.0 mm, z = 6.30 mm'),
    ('0831_162037', 'z-scan at x = 3.5 mm, z = 5.50 mm'),
    ('0831_162127', 'z-scan at x = 4.0 mm, z = 5.50 mm'),
]


def test_the_script_makes_twenty_runs_named_after_the_start_in_time_order(make_sample_data, tmp_path):
    old_dir = fake_old_directory(make_sample_data, tmp_path / 'data')
    out = tmp_path / 'sample'
    make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])

    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(['README.md', 'run_summary.txt', *(f'run_{label}' for label, _ in EXPECTED_RUNS)])
    rows = [line.split('\t') for line in (out / 'run_summary.txt').read_text(encoding='utf-8').splitlines()[1:]]
    assert [(r[0], r[4]) for r in rows] == EXPECTED_RUNS  # one line per run, in the order of the start times
    for label, _ in EXPECTED_RUNS:
        assert load_raw(npz(out, label)).nevents == 2
        assert yaml.safe_load(yaml_of(out, label).read_text(encoding='utf-8'))['run_number'] == label


def test_the_file_in_two_scans_is_one_run_with_both_conditions(make_sample_data, tmp_path):
    old_dir = fake_old_directory(make_sample_data, tmp_path / 'data')
    make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(tmp_path / 'sample')])
    sources = [yaml.safe_load(p.read_text(encoding='utf-8'))['converted_from']['file']
               for p in sorted((tmp_path / 'sample').glob('run_*/config/*.yaml'))]
    assert len(sources) == len(set(sources)) == 20  # 9 + 7 + 5 entries, 21 patterns, one file twice
    shared = yaml.safe_load(yaml_of(tmp_path / 'sample', '0831_154947').read_text(encoding='utf-8'))
    assert shared['converted_from']['file'] == 'wf_260831_154947_49.50Hz.npz'
    assert shared['condition'] == EXPECTED_RUNS[9][1]


def test_the_readme_lists_the_runs_and_the_caveats(make_sample_data, tmp_path):
    old_dir = fake_old_directory(make_sample_data, tmp_path / 'data')
    out = tmp_path / 'sample'
    make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])
    readme = (out / 'README.md').read_text(encoding='utf-8')
    for label, condition in EXPECTED_RUNS:
        assert f'| {label} | {condition} |' in readme
    assert 'run_0831_155253/data/run0831_155253-00.npz' in readme  # the rule, with an example
    assert 'UTC+9' in readme and '±1 秒' in readme and '測定 20 点' in readme
    assert 'kidpack-monitor --data-dir sample_data --run-number 0831_153857' in readme
    assert 'sample_data/run_0831_153857/data/run0831_153857-00.npz' in readme


def test_the_runs_of_the_script_can_be_chosen_by_name_in_the_monitor(make_sample_data, tmp_path, capsys):
    old_dir = fake_old_directory(make_sample_data, tmp_path / 'data')
    out = tmp_path / 'sample'
    make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])
    assert monitor_main(['--data-dir', str(out), '--run-number', '0831_154947', '--file-number', '0',
                         '--no-show', '--output-dir', str(tmp_path / 'png')]) == 0
    assert 'run0831_154947-00.npz' in capsys.readouterr().out
    assert monitor_main(['--data-dir', str(out), '--run-number', '0831_000000', '--no-show',
                         '--output-dir', str(tmp_path / 'png')]) == 1  # a run that does not exist


def test_the_script_does_not_overwrite_an_existing_sample_data(make_sample_data, tmp_path):
    old_dir = fake_old_directory(make_sample_data, tmp_path / 'data')
    out = tmp_path / 'sample'
    make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])
    before = tree_hashes(out)
    with pytest.raises(FileExistsError):
        make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])
    assert tree_hashes(out) == before


def test_a_partly_existing_output_is_refused_before_anything_is_written(make_sample_data, tmp_path):
    old_dir = fake_old_directory(make_sample_data, tmp_path / 'data')
    out = tmp_path / 'sample'
    (out / 'run_0831_162127').mkdir(parents=True)  # the last run: the others would be written first
    with pytest.raises(FileExistsError, match='run_0831_162127'):
        make_sample_data.main(['--old-data-dir', str(old_dir), '--output-dir', str(out)])
    assert sorted(p.name for p in out.iterdir()) == ['run_0831_162127']


def test_the_script_wants_exactly_one_file_per_pattern(make_sample_data, tmp_path):
    first_pattern = make_sample_data.SCANS[0][2][0][0]
    missing = fake_old_directory(make_sample_data, tmp_path / 'a', skip=first_pattern)
    with pytest.raises(SystemExit, match=r'matches 0 files'):
        make_sample_data.main(['--old-data-dir', str(missing), '--output-dir', str(tmp_path / 'o1')])

    twice = fake_old_directory(make_sample_data, tmp_path / 'b', extra='wf_260831_155543_49.00Hz.npz')
    with pytest.raises(SystemExit, match=r'matches 2 files'):
        make_sample_data.main(['--old-data-dir', str(twice), '--output-dir', str(tmp_path / 'o2')])
    assert not (tmp_path / 'o1').exists() and not (tmp_path / 'o2').exists()
