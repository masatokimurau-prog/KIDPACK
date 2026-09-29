import os
from datetime import datetime

import numpy as np
import pytest
import yaml

from kidpack.daq.backends.base import DaqError
from kidpack.daq.backends.simulator import SimulatedScope, SimulatedSg
from kidpack.daq.cli import main
from kidpack.daq.config import DaqConfig, SgConfig, TriggerConfig
from kidpack.daq.runner import run_daq
from kidpack.daq.summary import HEADER, append_run_summary

NPZ_KEYS = {'ch0', 'ch1', 'event_id', 'timestamp_unix_ns', 'npts', 'sample_rate',
            'ref_position', 'run_start_unix_ns'}
YAML_KEYS = {'run_number', 'file_number', 'events_per_file', 'nevents', 'start_time_utc',
             'stop_time_utc', 'daq_software_version', 'npts', 'sample_rate', 'ref_position',
             'trigger', 'status', 'errors', 'warnings'}


def make_cfg(tmp_path, **kw):
    params = dict(run_number=7, events_per_file=5, num_files=3, output_dir=str(tmp_path),
                  condition='temp 5.5K, lna 1.9V', npts=100)
    params.update(kw)
    return DaqConfig(**params)


class ScriptedScope(SimulatedScope):
    """Simulator that raises at the n-th acquire()."""

    def __init__(self, cfg, interrupt_at=None, fail_at=None):
        super().__init__(cfg, seed=0)
        self.interrupt_at, self.fail_at = interrupt_at, fail_at
        self.count = 0
        self.closed = False

    def acquire(self):
        self.count += 1
        if self.count == self.interrupt_at:
            raise KeyboardInterrupt
        if self.count == self.fail_at:
            raise DaqError('boom (trigger timeout)')
        return super().acquire()

    def close(self):
        self.closed = True


def load(run_dir, stem):
    with np.load(os.path.join(run_dir, 'data', f'{stem}.npz')) as d:
        arrays = {k: d[k] for k in d.files}
    with open(os.path.join(run_dir, 'config', f'{stem}.yaml')) as f:
        return arrays, yaml.safe_load(f)


def summary_rows(cfg):
    with open(cfg.summary_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines[0].split('\t'), [line.split('\t') for line in lines[1:]]


def test_full_run_layout_and_contract(tmp_path):
    cfg = make_cfg(tmp_path)
    result = run_daq(cfg, SimulatedScope(cfg, seed=1))

    run_dir = tmp_path / 'run_07'
    assert result.status == 'completed' and result.exit_code == 0
    assert result.total_events == 15 and result.files_written == 3
    assert sorted(os.listdir(run_dir)) == ['config', 'data', 'logs']
    assert sorted(os.listdir(run_dir / 'data')) == [f'run07-0{i}.npz' for i in range(3)]
    assert sorted(os.listdir(run_dir / 'config')) == [f'run07-0{i}.yaml' for i in range(3)]
    assert (run_dir / 'logs' / 'daq.log').stat().st_size > 0

    run_starts = set()
    for i in range(3):
        arrays, meta = load(str(run_dir), f'run07-0{i}')
        assert set(arrays) == NPZ_KEYS
        assert arrays['ch0'].dtype == arrays['ch1'].dtype == np.float32
        assert arrays['ch0'].shape == arrays['ch1'].shape == (5, 100)
        assert arrays['event_id'].dtype == np.uint32
        np.testing.assert_array_equal(arrays['event_id'], np.arange(5))  # resets per file
        assert arrays['timestamp_unix_ns'].dtype == np.int64
        assert np.all(np.diff(arrays['timestamp_unix_ns']) >= 0)
        assert int(arrays['npts']) == 100
        assert float(arrays['sample_rate']) == 2.5e9
        assert float(arrays['ref_position']) == 20.0
        run_starts.add(int(arrays['run_start_unix_ns']))
        assert int(arrays['run_start_unix_ns']) <= arrays['timestamp_unix_ns'][0]

        assert YAML_KEYS <= set(meta)
        assert (meta['run_number'], meta['file_number'], meta['nevents']) == (7, i, 5)
        assert meta['events_per_file'] == 5 and meta['status'] == 'complete'
        assert meta['condition'] == 'temp 5.5K, lna 1.9V'
        assert meta['trigger']['source'] == 'VAL_EXTERNAL'
        assert meta['start_time_utc'] <= meta['stop_time_utc']
        assert meta['run_start_time_utc'] <= meta['start_time_utc']
        assert meta['sg'] == {'controlled_by_daq': False}
    assert len(run_starts) == 1  # every file carries the same run start time


def test_summary_line_per_run_appended_to_one_file(tmp_path):
    for run_number in (7, 8):
        cfg = make_cfg(tmp_path, run_number=run_number, num_files=1,
                       condition=f'condition of run {run_number}')
        run_daq(cfg, SimulatedScope(cfg, seed=1))

    header, rows = summary_rows(cfg)
    assert header == list(HEADER)
    assert [r[0] for r in rows] == ['7', '8']
    assert [r[4] for r in rows] == ['condition of run 7', 'condition of run 8']
    for _, start, stop, rate, _ in rows:
        assert datetime.fromisoformat(start) <= datetime.fromisoformat(stop)
        assert float(rate) > 0


def test_summary_condition_is_kept_on_one_line(tmp_path):
    path = str(tmp_path / 's.txt')
    append_run_summary(path, 1, 0, 10**9, None, 'a\tb\nc\r\nd')
    header, rows = summary_rows(type('C', (), {'summary_path': path}))
    assert rows == [['1', rows[0][1], rows[0][2], '', 'a b c  d']]


def test_existing_run_directory_is_never_reused(tmp_path):
    cfg = make_cfg(tmp_path, num_files=1)
    run_daq(cfg, SimulatedScope(cfg, seed=1))
    before = sorted(os.listdir(tmp_path / 'run_07' / 'data'))
    scope = ScriptedScope(cfg)
    with pytest.raises(FileExistsError):
        run_daq(cfg, scope)
    assert scope.closed and scope.count == 0
    assert sorted(os.listdir(tmp_path / 'run_07' / 'data')) == before
    assert len(summary_rows(cfg)[1]) == 1


def test_ctrl_c_keeps_the_partial_file_and_still_writes_the_summary(tmp_path):
    cfg = make_cfg(tmp_path)
    scope = ScriptedScope(cfg, interrupt_at=8)  # 3rd event of file 01
    sg = SimulatedSg()
    cfg.sg = SgConfig(frequency=5.49e9, power=-12.0)
    result = run_daq(cfg, scope, sg)

    assert result.status == 'interrupted' and result.exit_code == 130
    assert result.total_events == 7 and result.files_written == 2
    assert sorted(os.listdir(tmp_path / 'run_07' / 'data')) == ['run07-00.npz', 'run07-01.npz']
    arrays, meta = load(str(tmp_path / 'run_07'), 'run07-01')
    assert arrays['ch0'].shape == (2, 100)
    np.testing.assert_array_equal(arrays['event_id'], [0, 1])
    assert meta['status'] == 'interrupted' and meta['nevents'] == 2
    assert load(str(tmp_path / 'run_07'), 'run07-00')[1]['status'] == 'complete'
    assert sg.calls == [('start', 5.49e9, -12.0), ('stop',)]
    assert scope.closed
    assert len(summary_rows(cfg)[1]) == 1


def test_ctrl_c_before_any_event_of_a_file_writes_no_empty_file(tmp_path):
    cfg = make_cfg(tmp_path)
    result = run_daq(cfg, ScriptedScope(cfg, interrupt_at=6))  # first event of file 01
    assert result.status == 'interrupted' and result.total_events == 5
    assert os.listdir(tmp_path / 'run_07' / 'data') == ['run07-00.npz']


def test_device_error_is_recorded_and_data_kept(tmp_path):
    cfg = make_cfg(tmp_path)
    sg = SimulatedSg()
    cfg.sg = SgConfig(frequency=5.49e9, power=-12.0)
    result = run_daq(cfg, ScriptedScope(cfg, fail_at=3), sg)

    assert result.status == 'error' and result.exit_code == 1
    assert 'boom' in result.error
    arrays, meta = load(str(tmp_path / 'run_07'), 'run07-00')
    assert arrays['ch0'].shape == (2, 100)
    assert meta['status'] == 'error' and 'boom' in meta['errors'][0]
    assert sg.calls[-1] == ('stop',)  # SG is stopped even on errors
    assert len(summary_rows(cfg)[1]) == 1


def test_sg_settings_are_recorded_when_controlled_by_the_daq(tmp_path):
    cfg = make_cfg(tmp_path, num_files=1, sg=SgConfig(5.49e9, -12.0, 'PXI1Slot3'))
    sg = SimulatedSg()
    run_daq(cfg, SimulatedScope(cfg, seed=1), sg)
    assert sg.calls == [('start', 5.49e9, -12.0), ('stop',)]
    meta = load(str(tmp_path / 'run_07'), 'run07-00')[1]
    assert meta['sg'] == {'controlled_by_daq': True, 'frequency_hz': 5.49e9,
                          'power_dbm': -12.0, 'resource': 'PXI1Slot3'}


def test_random_trigger_run_records_one_event_per_interval(tmp_path):
    cfg = make_cfg(tmp_path, events_per_file=4, num_files=2,
                   trigger=TriggerConfig(mode='random', interval=0.03))
    result = run_daq(cfg, SimulatedScope(cfg, seed=1))

    assert result.status == 'completed' and result.total_events == 8
    stamps = np.concatenate([load(str(tmp_path / 'run_07'), f'run07-0{i}')[0]['timestamp_unix_ns']
                             for i in range(2)])
    gaps = np.diff(stamps) / 1e9
    # Triggers sit on a fixed grid: a single gap jitters with the OS timer (a late
    # wake-up shortens the next gap) but the total follows the grid without drift.
    assert np.all((gaps > 0.01) & (gaps < 0.09))
    assert stamps[-1] - stamps[0] == pytest.approx(7 * 0.03e9, abs=0.03e9)
    meta = load(str(tmp_path / 'run_07'), 'run07-00')[1]
    assert meta['trigger'] == {'mode': 'random', 'source': 'software', 'interval_s': 0.03}
    assert 0.8 / 0.03 < result.daq_rate_hz < 1.25 / 0.03  # ~ 1 event per interval


def test_edge_trigger_is_recorded_in_the_yaml(tmp_path):
    cfg = make_cfg(tmp_path, num_files=1)
    run_daq(cfg, SimulatedScope(cfg, seed=1))
    assert load(str(tmp_path / 'run_07'), 'run07-00')[1]['trigger'] == {
        'mode': 'edge', 'source': 'VAL_EXTERNAL', 'level': 2.2,
        'slope': 'positive', 'coupling': 'lf_reject'}


def test_actual_device_settings_replace_the_requested_ones_in_the_data_file(tmp_path):
    class CoercingScope(SimulatedScope):
        def configure(self):
            return {'sample_rate': 2.4e9, 'ref_position': 25.0}

    cfg = make_cfg(tmp_path, num_files=1)
    run_daq(cfg, CoercingScope(cfg, seed=1))
    arrays, meta = load(str(tmp_path / 'run_07'), 'run07-00')
    assert float(arrays['sample_rate']) == 2.4e9 and float(arrays['ref_position']) == 25.0
    assert meta['actual_settings'] == {'sample_rate': 2.4e9, 'ref_position': 25.0}


# --- command line ------------------------------------------------------------

def cli_args(tmp_path, *extra, run='3'):
    return ['--backend', 'simulator', '--run-number', run, '--events-per-file', '4',
            '--num-files', '2', '--npts', '64', '--output-dir', str(tmp_path), *extra]


def test_cli_end_to_end_with_simulator(tmp_path):
    assert main(cli_args(tmp_path, '--condition', 'x\ty',
                         '--sg-frequency', '5.49e9', '--sg-power', '-12')) == 0
    arrays, meta = load(str(tmp_path / 'run_03'), 'run03-01')
    assert arrays['ch0'].shape == (4, 64)
    assert meta['sg']['controlled_by_daq'] is True
    assert meta['command_line'][0] == 'kidpack-daq' and '--run-number' in meta['command_line']
    lines = (tmp_path / 'run_summary.txt').read_text().splitlines()
    assert len(lines) == 2 and lines[1].split('\t')[0] == '3' and lines[1].endswith('x y')


def test_cli_refuses_an_existing_run_number(tmp_path, capsys):
    assert main(cli_args(tmp_path)) == 0
    assert main(cli_args(tmp_path)) == 1
    assert 'already exists' in capsys.readouterr().err
    assert len((tmp_path / 'run_summary.txt').read_text().splitlines()) == 2  # header + 1 run


def test_cli_reports_a_missing_driver_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'niscope', None)  # import raises ImportError
    args = cli_args(tmp_path)
    args[args.index('--backend') + 1] = 'niscope'
    assert main(args) == 1
    assert 'could not set up the niscope backend' in capsys.readouterr().err
    assert not (tmp_path / 'run_03').exists()  # a failed start must not burn the run number
