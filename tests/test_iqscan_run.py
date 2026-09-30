import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import yaml

from kidpack.daq.backends.base import DaqError
from kidpack.iqscan.backends import SimulatedIqScope, SimulatedSweepSg
from kidpack.iqscan.cli import main
from kidpack.iqscan.config import IqScanConfig
from kidpack.iqscan.runner import run_iqscan
from kidpack.iqscan.summary import HEADER

F0, F1, N = 5.324e9, 5.328e9, 21


def make_cfg(tmp_path, **kw):
    params = dict(f_start=F0, f_stop=F1, num_points=N, output_dir=str(tmp_path), name='scan',
                  condition='temp 5.5K, lna 1.9V', power=-12.0)
    params.update(kw)
    return IqScanConfig(**params)


def bench(cfg, seed=1, scope_class=SimulatedIqScope, **kw):
    sg = SimulatedSweepSg()
    return scope_class(cfg, sg, seed=seed, **kw), sg


class ScriptedScope(SimulatedIqScope):
    """Simulator that fails at the n-th measure()."""

    def __init__(self, cfg, sg, seed=None, interrupt_at=None, fail_at=None):
        super().__init__(cfg, sg, seed)
        self.interrupt_at, self.fail_at = interrupt_at, fail_at
        self.count = 0
        self.closed = False

    def measure(self):
        self.count += 1
        if self.count == self.interrupt_at:
            raise KeyboardInterrupt
        if self.count == self.fail_at:
            raise DaqError('boom (acquisition timeout)')
        return super().measure()

    def close(self):
        self.closed = True


def load(tmp_path, stem='scan'):
    with np.load(tmp_path / f'{stem}.npz') as d:
        arrays = {k: d[k] for k in d.files}
    with open(tmp_path / f'{stem}.yaml') as f:
        return arrays, yaml.safe_load(f)


def summary_rows(cfg):
    with open(cfg.summary_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines[0].split('\t'), [line.split('\t') for line in lines[1:]]


def test_full_scan_writes_the_old_dd_layout_plus_extras(tmp_path):
    cfg = make_cfg(tmp_path)
    scope, sg = bench(cfg)
    result = run_iqscan(cfg, scope, sg)

    assert result.status == 'completed' and result.exit_code == 0 and result.points_measured == N
    assert sorted(os.listdir(tmp_path)) == ['iqscan_summary.txt', 'scan.npz', 'scan.yaml']
    arrays, meta = load(tmp_path)
    dd = arrays['dd']
    assert dd.shape == (N, 3) and dd.dtype == np.float64  # like the old np.savez(dd=...)
    np.testing.assert_array_equal(dd[:, 0], np.linspace(F0, F1, N))  # column 0: frequency [Hz]
    # columns 1 and 2: mean ch0 (I) and ch1 (Q) in volts, here the notch resonator model
    model = np.array([SimulatedIqScope.AMPLITUDE * 10 ** ((-12 + 10) / 20) *
                      scope.transmission(f) for f in dd[:, 0]])
    np.testing.assert_allclose(dd[:, 1], model.real, atol=5 * 2e-3 / np.sqrt(10000))
    np.testing.assert_allclose(dd[:, 2], model.imag, atol=5 * 2e-3 / np.sqrt(10000))
    assert np.argmin(np.hypot(dd[:, 1], dd[:, 2])) == N // 2  # resonance in the middle

    n_samples = 5000 * 2
    assert set(arrays) == {'dd', 'ch0_std', 'ch1_std', 'ch0_stderr', 'ch1_stderr', 'n_samples',
                           'timestamp_unix_ns', 'scan_start_unix_ns', 'power_dbm', 'sample_rate',
                           'npts', 'num_records'}
    assert (arrays['n_samples'] == n_samples).all()
    np.testing.assert_allclose(arrays['ch0_stderr'], arrays['ch0_std'] / np.sqrt(n_samples))
    assert arrays['timestamp_unix_ns'].dtype == np.int64
    assert np.all(np.diff(arrays['timestamp_unix_ns']) >= 0)
    assert int(arrays['scan_start_unix_ns']) <= arrays['timestamp_unix_ns'][0]
    assert float(arrays['power_dbm']) == -12.0 and float(arrays['sample_rate']) == 1e4
    assert int(arrays['npts']) == 5000 and int(arrays['num_records']) == 2

    assert meta['status'] == 'completed' and meta['errors'] == []
    assert meta['condition'] == 'temp 5.5K, lna 1.9V'
    assert meta['scan'] == {'f_start_hz': F0, 'f_stop_hz': F1, 'num_points_requested': N,
                            'num_points_measured': N, 'power_dbm': -12.0, 'settle_time_s': 0.0}
    assert meta['digitizer']['trigger'] == 'immediate' and meta['digitizer']['num_records'] == 2
    assert meta['signal_generator'] == {'resource': 'PXI1Slot3'}
    assert meta['start_time_utc'] <= meta['stop_time_utc']


def test_the_file_is_readable_by_the_old_analysis_scripts(tmp_path):
    cfg = make_cfg(tmp_path)
    run_iqscan(cfg, *bench(cfg))
    # exactly what analysis/ana_iqscan_tscan.py does with an iq_scan file
    data = np.load(tmp_path / 'scan.npz', allow_pickle=True)['dd']
    df = pd.DataFrame(data, columns=['freq', 'ch0', 'ch1'])
    df['freq'] = df['freq'].astype(float) / 1e9
    for ich in ['ch0', 'ch1']:
        df[ich] = df[ich].astype(float) * 1e3
    df['iq'] = df['ch0'] + 1j * df['ch1']
    assert len(df) == N and df['freq'].iloc[0] == pytest.approx(5.324)
    # and what plot_iq_scan.py does
    ff, ii, qq = np.load(tmp_path / 'scan.npz')['dd'].transpose()
    assert len(ff) == len(ii) == len(qq) == N


def test_the_generator_is_started_and_stopped_at_every_frequency(tmp_path):
    cfg = make_cfg(tmp_path)
    scope, sg = bench(cfg)
    run_iqscan(cfg, scope, sg)
    freqs = np.linspace(F0, F1, N)
    expected = [call for f in freqs for call in (('start', f, -12.0), ('stop',))]
    assert sg.calls == expected and not sg.on


def test_the_settle_time_is_waited_after_starting_the_generator(tmp_path):
    cfg = make_cfg(tmp_path, settle_time=0.25, num_points=4)
    scope, sg = bench(cfg)
    order = []
    scope_measure = scope.measure
    scope.measure = lambda: order.append('measure') or scope_measure()
    run_iqscan(cfg, scope, sg, sleep=lambda s: order.append(('sleep', s, sg.on)))
    assert order == [('sleep', 0.25, True), 'measure'] * 4  # generator already on when waiting

    cfg0 = make_cfg(tmp_path, name='nowait', num_points=2)
    slept = []
    run_iqscan(cfg0, *bench(cfg0), sleep=slept.append)
    assert slept == []


def test_summary_line_per_scan_appended_to_one_file(tmp_path):
    for name in ('a', 'b'):
        cfg = make_cfg(tmp_path, name=name, condition=f'condition {name}', num_points=3)
        run_iqscan(cfg, *bench(cfg))
    header, rows = summary_rows(cfg)
    assert header == list(HEADER)
    assert [(r[0], r[5], r[6], r[7], r[8]) for r in rows] == [
        ('a', '3', '-12', 'completed', 'condition a'), ('b', '3', '-12', 'completed', 'condition b')]
    for r in rows:
        assert datetime.fromisoformat(r[1]) <= datetime.fromisoformat(r[2])
        assert (r[3], r[4]) == ('5.324000', '5.328000')


def test_default_file_name_is_iqscan_and_the_start_minute(tmp_path):
    cfg = make_cfg(tmp_path, name=None, num_points=2)
    run_iqscan(cfg, *bench(cfg))
    files = sorted(p.name for p in tmp_path.iterdir() if p.suffix == '.npz')
    assert len(files) == 1 and re.fullmatch(r'iqscan_\d{12}\.npz', files[0])


def test_an_existing_scan_is_never_overwritten(tmp_path):
    cfg = make_cfg(tmp_path, num_points=3)
    run_iqscan(cfg, *bench(cfg))
    before = (tmp_path / 'scan.npz').read_bytes(), (tmp_path / 'scan.yaml').read_bytes()
    scope, sg = bench(cfg, scope_class=ScriptedScope)
    with pytest.raises(FileExistsError):
        run_iqscan(cfg, scope, sg)
    assert scope.closed and scope.count == 0 and sg.calls == []  # instruments untouched
    assert ((tmp_path / 'scan.npz').read_bytes(), (tmp_path / 'scan.yaml').read_bytes()) == before
    assert len(summary_rows(cfg)[1]) == 1


def test_ctrl_c_keeps_the_points_measured_so_far(tmp_path):
    cfg = make_cfg(tmp_path)
    scope, sg = bench(cfg, scope_class=ScriptedScope, interrupt_at=6)  # during the 6th point
    result = run_iqscan(cfg, scope, sg)

    assert result.status == 'interrupted' and result.exit_code == 130
    assert result.points_measured == 5
    arrays, meta = load(tmp_path)
    assert arrays['dd'].shape == (5, 3)
    np.testing.assert_array_equal(arrays['dd'][:, 0], np.linspace(F0, F1, N)[:5])
    assert meta['status'] == 'interrupted'
    assert meta['scan']['num_points_measured'] == 5 and meta['scan']['num_points_requested'] == N
    assert sg.calls[-1] == ('stop',) and not sg.on and len(sg.calls) == 12  # 6 x (start, stop)
    assert scope.closed
    header, rows = summary_rows(cfg)
    assert (rows[0][5], rows[0][7]) == ('5', 'interrupted')


def test_stopping_before_the_first_point_writes_nothing(tmp_path):
    cfg = make_cfg(tmp_path)
    scope, sg = bench(cfg, scope_class=ScriptedScope, interrupt_at=1)
    result = run_iqscan(cfg, scope, sg)
    assert result.status == 'interrupted' and result.npz_path is None
    assert os.listdir(tmp_path) == [] and not sg.on


def test_a_device_error_is_recorded_and_the_data_kept(tmp_path):
    cfg = make_cfg(tmp_path)
    scope, sg = bench(cfg, scope_class=ScriptedScope, fail_at=4)
    result = run_iqscan(cfg, scope, sg)
    assert result.status == 'error' and result.exit_code == 1 and 'boom' in result.error
    arrays, meta = load(tmp_path)
    assert arrays['dd'].shape == (3, 3)
    assert meta['status'] == 'error' and 'boom' in meta['errors'][0]
    assert not sg.on
    assert [c[0] for c in sg.calls] == ['start', 'stop'] * 4  # the failed point was stopped too


def test_a_failing_generator_stop_does_not_hide_the_original_error(tmp_path):
    class BrokenStopSg(SimulatedSweepSg):
        def stop(self):
            super().stop()
            raise RuntimeError('generator does not answer')

    cfg = make_cfg(tmp_path)
    sg = BrokenStopSg()
    result = run_iqscan(cfg, ScriptedScope(cfg, sg, seed=1, fail_at=2), sg)
    assert result.status == 'error' and 'boom' in result.error  # not 'generator does not answer'
    assert load(tmp_path)[0]['dd'].shape == (1, 3)


def test_the_generator_is_stopped_even_if_starting_it_fails(tmp_path):
    class FailingStartSg(SimulatedSweepSg):
        def start(self, frequency, power):
            super().start(frequency, power)
            raise RuntimeError('cannot start')

    cfg = make_cfg(tmp_path)
    sg = FailingStartSg()
    result = run_iqscan(cfg, ScriptedScope(cfg, sg, seed=1), sg)
    assert result.status == 'error' and 'cannot start' in result.error
    assert sg.calls == [('start', F0, -12.0), ('stop',)] and not sg.on
    assert os.listdir(tmp_path) == []  # nothing was measured


# --- command line --------------------------------------------------------------

def cli_args(tmp_path, *extra):
    return ['--backend', 'simulator', '--f-start', '5.324e9', '--f-stop', '5.328e9',
            '--num-points', '11', '--output-dir', str(tmp_path), *extra]


def test_cli_end_to_end_with_simulator(tmp_path):
    assert main(cli_args(tmp_path, '--name', 'cli_scan', '--power', '-20',
                         '--condition', 'x\ty')) == 0
    arrays, meta = load(tmp_path, 'cli_scan')
    assert arrays['dd'].shape == (11, 3) and float(arrays['power_dbm']) == -20
    assert meta['command_line'][0] == 'kidpack-iqscan' and '--f-start' in meta['command_line']
    lines = (tmp_path / 'iqscan_summary.txt').read_text().splitlines()
    assert len(lines) == 2 and lines[1].endswith('x y')


def test_cli_refuses_an_existing_scan_name(tmp_path, capsys):
    assert main(cli_args(tmp_path, '--name', 's')) == 0
    assert main(cli_args(tmp_path, '--name', 's')) == 1
    assert 'already exists' in capsys.readouterr().err
    assert len((tmp_path / 'iqscan_summary.txt').read_text().splitlines()) == 2


def test_cli_reports_a_missing_driver_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'niscope', None)  # import raises ImportError
    args = cli_args(tmp_path)
    args[args.index('--backend') + 1] = 'niscope'
    assert main(args) == 1
    assert 'could not set up the niscope backend' in capsys.readouterr().err
    assert os.listdir(tmp_path) == []
