import hashlib
import os

import numpy as np
import pytest

from kidpack.daq.backends.simulator import SimulatedScope
from kidpack.daq.config import DaqConfig
from kidpack.daq.runner import run_daq
from kidpack.monitor.cli import find_latest_raw_file, main
from kidpack.monitor.pulses import (make_pulse_view, plot_iq_plane, plot_waveforms, rebin,
                                    select_events)
from kidpack.rawdata import RawData, time_axis_s

PED = (0.030, -0.060)  # V
AMP = 0.010  # V


def synthetic_raw(nevents=20, npts=1000, sample_rate=1e9, ref=20.0, baseline=None, seed=0):
    """Pulse of known amplitude and (per-event) phase on a known pedestal."""
    rng = np.random.default_rng(seed)
    thetas = np.linspace(-1.0, 1.0, nevents)
    t = np.arange(npts) - int(npts * ref / 100)
    pulse = np.where(t >= 0, np.exp(-np.clip(t, 0, None) / 100.0), 0.0)
    i = PED[0] + AMP * np.cos(thetas)[:, None] * pulse + 2e-6 * rng.standard_normal((nevents, npts))
    q = PED[1] + AMP * np.sin(thetas)[:, None] * pulse + 2e-6 * rng.standard_normal((nevents, npts))
    if baseline is not None:  # (first 100 samples offset, rest of the pre-trigger region offset)
        i[:, :100] += baseline[0]
        i[:, 100:int(npts * ref / 100)] += baseline[1]
    raw = RawData('synthetic.npz', i.astype(np.float32), q.astype(np.float32),
                  np.arange(nevents), npts, sample_rate, ref)
    return raw, thetas


def test_pedestal_peak_phase_and_projection_are_recovered():
    raw, thetas = synthetic_raw()
    view = make_pulse_view(raw, np.arange(raw.nevents))

    np.testing.assert_allclose(view.ped_i, PED[0], atol=1e-5)
    np.testing.assert_allclose(view.ped_q, PED[1], atol=1e-5)
    np.testing.assert_allclose(view.proj_theta, thetas, atol=2e-3)
    np.testing.assert_allclose(view.proj_max, AMP, rtol=2e-3)
    # the projected waveform peaks at the full amplitude, sits at zero before the pulse
    np.testing.assert_allclose(view.proj.max(axis=1), AMP, rtol=2e-3)
    assert np.abs(view.proj[:, :190]).max() < 1e-4
    # I and Q are left as acquired (pedestal not subtracted)
    np.testing.assert_allclose(view.i[:, :190].mean(axis=1), PED[0], atol=1e-5)


def test_projection_is_a_rotation_of_the_pedestal_subtracted_iq():
    raw, _ = synthetic_raw()
    view = make_pulse_view(raw, [3, 7])
    vc = (view.i - view.ped_i[:, None]) + 1j * (view.q - view.ped_q[:, None])
    np.testing.assert_allclose(view.proj, np.real(vc * np.exp(-1j * view.proj_theta)[:, None]),
                               atol=1e-7)
    assert np.all(np.abs(view.proj) <= np.abs(vc) + 1e-7)


def test_only_the_requested_events_are_used():
    raw, _ = synthetic_raw()
    view = make_pulse_view(raw, [5, 2])
    np.testing.assert_array_equal(view.event_id, [5, 2])
    np.testing.assert_array_equal(view.i, raw.ch0[[5, 2]])


def test_rebinning_shortens_the_waveforms_and_the_time_axis_consistently():
    raw, _ = synthetic_raw(npts=1000)
    view = make_pulse_view(raw, [0, 1], rebin_factor=4)
    assert view.i.shape == view.q.shape == view.proj.shape == (2, 250)
    np.testing.assert_allclose(view.tbin_us, time_axis_s(250, 1e9 / 4, 20.0) * 1e6)
    np.testing.assert_allclose(view.i, raw.ch0[[0, 1]].reshape(2, 250, 4).mean(axis=2), rtol=1e-6)


def test_pedestal_window_is_the_pretrigger_region_or_the_first_100_ns():
    raw, _ = synthetic_raw(baseline=(0.010, 0.050))  # first 100 samples differ from the rest
    normal = make_pulse_view(raw, [0])
    alpha = make_pulse_view(raw, [0], alpha=True)  # 100 ns = 100 samples at 1 GS/s
    assert normal.ped_i[0] == pytest.approx(PED[0] + (0.010 * 100 + 0.050 * 100) / 200, abs=1e-5)
    assert alpha.ped_i[0] == pytest.approx(PED[0] + 0.010, abs=1e-5)


def test_no_pretrigger_samples_means_no_pedestal():
    raw, _ = synthetic_raw(ref=0.0)
    with pytest.raises(ValueError, match='pedestal'):
        make_pulse_view(raw, [0])
    make_pulse_view(raw, [0], alpha=True)  # the 100 ns window still works


def test_rebin_and_select_events():
    np.testing.assert_array_equal(rebin(np.arange(10.0), 3), [1.0, 4.0, 7.0])  # remainder dropped
    np.testing.assert_array_equal(rebin(np.ones((2, 6)), 1), np.ones((2, 6)))
    assert rebin(np.zeros((4, 12)), 5).shape == (4, 2)
    np.testing.assert_array_equal(select_events(1000), np.arange(16))
    np.testing.assert_array_equal(select_events(1000, stride=2), np.arange(0, 32, 2))
    np.testing.assert_array_equal(select_events(5), np.arange(5))
    with pytest.raises(ValueError):
        select_events(10, stride=0)


# --- figures -----------------------------------------------------------------

def visible_axes(fig):
    return [ax for ax in fig.axes if ax.axison]


def test_pc1_and_pc2_have_sixteen_panels_in_a_4x4_grid():
    raw, _ = synthetic_raw()
    view = make_pulse_view(raw, select_events(raw.nevents))
    for make_figure, extra in ((plot_iq_plane, 1), (plot_waveforms, 0)):  # pc1 has a colorbar
        fig = make_figure(view)
        panels = visible_axes(fig)[:16]
        assert len(visible_axes(fig)) == 16 + extra
        assert [ax.get_title() for ax in panels] == [f'Event #{k}' for k in range(16)]
        assert len({ax.get_subplotspec().rowspan.start for ax in panels}) == 4
        assert len({ax.get_subplotspec().colspan.start for ax in panels}) == 4


def test_pc1_shows_iq_in_mv_with_pedestal_and_peak_markers():
    raw, _ = synthetic_raw()
    view = make_pulse_view(raw, [0])
    ax = plot_iq_plane(view).axes[0]
    xy = ax.collections[0].get_offsets()
    assert xy.shape[0] == 1000 // 5  # extra 5-sample averaging of the displayed points
    assert xy[:, 0].mean() == pytest.approx(PED[0] * 1e3, rel=0.2)  # mV
    ped, peak = ax.lines
    assert (ped.get_xdata()[0], ped.get_ydata()[0]) == pytest.approx((PED[0] * 1e3, PED[1] * 1e3), abs=1e-2)
    px, py = peak.get_xdata()[0], peak.get_ydata()[0]
    assert np.hypot(px - ped.get_xdata()[0], py - ped.get_ydata()[0]) == pytest.approx(AMP * 1e3, rel=2e-3)
    assert ax.get_xlabel() == 'I [mV]' or ax.get_ylabel() == 'Q [mV]'


def test_pc2_plots_i_q_and_proj_in_mv_versus_microseconds():
    raw, _ = synthetic_raw()
    view = make_pulse_view(raw, [0])
    ax = plot_waveforms(view).axes[0]
    assert [line.get_label() for line in ax.lines] == ['I', 'Q', 'Proj']
    np.testing.assert_allclose(ax.lines[2].get_ydata(), view.proj[0] * 1e3)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), view.tbin_us)
    assert ax.get_xlabel() == 'time [µs]' and ax.get_ylabel() == 'voltage [mV]'


def test_fewer_events_than_panels_leave_the_rest_blank():
    raw, _ = synthetic_raw(nevents=5)
    view = make_pulse_view(raw, select_events(raw.nevents))
    fig = plot_waveforms(view)
    assert len(visible_axes(fig)) == 5
    assert len(fig.axes) == 16


# --- command line --------------------------------------------------------------

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_run(data_dir, run_number, events=20, npts=200):
    cfg = DaqConfig(run_number=run_number, events_per_file=events, num_files=1,
                    output_dir=str(data_dir), npts=npts)
    run_daq(cfg, SimulatedScope(cfg, seed=run_number))
    return data_dir / f'run_{run_number:02d}' / 'data' / f'run{run_number:02d}-00.npz'


def test_cli_writes_pc1_and_pc2_and_leaves_the_data_alone(tmp_path, capsys):
    data_dir = tmp_path / 'data'
    path = make_run(data_dir, 1)
    before = (sha(path), sorted(os.listdir(data_dir)), sorted(os.listdir(path.parent.parent)))

    assert main([str(path), '--output-dir', str(tmp_path / 'out')]) == 0
    for name in ('pc1.png', 'pc2.png'):
        assert (tmp_path / 'out' / name).read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    assert 'events 0..15 (16)' in capsys.readouterr().out
    assert (sha(path), sorted(os.listdir(data_dir)), sorted(os.listdir(path.parent.parent))) == before


def test_cli_uses_the_newest_file_by_modification_time(tmp_path, capsys):
    data_dir = tmp_path / 'data'
    older = make_run(data_dir, 2)
    newer = make_run(data_dir, 1)  # lower run number, but written later
    os.utime(older, (1_000_000_000, 1_000_000_000))
    os.utime(newer, (2_000_000_000, 2_000_000_000))
    (newer.parent / 'run01-01.npz.tmp').write_bytes(b'half written')  # must never be picked up
    os.utime(newer.parent / 'run01-01.npz.tmp', (3_000_000_000, 3_000_000_000))

    assert find_latest_raw_file(str(data_dir)) == str(newer)
    assert main(['--data-dir', str(data_dir), '--output-dir', str(tmp_path / 'out')]) == 0
    assert str(newer) in capsys.readouterr().out


def test_cli_options_are_applied(tmp_path, capsys):
    path = make_run(tmp_path / 'data', 1, events=40)
    assert main([str(path), '--output-dir', str(tmp_path / 'o'), '--stride', '2',
                 '--rebin', '4', '--alpha']) == 0
    assert 'events 0..30 (16)' in capsys.readouterr().out


def test_cli_reports_problems_without_a_traceback(tmp_path, capsys):
    assert main(['--data-dir', str(tmp_path / 'empty'), '--output-dir', str(tmp_path)]) == 1
    assert 'no raw file found' in capsys.readouterr().err

    np.savez(tmp_path / 'bad.npz', ch0=np.zeros((2, 4)))
    assert main([str(tmp_path / 'bad.npz'), '--output-dir', str(tmp_path)]) == 1
    assert 'missing key' in capsys.readouterr().err

    assert main([str(tmp_path / 'nonexistent.npz'), '--output-dir', str(tmp_path)]) == 1
    for bad in (['--rebin', '0'], ['--stride', '0']):
        with pytest.raises(SystemExit):
            main([str(tmp_path / 'bad.npz'), *bad])
