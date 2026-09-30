import hashlib
import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kidpack.iqscan.backends import SimulatedIqScope, SimulatedSweepSg
from kidpack.iqscan.config import IqScanConfig
from kidpack.iqscan.plot import FIGSIZE, find_latest_scan, main, plot_scan, s21
from kidpack.iqscan.reader import IqScan, IqScanError, load_iqscan
from kidpack.iqscan.runner import run_iqscan

F = np.linspace(5.324e9, 5.328e9, 41)


def synthetic(name='scan', delay=50e-9, amplitude=0.01, power=None, dip=True, gain=1.0):
    """A notch resonator times a cable delay: known S21, stored as ch0/ch1 in volts."""
    x = 2 * (F - 5.326e9) / 0.4e6
    true = (1 - (0.6 / (1 + 1j * x) if dip else 0)) * np.exp(-2j * np.pi * F * delay)
    iq = amplitude * gain * true
    return IqScan(f'/x/{name}.npz', F.copy(), iq.real, iq.imag, power), true


def write_dd(path, scan, extra=()):
    np.savez(path, dd=np.column_stack([scan.frequency, scan.i, scan.q]),
             **{k: v for k, v in extra})
    return path


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# --- reader ---------------------------------------------------------------------

def test_reads_a_file_of_kidpack_iqscan(tmp_path):
    cfg = IqScanConfig(f_start=5.324e9, f_stop=5.328e9, num_points=11, output_dir=str(tmp_path),
                       name='s', power=-17.0)
    sg = SimulatedSweepSg()
    run_iqscan(cfg, SimulatedIqScope(cfg, sg, seed=1), sg)
    scan = load_iqscan(tmp_path / 's.npz')
    assert scan.name == 's' and scan.power_dbm == -17.0
    np.testing.assert_array_equal(scan.frequency, np.linspace(5.324e9, 5.328e9, 11))
    assert scan.iq.dtype == np.complex128 and len(scan.iq) == 11


def test_reads_the_dd_only_files_of_the_old_iq_scan_py(tmp_path):
    scan, _ = synthetic()
    old = write_dd(tmp_path / 'iq_scan_202607221241.npz', scan)  # only dd, like bb.npz
    loaded = load_iqscan(old)
    assert loaded.power_dbm is None
    np.testing.assert_array_equal(loaded.iq, scan.iq)  # ch0 + i ch1
    assert loaded.name == 'iq_scan_202607221241'


def test_reading_never_modifies_the_file(tmp_path):
    path = write_dd(tmp_path / 'a.npz', synthetic()[0])
    before = sha(path)
    load_iqscan(path)
    assert sha(path) == before and sorted(os.listdir(tmp_path)) == ['a.npz']


@pytest.mark.parametrize('arrays, message', [
    ({'x': np.zeros((3, 3))}, 'no "dd"'),
    ({'dd': np.zeros((3, 2))}, r'\(n, 3\)'),
    ({'dd': np.zeros(3)}, r'\(n, 3\)'),
    ({'dd': np.zeros((0, 3))}, r'\(n, 3\)'),
])
def test_files_that_are_not_iq_scans_are_rejected(tmp_path, arrays, message):
    np.savez(tmp_path / 'bad.npz', **arrays)
    with pytest.raises(IqScanError, match=message):
        load_iqscan(tmp_path / 'bad.npz')


# --- S21 -------------------------------------------------------------------------

def test_without_a_reference_the_stored_values_are_used_as_they_are():
    scan, _ = synthetic()
    np.testing.assert_array_equal(s21(scan), scan.i + 1j * scan.q)


def test_the_reference_scan_divides_out_gain_and_cable_phase():
    scan, true = synthetic()  # notch x cable phase, amplitude 0.01
    ref, _ = synthetic('ref', dip=False, amplitude=0.02)  # through path: same cable, twice the gain
    cable_phase = np.exp(-2j * np.pi * F * 50e-9)
    notch = true / cable_phase
    # S21 = scan / ref: the cable phase cancels, what is left is the notch times 0.01 / 0.02
    np.testing.assert_allclose(s21(scan, ref), notch * 0.5, rtol=1e-12)
    assert abs(s21(scan, ref)).max() == pytest.approx(0.5, rel=1e-2)  # off resonance (scan edge)


def test_a_reference_at_other_frequencies_is_refused():
    scan, _ = synthetic()
    other = IqScan('/x/other.npz', F + 1e6, scan.i, scan.q)
    with pytest.raises(IqScanError, match='same frequencies'):
        s21(scan, other)
    short = IqScan('/x/short.npz', F[:-1], scan.i[:-1], scan.q[:-1])
    with pytest.raises(IqScanError, match='same frequencies'):
        s21(scan, short)


# --- figure ----------------------------------------------------------------------

def three_axes(fig):
    assert len(fig.axes) == 3
    return fig.axes


def test_three_panels_ch0_vs_ch1_magnitude_and_phase():
    scan, _ = synthetic(power=-10.0)
    fig = plot_scan(scan)
    ax_iq, ax_mag, ax_phase = three_axes(fig)

    curve = ax_iq.lines[0]
    np.testing.assert_allclose(curve.get_xdata(), scan.i * 1e3)  # ch0 [mV] on x
    np.testing.assert_allclose(curve.get_ydata(), scan.q * 1e3)  # ch1 [mV] on y
    first = ax_iq.lines[1]
    assert (first.get_xdata()[0], first.get_ydata()[0]) == pytest.approx((scan.i[0] * 1e3, scan.q[0] * 1e3))
    assert ax_iq.get_xlabel() == 'ch0 (I) [mV]' and ax_iq.get_ylabel() == 'ch1 (Q) [mV]'
    assert ax_iq.get_aspect() == 1.0  # circles look like circles

    np.testing.assert_allclose(ax_mag.lines[0].get_xdata(), F / 1e9)  # frequency in GHz
    np.testing.assert_allclose(ax_mag.lines[0].get_ydata(), np.abs(scan.iq) * 1e3)
    np.testing.assert_allclose(ax_phase.lines[0].get_xdata(), F / 1e9)
    assert ax_mag.get_xlabel() == ax_phase.get_xlabel() == 'Frequency [GHz]'
    assert 'uncalibrated' in ax_mag.get_ylabel() and ax_phase.get_ylabel().startswith('arg S21 [rad]')
    assert 'scan' in fig._suptitle.get_text() and '-10 dBm' in fig._suptitle.get_text()


def test_the_resonance_shows_up_at_its_frequency():
    scan, _ = synthetic()
    ax_mag = plot_scan(scan).axes[1]
    x, y = ax_mag.lines[0].get_data()
    assert x[np.argmin(y)] == pytest.approx(5.326)


def test_phase_is_continuous_by_default_and_folded_with_wrap():
    scan, _ = synthetic(delay=2e-6)  # ~8 turns of cable phase over the 4 MHz scan, 1.3 rad per point
    continuous = plot_scan(scan).axes[2].lines[0].get_ydata()
    folded = plot_scan(scan, wrap=True).axes[2].lines[0].get_ydata()
    assert np.abs(np.diff(continuous)).max() < np.pi  # no 2 pi jumps
    assert continuous.max() - continuous.min() > 2 * np.pi
    assert folded.max() <= np.pi and folded.min() >= -np.pi
    np.testing.assert_allclose(np.exp(1j * continuous), np.exp(1j * folded), atol=1e-12)


def test_db_magnitude():
    scan, _ = synthetic()
    y = plot_scan(scan, db=True).axes[1].lines[0].get_ydata()
    np.testing.assert_allclose(y, 20 * np.log10(np.abs(scan.iq)))
    assert 'dB' in plot_scan(scan, db=True).axes[1].get_ylabel()


def test_with_a_reference_the_panels_show_the_normalised_s21():
    scan, true = synthetic()
    ref, _ = synthetic('ref', dip=False)
    fig = plot_scan(scan, ref)
    ax_iq, ax_mag, ax_phase = three_axes(fig)
    values = s21(scan, ref)
    np.testing.assert_allclose(ax_iq.lines[0].get_xdata(), values.real)  # dimensionless, no mV
    np.testing.assert_allclose(ax_mag.lines[0].get_ydata(), np.abs(values))
    assert ax_mag.get_ylabel() == '|S21|' and ax_iq.get_xlabel() == 'Re S21'
    assert 'ref ref' in fig._suptitle.get_text() and 'uncalibrated' not in fig._suptitle.get_text()
    # the notch: |S21| falls to 1 - 0.6 = 0.4 at resonance and is ~1 far away
    assert ax_mag.lines[0].get_ydata().min() == pytest.approx(0.4, abs=0.02)


# --- command line ------------------------------------------------------------------

def scan_files(tmp_path):
    scan, _ = synthetic()
    ref, _ = synthetic('ref', dip=False)
    return (write_dd(tmp_path / 'scan.npz', scan, [('power_dbm', np.float64(-10))]),
            write_dd(tmp_path / 'ref.npz', ref))


def png_size(path):
    with open(path, 'rb') as f:
        header = f.read(24)
    assert header[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', header[16:24])


def test_the_figure_is_shown_in_a_window_and_nothing_is_written_by_default(tmp_path, shown):
    scan_path, _ = scan_files(tmp_path)
    before = sorted(os.listdir(tmp_path))
    assert main([str(scan_path)]) == 0
    assert shown == [[FIGSIZE]]  # one window with the three plots
    assert plt.get_fignums() == []  # closed again afterwards
    assert sorted(os.listdir(tmp_path)) == before  # no file created


def test_save_writes_a_png_as_well(tmp_path, shown):
    scan_path, _ = scan_files(tmp_path)
    assert main([str(scan_path), '--save', str(tmp_path / 'sub' / 'q.png')]) == 0
    assert png_size(tmp_path / 'sub' / 'q.png') == (1500, 480)
    assert len(shown) == 1


def test_no_show_only_saves(tmp_path, shown):
    scan_path, _ = scan_files(tmp_path)
    assert main([str(scan_path), '--no-show', '--save', str(tmp_path / 'q.png')]) == 0
    assert shown == [] and plt.get_fignums() == [] and (tmp_path / 'q.png').exists()
    with pytest.raises(SystemExit):
        main([str(scan_path), '--no-show'])  # would do nothing


def test_the_data_files_are_left_alone(tmp_path):
    scan_path, ref_path = scan_files(tmp_path)
    before = sha(scan_path), sha(ref_path)
    assert main([str(scan_path), '--ref', str(ref_path), '--no-show', '--save', str(tmp_path / 'q.png')]) == 0
    assert (sha(scan_path), sha(ref_path)) == before


def test_options_reach_the_figure(tmp_path, monkeypatch):
    scan_path, ref_path = scan_files(tmp_path)
    figures = []
    import kidpack.iqscan.plot as plot_module
    original = plot_module.plot_scan
    monkeypatch.setattr(plot_module, 'plot_scan',
                        lambda *a, **k: figures.append(k) or original(*a, **k))
    assert main([str(scan_path), '--ref', str(ref_path), '--db', '--wrap', '--no-show',
                 '--save', str(tmp_path / 'q.png')]) == 0
    assert (figures[0]['db'], figures[0]['wrap']) == (True, True)


def test_without_a_file_the_newest_scan_of_the_data_dir_is_used(tmp_path, capsys):
    scan, _ = synthetic()
    old = write_dd(tmp_path / 'aaa.npz', scan)
    new = write_dd(tmp_path / 'zzz_named_by_hand.npz', scan)
    os.utime(old, (2_000_000_000, 2_000_000_000))  # 'aaa' is the newest although first by name
    os.utime(new, (1_000_000_000, 1_000_000_000))
    (tmp_path / 'half.npz.tmp').write_bytes(b'x')
    os.utime(tmp_path / 'half.npz.tmp', (3_000_000_000, 3_000_000_000))
    assert find_latest_scan(str(tmp_path)) == str(old)
    assert main(['--data-dir', str(tmp_path), '--no-show', '--save', str(tmp_path / 'q.png')]) == 0
    assert str(old) in capsys.readouterr().out


def test_problems_are_reported_without_a_traceback(tmp_path, capsys, shown):
    scan_path, ref_path = scan_files(tmp_path)
    assert main(['--data-dir', str(tmp_path / 'empty'), '--no-show', '--save', str(tmp_path / 'q.png')]) == 1
    assert 'no IQ-scan file found' in capsys.readouterr().err

    np.savez(tmp_path / 'notascan.npz', x=np.zeros(3))
    assert main([str(tmp_path / 'notascan.npz')]) == 1
    assert 'no "dd"' in capsys.readouterr().err

    assert main([str(tmp_path / 'missing.npz')]) == 1
    assert main([str(scan_path), '--ref', str(tmp_path / 'missing.npz')]) == 1

    shifted = synthetic('shifted')[0]
    shifted.frequency = shifted.frequency + 1e6
    other = write_dd(tmp_path / 'shifted.npz', shifted)
    assert main([str(scan_path), '--ref', str(other)]) == 1
    assert 'same frequencies' in capsys.readouterr().err
    assert shown == [] and plt.get_fignums() == []  # no window for bad input
