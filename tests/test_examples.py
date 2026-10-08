"""The sample macros in examples/ (handed to students): they must keep working and stay independent."""
import ast
import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kidpack.daq.backends.simulator import SimulatedScope
from kidpack.daq.config import DaqConfig
from kidpack.daq.runner import run_daq
from kidpack.iqscan.plot import plot_scan
from kidpack.iqscan.reader import load_iqscan
from kidpack.monitor.pulses import make_pulse_view, plot_iq_plane, plot_waveforms, select_events
from kidpack.rawdata import load_raw

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
TOYMC = EXAMPLES / '01_kid_response_toymc.py'
ANALYSIS = EXAMPLES / '02_pulse_analysis.py'
WAVEFORMS = EXAMPLES / '03_plot_waveforms.py'
IQSCAN = EXAMPLES / '04_plot_iqscan.py'


@pytest.fixture(scope='module')
def toy():
    spec = importlib.util.spec_from_file_location('toy_example', TOYMC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(toy, **changes):
    return {**toy.SETTINGS, **changes}


# --- independence -------------------------------------------------------------------

@pytest.mark.parametrize('path', sorted(EXAMPLES.glob('[0-9][0-9]_*.py')), ids=lambda p: p.name)
def test_every_example_is_a_stand_alone_macro(path):
    """Only numpy / matplotlib (and a little standard library): no scipy, no kidpack, no local helpers."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split('.')[0])
    standard = {'sys', 'os', 'math'}  # (small) part of the standard library
    assert imported <= standard | {'numpy', 'matplotlib'}, imported


# --- the S21 formula ------------------------------------------------------------------

def test_on_resonance_the_dip_has_the_depth_ql_over_qc(toy):
    fr = 5.4e9
    assert toy.s21_notch(fr, fr, 500, 1000) == pytest.approx(0.5)  # 1 - Ql/Qc
    assert toy.s21_notch(fr, fr, 900, 900) == pytest.approx(0.0)  # critical coupling: no transmission
    assert toy.s21_notch(fr, fr, 500, 1000, a=2.0) == pytest.approx(1.0)  # a scales everything


def test_far_from_resonance_only_the_envelope_is_left(toy):
    fr, f = 5.4e9, 5.4e9 * 3
    value = toy.s21_notch(f, fr, 500, 1000, a=1.5, alpha=0.3, tau=1e-10)
    assert value == pytest.approx(1.5 * np.exp(0.3j) * np.exp(-2j * np.pi * f * 1e-10), abs=1e-2)


def test_for_phi_zero_the_curve_is_symmetric_and_the_iq_loop_is_a_circle(toy):
    fr, Ql, Qc = 5.4e9, 800, 1600
    f = fr * (1 + np.linspace(-20, 20, 4001) / Ql)
    value = toy.s21_notch(f, fr, Ql, Qc)
    np.testing.assert_allclose(np.abs(value), np.abs(value)[::-1], atol=1e-12)  # symmetric about fr
    centre, radius = 1 - Ql / Qc / 2, Ql / Qc / 2
    np.testing.assert_allclose(np.abs(value - centre), radius, atol=1e-9)  # a circle of diameter Ql/Qc


def test_the_parameters_do_what_their_names_say(toy):
    f = np.linspace(5.39e9, 5.41e9, 50)
    base = toy.s21_notch(f, 5.4e9, 800, 1600)
    np.testing.assert_allclose(toy.s21_notch(f, 5.4e9, 800, 1600, a=3.0), 3 * base)  # amplitude
    np.testing.assert_allclose(toy.s21_notch(f, 5.4e9, 800, 1600, alpha=0.7), base * np.exp(0.7j))
    np.testing.assert_allclose(toy.s21_notch(f, 5.4e9, 800, 1600, tau=2e-9),
                               base * np.exp(-2j * np.pi * f * 2e-9))  # cable delay
    skewed = toy.s21_notch(f, 5.4e9, 800, 1600, phi=0.8)  # asymmetry breaks the symmetry
    assert not np.allclose(np.abs(skewed), np.abs(skewed)[::-1])


def test_qi_is_the_internal_q(toy):
    Ql, Qc = 800.0, 1600.0
    Qi = toy.Qi_from(Ql, Qc)
    assert 1 / Ql == pytest.approx(1 / Qi + 1 / Qc)
    assert Qi == pytest.approx(1600.0)


# --- the temperature model (values of the analysis/toymc_kidresponse_test.py macro) ------

def test_the_temperature_model_reproduces_the_reference_macro(toy):
    assert toy.fr_T(5.5) == pytest.approx(5.380339e9, rel=1e-7)
    assert toy.Ql_T(5.5) == pytest.approx(911.3, abs=0.1)
    assert toy.Qc_T(5.5) == pytest.approx(1644.4, abs=0.1)
    T = np.linspace(1, 9, 50)
    assert np.all(np.diff(toy.fr_T(T)) < 0)  # a warmer film resonates at a lower frequency
    assert toy.Ql_T(np.array([9.0, 9.1]))[0] == 1.0  # never below 1


# --- the simulation -----------------------------------------------------------------------

def t_peak(tau_rise=50.0, tau_decay=200.0):
    """Time of the maximum of exp(-t/tau_decay) - exp(-t/tau_rise)."""
    return np.log(tau_rise / tau_decay) / (1 / tau_decay - 1 / tau_rise)


def test_the_temperature_pulse_starts_and_ends_at_the_base_temperature(toy):
    r = toy.simulate(**settings(toy, T_base=5.5, delta_T=1.5))
    assert r['T'][0] == pytest.approx(5.5)
    assert r['T'][-1] == pytest.approx(5.5, abs=0.02)  # 1.5 K x e^-5 is left after 1000 ns
    assert abs(np.argmax(r['T']) - t_peak()) <= 1
    expected_peak = 5.5 + 1.5 * (np.exp(-t_peak() / 200) - np.exp(-t_peak() / 50))
    assert r['T'].max() == pytest.approx(expected_peak, abs=1e-3)  # lower than T_hot: finite rise time
    assert r['settings']['T_hot'] == pytest.approx(7.0)  # T_hot = T_base + delta_T


def test_the_pulse_is_the_same_whatever_the_base_temperature(toy):
    shapes = [toy.simulate(**settings(toy, T_base=T_base, delta_T=0.5))['T'] - T_base
              for T_base in (4.5, 5.5, 6.5)]
    np.testing.assert_allclose(shapes[0], shapes[1], atol=1e-12)
    np.testing.assert_allclose(shapes[2], shapes[1], atol=1e-12)


def test_the_readout_is_tuned_to_the_resonance_at_the_base_temperature_by_default(toy):
    r = toy.simulate(**settings(toy))
    assert toy.SETTINGS['f_readout'] is None
    assert r['settings']['f_readout'] == pytest.approx(toy.fr_T(5.5))  # 5.3803 GHz
    assert r['s21'][0] == pytest.approx(1 - toy.Ql_T(5.5) / toy.Qc_T(5.5))  # on resonance: the bottom of the dip
    for T_base in (4.5, 6.5):  # ... at every base temperature
        r = toy.simulate(**settings(toy, T_base=T_base))
        assert r['settings']['f_readout'] == pytest.approx(toy.fr_T(T_base))
        assert r['s21'][0] == pytest.approx(1 - toy.Ql_T(T_base) / toy.Qc_T(T_base))


def test_a_given_readout_frequency_is_kept(toy):
    r = toy.simulate(**settings(toy, f_readout=5.38e9))
    assert r['settings']['f_readout'] == 5.38e9
    rest = toy.s21_notch(5.38e9, toy.fr_T(5.5), toy.Ql_T(5.5), toy.Qc_T(5.5))
    assert r['s21'][0] == pytest.approx(rest)
    # and it stays there when the base temperature changes
    assert toy.simulate(**settings(toy, T_base=4.5, f_readout=5.38e9))['settings']['f_readout'] == 5.38e9


def test_no_pulse_no_response(toy):
    r = toy.simulate(**settings(toy, delta_T=0.0))
    np.testing.assert_allclose(r['s21'], r['s21'][0])
    assert toy.peak_response(r) == pytest.approx(0.0, abs=1e-12)


def test_heating_moves_the_resonance_away_from_the_readout_and_the_dip_gets_shallower(toy):
    r = toy.simulate(**settings(toy))  # readout on the resonance
    assert r['fr'].min() < r['fr'][0]
    assert np.abs(r['s21']).max() > np.abs(r['s21'][0]) + 0.1


def test_a_readout_far_from_the_resonance_sees_almost_nothing(toy):
    on = toy.peak_response(toy.simulate(**settings(toy)))
    off = toy.peak_response(toy.simulate(**settings(toy, f_readout=5.25e9)))
    assert off < 0.05 * on


def test_ql_and_qc_are_constant_unless_they_are_asked_to_follow_the_temperature(toy):
    constant = toy.simulate(**settings(toy))
    assert np.all(constant['Ql'] == constant['Ql'][0]) and constant['Ql'][0] == pytest.approx(toy.Ql_T(5.5))
    given = toy.simulate(**settings(toy, Ql=700.0, Qc=1500.0))
    assert given['Ql'][0] == 700.0 and given['Qc'][0] == 1500.0
    moving = toy.simulate(**settings(toy, q_changes_with_T=True, Ql=700.0))
    assert moving['Ql'].min() < moving['Ql'][0] == pytest.approx(toy.Ql_T(5.5))  # given Ql is ignored
    assert moving['Qc'].max() > moving['Qc'][0]


def test_a_pulse_that_would_break_the_superconductor_is_refused(toy):
    with pytest.raises(ValueError, match='delta_T'):
        toy.simulate(**settings(toy, delta_T=30.0))
    with pytest.raises(ValueError, match='T_base'):
        toy.simulate(**settings(toy, T_base=9.1, delta_T=1.0))


def test_the_macro_has_no_fit_any_more(toy):
    assert not hasattr(toy, 'decay_fit') and not hasattr(toy, 'curve_fit')


# --- the comparison ---------------------------------------------------------------------

def test_compare_overlays_one_curve_per_value_and_prints_a_table(toy, capsys):
    values = [0.1, 0.5, 1.0]
    fig = toy.compare('delta_T', values)
    assert len(fig.axes) == 6
    assert all(len(ax.lines) == 3 for ax in fig.axes[:5])  # one curve per value
    text = capsys.readouterr().out
    assert 'delta_T' in text and 'tau' not in text  # the table has the size of the response only
    assert text.count('max|dS21|') == 3  # one line per value
    assert 'size of the response' in text  # under a header
    sizes = fig.axes[5].lines[0].get_ydata()
    assert np.all(np.diff(sizes) > 0)  # a bigger pulse, a bigger response


def test_comparing_the_base_temperature_keeps_the_pulse_size_fixed(toy, capsys):
    values = [4.5, 5.5, 6.5]
    fig = toy.compare('T_base', values)
    assert fig.axes[0].get_legend_handles_labels()[1] == ['T_base = 4.5', 'T_base = 5.5', 'T_base = 6.5']
    assert 'delta_T = 0.5 fixed' in fig._suptitle.get_text()  # and the title says so
    assert 'comparison of T_base' in fig._suptitle.get_text()

    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith('T_base = ')]
    assert len(rows) == 3
    for value, row in zip(values, rows):
        t_hot = float(row.split('T_hot = ')[1].split(' K')[0])
        assert t_hot - value == pytest.approx(0.5)  # delta_T is the same in every row, T_hot moves
    f_readout = [float(row.split('f_readout = ')[1].split(' GHz')[0]) for row in rows]
    assert f_readout == sorted(f_readout, reverse=True)  # each readout follows its resonance (it falls with T)


def test_the_response_depends_on_the_base_temperature_for_the_same_pulse(toy):
    results = [toy.simulate(**settings(toy, T_base=T_base)) for T_base in (4.5, 5.0, 5.5, 6.0, 6.5)]
    depth = [r['Ql'][0] / r['Qc'][0] for r in results]
    assert all(a > b for a, b in zip(depth, depth[1:]))  # the dip gets shallower as it gets warmer
    rest = [r['s21'][0].real for r in results]
    assert all(a < b for a, b in zip(rest, rest[1:]))  # so S21 at rest rises
    sizes = [toy.peak_response(r) for r in results]
    assert max(sizes) - min(sizes) > 0.1  # and the response is not the same


def test_compare_shows_frequencies_in_ghz(toy, capsys):
    fig = toy.compare('f_readout', [5.377e9, 5.380e9])
    assert 'f_readout = 5.377 GHz' in fig.axes[0].get_legend_handles_labels()[1]
    assert '5.380 GHz' in capsys.readouterr().out
    assert 'T_base = 5.5, delta_T = 0.5 fixed' in fig._suptitle.get_text()


@pytest.mark.parametrize('name, values', [('phi', [-0.5, 0.5]), ('q_changes_with_T', [False, True]),
                                          ('Ql', [500.0, 900.0]), ('cable_delay', [0.0, 5e-9]),
                                          ('T_base', [5.0, 6.0]), ('delta_T', [0.2, 0.8])])
def test_any_setting_can_be_compared(toy, name, values):
    assert len(toy.compare(name, values).axes) == 6


# --- running the script -----------------------------------------------------------------

def test_the_script_runs_and_opens_its_figures(monkeypatch, shown):
    monkeypatch.setattr(sys, 'argv', ['01_kid_response_toymc.py'])
    runpy.run_path(str(TOYMC), run_name='__main__')
    assert len(shown) == 1  # one plt.show() ...
    assert len(shown[0]) == 2 + 2  # ... with 2 basic figures and the 2 default comparisons
    assert 'comparison of f_readout' in plt.figure(3)._suptitle.get_text()  # figure 3
    assert 'comparison of T_base' in plt.figure(4)._suptitle.get_text()  # figure 4: not T_hot any more


def test_the_two_optional_arguments_are_T_hot_and_T_base(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['01_kid_response_toymc.py', '6.5', '5.0'])
    runpy.run_path(str(TOYMC), run_name='__main__')
    temperature = plt.figure(2).axes[0].lines[0].get_ydata()  # figure 2: the time response
    assert temperature[0] == pytest.approx(5.0)  # T_base = 5.0, so delta_T = 6.5 - 5.0 = 1.5
    assert temperature.max() == pytest.approx(5.0 + 1.5 * (np.exp(-t_peak() / 200) - np.exp(-t_peak() / 50)), abs=1e-3)
    assert 'delta_T = 1.5 K' in plt.figure(2).axes[1].get_title()


def test_with_only_T_hot_the_base_temperature_stays_as_it_is(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['01_kid_response_toymc.py', '6.0'])
    runpy.run_path(str(TOYMC), run_name='__main__')
    temperature = plt.figure(2).axes[0].lines[0].get_ydata()
    assert temperature[0] == pytest.approx(5.5)  # the default T_base
    assert 'delta_T = 0.5 K' in plt.figure(2).axes[1].get_title()  # 6.0 - 5.5


# --- in a Jupyter notebook ------------------------------------------------------------------------

def test_in_a_notebook_cell_the_arguments_of_jupyter_are_not_the_arguments_of_the_macro(monkeypatch, shown):
    """Pasted into a cell the file runs as __main__ with sys.argv = ['...ipykernel_launcher.py', '-f', kernel.json]."""
    monkeypatch.setattr(sys, 'argv', ['/x/ipykernel_launcher.py', '-f', '/x/kernel-1234.json'])
    runpy.run_path(str(TOYMC), run_name='__main__')  # (it used to stop with: could not convert string to float: '-f')
    assert len(shown) == 1 and len(shown[0]) == 4
    assert 'delta_T = 0.5 K' in plt.figure(2).axes[1].get_title()  # the settings of the file


def test_a_bad_argument_of_the_command_line_is_still_an_error(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['01_kid_response_toymc.py', 'abc'])
    with pytest.raises(ValueError, match='abc'):
        runpy.run_path(str(TOYMC), run_name='__main__')


def test_main_takes_T_hot_and_T_base_as_arguments(toy, shown):
    toy.main(6.5, 5.0)
    temperature = plt.figure(2).axes[0].lines[0].get_ydata()
    assert temperature[0] == pytest.approx(5.0) and 'delta_T = 1.5 K' in plt.figure(2).axes[1].get_title()
    assert 'T_base = 5, delta_T = 1.5 fixed' in plt.figure(3)._suptitle.get_text()  # the comparisons use it too


def test_running_main_again_gives_the_same_result_the_settings_are_not_changed(toy, shown):
    before = dict(toy.SETTINGS)
    toy.main(6.5, 5.0)
    assert toy.SETTINGS == before  # (a notebook cell is run again and again)
    plt.close('all')
    toy.main()
    assert 'delta_T = 0.5 K' in plt.figure(2).axes[1].get_title() and toy.SETTINGS == before


def test_compare_can_be_given_the_settings_to_start_from(toy):
    fig = toy.compare('T_base', [5.0, 6.0], settings={**toy.SETTINGS, 'delta_T': 1.0})
    assert 'delta_T = 1' in fig._suptitle.get_text()
    assert toy.SETTINGS['delta_T'] == 0.5  # the module's own settings are not touched


def test_the_toy_mc_in_a_real_jupyter_kernel(tmp_path):
    """The file pasted into a cell, %run, and main(T_hot, T_base): all run and show the 4 figures.

    Needs ipykernel and nbclient (not installed by kidpack): skipped where there is no Jupyter.
    """
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    source = TOYMC.read_text(encoding='utf-8')
    cells = [source, f'%run {TOYMC}', 'main(6.0, 5.0)']
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(c) for c in cells]
    nbclient.NotebookClient(notebook, timeout=180, kernel_name='python3', allow_errors=False,
                            resources={'metadata': {'path': str(tmp_path)}}).execute()
    for cell in notebook.cells:
        figures = [o for o in cell.outputs if o.output_type == 'display_data' and 'image/png' in o.data]
        assert len(figures) == 4


# =============================================================================================
# 02_pulse_analysis.py
# =============================================================================================
NPTS, SAMPLE_RATE, REF_POSITION = 5000, 2.5e9, 20.0
COLUMNS = ['proj_half_max_left', 'proj_half_max_right', 'proj_max', 'proj_max_t', 'proj_integ',
           'ch0_ped', 'ch1_ped']
US_PER_BIN = 5 / SAMPLE_RATE * 1e6  # one bin = 5 samples


def pulse_shape(decay_len=1000):
    """A triangle: 0 up to sample 1000 (the trigger), rises to 1 at sample 1400, falls linearly to 0."""
    s = np.arange(NPTS)
    p = np.zeros(NPTS)
    rise = (s >= 1000) & (s < 1400)
    p[rise] = (s[rise] - 1000) / 400
    fall = (s >= 1400) & (s <= 1400 + decay_len)
    p[fall] = 1 - (s[fall] - 1400) / decay_len
    return p


def write_pulses(path, nwf=12, decay_len=1000, ref_position=REF_POSITION):
    """Noise-free pulses of known size, phase and pedestal. Returns their parameters."""
    i = np.arange(nwf)
    amp, theta = 0.004 + 0.0005 * i, 0.3 + 0.35 * i  # phases all over the IQ plane
    ped_i, ped_q = 0.02 + 0.001 * i, -0.03 + 0.0007 * i
    p = pulse_shape(decay_len)
    ch0 = ped_i[:, None] + (amp * np.cos(theta))[:, None] * p
    ch1 = ped_q[:, None] + (amp * np.sin(theta))[:, None] * p
    np.savez(path, ch0=ch0.astype(np.float32), ch1=ch1.astype(np.float32), npts=np.int32(NPTS),
             sample_rate=np.float64(SAMPLE_RATE), ref_position=np.float64(ref_position))
    return dict(amp=amp, theta=theta, ped_i=ped_i, ped_q=ped_q, p=p)


def run_analysis(monkeypatch, tmp_path, path):
    """Run the macro on ``path`` from ``tmp_path`` (it writes its csv into the current directory)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['02_pulse_analysis.py', str(path)])
    runpy.run_path(str(ANALYSIS), run_name='__main__')
    return np.genfromtxt(tmp_path / (Path(path).stem + '_ana.csv'), delimiter=',', names=True)


def expected_half_widths(p, amp):
    """Half-max times of a known noise-free pulse, found with np.interp (not the macro's code)."""
    bins = p[:NPTS // 5 * 5].reshape(-1, 5).mean(axis=1)  # the rebinned pulse (5 samples per bin)
    k = int(np.argmax(bins))
    half = bins[k] / 2
    left = k - np.interp(half, bins[200:k + 1], np.arange(200, k + 1))  # rising edge, bins 200 ... k
    end = 481  # the pulse is back to 0 at sample 2400 (bin 480)
    right = np.interp(half, bins[k:end][::-1], np.arange(k, end)[::-1]) - k
    return left * US_PER_BIN, right * US_PER_BIN, bins[k] * amp, (k - 200) * US_PER_BIN


def test_the_macro_finds_the_known_pulses(monkeypatch, tmp_path):
    truth = write_pulses(tmp_path / 'pulses.npz')
    table = run_analysis(monkeypatch, tmp_path, tmp_path / 'pulses.npz')
    assert table.dtype.names == tuple(COLUMNS)
    assert len(table) == 12

    left, right, top, t_peak = expected_half_widths(truth['p'], truth['amp'])
    np.testing.assert_allclose(table['ch0_ped'], truth['ped_i'], atol=1e-7)
    np.testing.assert_allclose(table['ch1_ped'], truth['ped_q'], atol=1e-7)
    np.testing.assert_allclose(table['proj_max'], top, rtol=1e-5)  # amplitude of the (5-sample averaged) peak
    np.testing.assert_allclose(table['proj_max_t'], t_peak, atol=1e-9)  # bin 280 = 0.16 us after the trigger
    np.testing.assert_allclose(table['proj_half_max_left'], left, atol=1e-6)  # ~ 200 samples = 0.08 us
    np.testing.assert_allclose(table['proj_half_max_right'], right, atol=1e-6)  # ~ 500 samples = 0.2 us
    assert t_peak == pytest.approx(0.16)


def test_proj_integ_is_the_sum_over_the_original_samples(monkeypatch, tmp_path):
    """Bin sums times REBIN: the same as summing the raw samples of the first microsecond."""
    truth = write_pulses(tmp_path / 'pulses.npz')
    table = run_analysis(monkeypatch, tmp_path, tmp_path / 'pulses.npz')
    raw_sum = truth['p'][1000:1000 + 2500].sum()  # trigger ... 1 us later (2500 samples at 2.5 GS/s)
    assert raw_sum == pytest.approx(700.0)
    np.testing.assert_allclose(table['proj_integ'], truth['amp'] * raw_sum, rtol=1e-5)


def test_the_result_does_not_depend_on_the_phase_of_the_pulse(monkeypatch, tmp_path):
    """proj rotates the pulse onto the real axis: every direction in the IQ plane gives the same answer."""
    path = tmp_path / 'phases.npz'
    p = pulse_shape()
    theta = np.linspace(-np.pi, np.pi, 9)
    np.savez(path, ch0=(0.01 * np.cos(theta)[:, None] * p).astype(np.float32),
             ch1=(0.01 * np.sin(theta)[:, None] * p).astype(np.float32), npts=np.int32(NPTS),
             sample_rate=np.float64(SAMPLE_RATE), ref_position=np.float64(REF_POSITION))
    table = run_analysis(monkeypatch, tmp_path, path)
    for name in ('proj_max', 'proj_max_t', 'proj_half_max_left', 'proj_half_max_right', 'proj_integ'):
        assert np.ptp(table[name]) < 1e-5 * np.abs(table[name]).max(), name


def test_a_pulse_that_never_falls_to_half_gives_nan_on_the_right_only(monkeypatch, tmp_path):
    write_pulses(tmp_path / 'slow.npz', nwf=3, decay_len=20000)  # still at 0.82 at the end of the record
    table = run_analysis(monkeypatch, tmp_path, tmp_path / 'slow.npz')
    assert np.all(np.isnan(table['proj_half_max_right']))
    assert np.all(np.isfinite(table['proj_half_max_left']))
    assert np.all(np.isfinite(table['proj_max'])) and np.all(np.isfinite(table['proj_integ']))


def test_it_plots_the_two_pedestal_histograms(monkeypatch, tmp_path, shown):
    write_pulses(tmp_path / 'pulses.npz', nwf=30)
    run_analysis(monkeypatch, tmp_path, tmp_path / 'pulses.npz')
    assert len(shown) == 1 and len(shown[0]) == 1  # one window ...
    fig = plt.figure(plt.get_fignums()[0])
    ax = fig.axes
    assert len(ax) == 2  # ... with the two histograms
    assert ax[0].get_xlabel() == 'ch0_ped [mV]' and ax[1].get_xlabel() == 'ch1_ped [mV]'
    for axis in ax:
        assert sum(patch.get_height() for patch in axis.patches) == 30  # every event is counted once
    edges = [patch.get_x() for patch in ax[0].patches]
    assert min(edges) >= 20 - 1e-6 and max(edges) <= 20 + 29 + 1  # ped_i in mV: 20 ... 49


def test_it_writes_only_its_csv_and_leaves_the_data_alone(monkeypatch, tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    write_pulses(data_dir / 'pulses.npz')
    before = hashlib.sha256((data_dir / 'pulses.npz').read_bytes()).hexdigest()
    work = tmp_path / 'work'
    work.mkdir()
    run_analysis(monkeypatch, work, data_dir / 'pulses.npz')
    assert hashlib.sha256((data_dir / 'pulses.npz').read_bytes()).hexdigest() == before
    assert [p.name for p in data_dir.iterdir()] == ['pulses.npz']  # nothing written next to the data
    assert [p.name for p in work.iterdir()] == ['pulses_ana.csv']  # only the table, where we are


def test_the_csv_holds_the_seven_columns_in_the_documented_order(monkeypatch, tmp_path):
    write_pulses(tmp_path / 'pulses.npz', nwf=4)
    run_analysis(monkeypatch, tmp_path, tmp_path / 'pulses.npz')
    first_line = (tmp_path / 'pulses_ana.csv').read_text().splitlines()[0]
    assert first_line == ','.join(COLUMNS)


def test_it_reads_the_files_of_the_kidpack_daq_too(monkeypatch, tmp_path):
    from kidpack.daq.backends.simulator import SimulatedScope
    from kidpack.daq.config import DaqConfig
    from kidpack.daq.runner import run_daq

    cfg = DaqConfig(run_number=1, events_per_file=50, num_files=1, output_dir=str(tmp_path / 'out'),
                    npts=5000)
    run_daq(cfg, SimulatedScope(cfg, seed=3))
    table = run_analysis(monkeypatch, tmp_path, tmp_path / 'out' / 'run_01' / 'data' / 'run01-00.npz')
    assert len(table) == 50
    assert np.all(table['proj_max'] > 0.004)  # the simulator's pulse is 5 mV
    assert np.all(np.abs(table['proj_max_t'] - 0.0) < 0.05)  # right at the trigger
    assert np.all(np.abs(table['ch0_ped']) < 1e-3)


@pytest.mark.parametrize('argv, message', [([], '使い方'), (['a.npz', 'b.npz'], '使い方')])
def test_wrong_arguments_print_the_usage(monkeypatch, capsys, argv, message):
    monkeypatch.setattr(sys, 'argv', ['02_pulse_analysis.py', *argv])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ANALYSIS), run_name='__main__')
    assert exit_info.value.code == 1 and message in capsys.readouterr().out


def test_a_missing_file_is_reported_in_plain_words(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['02_pulse_analysis.py', str(tmp_path / 'nothing.npz')])
    with pytest.raises(SystemExit, match='ファイルが見つかりません'):
        runpy.run_path(str(ANALYSIS), run_name='__main__')


def test_no_samples_before_the_trigger_means_no_pedestal(monkeypatch, tmp_path):
    write_pulses(tmp_path / 'notrigger.npz', nwf=2, ref_position=0.0)
    monkeypatch.setattr(sys, 'argv', ['02_pulse_analysis.py', str(tmp_path / 'notrigger.npz')])
    with pytest.raises(SystemExit, match='ref_position'):
        runpy.run_path(str(ANALYSIS), run_name='__main__')


# --- in a Jupyter notebook ------------------------------------------------------------------------

JUPYTER_ARGV = ['/x/ipykernel_launcher.py', '-f', '/x/kernel-1234.json']


def with_filename(tmp_path, filename):
    """What a student does in a notebook: the macro with the file name written after FILENAME = ."""
    source = ANALYSIS.read_text(encoding='utf-8')
    assert source.count('FILENAME = None') == 1  # the line to edit
    path = tmp_path / 'macro_with_filename.py'
    path.write_text(source.replace('FILENAME = None', f'FILENAME = {str(filename)!r}'), encoding='utf-8')
    return path


def test_in_a_notebook_cell_the_file_name_is_FILENAME(monkeypatch, tmp_path):
    """Pasted into a cell, sys.argv is Jupyter's ['-f', ...json]: not an argument of the macro."""
    write_pulses(tmp_path / 'pulses.npz', nwf=5)
    macro = with_filename(tmp_path, tmp_path / 'pulses.npz')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', JUPYTER_ARGV)
    namespace = runpy.run_path(str(macro), run_name='__main__')
    table = np.genfromtxt(tmp_path / 'pulses_ana.csv', delimiter=',', names=True)
    assert table.dtype.names == tuple(COLUMNS) and len(table) == 5
    assert namespace['proj_max'].shape == (5,)  # the results are there for the next cell


def test_in_a_notebook_without_FILENAME_the_usage_says_where_to_write_it(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', JUPYTER_ARGV)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ANALYSIS), run_name='__main__')
    out = capsys.readouterr().out
    assert exit_info.value.code == 1 and '使い方' in out and 'FILENAME' in out


def test_a_missing_FILENAME_is_reported_in_plain_words(monkeypatch, tmp_path):
    macro = with_filename(tmp_path, tmp_path / 'nothing.npz')
    monkeypatch.setattr(sys, 'argv', JUPYTER_ARGV)
    with pytest.raises(SystemExit, match='ファイルが見つかりません'):
        runpy.run_path(str(macro), run_name='__main__')


def test_an_argument_of_the_command_line_wins_over_FILENAME(monkeypatch, tmp_path):
    write_pulses(tmp_path / 'one.npz', nwf=3)
    write_pulses(tmp_path / 'two.npz', nwf=4)
    macro = with_filename(tmp_path, tmp_path / 'one.npz')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['02_pulse_analysis.py', str(tmp_path / 'two.npz')])  # also %run macro.py two.npz
    runpy.run_path(str(macro), run_name='__main__')
    assert len(np.genfromtxt(tmp_path / 'two_ana.csv', delimiter=',', names=True)) == 4
    assert not (tmp_path / 'one_ana.csv').exists()


def test_the_analysis_in_a_real_jupyter_kernel(tmp_path):
    """The macro pasted into a cell (with FILENAME), and %run with a file name, in a real kernel.

    Needs ipykernel and nbclient (not installed by kidpack): skipped where there is no Jupyter.
    """
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    write_pulses(tmp_path / 'pulses.npz', nwf=6)
    pasted = with_filename(tmp_path, tmp_path / 'pulses.npz').read_text(encoding='utf-8')
    cells = [pasted, 'print(proj_max.shape)', f'%run {ANALYSIS} {tmp_path / "pulses.npz"}', pasted]
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(c) for c in cells]
    nbclient.NotebookClient(notebook, timeout=180, kernel_name='python3', allow_errors=False,
                            resources={'metadata': {'path': str(tmp_path)}}).execute()
    assert [o['text'] for o in notebook.cells[1].outputs if o.output_type == 'stream'] == ['(6,)\n']
    for number in (0, 2, 3):  # the histograms of the pedestals
        figures = [o for o in notebook.cells[number].outputs if o.output_type == 'display_data' and 'image/png' in o.data]
        assert len(figures) == 1
    assert (tmp_path / 'pulses_ana.csv').is_file()


# =============================================================================================
# 03_plot_waveforms.py and 04_plot_iqscan.py: the figures of kidpack-monitor and kidpack-iqplot
# =============================================================================================

def edited(tmp_path, macro, **changes):
    """The macro with some settings edited, as a student does: {'FILENAME = None': "FILENAME = 'x.npz'"}."""
    source = macro.read_text(encoding='utf-8')
    for old, new in changes.items():
        assert source.count(old) == 1, old  # the line to edit is there, once
        source = source.replace(old, new)
    path = tmp_path / f'edited_{macro.name}'
    path.write_text(source, encoding='utf-8')
    return path


def filename_setting(path):
    """The edit that writes the file name after FILENAME = (at the start of a line: REF_FILENAME is another line)."""
    return {'\nFILENAME = None': f'\nFILENAME = {str(path)!r}'}


def run_macro(monkeypatch, macro, *argv):
    monkeypatch.setattr(sys, 'argv', [macro.name, *map(str, argv)])
    return runpy.run_path(str(macro), run_name='__main__')


def daq_file(tmp_path, events=40, npts=2000):
    """A raw file written by the DAQ (simulator): has event_id, noise and a pulse in every event."""
    cfg = DaqConfig(run_number=1, events_per_file=events, num_files=1, output_dir=str(tmp_path / 'daq'),
                    npts=npts)
    run_daq(cfg, SimulatedScope(cfg, seed=3))
    return tmp_path / 'daq' / 'run_01' / 'data' / 'run01-00.npz'


def waveform_file(tmp_path, name='pulses.npz'):
    write_pulses(tmp_path / name, nwf=20)  # no event_id: like a file of the old DAQ macro
    return tmp_path / name


F_SCAN = np.linspace(5.324e9, 5.328e9, 41)


def scan_file(tmp_path, name='scan.npz', dip=True, delay=50e-9, power=None, gain=0.01):
    """A notch resonator times a cable delay and a gain, stored like kidpack-iqscan / the old iq_scan.py."""
    x = 2 * (F_SCAN - 5.326e9) / 0.4e6
    true = (1 - (0.6 / (1 + 1j * x) if dip else 0)) * np.exp(-2j * np.pi * F_SCAN * delay)
    iq = gain * true
    extra = {} if power is None else {'power_dbm': power}
    np.savez(tmp_path / name, dd=np.column_stack([F_SCAN, iq.real, iq.imag]), **extra)
    return tmp_path / name


def same_lines(a, b, rtol=1e-5, atol=1e-6):
    assert len(a.lines) == len(b.lines)
    for la, lb in zip(a.lines, b.lines):
        np.testing.assert_allclose(la.get_xdata(), lb.get_xdata(), rtol=rtol, atol=atol)
        np.testing.assert_allclose(la.get_ydata(), lb.get_ydata(), rtol=rtol, atol=atol)
        assert la.get_label() == lb.get_label() and la.get_marker() == lb.get_marker()


# --- 03: the same figures as kidpack-monitor ------------------------------------------------------

def monitor_figures(path, stride=1):
    raw = load_raw(path)
    view = make_pulse_view(raw, select_events(raw.nevents, stride), rebin_factor=5)
    return plot_iq_plane(view), plot_waveforms(view)


@pytest.mark.parametrize('stride', [1, 2])
def test_03_draws_what_kidpack_monitor_draws(monkeypatch, tmp_path, stride):
    path = daq_file(tmp_path)
    changes = {} if stride == 1 else {'STRIDE = 1 ': f'STRIDE = {stride} '}
    ns = run_macro(monkeypatch, edited(tmp_path, WAVEFORMS, **changes), path)
    pc1, pc2 = monitor_figures(path, stride)

    assert ns['ax1'].shape == ns['ax2'].shape == (4, 4)
    for k in range(16):
        mine1, mine2, ref1, ref2 = ns['ax1'].flat[k], ns['ax2'].flat[k], pc1.axes[k], pc2.axes[k]
        assert mine1.get_title() == ref1.get_title() == mine2.get_title() == ref2.get_title()
        for mine, ref in ((mine1, ref1), (mine2, ref2)):
            assert (mine.get_xlabel(), mine.get_ylabel()) == (ref.get_xlabel(), ref.get_ylabel())
        same_lines(mine1, ref1)  # pedestal and peak
        same_lines(mine2, ref2)  # I, Q, Proj and the two stars
        np.testing.assert_allclose(mine1.collections[0].get_offsets(), ref1.collections[0].get_offsets(),
                                   rtol=1e-5, atol=1e-4)  # the points of the IQ plane (mV)
        np.testing.assert_allclose(mine1.collections[0].get_array(), ref1.collections[0].get_array(), rtol=1e-5)
    assert [a.get_title() for a in ns['ax1'].flat][:3] == [f'Event #{stride * i}' for i in range(3)]


def test_03_the_red_star_of_the_iq_plane_is_the_star_of_the_waveforms(monkeypatch, tmp_path):
    ns = run_macro(monkeypatch, WAVEFORMS, daq_file(tmp_path))
    for k in range(16):
        star_iq = ns['ax1'].flat[k].lines[1]  # (I, Q) of the peak
        star_i, star_q = ns['ax2'].flat[k].lines[3], ns['ax2'].flat[k].lines[4]
        assert star_iq.get_xdata()[0] == pytest.approx(star_i.get_ydata()[0])
        assert star_iq.get_ydata()[0] == pytest.approx(star_q.get_ydata()[0])
        assert star_i.get_xdata()[0] == star_q.get_xdata()[0]  # at the same time
        proj = ns['ax2'].flat[k].lines[2].get_ydata()
        assert proj.max() == pytest.approx(np.max(np.abs(ns['vc'][k])) * 1e3, rel=1e-5)  # proj max, in mV


def test_03_known_pulses_give_known_waveforms(monkeypatch, tmp_path):
    truth = write_pulses(tmp_path / 'pulses.npz', nwf=16)
    ns = run_macro(monkeypatch, WAVEFORMS, tmp_path / 'pulses.npz')
    for k in (0, 7, 15):
        np.testing.assert_allclose(ns['ped0'][k], truth['ped_i'][k], atol=1e-6)
        np.testing.assert_allclose(ns['ped1'][k], truth['ped_q'][k], atol=1e-6)
        peak_phase = np.angle(ns['vc'][k, ns['peak_bin'][k]])
        assert np.exp(1j * peak_phase) == pytest.approx(np.exp(1j * truth['theta'][k]), abs=1e-3)
        assert ns['proj'][k].min() > -1e-4  # the pulse is all in the direction of the peak: no negative part
    assert [a.get_title() for a in ns['ax1'].flat] == [f'Event #{i}' for i in range(16)]  # no event_id: 0, 1, ...


def test_03_fewer_than_16_events_leave_the_other_panels_blank(monkeypatch, tmp_path):
    path = daq_file(tmp_path, events=5)
    ns = run_macro(monkeypatch, WAVEFORMS, path)
    pc1, pc2 = monitor_figures(path)
    for k in range(16):
        assert ns['ax1'].flat[k].axison == ns['ax2'].flat[k].axison == (k < 5)
        assert (ns['ax2'].flat[k].get_xlabel(), ns['ax2'].flat[k].get_ylabel()) == (
            pc2.axes[k].get_xlabel(), pc2.axes[k].get_ylabel())  # the x label is on the last panel of a column
    assert ns['nplot'] == 5


def test_03_saves_the_figures_only_if_asked(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run_macro(monkeypatch, WAVEFORMS, waveform_file(tmp_path))
    assert not (tmp_path / 'pc1.png').exists() and not (tmp_path / 'pc2.png').exists()
    run_macro(monkeypatch, edited(tmp_path, WAVEFORMS, **{'SAVE = False': 'SAVE = True'}), tmp_path / 'pulses.npz')
    assert (tmp_path / 'pc1.png').stat().st_size > 10000 and (tmp_path / 'pc2.png').stat().st_size > 10000


def test_03_opens_two_figures_with_one_show(monkeypatch, tmp_path, shown):
    run_macro(monkeypatch, WAVEFORMS, waveform_file(tmp_path))
    assert len(shown) == 1 and len(shown[0]) == 2


def test_03_refuses_what_it_cannot_do(monkeypatch, tmp_path):
    write_pulses(tmp_path / 'notrigger.npz', nwf=4, ref_position=0.0)
    with pytest.raises(SystemExit, match='ref_position'):
        run_macro(monkeypatch, WAVEFORMS, tmp_path / 'notrigger.npz')
    for setting in ('REBIN = 0', 'STRIDE = 0'):
        old = setting.split()[0] + ' = '
        source = WAVEFORMS.read_text(encoding='utf-8')
        line = next(x for x in source.splitlines() if x.startswith(old))
        macro = edited(tmp_path, WAVEFORMS, **{line: setting + '  # test'})
        with pytest.raises(SystemExit, match='1 以上'):
            run_macro(monkeypatch, macro, waveform_file(tmp_path))


# --- 04: the same figure as kidpack-iqplot -----------------------------------------------------------

def same_scan_figure(mine, ref):
    assert len(mine.axes) == len(ref.axes) == 3
    for a, b in zip(mine.axes, ref.axes):
        same_lines(a, b, rtol=1e-9, atol=1e-12)
        assert (a.get_xlabel(), a.get_ylabel()) == (b.get_xlabel(), b.get_ylabel())
    assert mine._suptitle.get_text() == ref._suptitle.get_text()


@pytest.mark.parametrize('db, wrap', [(False, False), (True, False), (False, True)])
def test_04_draws_what_kidpack_iqplot_draws(monkeypatch, tmp_path, db, wrap):
    path = scan_file(tmp_path, delay=2e-6, power=-12.5)  # 2 us of cable: the phase turns many times
    changes = {}
    if db:
        changes['DB = False'] = 'DB = True'
    if wrap:
        changes['WRAP = False'] = 'WRAP = True'
    ns = run_macro(monkeypatch, edited(tmp_path, IQSCAN, **changes), path)
    same_scan_figure(ns['fig'], plot_scan(load_iqscan(path), db=db, wrap=wrap))
    phase = ns['ax'][2].lines[0].get_ydata()
    assert (np.abs(phase) <= np.pi + 1e-12).all() if wrap else (phase.max() - phase.min() > 4 * np.pi)
    assert '-12.5 dBm' in ns['fig']._suptitle.get_text() and 'uncalibrated' in ns['fig']._suptitle.get_text()


def test_04_reads_the_files_of_the_old_iq_scan_py(monkeypatch, tmp_path):
    path = scan_file(tmp_path)  # only dd, no power_dbm
    ns = run_macro(monkeypatch, IQSCAN, path)
    assert 'dBm' not in ns['fig']._suptitle.get_text() and ns['power_dbm'] is None
    # mV, not V: the dip is the resonator (0.4 of the off-resonance size), the circle is 10 mV across
    magnitude = ns['ax'][1].lines[0].get_ydata()
    assert magnitude.min() == pytest.approx(0.4 * 10, rel=1e-3) and magnitude.max() < 10.3


def test_04_with_a_reference_it_draws_s21(monkeypatch, tmp_path):
    scan = scan_file(tmp_path, 'scan.npz', delay=80e-9, gain=0.02)
    ref = scan_file(tmp_path, 'through.npz', dip=False, delay=80e-9, gain=0.02)  # the same cable and gain, no resonator
    ns = run_macro(monkeypatch, IQSCAN, scan, ref)
    x = 2 * (F_SCAN - 5.326e9) / 0.4e6
    true = 1 - 0.6 / (1 + 1j * x)  # the cable and the gain cancel
    np.testing.assert_allclose(ns['ax'][1].lines[0].get_ydata(), np.abs(true), rtol=1e-9)
    np.testing.assert_allclose(ns['ax'][0].lines[0].get_xdata(), true.real, atol=1e-12)
    np.testing.assert_allclose(ns['ax'][2].lines[0].get_ydata(), np.unwrap(np.angle(true)), atol=1e-9)
    assert ns['ax'][1].get_ylabel() == '|S21|' and 'ref through' in ns['fig']._suptitle.get_text()
    same_scan_figure(ns['fig'], plot_scan(load_iqscan(scan), load_iqscan(ref)))


def test_04_reference_is_given_as_the_second_argument_or_REF_FILENAME(monkeypatch, tmp_path):
    scan_file(tmp_path, 'scan.npz')
    ref = scan_file(tmp_path, 'through.npz', dip=False)
    by_argument = run_macro(monkeypatch, IQSCAN, tmp_path / 'scan.npz', ref)['ax'][1].lines[0].get_ydata()
    macro = edited(tmp_path, IQSCAN, **{'REF_FILENAME = None': f'REF_FILENAME = {str(ref)!r}'})
    by_variable = run_macro(monkeypatch, macro, tmp_path / 'scan.npz')['ax'][1].lines[0].get_ydata()
    np.testing.assert_array_equal(by_argument, by_variable)


def test_04_a_reference_at_other_frequencies_is_refused(monkeypatch, tmp_path):
    scan_file(tmp_path, 'scan.npz')
    other = tmp_path / 'other.npz'
    np.savez(other, dd=np.column_stack([F_SCAN + 1e6, np.ones(41), np.zeros(41)]))
    with pytest.raises(SystemExit, match='周波数が違います'):
        run_macro(monkeypatch, IQSCAN, tmp_path / 'scan.npz', other)
    short = tmp_path / 'short.npz'
    np.savez(short, dd=np.column_stack([F_SCAN[:5], np.ones(5), np.zeros(5)]))
    with pytest.raises(SystemExit, match='周波数が違います'):
        run_macro(monkeypatch, IQSCAN, tmp_path / 'scan.npz', short)


def test_04_a_file_that_is_not_an_iq_scan_is_reported(monkeypatch, tmp_path):
    np.savez(tmp_path / 'wf.npz', ch0=np.zeros((2, 5)))
    with pytest.raises(SystemExit, match='"dd" がありません'):
        run_macro(monkeypatch, IQSCAN, tmp_path / 'wf.npz')
    np.savez(tmp_path / 'bad.npz', dd=np.zeros((4, 2)))
    with pytest.raises(SystemExit, match='形は'):
        run_macro(monkeypatch, IQSCAN, tmp_path / 'bad.npz')
    scan = scan_file(tmp_path)
    with pytest.raises(SystemExit, match='IQ スキャンのファイルではありません'):
        run_macro(monkeypatch, IQSCAN, scan, tmp_path / 'wf.npz')


def test_04_saves_the_figure_only_if_asked(monkeypatch, tmp_path, shown):
    scan = scan_file(tmp_path)
    run_macro(monkeypatch, IQSCAN, scan)
    assert list(tmp_path.glob('*.png')) == [] and len(shown) == 1 and len(shown[0]) == 1
    target = tmp_path / 'figure.png'
    run_macro(monkeypatch, edited(tmp_path, IQSCAN, **{'SAVE_PNG = None': f'SAVE_PNG = {str(target)!r}'}), scan)
    assert target.stat().st_size > 10000


# --- 03 and 04 in a notebook: the same rules as 02 -----------------------------------------------------

NEW_MACROS = [pytest.param(WAVEFORMS, waveform_file, id='03_plot_waveforms'),
              pytest.param(IQSCAN, scan_file, id='04_plot_iqscan')]


@pytest.mark.parametrize('macro, make_file', NEW_MACROS)
def test_new_macros_print_the_usage_for_no_file_or_too_many(monkeypatch, capsys, tmp_path, macro, make_file):
    for argv in ([], ['a.npz', 'b.npz', 'c.npz']) if macro == IQSCAN else ([], ['a.npz', 'b.npz']):
        with pytest.raises(SystemExit) as exit_info:
            run_macro(monkeypatch, macro, *argv)
        assert exit_info.value.code == 1 and '使い方' in capsys.readouterr().out


@pytest.mark.parametrize('macro, make_file', NEW_MACROS)
def test_new_macros_in_a_notebook_without_FILENAME_say_where_to_write_it(monkeypatch, capsys, macro, make_file):
    monkeypatch.setattr(sys, 'argv', JUPYTER_ARGV)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(macro), run_name='__main__')
    out = capsys.readouterr().out
    assert exit_info.value.code == 1 and '使い方' in out and 'FILENAME' in out


@pytest.mark.parametrize('macro, make_file', NEW_MACROS)
def test_new_macros_report_a_missing_file_in_plain_words(monkeypatch, tmp_path, macro, make_file):
    with pytest.raises(SystemExit, match='ファイルが見つかりません'):
        run_macro(monkeypatch, macro, tmp_path / 'nothing.npz')
    edited_macro = edited(tmp_path, macro, **filename_setting(tmp_path / 'nothing.npz'))
    monkeypatch.setattr(sys, 'argv', JUPYTER_ARGV)
    with pytest.raises(SystemExit, match='ファイルが見つかりません'):
        runpy.run_path(str(edited_macro), run_name='__main__')


@pytest.mark.parametrize('macro, make_file', NEW_MACROS)
def test_new_macros_take_the_file_name_from_FILENAME_in_a_notebook(monkeypatch, tmp_path, macro, make_file):
    path = make_file(tmp_path)
    edited_macro = edited(tmp_path, macro, **filename_setting(path))
    monkeypatch.setattr(sys, 'argv', JUPYTER_ARGV)  # Jupyter's own arguments are not the macro's
    namespace = runpy.run_path(str(edited_macro), run_name='__main__')
    assert namespace['filename'] == str(path)


@pytest.mark.parametrize('macro, make_file', NEW_MACROS)
def test_new_macros_prefer_an_argument_of_the_command_line_to_FILENAME(monkeypatch, tmp_path, macro, make_file):
    one, two = make_file(tmp_path, 'one.npz'), make_file(tmp_path, 'two.npz')
    edited_macro = edited(tmp_path, macro, **filename_setting(one))
    namespace = run_macro(monkeypatch, edited_macro, two)
    assert namespace['filename'] == str(two)


@pytest.mark.parametrize('macro, make_file', NEW_MACROS)
def test_new_macros_in_a_real_jupyter_kernel(tmp_path, macro, make_file):
    """Pasted into a cell (with FILENAME) and %run with a file name, in a real kernel; skipped without Jupyter."""
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    path = make_file(tmp_path)
    pasted = edited(tmp_path, macro, **filename_setting(path)).read_text(encoding='utf-8')
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(c) for c in (pasted, f'%run {macro} {path}', pasted)]
    nbclient.NotebookClient(notebook, timeout=240, kernel_name='python3', allow_errors=False,
                            resources={'metadata': {'path': str(tmp_path)}}).execute()
    figures_per_cell = 2 if macro == WAVEFORMS else 1
    for cell in notebook.cells:
        figures = [o for o in cell.outputs if o.output_type == 'display_data' and 'image/png' in o.data]
        assert len(figures) == figures_per_cell
