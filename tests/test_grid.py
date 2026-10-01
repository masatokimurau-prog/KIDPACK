"""The plots must have their grid whatever the matplotlib settings of the PC are.

``ax.grid()`` without an argument *toggles* the grid, so on a PC whose matplotlibrc
already says ``axes.grid : True`` it switched the grid off. Every plot calls
``ax.grid(True)`` instead; these tests run all plots under both settings.
"""
import runpy
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kidpack.iqscan.plot import plot_scan
from kidpack.iqscan.reader import IqScan
from kidpack.monitor.pulses import make_pulse_view, plot_iq_plane, plot_waveforms
from kidpack.rawdata import RawData

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


@pytest.fixture(params=[False, True], ids=['axes.grid=False', 'axes.grid=True'])
def pc_setting(request):
    """matplotlib settings of a PC where the grid is off / already on by default."""
    with matplotlib.rc_context({'axes.grid': request.param}):
        yield request.param


def plotted_axes(fig):
    """The axes that show a plot (not the colour bars, not the blank panels)."""
    return [ax for ax in fig.axes if ax.axison and ax.get_label() != '<colorbar>']


def has_grid(ax):
    return (ax.xaxis.get_gridlines()[0].get_visible() and ax.yaxis.get_gridlines()[0].get_visible())


def assert_all_gridded(fig):
    axes = plotted_axes(fig)
    assert axes
    missing = [ax.get_ylabel() or ax.get_title() for ax in axes if not has_grid(ax)]
    assert not missing, f'no grid in: {missing}'


def test_the_matplotlib_setting_really_changes_the_default(pc_setting):
    fig, ax = plt.subplots()
    assert has_grid(ax) is pc_setting  # (so that the tests below test something)


def test_iq_scan_plot_has_a_grid_in_all_three_panels(pc_setting):
    f = np.linspace(5.324e9, 5.328e9, 41)
    iq = 0.01 * (1 - 0.6 / (1 + 2j * (f - 5.326e9) / 0.4e6))
    fig = plot_scan(IqScan('/x/scan.npz', f, iq.real, iq.imag))
    assert len(plotted_axes(fig)) == 3
    assert_all_gridded(fig)


def test_monitor_plots_have_a_grid_in_every_panel(pc_setting):
    rng = np.random.default_rng(0)
    raw = RawData('x.npz', rng.normal(0, 1e-3, (16, 1000)).astype(np.float32),
                  rng.normal(0, 1e-3, (16, 1000)).astype(np.float32), np.arange(16), 1000, 1e9, 20.0)
    view = make_pulse_view(raw, np.arange(16))
    for make_figure in (plot_iq_plane, plot_waveforms):
        fig = make_figure(view)
        assert len(plotted_axes(fig)) == 16
        assert_all_gridded(fig)


def test_the_toy_mc_sample_has_a_grid_in_all_its_figures(pc_setting, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['01_kid_response_toymc.py'])
    runpy.run_path(str(EXAMPLES / '01_kid_response_toymc.py'), run_name='__main__')
    assert len(plt.get_fignums()) == 4
    for number in plt.get_fignums():
        assert_all_gridded(plt.figure(number))


def test_the_pulse_analysis_sample_has_a_grid_in_its_histograms(pc_setting, monkeypatch, tmp_path):
    s = np.arange(5000)
    pulse = np.where(s >= 1000, np.exp(-np.clip(s - 1000, 0, None) / 300.0), 0.0)
    rng = np.random.default_rng(1)
    np.savez(tmp_path / 'p.npz', ch0=(0.02 + 0.005 * pulse + 1e-4 * rng.standard_normal((20, 5000))).astype(np.float32),
             ch1=(-0.03 + 0.002 * pulse + 1e-4 * rng.standard_normal((20, 5000))).astype(np.float32),
             npts=np.int32(5000), sample_rate=np.float64(2.5e9), ref_position=np.float64(20.0))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['02_pulse_analysis.py', str(tmp_path / 'p.npz')])
    runpy.run_path(str(EXAMPLES / '02_pulse_analysis.py'), run_name='__main__')
    fig = plt.figure(plt.get_fignums()[0])
    assert len(plotted_axes(fig)) == 2
    assert_all_gridded(fig)
