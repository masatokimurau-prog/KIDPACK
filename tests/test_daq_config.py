from pathlib import Path

import pytest

from kidpack.daq.config import build_parser, config_from_args, default_output_dir

REQUIRED = ['--run-number', '3', '--events-per-file', '1000', '--num-files', '2']


def parse(*extra):
    parser = build_parser()
    return config_from_args(parser.parse_args([*REQUIRED, *extra]), parser)


def test_defaults_are_the_original_kid_py_values_except_the_trigger_coupling():
    cfg = parse()
    assert (cfg.run_number, cfg.events_per_file, cfg.num_files) == (3, 1000, 2)
    assert cfg.resource == 'PXI2Slot2'
    assert cfg.sample_rate == 2.5e9
    assert cfg.npts == 5000  # 2.5 GS/s x 2 us
    assert cfg.ref_position == 20
    assert cfg.ch0.vertical_range == cfg.ch1.vertical_range == 0.01
    assert cfg.ch0.coupling == cfg.ch1.coupling == 'dc'
    assert cfg.ch0.offset == cfg.ch1.offset == 0.0
    assert cfg.impedance == 50
    assert cfg.bandwidth == -1
    assert (cfg.trigger.source, cfg.trigger.level) == ('VAL_EXTERNAL', 2.2)
    assert (cfg.trigger.slope, cfg.trigger.coupling) == ('positive', 'dc')  # kid.py had LF_REJECT
    assert cfg.fetch_timeout == 100
    assert cfg.sg is None
    assert cfg.backend == 'niscope'


@pytest.mark.parametrize('missing', ['--run-number', '--events-per-file', '--num-files'])
def test_required_arguments(missing):
    args = list(REQUIRED)
    i = args.index(missing)
    del args[i:i + 2]
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)


def test_record_length_from_time_window_or_npts():
    assert parse('--time-window', '5e-6').npts == 12500
    assert parse('--sample-rate', '1e9', '--time-window', '1e-6').npts == 1000
    assert parse('--npts', '123').npts == 123
    with pytest.raises(SystemExit):
        parse('--npts', '10', '--time-window', '1e-6')


def test_every_hardcoded_setting_can_be_overridden():
    cfg = parse('--resource', 'PXI1Slot9', '--ref-position', '50',
                '--ch0-range', '0.2', '--ch0-coupling', 'ac', '--ch0-offset', '-0.014',
                '--ch1-range', '0.05', '--ch1-coupling', 'gnd', '--ch1-offset', '0.001',
                '--impedance', '1e6', '--bandwidth', '175e6', '--fetch-timeout', '5',
                '--trigger-source', '1', '--trigger-level', '-8e-5',
                '--trigger-slope', 'negative', '--trigger-coupling', 'dc',
                '--condition', 'temp 5.5K', '--output-dir', '/x', '--summary-file', '/y.txt')
    assert cfg.resource == 'PXI1Slot9' and cfg.ref_position == 50
    assert (cfg.ch0.vertical_range, cfg.ch0.coupling, cfg.ch0.offset) == (0.2, 'ac', -0.014)
    assert (cfg.ch1.vertical_range, cfg.ch1.coupling, cfg.ch1.offset) == (0.05, 'gnd', 0.001)
    assert (cfg.impedance, cfg.bandwidth, cfg.fetch_timeout) == (1e6, 175e6, 5)
    assert cfg.trigger.source == '1' and cfg.trigger.level == -8e-5
    assert (cfg.trigger.slope, cfg.trigger.coupling) == ('negative', 'dc')
    assert cfg.condition == 'temp 5.5K'
    assert cfg.output_dir == '/x' and cfg.summary_path == '/y.txt'


def test_negative_numbers_are_accepted_as_values():
    cfg = parse('--bandwidth', '-1', '--ch1-offset', '-0.014')
    assert cfg.bandwidth == -1 and cfg.ch1.offset == -0.014


def test_negative_numbers_in_exponent_notation_are_accepted_as_values():
    # plain argparse reads "-8e-5" as an option and fails
    cfg = parse('--trigger-level', '-8e-5', '--ch0-offset', '-1.4E-2', '--ch1-offset=-2e-3')
    assert cfg.trigger.level == -8e-5
    assert cfg.ch0.offset == -1.4e-2 and cfg.ch1.offset == -2e-3


def test_edge_trigger_is_the_default_mode():
    assert parse().trigger.mode == 'edge'


def test_random_trigger_defaults_to_once_per_second():
    cfg = parse('--random-trigger')
    assert (cfg.trigger.mode, cfg.trigger.interval) == ('random', 1.0)
    assert parse('--random-trigger', '--random-trigger-interval', '0.25').trigger.interval == 0.25


@pytest.mark.parametrize('bad', [
    ['--random-trigger-interval', '2'],  # without --random-trigger
    ['--random-trigger', '--trigger-level', '1.0'],  # edge settings are meaningless
    ['--random-trigger', '--trigger-source', '1'],
    ['--random-trigger', '--trigger-slope', 'negative'],
    ['--random-trigger', '--trigger-coupling', 'lf_reject'],  # not the default (dc) any more
    ['--random-trigger', '--random-trigger-interval', '0'],
])
def test_random_trigger_option_conflicts(bad):
    with pytest.raises(SystemExit):
        parse(*bad)


def test_runs_are_saved_under_kidpack_data_by_default():
    out = Path(default_output_dir())
    assert out.name == 'data' and out.is_absolute()
    assert (out.parent / 'pyproject.toml').is_file()  # <checkout>/data, whatever the cwd
    assert parse().output_dir == str(out)


def test_default_output_dir_does_not_depend_on_the_current_directory(tmp_path, monkeypatch):
    expected = default_output_dir()
    monkeypatch.chdir(tmp_path)
    assert default_output_dir() == expected == parse().output_dir


def test_summary_path_defaults_to_output_dir():
    assert parse('--output-dir', 'out').summary_path.endswith('out/run_summary.txt')


def test_sg_is_optional_but_all_or_nothing():
    cfg = parse('--sg-frequency', '5.49e9', '--sg-power', '-12')
    assert (cfg.sg.frequency, cfg.sg.power, cfg.sg.resource) == (5.49e9, -12, 'PXI2Slot3')
    assert parse('--sg-frequency', '5e9', '--sg-power', '0', '--sg-resource', 'PXI1Slot5').sg.resource == 'PXI1Slot5'
    for bad in (['--sg-frequency', '5e9'], ['--sg-power', '-10'], ['--sg-resource', 'PXI1Slot5']):
        with pytest.raises(SystemExit):
            parse(*bad)


@pytest.mark.parametrize('bad', [
    ['--run-number', '-1'], ['--events-per-file', '0'], ['--num-files', '0'],
    ['--sample-rate', '0'], ['--ref-position', '101'], ['--ch0-range', '0'],
    ['--fetch-timeout', '0'], ['--ch0-coupling', 'xx'], ['--trigger-slope', 'up'],
])
def test_invalid_values_are_rejected(bad):
    parser = build_parser()
    args = [*REQUIRED, *bad]
    # later occurrences of an option override earlier ones
    with pytest.raises(SystemExit):
        config_from_args(parser.parse_args(args), parser)


def test_the_trigger_coupling_can_still_be_set_to_every_value_and_defaults_to_dc():
    assert parse().trigger.coupling == 'dc'
    for value in ('lf_reject', 'hf_reject', 'dc', 'ac', 'ac_plus_hf_reject'):
        assert parse('--trigger-coupling', value).trigger.coupling == value
