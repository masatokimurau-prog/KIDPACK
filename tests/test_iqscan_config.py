import numpy as np
import pytest

from kidpack.iqscan.config import build_parser, config_from_args, default_scan_dir

REQUIRED = ['--f-start', '5.324e9', '--f-stop', '5.328e9', '--num-points', '51']


def parse(*extra, required=REQUIRED):
    parser = build_parser()
    return config_from_args(parser.parse_args([*required, *extra]), parser)


def test_defaults_reproduce_the_seed_iq_scan_py():
    cfg = parse()
    assert cfg.power == -10
    assert cfg.sample_rate == 1e4 and cfg.npts == 5000  # 0.5 s at 10 kS/s
    assert cfg.num_records == 2 and cfg.ref_position == 50
    assert cfg.ch0.vertical_range == cfg.ch1.vertical_range == 0.1
    assert cfg.ch0.coupling == cfg.ch1.coupling == 'dc'
    assert cfg.ch0.offset == cfg.ch1.offset == 0.0
    assert cfg.impedance == 50 and cfg.fetch_timeout == 17
    assert cfg.resource == 'PXI2Slot2' and cfg.sg_resource == 'PXI1Slot3'
    assert cfg.settle_time == 0 and cfg.backend == 'niscope'


@pytest.mark.parametrize('missing', ['--f-start', '--f-stop', '--num-points'])
def test_the_scan_range_has_no_default(missing):
    args = list(REQUIRED)
    i = args.index(missing)
    del args[i:i + 2]
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)


def test_frequencies_are_evenly_spaced_with_both_ends_included():
    freqs = parse().frequencies()
    np.testing.assert_array_equal(freqs, np.linspace(5.324e9, 5.328e9, 51))
    assert freqs[0] == 5.324e9 and freqs[-1] == 5.328e9
    assert list(parse(required=['--f-start', '1e9', '--f-stop', '2e9', '--num-points', '1']).frequencies()) == [1e9]
    # a descending scan is allowed
    assert parse(required=['--f-start', '2e9', '--f-stop', '1e9', '--num-points', '3']).frequencies()[1] == 1.5e9


def test_record_length_from_time_window_or_npts():
    assert parse('--time-window', '1').npts == 10000
    assert parse('--sample-rate', '2e4', '--time-window', '0.25').npts == 5000
    assert parse('--npts', '123').npts == 123
    with pytest.raises(SystemExit):
        parse('--npts', '10', '--time-window', '1')


def test_every_hardcoded_setting_can_be_overridden():
    cfg = parse('--power', '-25', '--settle-time', '0.1', '--num-records', '4',
                '--ref-position', '20', '--resource', 'PXI1Slot9', '--sg-resource', 'PXI1Slot5',
                '--ch0-range', '0.5', '--ch0-coupling', 'ac', '--ch0-offset', '-0.01',
                '--ch1-range', '0.2', '--ch1-coupling', 'gnd', '--ch1-offset', '0.02',
                '--impedance', '1e6', '--fetch-timeout', '60', '--condition', 'temp 5.5K',
                '--name', 'my_scan', '--output-dir', '/x', '--summary-file', '/y.txt')
    assert (cfg.power, cfg.settle_time, cfg.num_records, cfg.ref_position) == (-25, 0.1, 4, 20)
    assert (cfg.resource, cfg.sg_resource) == ('PXI1Slot9', 'PXI1Slot5')
    assert (cfg.ch0.vertical_range, cfg.ch0.coupling, cfg.ch0.offset) == (0.5, 'ac', -0.01)
    assert (cfg.ch1.vertical_range, cfg.ch1.coupling, cfg.ch1.offset) == (0.2, 'gnd', 0.02)
    assert (cfg.impedance, cfg.fetch_timeout) == (1e6, 60)
    assert cfg.condition == 'temp 5.5K' and cfg.name == 'my_scan'
    assert cfg.output_dir == '/x' and cfg.summary_path == '/y.txt'


def test_negative_numbers_including_exponent_notation_are_accepted():
    cfg = parse('--power', '-30', '--ch0-offset', '-1e-3', '--ch1-offset=-2.5E-3')
    assert cfg.power == -30 and cfg.ch0.offset == -1e-3 and cfg.ch1.offset == -2.5e-3


def test_scans_are_saved_under_kidpack_data_iqscan_by_default():
    assert default_scan_dir().replace('\\', '/').endswith('/data/iqscan')
    cfg = parse()
    assert cfg.output_dir == default_scan_dir()
    assert cfg.summary_path == default_scan_dir() + '/iqscan_summary.txt'


@pytest.mark.parametrize('bad', [
    ['--num-points', '0'], ['--f-start', '0'], ['--f-stop', '-1e9'], ['--sample-rate', '0'],
    ['--num-records', '0'], ['--ref-position', '101'], ['--ch1-range', '0'],
    ['--fetch-timeout', '0'], ['--settle-time', '-1'], ['--impedance', '0'],
    ['--power', 'nan'], ['--name', '../escape'], ['--name', 'a/b'], ['--name', ''],
    ['--ch0-coupling', 'xx'],
])
def test_invalid_values_are_rejected(bad):
    with pytest.raises(SystemExit):
        parse(*bad)
