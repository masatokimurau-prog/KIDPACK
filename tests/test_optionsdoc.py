"""docs/OPTIONS.md must say what the commands really accept."""
from pathlib import Path

import re

import pytest

from kidpack import optionsdoc
from kidpack.daq.config import build_parser as daq_parser, config_from_args as daq_config
from kidpack.iqscan.config import build_parser as iqscan_parser, config_from_args as iqscan_config
from kidpack.iqscan.plot import main as iqplot_main
from kidpack.monitor.cli import main as monitor_main
from kidpack.optionsdoc import ALLOWED, COMMANDS, RULES, command_options, render

DOC = Path(__file__).resolve().parents[1] / 'docs' / 'OPTIONS.md'

BASE = {
    'kidpack-daq': ['--events-per-file', '1', '--num-files', '1'],
    'kidpack-iqscan': ['--f-start', '5e9', '--f-stop', '6e9', '--num-points', '3'],
    'kidpack-monitor': [],
    'kidpack-iqplot': [],
}


@pytest.fixture
def accepted(tmp_path):
    """accepted(command, argv): does the command take these arguments (parsing and validation)?

    monitor / iqplot are run on a directory without data: accepted arguments end with
    'nothing to plot' (return code 1); rejected ones exit with a usage error.
    """
    nowhere = str(tmp_path / 'nothing')

    def check(command, argv):
        argv = list(argv)
        try:
            if command == 'kidpack-daq':
                parser = daq_parser()  # (an empty output directory: the run number is taken from it)
                daq_config(parser.parse_args(BASE[command] + ['--output-dir', nowhere] + argv), parser)
            elif command == 'kidpack-iqscan':
                parser = iqscan_parser()
                iqscan_config(parser.parse_args(BASE[command] + argv), parser)
            elif command == 'kidpack-monitor':
                monitor_main(argv + ['--data-dir', nowhere])
            else:
                iqplot_main(argv + ['--data-dir', nowhere])
        except SystemExit:
            return False
        return True

    return check


def section_of(text, name):
    """The part of the document that describes one command."""
    section = text[text.index(f'## {name}\n'):]
    return section[:section.index('\n## ')] if '\n## ' in section else section


def test_every_table_row_has_exactly_four_cells():
    """A stray | (e.g. in a metavar like N|test) would split a cell and shift the columns."""
    text = DOC.read_text(encoding='utf-8')
    # the rows of the option tables (the overview table at the top has two columns by design)
    rows = [line for line in text.splitlines() if line.startswith('| `')]
    assert len(rows) > 50
    assert any(r'N\|test' in row for row in rows)  # the case this test is about
    for row in rows:
        assert len(re.split(r'(?<!\\)\|', row)) == 6, row  # '' | 4 cells | ''


def test_the_committed_document_is_what_the_parsers_generate():
    assert DOC.read_text(encoding='utf-8') == render(), (
        'docs/OPTIONS.md is out of date: run  python -m kidpack.optionsdoc --output docs/OPTIONS.md')


def test_every_option_of_every_command_has_a_row_and_nothing_else_does():
    text = DOC.read_text(encoding='utf-8')
    documented = set()
    for name, _, builder in COMMANDS:
        section = section_of(text, name)
        for option in command_options(name, builder):
            assert f'| `{option}`' in section or f'| `{option}` (positional)' in section, (name, option)
            documented.add((name, option))
    assert set(ALLOWED) <= documented  # no note for an option that no longer exists


def test_the_notes_cover_exactly_the_options_without_a_fixed_list_of_values():
    for name, _, builder in COMMANDS:
        for option, action in command_options(name, builder).items():
            free = action.nargs != 0 and not action.choices
            assert ((name, option) in ALLOWED) == free, (name, option)


def test_choices_and_defaults_in_the_document_are_those_of_the_parsers():
    text = DOC.read_text(encoding='utf-8')
    for name, _, builder in COMMANDS:
        for option, action in command_options(name, builder).items():
            row = next(line for line in section_of(text, name).splitlines() if line.startswith(f'| `{option}`'))
            for choice in action.choices or ():
                assert f'`{choice}`' in row
            if action.default not in (None, False) and not action.required and not isinstance(action.default, str):
                assert optionsdoc._number(action.default) in row, (name, option)


def test_the_trigger_coupling_of_the_daq_defaults_to_dc_in_the_document():
    row = next(line for line in section_of(DOC.read_text(), 'kidpack-daq').splitlines()
               if line.startswith('| `--trigger-coupling`'))
    assert row.split('|')[3].strip() == '`dc`'
    assert '`lf_reject`' in row and '`ac_plus_hf_reject`' in row  # and the other values stay allowed


# --- the allowed values of the free options, checked against the real commands -------------------

def v(option, bad=(), ok=(), companions=()):
    return {'bad': [[*companions, option, x] for x in bad], 'ok': [[*companions, option, x] for x in ok]}


def numbers(option, bad, ok):
    return v(option, bad, ok)


VALUES = {
    # kidpack-daq
    ('kidpack-daq', '--run-number'): v('--run-number', ['-1', 'abc', 'Test', '1.5', '', '0831_155251'],
                                       ['0', '12', 'test']),  # (the DAQ writes numbers and test only)
    ('kidpack-daq', '--events-per-file'): v('--events-per-file', ['0', '-5', '1.5'], ['1', '1000']),
    ('kidpack-daq', '--num-files'): v('--num-files', ['0', '-1'], ['1', '50']),
    ('kidpack-daq', '--condition'): v('--condition', [], ['temp 5.5K; lna 1.9V', '']),
    ('kidpack-daq', '--output-dir'): v('--output-dir', [], ['data', '/tmp/x']),
    ('kidpack-daq', '--summary-file'): v('--summary-file', [], ['s.txt']),
    ('kidpack-daq', '--resource'): v('--resource', [], ['PXI2Slot2', 'PXI1Slot4']),
    ('kidpack-daq', '--sample-rate'): v('--sample-rate', ['0', '-1e9', 'abc'], ['2.5e9', '1e6']),
    ('kidpack-daq', '--time-window'): v('--time-window', ['0', '-1e-6'], ['2e-6', '1e-6']),
    ('kidpack-daq', '--npts'): v('--npts', ['0', '-1', '2.5'], ['1', '5000']),
    ('kidpack-daq', '--ref-position'): v('--ref-position', ['-1', '100.5'], ['0', '20', '100']),
    ('kidpack-daq', '--fetch-timeout'): v('--fetch-timeout', ['0', '-3'], ['0.5', '100']),
    ('kidpack-daq', '--ch0-range'): v('--ch0-range', ['0', '-0.1'], ['0.01', '5']),
    ('kidpack-daq', '--ch0-offset'): v('--ch0-offset', ['abc'], ['-0.5', '0', '1e-3', '-1e-3']),
    ('kidpack-daq', '--ch1-range'): v('--ch1-range', ['0', '-0.1'], ['0.01', '5']),
    ('kidpack-daq', '--ch1-offset'): v('--ch1-offset', ['abc'], ['-0.5', '0', '1e-3', '-1e-3']),
    ('kidpack-daq', '--impedance'): v('--impedance', ['0', '-50'], ['50', '1e6']),
    ('kidpack-daq', '--bandwidth'): v('--bandwidth', ['abc'], ['-1', '175e6']),
    ('kidpack-daq', '--random-trigger-interval'): v('--random-trigger-interval', ['0', '-1'], ['0.5', '2'],
                                                    companions=['--random-trigger']),
    ('kidpack-daq', '--trigger-source'): v('--trigger-source', [], ['VAL_EXTERNAL', '0', '1']),
    ('kidpack-daq', '--trigger-level'): v('--trigger-level', ['abc'], ['2.2', '-8e-5', '0']),
    ('kidpack-daq', '--sg-frequency'): v('--sg-frequency', ['abc'], ['5.49e9'], companions=['--sg-power', '-2']),
    ('kidpack-daq', '--sg-power'): v('--sg-power', ['abc'], ['-2', '0', '-30.5'], companions=['--sg-frequency', '5e9']),
    ('kidpack-daq', '--sg-resource'): v('--sg-resource', [], ['PXI2Slot3'], companions=['--sg-frequency', '5e9', '--sg-power', '0']),
    # kidpack-monitor
    ('kidpack-monitor', 'file'): {'bad': [], 'ok': [['some.npz']]},
    ('kidpack-monitor', '--run-number'): v('--run-number', ['-1', 'abc', 'Test', '1.5', '0831_15525', '1331_155251',
                                                             '0831-155251', '../0831_155251'],
                                           ['0', '12', 'test', '0831_155251', '1231_235959']),
    ('kidpack-monitor', '--file-number'): v('--file-number', ['-1', 'abc', '1.5'], ['0', '3'],
                                            companions=['--run-number', '1']),
    ('kidpack-monitor', '--data-dir'): v('--data-dir', [], ['data']),
    ('kidpack-monitor', '--output-dir'): v('--output-dir', [], ['out']),
    ('kidpack-monitor', '--rebin'): v('--rebin', ['0', '-1', '2.5'], ['1', '5', '10']),
    ('kidpack-monitor', '--stride'): v('--stride', ['0', '-1'], ['1', '2']),
    # kidpack-iqscan
    ('kidpack-iqscan', '--f-start'): v('--f-start', ['0', '-1e9', 'abc'], ['4.4e9']),
    ('kidpack-iqscan', '--f-stop'): v('--f-stop', ['0', '-1e9'], ['4.4e9', '1e9']),  # below f-start is fine
    ('kidpack-iqscan', '--num-points'): v('--num-points', ['0', '-1', '2.5'], ['1', '101']),
    ('kidpack-iqscan', '--power'): v('--power', ['nan', 'abc'], ['-2', '0', '-60.5']),
    ('kidpack-iqscan', '--settle-time'): v('--settle-time', ['-0.1'], ['0', '0.5']),
    ('kidpack-iqscan', '--condition'): v('--condition', [], ['temp 5.5K']),
    ('kidpack-iqscan', '--output-dir'): v('--output-dir', [], ['out']),
    ('kidpack-iqscan', '--name'): v('--name', ['', 'a/b', '../x', '.', '..'], ['scan_1', 'T5.5K_-30dBm']),
    ('kidpack-iqscan', '--summary-file'): v('--summary-file', [], ['s.txt']),
    ('kidpack-iqscan', '--resource'): v('--resource', [], ['PXI2Slot2']),
    ('kidpack-iqscan', '--sample-rate'): v('--sample-rate', ['0', '-1'], ['1e4', '2e4']),
    ('kidpack-iqscan', '--time-window'): v('--time-window', ['0', '-1'], ['1', '0.5']),
    ('kidpack-iqscan', '--npts'): v('--npts', ['0', '-1'], ['1', '10000']),
    ('kidpack-iqscan', '--num-records'): v('--num-records', ['0', '-1', '1.5'], ['1', '4']),
    ('kidpack-iqscan', '--ref-position'): v('--ref-position', ['-1', '101'], ['0', '50', '100']),
    ('kidpack-iqscan', '--ch0-range'): v('--ch0-range', ['0', '-1'], ['0.1', '1']),
    ('kidpack-iqscan', '--ch0-offset'): v('--ch0-offset', ['abc'], ['-0.1', '0', '1e-3']),
    ('kidpack-iqscan', '--ch1-range'): v('--ch1-range', ['0', '-1'], ['0.1', '1']),
    ('kidpack-iqscan', '--ch1-offset'): v('--ch1-offset', ['abc'], ['-0.1', '0', '1e-3']),
    ('kidpack-iqscan', '--impedance'): v('--impedance', ['0', '-50'], ['50', '1e6']),
    ('kidpack-iqscan', '--fetch-timeout'): v('--fetch-timeout', ['0', '-1'], ['1', '17']),
    ('kidpack-iqscan', '--sg-resource'): v('--sg-resource', [], ['PXI2Slot3']),
    # kidpack-iqplot
    ('kidpack-iqplot', 'file'): {'bad': [], 'ok': [['some.npz']]},
    ('kidpack-iqplot', '--data-dir'): v('--data-dir', [], ['data']),
    ('kidpack-iqplot', '--ref'): v('--ref', [], ['ref.npz']),
    ('kidpack-iqplot', '--save'): v('--save', [], ['q.png']),
}


def test_every_note_has_test_cases_and_vice_versa():
    assert set(VALUES) == set(ALLOWED)


@pytest.mark.parametrize('key', sorted(VALUES), ids=lambda k: f'{k[0]} {k[1]}')
def test_the_documented_values_are_accepted_and_the_others_rejected(accepted, key):
    command = key[0]
    for argv in VALUES[key]['ok']:
        assert accepted(command, argv), f'{command} should accept {argv}'
    for argv in VALUES[key]['bad']:
        assert not accepted(command, argv), f'{command} should reject {argv}'


@pytest.mark.parametrize('command', [name for name, _, _ in COMMANDS])
def test_every_choice_is_accepted_and_anything_else_is_rejected(accepted, command):
    builder = dict((n, b) for n, _, b in COMMANDS)[command]
    for option, action in command_options(command, builder).items():
        for choice in action.choices or ():
            assert accepted(command, [option, str(choice)]), (command, option, choice)
        if action.choices:
            assert not accepted(command, [option, 'not-a-choice']), (command, option)


RULE_CASES = [
    # (command, arguments that must be rejected, arguments that must be accepted)
    ('kidpack-daq', [], [[], ['--run-number', 'test'], ['--run-number', '4']]),  # the run number is optional
    ('kidpack-monitor', [['--file-number', '1'], ['some.npz', '--run-number', '3'], ['some.npz', '--file-number', '1']],
     [['--run-number', '3'], ['--run-number', '3', '--file-number', '1'], ['--run-number', 'test'], ['some.npz']]),
    ('kidpack-daq', [['--time-window', '1e-6', '--npts', '100']], [['--time-window', '1e-6'], ['--npts', '100']]),
    ('kidpack-daq', [['--sg-frequency', '5e9'], ['--sg-power', '0'], ['--sg-resource', 'X'],
                     ['--sg-resource', 'X', '--sg-power', '0']],
     [['--sg-frequency', '5e9', '--sg-power', '0'], ['--sg-frequency', '5e9', '--sg-power', '0', '--sg-resource', 'X']]),
    ('kidpack-daq', [['--random-trigger-interval', '1']], [['--random-trigger', '--random-trigger-interval', '1']]),
    ('kidpack-daq', [['--random-trigger', '--trigger-level', '1'], ['--random-trigger', '--trigger-source', '1'],
                     ['--random-trigger', '--trigger-slope', 'negative'],
                     ['--random-trigger', '--trigger-coupling', 'ac']],
     [['--random-trigger'], ['--random-trigger', '--trigger-coupling', 'dc'],
      ['--trigger-level', '1', '--trigger-coupling', 'ac']]),
    ('kidpack-iqscan', [['--time-window', '1', '--npts', '100']], [['--time-window', '1'], ['--npts', '100']]),
    ('kidpack-iqplot', [['--no-show']], [['--no-show', '--save', 'q.png'], ['--save', 'q.png']]),
]


@pytest.mark.parametrize('command, bad, ok', RULE_CASES, ids=[f'{c[0]}-{i}' for i, c in enumerate(RULE_CASES)])
def test_the_rules_between_options_hold(accepted, command, bad, ok):
    for argv in ok:
        assert accepted(command, argv), f'{command} should accept {argv}'
    for argv in bad:
        assert not accepted(command, argv), f'{command} should reject {argv}'


def test_every_command_has_its_rules_listed():
    assert set(RULES) == {name for name, _, _ in COMMANDS}
    assert all(RULES[name] for name in RULES)
