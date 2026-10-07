"""Generate docs/OPTIONS.md: every option of every kidpack command, its allowed values and default.

    python -m kidpack.optionsdoc                        # print the Markdown
    python -m kidpack.optionsdoc --output docs/OPTIONS.md

The tables are built from the real argument parsers, so choices, defaults and
descriptions cannot go out of date. Only the "allowed values" of numeric and free-text
options, and the rules between options, are written by hand below; the tests check
every one of them against the real parsers (tests/test_optionsdoc.py), and fail if
docs/OPTIONS.md is not what this module generates.
"""
import argparse
import re
import sys
from pathlib import Path

from kidpack.daq.config import build_parser as daq_parser
from kidpack.daq.config import default_output_dir
from kidpack.iqscan.config import build_parser as iqscan_parser
from kidpack.iqscan.plot import build_parser as iqplot_parser
from kidpack.monitor.cli import build_parser as monitor_parser

COMMANDS = [
    ('kidpack-daq', 'Pulse DAQ: acquires waveforms with the NI-SCOPE digitizer', daq_parser),
    ('kidpack-monitor', 'Online check of a raw file: pc1 / pc2 of 16 events', monitor_parser),
    ('kidpack-iqscan', 'IQ scan: transmission versus frequency', iqscan_parser),
    ('kidpack-iqplot', 'Quick look at an IQ scan', iqplot_parser),
]

# Allowed values of the options that are neither flags nor have a fixed list of choices.
_COUPLING_NOTE = 'see choices'
ALLOWED = {
    # ---- kidpack-daq ----
    ('kidpack-daq', '--run-number'): 'integer, 0 or more; or the word `test` (see Rules)',
    ('kidpack-daq', '--events-per-file'): 'integer, 1 or more',
    ('kidpack-daq', '--num-files'): 'integer, 1 or more',
    ('kidpack-daq', '--condition'): 'text (tabs and line breaks become spaces)',
    ('kidpack-daq', '--output-dir'): 'directory path',
    ('kidpack-daq', '--summary-file'): 'file path',
    ('kidpack-daq', '--resource'): 'NI-SCOPE resource name, e.g. `PXI2Slot2`',
    ('kidpack-daq', '--sample-rate'): 'number [Hz], more than 0 (the digitizer takes the nearest rate it has; max 2.5e9)',
    ('kidpack-daq', '--time-window'): 'number [s], more than 0 (not together with `--npts`)',
    ('kidpack-daq', '--npts'): 'integer, 1 or more (not together with `--time-window`)',
    ('kidpack-daq', '--ref-position'): 'number [%], 0 to 100',
    ('kidpack-daq', '--fetch-timeout'): 'number [s], more than 0',
    ('kidpack-daq', '--ch0-range'): 'number [V], more than 0 (the digitizer takes the nearest range it has)',
    ('kidpack-daq', '--ch0-offset'): 'number [V], any',
    ('kidpack-daq', '--ch1-range'): 'number [V], more than 0 (the digitizer takes the nearest range it has)',
    ('kidpack-daq', '--ch1-offset'): 'number [V], any',
    ('kidpack-daq', '--impedance'): 'number [ohm], more than 0 (normally 50 or 1e6)',
    ('kidpack-daq', '--bandwidth'): 'number [Hz]; -1 = full bandwidth',
    ('kidpack-daq', '--random-trigger-interval'): 'number [s], more than 0 (only with `--random-trigger`)',
    ('kidpack-daq', '--trigger-source'): 'text: `VAL_EXTERNAL` (external trigger input) or a channel name such as `0`, `1`',
    ('kidpack-daq', '--trigger-level'): 'number [V], any',
    ('kidpack-daq', '--sg-frequency'): 'number [Hz] (together with `--sg-power`)',
    ('kidpack-daq', '--sg-power'): 'number [dBm] (together with `--sg-frequency`)',
    ('kidpack-daq', '--sg-resource'): 'resource name (only with `--sg-frequency` and `--sg-power`)',
    # ---- kidpack-monitor ----
    ('kidpack-monitor', 'file'): 'path of a raw `.npz` file',
    ('kidpack-monitor', '--data-dir'): 'directory path',
    ('kidpack-monitor', '--output-dir'): 'directory path',
    ('kidpack-monitor', '--rebin'): 'integer, 1 or more',
    ('kidpack-monitor', '--stride'): 'integer, 1 or more',
    # ---- kidpack-iqscan ----
    ('kidpack-iqscan', '--f-start'): 'number [Hz], more than 0',
    ('kidpack-iqscan', '--f-stop'): 'number [Hz], more than 0 (may be below `--f-start`)',
    ('kidpack-iqscan', '--num-points'): 'integer, 1 or more',
    ('kidpack-iqscan', '--power'): 'number [dBm], any finite value',
    ('kidpack-iqscan', '--settle-time'): 'number [s], 0 or more',
    ('kidpack-iqscan', '--condition'): 'text (tabs and line breaks become spaces)',
    ('kidpack-iqscan', '--output-dir'): 'directory path',
    ('kidpack-iqscan', '--name'): 'file name without directory and extension (not `.` or `..`)',
    ('kidpack-iqscan', '--summary-file'): 'file path',
    ('kidpack-iqscan', '--resource'): 'NI-SCOPE resource name, e.g. `PXI2Slot2`',
    ('kidpack-iqscan', '--sample-rate'): 'number [Hz], more than 0',
    ('kidpack-iqscan', '--time-window'): 'number [s], more than 0 (not together with `--npts`)',
    ('kidpack-iqscan', '--npts'): 'integer, 1 or more (not together with `--time-window`)',
    ('kidpack-iqscan', '--num-records'): 'integer, 1 or more',
    ('kidpack-iqscan', '--ref-position'): 'number [%], 0 to 100',
    ('kidpack-iqscan', '--ch0-range'): 'number [V], more than 0',
    ('kidpack-iqscan', '--ch0-offset'): 'number [V], any',
    ('kidpack-iqscan', '--ch1-range'): 'number [V], more than 0',
    ('kidpack-iqscan', '--ch1-offset'): 'number [V], any',
    ('kidpack-iqscan', '--impedance'): 'number [ohm], more than 0 (normally 50 or 1e6)',
    ('kidpack-iqscan', '--fetch-timeout'): 'number [s], more than 0',
    ('kidpack-iqscan', '--sg-resource'): 'signal generator resource name, e.g. `PXI2Slot3`',
    # ---- kidpack-iqplot ----
    ('kidpack-iqplot', 'file'): 'path of an IQ-scan `.npz` file',
    ('kidpack-iqplot', '--data-dir'): 'directory path',
    ('kidpack-iqplot', '--ref'): 'path of an IQ-scan `.npz` file with the same frequencies',
    ('kidpack-iqplot', '--save'): 'path of the PNG file to write',
}

# Rules between options (checked by the tests).
RULES = {
    'kidpack-daq': [
        '`--run-number` is optional: without it the next free number is used (the highest of the '
        '`run_NN` directories and of the run summary, plus 1; 1 if there is none).',
        'A run number that already has a directory is refused. The exception is `--run-number test`: '
        'it writes to `run_test/` and replaces the previous test run (its files are deleted once the '
        'instruments are ready), unless that directory holds files that are not DAQ output (then '
        'nothing is deleted). Test runs never count as numbers; the summary gets one line per test run.',
        '`--time-window` and `--npts` cannot be given together.',
        '`--sg-frequency` and `--sg-power` must be given together; `--sg-resource` needs both.',
        '`--random-trigger-interval` needs `--random-trigger`.',
        '`--random-trigger` cannot be combined with `--trigger-source`, `--trigger-level`, '
        '`--trigger-slope` or `--trigger-coupling` set to something else than their default.',
    ],
    'kidpack-monitor': [
        'Without `file`, the newest `run*-*.npz` under `--data-dir` is used.',
    ],
    'kidpack-iqscan': [
        '`--time-window` and `--npts` cannot be given together.',
        'An existing scan file (same `--name`, or the same start minute) is never overwritten.',
    ],
    'kidpack-iqplot': [
        '`--no-show` needs `--save` (otherwise there would be nothing to do).',
        '`--ref` must have been measured at the same frequencies as `file`.',
        'Without `file`, the newest `*.npz` directly under `--data-dir` is used.',
    ],
}

_DEFAULT_GROUPS = ('positional arguments', 'options', 'optional arguments')  # titles differ by Python version


def _root():
    """The KIDPACK checkout (shown as <KIDPACK> so that the document does not depend on the machine)."""
    path = Path(default_output_dir())
    return str(path.parent) if path.is_absolute() else None


def _clean(text):
    root = _root()
    return text.replace(root, '<KIDPACK>') if root else text


def _number(value):
    text = f'{value:g}'
    for old, new in (('e+0', 'e'), ('e+', 'e'), ('e-0', 'e-')):
        text = text.replace(old, new)
    return text


def _default_text(action):
    if action.required:
        return 'required'
    if isinstance(action, argparse._StoreTrueAction):
        return 'off'
    if action.default is None:
        return '-'
    if isinstance(action.default, bool):
        return str(action.default)
    if isinstance(action.default, (int, float)):
        return f'`{_number(action.default)}`'
    return f'`{_clean(str(action.default))}`' if str(action.default) else '(empty)'


def _help_text(action):
    text = action.help or ''
    try:
        text = text % {'default': action.default}
    except (TypeError, ValueError, KeyError):
        pass
    text = ' '.join(text.split())
    if action.default is not None:  # the Default column already says it
        text = re.sub(r'\s*\(default: [^)]*\)', '', text)
    return _clean(text).replace('|', '\\|')


def _value_text(command, action):
    option = action.option_strings[0] if action.option_strings else action.dest
    if isinstance(action, argparse._StoreTrueAction):
        return '(flag: no value)'
    if action.choices:
        return ', '.join(f'`{choice}`' for choice in action.choices)
    return ALLOWED[(command, option)]


def _name_text(action):
    if action.option_strings:
        metavar = f' `{action.metavar}`'.replace('|', '\\|') if action.metavar else ''  # | would split the table cell
        return f"`{action.option_strings[0]}`" + metavar
    return f'`{action.metavar or action.dest}` (positional)'


def options_of(parser):
    """[(group title or None, [actions])] in the order they were defined, without -h/--help.

    The positional arguments and the plain options (no group title of our own) share one table.
    """
    groups, plain = [], []
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if not isinstance(a, argparse._HelpAction)]
        if not actions:
            continue
        if group.title in _DEFAULT_GROUPS:
            plain += actions
        else:
            groups.append((group.title, actions))
    return ([(None, plain)] if plain else []) + groups


def command_options(command, builder):
    """{option string or positional name: action} of one command (used by the tests)."""
    return {(a.option_strings[0] if a.option_strings else a.dest): a
            for _, actions in options_of(builder()) for a in actions}


def render():
    lines = [
        '# Command-line options',
        '',
        '*Generated by `python -m kidpack.optionsdoc --output docs/OPTIONS.md` from the real '
        'argument parsers; do not edit by hand (a test fails if this file is out of date). '
        '`<KIDPACK>` is the checkout of this repository.*',
        '',
        'The same information is printed by `<command> --help`. A value written like `5.4e9` means '
        '5.4 x 10^9; numbers can be given in that form (`--sample-rate 2.5e9`). Negative values '
        'can be given as usual (`--power -2`, `--trigger-level -8e-5`).',
        '',
    ]
    lines += ['| Command | Purpose |', '|---|---|']
    lines += [f'| [`{name}`](#{name}) | {purpose} |' for name, purpose, _ in COMMANDS]
    lines.append('')

    for name, purpose, builder in COMMANDS:
        parser = builder()
        lines += [f'## {name}', '', _clean(' '.join((parser.description or purpose).split())), '']
        for title, actions in options_of(parser):
            if title:
                lines += [f'### {title}', '']
            lines += ['| Option | Allowed values | Default | Meaning |', '|---|---|---|---|']
            for action in actions:
                lines.append(f'| {_name_text(action)} | {_value_text(name, action)} | '
                             f'{_default_text(action)} | {_help_text(action)} |')
            lines.append('')
        lines += ['### Rules', ''] + [f'- {rule}' for rule in RULES[name]] + ['']
    return '\n'.join(lines).rstrip('\n') + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m kidpack.optionsdoc', description=__doc__.splitlines()[0])
    parser.add_argument('--output', metavar='PATH', help='write the Markdown here instead of printing it')
    args = parser.parse_args(argv)
    text = render()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        print(f'wrote {path}')
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
