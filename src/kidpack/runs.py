"""Run numbers, run directories and file names (shared by the DAQ and the monitor).

    run_07/data/run07-03.npz         run number 7, file number 3
    run_test/data/runtest-00.npz     the special run "test", replaced by every new test run
    run_0831_155251/data/run0831_155251-00.npz
                                     a run named after its start, MMDD_HHMMSS: files of the old
                                     DAQ macro, converted by kidpack.legacy (the DAQ only writes
                                     numbers and "test")
"""
import argparse
import glob
import os
import re

RUN_TEST = 'test'

_RUN_DIR = re.compile(r'run_(\d+)')
_DIGITS = re.compile(r'\d+', re.ASCII)
# MMDD_HHMMSS: month 01-12, day 01-31, hour 00-23, minute and second 00-59
_DATED = re.compile(r'(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])_([01][0-9]|2[0-3])[0-5][0-9][0-5][0-9]', re.ASCII)


def is_valid_run_number(value):
    """A run number is an integer (0 or more) or the word "test"."""
    return value == RUN_TEST or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def is_dated_label(value):
    """A run named after its start (month, day, hour, minute, second): "0831_155251"."""
    return isinstance(value, str) and _DATED.fullmatch(value) is not None


def run_number_arg(text):
    """argparse type of ``--run-number``: an integer, 0 or more, or "test"."""
    if text == RUN_TEST:
        return RUN_TEST
    if _DIGITS.fullmatch(text):
        return int(text)
    raise argparse.ArgumentTypeError(f"invalid run number {text!r}: an integer (0 or more) or 'test'")


def run_label_arg(text):
    """argparse type of ``--run-number`` where a run is only read: also a name like 0831_155251."""
    if is_dated_label(text):
        return text
    try:
        return run_number_arg(text)
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(
            f"invalid run {text!r}: an integer (0 or more), 'test' or a start time MMDD_HHMMSS") from None


def file_number_arg(text):
    """argparse type of ``--file-number``: an integer, 0 or more."""
    if _DIGITS.fullmatch(text):
        return int(text)
    raise argparse.ArgumentTypeError(f'invalid file number {text!r}: an integer (0 or more)')


def run_label(run_number):
    """07 for 7, test for "test", 0831_155251 for that: the XX of run_XX and runXX-YY."""
    if isinstance(run_number, str):
        if run_number == RUN_TEST or is_dated_label(run_number):
            return run_number
        raise ValueError(f"invalid run {run_number!r}: an integer, 'test' or a start time MMDD_HHMMSS")
    return f'{run_number:02d}'


def run_dir_name(run_number):
    return f'run_{run_label(run_number)}'


def file_stem(run_number, file_number):
    return f'run{run_label(run_number)}-{file_number:02d}'


def next_run_number(output_dir, summary_path=None):
    """The number after the highest one in use: 1 if there is none.

    In use are the run_NN directories of ``output_dir`` and the numbers in the first
    column of the run summary (so that the number of a run whose directory was deleted
    is not used again). "test" runs and runs named after their start are not numbers and
    are ignored.
    """
    numbers = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            match = _RUN_DIR.fullmatch(name)
            if match and os.path.isdir(os.path.join(output_dir, name)):
                numbers.append(int(match.group(1)))
    if summary_path and os.path.isfile(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            next(f, None)  # the header
            for line in f:
                first = line.split('\t', 1)[0].strip()
                if _DIGITS.fullmatch(first):
                    numbers.append(int(first))
    return max(numbers, default=0) + 1


def find_run_file(data_dir, run_number, file_number=None):
    """Path of a raw file of a run (a number, "test" or MMDD_HHMMSS); the highest file number
    if ``file_number`` is None.

    Raises FileNotFoundError with a message saying what is missing and what exists.
    """
    data = os.path.join(data_dir, run_dir_name(run_number), 'data')
    if not os.path.isdir(data):
        raise FileNotFoundError(f'no data of run {run_label(run_number)}: {data} does not exist')

    prefix = f'run{run_label(run_number)}-'
    available = {}
    for path in glob.glob(os.path.join(data, f'{prefix}*.npz')):
        number = os.path.basename(path)[len(prefix):-len('.npz')]
        if _DIGITS.fullmatch(number):
            available[int(number)] = path
    if not available:
        raise FileNotFoundError(f'no raw file in {data}')

    if file_number is None:
        return available[max(available)]
    if file_number not in available:
        numbers = ', '.join(str(n) for n in sorted(available))
        raise FileNotFoundError(f'file number {file_number} of run {run_label(run_number)} not found '
                                f'in {data} (file numbers there: {numbers})')
    return available[file_number]
