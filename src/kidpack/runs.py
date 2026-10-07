"""Run numbers, run directories and file names (shared by the DAQ and the monitor).

    run_07/data/run07-03.npz         run number 7, file number 3
    run_test/data/runtest-00.npz     the special run "test", replaced by every new test run
"""
import argparse
import glob
import os
import re

RUN_TEST = 'test'

_RUN_DIR = re.compile(r'run_(\d+)')
_DIGITS = re.compile(r'\d+', re.ASCII)


def is_valid_run_number(value):
    """A run number is an integer (0 or more) or the word "test"."""
    return value == RUN_TEST or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def run_number_arg(text):
    """argparse type of ``--run-number``: an integer, 0 or more, or "test"."""
    if text == RUN_TEST:
        return RUN_TEST
    if _DIGITS.fullmatch(text):
        return int(text)
    raise argparse.ArgumentTypeError(f"invalid run number {text!r}: an integer (0 or more) or 'test'")


def file_number_arg(text):
    """argparse type of ``--file-number``: an integer, 0 or more."""
    if _DIGITS.fullmatch(text):
        return int(text)
    raise argparse.ArgumentTypeError(f'invalid file number {text!r}: an integer (0 or more)')


def run_label(run_number):
    """07 for 7, test for "test": the XX of run_XX and runXX-YY."""
    return run_number if run_number == RUN_TEST else f'{run_number:02d}'


def run_dir_name(run_number):
    return f'run_{run_label(run_number)}'


def file_stem(run_number, file_number):
    return f'run{run_label(run_number)}-{file_number:02d}'


def next_run_number(output_dir, summary_path=None):
    """The number after the highest one in use: 1 if there is none.

    In use are the run_NN directories of ``output_dir`` and the numbers in the first
    column of the run summary (so that the number of a run whose directory was deleted
    is not used again). "test" runs are not numbers and are ignored.
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
    """Path of a raw file of a run; the highest file number if ``file_number`` is None.

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
