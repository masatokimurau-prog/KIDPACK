import argparse

import pytest

from kidpack.runs import (RUN_TEST, file_number_arg, file_stem, find_run_file, is_valid_run_number,
                          next_run_number, run_dir_name, run_label, run_number_arg)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x')
    return path


# --- run numbers and names ------------------------------------------------------------------

@pytest.mark.parametrize('value, valid', [(0, True), (7, True), (123, True), ('test', True),
                                          (-1, False), ('7', False), ('Test', False), (1.5, False),
                                          (None, False), (True, False)])
def test_what_is_a_run_number(value, valid):
    assert is_valid_run_number(value) is valid


def test_run_number_argument_is_an_integer_or_the_word_test():
    assert run_number_arg('0') == 0 and run_number_arg('12') == 12 and run_number_arg('007') == 7
    assert run_number_arg('test') == RUN_TEST == 'test'
    for bad in ('-1', '1.5', 'abc', 'Test', 'TEST', '', ' 3', '1e3', '²'):
        with pytest.raises(argparse.ArgumentTypeError):
            run_number_arg(bad)


def test_file_number_argument_is_a_non_negative_integer():
    assert file_number_arg('0') == 0 and file_number_arg('15') == 15
    for bad in ('-1', '1.5', 'x', '', 'test'):
        with pytest.raises(argparse.ArgumentTypeError):
            file_number_arg(bad)


def test_names_of_run_directories_and_files():
    assert (run_label(7), run_dir_name(7), file_stem(7, 3)) == ('07', 'run_07', 'run07-03')
    assert (run_label(123), run_dir_name(123), file_stem(123, 12)) == ('123', 'run_123', 'run123-12')
    assert (run_label('test'), run_dir_name('test'), file_stem('test', 0)) == ('test', 'run_test', 'runtest-00')


# --- the next free run number ------------------------------------------------------------------

def test_the_first_run_is_number_1(tmp_path):
    assert next_run_number(tmp_path) == 1
    assert next_run_number(tmp_path / 'does' / 'not' / 'exist') == 1


def test_the_next_number_follows_the_highest_run_directory(tmp_path):
    for n in (1, 2, 7):
        (tmp_path / f'run_{n:02d}').mkdir()
    assert next_run_number(tmp_path) == 8  # not "the first gap"
    (tmp_path / 'run_100').mkdir()
    assert next_run_number(tmp_path) == 101


def test_only_numbered_run_directories_count(tmp_path):
    (tmp_path / 'run_03').mkdir()
    (tmp_path / 'run_test').mkdir()  # a test run is not a number
    (tmp_path / 'run_abc').mkdir()
    (tmp_path / 'run_9x').mkdir()
    (tmp_path / 'myrun_50').mkdir()
    (tmp_path / 'run_40').write_text('a file, not a directory')
    (tmp_path / 'iqscan').mkdir()
    assert next_run_number(tmp_path) == 4


def test_numbers_in_the_summary_count_too_even_if_the_directory_is_gone(tmp_path):
    summary = tmp_path / 'run_summary.txt'
    summary.write_text('run_number\tstart_time\tstop_time\tdaq_rate_hz\tcondition\n'
                       '1\ta\tb\t1.0\t\n'
                       '12\ta\tb\t1.0\twas deleted\n'
                       'test\ta\tb\t1.0\t\n'
                       '3\ta\tb\t1.0\t\n')
    (tmp_path / 'run_05').mkdir()
    assert next_run_number(tmp_path, summary) == 13
    assert next_run_number(tmp_path, tmp_path / 'no_such_summary.txt') == 6  # a missing summary is fine
    assert next_run_number(tmp_path) == 6  # and so is none


def test_a_summary_without_numbers_does_not_matter(tmp_path):
    summary = tmp_path / 'run_summary.txt'
    summary.write_text('run_number\tstart_time\n')
    assert next_run_number(tmp_path, summary) == 1
    summary.write_text('')
    assert next_run_number(tmp_path, summary) == 1


# --- finding a file of a run -------------------------------------------------------------------

def make_run(root, label, file_numbers):
    for n in file_numbers:
        touch(root / f'run_{label}' / 'data' / f'run{label}-{n:02d}.npz')


def test_a_file_of_a_run_is_found_by_run_and_file_number(tmp_path):
    make_run(tmp_path, '12', [0, 1, 2, 3])
    assert find_run_file(tmp_path, 12, 3) == str(tmp_path / 'run_12' / 'data' / 'run12-03.npz')
    assert find_run_file(tmp_path, 12, 0).endswith('run12-00.npz')


def test_without_a_file_number_the_highest_one_is_taken(tmp_path):
    make_run(tmp_path, '12', [0, 2, 10, 11])  # 10 > 2 although '2' > '10' as text
    assert find_run_file(tmp_path, 12).endswith('run12-11.npz')


def test_the_test_run_is_found_like_any_other(tmp_path):
    make_run(tmp_path, 'test', [0, 1])
    assert find_run_file(tmp_path, 'test', 1).endswith('runtest-01.npz')
    assert find_run_file(tmp_path, 'test').endswith('runtest-01.npz')


def test_runs_do_not_get_mixed_up(tmp_path):
    make_run(tmp_path, '01', [0])
    make_run(tmp_path, '12', [0, 1])
    make_run(tmp_path, '123', [0, 1, 2])
    assert find_run_file(tmp_path, 12).endswith('run12-01.npz')
    assert find_run_file(tmp_path, 123).endswith('run123-02.npz')
    assert find_run_file(tmp_path, 1).endswith('run01-00.npz')


def test_half_written_and_foreign_files_are_ignored(tmp_path):
    make_run(tmp_path, '12', [0, 1])
    touch(tmp_path / 'run_12' / 'data' / 'run12-02.npz.tmp')  # being written
    touch(tmp_path / 'run_12' / 'data' / 'run12-x.npz')
    touch(tmp_path / 'run_12' / 'data' / 'notes.txt')
    assert find_run_file(tmp_path, 12).endswith('run12-01.npz')


def test_missing_things_are_reported_in_words(tmp_path):
    make_run(tmp_path, '12', [0, 1, 4])
    with pytest.raises(FileNotFoundError, match='no data of run 99'):
        find_run_file(tmp_path, 99)
    with pytest.raises(FileNotFoundError, match=r'file number 2 of run 12 not found.*0, 1, 4'):
        find_run_file(tmp_path, 12, 2)
    (tmp_path / 'run_13' / 'data').mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match='no raw file'):
        find_run_file(tmp_path, 13)
