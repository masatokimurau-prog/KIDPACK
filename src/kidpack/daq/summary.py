"""Run summary text file: one tab-separated line appended per DAQ run."""
import os

from kidpack.daq.timeutil import iso_local

HEADER = ('run_number', 'start_time', 'stop_time', 'daq_rate_hz', 'condition')


def _one_line(text):
    return str(text).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')


def append_run_summary(path, run_number, start_ns, stop_ns, daq_rate_hz, condition):
    """Append one run to the summary file, writing the header if it is new.

    Times are local time with UTC offset; ``daq_rate_hz`` is empty when the
    run recorded no events.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    rate = '' if daq_rate_hz is None else f'{daq_rate_hz:.2f}'
    row = (run_number, iso_local(start_ns), iso_local(stop_ns), rate, _one_line(condition))
    with open(path, 'a', encoding='utf-8', newline='') as f:
        if is_new:
            f.write('\t'.join(HEADER) + '\n')
        f.write('\t'.join(str(v) for v in row) + '\n')
