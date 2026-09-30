"""Scan summary text file: one tab-separated line appended per scan."""
import os

from kidpack.daq.summary import one_line
from kidpack.daq.timeutil import iso_local

HEADER = ('name', 'start_time', 'stop_time', 'f_start_ghz', 'f_stop_ghz', 'num_points',
          'power_dbm', 'status', 'condition')


def append_scan_summary(path, name, start_ns, stop_ns, f_start, f_stop, num_points, power,
                        status, condition):
    """Append one scan to the summary file, writing the header if it is new.

    Times are local time with UTC offset; ``num_points`` is the number of
    frequencies actually measured (fewer than requested if the scan was stopped).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    row = (name, iso_local(start_ns), iso_local(stop_ns), f'{f_start / 1e9:.6f}',
           f'{f_stop / 1e9:.6f}', num_points, f'{power:g}', status, one_line(condition))
    with open(path, 'a', encoding='utf-8', newline='') as f:
        if is_new:
            f.write('\t'.join(HEADER) + '\n')
        f.write('\t'.join(str(v) for v in row) + '\n')
