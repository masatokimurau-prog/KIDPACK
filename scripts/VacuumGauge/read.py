"""Log pressure from a Pfeiffer TPG 261 vacuum gauge to a CSV file.

The gauge is expected to be in continuous output mode; every line it sends
has the form ``status,value`` (e.g. ``0,1.2345E-03``). The pressure unit is
whatever is configured on the gauge.

Usage:
    python read.py OUT.csv [PORT] [--interval SECONDS]
    python read.py --list-ports

``--list-ports`` shows the serial ports this Python can see (to find PORT).

One row is written per interval, aligned to the wall clock (for the default
of 60 s: the first reading after every minute boundary).
"""
import argparse
import csv
import errno
import os
import platform
import re
import sys
import time
from datetime import datetime

import serial
from serial.tools import list_ports

BAUDRATE = 9600
READ_TIMEOUT_S = 3
NO_DATA_WARN_S = 120
RECONNECT_WAIT_S = 5
MAX_UNPARSED_WARNINGS = 3

STATUS_MESSAGES = {
    0: 'OK',
    1: 'Under Range',
    2: 'Over Range',
    3: 'Sensor Error',
    4: 'Sensor Off',
    5: 'No Sensor',
    6: 'Identification Error',
}
CSV_HEADER = ['timestamp', 'status', 'pressure']


def parse_tpg261_data(raw_string):
    """Parse one ``status,value`` line into ``(status, pressure)``.

    ``pressure`` is a float when status is 0 (measurement OK) and None
    otherwise. Returns None if the line is not in the expected format.
    """
    parts = [p.strip() for p in raw_string.split(',')]
    if len(parts) != 2:
        return None
    try:
        status = int(parts[0])
        pressure = float(parts[1]) if status == 0 else None
    except ValueError:
        return None
    return status, pressure


def status_text(status):
    return STATUS_MESSAGES.get(status, f'Unknown Status {status}')


def ensure_header(csv_name):
    """Write the header row if the CSV does not exist yet (or is empty)."""
    if not os.path.exists(csv_name) or os.path.getsize(csv_name) == 0:
        with open(csv_name, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f'Created {csv_name}')


def append_row(csv_name, timestamp, status, pressure):
    pressure_str = '' if pressure is None else f'{pressure:.4E}'
    with open(csv_name, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([timestamp, status, pressure_str])


def open_port(port):
    ser = serial.Serial(port=port,
                        baudrate=BAUDRATE,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=READ_TIMEOUT_S)
    ser.dtr = True
    ser.rts = True
    time.sleep(0.1)
    ser.reset_input_buffer()
    return ser


def visible_ports():
    """[(device, description)] of the serial ports this Python can see."""
    try:
        return sorted((p.device, p.description) for p in list_ports.comports())
    except (OSError, ValueError):  # a platform whose ports cannot be listed
        return []


def wsl_version(release):
    """1 or 2 if ``release`` (the kernel release, ``uname -r``) is that of Windows Subsystem for Linux."""
    if 'WSL2' in release or 'microsoft-standard' in release:  # 5.15.153.1-microsoft-standard-WSL2
        return 2
    if 'Microsoft' in release:  # 4.4.0-19041-Microsoft
        return 1
    return None


def explain_open_failure(port, error, system=None, release=None):
    """Lines saying what is probably wrong when ``port`` cannot be opened, and what to try.

    ``system``: sys.platform of the Python that tried (``win32`` = a native Windows Python);
    ``release``: its kernel release (to tell WSL1 from WSL2).
    """
    system = sys.platform if system is None else system
    release = platform.release() if release is None else release
    text = str(error)
    not_found = (getattr(error, 'errno', None) == errno.ENOENT or 'FileNotFoundError' in text
                 or 'No such file' in text)
    denied = (getattr(error, 'errno', None) in (errno.EACCES, errno.EBUSY) or 'PermissionError' in text
              or 'Permission denied' in text or 'Access is denied' in text)
    windows_name = re.fullmatch(r'COM(\d+)', port, re.IGNORECASE)
    lines = []
    if windows_name and system != 'win32':
        device = f'/dev/ttyS{int(windows_name.group(1)) - 1}' if int(windows_name.group(1)) >= 1 else None
        windows_python = (f'Run the script with the Windows Python (in PowerShell: py read.py OUT.csv {port}; '
                          'pyserial for it: py -m pip install pyserial)')
        wsl = wsl_version(release) if system == 'linux' else None
        if wsl == 2:
            lines += [
                f'{port} is a Windows port name, and this Ubuntu is WSL2 (kernel {release}): '
                'WSL2 cannot see the Windows COM ports.',
                windows_python + ',',
                'or attach the USB-serial adapter to WSL2 with usbipd-win (it then shows up as '
                '/dev/ttyUSB0 and needs a driver in the WSL2 kernel; Windows cannot use it meanwhile).']
        elif wsl == 1 and device:
            lines += [
                f'{port} is a Windows port name; in WSL1 (kernel {release}) the port is the device file {device}:',
                f'  python3 read.py OUT.csv {device}',
                f'("Permission denied" there: sudo chmod 666 {device}, or join the dialout group.)']
        else:
            lines += [
                f'{port} is a Windows port name, but this Python is not a native Windows Python '
                f'(sys.platform = {system!r}: WSL, Cygwin or MSYS2?), so it looks for a file called {port}.',
                windows_python + ',',
                'or give the device file: in Cygwin, MSYS2 and WSL1 COM<n> is /dev/ttyS<n-1> '
                '(COM3 is /dev/ttyS2). WSL2 cannot see the Windows COM ports at all (without usbipd-win).']
    elif not_found:
        lines.append(f'{port} does not exist for this Python: use one of the ports below '
                     '(the names of the Device Manager on Windows).')
    elif denied:
        lines.append(f'{port} exists but cannot be opened: another program (a second read.py, the '
                     "gauge's own software, a terminal) may have it open; on Linux you may need to be "
                     'in the dialout group.')
    ports = visible_ports()
    if ports:
        lines.append('Serial ports this Python can see:')
        lines += [f'  {device}: {description}' for device, description in ports]
    else:
        lines.append('This Python sees no serial port (python read.py --list-ports shows what it sees).')
    return lines


def reconnect(port):
    """Retry opening the port until it succeeds."""
    while True:
        time.sleep(RECONNECT_WAIT_S)
        try:
            ser = open_port(port)
        except serial.SerialException as e:
            print(f'Reconnect to {port} failed: {e}')
        else:
            print(f'Reconnected to {port}.')
            return ser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('csv_name', nargs='?', help='output CSV file (appended to)')
    parser.add_argument('port', nargs='?', default='COM3',
                        help='serial port (default: COM3)')
    parser.add_argument('--interval', type=float, default=60.0,
                        help='seconds between CSV rows (default: 60)')
    parser.add_argument('--list-ports', action='store_true',
                        help='show the serial ports this Python can see and exit')
    args = parser.parse_args(argv)

    if args.list_ports:
        ports = visible_ports()
        print(f'Python: {sys.platform} ({sys.executable})')
        print('Serial ports this Python can see:' if ports else 'This Python sees no serial port.')
        for device, description in ports:
            print(f'  {device}: {description}')
        return 0
    if args.csv_name is None:
        parser.error('the following arguments are required: csv_name (or use --list-ports)')

    ensure_header(args.csv_name)
    try:
        ser = open_port(args.port)
    except serial.SerialException as e:
        print(f'Error opening serial port {args.port}: {e}')
        for line in explain_open_failure(args.port, e):
            print(line)
        return 1
    print(f'Connected to {args.port}. Reading data...')

    last_slot = None
    last_data = time.monotonic()
    no_data_warned = False
    n_unparsed = 0
    try:
        while True:
            try:
                response_bytes = ser.readline()
            except serial.SerialException as e:
                print(f'Serial error: {e}')
                ser.close()
                ser = reconnect(args.port)
                last_data = time.monotonic()
                no_data_warned = False
                continue

            raw_str = response_bytes.decode('ascii', errors='ignore').strip()
            result = parse_tpg261_data(raw_str) if raw_str else None
            if result is None:
                if raw_str:
                    n_unparsed += 1
                    if n_unparsed <= MAX_UNPARSED_WARNINGS:
                        print(f'Unrecognized line: {raw_str!r}')
                if (not no_data_warned
                        and time.monotonic() - last_data > NO_DATA_WARN_S):
                    print(f'WARNING: no valid data for {NO_DATA_WARN_S} s')
                    no_data_warned = True
                continue

            last_data = time.monotonic()
            no_data_warned = False
            status, pressure = result

            now = datetime.now()
            slot = int(now.timestamp() // args.interval)
            if slot == last_slot:
                continue
            last_slot = slot

            timestamp = now.isoformat(timespec='seconds')
            append_row(args.csv_name, timestamp, status, pressure)
            if pressure is None:
                print(f'{timestamp} - {status_text(status)}')
            else:
                print(f'{timestamp} - {pressure:.4E}')

    except KeyboardInterrupt:
        print('Exiting...')

    finally:
        ser.close()
        print(f'Serial port {args.port} closed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
