"""Log pressure from a Pfeiffer TPG 261 vacuum gauge to a CSV file.

The gauge is expected to be in continuous output mode; every line it sends
has the form ``status,value`` (e.g. ``0,1.2345E-03``). The pressure unit is
whatever is configured on the gauge.

Usage:
    python read.py OUT.csv [PORT] [--interval SECONDS]

One row is written per interval, aligned to the wall clock (for the default
of 60 s: the first reading after every minute boundary).
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import serial

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
    parser.add_argument('csv_name', help='output CSV file (appended to)')
    parser.add_argument('port', nargs='?', default='COM3',
                        help='serial port (default: COM3)')
    parser.add_argument('--interval', type=float, default=60.0,
                        help='seconds between CSV rows (default: 60)')
    args = parser.parse_args(argv)

    ensure_header(args.csv_name)
    try:
        ser = open_port(args.port)
    except serial.SerialException as e:
        print(f'Error opening serial port {args.port}: {e}')
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
