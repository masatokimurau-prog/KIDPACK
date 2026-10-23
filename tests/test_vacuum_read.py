import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

READ_PY = Path(__file__).resolve().parents[1] / 'scripts' / 'VacuumGauge' / 'read.py'


@pytest.fixture
def read():
    spec = importlib.util.spec_from_file_location('vacuum_read', READ_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('raw, expected', [
    ('0,1.2345E-03', (0, 1.2345e-3)),
    ('0,1.0000E+03', (0, 1e3)),
    (' 0 , 5.0000E-09 ', (0, 5e-9)),
    ('1,1.0000E-11', (1, None)),
    ('2,1.0000E+03', (2, None)),
    ('3,0.0000E+00', (3, None)),
    ('4,0.0000E+00', (4, None)),
    ('5,0.0000E+00', (5, None)),
    ('9,0.0000E+00', (9, None)),
])
def test_parse_valid(read, raw, expected):
    assert read.parse_tpg261_data(raw) == expected


@pytest.mark.parametrize('raw', [
    '', 'garbage', '0.1.2345E-03', '0,abc', '0,1,2', 'x,1.0E-3', '0',
])
def test_parse_invalid(read, raw):
    assert read.parse_tpg261_data(raw) is None


def test_ensure_header_written_once(read, tmp_path):
    path = tmp_path / 'p.csv'
    read.ensure_header(path)
    read.ensure_header(path)
    assert list(csv.reader(path.open())) == [read.CSV_HEADER]


def test_append_row(read, tmp_path):
    path = tmp_path / 'p.csv'
    read.append_row(path, '2026-09-29T12:00:00', 0, 1.2345e-3)
    read.append_row(path, '2026-09-29T12:01:00', 4, None)
    assert list(csv.reader(path.open())) == [
        ['2026-09-29T12:00:00', '0', '1.2345E-03'],
        ['2026-09-29T12:01:00', '4', ''],
    ]


# --- main loop with a fake serial port ------------------------------------

class Interrupt:
    pass


class SerialFailure:
    pass


@pytest.fixture
def run_main(read, tmp_path, monkeypatch):
    """Run main() against scripted events: (datetime, line) / SerialFailure / Interrupt."""
    def run(events):
        events = list(events)
        clock = {'now': None}

        class FakeSerial:
            opened = 0

            def __init__(self, **kwargs):
                FakeSerial.opened += 1

            def reset_input_buffer(self):
                pass

            def close(self):
                pass

            def readline(self):
                event = events.pop(0)
                if isinstance(event, Interrupt):
                    raise KeyboardInterrupt
                if isinstance(event, SerialFailure):
                    raise read.serial.SerialException('unplugged')
                clock['now'], line = event
                return line

        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock['now']

        monkeypatch.setattr(read.serial, 'Serial', FakeSerial)
        monkeypatch.setattr(read, 'datetime', FakeDatetime)
        monkeypatch.setattr(read, 'RECONNECT_WAIT_S', 0)
        path = tmp_path / 'p.csv'
        assert read.main([str(path), 'FAKE']) == 0
        return list(csv.reader(path.open())), FakeSerial.opened
    return run


def t(minute, second, micro=0):
    return datetime(2026, 9, 29, 12, minute, second, micro)


def test_one_row_per_minute_even_when_second_00_is_missed(run_main):
    rows, _ = run_main([
        (t(0, 0, 200000), b'0,1.0000E-03\r\n'),   # first reading -> written
        (t(0, 1, 200000), b'0,1.1000E-03\r\n'),   # same minute -> skipped
        (t(0, 59, 900000), b'0,1.2000E-03\r\n'),  # same minute -> skipped
        (t(1, 1, 400000), b'0,2.0000E-03\r\n'),   # :00 missed, new minute -> written
        (t(2, 0, 500000), b'4,0.0000E+00\r\n'),   # sensor off -> status, no pressure
        (t(2, 1), b'garbage\r\n'),                # unparsable -> ignored
        Interrupt(),
    ])
    assert rows == [
        ['timestamp', 'status', 'pressure'],
        ['2026-09-29T12:00:00', '0', '1.0000E-03'],
        ['2026-09-29T12:01:01', '0', '2.0000E-03'],
        ['2026-09-29T12:02:00', '4', ''],
    ]


def test_reconnects_after_serial_failure(run_main):
    rows, opened = run_main([
        (t(0, 0), b'0,1.0000E-03\r\n'),
        SerialFailure(),
        (t(1, 0), b'0,2.0000E-03\r\n'),
        Interrupt(),
    ])
    assert opened == 2
    assert [r[2] for r in rows[1:]] == ['1.0000E-03', '2.0000E-03']


# --- when the port cannot be opened -----------------------------------------

POSIX_NOT_FOUND = "could not open port COM3: [Errno 2] No such file or directory: 'COM3'"  # what pyserial says there
WINDOWS_NOT_FOUND = ("could not open port 'COM3': FileNotFoundError(2, 'The system cannot find the file "
                     "specified.', None, 2)")
WINDOWS_DENIED = "could not open port 'COM3': PermissionError(13, 'Access is denied.', None, 5)"


WSL1 = '4.4.0-19041-Microsoft'
WSL2 = '5.15.153.1-microsoft-standard-WSL2'


@pytest.mark.parametrize('release, version', [
    (WSL1, 1), (WSL2, 2), ('5.10.16.3-microsoft-standard-WSL2', 2), ('6.6.36.3-microsoft-standard-WSL2', 2),
    ('5.15.0-91-generic', None), ('25.5.0', None), ('', None)])
def test_wsl1_and_wsl2_are_told_apart_by_the_kernel_release(read, release, version):
    assert read.wsl_version(release) == version


@pytest.mark.parametrize('system', ['linux', 'cygwin', 'msys', 'darwin'])
def test_a_windows_port_name_on_a_python_that_is_not_windows_is_explained(read, system):
    error = read.serial.SerialException(2, POSIX_NOT_FOUND)
    text = '\n'.join(read.explain_open_failure('COM3', error, system, release='5.15.0-91-generic'))
    assert 'COM3 is a Windows port name' in text and repr(system) in text
    assert '/dev/ttyS2' in text and 'WSL2' in text and 'py read.py' in text  # what to do


def test_in_wsl2_the_com_ports_cannot_be_seen_so_the_windows_python_is_the_way(read):
    error = read.serial.SerialException(2, POSIX_NOT_FOUND)
    text = '\n'.join(read.explain_open_failure('COM3', error, 'linux', WSL2))
    assert 'this Ubuntu is WSL2' in text and WSL2 in text and 'cannot see the Windows COM ports' in text
    assert 'py read.py OUT.csv COM3' in text and 'py -m pip install pyserial' in text and 'usbipd-win' in text
    assert 'python3 read.py OUT.csv /dev/ttyS2' not in text  # that would not work there


@pytest.mark.parametrize('port, device', [('COM3', '/dev/ttyS2'), ('COM1', '/dev/ttyS0'), ('COM10', '/dev/ttyS9')])
def test_in_wsl1_the_port_is_the_ttyS_file_with_the_number_minus_one(read, port, device):
    error = read.serial.SerialException(2, f"could not open port {port}: [Errno 2] No such file or directory: '{port}'")
    text = '\n'.join(read.explain_open_failure(port, error, 'linux', WSL1))
    assert f'in WSL1 (kernel {WSL1}) the port is the device file {device}' in text
    assert f'python3 read.py OUT.csv {device}' in text and f'sudo chmod 666 {device}' in text
    assert 'usbipd' not in text


def test_the_same_port_name_on_windows_is_not_called_a_posix_problem(read, monkeypatch):
    monkeypatch.setattr(read, 'visible_ports', lambda: [('COM4', 'USB Serial Port (COM4)')])
    error = read.serial.SerialException(WINDOWS_NOT_FOUND)
    lines = read.explain_open_failure('COM3', error, 'win32')
    text = '\n'.join(lines)
    assert 'Windows port name' not in text and 'COM3 does not exist for this Python' in text
    assert '  COM4: USB Serial Port (COM4)' in lines  # what it does see


@pytest.mark.parametrize('port, platform, message', [
    ('COM3', 'win32', WINDOWS_DENIED),
    ('/dev/ttyUSB0', 'linux', 'could not open port /dev/ttyUSB0: [Errno 13] Permission denied'),
])
def test_a_port_that_is_in_use_or_not_allowed_is_explained(read, monkeypatch, port, platform, message):
    monkeypatch.setattr(read, 'visible_ports', lambda: [])
    text = '\n'.join(read.explain_open_failure(port, read.serial.SerialException(message), platform))
    assert f'{port} exists but cannot be opened' in text and 'another program' in text
    errno_version = read.serial.SerialException(13, message)  # with the errno that pyserial sets on POSIX
    assert 'another program' in '\n'.join(read.explain_open_failure(port, errno_version, platform))


def test_a_device_file_that_does_not_exist_is_explained(read, monkeypatch):
    monkeypatch.setattr(read, 'visible_ports', lambda: [('/dev/ttyUSB1', 'FT232R')])
    error = read.serial.SerialException(2, "could not open port /dev/ttyUSB0: [Errno 2] No such file or directory")
    lines = read.explain_open_failure('/dev/ttyUSB0', error, 'linux')
    assert any('does not exist for this Python' in line for line in lines)
    assert '  /dev/ttyUSB1: FT232R' in lines


def test_without_any_port_the_explanation_says_so(read, monkeypatch):
    monkeypatch.setattr(read, 'visible_ports', lambda: [])
    lines = read.explain_open_failure('COM3', read.serial.SerialException(WINDOWS_NOT_FOUND), 'win32')
    assert any('sees no serial port' in line and '--list-ports' in line for line in lines)


def test_the_visible_ports_are_sorted_and_a_failing_listing_is_no_ports(read, monkeypatch):
    class Port:
        def __init__(self, device, description):
            self.device, self.description = device, description

    monkeypatch.setattr(read.list_ports, 'comports', lambda: [Port('COM5', 'b'), Port('COM3', 'a')])
    assert read.visible_ports() == [('COM3', 'a'), ('COM5', 'b')]

    def broken():
        raise OSError('no listing here')
    monkeypatch.setattr(read.list_ports, 'comports', broken)
    assert read.visible_ports() == []


def test_main_prints_the_cause_and_returns_1_when_the_port_cannot_be_opened(read, monkeypatch, tmp_path, capsys):
    def refuse(**kwargs):
        raise read.serial.SerialException(2, POSIX_NOT_FOUND)

    monkeypatch.setattr(read.serial, 'Serial', refuse)
    monkeypatch.setattr(read, 'visible_ports', lambda: [('/dev/ttyS2', 'ttyS2')])
    assert read.main([str(tmp_path / 'p.csv')]) == 1
    out = capsys.readouterr().out
    assert f'Error opening serial port COM3: [Errno 2] {POSIX_NOT_FOUND}' in out  # the original message stays
    assert '  /dev/ttyS2: ttyS2' in out
    if read.sys.platform != 'win32':
        assert 'COM3 is a Windows port name' in out


def test_list_ports_shows_the_ports_and_the_python_and_needs_no_file(read, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(read, 'visible_ports', lambda: [('COM3', 'USB Serial Port (COM3)')])
    monkeypatch.chdir(tmp_path)
    assert read.main(['--list-ports']) == 0
    out = capsys.readouterr().out
    assert read.sys.platform in out and '  COM3: USB Serial Port (COM3)' in out
    assert list(tmp_path.iterdir()) == []  # nothing is created

    monkeypatch.setattr(read, 'visible_ports', lambda: [])
    assert read.main(['--list-ports']) == 0
    assert 'sees no serial port' in capsys.readouterr().out


def test_a_csv_name_is_still_required_without_list_ports(read, capsys):
    with pytest.raises(SystemExit) as exit_info:
        read.main([])
    assert exit_info.value.code == 2 and 'csv_name' in capsys.readouterr().err
