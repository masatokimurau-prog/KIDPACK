"""Scan control: frequency loop, safe shutdown, files and summary."""
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from kidpack.daq.runner import EXIT_CODES, log_to_console
from kidpack.iqscan.summary import append_scan_summary
from kidpack.iqscan.writer import IqScanWriter

log = logging.getLogger('kidpack.iqscan')


@dataclass
class ScanResult:
    status: str  # 'completed' | 'interrupted' | 'error'
    npz_path: Optional[str]  # None if not a single frequency was measured
    start_ns: int
    stop_ns: int
    points_measured: int
    error: Optional[str] = None

    @property
    def exit_code(self):
        return EXIT_CODES[self.status]


def run_iqscan(cfg, scope, sg, writer=None, command_line=None, sleep=time.sleep):
    """Run one scan and return a ScanResult; ``scope`` and ``sg`` are closed before returning.

    Per frequency, as in the seed ``iq_scan.py``: set frequency and power and start
    the generator, acquire, stop the generator. Whatever was measured is written
    even if the scan is stopped (Ctrl-C) or fails; the generator is always stopped.
    Raises FileExistsError, before touching any instrument, if the scan file exists.
    """
    writer = writer or IqScanWriter(cfg, command_line=command_line)
    writer.timestamp_source = scope.timestamp_source
    try:
        writer.check_new()
        return _scan(cfg, scope, sg, writer, sleep)
    finally:
        for backend in (sg, scope):
            try:
                backend.close()
            except Exception as e:
                log.warning('failed to close %s: %s', type(backend).__name__, e)


def _stop_sg(sg):
    """Stop the generator; a failure is logged, not raised, so it cannot hide the original error."""
    try:
        sg.stop()
    except Exception:
        log.exception('FAILED TO STOP THE SIGNAL GENERATOR - check that its output is off')


def _scan(cfg, scope, sg, writer, sleep):
    frequencies = cfg.frequencies()
    points, actual = [], {}
    status, error = 'completed', None
    start_ns = time.time_ns()

    with log_to_console(log):
        log.info('IQ scan %s: %d points, %.6f - %.6f GHz, %g dBm', writer.stem, cfg.num_points,
                 cfg.f_start / 1e9, cfg.f_stop / 1e9, cfg.power)
        log.info('condition: %s', cfg.condition or '(none)')
        log.info('config: %s', json.dumps(cfg.to_dict()))
        try:
            actual = scope.configure()
            log.info('actual device settings: %s', json.dumps(actual))
            for k, frequency in enumerate(frequencies):
                try:
                    sg.start(float(frequency), cfg.power)
                    if cfg.settle_time > 0:
                        sleep(cfg.settle_time)
                    m = scope.measure()
                finally:
                    _stop_sg(sg)  # also when the acquisition fails or is interrupted
                points.append(m)
                log.info('%d of %d: %.6f GHz, %.6f, %.6f', k + 1, cfg.num_points,
                         frequency / 1e9, m.mean0, m.mean1)
        except KeyboardInterrupt:
            status = 'interrupted'
            log.warning('stopped by user (Ctrl-C)')
        except Exception as e:
            status, error = 'error', f'{type(e).__name__}: {e}'
            log.exception('scan error')
        stop_ns = time.time_ns()

        npz_path = None
        if points:
            try:
                npz_path = writer.write(frequencies, points, start_ns=start_ns, stop_ns=stop_ns,
                                        status=status, actual=actual,
                                        errors=[error] if error else [])
                append_scan_summary(cfg.summary_path, writer.stem, start_ns, stop_ns,
                                    cfg.f_start, cfg.f_stop, len(points), cfg.power, status,
                                    cfg.condition)
            except Exception as e:
                status, error = 'error', error or f'{type(e).__name__}: {e}'
                log.exception('could not write the scan files')
        log.info('IQ scan %s %s: %d of %d points%s', writer.stem, status, len(points),
                 cfg.num_points, f' -> {npz_path}' if npz_path else ' (nothing written)')

    return ScanResult(status, npz_path, start_ns, stop_ns, len(points), error)
