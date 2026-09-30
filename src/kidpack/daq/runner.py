"""Run control: acquisition loop, file segmentation, logging, safe shutdown."""
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from kidpack.daq.summary import append_run_summary
from kidpack.daq.writer import RunWriter

log = logging.getLogger('kidpack.daq')

PROGRESS_INTERVAL_S = 10.0
EXIT_CODES = {'completed': 0, 'interrupted': 130, 'error': 1}


@dataclass
class RunResult:
    status: str  # 'completed' | 'interrupted' | 'error'
    run_dir: str
    run_start_ns: int
    run_stop_ns: int
    total_events: int
    files_written: int
    daq_rate_hz: Optional[float]
    error: Optional[str] = None

    @property
    def exit_code(self):
        return EXIT_CODES[self.status]


@contextlib.contextmanager
def log_to_console(logger=log):
    """Show ``logger``'s progress on stderr for the duration of a run (nothing is written to disk)."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def run_daq(cfg, scope, sg=None, writer=None, command_line=None):
    """Run one DAQ run and return a RunResult.

    ``scope`` and ``sg`` (optional) are backends; they are closed before
    returning. Raises FileExistsError, before touching anything, if the run
    directory already exists.
    """
    writer = writer or RunWriter(cfg, scope.timestamp_source, command_line)
    writer.timestamp_source = scope.timestamp_source
    try:
        writer.create()
        return _Run(cfg, scope, sg, writer).execute()
    finally:
        for backend in (sg, scope):
            if backend is not None:
                try:
                    backend.close()
                except Exception as e:
                    log.warning('failed to close %s: %s', type(backend).__name__, e)


class _Run:
    def __init__(self, cfg, scope, sg, writer):
        self.cfg, self.scope, self.sg, self.writer = cfg, scope, sg, writer
        self.actual = {}
        self.events = []  # events of the file being acquired
        self.file_number = -1
        self.total_events = 0
        self.total_acq_s = 0.0
        self.files_written = 0
        self._file_start_ns = self._file_stop_ns = None
        self._file_acq_s = 0.0

    def execute(self):
        cfg = self.cfg
        run_start_ns = self.writer.run_start_ns = time.time_ns()
        status, error = 'completed', None
        run_stop_ns = None

        with log_to_console():
            log.info('run %02d start: %d file(s) x %d events -> %s',
                     cfg.run_number, cfg.num_files, cfg.events_per_file, self.writer.run_dir)
            log.info('condition: %s', cfg.condition or '(none)')
            log.info('config: %s', json.dumps(cfg.to_dict()))
            try:
                self.actual = self.scope.configure()
                log.info('actual device settings: %s', json.dumps(self.actual))
                if self.sg is not None:
                    self.sg.start(cfg.sg.frequency, cfg.sg.power)
                    log.info('SG started: %g Hz, %g dBm', cfg.sg.frequency, cfg.sg.power)
                for file_number in range(cfg.num_files):
                    self._acquire_file(file_number)
            except KeyboardInterrupt:
                status = 'interrupted'
                log.warning('stopped by user (Ctrl-C)')
            except Exception as e:
                status, error = 'error', f'{type(e).__name__}: {e}'
                log.exception('DAQ error')

            run_stop_ns = self._file_stop_ns or time.time_ns()

            if self.events:
                self._flush_partial(status, error)
            self._stop_sg()

            rate = self.total_events / self.total_acq_s if self.total_events and self.total_acq_s > 0 else None
            try:
                append_run_summary(cfg.summary_path, cfg.run_number, run_start_ns,
                                   run_stop_ns, rate, cfg.condition)
            except OSError:
                log.exception('could not update the run summary %s', cfg.summary_path)

            log.info('run %02d %s: %d events in %d file(s), DAQ rate %s',
                     cfg.run_number, status, self.total_events, self.files_written,
                     'n/a' if rate is None else f'{rate:.2f} Hz')

        return RunResult(status, self.writer.run_dir, run_start_ns, run_stop_ns,
                         self.total_events, self.files_written, rate, error)

    def _acquire_file(self, file_number):
        self.file_number = file_number
        self.events = []
        self._file_start_ns = time.time_ns()
        started = last_report = time.perf_counter()
        try:
            for _ in range(self.cfg.events_per_file):
                self.events.append(self.scope.acquire())
                now = time.perf_counter()
                if now - last_report >= PROGRESS_INTERVAL_S:
                    last_report = now
                    log.info('file %02d: %d/%d events', file_number,
                             len(self.events), self.cfg.events_per_file)
        finally:
            self._file_acq_s = time.perf_counter() - started
            self._file_stop_ns = time.time_ns()
        self._flush('complete')

    def _flush(self, status, errors=()):
        n = len(self.events)
        rate = n / self._file_acq_s if n and self._file_acq_s > 0 else None
        path = self.writer.write_file(
            self.file_number, self.events, file_start_ns=self._file_start_ns,
            file_stop_ns=self._file_stop_ns, daq_rate_hz=rate, status=status,
            actual=self.actual, errors=errors)
        self.total_events += n
        self.total_acq_s += self._file_acq_s
        self.files_written += 1
        self.events = []
        log.info('file %02d %s: %d events, %s -> %s', self.file_number, status, n,
                 'rate n/a' if rate is None else f'{rate:.2f} Hz', path)

    def _flush_partial(self, status, error):
        try:
            self._flush(status, [error] if error else [])
        except Exception:
            log.exception('failed to write the partial file run%02d-%02d',
                          self.cfg.run_number, self.file_number)

    def _stop_sg(self):
        if self.sg is None:
            return
        try:
            self.sg.stop()
            log.info('SG stopped')
        except Exception:
            log.exception('failed to stop the SG')
