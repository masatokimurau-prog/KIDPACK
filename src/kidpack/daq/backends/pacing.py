"""Fixed-interval scheduling for software-issued (random) triggers."""
import time


class IntervalPacer:
    """Release one tick every ``interval`` seconds without accumulating drift.

    The first ``wait()`` returns one interval after it was called, so N
    events take N intervals and the resulting rate is exactly 1/interval.
    Ticks are scheduled on a fixed grid (t0 + interval, t0 + 2 * interval, ...),
    so the time spent between calls (writing a file, ...) does not shift later
    ticks. If the caller falls more than one interval behind, the missed ticks
    are dropped rather than fired as a burst, and the grid restarts from the
    current time.
    """

    def __init__(self, interval, clock=time.monotonic, sleep=time.sleep):
        self.interval = interval
        self._clock = clock
        self._sleep = sleep
        self._next = None

    def wait(self):
        now = self._clock()
        if self._next is None:
            self._next = now + self.interval
        if self._next > now:
            self._sleep(self._next - now)
        scheduled = self._next
        self._next = scheduled + self.interval
        if self._next <= self._clock():  # more than one interval late
            self._next = self._clock() + self.interval
        return scheduled
