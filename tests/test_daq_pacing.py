import pytest

from kidpack.daq.backends.pacing import IntervalPacer


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.slept = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def work(self, seconds):
        self.now += seconds


@pytest.fixture
def t():
    return FakeTime()


def pacer(t, interval=1.0):
    return IntervalPacer(interval, clock=t.clock, sleep=t.sleep)


def test_one_tick_per_interval_starting_one_interval_after_the_first_call(t):
    p = pacer(t)
    ticks = []
    for _ in range(4):
        p.wait()
        ticks.append(t.now)
    assert ticks == [101.0, 102.0, 103.0, 104.0]  # N events take N intervals: rate = 1/interval
    assert t.slept == [1.0, 1.0, 1.0, 1.0]


def test_work_between_ticks_does_not_shift_the_grid(t):
    p = pacer(t)
    ticks = []
    for _ in range(4):
        p.wait()
        ticks.append(t.now)
        t.work(0.3)  # e.g. fetch + bookkeeping
    assert ticks == pytest.approx([101.0, 102.0, 103.0, 104.0])  # no drift


def test_a_late_caller_catches_up_when_less_than_one_interval_behind(t):
    p = pacer(t)
    p.wait()  # 101.0
    t.work(1.4)  # e.g. writing a file: the 102.0 tick is missed by 0.4 s
    p.wait()
    assert t.now == pytest.approx(102.4)  # fires immediately ...
    p.wait()
    assert t.now == pytest.approx(103.0)  # ... and is back on the original grid


def test_missed_ticks_are_dropped_not_fired_as_a_burst(t):
    p = pacer(t)
    p.wait()  # 101.0
    t.work(5.5)  # five ticks missed
    p.wait()
    assert t.now == pytest.approx(106.5)  # one immediate trigger
    before = len(t.slept)
    p.wait()
    assert t.slept[before:] == [pytest.approx(1.0)]  # then a full interval, no burst
    assert t.now == pytest.approx(107.5)


def test_sub_second_interval(t):
    p = pacer(t, interval=0.25)
    for _ in range(3):
        p.wait()
    assert t.now == pytest.approx(100.75)
