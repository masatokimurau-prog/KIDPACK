"""Timestamp formatting helpers (all inputs are ns since the Unix epoch)."""
from datetime import datetime, timezone


def iso_utc(ns):
    """ISO 8601 in UTC, e.g. 2026-09-29T05:03:12.123+00:00."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec='milliseconds')


def iso_local(ns):
    """ISO 8601 in the local time zone, e.g. 2026-09-29T14:03:12+09:00."""
    return datetime.fromtimestamp(ns / 1e9).astimezone().isoformat(timespec='seconds')
