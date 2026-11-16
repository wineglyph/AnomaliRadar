"""Half-open time windows, the single abstraction used by masks and splits.

A ``TimeWindow`` is ``[start, end)``: ``start`` is inclusive, ``end`` is
exclusive. Half-open windows make contiguous splits unambiguous — a row exactly
on a boundary belongs to the later segment only — and they avoid the classic bug
where ``max(train) == min(calibration)``.

All timestamps in this project are naive local time, as stored in the CSV.
Timezone-aware inputs are converted to naive UTC instead of being silently
mixed with naive values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _naive(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A half-open interval ``[start, end)``."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start = _naive(self.start)
        end = _naive(self.end)
        if end < start:
            raise ValueError(f"window end {end} precedes start {start}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    # -- basic properties -------------------------------------------------- #
    @property
    def duration(self) -> pd.Timedelta:
        return self.end - self.start

    @property
    def duration_hours(self) -> float:
        return self.duration.total_seconds() / 3600.0

    def __contains__(self, value: Any) -> bool:
        ts = _naive(value)
        return bool(self.start <= ts < self.end)

    def __len__(self) -> int:
        """Number of whole minutes in the window (convenience for reporting)."""
        return int(self.duration / pd.Timedelta(minutes=1))

    # -- set operations ---------------------------------------------------- #
    def mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Boolean array selecting timestamps inside the window."""
        idx = pd.DatetimeIndex(index)
        return np.asarray((idx >= self.start) & (idx < self.end), dtype=bool)

    def intersect(self, other: "TimeWindow") -> "TimeWindow | None":
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if end <= start:
            return None
        return TimeWindow(start, end)

    def overlap_hours(self, other: "TimeWindow") -> float:
        inter = self.intersect(other)
        return 0.0 if inter is None else inter.duration_hours

    def shifted(self, **kwargs: Any) -> "TimeWindow":
        """Return a copy moved by a ``pd.Timedelta`` keyword (e.g. hours=1)."""
        delta = pd.Timedelta(**kwargs)
        return TimeWindow(self.start + delta, self.end + delta)

    def __str__(self) -> str:
        return f"[{self.start:%Y-%m-%d %H:%M}, {self.end:%Y-%m-%d %H:%M})"
