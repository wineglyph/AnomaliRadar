"""Labelled failure windows and detection windows.

The four windows are the maintenance reports published with MetroPT-3. The PDF
table is mis-numbered (#1 appears twice, then #3, #4); ``event_id`` is sequential
and ``pdf_label`` preserves the published text.

Representation detail: the published windows are *minute-inclusive* (e.g.
``2020-04-18 23:59``). They are stored here as half-open windows ending one
minute later, so ``[start, end)`` covers every minute the report names and no
row is lost at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from anomaliradar.timewindow import TimeWindow

#: Ground-truth windows, verified against the dataset PDF on Day 0.
FAILURE_EVENTS: tuple["EventWindow", ...] = ()


@dataclass(frozen=True, slots=True)
class EventWindow:
    """A labelled event plus the derived detection window."""

    event_id: int
    pdf_label: str
    kind: str
    window: TimeWindow

    @property
    def start(self) -> pd.Timestamp:
        return self.window.start

    @property
    def end(self) -> pd.Timestamp:
        return self.window.end

    @property
    def duration_hours(self) -> float:
        return self.window.duration_hours

    def detection_window(
        self,
        *,
        lead: pd.Timedelta = pd.Timedelta(hours=2),
        max_tau: pd.Timedelta = pd.Timedelta(hours=6),
    ) -> TimeWindow:
        """Window in which an alarm earns detection credit.

        ``tau = min(max_tau, |E| / 4)``; the start is ``t_start - lead``.
        """
        tau = min(max_tau, self.window.duration / 4)
        return TimeWindow(self.start - lead, self.start + tau)


FAILURE_EVENTS = (
    EventWindow(
        1,
        "#1",
        "air leak",
        # published end 2020-04-18 23:59
        TimeWindow("2020-04-18 00:00", "2020-04-19 00:00"),
    ),
    EventWindow(
        2,
        "#1 (duplicate)",
        "air leak",
        # published end 2020-05-30 06:00
        TimeWindow("2020-05-29 23:30", "2020-05-30 06:01"),
    ),
    EventWindow(
        3,
        "#3",
        "air leak",
        # published end 2020-06-07 14:30
        TimeWindow("2020-06-05 10:00", "2020-06-07 14:31"),
    ),
    EventWindow(
        4,
        "#4",
        "air leak",
        # published end 2020-07-15 19:00
        TimeWindow("2020-07-15 14:30", "2020-07-15 19:01"),
    ),
)


def failure_windows() -> tuple[EventWindow, ...]:
    """Return the labelled events (immutable, module-level constants)."""
    return FAILURE_EVENTS


def as_windows(events: Iterable[EventWindow] | None = None) -> tuple[TimeWindow, ...]:
    chosen = FAILURE_EVENTS if events is None else tuple(events)
    return tuple(e.window for e in chosen)


def label_mask(
    index: pd.DatetimeIndex,
    events: Sequence[EventWindow] | None = None,
) -> np.ndarray:
    """Boolean array: True where ``index`` falls inside any labelled window."""
    chosen = FAILURE_EVENTS if events is None else events
    mask = np.zeros(len(index), dtype=bool)
    for event in chosen:
        mask |= event.window.mask(index)
    return mask


def event_overlap_table(
    index: pd.DatetimeIndex,
    mask: np.ndarray,
    events: Sequence[EventWindow] | None = None,
) -> pd.DataFrame:
    """Per-event overlap of a boolean row mask with the labelled windows.

    Returns columns ``event_id``, ``window_hours``, ``masked_minutes``,
    ``masked_hours`` and ``masked_share``. ``masked_share`` is relative to the
    full wall-clock window, so a gap counts even though it has no grid values.
    """
    chosen = FAILURE_EVENTS if events is None else events
    rows = []
    for event in chosen:
        in_window = event.window.mask(index)
        total = int(in_window.sum())
        overlap = int((in_window & np.asarray(mask, dtype=bool)).sum())
        rows.append(
            {
                "event_id": event.event_id,
                "window_hours": round(event.duration_hours, 3),
                "window_minutes": total,
                "masked_minutes": overlap,
                "masked_hours": round(overlap / 60.0, 3),
                "masked_share": (overlap / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


#: The only clean out-of-sample false-alarm period, used by fold T.
TAIL_WINDOW = TimeWindow("2020-08-01 00:00", "2020-09-01 04:00")
