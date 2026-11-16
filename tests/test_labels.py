"""Labelled event windows, detection credit and overlap tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.labels import (
    FAILURE_EVENTS,
    TAIL_WINDOW,
    event_overlap_table,
    failure_windows,
    label_mask,
)


def test_four_verified_events() -> None:
    assert len(FAILURE_EVENTS) == 4
    assert [e.event_id for e in FAILURE_EVENTS] == [1, 2, 3, 4]
    assert [e.start for e in FAILURE_EVENTS] == [
        pd.Timestamp("2020-04-18 00:00"),
        pd.Timestamp("2020-05-29 23:30"),
        pd.Timestamp("2020-06-05 10:00"),
        pd.Timestamp("2020-07-15 14:30"),
    ]
    assert failure_windows() is FAILURE_EVENTS


def test_published_end_is_represented_inclusively() -> None:
    event = FAILURE_EVENTS[0]
    index = pd.DatetimeIndex(
        ["2020-04-17 23:59", "2020-04-18 00:00", "2020-04-18 23:59", "2020-04-19 00:00"]
    )
    assert label_mask(index, [event]).tolist() == [False, True, True, False]


def test_detection_window_uses_tau_capped_at_six_hours() -> None:
    e1 = FAILURE_EVENTS[0].detection_window()   # 24 h -> tau = 6 h
    assert e1.start == pd.Timestamp("2020-04-17 22:00")
    assert e1.end == pd.Timestamp("2020-04-18 06:00")
    assert e1.end - FAILURE_EVENTS[0].start == pd.Timedelta(hours=6)

    # Event 4 is 4 h 31 min, so tau = |E| / 4; the window also carries the 2 h lead.
    event4 = FAILURE_EVENTS[3]
    e4 = event4.detection_window()
    tau = event4.window.duration / 4
    assert e4.start == pd.Timestamp("2020-07-15 12:30")
    assert e4.end - event4.start == tau
    assert e4.duration == pd.Timedelta(hours=2) + tau


def test_event_overlap_table_reports_share_of_full_window() -> None:
    event = FAILURE_EVENTS[0]
    index = pd.date_range("2020-04-18 00:00", periods=1440, freq="1min")
    mask = np.zeros(len(index), dtype=bool)
    mask[:10] = True
    table = event_overlap_table(index, mask, [event])
    row = table.iloc[0]
    assert row["window_minutes"] == 1440
    assert row["masked_minutes"] == 10
    assert row["masked_share"] == pytest.approx(10 / 1440)


def test_tail_window_is_the_declared_august_period() -> None:
    assert TAIL_WINDOW.start == pd.Timestamp("2020-08-01 00:00")
    assert TAIL_WINDOW.end == pd.Timestamp("2020-09-01 04:00")
