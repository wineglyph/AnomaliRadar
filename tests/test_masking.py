"""Masks M1–M3: threshold behaviour, warm-up, label independence, reporting."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.labels import EventWindow
from anomaliradar.data.masking import (
    MaskConfig,
    array_signature,
    build_grid_masks,
    detect_frozen_blocks,
    detect_major_gaps,
    detect_native_masks,
    mask_summary,
)
from anomaliradar.data.pipeline import build_dataset
from anomaliradar.timewindow import TimeWindow
from helpers import drop_minutes, freeze_analog, make_native_frame


# --------------------------------------------------------------------------- #
# M1 frozen blocks
# --------------------------------------------------------------------------- #
def test_m1_detects_exactly_one_hour_of_constant_samples() -> None:
    frame = freeze_analog(make_native_frame(minutes=720, freq="10s"), 100, 460)
    blocks = detect_frozen_blocks(frame)
    assert len(blocks) == 1
    assert blocks.iloc[0]["rows"] == 360
    assert blocks.iloc[0]["start"] == frame["timestamp"].iloc[100]
    assert blocks.iloc[0]["end"] == frame["timestamp"].iloc[459]


def test_m1_ignores_359_rows() -> None:
    frame = freeze_analog(make_native_frame(minutes=720, freq="10s"), 100, 459)
    assert detect_frozen_blocks(frame).empty


def test_m1_requires_every_analog_channel_to_be_constant() -> None:
    frame = freeze_analog(make_native_frame(minutes=720, freq="10s"), 100, 460)
    frame.loc[frame.index[200], "TP3"] += 1.0
    # The changed row splits the run into 100 and 259 rows; neither reaches 360.
    assert detect_frozen_blocks(frame).empty


def test_m1_native_share_and_max_duration() -> None:
    frame = freeze_analog(make_native_frame(minutes=720, freq="10s"), 100, 460)
    masks = detect_native_masks(frame)
    assert masks.native_rows == 720
    assert masks.summary()["m1_native_rows"] == 360
    assert masks.summary()["m1_native_share"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# M2 major gaps
# --------------------------------------------------------------------------- #
def _two_timestamp_frame(minutes_between: int) -> pd.DataFrame:
    frame = make_native_frame(minutes=2)
    frame.loc[frame.index[1], "timestamp"] = frame["timestamp"].iloc[0] + pd.Timedelta(
        minutes=minutes_between
    )
    return frame


def test_m2_detects_gap_strictly_longer_than_one_hour() -> None:
    assert detect_major_gaps(_two_timestamp_frame(60)).empty
    gaps = detect_major_gaps(_two_timestamp_frame(61))
    assert len(gaps) == 1
    assert gaps.iloc[0]["hours"] == pytest.approx(61 / 60)


def test_m2_reports_actual_timestamps() -> None:
    frame = _two_timestamp_frame(90)
    gap = detect_major_gaps(frame).iloc[0]
    assert gap["start"] == frame["timestamp"].iloc[0]
    assert gap["end"] == frame["timestamp"].iloc[1]
    assert gap["missing_minutes"] == 89


# --------------------------------------------------------------------------- #
# M3 warm-up
# --------------------------------------------------------------------------- #
def test_m3_marks_thirty_minutes_after_a_major_gap() -> None:
    frame = make_native_frame(minutes=250)
    gapped = drop_minutes(frame, list(range(50, 140)))  # 90 missing minutes
    ds = build_dataset(gapped)
    index = ds.index

    gap_rows = (index > pd.Timestamp("2020-02-01 00:49")) & (
        index < pd.Timestamp("2020-02-01 02:20")
    )
    warm_rows = (index >= pd.Timestamp("2020-02-01 02:20")) & (
        index < pd.Timestamp("2020-02-01 02:50")
    )

    assert ds.masks.m1.sum() == 0
    assert int(ds.masks.m2.sum()) == int(gap_rows.sum()) == 90
    assert int(ds.masks.m3.sum()) == int(warm_rows.sum()) == 30
    assert np.array_equal(ds.masks.m3, warm_rows)


def test_m3_marks_thirty_minutes_after_a_frozen_block() -> None:
    frame = freeze_analog(make_native_frame(minutes=150), 20, 90)
    cfg = MaskConfig(frozen_min_rows=60)
    from anomaliradar.data.pipeline import DatasetConfig

    ds = build_dataset(frame, config=DatasetConfig(mask=cfg))
    expected_m1 = (ds.index >= pd.Timestamp("2020-02-01 00:20")) & (
        ds.index < pd.Timestamp("2020-02-01 01:30")
    )
    expected_m3 = (ds.index >= pd.Timestamp("2020-02-01 01:30")) & (
        ds.index < pd.Timestamp("2020-02-01 02:00")
    )
    assert np.array_equal(ds.masks.m1, expected_m1)
    assert np.array_equal(ds.masks.m3, expected_m3)


def test_m3_is_empty_for_short_breaks() -> None:
    frame = make_native_frame(minutes=60)
    ds = build_dataset(drop_minutes(frame, [20, 21]))
    assert ds.masks.m2.sum() == 0
    assert ds.masks.m3.sum() == 0


# --------------------------------------------------------------------------- #
# Label independence and signatures
# --------------------------------------------------------------------------- #
def test_mask_functions_have_no_label_parameter() -> None:
    for func in (detect_frozen_blocks, detect_major_gaps, detect_native_masks, build_grid_masks):
        params = set(inspect.signature(func).parameters)
        assert not params & {"labels", "events", "y", "target"}, (func.__name__, params)


def test_mask_signature_is_a_pure_function_of_the_sensor_frame() -> None:
    frame = freeze_analog(make_native_frame(minutes=400), 100, 180)
    first = detect_native_masks(frame)
    second = detect_native_masks(frame.copy())
    assert first.frozen_blocks.equals(second.frozen_blocks)
    assert first.major_gaps.equals(second.major_gaps)

    grid = build_dataset(frame).grid
    masks_a = build_grid_masks(grid, first)
    masks_b = build_grid_masks(grid, second)
    assert masks_a.signature() == masks_b.signature()


def test_array_signature_changes_with_content() -> None:
    a = np.zeros(10, dtype=bool)
    b = a.copy()
    b[3] = True
    assert array_signature(m1=a) != array_signature(m1=b)
    assert array_signature(m1=a) != array_signature(m2=a)


# --------------------------------------------------------------------------- #
# Reporting: M1 and M2 overlaps must be separate
# --------------------------------------------------------------------------- #
def test_mask_summary_reports_m1_and_m2_overlaps_separately() -> None:
    frame = freeze_analog(make_native_frame(minutes=300), 0, 60)
    gapped = drop_minutes(frame, list(range(150, 211)))  # 61-minute gap

    event_frozen = EventWindow(
        1, "synthetic-frozen", "test", TimeWindow("2020-02-01 00:10", "2020-02-01 00:50")
    )
    event_gap = EventWindow(
        2, "synthetic-gap", "test", TimeWindow("2020-02-01 02:30", "2020-02-01 03:30")
    )
    from anomaliradar.data.pipeline import DatasetConfig

    ds = build_dataset(
        gapped,
        events=(event_frozen, event_gap),
        config=DatasetConfig(mask=MaskConfig(frozen_min_rows=60)),
    )
    summary = mask_summary(ds.grid, ds.masks, ds.native_masks, ds.events)
    events = {row["event_id"]: row for row in summary["events"]}

    assert events[1]["m1_minutes"] == 40
    assert events[1]["m2_minutes"] == 0
    assert events[1]["m1_share"] == pytest.approx(1.0)

    assert events[2]["m1_minutes"] == 0
    assert events[2]["m2_minutes"] >= 30
    assert events[2]["m2_share"] > 0
    assert events[2]["m2_gap_count"] == 1

    # Global shares are reported on the grid and on native rows separately.
    assert summary["global"]["m1_native_rows"] == 60
    assert summary["global"]["grid_m1_rows"] == 60
    assert summary["global"]["m2_gaps"] == 1


def test_mask_summary_event_denominator_is_wall_clock() -> None:
    frame = make_native_frame(minutes=120)
    gapped = drop_minutes(frame, list(range(30, 91)))  # 61-minute gap
    event = EventWindow(
        1, "synthetic", "test", TimeWindow("2020-02-01 00:20", "2020-02-01 01:40")
    )
    ds = build_dataset(gapped, events=(event,))
    row = mask_summary(ds.grid, ds.masks, ds.native_masks, ds.events)["events"][0]
    assert row["window_minutes"] == 80
    assert row["m2_minutes"] == 61  # minutes 00:30..01:30 inside the window
    assert row["m2_share"] == pytest.approx(61 / 80)


def test_mask_config_defaults_match_the_protocol() -> None:
    cfg = MaskConfig()
    assert cfg.frozen_min_rows == 360
    assert cfg.major_gap == pd.Timedelta(hours=1)
    assert cfg.warmup == pd.Timedelta(minutes=30)
