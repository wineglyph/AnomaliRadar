"""Real-data audit: recomputed masks, event overlap, fold boundaries, splits.

Skipped automatically when the CSV is absent, so the suite never needs the
download. Every number here is recomputed by the data layer, not copied from a
report; the assertions are the Day 2 stop conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.labels import FAILURE_EVENTS, label_mask
from anomaliradar.data.splits import validate_folds


def test_real_shape_and_grid(real_dataset) -> None:
    ds = real_dataset
    assert ds.native_rows == 1_516_948
    assert len(ds.grid) == 306_960
    assert ds.grid.observed_rows == 252_720
    assert ds.grid.has_value_rows == 252_731
    assert ds.grid.frame["segment_id"].nunique() == 321
    assert len(ds.grid.breaks) == 320
    assert str(ds.index.min()) == "2020-02-01 00:00:00"
    assert str(ds.index.max()) == "2020-09-01 03:59:00"


def test_real_m1_frozen_blocks(real_dataset) -> None:
    global_stats = real_dataset.mask_summary()["global"]
    assert global_stats["m1_blocks"] == 7
    assert global_stats["m1_native_rows"] == 50_425
    assert global_stats["m1_native_share"] == pytest.approx(0.03324, abs=5e-4)
    assert global_stats["m1_max_hours"] == pytest.approx(51.4306, abs=1e-3)
    assert global_stats["grid_m1_rows"] == 10_155


def test_real_m2_gaps(real_dataset) -> None:
    global_stats = real_dataset.mask_summary()["global"]
    assert global_stats["m2_gaps"] == 160
    assert global_stats["m2_gap_hours"] == pytest.approx(868.513, abs=0.01)
    assert global_stats["grid_m2_rows"] == 52_108


def test_real_m3_warmup(real_dataset) -> None:
    global_stats = real_dataset.mask_summary()["global"]
    assert global_stats["grid_m3_after_frozen_rows"] == 7 * 30
    assert global_stats["grid_m3_after_gap_rows"] == 160 * 30
    assert global_stats["grid_m3_rows"] == 7 * 30 + 160 * 30


def test_real_mask_shares_are_reported_on_two_denominators(real_dataset) -> None:
    global_stats = real_dataset.mask_summary()["global"]
    assert global_stats["grid_m1_share"] == pytest.approx(0.03308, abs=5e-4)
    assert global_stats["grid_combined_share"] == pytest.approx(0.21755, abs=5e-4)
    assert global_stats["grid_unmasked_observed_share"] == pytest.approx(0.77593, abs=5e-4)
    assert global_stats["grid_unmasked_observed_rows"] == 238_179


def test_real_event_masks_are_reported_separately(real_dataset) -> None:
    events = {row["event_id"]: row for row in real_dataset.mask_summary()["events"]}

    # Event 1: a frozen block tail plus its 30-minute warm-up.
    assert events[1]["m1_minutes"] == 19
    assert events[1]["m2_minutes"] == 0
    assert events[1]["m3_minutes"] == 30
    assert events[1]["m2_gap_count"] == 0

    # Events 2 and 4: no mask overlap at all.
    assert events[2]["combined_minutes"] == 0
    assert events[4]["combined_minutes"] == 0

    # Event 3: two gaps touch the window; one of them starts inside the window.
    assert events[3]["m1_minutes"] == 0
    assert events[3]["m2_minutes"] == 238
    assert events[3]["m2_gap_count"] == 2
    assert events[3]["m3_minutes"] == 30


def test_event3_mask_ratio_is_under_the_ten_percent_limit(real_dataset) -> None:
    """Day 2 stop condition F6. Report this number verbatim if it ever fails."""
    row = {
        r["event_id"]: r
        for r in real_dataset.mask_summary()["events"]
    }[3]
    assert row["m2_share"] == pytest.approx(0.07553, abs=1e-4)
    assert row["combined_share"] < 0.10, (
        f"event 3 masked share {row['combined_share']:.3%} exceeds the 10% limit"
    )
    assert row["combined_share"] == pytest.approx(0.08505, abs=1e-4)


def test_real_state_distribution(real_dataset) -> None:
    summary = real_dataset.state_summary()
    assert summary["run_length_median_min"]["loaded"] <= 5
    assert summary["run_length_median_min"]["offloaded"] == 7
    assert summary["run_length_median_min"]["off"] == 14
    assert summary["confirmed_share"]["loaded"] == pytest.approx(0.05516, abs=5e-4)
    assert summary["other_state_share"] < 0.01


def test_real_folds_match_the_protocol(real_dataset) -> None:
    folds = real_dataset.folds()
    assert [f.fold_id for f in folds] == ["1", "2", "3", "4", "T"]
    validate_folds(folds)

    assert folds[0].test.start == pd.Timestamp("2020-04-17 12:00")
    assert folds[0].calibration.end == pd.Timestamp("2020-04-11 00:00")
    assert folds[3].calibration.end == pd.Timestamp("2020-07-08 14:30")
    assert folds[4].refit is False
    assert folds[4].test.start == pd.Timestamp("2020-08-01 00:00")


def test_real_split_ordering_and_disjointness(real_dataset) -> None:
    index = real_dataset.index
    for fold in real_dataset.folds():
        split = real_dataset.split(fold)
        if len(split.train) and len(split.calibration):
            assert index[split.train].max() < index[split.calibration].min()
        if len(split.calibration) and len(split.test):
            assert index[split.calibration].max() < index[split.test].min()
        assert set(split.train) & set(split.calibration) == set()
        assert set(split.train) & set(split.test) == set()
        assert set(split.calibration) & set(split.test) == set()


def test_real_split_excludes_masks_and_labels_from_fit(real_dataset) -> None:
    ds = real_dataset
    for fold in ds.folds():
        split = ds.split(fold)
        assert not ds.masks.combined[split.train].any()
        assert not ds.masks.combined[split.calibration].any()
        assert not label_mask(ds.index[split.train], ds.events).any()
        assert not label_mask(ds.index[split.calibration], ds.events).any()


def test_real_training_removes_earlier_events(real_dataset) -> None:
    ds = real_dataset
    splits = {fold.fold_id: ds.split(fold) for fold in ds.folds()}
    assert splits["1"].stats["train"]["excluded_label"] == 0
    assert splits["2"].stats["train"]["excluded_label"] == 1440   # event 1
    assert splits["3"].stats["train"]["excluded_label"] == 1440   # event 1
    assert splits["4"].stats["train"]["excluded_label"] == 4982   # events 1-3
    for fold_id in ("1", "2", "3", "4", "T"):
        assert splits[fold_id].stats["calibration"]["excluded_label"] == 0


def test_real_test_windows_retain_their_events(real_dataset) -> None:
    ds = real_dataset
    for fold in ds.folds()[:4]:
        split = ds.split(fold)
        event = FAILURE_EVENTS[int(fold.fold_id) - 1]
        labelled = label_mask(ds.index[split.test], ds.events)
        assert labelled.sum() > 0

        # The test *window* must span the event even when masks remove early rows.
        window_times = ds.index[split.test_window]
        assert window_times.min() <= event.start
        assert window_times.max() >= event.end - pd.Timedelta(minutes=1)

        # If the first kept row is after the event start, masks must explain it.
        kept_times = ds.index[split.test]
        if kept_times.min() > event.start:
            early = (ds.index >= event.start) & (ds.index < kept_times.min())
            assert ds.masks.combined[early].all()


def test_real_calibration_cells_are_the_six_declared_cells(real_dataset) -> None:
    ds = real_dataset
    expected = {
        f"{state}|{day}"
        for state in ("off", "offloaded", "loaded")
        for day in ("weekday", "weekend")
    }
    for fold in ds.folds():
        split = ds.split(fold)
        cells = set(ds.context["context_cell"].iloc[split.calibration].dropna().unique())
        assert cells == expected, (fold.fold_id, cells)


def test_fold1_calibration_cells_are_dense(real_dataset) -> None:
    ds = real_dataset
    split = ds.split(ds.folds()[0])
    counts = ds.context["context_cell"].iloc[split.calibration].value_counts()
    assert counts.min() >= 200
    assert counts["loaded|weekday"] == 701


def test_fold4_loaded_cells_are_a_known_calibration_risk(real_dataset) -> None:
    """Known Day 3 risk, not a Day 2 failure.

    Fold 4's calibration window (06-17 → 07-08) contains only 77 loaded minutes.
    Both fine cells are below the 50-row refusal threshold, so back-off must use
    the state-only cell (77 rows) and calibration for ``loaded`` is weak.
    """
    ds = real_dataset
    split = ds.split(ds.folds()[3])
    counts = ds.context["context_cell"].iloc[split.calibration].value_counts()
    assert counts["loaded|weekday"] == 48
    assert counts["loaded|weekend"] == 29
    assert counts.get("loaded|weekday", 0) + counts.get("loaded|weekend", 0) == 77
    # The pooled state cell still clears the 50-row refusal floor, but not 200.
    assert 50 <= 77 < 200


def test_real_tail_split(real_dataset) -> None:
    ds = real_dataset
    tail = ds.folds()[-1]
    split = ds.split(tail)
    assert len(split.test) == 35_940
    assert len(split.detection) == 0
    assert ds.index[split.test].min() == pd.Timestamp("2020-08-01 00:00")
    assert ds.index[split.test].max() == pd.Timestamp("2020-09-01 03:59:00")
    assert len(split.train) == 150_307
    assert len(split.calibration) == 20_488
