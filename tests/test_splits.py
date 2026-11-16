"""Chronological fold construction, ordering and split materialisation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.labels import FAILURE_EVENTS, label_mask
from anomaliradar.data.pipeline import build_dataset
from anomaliradar.data.splits import (
    FoldSpec,
    SplitError,
    build_folds,
    fold_table,
    materialise_split,
    validate_folds,
)
from anomaliradar.timewindow import TimeWindow
from helpers import make_native_frame, synthetic_events


@pytest.fixture(scope="module")
def long_ds():
    events = synthetic_events("2020-03-10 00:00", "2020-04-10 00:00")
    frame = make_native_frame(minutes=80 * 24 * 60)  # 2020-02-01 → 2020-04-20
    return build_dataset(frame, events=events)


def test_fold_boundaries_match_the_pre_registered_table() -> None:
    folds = build_folds(pd.Timestamp("2020-02-01"), FAILURE_EVENTS)
    assert [f.fold_id for f in folds] == ["1", "2", "3", "4"]

    f1 = folds[0]
    assert f1.train.start == pd.Timestamp("2020-02-01 00:00")
    assert f1.train.end == pd.Timestamp("2020-03-21 00:00")
    assert f1.calibration.start == pd.Timestamp("2020-03-21 00:00")
    assert f1.calibration.end == pd.Timestamp("2020-04-11 00:00")
    assert f1.test.start == pd.Timestamp("2020-04-17 12:00")
    assert f1.test.end == pd.Timestamp("2020-04-20 00:00")

    f4 = folds[3]
    assert f4.calibration.end == pd.Timestamp("2020-07-08 14:30")
    assert (f4.calibration.end - f4.calibration.start) == pd.Timedelta(days=21)


def test_fold_table_is_deterministic() -> None:
    first = fold_table(build_folds(pd.Timestamp("2020-02-01"), FAILURE_EVENTS))
    second = fold_table(build_folds(pd.Timestamp("2020-02-01"), FAILURE_EVENTS))
    pd.testing.assert_frame_equal(first, second)
    assert first["embargo_days"].tolist() == [6.5, 6.5, 6.5, 6.5]


def test_tail_fold_reuses_fold_four() -> None:
    folds = build_folds(pd.Timestamp("2020-02-01"), FAILURE_EVENTS, include_tail=True)
    assert [f.fold_id for f in folds] == ["1", "2", "3", "4", "T"]
    tail, base = folds[-1], folds[3]
    assert tail.train == base.train
    assert tail.calibration == base.calibration
    assert tail.refit is False
    assert tail.inherits_from == "4"
    assert tail.test.start == pd.Timestamp("2020-08-01 00:00")


def test_chronological_ordering_holds_on_materialised_rows(long_ds) -> None:
    index = long_ds.index
    for fold in long_ds.folds():
        split = long_ds.split(fold)
        assert index[split.train].max() < index[split.calibration].min()
        assert index[split.calibration].max() < index[split.test].min()


def test_train_calibration_test_sets_are_disjoint(long_ds) -> None:
    for fold in long_ds.folds():
        split = long_ds.split(fold)  # __post_init__ already checks; repeat explicitly
        assert set(split.train) & set(split.calibration) == set()
        assert set(split.train) & set(split.test) == set()
        assert set(split.calibration) & set(split.test) == set()


def test_masked_rows_never_enter_train_or_calibration(long_ds) -> None:
    for fold in long_ds.folds():
        split = long_ds.split(fold)
        assert not long_ds.masks.combined[split.train].any()
        assert not long_ds.masks.combined[split.calibration].any()
        assert long_ds.usable[split.train].all()
        assert long_ds.usable[split.calibration].all()


def test_labels_never_enter_train_or_calibration(long_ds) -> None:
    index = long_ds.index
    # Fold 2's training window contains synthetic event 1.
    fold2 = long_ds.folds()[1]
    split = long_ds.split(fold2)
    assert split.stats["train"]["excluded_label"] > 0
    assert not label_mask(index[split.train], long_ds.events).any()
    assert not label_mask(index[split.calibration], long_ds.events).any()


def test_calibration_never_contains_a_labelled_row() -> None:
    events = synthetic_events("2020-02-20 00:00")
    frame = make_native_frame(minutes=40 * 24 * 60)
    ds = build_dataset(frame, events=events)

    fold = FoldSpec(
        fold_id="synthetic",
        event_id=1,
        train=TimeWindow("2020-02-01", "2020-02-10"),
        calibration=TimeWindow("2020-02-10", "2020-03-01"),
        test=TimeWindow("2020-03-10", "2020-03-11"),
        detection=TimeWindow("2020-03-10", "2020-03-10 02:00"),
        event_start=pd.Timestamp("2020-03-17"),
    )
    split = ds.split(fold)
    assert split.stats["calibration"]["excluded_label"] == 120  # the 2-hour event
    assert not label_mask(ds.index[split.calibration], events).any()


def test_test_split_retains_labelled_rows(long_ds) -> None:
    split = long_ds.split(long_ds.folds()[0])
    labelled = label_mask(long_ds.index[split.test], long_ds.events)
    assert labelled.sum() == 120  # the synthetic event is 2 h long
    assert split.detection.size > 0
    assert set(split.detection) <= set(split.test)


def test_detection_window_is_the_creditable_subset(long_ds) -> None:
    fold = long_ds.folds()[0]
    split = long_ds.split(fold)
    detection_times = long_ds.index[split.detection]
    assert detection_times.min() >= fold.detection.start
    assert detection_times.max() < fold.detection.end


def test_overlapping_fold_is_rejected() -> None:
    folds = build_folds(pd.Timestamp("2020-02-01"), FAILURE_EVENTS)
    bad = replace(
        folds[0],
        test=TimeWindow(
            folds[0].calibration.start + pd.Timedelta(hours=1),
            folds[0].test.end,
        ),
    )
    with pytest.raises(SplitError, match="overlaps test"):
        validate_folds([bad])


def test_short_embargo_is_rejected() -> None:
    folds = build_folds(pd.Timestamp("2020-02-01"), FAILURE_EVENTS)
    bad = replace(
        folds[0],
        test=TimeWindow(
            folds[0].calibration.end + pd.Timedelta(days=1),
            folds[0].test.end,
        ),
    )
    with pytest.raises(SplitError, match="embargo"):
        validate_folds([bad])


def test_test_window_contains_the_event() -> None:
    for fold, event in zip(build_folds(pd.Timestamp("2020-02-01"), FAILURE_EVENTS), FAILURE_EVENTS):
        assert fold.test.start <= event.start
        assert fold.test.end >= event.end
        assert fold.detection.start == event.start - pd.Timedelta(hours=2)


def test_split_to_dict_has_counts(long_ds) -> None:
    split = long_ds.split(long_ds.folds()[0])
    payload = split.to_dict()
    assert payload["fold_id"] == "1"
    assert payload["train_rows"] == len(split.train)
    assert payload["calibration_rows"] == len(split.calibration)
    assert payload["calibration"]["kept"] == len(split.calibration)
