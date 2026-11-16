"""Leakage invariants L3–L7 and L9, exercised end-to-end.

L1/L2 live in ``test_splits.py``; L5 partly there too. This module focuses on
future poison, same-row regime, mask/label exclusion, and gap resets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.features import FEATURE_COLUMNS, feature_matrix
from anomaliradar.data.labels import label_mask
from anomaliradar.data.pipeline import build_dataset
from anomaliradar.data.schema import ANALOG, DIGITAL
from helpers import drop_minutes, make_native_frame, synthetic_events


def test_poisoning_the_future_does_not_change_past_features() -> None:
    """L3: no feature at t may read a row with timestamp > t."""
    frame = make_native_frame(minutes=40 * 24 * 60)
    base = build_dataset(frame)
    cutoff = pd.Timestamp("2020-02-20 12:00")

    poisoned = frame.copy()
    future = poisoned["timestamp"] > cutoff
    for col in ANALOG:
        poisoned.loc[future, col] = poisoned.loc[future, col] + 0.5
    for col in DIGITAL:
        poisoned.loc[future, col] = 1 - poisoned.loc[future, col]
    changed = build_dataset(poisoned, validate=False)

    past = base.index <= cutoff
    pd.testing.assert_frame_equal(
        base.context.loc[past, list(FEATURE_COLUMNS)],
        changed.context.loc[past, list(FEATURE_COLUMNS)],
    )
    # Sanity: the poison did take effect in the future.
    assert changed.grid.frame.loc[cutoff + pd.Timedelta(hours=1), "Reservoirs"] != pytest.approx(
        base.grid.frame.loc[cutoff + pd.Timedelta(hours=1), "Reservoirs"]
    )


def test_future_poison_does_not_change_past_masks_or_state() -> None:
    frame = make_native_frame(minutes=10 * 24 * 60)
    base = build_dataset(frame)
    cutoff = pd.Timestamp("2020-02-05 12:00")

    poisoned = frame.copy()
    future = poisoned["timestamp"] > cutoff
    poisoned.loc[future, "Motor_current"] = 9.0
    poisoned.loc[future, "DV_eletric"] = 1
    changed = build_dataset(poisoned, validate=False)

    past = base.index <= cutoff
    assert np.array_equal(base.masks.combined[past], changed.masks.combined[past])
    assert base.state.loc[past, "state"].equals(changed.state.loc[past, "state"])


def test_regime_context_is_always_one_row_lagged() -> None:
    """L7: context uses s_{t-1}; same-row s_t is present but never a feature."""
    states = (["off"] * 30 + ["loaded"] * 20 + ["offloaded"] * 40 + ["off"] * 30)
    ds = build_dataset(make_native_frame(minutes=len(states), states=states))
    for pos in range(1, len(ds.index)):
        assert ds.context["state_lag1"].iloc[pos] == ds.context["state"].iloc[pos - 1]
    assert pd.isna(ds.context["state_lag1"].iloc[0])
    assert "state" not in feature_matrix(ds.context).columns


def test_feature_matrix_contains_no_diagnostic_or_label_columns() -> None:
    ds = build_dataset(make_native_frame(minutes=300))
    columns = set(feature_matrix(ds.context).columns)
    forbidden = {
        "state",
        "raw_state",
        "run_length_min",
        "is_transition",
        "is_other_state",
        "m1_frozen",
        "m2_gap",
        "label",
        "event_id",
    }
    assert columns.isdisjoint(forbidden)


def test_masks_do_not_depend_on_labels() -> None:
    frame = make_native_frame(minutes=400)
    events_a = synthetic_events("2020-02-01 02:00")
    events_b = synthetic_events("2020-02-01 05:00")

    ds_a = build_dataset(frame, events=events_a)
    ds_b = build_dataset(frame, events=events_b)
    assert ds_a.masks.signature() == ds_b.masks.signature()
    assert np.array_equal(ds_a.masks.combined, ds_b.masks.combined)
    assert ds_a.fingerprint() == ds_b.fingerprint()


def test_masked_and_labelled_rows_are_absent_from_fit_sets() -> None:
    events = synthetic_events("2020-03-10 00:00", "2020-04-10 00:00")
    frame = make_native_frame(minutes=80 * 24 * 60)
    ds = build_dataset(frame, events=events)

    for fold in ds.folds():
        split = ds.split(fold)
        assert not ds.masks.combined[split.train].any()
        assert not ds.masks.combined[split.calibration].any()
        assert not label_mask(ds.index[split.train], events).any()
        assert not label_mask(ds.index[split.calibration], events).any()


def test_no_interpolation_across_a_long_gap() -> None:
    frame = make_native_frame(minutes=20)
    frame["Reservoirs"] = 8.0 + 0.1 * np.arange(20, dtype=float)
    ds = build_dataset(drop_minutes(frame, [5, 6, 7]))

    for minute in (5, 6, 7):
        assert pd.isna(ds.grid.frame["Reservoirs"].iloc[minute])
    assert len(ds.grid.breaks) == 1
    # Values on either side are the real observations, not a straight line.
    assert ds.grid.frame["Reservoirs"].iloc[4] == pytest.approx(8.4)
    assert ds.grid.frame["Reservoirs"].iloc[8] == pytest.approx(8.8)


def test_gap_resets_run_length_and_transition_state() -> None:
    frame = make_native_frame(minutes=30, states=["loaded"] * 30)
    ds = build_dataset(drop_minutes(frame, [10, 11, 12]))

    first_after = 13
    assert ds.state["state"].iloc[first_after] == "loaded"
    assert ds.state["run_length_min"].iloc[first_after] == 1
    assert bool(ds.state["is_transition"].iloc[first_after]) is False
    # The lag feature at the first row after the break has no context.
    assert pd.isna(ds.context["state_lag1"].iloc[first_after])


def test_short_gap_is_filled_but_flagged_not_silently_interpolated() -> None:
    frame = make_native_frame(minutes=10)
    frame["Reservoirs"] = 8.0 + 0.1 * np.arange(10, dtype=float)
    ds = build_dataset(drop_minutes(frame, [5]))

    assert bool(ds.grid.frame["ffilled"].iloc[5]) is True
    assert bool(ds.grid.frame["observed"].iloc[5]) is False
    assert ds.grid.frame["Reservoirs"].iloc[5] == pytest.approx(8.4)  # copied from minute 4
    assert ds.grid.frame["Reservoirs"].iloc[5] != pytest.approx(8.5)  # not a mean of neighbours
