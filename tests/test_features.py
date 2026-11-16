"""Causal context features: lag, run length, transitions, calendar, R1 partition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.features import (
    FEATURE_COLUMNS,
    NON_FEATURE_COLUMNS,
    add_context_features,
    assert_feature_columns_are_causal,
    feature_matrix,
)
from anomaliradar.data.grid import to_minute_grid
from anomaliradar.data.pipeline import build_dataset
from anomaliradar.data.schema import (
    REGIME_CHANNELS,
    SCORE_A_EXCLUDED,
    VALUE_CHANNELS,
    SchemaError,
)
from anomaliradar.data.state import run_length, transitions
from helpers import make_native_frame


@pytest.fixture(scope="module")
def ds():
    return build_dataset(make_native_frame(minutes=240))


def _context_with_states(ds, states: list[str]):
    state = ds.state.copy()
    state["state"] = states
    state["run_length_min"] = run_length(state["state"], state["segment_id"])
    state["is_transition"] = transitions(state["state"], state["segment_id"])
    state["is_other_state"] = state["state"] == "other"
    return add_context_features(ds.grid, state)


def test_state_lag_is_exactly_one_row(ds) -> None:
    context = ds.context
    assert pd.isna(context["state_lag1"].iloc[0])
    for pos in (1, 2, 50, 200):
        assert context["state_lag1"].iloc[pos] == context["state"].iloc[pos - 1]
    assert context["run_length_lag1_min"].iloc[10] == context["run_length_min"].iloc[9]


def test_same_row_state_is_not_in_the_feature_set(ds) -> None:
    assert set(FEATURE_COLUMNS) & set(NON_FEATURE_COLUMNS) == set()
    assert "state" not in FEATURE_COLUMNS
    assert "run_length_min" not in FEATURE_COLUMNS
    columns = feature_matrix(ds.context).columns
    assert list(columns) == list(FEATURE_COLUMNS)


def test_changing_the_same_row_state_does_not_change_row_features(ds) -> None:
    baseline = ds.context
    states = baseline["state"].astype(object).tolist()
    states[10] = "loaded" if states[10] != "loaded" else "off"
    perturbed = _context_with_states(ds, states)

    pd.testing.assert_series_equal(
        baseline.loc[baseline.index[10], list(FEATURE_COLUMNS) + ["context_cell"]],
        perturbed.loc[perturbed.index[10], list(FEATURE_COLUMNS) + ["context_cell"]],
    )
    # The perturbation is real: the next row's lag must change.
    assert perturbed["state_lag1"].iloc[11] != baseline["state_lag1"].iloc[11]


def test_transitions_are_counted_strictly_in_the_past(ds) -> None:
    states = ["off"] * 10 + ["loaded"] * 230
    context = _context_with_states(ds, states)

    assert bool(context["is_transition"].iloc[10]) is True
    assert context["transitions_1h_lag1"].iloc[10] == 0
    assert context["transitions_1h_lag1"].iloc[11] == 1
    assert context["transitions_1h_lag1"].iloc[70] == 1
    assert context["transitions_1h_lag1"].iloc[71] == 0  # left the 60-row window


def test_lag_resets_at_a_segment_boundary(ds) -> None:
    state = ds.state.copy()
    state["state"] = "offloaded"
    state["segment_id"] = pd.Series(
        [0] * 20 + [1] * (len(state) - 20), index=state.index, dtype="Int64"
    )
    context = add_context_features(ds.grid, state)
    assert pd.isna(context["state_lag1"].iloc[20])
    assert context["state_lag1"].iloc[21] == "offloaded"


def test_day_type_and_hour_are_known_in_advance(ds) -> None:
    first = ds.index[0]
    assert ds.context["day_type"].iloc[0] == ("weekend" if first.dayofweek >= 5 else "weekday")
    assert ds.context["hour"].iloc[0] == first.hour

    monday_ds = build_dataset(make_native_frame(start="2020-02-03 00:00", minutes=120))
    assert set(monday_ds.context["day_type"]) == {"weekday"}
    assert monday_ds.context["hour"].iloc[0] == 0
    assert monday_ds.context["hour"].iloc[99] == 1


def test_context_cell_combines_lagged_state_and_day_type(ds) -> None:
    row = ds.context.iloc[1]
    assert row["context_cell"] == f"{row['state_lag1']}|{row['day_type']}"
    assert pd.isna(ds.context["context_cell"].iloc[0])


def test_feature_guard_rejects_same_row_columns() -> None:
    with pytest.raises(SchemaError, match="same-row"):
        assert_feature_columns_are_causal(("state",))
    with pytest.raises(SchemaError, match="same-row"):
        assert_feature_columns_are_causal(("run_length_min",))


def test_anti_circularity_partition() -> None:
    assert set(VALUE_CHANNELS) & set(REGIME_CHANNELS) == set()
    assert set(SCORE_A_EXCLUDED) == set(REGIME_CHANNELS) | {"TP2"}
    assert "TP2" not in VALUE_CHANNELS
    for channel in ("Motor_current", "DV_eletric", "COMP", "MPG", "TP2"):
        assert channel not in VALUE_CHANNELS


def test_grid_aggregation_does_not_create_a_same_row_feature() -> None:
    """A single-minute state change leaves that minute's feature row untouched."""
    frame = make_native_frame(minutes=30)
    base = build_dataset(frame)
    perturbed_frame = frame.copy()
    minute = base.index[10]
    rows = perturbed_frame["timestamp"] == minute
    perturbed_frame.loc[rows, "Motor_current"] = 9.0
    perturbed_frame.loc[rows, "DV_eletric"] = 1
    changed = build_dataset(perturbed_frame)

    pd.testing.assert_series_equal(
        base.context.loc[minute, list(FEATURE_COLUMNS)],
        changed.context.loc[minute, list(FEATURE_COLUMNS)],
    )
