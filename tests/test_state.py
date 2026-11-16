"""Operating-state encoding, hysteresis and run length."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.grid import to_minute_grid
from anomaliradar.data.state import (
    StateConfig,
    attach_state,
    confirm_state,
    raw_state,
    run_length,
    state_run_table,
    transitions,
)
from helpers import make_native_frame


def _confirmed(raw: list[str], segments: list[int] | None = None, dwell: int = 3) -> list:
    if segments is None:
        segments = [0] * len(raw)
    state = confirm_state(
        pd.Series(raw, dtype=object),
        pd.Series(segments, dtype="Int64"),
        StateConfig(min_dwell=dwell),
    )
    return state.tolist()


def test_raw_state_mapping() -> None:
    frame = pd.DataFrame(
        {
            "Motor_current": [0.0, 3.5, 6.5, 7.0, 9.0, 1.0, 6.5],
            "DV_eletric":    [0,   0,   0,   1,   0,   0,   1],
        }
    )
    assert raw_state(frame).tolist() == [
        "off",
        "offloaded",
        "other",
        "loaded",
        "loaded",   # start is absorbed into loaded
        "other",
        "loaded",
    ]


def test_missing_sensor_value_is_not_other() -> None:
    frame = pd.DataFrame({"Motor_current": [np.nan], "DV_eletric": [1.0]})
    assert pd.isna(raw_state(frame).iloc[0])


def test_hysteresis_ignores_a_two_minute_blip() -> None:
    raw = ["off"] * 4 + ["loaded"] * 2 + ["off"] * 4
    assert _confirmed(raw) == ["off"] * 10


def test_hysteresis_confirms_three_minute_dwell() -> None:
    raw = ["off", "off", "loaded", "loaded", "loaded", "off"]
    # The third consecutive candidate confirms; the following off is still pending.
    assert _confirmed(raw) == ["off", "off", "off", "off", "loaded", "loaded"]


def test_hysteresis_delay_is_at_most_two_steps() -> None:
    raw = ["off"] * 3 + ["loaded"] * 5
    confirmed = _confirmed(raw)
    first_loaded = confirmed.index("loaded")
    # Raw transition at row 3; third consecutive candidate is row 5, i.e. L-1.
    assert first_loaded == 5


def test_hysteresis_resets_at_segment_break() -> None:
    raw = ["loaded"] * 4 + ["offloaded", "offloaded", "offloaded"]
    segments = [0, 0, 0, 0, 1, 1, 1]
    confirmed = _confirmed(raw, segments)
    # No state is carried across the break: the new segment starts from its raw state.
    assert confirmed == ["loaded"] * 4 + ["offloaded"] * 3


def test_hysteresis_does_not_bridge_missing_rows() -> None:
    raw = pd.Series(["loaded", "loaded", np.nan, "loaded", "loaded"], dtype=object)
    segments = pd.Series([0, 0, 0, 0, 0], dtype="Int64")
    confirmed = confirm_state(raw, segments)
    assert confirmed.iloc[2] is np.nan or pd.isna(confirmed.iloc[2])
    # The run restarts after the missing row.
    assert confirmed.tolist()[:2] == ["loaded", "loaded"]


def test_run_length_counts_and_resets() -> None:
    state = pd.Series(["off", "off", "loaded", "loaded", "loaded"])
    segment = pd.Series([0, 0, 0, 0, 0], dtype="Int64")
    assert run_length(state, segment).tolist() == [1, 2, 1, 2, 3]


def test_run_length_resets_at_segment_boundary() -> None:
    state = pd.Series(["off", "off", "loaded", "loaded", "loaded"])
    segment = pd.Series([0, 0, 0, 1, 1], dtype="Int64")
    assert run_length(state, segment).tolist() == [1, 2, 1, 1, 2]


def test_run_length_is_nan_without_state() -> None:
    state = pd.Series(["off", np.nan, "off"])
    segment = pd.Series([0, 0, 0], dtype="Int64")
    lengths = run_length(state, segment)
    assert lengths.iloc[0] == 1
    assert pd.isna(lengths.iloc[1])
    assert lengths.iloc[2] == 1


def test_transitions_are_within_segment_and_not_at_segment_start() -> None:
    state = pd.Series(["off", "off", "loaded", "loaded", "off"])
    segment = pd.Series([0, 0, 0, 0, 1], dtype="Int64")
    flags = transitions(state, segment)
    assert flags.tolist() == [False, False, True, False, False]


def test_attach_state_end_to_end() -> None:
    raw = ["off"] * 20 + ["loaded"] * 5 + ["off"] * 20
    frame = make_native_frame(minutes=len(raw), states=raw)
    grid = to_minute_grid(frame)
    state = attach_state(grid)

    assert set(state["state"].dropna().unique()) <= {"off", "loaded"}
    # The 5-minute loaded span survives hysteresis.
    assert (state["state"] == "loaded").sum() == 5
    assert state["run_length_min"].max() == 22  # the leading off run

    runs = state_run_table(state)
    assert runs["minutes"].tolist() == [22, 5, 18]


def test_state_config_rejects_zero_dwell() -> None:
    with pytest.raises(ValueError, match="min_dwell"):
        StateConfig(min_dwell=0)


def test_hysteresis_dwell_is_independent_of_alarm_debounce() -> None:
    """The data layer's dwell is 3 minutes; alarm debounce is a 30-minute policy."""
    assert StateConfig().min_dwell == 3
    # 30 is a legal dwell value in principle. The point is that the data layer
    # never uses it: state definition and alarm debounce are separate concerns.
    assert StateConfig(min_dwell=30).min_dwell == 30
