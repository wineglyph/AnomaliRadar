"""1-minute grid: aggregation, missingness, forward-fill limits and breaks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.grid import to_minute_grid
from anomaliradar.data.pipeline import build_dataset
from helpers import drop_minutes, make_native_frame


def test_minute_mean_aggregation() -> None:
    frame = make_native_frame(minutes=12, freq="10s")
    frame["Reservoirs"] = np.arange(12, dtype=float)
    grid = to_minute_grid(frame)

    assert len(grid) == 2
    assert grid.frame["n_obs"].tolist() == [6, 6]
    assert grid.frame["Reservoirs"].iloc[0] == pytest.approx(2.5)
    assert grid.frame["Reservoirs"].iloc[1] == pytest.approx(8.5)


def test_digital_channel_uses_last_sample_not_mean() -> None:
    frame = make_native_frame(minutes=6, freq="10s")
    frame["DV_eletric"] = [0, 0, 0, 0, 1, 1]
    grid = to_minute_grid(frame)
    # The minute contains both 0 and 1; mean would invent 1/3. Last is 1.
    assert grid.frame["DV_eletric"].iloc[0] == 1


def test_single_missing_minute_is_forward_filled_and_flagged() -> None:
    frame = make_native_frame(minutes=6)
    frame["Reservoirs"] = np.arange(6, dtype=float) + 10.0
    gapped = drop_minutes(frame, [2])
    grid = to_minute_grid(gapped)

    assert len(grid) == 6
    row = grid.frame.iloc[2]
    assert bool(row["observed"]) is False
    assert bool(row["ffilled"]) is True
    # Forward fill copies minute 1, it does not interpolate or write zero.
    assert row["Reservoirs"] == pytest.approx(11.0)
    assert row["Reservoirs"] != 0
    # A 2-minute timestamp delta is not a break.
    assert len(grid.breaks) == 0
    assert grid.frame["segment_id"].nunique() == 1


def test_two_missing_minutes_are_not_interpolated() -> None:
    frame = make_native_frame(minutes=6)
    frame["Reservoirs"] = np.arange(6, dtype=float) + 10.0
    gapped = drop_minutes(frame, [2, 3])
    grid = to_minute_grid(gapped)

    assert len(grid) == 6
    for pos in (2, 3):
        assert pd.isna(grid.frame["Reservoirs"].iloc[pos])
        assert bool(grid.frame["observed"].iloc[pos]) is False
        assert bool(grid.frame["ffilled"].iloc[pos]) is False
        assert bool(grid.frame["has_value"].iloc[pos]) is False

    assert len(grid.breaks) == 1
    brk = grid.breaks.iloc[0]
    assert brk["delta_minutes"] == 3
    assert brk["missing_minutes"] == 2


def test_segment_id_increments_only_at_a_break() -> None:
    frame = make_native_frame(minutes=6)
    grid = to_minute_grid(drop_minutes(frame, [2, 3]))
    assert grid.frame["segment_id"].tolist() == [0, 0, 0, 0, 1, 1]


def test_grid_is_contiguous_minutes() -> None:
    grid = to_minute_grid(make_native_frame(minutes=10))
    deltas = grid.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(minutes=1)


def test_grid_is_deterministic() -> None:
    frame = make_native_frame(minutes=20, seed=7)
    first = to_minute_grid(frame)
    second = to_minute_grid(frame.copy())
    pd.testing.assert_frame_equal(first.frame, second.frame)
    pd.testing.assert_frame_equal(first.breaks, second.breaks)


def test_forward_fill_parameter_changes_only_the_limit() -> None:
    frame = make_native_frame(minutes=6)
    gapped = drop_minutes(frame, [2, 3])
    strict = to_minute_grid(gapped, max_ffill_minutes=2)
    loose = to_minute_grid(gapped, max_ffill_minutes=4)

    assert strict.frame["has_value"].sum() == 4
    assert loose.frame["has_value"].sum() == 6
    # Even when filled, the values are copied forward, never interpolated.
    assert loose.frame["Reservoirs"].iloc[2] == pytest.approx(
        loose.frame["Reservoirs"].iloc[1]
    )


def test_state_and_run_length_reset_after_a_gap() -> None:
    frame = make_native_frame(minutes=8, states=["loaded"] * 8)
    gapped = drop_minutes(frame, [2, 3])
    ds = build_dataset(gapped)

    # First row after the break starts a fresh run.
    assert ds.state["state"].iloc[4] == "loaded"
    assert ds.state["run_length_min"].iloc[4] == 1
    # Missing rows carry no state at all.
    assert ds.state["state"].iloc[2:4].isna().all()


def test_grid_rejects_unsorted_input() -> None:
    frame = make_native_frame(minutes=5)
    broken = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    with pytest.raises(Exception, match="sorted ascending"):
        to_minute_grid(broken)
