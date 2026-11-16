"""Operating state, 3-minute hysteresis, and run length.

Raw state encoding (verified Day 0, dataset PDF + data)::

    off        Motor_current < 0.5
    loaded     Motor_current >= 8, or 5 <= Motor_current < 8 with DV_eletric == 1
    offloaded  2 <= Motor_current < 5 with DV_eletric == 0
    other      everything else (transitions / ambiguous rows)

``start`` (``Motor_current >= 8``) is absorbed into ``loaded``: it occurs in a
few dozen rows, is too rare to calibrate, and physically precedes loaded
operation.

The raw state fragments on single-row blips (event 3 splits into 143 raw loaded
episodes). A candidate state must therefore persist for ``min_dwell = 3``
consecutive grid minutes before it is confirmed; meanwhile the previous
confirmed state is retained. This is **not** alarm debounce: hysteresis is a
data-layer state definition, alarm debounce is a 30-minute alarm-layer policy
and is deliberately not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaliradar.data.grid import MinuteGrid

STATE_LABELS: tuple[str, ...] = ("off", "offloaded", "loaded", "other")


@dataclass(frozen=True, slots=True)
class StateConfig:
    min_dwell: int = 3          # minutes (grid rows)
    off_below: float = 0.5
    start_at: float = 8.0
    loaded_low: float = 5.0
    loaded_high: float = 8.0
    offloaded_low: float = 2.0
    offloaded_high: float = 5.0

    def __post_init__(self) -> None:
        if self.min_dwell < 1:
            raise ValueError("min_dwell must be >= 1")


def raw_state(frame: pd.DataFrame, config: StateConfig | None = None) -> pd.Series:
    """Map ``Motor_current``/``DV_eletric`` to a 4-label state per row.

    ``start`` is absorbed into ``loaded``. Rows without values are NaN, never
    ``other``, so that missing data cannot masquerade as a transition.
    """
    cfg = config or StateConfig()
    current = frame["Motor_current"].to_numpy(dtype="float64")
    dv = frame["DV_eletric"].to_numpy(dtype="float64")

    state = np.select(
        [
            current < cfg.off_below,
            current >= cfg.start_at,
            (current >= cfg.loaded_low) & (current < cfg.loaded_high) & (dv == 1),
            (current >= cfg.offloaded_low) & (current < cfg.offloaded_high) & (dv == 0),
        ],
        ["off", "loaded", "loaded", "offloaded"],
        default="other",
    ).astype(object)

    no_value = np.isnan(current) | np.isnan(dv)
    state[no_value] = np.nan
    return pd.Series(state, index=frame.index, name="raw_state")


def confirm_state(
    raw: pd.Series,
    segment_id: pd.Series,
    config: StateConfig | None = None,
) -> pd.Series:
    """Apply minimum-dwell hysteresis independently inside every segment.

    A candidate is confirmed on its ``min_dwell``-th consecutive sample, so the
    confirmed transition lags the raw one by at most ``min_dwell - 1`` minutes.
    A break in ``segment_id`` restarts the confirmation: no state is carried
    across a data gap.
    """
    cfg = config or StateConfig()
    out = pd.Series(np.nan, index=raw.index, dtype=object, name="state")

    raw_values = raw.to_numpy(dtype=object)
    segments = pd.Series(segment_id).to_numpy()

    # Contiguous groups of equal segment id (segments are already monotone).
    if len(raw_values) == 0:
        return out
    boundaries = np.flatnonzero(
        np.r_[True, ~pd.Series(segments).eq(pd.Series(segments).shift()).to_numpy()[1:]]
    )
    boundaries = np.append(boundaries, len(raw_values))

    confirmed = np.full(len(raw_values), np.nan, dtype=object)
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        current: object = None
        pending: object = None
        pending_count = 0
        for pos in range(int(b0), int(b1)):
            candidate = raw_values[pos]
            if candidate is None or (isinstance(candidate, float) and np.isnan(candidate)):
                confirmed[pos] = np.nan
                current = None
                pending = None
                pending_count = 0
                continue
            if current is None:
                current = candidate
                confirmed[pos] = current
                pending = None
                pending_count = 0
                continue
            if candidate == current:
                confirmed[pos] = current
                pending = None
                pending_count = 0
                continue

            if pending is not None and candidate == pending:
                pending_count += 1
            else:
                pending = candidate
                pending_count = 1

            if pending_count >= cfg.min_dwell:
                current = candidate
                pending = None
                pending_count = 0
            confirmed[pos] = current

    out[:] = confirmed
    return out


def run_length(state: pd.Series, segment_id: pd.Series) -> pd.Series:
    """Consecutive minutes in the current confirmed state, reset at breaks.

    The first row of a run has length 1. Missing-state rows are NaN.
    """
    state = pd.Series(state)
    segment = pd.Series(segment_id, index=state.index)

    new_run = (
        state.ne(state.shift())
        | segment.ne(segment.shift())
        | state.isna()
        | state.shift().isna()
    )
    run_id = new_run.cumsum()
    lengths = state.groupby(run_id).cumcount() + 1
    lengths = lengths.astype("float64")
    lengths[state.isna()] = np.nan
    return lengths.rename("run_length_min")


def transitions(state: pd.Series, segment_id: pd.Series) -> pd.Series:
    """True at the first row of a new confirmed state, inside a segment."""
    state = pd.Series(state)
    segment = pd.Series(segment_id, index=state.index)
    changed = (
        state.ne(state.shift())
        & state.notna()
        & state.shift().notna()
        & segment.eq(segment.shift())
    )
    return changed.astype(bool).rename("is_transition")


def attach_state(grid: MinuteGrid, config: StateConfig | None = None) -> pd.DataFrame:
    """Build the state frame for a grid: raw, confirmed, run length, transitions."""
    cfg = config or StateConfig()
    segment = grid.frame["segment_id"]
    raw = raw_state(grid.frame, cfg)
    confirmed = confirm_state(raw, segment, cfg)
    frame = pd.DataFrame(
        {
            "segment_id": segment,
            "raw_state": raw,
            "state": confirmed,
            "run_length_min": run_length(confirmed, segment),
            "is_transition": transitions(confirmed, segment),
        },
        index=grid.index,
    )
    frame["is_other_state"] = (frame["state"] == "other").astype(bool)
    return frame


def state_run_table(state_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise confirmed-state episodes (duration in minutes)."""
    state = state_frame["state"]
    segment = state_frame["segment_id"]
    new_run = state.ne(state.shift()) | segment.ne(segment.shift()) | state.isna()
    run_id = new_run.cumsum()

    valid = state.notna()
    sub = state_frame.loc[valid, ["state"]].copy()
    sub["run_id"] = run_id.loc[valid].to_numpy()
    table = (
        sub.groupby(["run_id", "state"], sort=False)
        .size()
        .rename("minutes")
        .reset_index()[["state", "minutes"]]
    )
    return table
