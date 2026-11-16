"""Causal context features.

The only context this project uses as a *feature* is the **previous** confirmed
state and its consequences. Same-row state is retained in the frame for
diagnostics and for the M4 calibration filter, but it is not in
``FEATURE_COLUMNS`` and no model may read it (rules R1/R2, invariant L7).

``state_lag1`` at row ``t`` is the confirmed state at ``t-1`` *within the same
segment*. At a segment start it is NaN: context never crosses a data gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaliradar.data.grid import MinuteGrid
from anomaliradar.data.schema import SchemaError

#: Columns a model is allowed to consume as context.
FEATURE_COLUMNS: tuple[str, ...] = (
    "state_lag1",
    "run_length_lag1_min",
    "transitions_1h_lag1",
    "day_type",
    "hour",
)

#: Same-row columns that must never be features (kept for diagnostics only).
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "raw_state",
    "state",
    "run_length_min",
    "is_transition",
    "is_other_state",
)

CONTEXT_CELL_COLUMN = "context_cell"


def _shift_within_segment(values: pd.Series, segment: pd.Series, periods: int = 1) -> pd.Series:
    grouped = values.groupby(segment, sort=False, dropna=False)
    return grouped.shift(periods)


def _rolling_sum_within_segment(values: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    rolled = (
        values.groupby(segment, sort=False, dropna=False)
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    return rolled


def add_context_features(grid: MinuteGrid, state_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach lagged state, lagged run length, past-hour transitions and calendar.

    Parameters
    ----------
    grid:
        The 1-minute grid, used only for its index and segment ids.
    state_frame:
        Output of :func:`anomaliradar.data.state.attach_state`.

    Returns
    -------
    pandas.DataFrame
        One row per grid minute. ``FEATURE_COLUMNS`` are strictly causal;
        ``NON_FEATURE_COLUMNS`` are present but explicitly forbidden.
    """
    segment = state_frame["segment_id"].astype("Int64")
    index = grid.index

    state = state_frame["state"]
    state_lag1 = _shift_within_segment(state, segment, 1)
    run_lag1 = _shift_within_segment(state_frame["run_length_min"], segment, 1)

    # A transition at row u is only usable at u+1, so shift first and then count
    # transitions in the 60 rows strictly before t.
    transition_lag = _shift_within_segment(
        state_frame["is_transition"].astype("float64"), segment, 1
    )
    transitions_1h = _rolling_sum_within_segment(transition_lag, segment, 60)

    day_type = pd.Series(
        np.where(index.dayofweek >= 5, "weekend", "weekday"),
        index=index,
        name="day_type",
    )
    hour = pd.Series(index.hour.astype("int64"), index=index, name="hour")

    cells = pd.Series(pd.NA, index=index, dtype="object", name=CONTEXT_CELL_COLUMN)
    valid = state_lag1.notna()
    cells.loc[valid] = (
        state_lag1.loc[valid].astype(str) + "|" + day_type.loc[valid].astype(str)
    )

    frame = pd.DataFrame(
        {
            "segment_id": segment,
            "state_lag1": state_lag1,
            "run_length_lag1_min": run_lag1,
            "transitions_1h_lag1": transitions_1h,
            "day_type": day_type,
            "hour": hour,
            CONTEXT_CELL_COLUMN: cells,
        },
        index=index,
    )

    # Keep diagnostics next to features so downstream code can audit them.
    for col in NON_FEATURE_COLUMNS:
        frame[col] = state_frame[col]
    return frame


def feature_matrix(context: pd.DataFrame) -> pd.DataFrame:
    """Select only the causal feature columns, in a stable order."""
    missing = [c for c in FEATURE_COLUMNS if c not in context.columns]
    if missing:
        raise SchemaError(f"missing feature columns: {missing}")
    return context.loc[:, list(FEATURE_COLUMNS)].copy()


def assert_feature_columns_are_causal(columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS) -> None:
    """Fail fast if a same-row state column is proposed as a feature (L7)."""
    forbidden = set(columns) & set(NON_FEATURE_COLUMNS)
    if forbidden:
        raise SchemaError(f"same-row state columns cannot be features: {sorted(forbidden)}")
    if "state" in columns:
        raise SchemaError("same-row 'state' cannot be a feature (R2/L7)")
