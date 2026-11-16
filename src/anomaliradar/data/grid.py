"""1-minute analysis grid.

Native sampling is ~10 s with jitter. Rolling statistics computed on row indices
therefore mix time scales, so the data layer aggregates to one row per minute.

Rules (``docs/methodology.md`` §4.1):

* a minute with no sample is *missing* (NaN), never zero;
* forward-fill is allowed only when the surrounding observations are at most
  ``max_ffill_minutes`` apart (default 2 min, one missing minute);
* any larger gap is a **break**: no interpolation, no forward-fill, and a new
  ``segment_id`` so that state and run length reset. Breaks longer than one hour
  are additionally masked as warm-up by :mod:`anomaliradar.data.masking`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaliradar.data.schema import ANALOG, DIGITAL, TIMESTAMP, SchemaError

ANALOG_COLUMNS = list(ANALOG)
DIGITAL_COLUMNS = list(DIGITAL)


@dataclass(frozen=True, slots=True)
class MinuteGrid:
    """Aggregated series plus a table of breaks."""

    frame: pd.DataFrame
    breaks: pd.DataFrame
    max_ffill_minutes: int

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def observed_rows(self) -> int:
        return int(self.frame["observed"].sum())

    @property
    def has_value_rows(self) -> int:
        return int(self.frame["has_value"].sum())

    @property
    def wall_clock_hours(self) -> float:
        return len(self.frame) * 60.0 / 3600.0

    def values(self, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
        cols = list(columns) if columns is not None else list(ANALOG) + list(DIGITAL)
        return self.frame.loc[:, cols]


def to_minute_grid(
    frame: pd.DataFrame,
    *,
    freq: str = "1min",
    max_ffill_minutes: int = 2,
) -> MinuteGrid:
    """Aggregate a validated native frame onto a contiguous 1-minute grid.

    Analog channels use the mean of the samples in the minute; digital channels
    use the last sample, because averaging a binary switch would invent an
    intermediate value. The grid spans every minute between the first and last
    observation, including minutes that are entirely missing.
    """
    if max_ffill_minutes < 1:
        raise ValueError("max_ffill_minutes must be >= 1")
    if TIMESTAMP not in frame.columns:
        raise SchemaError(f"missing required columns: ['{TIMESTAMP}']")

    ts = frame[TIMESTAMP]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise SchemaError("timestamp must be datetime before gridding")
    if not ts.is_monotonic_increasing:
        raise SchemaError("timestamps must be sorted ascending before gridding")

    bins = ts.dt.floor(freq)
    grouped = frame.groupby(bins, sort=True)

    aggregated = grouped[ANALOG_COLUMNS].mean()
    aggregated = aggregated.join(grouped[DIGITAL_COLUMNS].last())
    aggregated["n_obs"] = grouped.size().astype("int64")

    full_index = pd.date_range(
        aggregated.index.min(),
        aggregated.index.max(),
        freq=freq,
        name=TIMESTAMP,
    )
    aggregated = aggregated.reindex(full_index)

    observed = aggregated["n_obs"].notna().to_numpy()
    n_obs = aggregated["n_obs"].fillna(0).astype("int64")

    # Timestamps of the nearest real observation on each side of every row.
    idx = aggregated.index
    observed_ts = pd.Series(idx.to_numpy(), index=idx).where(observed)
    prev_obs = observed_ts.ffill()
    next_obs = observed_ts.bfill()

    bracket = next_obs - prev_obs
    fillable = (
        (~observed)
        & bracket.notna()
        & (bracket <= pd.Timedelta(minutes=max_ffill_minutes))
    ).to_numpy()

    for col in (*ANALOG_COLUMNS, *DIGITAL_COLUMNS):
        original = aggregated[col]
        filled = original.ffill()
        aggregated[col] = original.where(~fillable, filled)

    aggregated["n_obs"] = n_obs
    aggregated["observed"] = observed
    aggregated["ffilled"] = fillable
    aggregated["has_value"] = observed | fillable

    # -- segment ids and breaks ------------------------------------------- #
    obs_index = idx[observed]
    obs_series = pd.Series(obs_index, index=np.arange(len(obs_index)))
    deltas = obs_series.diff()
    threshold = pd.Timedelta(minutes=max_ffill_minutes)
    is_break = (deltas > threshold).to_numpy()

    segment_obs = np.cumsum(is_break).astype("int64")
    segment = pd.Series(np.nan, index=idx, dtype="float64")
    segment.loc[observed] = segment_obs
    segment = segment.ffill()
    aggregated["segment_id"] = segment.astype("Int64")

    rows = []
    for pos in np.flatnonzero(is_break):
        delta = deltas.iloc[pos]
        missing = int(delta / pd.Timedelta(minutes=1)) - 1
        rows.append(
            {
                "start": obs_index[pos - 1],
                "end": obs_index[pos],
                "delta_minutes": round(delta.total_seconds() / 60.0, 3),
                "missing_minutes": missing,
            }
        )
    breaks = pd.DataFrame(rows, columns=["start", "end", "delta_minutes", "missing_minutes"])

    aggregated.index.name = TIMESTAMP
    return MinuteGrid(
        frame=aggregated,
        breaks=breaks,
        max_ffill_minutes=max_ffill_minutes,
    )
