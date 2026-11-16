"""Segment-aware causal feature helpers shared by scores and baselines.

Every rolling statistic is computed **within a data segment**. A segment ends at
any native gap longer than the forward-fill tolerance, so no statistic bridges a
gap (L9). All helpers take a value series and a segment id series aligned on the
same index.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def segment_shift(values: pd.Series, segment: pd.Series, periods: int = 1) -> pd.Series:
    """Shift within each segment; the first ``periods`` rows are NaN."""
    return values.groupby(segment, sort=False, dropna=False).shift(periods)


def segment_rolling(
    values: pd.Series,
    segment: pd.Series,
    window: int,
    how: str = "mean",
) -> pd.Series:
    """Segment-aware rolling ``mean`` or ``median``; skips NaN values."""
    if window < 1:
        raise ValueError("window must be >= 1")
    grouped = values.groupby(segment, sort=False, dropna=False)
    rolled = getattr(grouped.rolling(window, min_periods=1), how)()
    return rolled.reset_index(level=0, drop=True).sort_index()


def segment_ewm(
    values: pd.Series,
    segment: pd.Series,
    halflife: float,
    *,
    shift: int = 0,
) -> pd.Series:
    """Segment-aware EWMA. With ``shift=1`` the value at ``t`` uses only ``< t``."""
    shifted = values.groupby(segment, sort=False, dropna=False).shift(shift)
    ewm = (
        shifted.groupby(segment, sort=False, dropna=False)
        .ewm(halflife=halflife, min_periods=1)
        .mean()
    )
    return ewm.reset_index(level=0, drop=True).sort_index()


def _expanding_mad(x: np.ndarray, window: int) -> np.ndarray:
    """Causal MAD for the first ``window-1`` rows of a segment (expanding)."""
    out = np.full(x.shape[0], np.nan, dtype=float)
    for end in range(min(window - 1, x.shape[0])):
        chunk = x[: end + 1]
        med = np.median(chunk)
        out[end] = np.median(np.abs(chunk - med))
    return out


def segment_rolling_mad(
    values: pd.Series,
    segment: pd.Series,
    window: int,
    *,
    chunk_rows: int = 4096,
) -> pd.Series:
    """Exact causal rolling median absolute deviation within each segment.

    The rolling window ends at the current row. When the series passed here has
    already been shifted by one, the statistic is strictly causal. The first
    ``window-1`` rows of each segment use an expanding window.

    Long segments are processed in bounded chunks so that the 6-hour baseline
    window does not allocate a full segment-by-window matrix.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    values = pd.Series(values)
    segment = pd.Series(segment, index=values.index)
    out = np.full(len(values), np.nan, dtype=float)

    # Contiguous-run scan; segments are already monotone in the grid.
    seg_values = segment.to_numpy()
    x = values.to_numpy(dtype=float)
    boundaries = np.flatnonzero(np.r_[True, seg_values[1:] != seg_values[:-1]])
    boundaries = np.append(boundaries, len(seg_values))

    from numpy.lib.stride_tricks import sliding_window_view

    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        chunk = x[b0:b1]
        n = len(chunk)
        if n == 0:
            continue
        if n >= window:
            res = np.full(n, np.nan, dtype=float)
            # Expanding warm-up for the first window-1 rows.
            warm = _expanding_mad(chunk, window)
            res[: window - 1] = warm[: window - 1]
            for start in range(window - 1, n, chunk_rows):
                end = min(start + chunk_rows, n)
                windows = sliding_window_view(
                    chunk[start - window + 1 : end], window
                )
                med = np.median(windows, axis=1)
                mad = np.median(np.abs(windows - med[:, None]), axis=1)
                res[start:end] = mad
        else:
            res = _expanding_mad(chunk, window)
        out[b0:b1] = res
    return pd.Series(out, index=values.index, name="rolling_mad")


def robust_location_scale(
    values: np.ndarray,
    *,
    floor_quantile: float = 0.5,
) -> tuple[float, float]:
    """Median and 1.4826·MAD, with a positive floor from the |x| distribution."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = float(np.quantile(np.abs(x - med), floor_quantile))
    if scale <= 0:
        scale = 1e-9
    return med, scale


def safe_standardize(
    frame: pd.DataFrame,
    stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Median-impute and z-score numeric columns using train-only statistics.

    If ``stats`` is None the statistics are estimated from ``frame`` (which must
    therefore be the training frame). Missing values are imputed with the train
    median, never with a test statistic.
    """
    out = frame.copy()
    learned: dict[str, tuple[float, float]] = {} if stats is None else dict(stats)
    for col in out.columns:
        values = out[col].to_numpy(dtype=float)
        if stats is None or col not in learned:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                center, scale = 0.0, 1.0
            else:
                center = float(np.median(finite))
                scale = float(np.std(finite))
                if scale <= 0:
                    scale = 1.0
            learned[col] = (center, scale)
        center, scale = learned[col]
        values = np.where(np.isfinite(values), values, center)
        out[col] = (values - center) / scale
    return out, learned


def apply_standardize(
    frame: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        center, scale = stats[col]
        values = out[col].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, center)
        out[col] = (values - center) / scale
    return out
