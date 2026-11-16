"""Pre-registered baselines, all converted to the same p-value + alarm pipeline.

B1 global robust z
    Per channel, median and 1.4826·MAD estimated on **training rows**, then
    ``z_t = max_c |y_c,t - med_c| / scale_c``.

B2 rolling robust z
    The same statistic with a trailing 6 h median and MAD, computed inside each
    data segment on values shifted by one minute. The first 359 minutes use an
    expanding window; no future row is read. The dispersion floor is the median
    positive train-window MAD, so a quiet window cannot produce an unbounded z.

B3 Isolation Forest (raw)
    Five approved channels at the current minute.

B4 Isolation Forest + context
    B3 plus calendar (hour sin/cos, weekend) and lagged state (one-hots, lagged
    run length, lagged past-hour transitions).

All four are fit on training rows only and then calibrated with the *same*
Mondrian hierarchy and budget solver as Scores A/B (protocol C8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaliradar.model.common import (
    robust_location_scale,
    segment_rolling,
    segment_rolling_mad,
    segment_shift,
)
from anomaliradar.model.config import VALUE_CHANNELS, BaselineConfig
from anomaliradar.model.score_a import context_design


@dataclass(frozen=True, slots=True)
class GlobalZModel:
    channels: tuple[str, ...]
    stats: dict[str, tuple[float, float]]

    def score(self, grid_frame: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(grid_frame), dtype=float)
        for channel in self.channels:
            center, scale = self.stats[channel]
            values = pd.to_numeric(grid_frame[channel], errors="coerce").to_numpy(
                dtype=float
            )
            z = np.abs(values - center) / scale
            z = np.where(np.isfinite(z), z, 0.0)
            score = np.maximum(score, z)
        return score

    def to_dict(self) -> dict:
        return {
            "channels": list(self.channels),
            "stats": {k: {"center": v[0], "scale": v[1]} for k, v in self.stats.items()},
        }


def fit_global_z(
    grid_frame: pd.DataFrame,
    train_positions: np.ndarray,
    *,
    channels: tuple[str, ...] = VALUE_CHANNELS,
    config: BaselineConfig | None = None,
) -> GlobalZModel:
    cfg = config or BaselineConfig()
    train_positions = np.asarray(train_positions, dtype=int)
    stats: dict[str, tuple[float, float]] = {}
    for channel in channels:
        values = pd.to_numeric(grid_frame[channel], errors="coerce").to_numpy(
            dtype=float
        )
        center, scale = robust_location_scale(
            values[train_positions], floor_quantile=cfg.mad_floor_quantile
        )
        if not np.isfinite(center):
            center, scale = 0.0, 1.0
        stats[channel] = (float(center), float(max(scale, 1e-9)))
    return GlobalZModel(channels=tuple(channels), stats=stats)


@dataclass(frozen=True, slots=True)
class RollingZComponents:
    channels: tuple[str, ...]
    window: int
    center: dict[str, np.ndarray]
    mad: dict[str, np.ndarray]

    def score(
        self,
        grid_frame: pd.DataFrame,
        train_positions: np.ndarray,
        *,
        floor_quantile: float = 0.5,
    ) -> np.ndarray:
        score = np.zeros(len(grid_frame), dtype=float)
        for channel in self.channels:
            center = self.center[channel]
            mad = self.mad[channel]
            train_mad = mad[np.asarray(train_positions, dtype=int)]
            train_mad = train_mad[np.isfinite(train_mad) & (train_mad > 0)]
            floor = (
                float(np.quantile(train_mad, floor_quantile))
                if train_mad.size
                else 1e-6
            )
            scale = np.maximum(1.4826 * mad, max(floor, 1e-9))
            values = pd.to_numeric(grid_frame[channel], errors="coerce").to_numpy(
                dtype=float
            )
            z = np.abs(values - center) / scale
            z = np.where(np.isfinite(z), z, 0.0)
            score = np.maximum(score, z)
        return score


def build_rolling_z_components(
    grid_frame: pd.DataFrame,
    *,
    channels: tuple[str, ...] = VALUE_CHANNELS,
    config: BaselineConfig | None = None,
) -> RollingZComponents:
    """Compute the causal trailing 6 h center and MAD once per dataset."""
    cfg = config or BaselineConfig()
    segment = grid_frame["segment_id"]
    center: dict[str, np.ndarray] = {}
    mad: dict[str, np.ndarray] = {}
    for channel in channels:
        values = pd.to_numeric(grid_frame[channel], errors="coerce")
        past = segment_shift(values, segment, 1)
        center[channel] = segment_rolling(
            past, segment, cfg.rolling_window, "median"
        ).to_numpy(dtype=float)
        mad[channel] = segment_rolling_mad(
            past, segment, cfg.rolling_window
        ).to_numpy(dtype=float)
    return RollingZComponents(
        channels=tuple(channels),
        window=cfg.rolling_window,
        center=center,
        mad=mad,
    )


@dataclass(frozen=True, slots=True)
class IsolationForestModel:
    model: IsolationForest
    columns: tuple[str, ...]
    fill: dict[str, float]

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        design = frame.loc[:, list(self.columns)].copy()
        for column in self.columns:
            values = pd.to_numeric(design[column], errors="coerce").to_numpy(
                dtype=float
            )
            values = np.where(np.isfinite(values), values, self.fill[column])
            design[column] = values
        return -self.model.score_samples(design.to_numpy(dtype=float))


def _iforest_design(
    grid_frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    channels: tuple[str, ...],
    with_context: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=grid_frame.index)
    for channel in channels:
        frame[channel] = pd.to_numeric(grid_frame[channel], errors="coerce")
    if with_context:
        ctx = context_design(context, "hybrid")
        for column in ctx.columns:
            frame[column] = ctx[column].to_numpy(dtype=float)
    return frame


def fit_iforest(
    grid_frame: pd.DataFrame,
    context: pd.DataFrame,
    train_positions: np.ndarray,
    *,
    with_context: bool,
    channels: tuple[str, ...] = VALUE_CHANNELS,
    config: BaselineConfig | None = None,
) -> IsolationForestModel:
    cfg = config or BaselineConfig()
    design = _iforest_design(
        grid_frame, context, channels=channels, with_context=with_context
    )
    train = design.iloc[np.asarray(train_positions, dtype=int)]
    fill: dict[str, float] = {}
    for column in design.columns:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        fill[column] = float(np.median(finite)) if finite.size else 0.0
    train = train.copy()
    for column in design.columns:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
        train[column] = np.where(np.isfinite(values), values, fill[column])

    model = IsolationForest(
        n_estimators=cfg.iforest_estimators,
        random_state=cfg.random_state,
        n_jobs=1,
    )
    model.fit(train.to_numpy(dtype=float))
    return IsolationForestModel(
        model=model,
        columns=tuple(design.columns),
        fill=fill,
    )


def iforest_scores(
    model: IsolationForestModel,
    grid_frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    with_context: bool,
    channels: tuple[str, ...] = VALUE_CHANNELS,
) -> np.ndarray:
    design = _iforest_design(
        grid_frame, context, channels=channels, with_context=with_context
    )
    return model.score(design)
