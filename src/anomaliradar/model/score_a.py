"""Score A — value anomaly.

Design (``docs/methodology.md`` §5):

* five approved value channels, TP2 excluded (R4);
* per-channel ridge regression on **strictly causal** features: lags 1 and 5,
  rolling means over the past 15/60/360 minutes, and rolling MAD over the past
  60 minutes, all computed inside a data segment;
* four context arms: none, calendar, lagged state, hybrid;
* residual ``e = y - y_hat``, causal EWMA scale of ``|e|`` (half-life 1 h,
  shifted by one minute), floored by the train median absolute residual;
* per-channel nonconformity ``log1p(|e| / max(scale, floor))`` and
  ``a_t = max_c a_{c,t}``.

The model never sees a same-row state column. Context is lagged ``s_{t-1}`` and
``r_{t-1}`` only. All fitting and standardization statistics come from the
training rows passed by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from anomaliradar.model.common import (
    apply_standardize,
    safe_standardize,
    segment_ewm,
    segment_rolling,
    segment_rolling_mad,
    segment_shift,
)
from anomaliradar.model.config import CONTEXT_ARMS, ScoreAConfig


@dataclass(frozen=True, slots=True)
class ChannelModel:
    channel: str
    feature_columns: tuple[str, ...]
    stats: dict[str, tuple[float, float]]
    ridge: Ridge
    scale_floor: float

    def design(self, features: pd.DataFrame, context: pd.DataFrame, arm: str) -> pd.DataFrame:
        parts = [features.loc[:, list(self.feature_columns)]]
        ctx = context_design(context, arm)
        if len(ctx.columns):
            parts.append(ctx)
        frame = pd.concat(parts, axis=1)
        return apply_standardize(frame, self.stats)

    def predict(self, features: pd.DataFrame, context: pd.DataFrame, arm: str) -> np.ndarray:
        design = self.design(features, context, arm)
        return self.ridge.predict(design.to_numpy(dtype=float))


@dataclass(frozen=True, slots=True)
class ScoreAModel:
    arm: str
    channels: tuple[str, ...]
    models: dict[str, ChannelModel]
    config: ScoreAConfig

    def predict(self, features: dict[str, pd.DataFrame], context: pd.DataFrame) -> pd.DataFrame:
        return score_a_frame(self, features, context)


def _context_columns(arm: str) -> tuple[str, ...]:
    calendar = ("hour_sin", "hour_cos", "weekend")
    state = ("state_offloaded", "state_loaded", "state_unknown", "run_length_lag1_min", "transitions_1h_lag1")
    if arm == "none":
        return ()
    if arm == "calendar":
        return calendar
    if arm == "state":
        return state
    if arm == "hybrid":
        return calendar + state
    raise ValueError(f"unknown context arm {arm!r}; expected one of {CONTEXT_ARMS}")


def context_design(context: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Build the context block for one arm, using only lagged state and calendar."""
    cols = _context_columns(arm)
    if not cols:
        return pd.DataFrame(index=context.index)
    index = context.index
    out = pd.DataFrame(index=index)

    if arm in ("calendar", "hybrid"):
        hour = context["hour"].to_numpy(dtype=float)
        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        out["weekend"] = (context["day_type"].astype(str) == "weekend").astype(float).to_numpy()

    if arm in ("state", "hybrid"):
        state = context["state_lag1"].astype(object)
        out["state_offloaded"] = (state == "offloaded").astype(float).to_numpy()
        out["state_loaded"] = (state == "loaded").astype(float).to_numpy()
        # Missing context is its own indicator; the raw string never enters a model.
        out["state_unknown"] = (~state.isin(("off", "offloaded", "loaded"))).astype(float).to_numpy()
        out["run_length_lag1_min"] = pd.to_numeric(
            context["run_length_lag1_min"], errors="coerce"
        ).to_numpy(dtype=float)
        out["transitions_1h_lag1"] = pd.to_numeric(
            context["transitions_1h_lag1"], errors="coerce"
        ).to_numpy(dtype=float)

    return out.loc[:, list(cols)]


def reference_context_design(context: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Context block with every context variable set to its reference level.

    Reference: weekday, noon, lagged state ``off``, zero recent transitions.
    Used only for the context-effect diagnostic, never for scoring.
    """
    rows = len(context)
    out = pd.DataFrame(index=context.index)
    if arm in ("calendar", "hybrid"):
        out["hour_sin"] = np.zeros(rows)
        out["hour_cos"] = np.full(rows, -1.0)   # cos(pi) at 12:00
        out["weekend"] = np.zeros(rows)
    if arm in ("state", "hybrid"):
        out["state_offloaded"] = np.zeros(rows)
        out["state_loaded"] = np.zeros(rows)
        out["state_unknown"] = np.zeros(rows)
        out["run_length_lag1_min"] = np.zeros(rows)
        out["transitions_1h_lag1"] = np.zeros(rows)
    return out.loc[:, list(_context_columns(arm))]


def build_score_a_features(
    grid_frame: pd.DataFrame,
    config: ScoreAConfig,
) -> dict[str, pd.DataFrame]:
    """Build the per-channel causal feature blocks once per dataset."""
    segment = grid_frame["segment_id"]
    features: dict[str, pd.DataFrame] = {}
    for channel in config.channels:
        values = pd.to_numeric(grid_frame[channel], errors="coerce")
        past = segment_shift(values, segment, 1)
        block = pd.DataFrame(index=grid_frame.index)
        for lag in config.lags:
            block[f"lag{lag}"] = segment_shift(values, segment, lag)
        for window in config.rolling_windows:
            block[f"mean{window}"] = segment_rolling(past, segment, window, "mean")
        block[f"mad{config.dispersion_window}"] = segment_rolling_mad(
            past, segment, config.dispersion_window
        )
        features[channel] = block
    return features


def fit_score_a(
    features: dict[str, pd.DataFrame],
    context: pd.DataFrame,
    grid_frame: pd.DataFrame,
    train_positions: np.ndarray,
    *,
    arm: str,
    config: ScoreAConfig | None = None,
) -> ScoreAModel:
    """Fit one ridge model per channel on the supplied training rows only."""
    cfg = config or ScoreAConfig()
    train_positions = np.asarray(train_positions, dtype=int)
    if train_positions.size == 0:
        raise ValueError("cannot fit Score A on zero training rows")

    ctx = context_design(context, arm)
    models: dict[str, ChannelModel] = {}
    for channel in cfg.channels:
        block = features[channel]
        parts = [block]
        if len(ctx.columns):
            parts.append(ctx)
        design = pd.concat(parts, axis=1)
        target = pd.to_numeric(grid_frame[channel], errors="coerce").to_numpy(dtype=float)
        keep = train_positions[np.isfinite(target[train_positions])]
        if keep.size < 30:
            raise ValueError(f"not enough finite training rows for {channel}: {keep.size}")

        train_design = design.iloc[keep]
        standardized, stats = safe_standardize(train_design)
        ridge = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True)
        ridge.fit(standardized.to_numpy(dtype=float), target[keep])

        pred_train = ridge.predict(standardized.to_numpy(dtype=float))
        residual_train = np.abs(target[keep] - pred_train)
        floor = float(np.quantile(residual_train, cfg.scale_floor_quantile))
        floor = max(floor, 1e-9)
        models[channel] = ChannelModel(
            channel=channel,
            feature_columns=tuple(block.columns),
            stats=stats,
            ridge=ridge,
            scale_floor=floor,
        )
    return ScoreAModel(
        arm=arm,
        channels=tuple(cfg.channels),
        models=models,
        config=cfg,
    )


def score_a_frame(
    model: ScoreAModel,
    features: dict[str, pd.DataFrame],
    context: pd.DataFrame,
    *,
    grid_frame: pd.DataFrame | None = None,
    segment: pd.Series | None = None,
) -> pd.DataFrame:
    """Predict every row and return residual, scale and nonconformity columns."""
    cfg = model.config
    if segment is None:
        if grid_frame is None:
            raise ValueError("either grid_frame or segment must be supplied")
        segment = grid_frame["segment_id"]

    out = pd.DataFrame(index=context.index)
    per_channel = []
    effects = []
    for channel in cfg.channels:
        cm = model.models[channel]
        block = features[channel]
        parts = [block]
        ctx = context_design(context, model.arm)
        if len(ctx.columns):
            parts.append(ctx)
        design, _ = safe_standardize(
            pd.concat(parts, axis=1), stats=cm.stats
        )
        pred = cm.ridge.predict(design.to_numpy(dtype=float))

        if grid_frame is None:
            raise ValueError("grid_frame is required to compute residuals")
        target = pd.to_numeric(grid_frame[channel], errors="coerce").to_numpy(dtype=float)
        residual = target - pred
        abs_residual = pd.Series(np.abs(residual), index=context.index)
        scale = segment_ewm(
            abs_residual, segment, halflife=cfg.scale_halflife_min, shift=1
        )
        scale = scale.to_numpy(dtype=float)
        scale = np.where(np.isfinite(scale), scale, cm.scale_floor)
        denom = np.maximum(scale, cm.scale_floor)
        raw = np.abs(residual) / denom
        a_c = np.log1p(raw) if cfg.log1p_scores else raw

        out[f"resid_{channel}"] = residual
        out[f"scale_{channel}"] = denom
        out[f"pred_{channel}"] = pred
        out[f"a_{channel}"] = a_c
        per_channel.append(a_c)

        # Context effect: actual context vs the reference context, same residual
        # scale. Only meaningful for arms that use context.
        if model.arm != "none":
            ref_ctx = reference_context_design(context, model.arm)
            ref_parts = [block]
            if len(ref_ctx.columns):
                ref_parts.append(ref_ctx)
            ref_design, _ = safe_standardize(
                pd.concat(ref_parts, axis=1), stats=cm.stats
            )
            pred_ref = cm.ridge.predict(ref_design.to_numpy(dtype=float))
            effects.append(np.abs(pred - pred_ref) / denom)

    matrix = np.vstack(per_channel)
    argmax = np.argmax(matrix, axis=0)
    out["a_channel"] = [cfg.channels[i] for i in argmax]
    out["a"] = matrix.max(axis=0)
    if effects:
        out["context_effect"] = np.vstack(effects).max(axis=0)
    else:
        out["context_effect"] = 0.0
    return out
