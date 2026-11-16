"""Pipeline orchestration for the Day 2 data layer.

``build_dataset`` is the only public entry point downstream code needs::

    ds = build_dataset(load_native())
    folds = ds.folds()                 # chronological, includes tail
    split = ds.split(folds[0])         # positional indices, leakage-checked
    X = ds.feature_matrix()            # causal context features only

No model, score, calibration or alarm object is created here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from anomaliradar.data.features import (
    FEATURE_COLUMNS,
    add_context_features,
    feature_matrix as _feature_matrix,
    assert_feature_columns_are_causal,
)
from anomaliradar.data.grid import MinuteGrid, to_minute_grid
from anomaliradar.data.labels import FAILURE_EVENTS, EventWindow, TAIL_WINDOW
from anomaliradar.data.masking import (
    GridMasks,
    MaskConfig,
    NativeMasks,
    build_grid_masks,
    detect_native_masks,
    mask_summary,
)
from anomaliradar.data.schema import validate_schema
from anomaliradar.data.splits import (
    FoldSpec,
    SplitIndices,
    build_folds,
    materialise_split,
)
from anomaliradar.data.state import StateConfig, attach_state, state_run_table

#: The protocol starts the analysis on 2020-02-01, the first complete month.
DEFAULT_SERIES_START = pd.Timestamp("2020-02-01")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    mask: MaskConfig = field(default_factory=MaskConfig)
    state: StateConfig = field(default_factory=StateConfig)
    grid_freq: str = "1min"
    max_ffill_minutes: int = 2


@dataclass(frozen=True, slots=True)
class Dataset:
    native_rows: int
    grid: MinuteGrid
    state: pd.DataFrame
    context: pd.DataFrame
    native_masks: NativeMasks
    masks: GridMasks
    events: tuple[EventWindow, ...]
    config: DatasetConfig = field(default_factory=DatasetConfig)

    # -- arrays used by splits -------------------------------------------- #
    @property
    def index(self) -> pd.DatetimeIndex:
        return self.grid.index

    @property
    def usable(self) -> np.ndarray:
        """Rows with a value after the ≤2-minute forward-fill rule."""
        return self.grid.frame["has_value"].to_numpy(dtype=bool)

    @property
    def transition(self) -> np.ndarray:
        """M4: confirmed state is ``other`` (excluded from fit/calibration)."""
        return self.state["is_other_state"].to_numpy(dtype=bool)

    @property
    def context_ready(self) -> np.ndarray:
        """Rows with a defined, calibratable lagged cell.

        ``other`` is not a calibratable state: it is a transition artefact. A row
        whose *lagged* state is ``other`` must not enter calibration even if its
        current state has already resolved.
        """
        cell_ok = self.context["context_cell"].notna().to_numpy(dtype=bool)
        state_ok = ~self.context["state_lag1"].isin(("other",)).to_numpy(dtype=bool)
        return cell_ok & state_ok

    # -- folds ------------------------------------------------------------- #
    def folds(self, *, include_tail: bool | None = None) -> list[FoldSpec]:
        """Build folds; the tail is included only when the data covers it."""
        if include_tail is None:
            include_tail = bool(len(self.index)) and self.index.max() >= TAIL_WINDOW.start
        series_start = max(self.index.min().floor("1min"), DEFAULT_SERIES_START)
        return build_folds(
            series_start,
            self.events,
            include_tail=include_tail,
        )

    def split(self, fold: FoldSpec) -> SplitIndices:
        return materialise_split(
            self.index,
            fold,
            mask=self.masks.combined,
            usable=self.usable,
            transition=self.transition,
            context_ready=self.context_ready,
            events=self.events,
        )

    # -- feature views ----------------------------------------------------- #
    def feature_matrix(self) -> pd.DataFrame:
        assert_feature_columns_are_causal(FEATURE_COLUMNS)
        return _feature_matrix(self.context)

    # -- reporting --------------------------------------------------------- #
    def mask_summary(self) -> dict:
        return mask_summary(self.grid, self.masks, self.native_masks, self.events)

    def state_summary(self) -> dict:
        confirmed = self.state["state"]
        counts = confirmed.value_counts(dropna=True)
        shares = (counts / len(confirmed)).to_dict()
        runs = state_run_table(self.state)
        medians = runs.groupby("state")["minutes"].median().to_dict()
        episode_counts = runs.groupby("state")["minutes"].size().to_dict()
        raw_counts = self.state["raw_state"].value_counts(dropna=True)
        return {
            "confirmed_counts": {str(k): int(v) for k, v in counts.items()},
            "confirmed_share": {str(k): round(float(v), 6) for k, v in shares.items()},
            "raw_counts": {str(k): int(v) for k, v in raw_counts.items()},
            "episodes": {str(k): int(v) for k, v in episode_counts.items()},
            "run_length_median_min": {str(k): float(v) for k, v in medians.items()},
            "other_state_rows": int(self.transition.sum()),
            "other_state_share": float(self.transition.mean()),
            "segments": int(self.state["segment_id"].nunique()),
        }

    def context_summary(self) -> dict:
        cells = self.context["context_cell"].value_counts(dropna=True)
        return {
            "cells": {str(k): int(v) for k, v in cells.items()},
            "missing_state_lag1": int(self.context["state_lag1"].isna().sum()),
            "feature_nan_share": {
                c: round(float(self.context[c].isna().mean()), 6)
                for c in FEATURE_COLUMNS
            },
            "feature_columns": list(FEATURE_COLUMNS),
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.masks.signature().encode("ascii"))
        # ``tobytes()`` on an object array returns pointer addresses, not values.
        # Hash the object column through pandas so the digest is content-stable
        # across processes and independent of Python's string interning.
        state_hashes = pd.util.hash_pandas_object(
            self.state["state"].fillna("<na>"), index=False
        ).to_numpy(dtype="uint64")
        digest.update(state_hashes.tobytes())
        digest.update(np.asarray(self.index.asi8, dtype="int64").tobytes())
        return digest.hexdigest()

    def summary(self) -> dict:
        return {
            "native_rows": int(self.native_rows),
            "grid_rows": int(len(self.grid)),
            "grid_observed_rows": int(self.grid.observed_rows),
            "grid_has_value_rows": int(self.grid.has_value_rows),
            "grid_wall_clock_hours": round(self.grid.wall_clock_hours, 3),
            "first_timestamp": str(self.index.min()),
            "last_timestamp": str(self.index.max()),
            "segments": int(self.grid.frame["segment_id"].nunique()),
            "breaks_longer_than_ffill": int(len(self.grid.breaks)),
            "masks": self.mask_summary(),
            "state": self.state_summary(),
            "context": self.context_summary(),
            "labels": [
                {
                    "event_id": e.event_id,
                    "pdf_label": e.pdf_label,
                    "start": str(e.start),
                    "end": str(e.end),
                    "hours": round(e.duration_hours, 3),
                }
                for e in self.events
            ],
            "fingerprint": self.fingerprint(),
        }


def build_dataset(
    frame: pd.DataFrame,
    *,
    events: Sequence[EventWindow] | None = None,
    config: DatasetConfig | None = None,
    validate: bool = True,
) -> Dataset:
    """Build the complete data layer from a validated native frame.

    The function has **no label argument**: labels are attached only for split
    materialisation, never for mask construction, state, or features.
    """
    cfg = config or DatasetConfig()
    if validate:
        validate_schema(frame)

    native_masks = detect_native_masks(frame, cfg.mask)
    grid = to_minute_grid(
        frame,
        freq=cfg.grid_freq,
        max_ffill_minutes=cfg.max_ffill_minutes,
    )
    state = attach_state(grid, cfg.state)
    context = add_context_features(grid, state)
    masks = build_grid_masks(grid, native_masks, cfg.mask)

    chosen_events = tuple(FAILURE_EVENTS if events is None else events)
    return Dataset(
        native_rows=len(frame),
        grid=grid,
        state=state,
        context=context,
        native_masks=native_masks,
        masks=masks,
        events=chosen_events,
        config=cfg,
    )
