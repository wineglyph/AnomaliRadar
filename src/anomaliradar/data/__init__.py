"""Data layer for AnomaliRadar (Day 2).

Import order is acyclic: ``schema`` → ``loader``/``labels`` → ``grid`` →
``masking``/``state`` → ``features`` → ``splits`` → ``pipeline``.
"""

from __future__ import annotations

from anomaliradar.data.grid import MinuteGrid, to_minute_grid
from anomaliradar.data.labels import (
    FAILURE_EVENTS,
    EventWindow,
    label_mask,
    failure_windows,
)
from anomaliradar.data.loader import default_dataset_path, load_native
from anomaliradar.data.masking import (
    GridMasks,
    MaskConfig,
    NativeMasks,
    build_grid_masks,
    detect_frozen_blocks,
    detect_major_gaps,
)
from anomaliradar.data.pipeline import Dataset, build_dataset
from anomaliradar.data.schema import (
    ANALOG,
    DIGITAL,
    REGIME_CHANNELS,
    SCORE_A_EXCLUDED,
    VALUE_CHANNELS,
    SchemaError,
    normalise_columns,
    validate_schema,
)
from anomaliradar.data.splits import FoldSpec, SplitIndices, build_folds, validate_folds
from anomaliradar.data.state import confirm_state, raw_state, run_length

__all__ = [
    "ANALOG",
    "DIGITAL",
    "REGIME_CHANNELS",
    "SCORE_A_EXCLUDED",
    "VALUE_CHANNELS",
    "Dataset",
    "EventWindow",
    "FAILURE_EVENTS",
    "FoldSpec",
    "GridMasks",
    "MaskConfig",
    "MinuteGrid",
    "NativeMasks",
    "SchemaError",
    "SplitIndices",
    "build_dataset",
    "build_folds",
    "build_grid_masks",
    "confirm_state",
    "default_dataset_path",
    "detect_frozen_blocks",
    "detect_major_gaps",
    "failure_windows",
    "label_mask",
    "load_native",
    "normalise_columns",
    "raw_state",
    "run_length",
    "to_minute_grid",
    "validate_folds",
    "validate_schema",
]
