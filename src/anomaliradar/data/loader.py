"""Deterministic loader for the raw MetroPT-3 CSV.

Responsibilities are deliberately narrow: read the file, normalise the column
names, parse timestamps, sort deterministically, validate, and return a frame.
No aggregation, no masks, no labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anomaliradar.data.schema import (
    TIMESTAMP,
    SchemaError,
    coerce_timestamp,
    drop_unnamed,
    normalise_columns,
    validate_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "metropt3" / "MetroPT3(AirCompressor).csv"


def default_dataset_path() -> Path:
    return DEFAULT_CSV


def load_native(
    path: str | Path | None = None,
    *,
    validate: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Load the raw CSV deterministically.

    Parameters
    ----------
    path:
        CSV location. Defaults to the project's download location.
    validate:
        Run the full schema contract. Disable only for deliberately broken
        fixtures.
    sort:
        Stable-sort by timestamp. The real file is already sorted; sorting is
        applied only when needed so byte-identical input yields identical
        output.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` + 7 analog + 8 digital, one row per native
        sample, monotonically increasing timestamps, RangeIndex.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MetroPT-3 CSV not found at {csv_path}; run scripts/download_data.py"
        )

    frame = pd.read_csv(csv_path, low_memory=False)
    frame = normalise_columns(frame)
    frame = drop_unnamed(frame)
    frame = coerce_timestamp(frame)

    if sort and not frame[TIMESTAMP].is_monotonic_increasing:
        frame = frame.sort_values(TIMESTAMP, kind="stable")

    frame = frame.reset_index(drop=True)

    if validate:
        validate_schema(frame)
    return frame


def assert_no_future_rows(frame: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """Raise if any row is newer than ``cutoff`` (guard for manual slicing)."""
    if len(frame) and frame[TIMESTAMP].max() > cutoff:
        raise SchemaError(f"frame contains rows after cutoff {cutoff}")
