"""MetroPT-3 schema contract and channel roles.

The column names below are verified against the payload of
``MetroPT3(AirCompressor).csv`` (UCI ID 791), not against the UCI web page.
Three columns are spelled differently on the web page/PDF; the aliases map the
documented spelling onto the payload spelling.

Channel roles encode the anti-circularity rules R1–R5 from
``docs/methodology.md`` §5.2. They live here, next to the schema, so that no
downstream module can accidentally score a regime-defining channel.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
ANALOG: tuple[str, ...] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)
DIGITAL: tuple[str, ...] = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)
TIMESTAMP = "timestamp"
REQUIRED: tuple[str, ...] = (TIMESTAMP, *ANALOG, *DIGITAL)

#: Target channels for Score A. Deliberately excludes every regime-defining
#: channel (R1) and TP2 (R4).
VALUE_CHANNELS: tuple[str, ...] = (
    "Reservoirs",
    "TP3",
    "H1",
    "DV_pressure",
    "Oil_temperature",
)

#: Channels that define the operating state. Never Score A targets (R1).
REGIME_CHANNELS: tuple[str, ...] = ("Motor_current", "DV_eletric", "COMP", "MPG")

#: Channels that must never be a Score A target, for the stated reason.
SCORE_A_EXCLUDED: tuple[str, ...] = (*REGIME_CHANNELS, "TP2")

#: Channels that are physically constant while the compressor is idle; TP2 is
#: included here because M1 is a data-quality check, not a scoring decision.
FROZEN_CHECK_CHANNELS: tuple[str, ...] = ANALOG

# Documented spelling differences between the UCI description and the CSV.
COLUMN_ALIASES: dict[str, str] = {
    "DV_electric": "DV_eletric",
    "DV electric": "DV_eletric",
    "TOWERS": "Towers",
    "Caudal_impulse": "Caudal_impulses",
    "Caudal Impulse": "Caudal_impulses",
}

# Loose physical ranges: these catch unit/parse errors, not operational
# outliers (those are the signal). Negative lower bounds allow the small sensor
# zero-offsets observed on Day 0.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "TP2": (-1.0, 15.0),
    "TP3": (-1.0, 15.0),
    "H1": (-1.0, 15.0),
    "DV_pressure": (-1.0, 15.0),
    "Reservoirs": (-1.0, 15.0),
    "Oil_temperature": (-40.0, 120.0),
    "Motor_current": (-1.0, 30.0),
}


class SchemaError(ValueError):
    """Raised when a frame violates the MetroPT-3 contract."""


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip column whitespace and map documented spellings to payload names."""
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns=COLUMN_ALIASES)
    duplicates = out.columns[out.columns.duplicated()].tolist()
    if duplicates:
        raise SchemaError(f"duplicate column names after normalisation: {duplicates}")
    return out


def drop_unnamed(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the unnamed decisecond counter column present in the raw CSV."""
    keep = [c for c in frame.columns if c and not str(c).startswith("Unnamed")]
    return frame.loc[:, keep]


def coerce_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse the timestamp column deterministically and reject unparseable rows."""
    if TIMESTAMP not in frame.columns:
        raise SchemaError(f"missing required columns: ['{TIMESTAMP}']")
    out = frame.copy()
    raw = out[TIMESTAMP]
    if pd.api.types.is_datetime64_any_dtype(raw):
        parsed = raw
    else:
        try:
            parsed = pd.to_datetime(raw, errors="raise", format="mixed")
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            raise SchemaError(f"timestamp column is not parseable: {exc}") from exc
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    out[TIMESTAMP] = parsed
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")


def validate_schema(frame: pd.DataFrame, *, check_ranges: bool = True) -> None:
    """Validate the contract every consumer of MetroPT-3 may assume.

    Raises
    ------
    SchemaError
        With a message naming the violated rule. The function never mutates the
        frame and never looks at labels.
    """
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError("expected a pandas DataFrame")
    if len(frame) == 0:
        raise SchemaError("frame is empty")

    _require_columns(frame, REQUIRED)

    ts = frame[TIMESTAMP]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise SchemaError("timestamp must be parsed to datetime")
    if ts.isna().any():
        raise SchemaError("unparseable timestamps present")
    if not ts.is_monotonic_increasing:
        raise SchemaError("timestamps must be sorted ascending")
    if ts.duplicated().any():
        raise SchemaError("duplicate timestamps present")

    for col in (*ANALOG, *DIGITAL):
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise SchemaError(f"{col} must be numeric")

    for col in DIGITAL:
        uniques = set(pd.unique(frame[col].dropna()))
        if not uniques <= {0, 1}:
            raise SchemaError(f"{col} must be binary, found {sorted(uniques)[:5]}")

    if check_ranges:
        for col, (lo, hi) in PLAUSIBLE_RANGES.items():
            observed = frame[col].dropna()
            if not observed.between(lo, hi).all():
                bad = observed[~observed.between(lo, hi)].iloc[0]
                raise SchemaError(
                    f"{col} outside plausible range [{lo}, {hi}]: found {bad}"
                )

    # Native rows are expected to be complete (Day 0: zero missing values).
    for col in REQUIRED:
        if frame[col].isna().any():
            raise SchemaError(f"{col} contains missing values in the native frame")


def require_value_channels() -> None:
    """Assert the anti-circularity partition at import time (R1, R4)."""
    overlap = set(VALUE_CHANNELS) & set(SCORE_A_EXCLUDED)
    if overlap:
        raise SchemaError(f"Score A targets overlap forbidden channels: {sorted(overlap)}")
    if "TP2" in VALUE_CHANNELS:
        raise SchemaError("TP2 must not be a Score A target (R4)")


require_value_channels()
