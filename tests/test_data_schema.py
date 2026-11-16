"""Schema contract tests for MetroPT-3.

These tests encode the assumptions the rest of the project relies on. They run
against a small synthetic fixture so CI never needs the 208 MB download; a
separate, automatically-skipped test validates the real file when present.

The contract itself lives in :mod:`anomaliradar.data.schema`; this module no
longer duplicates it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.loader import load_native
from anomaliradar.data.schema import (
    ANALOG,
    COLUMN_ALIASES,
    DIGITAL,
    REQUIRED,
    SchemaError,
    normalise_columns,
    validate_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = REPO_ROOT / "data" / "raw" / "metropt3" / "MetroPT3(AirCompressor).csv"


# --------------------------------------------------------------------------- #
# Synthetic fixture
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    n = 600
    start = pd.Timestamp("2020-03-01 00:00:00")
    rng = np.random.default_rng(0)
    ts = pd.date_range(start, periods=n, freq="min")
    comp = np.zeros(n, dtype=int)
    dv = np.zeros(n, dtype=int)
    comp[200:400] = 1          # intake active
    dv[100:500] = 1            # outlet active
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "TP2": rng.normal(8.5, 0.1, n),
            "TP3": rng.normal(8.4, 0.1, n),
            "H1": rng.normal(0.2, 0.05, n),
            "DV_pressure": rng.normal(0.05, 0.02, n),
            "Reservoirs": rng.normal(8.3, 0.15, n),
            "Oil_temperature": rng.normal(45.0, 2.0, n),
            "Motor_current": np.where(dv == 1, 7.0, 4.0) + rng.normal(0, 0.2, n),
            "COMP": comp,
            "DV_eletric": dv,
            "Towers": rng.integers(0, 2, n),
            "MPG": comp.copy(),
            "LPS": np.zeros(n, dtype=int),
            "Pressure_switch": rng.integers(0, 2, n),
            "Oil_level": np.zeros(n, dtype=int),
            "Caudal_impulses": np.zeros(n, dtype=int),
        }
    )
    frame["TP2"] = frame["TP2"].round(3)
    return frame


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_synthetic_fixture_passes_contract(synthetic_frame: pd.DataFrame) -> None:
    validate_schema(synthetic_frame)


def test_missing_column_is_rejected(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_schema(synthetic_frame.drop(columns=["Motor_current"]))


def test_string_timestamp_is_rejected(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.copy()
    broken["timestamp"] = broken["timestamp"].astype(str)
    with pytest.raises(SchemaError, match="datetime"):
        validate_schema(broken)


def test_unsorted_timestamps_are_rejected(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(SchemaError, match="sorted ascending"):
        validate_schema(broken)


def test_duplicate_timestamps_are_rejected(synthetic_frame: pd.DataFrame) -> None:
    # Duplicate in place: appending a row would break monotonic order first and
    # the test would pass for the wrong reason.
    broken = synthetic_frame.copy()
    broken.loc[broken.index[1], "timestamp"] = broken["timestamp"].iloc[0]
    with pytest.raises(SchemaError, match="duplicate timestamps"):
        validate_schema(broken)


def test_non_binary_digital_signal_is_rejected(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.copy()
    broken.loc[broken.index[0], "COMP"] = 2
    with pytest.raises(SchemaError, match="binary"):
        validate_schema(broken)


def test_impossible_motor_current_is_rejected(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.copy()
    broken.loc[broken.index[0], "Motor_current"] = 9999.0
    with pytest.raises(SchemaError, match="plausible range"):
        validate_schema(broken)


def test_missing_value_is_rejected_in_native_frame(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.copy()
    broken.loc[broken.index[0], "Reservoirs"] = np.nan
    with pytest.raises(SchemaError, match="missing values"):
        validate_schema(broken)


def test_regime_definition_is_exhaustive(synthetic_frame: pd.DataFrame) -> None:
    """The COMP/DV_eletric -> regime mapping must cover every observed combination."""
    combos = set(zip(synthetic_frame["COMP"], synthetic_frame["DV_eletric"]))
    assert combos <= {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_uci_page_spellings_are_normalised_to_csv_names(synthetic_frame: pd.DataFrame) -> None:
    """The UCI page/PDF spell three columns differently from the CSV payload."""
    documented = synthetic_frame.rename(
        columns={"DV_eletric": "DV_electric", "Towers": "TOWERS", "Caudal_impulses": "Caudal_impulse"}
    )
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_schema(documented)
    validate_schema(normalise_columns(documented))


def test_schema_constants_are_consistent() -> None:
    assert set(REQUIRED) == {"timestamp", *ANALOG, *DIGITAL}
    assert set(COLUMN_ALIASES.values()) <= set(REQUIRED)


@pytest.mark.skipif(not REAL_CSV.exists(), reason="MetroPT-3 not downloaded (run scripts/download_data.py)")
def test_real_dataset_satisfies_contract() -> None:
    frame = load_native(REAL_CSV)
    validate_schema(frame)
    assert len(frame) > 1_000_000
    # Verified facts from Day 0 inspection, asserted so regressions are caught.
    assert frame["timestamp"].duplicated().sum() == 0
    assert frame["timestamp"].diff().dt.total_seconds().median() == 10
    assert frame["Towers"].nunique() == 2
