"""Session-scoped fixtures for the real dataset.

All synthetic builders live in ``tests/helpers.py`` so importing them does not
drag pytest fixtures into ordinary modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = REPO_ROOT / "data" / "raw" / "metropt3" / "MetroPT3(AirCompressor).csv"


@pytest.fixture(scope="session")
def real_native() -> pd.DataFrame:
    if not REAL_CSV.exists():
        pytest.skip(f"MetroPT-3 not downloaded at {REAL_CSV}")
    from anomaliradar.data.loader import load_native

    return load_native(REAL_CSV)


@pytest.fixture(scope="session")
def real_dataset(real_native: pd.DataFrame):
    from anomaliradar.data.pipeline import build_dataset

    return build_dataset(real_native)
