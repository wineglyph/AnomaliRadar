"""Read-only loaders for the frozen AnomaliRadar experiment artefacts.

Design rules
------------
* Every number shown by the application comes from ``reports/experiments`` or
  from the locally downloaded MetroPT-3 CSV.
* No model is fitted here. The experiment outputs are treated as frozen.
* Loaders are cached with :func:`functools.lru_cache` so the page code can call
  them freely and the same objects are reused across reruns.
* Missing artefacts are reported explicitly. The app never substitutes demo or
  synthetic values.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORTS = ROOT / "reports" / "experiments"
METRICS = REPORTS / "metrics"
ALARMS = REPORTS / "alarms" / "alarms.csv"
PREDICTIONS = REPORTS / "predictions"
FIGURES = REPORTS / "figures"
RAW_CSV = ROOT / "data" / "raw" / "metropt3" / "MetroPT3(AirCompressor).csv"
RAW_MANIFEST = ROOT / "data" / "raw" / "manifest.json"

# Make the package importable when Streamlit is launched from the repository
# root without an editable install.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------------- #
# Method presentation
# --------------------------------------------------------------------------- #
METHOD_LABELS: dict[str, str] = {
    "P1_A_hybrid": "Score A — hybrid",
    "P1_A_none": "Score A — no context",
    "P1_A_state": "Score A — state only",
    "P1_A_calendar": "Score A — calendar only",
    "P2_B_B1": "Score B — B1",
    "P2_B_B3": "Score B — B3",
    "P3_or": "OR union (recommended)",
    "P3_or_bonferroni": "OR union — Bonferroni column",
    "P3_and": "AND union",
    "P3_weighted": "Weighted λ=0.5",
    "P3_weighted_l000": "Weighted λ=0.00",
    "P3_weighted_l025": "Weighted λ=0.25",
    "P3_weighted_l050": "Weighted λ=0.50",
    "P3_weighted_l075": "Weighted λ=0.75",
    "P3_weighted_l100": "Weighted λ=1.00",
    "B1_global_z": "Baseline — global robust z",
    "B2_rolling_z": "Baseline — rolling robust z",
    "B3_iforest_raw": "Baseline — IForest raw",
    "B4_iforest_context": "Baseline — IForest + context",
    "Mnone_P3_or": "OR — masks disabled (ablation)",
    "Mhyg_P3_or": "OR — hygienic masks (ablation)",
}

#: Methods shown on the overview and comparison pages, in reading order.
COMPARISON_METHODS: tuple[str, ...] = (
    "P1_A_hybrid",
    "P2_B_B1",
    "P3_or",
    "P3_and",
    "P3_weighted",
    "B1_global_z",
    "B2_rolling_z",
    "B3_iforest_raw",
    "B4_iforest_context",
)

#: Methods with a temperature-like p-value stream exported per minute.
SCORE_METHODS: tuple[str, ...] = ("P1_A_hybrid", "P2_B_B1", "P3_or")

EVENT_LABELS: dict[int, str] = {
    1: "Event 1 · 2020-04-18",
    2: "Event 2 · 2020-05-29/30",
    3: "Event 3 · 2020-06-05/07",
    4: "Event 4 · 2020-07-15",
}


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


# --------------------------------------------------------------------------- #
# Generic loaders
# --------------------------------------------------------------------------- #
def artifact_exists(path: Path) -> bool:
    return Path(path).exists()


def missing_artifacts() -> list[str]:
    required = [
        REPORTS / "run_manifest.json",
        METRICS / "summary.json",
        METRICS / "fold_results.csv",
        METRICS / "event_outcomes.csv",
        METRICS / "context_diagnostics.csv",
        METRICS / "mask_coverage.csv",
        METRICS / "falsification.json",
        ALARMS,
        *[PREDICTIONS / f"fold_{i}.csv" for i in range(1, 5)],
    ]
    return [str(p.relative_to(ROOT)) for p in required if not p.exists()]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_summary() -> dict[str, Any]:
    return _read_json(METRICS / "summary.json")


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    return _read_json(REPORTS / "run_manifest.json")


@lru_cache(maxsize=1)
def load_falsification() -> dict[str, Any]:
    return _read_json(METRICS / "falsification.json")


@lru_cache(maxsize=1)
def load_fold_results() -> pd.DataFrame:
    frame = pd.read_csv(METRICS / "fold_results.csv")
    frame["fold"] = frame["fold"].astype(str)
    return frame


@lru_cache(maxsize=1)
def load_event_outcomes() -> pd.DataFrame:
    frame = pd.read_csv(METRICS / "event_outcomes.csv")
    frame["fold"] = frame["fold"].astype(str)
    return frame


@lru_cache(maxsize=1)
def load_context_diagnostics() -> pd.DataFrame:
    frame = pd.read_csv(METRICS / "context_diagnostics.csv")
    frame["fold"] = frame["fold"].astype(str)
    return frame


@lru_cache(maxsize=1)
def load_mask_coverage() -> pd.DataFrame:
    return pd.read_csv(METRICS / "mask_coverage.csv")


@lru_cache(maxsize=1)
def load_cell_sizes() -> pd.DataFrame:
    return pd.read_csv(METRICS / "calibration_cell_sizes.csv")


@lru_cache(maxsize=1)
def load_alarms() -> pd.DataFrame:
    frame = pd.read_csv(ALARMS)
    frame["fold"] = frame["fold"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


@lru_cache(maxsize=4)
def load_predictions(fold: str) -> pd.DataFrame:
    path = PREDICTIONS / f"fold_{fold}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


@lru_cache(maxsize=1)
def load_raw_manifest() -> dict[str, Any] | None:
    if not RAW_MANIFEST.exists():
        return None
    return _read_json(RAW_MANIFEST)


def dataset_available() -> bool:
    return RAW_CSV.exists()


@lru_cache(maxsize=1)
def load_dataset():
    """Build the deterministic data layer once for data-explorer pages.

    The returned object is the same :class:`anomaliradar.data.pipeline.Dataset`
    used by the experiment. It is read-only in this application. The import is
    local so the metric-only pages still work when the dataset is absent.
    """
    from anomaliradar.data.loader import load_native
    from anomaliradar.data.pipeline import build_dataset

    return build_dataset(load_native())


# --------------------------------------------------------------------------- #
# Folds, events, metrics
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def fold_table() -> pd.DataFrame:
    rows = load_manifest()["fold_table"]
    frame = pd.DataFrame(rows)
    for column in ("calibration_start", "calibration_end", "train_start", "train_end",
                   "test_start", "test_end", "detection_start", "detection_end",
                   "event_start", "event_end"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["fold"] = frame["fold_id"].astype(str)
    return frame


@lru_cache(maxsize=1)
def events() -> pd.DataFrame:
    rows = load_manifest()["dataset_summary"]["labels"]
    frame = pd.DataFrame(rows)
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    return frame


def fold_window(fold: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    table = fold_table()
    row = table[table["fold"] == str(fold)]
    if row.empty:
        return None
    row = row.iloc[0]
    start = row["test_start"]
    end = row["test_end"]
    if pd.isna(start) or pd.isna(end):
        return None
    return pd.Timestamp(start), pd.Timestamp(end)


def available_budgets() -> list[float]:
    budgets = sorted(load_fold_results()["budget"].dropna().unique().tolist())
    return [float(b) for b in budgets]


def method_row(fold: str, method: str, budget: float) -> pd.Series | None:
    frame = load_fold_results()
    match = frame[
        (frame["fold"] == str(fold))
        & (frame["method"] == method)
        & (frame["budget"].round(6) == round(float(budget), 6))
    ]
    return None if match.empty else match.iloc[0]


def alarms_for(fold: str, method: str, budget: float) -> pd.DataFrame:
    frame = load_alarms()
    match = frame[
        (frame["fold"] == str(fold))
        & (frame["method"] == method)
        & (frame["budget"].round(6) == round(float(budget), 6))
    ]
    return match.sort_values("timestamp").reset_index(drop=True)


def pooled_comparison(budget: float, methods: tuple[str, ...] = COMPARISON_METHODS) -> pd.DataFrame:
    """Pool event-fold counts and report tail numbers side by side.

    Rates are pooled counts divided by pooled observed days, matching the
    event-fold aggregation in ``summary.json``. Recall is the number of
    detected event folds out of the four labelled events.
    """
    frame = load_fold_results()
    event = frame[(frame["fold"].isin(["1", "2", "3", "4"])) & (frame["budget"] == budget)]
    tail = frame[(frame["fold"] == "T") & (frame["budget"] == budget)]

    rows: list[dict[str, Any]] = []
    for method in methods:
        ev = event[event["method"] == method]
        tl = tail[tail["method"] == method]
        if ev.empty:
            continue
        observed = float(ev["observed_days"].sum())
        alarms = int(ev["alarms"].sum())
        false = int(ev["false_alarms"].sum())
        detected = int(ev["event_recall_count"].sum())
        tail_days = float(tl["observed_days"].iloc[0]) if not tl.empty else float("nan")
        tail_alarms = int(tl["alarms"].iloc[0]) if not tl.empty else 0
        rows.append(
            {
                "method": method,
                "method_label": label(method),
                "recall_count": detected,
                "recall": f"{detected}/4",
                "alarms": alarms,
                "false_alarms": false,
                "observed_days": observed,
                "alarms_per_day": alarms / observed if observed else float("nan"),
                "false_per_day": false / observed if observed else float("nan"),
                "mean_abs_budget_error": float(ev["budget_error"].abs().mean()),
                "tail_alarms": tail_alarms,
                "tail_days": tail_days,
                "tail_per_day": tail_alarms / tail_days if tail_days else float("nan"),
                "tail_relative_error": (tail_alarms / tail_days - budget) / budget
                if tail_days
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def event_matrix(budget: float, methods: tuple[str, ...] = COMPARISON_METHODS) -> pd.DataFrame:
    """Event-level detected flags and delays, one row per method."""
    frame = load_fold_results()
    subset = frame[
        frame["fold"].isin(["1", "2", "3", "4"])
        & (frame["method"].isin(methods))
        & (frame["budget"] == budget)
    ]
    rows = []
    for method in methods:
        block = subset[subset["method"] == method].set_index("fold")
        if block.empty:
            continue
        row: dict[str, Any] = {"method": method, "method_label": label(method)}
        for event_id in (1, 2, 3, 4):
            fold = str(event_id)
            if fold in block.index:
                detected = bool(block.loc[fold, f"event_{event_id}"])
                delay = block.loc[fold, f"delay_{event_id}"]
                row[f"E{event_id}"] = "detected" if detected else "missed"
                row[f"delay_{event_id}"] = None if pd.isna(delay) else float(delay)
            else:
                row[f"E{event_id}"] = "not evaluated"
                row[f"delay_{event_id}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_diagnostics(method: str, budget: float) -> pd.DataFrame:
    frame = load_fold_results()
    subset = frame[
        (frame["method"] == method)
        & (frame["budget"] == budget)
        & (frame["fold"].isin(["1", "2", "3", "4", "T"]))
    ].copy()
    keep = [
        "fold",
        "alpha",
        "cal_reference_rows",
        "cal_selection_rows",
        "cal_selection_refused",
        "cal_selection_levels",
        "cal_test_refused",
        "cal_alarm_n_lt_50",
        "cal_alarm_n_lt_200",
        "cal_alarm_max_depth",
        "cal_alarm_levels",
        "cal_budget_reachable",
        "cal_budget_coarse",
        "cal_calibration_within_50pct",
        "cal_calibration_relative_error",
    ]
    return subset[[c for c in keep if c in subset.columns]].sort_values("fold")


def figure_path(name: str) -> Path:
    return FIGURES / name


def available_figures() -> list[str]:
    if not FIGURES.exists():
        return []
    return sorted(p.name for p in FIGURES.glob("*.png"))


def provenance() -> dict[str, str]:
    manifest = load_manifest()
    summary = load_summary()
    return {
        "config_hash": str(summary.get("config_hash", manifest.get("config_hash", ""))),
        "fingerprint": str(manifest.get("dataset_fingerprint", "")),
        "python": str(manifest.get("python", "")),
        "numpy": str(manifest.get("numpy", "")),
        "pandas": str(manifest.get("pandas", "")),
    }
