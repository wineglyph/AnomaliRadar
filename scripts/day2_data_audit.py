#!/usr/bin/env python3
"""Day 2 data-layer audit for MetroPT-3.

Recomputes every number quoted in ``docs/day2_report.md`` from the raw CSV using
the package itself, then writes machine-readable artefacts:

    reports/day2/data_audit.json          full snapshot
    reports/day2/masks_by_event.csv       M1/M2/M3 overlap per labelled event
    reports/day2/folds.csv                chronological fold boundaries
    reports/day2/splits.csv               row counts per split and exclusion reason
    reports/day2/calibration_cells.csv    calibration cell counts per fold

The script is read-only with respect to the raw data and implements no model,
score, calibration or alarm. It is deterministic: no random seeds, no sampling,
and every table is sorted before writing.

Usage:
    .venv/bin/python scripts/day2_data_audit.py

Exit codes:
    0  audit completed and the Day 2 stop condition (event 3 < 10%) holds
    1  the stop condition failed; the value is still written to the artefacts
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

# pandas 2.3.3 + numpy 2.5.4 emit this on every Timedelta construction. It is an
# upstream compatibility warning, not an audit finding.
warnings.filterwarnings(
    "ignore", message=".*generic.*unit for NumPy timedelta is deprecated.*"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anomaliradar.data.labels import FAILURE_EVENTS, label_mask  # noqa: E402
from anomaliradar.data.loader import default_dataset_path, load_native  # noqa: E402
from anomaliradar.data.pipeline import build_dataset  # noqa: E402

OUT = REPO_ROOT / "reports" / "day2"


def _jsonable(obj):
    """Recursively convert pandas/numpy scalars to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (TypeError, ValueError):
            return str(obj)
    return obj


def build_tables(ds):
    """Return the four report tables and the split payload."""
    splits: dict[str, dict] = {}
    split_rows: list[dict] = []
    cell_rows: list[dict] = []

    for fold in ds.folds():
        split = ds.split(fold)
        payload = split.to_dict()
        splits[fold.fold_id] = payload

        for name in ("train", "calibration", "test"):
            stats = split.stats[name]
            split_rows.append(
                {
                    "fold": fold.fold_id,
                    "split": name,
                    **{k: int(v) for k, v in stats.items()},
                    "kept_rows": int(getattr(split, name).size),
                }
            )
        split_rows.append(
            {
                "fold": fold.fold_id,
                "split": "detection",
                "kept_rows": int(split.detection.size),
            }
        )
        split_rows.append(
            {
                "fold": fold.fold_id,
                "split": "test_window",
                "kept_rows": int(split.test_window.size),
            }
        )

        cells = (
            ds.context["context_cell"]
            .iloc[split.calibration]
            .value_counts()
            .sort_index()
        )
        for cell, count in cells.items():
            cell_rows.append({"fold": fold.fold_id, "cell": str(cell), "rows": int(count)})

    mask_rows = pd.DataFrame(ds.mask_summary()["events"]).sort_values("event_id")
    fold_rows = pd.DataFrame(
        [
            {
                "fold": fold.fold_id,
                "event": fold.event_id,
                "train_start": fold.train.start,
                "train_end": fold.train.end,
                "calibration_start": fold.calibration.start,
                "calibration_end": fold.calibration.end,
                "test_start": fold.test.start,
                "test_end": fold.test.end,
                "detection_start": fold.detection.start if fold.detection else pd.NaT,
                "detection_end": fold.detection.end if fold.detection else pd.NaT,
                "embargo_days": round(
                    (fold.test.start - fold.calibration.end) / pd.Timedelta(days=1), 3
                ),
                "refit": fold.refit,
                "inherits_from": fold.inherits_from or "",
            }
            for fold in ds.folds()
        ]
    )

    return {
        "masks_by_event": mask_rows,
        "folds": fold_rows,
        "splits": pd.DataFrame(split_rows).sort_values(["fold", "split"]),
        "calibration_cells": pd.DataFrame(cell_rows).sort_values(["fold", "cell"]),
        "split_payload": splits,
    }


def leakage_checks(ds, tables) -> dict:
    """Recompute the data-side leakage invariants and return their verdicts."""
    index = ds.index
    events = ds.events
    results: dict[str, dict] = {}

    for fold in ds.folds():
        split = ds.split(fold)
        checks = {
            "train_before_calibration": bool(
                index[split.train].max() < index[split.calibration].min()
            )
            if len(split.train) and len(split.calibration)
            else True,
            "calibration_before_test": bool(
                index[split.calibration].max() < index[split.test].min()
            )
            if len(split.calibration) and len(split.test)
            else True,
            "sets_disjoint": bool(
                not (
                    set(split.train.tolist()) & set(split.calibration.tolist())
                    or set(split.train.tolist()) & set(split.test.tolist())
                    or set(split.calibration.tolist()) & set(split.test.tolist())
                )
            ),
            "no_masked_rows_in_fit": bool(
                not ds.masks.combined[split.train].any()
                and not ds.masks.combined[split.calibration].any()
            ),
            "no_labels_in_fit": bool(
                not label_mask(index[split.train], events).any()
                and not label_mask(index[split.calibration], events).any()
            ),
            "detection_subset_of_test": bool(set(split.detection.tolist()) <= set(split.test.tolist())),
        }
        checks["embargo_days"] = round(
            (fold.test.start - fold.calibration.end) / pd.Timedelta(days=1), 5
        )
        checks["calibration_cells"] = sorted(
            ds.context["context_cell"].iloc[split.calibration].dropna().unique().tolist()
        )
        results[fold.fold_id] = checks

    return results


def main() -> int:
    csv_path = default_dataset_path()
    if not csv_path.exists():
        print(f"missing CSV: {csv_path}", file=sys.stderr)
        print("run scripts/download_data.py first", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)

    native = load_native(csv_path)
    ds = build_dataset(native)
    tables = build_tables(ds)

    summary = ds.summary()
    leakage = leakage_checks(ds, tables)

    event3 = next(
        row for row in summary["masks"]["events"] if row["event_id"] == 3
    )
    stop_ok = bool(event3["combined_share"] < 0.10)

    audit = {
        "generated_by": "scripts/day2_data_audit.py",
        "dataset": {
            "csv": str(csv_path.relative_to(REPO_ROOT)),
            "native_rows": ds.native_rows,
            "first_timestamp": str(ds.index.min()),
            "last_timestamp": str(ds.index.max()),
        },
        "grid": {
            "grid_rows": summary["grid_rows"],
            "observed_rows": summary["grid_observed_rows"],
            "has_value_rows": summary["grid_has_value_rows"],
            "wall_clock_hours": summary["grid_wall_clock_hours"],
            "segments": summary["segments"],
            "breaks": summary["breaks_longer_than_ffill"],
        },
        "masks": summary["masks"],
        "state": summary["state"],
        "context": summary["context"],
        "folds": tables["split_payload"],
        "leakage": leakage,
        "stop_condition": {
            "rule": "event 3 combined mask share < 10%",
            "event3_combined_share": event3["combined_share"],
            "event3_combined_minutes": event3["combined_minutes"],
            "event3_window_minutes": event3["window_minutes"],
            "passed": stop_ok,
        },
        "fingerprint": summary["fingerprint"],
    }

    (OUT / "data_audit.json").write_text(
        json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tables["masks_by_event"].to_csv(OUT / "masks_by_event.csv", index=False)
    tables["folds"].to_csv(OUT / "folds.csv", index=False)
    tables["splits"].to_csv(OUT / "splits.csv", index=False)
    tables["calibration_cells"].to_csv(OUT / "calibration_cells.csv", index=False)

    bands = ", ".join(
        f"{e['event_id']}: {e['combined_share']:.4%}" for e in summary["masks"]["events"]
    )
    print(f"native rows      : {ds.native_rows:,}")
    print(f"grid rows        : {summary['grid_rows']:,}")
    print(f"observed rows    : {summary['grid_observed_rows']:,}")
    print(f"event mask shares: {bands}")
    print(
        "stop condition   : event 3 = "
        f"{event3['combined_share']:.4%} ({'PASS' if stop_ok else 'FAIL'} < 10%)"
    )
    print(f"fingerprint      : {summary['fingerprint'][:16]}")
    print(f"artefacts        : {OUT.relative_to(REPO_ROOT)}/")

    return 0 if stop_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
