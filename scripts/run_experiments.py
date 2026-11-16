#!/usr/bin/env python3
"""Run the complete AnomaliRadar Day 3 experiment suite.

Examples:
    .venv/bin/python scripts/run_experiments.py
    .venv/bin/python scripts/run_experiments.py --folds 1 --no-ablations --no-baselines
    .venv/bin/python scripts/run_experiments.py --output reports/experiments

The script is deterministic for a fixed config and dataset fingerprint. It writes
``reports/experiments/run_manifest.json`` plus metrics, alarms and predictions.
It does not create any Streamlit application.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message=".*generic.*unit for NumPy timedelta is deprecated.*"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anomaliradar.model.config import ExperimentConfig  # noqa: E402
from anomaliradar.model.runner import run_experiments  # noqa: E402


def _jsonable(obj):
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "reports" / "experiments"),
        help="output directory",
    )
    parser.add_argument(
        "--folds",
        default=None,
        help="comma-separated fold ids, e.g. 1,2 or 4,T; default all",
    )
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    fold_ids = None
    if args.folds:
        fold_ids = [part.strip() for part in args.folds.split(",") if part.strip()]

    result = run_experiments(
        output_dir=Path(args.output),
        config=ExperimentConfig(),
        fold_ids=fold_ids,
        with_ablations=not args.no_ablations,
        with_baselines=not args.no_baselines,
        save_predictions=not args.no_predictions,
        verbose=not args.quiet,
    )
    summary = result["summary"]
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
