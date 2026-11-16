"""Day 3 reproducibility and mask-handling tests.

The reproducibility test runs the real fold-1 pipeline twice with identical
configuration and asserts byte-stable metrics/alarms. It is the slowest test in
the suite (~40 s per run); the rest of the suite stays fast.

The mask tests use the real dataset and assert the operational contract:
masked minutes are not observed, never scored, and therefore cannot generate an
alarm or a false alarm.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.labels import label_mask
from anomaliradar.model.config import ExperimentConfig
from anomaliradar.model.evaluation import exposure_days
from anomaliradar.model.runner import run_experiments

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = REPO_ROOT / "data" / "raw" / "metropt3" / "MetroPT3(AirCompressor).csv"


def test_config_hash_is_frozen_and_deterministic() -> None:
    config = ExperimentConfig()
    # Changing this hash requires a protocol changelog entry; it freezes the
    # entire pre-registered parameter set.
    assert config.hash() == "06954d34ced38410"
    assert ExperimentConfig().hash() == config.hash()
    other = dataclasses.replace(config, seed=config.seed + 1)
    assert other.hash() != config.hash()


@pytest.mark.skipif(
    not REAL_CSV.exists(),
    reason="MetroPT-3 not downloaded (run scripts/download_data.py)",
)
def test_fold1_run_is_reproducible(tmp_path: Path) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    common = dict(
        fold_ids=["1"],
        with_ablations=False,
        with_baselines=False,
        save_predictions=False,
        verbose=False,
    )
    first = run_experiments(output_dir=out_a, config=ExperimentConfig(), **common)
    second = run_experiments(output_dir=out_b, config=ExperimentConfig(), **common)

    assert first["manifest"]["config_hash"] == second["manifest"]["config_hash"]
    assert first["manifest"]["dataset_fingerprint"] == second["manifest"]["dataset_fingerprint"]

    metrics_a = pd.read_csv(out_a / "metrics" / "fold_results.csv")
    metrics_b = pd.read_csv(out_b / "metrics" / "fold_results.csv")
    pd.testing.assert_frame_equal(metrics_a, metrics_b, check_exact=True)

    alarms_a = pd.read_csv(out_a / "alarms" / "alarms.csv")
    alarms_b = pd.read_csv(out_b / "alarms" / "alarms.csv")
    assert len(alarms_a) > 0
    pd.testing.assert_frame_equal(alarms_a, alarms_b, check_exact=True)

    with open(out_a / "metrics" / "summary.json", encoding="utf-8") as fh:
        summary_a = json.load(fh)
    with open(out_b / "metrics" / "summary.json", encoding="utf-8") as fh:
        summary_b = json.load(fh)
    assert summary_a == summary_b


def test_real_fold3_masked_rows_are_never_scored(real_dataset) -> None:
    dataset = real_dataset
    fold = next(f for f in dataset.folds() if f.fold_id == "3")
    split = dataset.split(fold)
    masked = dataset.masks.combined

    assert masked[split.train].sum() == 0
    assert masked[split.calibration].sum() == 0
    assert masked[split.test].sum() == 0
    assert masked[split.detection].sum() == 0

    # The event-3 window really does contain masked minutes, so this is not a
    # vacuous assertion: masking removes observations from the scorable set.
    in_window = fold.test.mask(dataset.index)
    assert masked[in_window].any()
    assert len(split.test_window) > len(split.test)


def test_real_masked_event_minutes_cannot_be_detected(real_dataset) -> None:
    dataset = real_dataset
    fold = next(f for f in dataset.folds() if f.fold_id == "3")
    split = dataset.split(fold)
    masked = dataset.masks.combined

    # Event 3 contains masked minutes (two timestamp gaps). Those minutes are
    # absent from both the detection credit set and the scored test set, so they
    # can neither trigger an alarm nor be counted as a false alarm.
    event_mask = (dataset.index >= fold.event_start) & (dataset.index < fold.event_end)
    masked_in_event = event_mask & masked
    assert masked_in_event.any()
    assert not np.any(masked_in_event[split.test])
    assert not np.any(masked_in_event[split.detection])
    assert masked[split.detection].sum() == 0
    assert masked[split.test].sum() == 0


def test_real_fit_rows_exclude_labels_and_masks(real_dataset) -> None:
    dataset = real_dataset
    fold = next(f for f in dataset.folds() if f.fold_id == "4")
    split = dataset.split(fold)
    labelled = label_mask(dataset.index, dataset.events)
    assert labelled[split.train].sum() == 0
    assert labelled[split.calibration].sum() == 0
    # Test labels are deliberately retained: they are the evaluation.
    assert labelled[split.test].sum() > 0


def test_exposure_counts_observed_minutes_not_wall_clock() -> None:
    observed = pd.date_range("2020-03-01", periods=2_880, freq="1min")
    # A 3-day wall-clock window with only two observed days must be two days.
    assert exposure_days(observed) == pytest.approx(2.0)
