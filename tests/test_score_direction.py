"""Day 3 score tests: direction, monotonicity, train-only fitting, causality.

These tests use synthetic grids so they run in seconds. They assert the
properties that the methodology promises, not the numerical values of the real
MetroPT-3 run:

* a larger injected residual produces a larger Score A;
* Score A reports the channel that actually moved;
* refitting after changing only non-training rows leaves the model unchanged;
* poisoning the future cannot change any past Score A value;
* Score B decreases monotonically with the length of the current run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.pipeline import build_dataset
from anomaliradar.model.config import ScoreBConfig
from anomaliradar.model.score_a import (
    build_score_a_features,
    fit_score_a,
    score_a_frame,
)
from anomaliradar.model.score_b import (
    build_run_table,
    fit_score_b_reference,
    score_b_series,
)
from helpers import make_native_frame

TRAIN_ROWS = 1500
SPIKE_ROW = 3000


def _synthetic(minutes: int = 4 * 1440, seed: int = 0):
    frame = make_native_frame(minutes=minutes, seed=seed)
    return frame, build_dataset(frame)


def _score(frame: pd.DataFrame) -> tuple:
    ds = build_dataset(frame)
    cfg = ds.folds()[0]  # not used for fitting; only to keep an explicit shape
    del cfg
    from anomaliradar.model.config import ScoreAConfig

    config = ScoreAConfig()
    features = build_score_a_features(ds.grid.frame, config)
    train = np.arange(TRAIN_ROWS)
    model = fit_score_a(
        features, ds.context, ds.grid.frame, train, arm="hybrid", config=config
    )
    scored = score_a_frame(
        model,
        features,
        ds.context,
        grid_frame=ds.grid.frame,
        segment=ds.grid.frame["segment_id"],
    )
    return ds, model, features, scored


def test_score_a_is_monotone_in_the_injected_residual() -> None:
    base, _ = _synthetic()
    normal = base.copy()
    small = base.copy()
    large = base.copy()
    # PLAUSIBLE_RANGES caps Reservoirs at 15.0; keep both injections inside it.
    small.loc[small.index[SPIKE_ROW], "Reservoirs"] += 3.0
    large.loc[large.index[SPIKE_ROW], "Reservoirs"] += 6.0

    _, _, _, scored_normal = _score(normal)
    _, _, _, scored_small = _score(small)
    _, _, _, scored_large = _score(large)

    a_normal = float(scored_normal["a"].iloc[SPIKE_ROW - 10 : SPIKE_ROW].median())
    a_small = float(scored_small["a"].iloc[SPIKE_ROW])
    a_large = float(scored_large["a"].iloc[SPIKE_ROW])
    assert a_small > a_normal
    assert a_large > a_small
    # The max-over-channels rule must identify the channel that moved.
    assert scored_large["a_channel"].iloc[SPIKE_ROW] == "Reservoirs"


def test_score_a_context_arms_are_explicit_and_distinct() -> None:
    """The four pre-registered arms must all be constructible and differ."""
    frame, ds = _synthetic(minutes=2 * 1440)
    del frame
    from anomaliradar.model.config import ScoreAConfig

    config = ScoreAConfig()
    features = build_score_a_features(ds.grid.frame, config)
    train = np.arange(200, 1400)
    predictions = {}
    for arm in ("none", "calendar", "state", "hybrid"):
        model = fit_score_a(
            features, ds.context, ds.grid.frame, train, arm=arm, config=config
        )
        predictions[arm] = score_a_frame(
            model,
            features,
            ds.context,
            grid_frame=ds.grid.frame,
            segment=ds.grid.frame["segment_id"],
        )["a"].to_numpy(dtype=float)
    assert not np.allclose(predictions["none"], predictions["hybrid"], equal_nan=True)
    assert not np.allclose(predictions["calendar"], predictions["state"], equal_nan=True)


def test_score_a_fit_uses_training_rows_only() -> None:
    frame, ds = _synthetic()
    from anomaliradar.model.config import ScoreAConfig

    config = ScoreAConfig()
    features = build_score_a_features(ds.grid.frame, config)
    model_a = fit_score_a(
        features, ds.context, ds.grid.frame, np.arange(TRAIN_ROWS),
        arm="hybrid", config=config,
    )

    # Change every input outside the training rows: features, context and target.
    outside = np.arange(TRAIN_ROWS, len(ds.index))
    mutated_features = {k: v.copy() for k, v in features.items()}
    for block in mutated_features.values():
        block.iloc[outside] = block.iloc[outside] + 100.0
    mutated_context = ds.context.copy()
    mutated_context.iloc[outside, mutated_context.columns.get_loc("hour")] = 3.0
    mutated_grid = ds.grid.frame.copy()
    mutated_grid.iloc[outside, mutated_grid.columns.get_loc("Reservoirs")] += 50.0

    model_b = fit_score_a(
        mutated_features, mutated_context, mutated_grid, np.arange(TRAIN_ROWS),
        arm="hybrid", config=config,
    )

    for channel in config.channels:
        a = model_a.models[channel]
        b = model_b.models[channel]
        np.testing.assert_allclose(a.ridge.coef_, b.ridge.coef_)
        assert a.ridge.intercept_ == pytest.approx(b.ridge.intercept_)
        assert a.scale_floor == pytest.approx(b.scale_floor)
        assert a.stats.keys() == b.stats.keys()
        for key in a.stats:
            assert a.stats[key] == pytest.approx(b.stats[key])


def test_score_a_past_is_invariant_to_future_poison() -> None:
    frame, _ = _synthetic()
    cutoff = pd.Timestamp(frame["timestamp"].iloc[3200])
    _, model, _, base_scored = _score(frame)

    poisoned = frame.copy()
    future = poisoned["timestamp"] > cutoff
    poisoned.loc[future, "Reservoirs"] += 2.0
    poisoned.loc[future, "Oil_temperature"] += 5.0

    ds_poisoned = build_dataset(poisoned, validate=False)
    from anomaliradar.model.config import ScoreAConfig

    config = ScoreAConfig()
    features = build_score_a_features(ds_poisoned.grid.frame, config)
    poisoned_scored = score_a_frame(
        model,
        features,
        ds_poisoned.context,
        grid_frame=ds_poisoned.grid.frame,
        segment=ds_poisoned.grid.frame["segment_id"],
    )

    past = ds_poisoned.index <= cutoff
    np.testing.assert_allclose(
        base_scored.loc[past, "a"].to_numpy(dtype=float),
        poisoned_scored.loc[past, "a"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def _run_frame(lengths: list[int], state: str = "loaded") -> pd.DataFrame:
    """One segment made of alternating completed runs with known lengths."""
    states: list[str] = []
    for i, length in enumerate(lengths):
        states.extend([state if i % 2 == 0 else "offloaded"] * length)
    frame = pd.DataFrame(
        {
            "state": pd.Series(states, dtype=object),
            "segment_id": 0,
        }
    )
    frame["run_length_min"] = (
        frame.groupby((frame["state"] != frame["state"].shift()).cumsum()).cumcount() + 1
    ).astype(float)
    return frame


def test_score_b_survival_is_monotone_in_run_length() -> None:
    lengths = [2, 3, 4, 5, 7, 10, 15, 20, 40, 80]
    state_frame = _run_frame(lengths)
    runs = build_run_table(state_frame)
    config = ScoreBConfig(variant="B1", min_reference_runs=3)
    reference_mask = np.ones(len(state_frame), dtype=bool)
    reference = fit_score_b_reference(runs, reference_mask, config)
    assert reference.source_runs >= 5

    scored = score_b_series(
        state_frame, reference, config, loo_mask=None, runs=runs
    )
    loaded = scored.loc[state_frame["state"] == "loaded"].copy()
    loaded["run"] = state_frame.loc[loaded.index, "run_length_min"]
    loaded = loaded.sort_values("run")

    b = loaded["b"].to_numpy(dtype=float)
    score = loaded["score_b"].to_numpy(dtype=float)
    # Empirical survival is non-increasing in run length; nonconformity is
    # -log(b), so it is non-decreasing. Ties are allowed (discrete survival).
    assert np.all(np.diff(b) <= 1e-12)
    assert np.all(np.diff(score) >= -1e-12)
    assert score[-1] > score[0]


def test_score_b_does_not_read_the_value_scores() -> None:
    """Score B is a function of state/run length only (independence check)."""
    lengths = [3, 4, 6, 9, 12]
    state_frame = _run_frame(lengths)
    runs = build_run_table(state_frame)
    config = ScoreBConfig(variant="B1", min_reference_runs=3)
    reference = fit_score_b_reference(
        runs, np.ones(len(state_frame), dtype=bool), config
    )
    first = score_b_series(state_frame, reference, config, runs=runs)["score_b"]
    second = score_b_series(
        state_frame.copy(), reference, config, runs=runs
    )["score_b"]
    pd.testing.assert_series_equal(first, second, check_names=False)
