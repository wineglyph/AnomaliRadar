"""Day 3 calibration tests: direction, hierarchy, refusal and separation.

The tests pin the pre-registered conformal behaviour:

* larger nonconformity -> smaller (or equal) p-value;
* cell -> state -> global back-off with ``n_min=200`` and ``n_refuse=50``;
* fewer than 50 reference rows -> refusal (``p=1``, ``eligible=False``);
* the chronological reference/selection split is disjoint and the calibrator
  never fits on selection rows, labels or masked rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.model.calibration import (
    CalibrationConfig,
    ConformalCalibrator,
    chronological_split,
)


def _fit(scores, cells, states, reference):
    return ConformalCalibrator(CalibrationConfig()).fit(
        np.asarray(scores, dtype=float),
        np.asarray(cells, dtype=object),
        np.asarray(states, dtype=object),
        np.asarray(reference, dtype=bool),
    )


def test_p_value_is_monotone_decreasing_in_the_score() -> None:
    scores = np.linspace(0.0, 100.0, 400)
    cells = np.full(scores.size, "loaded|weekday", dtype=object)
    states = np.full(scores.size, "loaded", dtype=object)
    calibrator = _fit(scores, cells, states, np.ones(scores.size, dtype=bool))
    query = np.array([-1.0, 1.0, 25.0, 99.0, 100.0, 500.0])
    result = calibrator.p_values(query, cells[: query.size], states[: query.size])
    assert np.all(np.diff(result.p) < 0.0)
    assert result.p[0] == pytest.approx(1.0)
    assert 0.0 < result.p[-1] < 0.01
    assert result.eligible.all()


def test_cell_to_state_back_off() -> None:
    # cell A: 250 rows (>= n_min) -> cell
    # cell B: 10 rows, state loaded has 260 (>= n_min) -> state
    scores = np.arange(600, dtype=float)
    cells = np.array(["A"] * 250 + ["B"] * 10 + ["C"] * 340, dtype=object)
    states = np.array(["loaded"] * 260 + ["offloaded"] * 340, dtype=object)
    calibrator = _fit(scores, cells, states, np.ones(600, dtype=bool))

    result = calibrator.p_values(
        np.array([100.0, 100.0]),
        np.array(["A", "B"], dtype=object),
        np.array(["loaded", "loaded"], dtype=object),
    )
    assert list(result.level) == ["cell", "state"]
    assert list(result.n) == [250, 260]
    assert list(result.depth) == [0, 1]
    assert result.eligible.all()


def test_state_to_global_back_off_when_every_smaller_level_is_under_50() -> None:
    scores = np.arange(300, dtype=float)
    cells = np.array(["rare"] * 5 + ["common"] * 295, dtype=object)
    states = np.array(["other"] * 30 + ["loaded"] * 270, dtype=object)
    calibrator = _fit(scores, cells, states, np.ones(300, dtype=bool))
    result = calibrator.p_values(
        np.array([100.0]),
        np.array(["rare"], dtype=object),
        np.array(["other"], dtype=object),
    )
    assert result.level[0] == "global"
    assert result.depth[0] == 2
    assert result.n[0] == 300
    assert result.eligible[0]


def test_low_density_state_is_used_before_global_and_is_flagged() -> None:
    scores = np.arange(300, dtype=float)
    cells = np.array(["rare"] * 5 + ["common"] * 295, dtype=object)
    states = np.array(["loaded"] * 100 + ["offloaded"] * 200, dtype=object)
    calibrator = _fit(scores, cells, states, np.ones(300, dtype=bool))
    result = calibrator.p_values(
        np.array([100.0]),
        np.array(["rare"], dtype=object),
        np.array(["loaded"], dtype=object),
    )
    assert result.level[0] == "state_low"
    assert result.n[0] == 100
    assert result.eligible[0]


def test_small_cell_is_refused_and_can_never_alarm() -> None:
    scores = np.arange(40, dtype=float)
    calibrator = _fit(
        scores,
        np.full(40, "rare", dtype=object),
        np.full(40, "loaded", dtype=object),
        np.ones(40, dtype=bool),
    )
    result = calibrator.p_values(
        np.array([1e9]),
        np.array(["rare"], dtype=object),
        np.array(["loaded"], dtype=object),
    )
    assert result.level[0] == "refused"
    assert result.eligible[0] == pytest.approx(0.0)
    assert result.p[0] == 1.0
    assert result.depth[0] == -1


def test_missing_keys_back_off_instead_of_crashing() -> None:
    scores = np.arange(400, dtype=float)
    cells = np.array([None] * 400, dtype=object)
    states = np.array(["loaded"] * 400, dtype=object)
    calibrator = _fit(scores, cells, states, np.ones(400, dtype=bool))
    result = calibrator.p_values(
        np.array([10.0, 900.0]),
        np.array([None, "unseen-cell"], dtype=object),
        np.array(["loaded", "loaded"], dtype=object),
    )
    assert result.eligible.all()
    assert result.level[0] == "state"
    assert result.level[1] == "state"


def test_chronological_split_is_disjoint_and_ordered() -> None:
    index = pd.date_range("2020-03-01", periods=120, freq="1min")
    positions = np.arange(120)
    split = chronological_split(positions, index, fraction=2.0 / 3.0)
    assert split.reference_rows == 80
    assert split.selection_rows == 40
    assert not np.any(split.reference & split.selection)
    assert np.array_equal(
        np.flatnonzero(split.reference | split.selection), positions
    )
    assert index[np.flatnonzero(split.reference)].max() < split.cutoff
    assert split.cutoff == index[np.flatnonzero(split.selection).min()]


def test_fit_ignores_rows_outside_the_reference_mask() -> None:
    rng = np.random.default_rng(7)
    scores = rng.normal(size=500)
    reference = np.zeros(500, dtype=bool)
    reference[:300] = True
    cells = np.full(500, "loaded|weekday", dtype=object)
    states = np.full(500, "loaded", dtype=object)

    first = _fit(scores, cells, states, reference)
    mutated = scores.copy()
    mutated[300:] += 1_000_000.0
    second = _fit(mutated, cells, states, reference)

    query = np.array([-2.0, 0.0, 2.0])
    q_cells = np.array(["loaded|weekday"] * 3, dtype=object)
    q_states = np.array(["loaded"] * 3, dtype=object)
    np.testing.assert_allclose(
        first.p_values(query, q_cells, q_states).p,
        second.p_values(query, q_cells, q_states).p,
    )
    assert first.reference_rows == 300
    assert second.reference_rows == 300


def test_masked_reference_rows_are_not_calibration_evidence() -> None:
    rng = np.random.default_rng(11)
    scores = rng.normal(size=600)
    reference = np.ones(600, dtype=bool)
    reference[200:400] = False  # stand-in for masked quality rows
    cells = np.full(600, "loaded|weekday", dtype=object)
    states = np.full(600, "loaded", dtype=object)

    calibrator = _fit(scores, cells, states, reference)
    assert calibrator.reference_rows == 400
    assert calibrator.global_scores.size == 400
    assert calibrator.cell_scores["loaded|weekday"].size == 400
