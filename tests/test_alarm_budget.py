"""Day 3 alarm-layer tests: debounce, budget solver and refusal.

The budget solver is calibration-only. These tests never pass test data to it;
they check monotonicity, unreachable budgets, refusal and the 30-minute
debounce, which is distinct from the 3-minute state hysteresis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaliradar.data.state import StateConfig
from anomaliradar.model.alarm import (
    AlarmConfig,
    debounce_positions,
    emit_alarms,
    solve_budget,
)


def _timestamps(minutes: int, start: str = "2020-03-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=minutes, freq="1min")


def test_debounce_keeps_the_first_candidate_and_spaces_alarms() -> None:
    timestamps = _timestamps(121)
    p = np.full(121, 0.001)
    chosen = debounce_positions(timestamps, p, alpha=0.01, debounce_minutes=30)
    assert chosen.tolist() == [0, 30, 60, 90, 120]
    gaps = np.diff(timestamps[chosen]) / pd.Timedelta(minutes=1)
    assert gaps.min() == 30


def test_debounce_uses_clock_time_across_gaps() -> None:
    # A 90-minute data gap: the next candidate is immediately eligible after it.
    timestamps = pd.DatetimeIndex(
        list(pd.date_range("2020-03-01 00:00", periods=5, freq="1min"))
        + list(pd.date_range("2020-03-01 01:35", periods=5, freq="1min"))
    )
    p = np.full(len(timestamps), 0.001)
    chosen = debounce_positions(timestamps, p, alpha=0.01, debounce_minutes=30)
    assert chosen.tolist() == [0, 5]


def test_refused_rows_can_never_fire_even_at_alpha_one() -> None:
    timestamps = _timestamps(60)
    p = np.zeros(60)
    eligible = np.zeros(60, dtype=bool)
    assert debounce_positions(
        timestamps, p, alpha=1.0, debounce_minutes=30, eligible=eligible
    ).size == 0

    eligible[10] = True
    chosen = debounce_positions(
        timestamps, p, alpha=1.0, debounce_minutes=30, eligible=eligible
    )
    assert chosen.tolist() == [10]


def test_budget_solver_is_accurate_on_an_exchangeable_null() -> None:
    rng = np.random.default_rng(3)
    minutes = 14_400  # ten observed days
    timestamps = _timestamps(minutes)
    p = rng.uniform(size=minutes)

    solution = solve_budget(timestamps, p, 1.0, AlarmConfig())
    assert solution.reachable
    assert solution.alpha > 0.0
    assert abs(solution.calibration_rate - 1.0) <= 0.5
    alarms = emit_alarms(timestamps, p, solution, AlarmConfig())
    assert abs(len(alarms) / 10.0 - 1.0) <= 0.5


def test_budget_solver_is_monotone_in_the_requested_budget() -> None:
    rng = np.random.default_rng(5)
    timestamps = _timestamps(14_400)
    p = rng.uniform(size=14_400)
    solutions = {
        budget: solve_budget(timestamps, p, budget, AlarmConfig())
        for budget in (0.5, 1.0, 2.0, 4.0)
    }
    alphas = [solutions[b].alpha for b in (0.5, 1.0, 2.0, 4.0)]
    assert np.all(np.diff(alphas) >= 0.0)
    assert solutions[4.0].calibration_rate > solutions[1.0].calibration_rate


def test_unreachable_and_coarse_budgets_are_reported_not_silently_changed() -> None:
    timestamps = _timestamps(1_440)
    p = np.full(1_440, 0.5)

    coarse = solve_budget(timestamps, p, 0.25, AlarmConfig())
    assert coarse.coarse
    assert not coarse.reachable
    assert coarse.floor_rate == pytest.approx(1.0)
    assert coarse.calibration_alarms == 0

    reachable = solve_budget(timestamps, p, 1.0, AlarmConfig())
    assert reachable.reachable


def test_solver_never_uses_refused_rows_to_reach_the_budget() -> None:
    rng = np.random.default_rng(9)
    timestamps = _timestamps(14_400)
    p = rng.uniform(size=14_400)
    eligible = np.zeros(14_400, dtype=bool)
    eligible[::7] = True

    solution = solve_budget(
        timestamps, p, 2.0, AlarmConfig(), eligible=eligible
    )
    chosen = debounce_positions(
        timestamps, p, solution.alpha, 30, eligible=eligible
    )
    assert np.all(eligible[chosen])
    assert solution.calibration_alarms == chosen.size


def test_solver_output_depends_only_on_calibration_inputs() -> None:
    """The same calibration input must give the same alpha on any test slice."""
    rng = np.random.default_rng(13)
    timestamps = _timestamps(14_400)
    p = rng.uniform(size=14_400)
    solution = solve_budget(timestamps, p, 1.0, AlarmConfig())

    test_timestamps = _timestamps(2_880, start="2020-04-01")
    test_p = np.full(2_880, 0.0001)
    test_alarms = emit_alarms(test_timestamps, test_p, solution, AlarmConfig())
    assert test_alarms.alpha == solution.alpha
    assert len(test_alarms) > 0  # all test rows pass the frozen alpha


def test_debounce_is_not_state_hysteresis() -> None:
    assert AlarmConfig().debounce_minutes == 30
    assert StateConfig().min_dwell == 3
    assert AlarmConfig().debounce_minutes != StateConfig().min_dwell
