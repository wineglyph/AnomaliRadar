"""Alarm layer: 30-minute debounce and calibration-only budget solving.

The alarm layer is deliberately small and shared by every method:

1. a row is a *candidate* if its fused p-value is at most ``alpha``;
2. candidates are scanned chronologically and one alarm is emitted; the next
   candidate is only eligible 30 minutes later (**debounce**, not the 3-minute
   state hysteresis);
3. ``alpha`` is chosen on the **calibration selection** part only, never on test
   data and never using labels.

The solver reports the floor rate when the requested budget cannot be reached,
and marks a budget *coarse* when the target count over the exposure is below one
alarm. It never silently changes the requested budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from anomaliradar.model.config import AlarmConfig


@dataclass(frozen=True, slots=True)
class BudgetSolution:
    budget: float
    alpha: float
    calibration_alarms: int
    calibration_days: float
    calibration_rate: float
    target_alarms: float
    reachable: bool
    floor_rate: float
    coarse: bool
    absolute_error: float

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "alpha": self.alpha,
            "calibration_alarms": self.calibration_alarms,
            "calibration_days": self.calibration_days,
            "calibration_rate": self.calibration_rate,
            "target_alarms": self.target_alarms,
            "reachable": self.reachable,
            "floor_rate": self.floor_rate,
            "coarse": self.coarse,
            "absolute_error": self.absolute_error,
        }


@dataclass(frozen=True, slots=True)
class AlarmSet:
    positions: np.ndarray
    timestamps: pd.DatetimeIndex
    p: np.ndarray
    alpha: float
    budget: float
    calibration: BudgetSolution

    def __len__(self) -> int:
        return int(self.positions.size)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "position": self.positions,
                "timestamp": self.timestamps,
                "p": self.p,
                "alpha": self.alpha,
                "budget": self.budget,
            }
        )


def _minute_offsets(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Integer minute offsets; actual clock time, so gaps are respected."""
    values = pd.DatetimeIndex(timestamps).asi8
    return values // 60_000_000_000


def debounce_positions(
    timestamps: pd.DatetimeIndex,
    p: np.ndarray,
    alpha: float,
    debounce_minutes: int = 30,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy chronological selection with a minimum spacing of ``delta`` minutes.

    The first candidate after the previous alarm is kept, which is exactly the
    pre-registered "first triggering timestamp" rule. The loop runs once per
    emitted alarm, not once per candidate row, so it is cheap even when ``alpha``
    is large.
    """
    p = np.asarray(p, dtype=float)
    allow = np.ones(p.shape[0], dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    candidate = np.flatnonzero(allow & np.isfinite(p) & (p <= alpha))
    if candidate.size == 0:
        return np.array([], dtype=int)
    ts = _minute_offsets(pd.DatetimeIndex(timestamps))
    sub = ts[candidate]
    selected: list[int] = []
    cursor = 0
    total = sub.size
    while cursor < total:
        selected.append(int(candidate[cursor]))
        next_time = sub[cursor] + debounce_minutes
        cursor = int(np.searchsorted(sub, next_time, side="left"))
        if cursor >= total:
            break
    return np.asarray(selected, dtype=int)


def _rate(
    timestamps: pd.DatetimeIndex,
    p: np.ndarray,
    alpha: float,
    debounce_minutes: int,
    exposure_days: float,
    eligible: np.ndarray | None = None,
) -> int:
    del exposure_days
    return int(
        debounce_positions(timestamps, p, alpha, debounce_minutes, eligible).size
    )


def solve_budget(
    timestamps: pd.DatetimeIndex,
    p: np.ndarray,
    budget: float,
    config: AlarmConfig | None = None,
    *,
    exposure_days: float | None = None,
    eligible: np.ndarray | None = None,
) -> BudgetSolution:
    """Choose ``alpha`` on calibration data to reach ``budget`` alarms/day.

    Only ``timestamps`` and ``p`` are used. There is no label argument and no
    parameter that can read test data.
    """
    cfg = config or AlarmConfig()
    p = np.asarray(p, dtype=float)
    timestamps = pd.DatetimeIndex(timestamps)
    n = p.size
    days = (
        float(exposure_days)
        if exposure_days is not None
        else (n / 1440.0 if n else 0.0)
    )
    if n == 0 or days <= 0.0:
        return BudgetSolution(
            budget=budget,
            alpha=0.0,
            calibration_alarms=0,
            calibration_days=max(days, 0.0),
            calibration_rate=0.0,
            target_alarms=0.0,
            reachable=False,
            floor_rate=float("inf"),
            coarse=False,
            absolute_error=budget,
        )

    target = float(budget) * days
    floor_rate = 1.0 / days

    finite_p = p[np.isfinite(p)]
    if finite_p.size == 0:
        return BudgetSolution(
            budget=budget,
            alpha=0.0,
            calibration_alarms=0,
            calibration_days=days,
            calibration_rate=0.0,
            target_alarms=target,
            reachable=False,
            floor_rate=floor_rate,
            coarse=target < 1.0,
            absolute_error=budget,
        )

    candidates: set[float] = {0.0, 1.0, float(finite_p.max()), float(finite_p.min())}

    # 20-point quantile grid (pre-registered tie-break grid).
    for q in np.linspace(0.0, 1.0, cfg.solver_extra_grid):
        candidates.add(float(np.quantile(finite_p, q)))

    # A target-focused grid: candidates whose raw exceedance count brackets the
    # requested alarm count. Debounce needs more candidates than alarms, so scan
    # multipliers above one as well.
    sorted_p = np.sort(finite_p)
    for multiplier in np.geomspace(0.5, 8.0, 12):
        k = int(max(1, min(finite_p.size, round(target * multiplier))))
        candidates.add(float(sorted_p[k - 1]))

    # Bisection over alpha (40 iterations as pre-registered). The alarm count is
    # monotone non-decreasing in alpha, so a plain bisection is valid.
    lo, hi = 0.0, 1.0
    _rate(timestamps, p, hi, cfg.debounce_minutes, days, eligible)
    for _ in range(cfg.solver_grid):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _rate(timestamps, p, mid, cfg.debounce_minutes, days, eligible) >= target:
            hi = mid
        else:
            lo = mid
    candidates.update((lo, hi))

    best: tuple[float, float, int, float] | None = None  # (error, alpha, count, rate)
    for alpha in sorted(candidates):
        count = _rate(timestamps, p, alpha, cfg.debounce_minutes, days, eligible)
        rate = count / days
        error = abs(rate - budget)
        # Tie-break towards the smaller alpha: over-alarming is the worse
        # operational error, per protocol §6.3.
        candidate_key = (round(error, 12), alpha)
        if best is None or candidate_key < (round(best[0], 12), best[1]):
            best = (error, float(alpha), count, rate)

    assert best is not None
    error, alpha, count, rate = best
    # A requested budget below the single-alarm floor cannot be achieved.
    reachable = bool(floor_rate <= budget + 1e-12) and bool(np.isfinite(rate))
    if not reachable:
        alpha = 0.0
        count = _rate(timestamps, p, 0.0, cfg.debounce_minutes, days, eligible)
        rate = count / days

    return BudgetSolution(
        budget=float(budget),
        alpha=float(alpha),
        calibration_alarms=int(count),
        calibration_days=days,
        calibration_rate=float(rate),
        target_alarms=float(target),
        reachable=reachable,
        floor_rate=float(floor_rate),
        coarse=bool(target < 1.0),
        absolute_error=float(abs(rate - budget)),
    )


def emit_alarms(
    timestamps: pd.DatetimeIndex,
    p: np.ndarray,
    solution: BudgetSolution,
    config: AlarmConfig | None = None,
    *,
    eligible: np.ndarray | None = None,
) -> AlarmSet:
    """Apply the frozen ``alpha`` and debounce to a set of scored rows."""
    cfg = config or AlarmConfig()
    chosen = debounce_positions(
        timestamps, p, solution.alpha, cfg.debounce_minutes, eligible
    )
    return AlarmSet(
        positions=chosen,
        timestamps=pd.DatetimeIndex(pd.DatetimeIndex(timestamps)[chosen]),
        p=np.asarray(p, dtype=float)[chosen],
        alpha=solution.alpha,
        budget=solution.budget,
        calibration=solution,
    )


def constant_threshold_solver(
    timestamps: pd.DatetimeIndex,
    p: np.ndarray,
    budget: float,
    config: AlarmConfig | None = None,
) -> BudgetSolution:
    """Alias kept explicit at call sites: the solver is calibration-only."""
    return solve_budget(timestamps, p, budget, config)
