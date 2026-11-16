"""Mondrian conformal calibration with hierarchical back-off.

Higher nonconformity -> smaller tail probability:

    p = (1 + #{s in S_g : s >= a}) / (|S_g| + 1)

Hierarchy: ``(state, day_type)`` -> ``state`` -> ``global``. A level is used if
it has at least ``n_min=200`` reference scores. If no level reaches 200, the first
level with at least ``n_refuse=50`` is used and reported as low-density. If no
level reaches 50, the row is **refused**: ``p=1`` and no alarm may be emitted.

Statistical audit (pre-registered deviation C1): the calibration window is split
chronologically into an earlier **reference** part (2/3) that builds the
conformal distributions and a later **selection** part (1/3) used only to choose
the alarm threshold ``alpha``. Test rows use the reference distributions. This
removes the optimistic in-sample budget estimate that would arise if the same
observations built the reference and selected the threshold.

No finite-sample distribution-free guarantee is claimed: the series is not
exchangeable. Results are empirical calibration and empirical false-alarm
control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from anomaliradar.model.config import CalibrationConfig


@dataclass(frozen=True, slots=True)
class CalibrationSplit:
    reference: np.ndarray
    selection: np.ndarray
    cutoff: pd.Timestamp
    reference_rows: int
    selection_rows: int


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    p: np.ndarray
    level: np.ndarray
    n: np.ndarray
    depth: np.ndarray
    eligible: np.ndarray

    def summary(self) -> dict:
        levels = pd.Series(self.level).value_counts(dropna=False)
        return {
            "levels": {str(k): int(v) for k, v in levels.items()},
            "refused_rows": int((~self.eligible).sum()),
            "median_effective_n": float(np.median(self.n[self.eligible]))
            if self.eligible.any()
            else 0.0,
        }


@dataclass(slots=True)
class ConformalCalibrator:
    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    cell_scores: dict[str, np.ndarray] = field(default_factory=dict)
    state_scores: dict[str, np.ndarray] = field(default_factory=dict)
    global_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    reference_rows: int = 0

    # -- fitting ----------------------------------------------------------- #
    def fit(
        self,
        scores: np.ndarray,
        cell_keys: np.ndarray,
        state_keys: np.ndarray,
        reference_mask: np.ndarray,
    ) -> "ConformalCalibrator":
        scores = np.asarray(scores, dtype=float)
        cell_keys = np.asarray(cell_keys, dtype=object)
        state_keys = np.asarray(state_keys, dtype=object)
        reference_mask = np.asarray(reference_mask, dtype=bool)

        valid = reference_mask & np.isfinite(scores)
        self.reference_rows = int(valid.sum())
        if self.reference_rows == 0:
            raise ValueError("calibration reference contains no finite scores")

        def _text(value: object) -> str | None:
            if value is None or value is pd.NA:
                return None
            if isinstance(value, float) and np.isnan(value):
                return None
            text = str(value)
            return None if text in ("nan", "None", "<NA>") else text

        self.global_scores = np.sort(scores[valid])
        self.cell_scores = {}
        self.state_scores = {}
        valid_rows = np.flatnonzero(valid)
        cell_text = np.array([_text(v) for v in cell_keys[valid]], dtype=object)
        state_text = np.array([_text(v) for v in state_keys[valid]], dtype=object)
        for key in pd.unique(cell_text):
            if key is None:
                continue
            sel = valid_rows[cell_text == key]
            if sel.size:
                self.cell_scores[str(key)] = np.sort(scores[sel])
        for key in pd.unique(state_text):
            if key is None:
                continue
            sel = valid_rows[state_text == key]
            if sel.size:
                self.state_scores[str(key)] = np.sort(scores[sel])
        return self

    # -- scoring ----------------------------------------------------------- #
    @staticmethod
    def _p_value(sorted_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
        n = sorted_scores.size
        if n == 0:
            return np.ones(values.shape[0], dtype=float)
        count_ge = n - np.searchsorted(sorted_scores, values, side="left")
        return (1.0 + count_ge) / (n + 1.0)

    def _choose(
        self, cell_key: object, state_key: object
    ) -> tuple[np.ndarray | None, str, int]:
        """Hierarchical back-off decision for one (cell, state) pair."""
        n_min = self.config.n_min
        n_refuse = self.config.n_refuse
        cell_arr = (
            self.cell_scores.get(str(cell_key))
            if cell_key is not None
            else None
        )
        state_arr = (
            self.state_scores.get(str(state_key))
            if state_key is not None
            else None
        )
        if cell_arr is not None and cell_arr.size >= n_min:
            return cell_arr, "cell", 0
        if state_arr is not None and state_arr.size >= n_min:
            return state_arr, "state", 1
        if state_arr is not None and state_arr.size >= n_refuse:
            return state_arr, "state_low", 1
        if self.global_scores.size >= n_min:
            return self.global_scores, "global", 2
        if self.global_scores.size >= n_refuse:
            return self.global_scores, "global_low", 2
        return None, "refused", -1

    def p_values(
        self,
        scores: np.ndarray,
        cell_keys: np.ndarray,
        state_keys: np.ndarray,
    ) -> CalibrationResult:
        scores = np.asarray(scores, dtype=float)
        cell_keys = np.asarray(cell_keys, dtype=object)
        state_keys = np.asarray(state_keys, dtype=object)
        n_rows = scores.shape[0]

        p = np.ones(n_rows, dtype=float)
        level = np.empty(n_rows, dtype=object)
        level[:] = "refused"
        n_eff = np.zeros(n_rows, dtype=int)
        depth = np.full(n_rows, -1, dtype=int)
        eligible = np.zeros(n_rows, dtype=bool)

        finite = np.isfinite(scores)
        # Missing keys are represented as None so that a missing cell can still
        # back off to a valid state or to global.
        def key(value: object) -> object:
            if value is None:
                return None
            if isinstance(value, float) and np.isnan(value):
                return None
            if value is pd.NA:
                return None
            text = str(value)
            return None if text in ("nan", "None", "<NA>") else text

        # The hierarchy decision depends only on the (cell, state) pair, so the
        # loop runs over at most a handful of groups, not over every row.
        groups: dict[tuple[object, object], list[int]] = {}
        for i in np.flatnonzero(finite):
            groups.setdefault((key(cell_keys[i]), key(state_keys[i])), []).append(int(i))
        for (cell_key, state_key), indices in groups.items():
            chosen, level_name, chosen_depth = self._choose(cell_key, state_key)
            if chosen is None:
                continue
            rows = np.asarray(indices, dtype=int)
            p[rows] = self._p_value(chosen, scores[rows])
            level[rows] = level_name
            n_eff[rows] = chosen.size
            depth[rows] = chosen_depth
            eligible[rows] = True

        level[~finite] = "insufficient"
        return CalibrationResult(
            p=p,
            level=level,
            n=n_eff,
            depth=depth,
            eligible=eligible,
        )

    # -- diagnostics ------------------------------------------------------- #
    def cell_table(self) -> pd.DataFrame:
        rows = []
        for key, values in sorted(self.cell_scores.items()):
            rows.append({"level": "cell", "key": key, "n": int(values.size)})
        for key, values in sorted(self.state_scores.items()):
            rows.append({"level": "state", "key": key, "n": int(values.size)})
        rows.append({"level": "global", "key": "global", "n": int(self.global_scores.size)})
        return pd.DataFrame(rows, columns=["level", "key", "n"])


def chronological_split(
    calibration_positions: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    fraction: float = 2.0 / 3.0,
) -> CalibrationSplit:
    """Split calibration rows into an earlier reference and a later selection set."""
    positions = np.sort(np.asarray(calibration_positions, dtype=int))
    if positions.size < 2:
        raise ValueError("calibration window has fewer than two rows")
    cut = int(np.floor(positions.size * fraction))
    cut = min(max(cut, 1), positions.size - 1)
    ref_pos = positions[:cut]
    sel_pos = positions[cut:]
    reference = np.zeros(len(index), dtype=bool)
    selection = np.zeros(len(index), dtype=bool)
    reference[ref_pos] = True
    selection[sel_pos] = True
    cutoff = index[sel_pos[0]]
    return CalibrationSplit(
        reference=reference,
        selection=selection,
        cutoff=cutoff,
        reference_rows=int(ref_pos.size),
        selection_rows=int(sel_pos.size),
    )
