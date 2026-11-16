"""Score B — state-persistence anomaly.

Primary statistic (``docs/methodology.md`` §6.1): for the current confirmed state
``s`` and run length ``r``, the empirical survival of completed calibration runs

    b = (1 + #{d in D_s : d >= r}) / (|D_s| + 1)

Small ``b`` means the state has persisted longer than calibration suggests.
``b`` is a nonconformity score, not a p-value: at an arbitrary time you are more
likely to be inside a long run than a short one (length-biased sampling), so it
is calibrated in the same Mondrian layer as Score A.

Calibration hygiene:

* reference runs must be **completed** and lie wholly inside the reference part
  of the calibration window; censored runs at a segment end are excluded;
* when scoring a reference row, its own run is removed from ``D_s`` (leave one
  run out), because otherwise the reference distribution would contain the point
  being scored;
* test rows are not in ``D_s`` and use the full reference distribution.

B3 (discrete-time hazard) is provided as a pre-registered ablation. B2 (cycle
period deviation) is not implemented: it needs a stable cycle-period estimator,
and the data contain many missing cycles, so it cannot be run without adding a
fragile component to the primary experiment (documented deviation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaliradar.model.config import ScoreBConfig

CALIBRATABLE_STATES: tuple[str, ...] = ("off", "offloaded", "loaded")


@dataclass(frozen=True, slots=True)
class RunTable:
    frame: pd.DataFrame

    @property
    def completed(self) -> pd.DataFrame:
        return self.frame.loc[self.frame["completed"]]


@dataclass(frozen=True, slots=True)
class ScoreBReference:
    variant: str
    state_lengths: dict[str, np.ndarray]
    pooled_lengths: np.ndarray
    state_hazard: dict[str, tuple[np.ndarray, np.ndarray]]
    pooled_hazard: tuple[np.ndarray, np.ndarray]
    source_runs: int

    def level_for(self, state: object, config: ScoreBConfig) -> tuple[str, np.ndarray]:
        if isinstance(state, str) and state in self.state_lengths:
            lengths = self.state_lengths[state]
            if lengths.size >= config.min_reference_runs:
                return "state", lengths
        if self.pooled_lengths.size >= config.min_reference_runs:
            return "pooled", self.pooled_lengths
        return "insufficient", np.array([], dtype=float)


def build_run_table(state_frame: pd.DataFrame) -> RunTable:
    """Extract maximal confirmed-state runs and mark whether each is completed."""
    state = state_frame["state"]
    segment = state_frame["segment_id"]
    n = len(state_frame)
    new_run = (
        state.ne(state.shift())
        | segment.ne(segment.shift())
        | state.isna()
        | state.shift().isna()
    )
    run_id = new_run.cumsum().to_numpy()
    state_values = state.to_numpy(dtype=object)
    seg_values = segment.to_numpy()

    rows = []
    positions = np.arange(n)
    valid = ~pd.isna(state_values)
    for rid in pd.unique(run_id[valid]):
        pos = positions[(run_id == rid) & valid]
        if pos.size == 0:
            continue
        start, end = int(pos[0]), int(pos[-1])
        run_state = state_values[start]
        length = end - start + 1
        next_pos = end + 1
        completed = bool(
            next_pos < n
            and seg_values[next_pos] == seg_values[end]
            and not pd.isna(state_values[next_pos])
            and state_values[next_pos] != run_state
        )
        rows.append(
            {
                "run_id": int(rid),
                "state": str(run_state),
                "start": start,
                "end": end,
                "length": int(length),
                "segment_id": int(seg_values[start]),
                "completed": completed,
            }
        )
    return RunTable(
        frame=pd.DataFrame(
            rows,
            columns=[
                "run_id",
                "state",
                "start",
                "end",
                "length",
                "segment_id",
                "completed",
            ],
        )
    )


def _hazard(lengths: np.ndarray, smoothing: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time exit hazard and survival for lengths ``1..max``."""
    lengths = np.asarray(lengths, dtype=float)
    if lengths.size == 0:
        return np.array([1.0]), np.array([1.0])
    max_len = int(np.max(lengths))
    risk = np.arange(1, max_len + 1)
    at_risk = np.array([(lengths >= r).sum() for r in risk], dtype=float)
    exits = np.array([(lengths == r).sum() for r in risk], dtype=float)
    hazard = (exits + smoothing) / (at_risk + 2.0 * smoothing)
    survival = np.ones(max_len + 1, dtype=float)
    for r in range(1, max_len + 1):
        survival[r] = survival[r - 1] * (1.0 - hazard[r - 1])
    survival = np.clip(survival, 1e-12, 1.0)
    return risk, survival


def reference_run_rows(runs: RunTable, reference_mask: np.ndarray) -> np.ndarray:
    """Rows belonging to completed reference runs wholly inside the reference.

    These are the only Score B rows with a well-defined leave-one-run-out
    nonconformity. Rows from runs censored by the reference boundary are scored
    against a distribution that excludes their own run, which would make them
    artificial outliers; they are therefore not used to build the conformal
    reference (they remain scored as test/selection rows).
    """
    reference_mask = np.asarray(reference_mask, dtype=bool)
    mask = np.zeros(reference_mask.shape[0], dtype=bool)
    for row in runs.frame.itertuples(index=False):
        if not row.completed or row.state not in CALIBRATABLE_STATES:
            continue
        start, end = int(row.start), int(row.end)
        if reference_mask[start : end + 1].all():
            mask[start : end + 1] = True
    return mask


def fit_score_b_reference(
    runs: RunTable,
    reference_mask: np.ndarray,
    config: ScoreBConfig | None = None,
) -> ScoreBReference:
    """Build the state-specific completed-run distributions from reference rows."""
    cfg = config or ScoreBConfig()
    reference_mask = np.asarray(reference_mask, dtype=bool)
    table = runs.frame
    usable = table["completed"] & table["state"].isin(CALIBRATABLE_STATES)
    mask = np.zeros(len(table), dtype=bool)
    for i, row in table.iterrows():
        if not usable.iloc[i]:
            continue
        start, end = int(row["start"]), int(row["end"])
        mask[i] = bool(reference_mask[start : end + 1].all())

    included = table.loc[mask]
    state_lengths: dict[str, np.ndarray] = {}
    hazard: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for state, group in included.groupby("state"):
        lengths = np.sort(group["length"].to_numpy(dtype=float))
        state_lengths[str(state)] = lengths
        hazard[str(state)] = _hazard(lengths, cfg.hazard_smoothing)

    pooled = np.sort(included["length"].to_numpy(dtype=float))
    return ScoreBReference(
        variant=cfg.variant,
        state_lengths=state_lengths,
        pooled_lengths=pooled,
        state_hazard=hazard,
        pooled_hazard=_hazard(pooled, cfg.hazard_smoothing),
        source_runs=int(len(included)),
    )


def _survival_b1(lengths: np.ndarray, run_length: np.ndarray) -> np.ndarray:
    """Empirical survival ``(1 + count(d >= r)) / (n + 1)``."""
    if lengths.size == 0:
        return np.full(run_length.shape[0], np.nan)
    sorted_lengths = np.sort(lengths)
    count_ge = lengths.size - np.searchsorted(sorted_lengths, run_length, side="left")
    return (1.0 + count_ge) / (lengths.size + 1.0)


def _survival_b3(
    risk: np.ndarray,
    survival: np.ndarray,
    run_length: np.ndarray,
) -> np.ndarray:
    """Piecewise-constant survival from the discrete hazard, tail at the last value."""
    idx = np.clip(run_length.astype(int), 1, len(survival) - 1)
    out = survival[idx]
    beyond = run_length > risk[-1]
    if np.any(beyond):
        # Continue with the final hazard estimate rather than assuming survival 0.
        tail_hazard = 1.0 - survival[-1] / survival[-2] if len(survival) > 1 else 1.0
        tail_hazard = float(np.clip(tail_hazard, 1e-9, 1.0))
        extra = np.power(1.0 - tail_hazard, run_length[beyond] - risk[-1])
        out = out.astype(float)
        out[beyond] = survival[-1] * extra
    return np.clip(out, 1e-12, 1.0)


def score_b_series(
    state_frame: pd.DataFrame,
    reference: ScoreBReference,
    config: ScoreBConfig | None = None,
    *,
    loo_mask: np.ndarray | None = None,
    runs: RunTable | None = None,
) -> pd.DataFrame:
    """Score every row with the calibrated-state survival statistic.

    ``loo_mask`` marks reference rows whose own run is in the reference
    distribution; those rows get the leave-one-run-out value. Test rows pass
    ``None``.
    """
    cfg = config or ScoreBConfig()
    state = state_frame["state"]
    run_length = pd.to_numeric(state_frame["run_length_min"], errors="coerce").to_numpy(
        dtype=float
    )
    state_values = state.to_numpy(dtype=object)
    n = len(state_frame)

    out = pd.DataFrame(index=state_frame.index)
    b = np.full(n, np.nan, dtype=float)
    score = np.full(n, np.nan, dtype=float)
    level = np.empty(n, dtype=object)
    level[:] = "insufficient"
    ref_n = np.zeros(n, dtype=int)

    loo = (
        np.zeros(n, dtype=bool)
        if loo_mask is None
        else np.asarray(loo_mask, dtype=bool)
    )
    run_of_row = np.full(n, -1, dtype=int)
    run_in_reference: set[int] = set()
    if runs is not None:
        table = runs.frame
        included = table["completed"] & table["state"].isin(CALIBRATABLE_STATES)
        for _, row in table.iterrows():
            start, end = int(row["start"]), int(row["end"])
            run_of_row[start : end + 1] = int(row["run_id"])
            if included.loc[row.name] and loo[start : end + 1].all():
                run_in_reference.add(int(row["run_id"]))

    for state_name in CALIBRATABLE_STATES:
        positions = np.flatnonzero(state_values == state_name)
        if positions.size == 0:
            continue
        level_name, lengths = reference.level_for(state_name, cfg)
        if level_name == "insufficient":
            continue
        r = run_length[positions]
        if cfg.variant == "B1":
            s = _survival_b1(lengths, r)
        elif cfg.variant == "B3":
            risk, survival = (
                reference.state_hazard[state_name]
                if level_name == "state"
                else reference.pooled_hazard
            )
            s = _survival_b3(risk, survival, r)
        else:
            raise ValueError(f"unknown Score B variant {cfg.variant!r}")

        # Leave one run out for reference rows: remove this row's own run from
        # the empirical distribution. For B1 this is exact; for B3 the hazard is
        # rebuilt from all reference runs, so B3 is reported as an ablation with
        # this small in-sample optimism documented.
        if cfg.variant == "B1" and run_in_reference:
            own = np.array(
                [
                    loo[p] and int(run_of_row[p]) in run_in_reference
                    for p in positions
                ],
                dtype=bool,
            )
            if own.any():
                sorted_lengths = np.sort(lengths)
                count_ge = lengths.size - np.searchsorted(
                    sorted_lengths, r, side="left"
                )
                # Full b = (1 + count_ge) / (n + 1). Removing the scored run
                # subtracts one from both the count and n:
                #   (1 + count_ge - 1) / ((n - 1) + 1) = count_ge / n.
                loo_s = count_ge / max(lengths.size, 1)
                s = np.where(own, loo_s, s)

        b[positions] = s
        score[positions] = -np.log(np.clip(s, 1e-12, 1.0))
        level[positions] = level_name
        ref_n[positions] = lengths.size

    out["b"] = b
    out["score_b"] = score
    out["ref_level"] = level
    out["ref_n"] = ref_n
    return out
