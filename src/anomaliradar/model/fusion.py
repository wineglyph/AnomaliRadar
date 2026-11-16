"""Fusion of the two calibrated scores.

The budget fixes the alarm count, so fusion is a **ranking** problem. The
pre-registered primary rule is ``OR`` = ``min(pA, pB)``. A Bonferroni-corrected
column is also produced for reporting; because it is a monotone rescaling it
does not change the alarm set at a fixed budget except where it clips at 1.

Missing or refused p-values (``p=1`` and ``eligible=False``) are handled
explicitly:

* ``or``: use the available score (union semantics);
* ``and``: a refused side cannot confirm, so the fused value is 1;
* ``weighted``: a refused side enters as the least anomalous value, 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FUSION_LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True, slots=True)
class FusionResult:
    p: np.ndarray
    fired_by: np.ndarray
    rule: str
    lambda_: float | None = None


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def fuse(
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    rule: str,
    lambda_: float = 0.5,
    eligible_a: np.ndarray | None = None,
    eligible_b: np.ndarray | None = None,
) -> FusionResult:
    """Fuse two calibrated p-value vectors.

    Parameters
    ----------
    rule:
        ``A``, ``B``, ``or``, ``or_bonferroni``, ``and`` or ``weighted``.
    lambda_:
        Weight on Score A for ``weighted``. Fixed at 0.5 for the primary run.
    eligible_a / eligible_b:
        Optional masks; an ineligible p-value is treated as ``1``.
    """
    a = _as_float(p_a).copy()
    b = _as_float(p_b).copy()
    if eligible_a is not None:
        a = np.where(np.asarray(eligible_a, dtype=bool) & np.isfinite(a), a, 1.0)
    if eligible_b is not None:
        b = np.where(np.asarray(eligible_b, dtype=bool) & np.isfinite(b), b, 1.0)
    a = np.where(np.isfinite(a), a, 1.0)
    b = np.where(np.isfinite(b), b, 1.0)
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    fired = np.where(a <= b, "A", "B").astype(object)

    if rule == "A":
        return FusionResult(p=a, fired_by=np.full(a.shape[0], "A", dtype=object), rule=rule)
    if rule == "B":
        return FusionResult(p=b, fired_by=np.full(b.shape[0], "B", dtype=object), rule=rule)
    if rule == "or":
        return FusionResult(p=np.minimum(a, b), fired_by=fired, rule=rule)
    if rule == "or_bonferroni":
        return FusionResult(
            p=np.minimum(1.0, 2.0 * np.minimum(a, b)),
            fired_by=fired,
            rule=rule,
        )
    if rule == "and":
        return FusionResult(p=np.maximum(a, b), fired_by=fired, rule=rule)
    if rule == "weighted":
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must lie in [0, 1]")
        return FusionResult(
            p=lambda_ * a + (1.0 - lambda_) * b,
            fired_by=fired,
            rule=rule,
            lambda_=lambda_,
        )
    raise ValueError(f"unknown fusion rule {rule!r}")


def weighted_grid(
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    lambdas: tuple[float, ...] = FUSION_LAMBDAS,
    eligible_a: np.ndarray | None = None,
    eligible_b: np.ndarray | None = None,
) -> dict[float, FusionResult]:
    """Evaluate the pre-registered weighted grid. Selection is reported, not hidden."""
    return {
        lam: fuse(
            p_a,
            p_b,
            rule="weighted",
            lambda_=lam,
            eligible_a=eligible_a,
            eligible_b=eligible_b,
        )
        for lam in lambdas
    }
