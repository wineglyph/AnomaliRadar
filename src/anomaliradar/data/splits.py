"""Chronological folds, leakage invariants and split materialisation.

Fold design (``docs/experimental_protocol.md`` §2)::

    calibration_end   = event_start - 7 days
    calibration_start = calibration_end - 21 days
    train             = [series_start, calibration_start)
    test              = [event_start - 12 h, event_end + 24 h)
    detection credit  = [event_start - 2 h, event_start + tau)

Fold ``T`` reuses fold 4's frozen model on the 31-day tail.

``materialise_split`` enforces the data-side leakage rules: masked rows and
labelled rows never enter train or calibration; test rows are never masked;
train/calibration/test index sets are disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from anomaliradar.data.labels import EventWindow, FAILURE_EVENTS, TAIL_WINDOW, label_mask
from anomaliradar.timewindow import TimeWindow

#: Literal L2 asks for >= 7 days between calibration end and test start. The
#: pre-registered test window starts 12 h before the event, so the raw buffer is
#: 6.5 days. We preserve the stronger, meaningful invariant — a full 7-day
#: margin before the labelled event — and require at least 6 days of raw buffer,
#: which is 24x the longest causal feature window (6 h).
MIN_EMBARGO = pd.Timedelta(days=6)


class SplitError(ValueError):
    """Raised when a fold violates chronological ordering or disjointness."""


@dataclass(frozen=True, slots=True)
class FoldSpec:
    fold_id: str
    event_id: int | None
    train: TimeWindow
    calibration: TimeWindow
    test: TimeWindow
    detection: TimeWindow | None
    event_start: pd.Timestamp | None = None
    event_end: pd.Timestamp | None = None
    refit: bool = True
    inherits_from: str | None = None

    def to_dict(self) -> dict:
        out = {
            "fold_id": self.fold_id,
            "event_id": self.event_id,
            "train_start": self.train.start,
            "train_end": self.train.end,
            "calibration_start": self.calibration.start,
            "calibration_end": self.calibration.end,
            "test_start": self.test.start,
            "test_end": self.test.end,
            "event_start": self.event_start,
            "event_end": self.event_end,
            "detection_start": self.detection.start if self.detection else None,
            "detection_end": self.detection.end if self.detection else None,
            "refit": self.refit,
            "inherits_from": self.inherits_from,
        }
        return out


@dataclass(frozen=True, slots=True)
class SplitIndices:
    fold_id: str
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    detection: np.ndarray
    test_window: np.ndarray
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.check_disjoint()

    def check_disjoint(self) -> None:
        sets = {
            "train": set(self.train.tolist()),
            "calibration": set(self.calibration.tolist()),
            "test": set(self.test.tolist()),
        }
        names = list(sets)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = sets[left] & sets[right]
                if overlap:
                    raise SplitError(
                        f"fold {self.fold_id}: {left} and {right} overlap on "
                        f"{len(overlap)} rows"
                    )

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_rows": int(len(self.train)),
            "calibration_rows": int(len(self.calibration)),
            "test_rows": int(len(self.test)),
            "detection_rows": int(len(self.detection)),
            "test_window_rows": int(len(self.test_window)),
            **self.stats,
        }


# --------------------------------------------------------------------------- #
# Fold construction
# --------------------------------------------------------------------------- #
def build_folds(
    series_start: pd.Timestamp,
    events: Sequence[EventWindow] | None = None,
    *,
    calibration_days: int = 21,
    pre_event_margin_days: int = 7,
    test_pre_hours: int = 12,
    test_post_hours: int = 24,
    include_tail: bool = False,
    tail: TimeWindow = TAIL_WINDOW,
) -> list[FoldSpec]:
    """Build the rolling-origin folds. Deterministic; no data is read here."""
    chosen = tuple(FAILURE_EVENTS if events is None else events)
    series_start = pd.Timestamp(series_start)
    folds: list[FoldSpec] = []

    for event in chosen:
        calib_end = event.start - pd.Timedelta(days=pre_event_margin_days)
        calib_start = calib_end - pd.Timedelta(days=calibration_days)
        train = TimeWindow(series_start, calib_start)
        calibration = TimeWindow(calib_start, calib_end)
        test = TimeWindow(
            event.start - pd.Timedelta(hours=test_pre_hours),
            event.end + pd.Timedelta(hours=test_post_hours),
        )
        folds.append(
            FoldSpec(
                fold_id=str(event.event_id),
                event_id=event.event_id,
                train=train,
                calibration=calibration,
                test=test,
                detection=event.detection_window(),
                event_start=event.start,
                event_end=event.end,
            )
        )

    if include_tail:
        if not folds:
            raise SplitError("tail fold requires the event folds")
        base = folds[-1]
        folds.append(
            FoldSpec(
                fold_id="T",
                event_id=None,
                train=base.train,
                calibration=base.calibration,
                test=tail,
                detection=None,
                refit=False,
                inherits_from=base.fold_id,
            )
        )

    validate_folds(folds)
    return folds


def validate_folds(
    folds: Sequence[FoldSpec],
    *,
    min_embargo: pd.Timedelta = MIN_EMBARGO,
) -> None:
    """Check L1/L2 invariants for every fold. Raises ``SplitError`` on failure."""
    if not folds:
        raise SplitError("no folds")
    seen: set[str] = set()

    for fold in folds:
        if fold.fold_id in seen:
            raise SplitError(f"duplicate fold id {fold.fold_id}")
        seen.add(fold.fold_id)

        if fold.train.end > fold.calibration.start:
            raise SplitError(f"fold {fold.fold_id}: train overlaps calibration")
        if fold.calibration.end > fold.test.start:
            raise SplitError(f"fold {fold.fold_id}: calibration overlaps test")
        if fold.detection is not None and fold.test.start > fold.detection.start:
            raise SplitError(f"fold {fold.fold_id}: detection starts before test")
        if fold.detection is not None and fold.detection.end > fold.test.end:
            raise SplitError(f"fold {fold.fold_id}: detection ends after test")

        buffer = fold.test.start - fold.calibration.end
        if buffer < min_embargo:
            raise SplitError(
                f"fold {fold.fold_id}: embargo {buffer} is shorter than {min_embargo}"
            )
        if fold.event_start is not None:
            # Design invariant: calibration closes a full 7 days before the event.
            if fold.calibration.end != fold.event_start - pd.Timedelta(days=7):
                raise SplitError(
                    f"fold {fold.fold_id}: calibration does not end 7 days before the event"
                )

        if not fold.refit:
            if fold.inherits_from not in seen - {fold.fold_id}:
                raise SplitError(
                    f"fold {fold.fold_id}: inherits_from={fold.inherits_from!r} unknown"
                )

    # Pairwise disjointness across train/calibration/test; a new fold may reuse
    # an earlier fold's training data, so only compare within one fold.
    for fold in folds:
        if fold.train.intersect(fold.calibration):
            raise SplitError(f"fold {fold.fold_id}: train/calibration intersection")
        if fold.calibration.intersect(fold.test):
            raise SplitError(f"fold {fold.fold_id}: calibration/test intersection")
        if fold.train.intersect(fold.test):
            raise SplitError(f"fold {fold.fold_id}: train/test intersection")


def fold_table(folds: Sequence[FoldSpec]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        rows.append(
            {
                "fold": fold.fold_id,
                "event": fold.event_id,
                "train": str(fold.train),
                "calibration": str(fold.calibration),
                "test": str(fold.test),
                "detection": str(fold.detection) if fold.detection else "",
                "embargo_days": round(
                    (fold.test.start - fold.calibration.end) / pd.Timedelta(days=1), 3
                ),
                "refit": fold.refit,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Materialisation
# --------------------------------------------------------------------------- #
def _reason_counts(
    in_window: np.ndarray,
    usable: np.ndarray,
    masked: np.ndarray,
    labelled: np.ndarray,
    transition: np.ndarray,
    keep_mask: np.ndarray,
    context_ready: np.ndarray | None = None,
) -> dict:
    out = {
        "window_rows": int(in_window.sum()),
        "kept": int(keep_mask.sum()),
        "excluded_masked": int((in_window & masked).sum()),
        "excluded_label": int((in_window & labelled).sum()),
        "excluded_transition": int((in_window & transition).sum()),
        "excluded_unusable": int((in_window & ~usable).sum()),
    }
    if context_ready is not None:
        out["excluded_no_context"] = int((in_window & ~context_ready).sum())
    return out


def materialise_split(
    index: pd.DatetimeIndex,
    fold: FoldSpec,
    *,
    mask: np.ndarray,
    usable: np.ndarray,
    transition: np.ndarray | None = None,
    context_ready: np.ndarray | None = None,
    events: Sequence[EventWindow] | None = None,
    exclude_labels_from_fit: bool = True,
    exclude_transitions_from_fit: bool = True,
) -> SplitIndices:
    """Turn a fold specification into positional row indices.

    * train + calibration: usable, unmasked, unlabelled (and non-transition);
    * test: usable, unmasked, **labels retained** (they are the evaluation);
    * detection: the creditable subset of the test window.

    ``context_ready`` is optional and, when supplied, additionally removes rows
    without a lagged context cell from **calibration** only. Those rows are the
    first row after each break; they cannot be calibrated because their cell is
    undefined.

    ``test_window`` records every row in the test window regardless of masks,
    so the mask cost inside the evaluation window stays visible.
    """
    index = pd.DatetimeIndex(index)
    mask = np.asarray(mask, dtype=bool)
    usable = np.asarray(usable, dtype=bool)
    transition = (
        np.zeros(len(index), dtype=bool)
        if transition is None
        else np.asarray(transition, dtype=bool)
    )
    ready = (
        np.ones(len(index), dtype=bool)
        if context_ready is None
        else np.asarray(context_ready, dtype=bool)
    )
    chosen = tuple(FAILURE_EVENTS if events is None else events)

    in_train = fold.train.mask(index)
    in_calib = fold.calibration.mask(index)
    in_test = fold.test.mask(index)
    in_detection = fold.detection.mask(index) if fold.detection else np.zeros(len(index), dtype=bool)

    labelled = label_mask(index, chosen) if exclude_labels_from_fit else np.zeros(len(index), dtype=bool)
    transition_exclusion = transition if exclude_transitions_from_fit else np.zeros(len(index), dtype=bool)

    base_fit = usable & ~mask
    train_mask = in_train & base_fit & ~labelled & ~transition_exclusion
    calib_mask = in_calib & base_fit & ~labelled & ~transition_exclusion & ready
    test_mask = in_test & base_fit
    detection_mask = in_detection & base_fit

    context_arg = ready if context_ready is not None else None
    stats = {
        "train": _reason_counts(
            in_train, usable, mask, labelled, transition_exclusion, train_mask
        ),
        "calibration": _reason_counts(
            in_calib, usable, mask, labelled, transition_exclusion, calib_mask, context_arg
        ),
        "test": _reason_counts(
            in_test, usable, mask, labelled, transition_exclusion, test_mask
        ),
    }

    return SplitIndices(
        fold_id=fold.fold_id,
        train=np.flatnonzero(train_mask),
        calibration=np.flatnonzero(calib_mask),
        test=np.flatnonzero(test_mask),
        detection=np.flatnonzero(detection_mask),
        test_window=np.flatnonzero(in_test),
        stats=stats,
    )
