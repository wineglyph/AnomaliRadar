"""Evaluation metrics fixed by ``docs/experimental_protocol.md`` §6.

Nothing here chooses a threshold or a method. It receives frozen alarms and
returns counts, rates and calibration diagnostics. Four events are reported as
four booleans plus ``x/4``; no significance test is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from anomaliradar.data.labels import EventWindow, label_mask

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True, slots=True)
class EventOutcome:
    event_id: int
    detected: bool
    delay_minutes: float | None
    first_alarm: pd.Timestamp | None
    alarms_in_detection: int
    detection_start: pd.Timestamp
    detection_end: pd.Timestamp
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    masked_detection_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "detected": self.detected,
            "delay_minutes": self.delay_minutes,
            "first_alarm": self.first_alarm,
            "alarms_in_detection": self.alarms_in_detection,
            "detection_start": self.detection_start,
            "detection_end": self.detection_end,
            "event_start": self.event_start,
            "event_end": self.event_end,
            "masked_detection_rows": self.masked_detection_rows,
        }


def event_related_mask(
    index: pd.DatetimeIndex,
    events: Sequence[EventWindow],
    *,
    pre_hours: float = 2.0,
    post_hours: float = 6.0,
) -> np.ndarray:
    """Rows close enough to a labelled event to be excluded from false alarms."""
    index = pd.DatetimeIndex(index)
    mask = np.zeros(len(index), dtype=bool)
    for event in events:
        start = event.start - pd.Timedelta(hours=pre_hours)
        end = event.end + pd.Timedelta(hours=post_hours)
        mask |= (index >= start) & (index < end)
    return mask


def event_outcomes(
    alarm_times: pd.DatetimeIndex,
    events: Sequence[EventWindow],
    *,
    masked: np.ndarray | None = None,
    index: pd.DatetimeIndex | None = None,
) -> list[EventOutcome]:
    """Per-event detection credit and first-alarm delay."""
    alarms = pd.DatetimeIndex(alarm_times)
    outcomes: list[EventOutcome] = []
    for event in events:
        window = event.detection_window()
        in_window = (alarms >= window.start) & (alarms < window.end)
        first = alarms[in_window].min() if in_window.any() else None
        delay = (
            float((first - event.start) / pd.Timedelta(minutes=1))
            if first is not None
            else None
        )
        masked_rows = 0
        if masked is not None and index is not None:
            in_det = window.mask(pd.DatetimeIndex(index))
            masked_rows = int((in_det & np.asarray(masked, dtype=bool)).sum())
        outcomes.append(
            EventOutcome(
                event_id=event.event_id,
                detected=bool(in_window.any()),
                delay_minutes=delay,
                first_alarm=first,
                alarms_in_detection=int(in_window.sum()),
                detection_start=window.start,
                detection_end=window.end,
                event_start=event.start,
                event_end=event.end,
                masked_detection_rows=masked_rows,
            )
        )
    return outcomes


def exposure_days(index: pd.DatetimeIndex) -> float:
    """Observed exposure: one row is one observed minute (masks already removed)."""
    return float(len(index)) / MINUTES_PER_DAY


def alarm_volume(
    alarm_times: pd.DatetimeIndex,
    observed_index: pd.DatetimeIndex,
    events: Sequence[EventWindow],
    *,
    pre_hours: float = 2.0,
    post_hours: float = 6.0,
) -> dict:
    """Total and false alarms per observed day, with exposure outside events."""
    alarms = pd.DatetimeIndex(alarm_times)
    observed_index = pd.DatetimeIndex(observed_index)
    total_days = exposure_days(observed_index)
    related = event_related_mask(
        observed_index, events, pre_hours=pre_hours, post_hours=post_hours
    )
    outside_index = observed_index[~related]
    outside_days = exposure_days(outside_index)
    outside = ~event_related_mask(
        alarms, events, pre_hours=pre_hours, post_hours=post_hours
    ) if len(alarms) else np.array([], dtype=bool)
    false_alarms = int(outside.sum())
    return {
        "alarms": int(len(alarms)),
        "observed_days": total_days,
        "alarms_per_day": (len(alarms) / total_days) if total_days else 0.0,
        "false_alarms": false_alarms,
        "outside_days": outside_days,
        "false_alarms_per_day": (false_alarms / outside_days) if outside_days else 0.0,
        "alarms_inside_event_related": int(len(alarms) - false_alarms),
    }


def budget_error(realized_rate: float, budget: float) -> float:
    return (realized_rate - budget) / budget if budget else float("nan")


def calibration_cells(
    p: np.ndarray,
    cell: np.ndarray,
    nominal_alpha: float,
    *,
    outside_event: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """Per-cell empirical FPR on non-event observed rows, and max deviation."""
    p = np.asarray(p, dtype=float)
    cell = np.asarray(cell, dtype=object)
    keep = np.asarray(outside_event, dtype=bool) & np.isfinite(p)
    rows = []
    for key in sorted({str(k) for k in cell[keep]}):
        selected = keep & np.array([str(k) == key for k in cell], dtype=bool)
        n = int(selected.sum())
        alarms = int((p[selected] <= nominal_alpha).sum())
        fpr = alarms / n if n else 0.0
        rows.append(
            {
                "cell": key,
                "n": n,
                "alarms": alarms,
                "empirical_fpr": fpr,
                "nominal_alpha": nominal_alpha,
                "deviation": abs(fpr - nominal_alpha),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=["cell", "n", "alarms", "empirical_fpr", "nominal_alpha", "deviation"],
    )
    max_deviation = float(frame["deviation"].max()) if len(frame) else 0.0
    return frame, max_deviation


def spearman_scores(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation on rows where both scores are finite."""
    a = pd.Series(np.asarray(a, dtype=float))
    b = pd.Series(np.asarray(b, dtype=float))
    keep = a.notna() & b.notna()
    if int(keep.sum()) < 3:
        return float("nan")
    return float(a[keep].corr(b[keep], method="spearman"))


def pointwise_diagnostics(
    score: np.ndarray,
    index: pd.DatetimeIndex,
    events: Sequence[EventWindow],
) -> dict:
    """Secondary ROC-AUC / PR-AUC, explicitly a diagnostic only."""
    labels = label_mask(pd.DatetimeIndex(index), events).astype(int)
    score = np.asarray(score, dtype=float)
    keep = np.isfinite(score)
    y = labels[keep]
    s = score[keep]
    if y.min() == y.max():
        return {"roc_auc": None, "pr_auc": None, "positive_rows": int(y.sum())}
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "positive_rows": int(y.sum()),
        "positive_share": float(y.mean()),
    }


def summarise_alarms(
    alarms: pd.DataFrame,
    observed_index: pd.DatetimeIndex,
    events: Sequence[EventWindow],
    *,
    budget: float,
    n_events: int | None = 4,
) -> dict:
    """One row of the primary results table."""
    times = pd.DatetimeIndex(alarms["timestamp"]) if len(alarms) else pd.DatetimeIndex([])
    volume = alarm_volume(times, observed_index, events)
    outcomes = event_outcomes(times, events)
    recall = sum(int(o.detected) for o in outcomes)
    recall_label = f"{recall}/{n_events}" if n_events is not None else "n/a"
    return {
        "budget": budget,
        "event_recall": recall_label,
        "event_recall_count": recall if n_events is not None else None,
        "detected": [o.detected for o in outcomes],
        "delays_minutes": [o.delay_minutes for o in outcomes],
        "alarms": volume["alarms"],
        "observed_days": volume["observed_days"],
        "alarms_per_day": volume["alarms_per_day"],
        "false_alarms": volume["false_alarms"],
        "false_alarms_per_day": volume["false_alarms_per_day"],
        "outside_days": volume["outside_days"],
        "realized_budget": volume["alarms_per_day"],
        "budget_error": budget_error(volume["alarms_per_day"], budget),
        **{f"event_{o.event_id}": o.detected for o in outcomes},
        **{f"delay_{o.event_id}": o.delay_minutes for o in outcomes},
    }


@dataclass(slots=True)
class EvaluationAccumulator:
    rows: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    cells: list[dict] = field(default_factory=list)

    def add(
        self,
        *,
        fold: str,
        method: str,
        budget: float,
        alarms: pd.DataFrame,
        observed_index: pd.DatetimeIndex,
        events: Sequence[EventWindow],
        p: np.ndarray,
        cell: np.ndarray,
        outside_event: np.ndarray,
        alpha: float,
        calibration: dict,
        extra: dict | None = None,
        n_events: int | None = 4,
    ) -> None:
        summary = summarise_alarms(
            alarms, observed_index, events, budget=budget, n_events=n_events
        )
        alarm_times = (
            pd.DatetimeIndex(alarms["timestamp"])
            if len(alarms)
            else pd.DatetimeIndex([])
        )
        frame, max_deviation = calibration_cells(
            p, cell, alpha, outside_event=outside_event
        )
        base = {
            "fold": fold,
            "method": method,
            **summary,
            "alpha": alpha,
            "max_cell_deviation": max_deviation,
            **{f"cal_{k}": v for k, v in calibration.items()},
        }
        if extra:
            base.update(extra)
        self.rows.append(base)

        for outcome in event_outcomes(alarm_times, events):
            self.events.append(
                {"fold": fold, "method": method, "budget": budget, **outcome.to_dict()}
            )

        if len(frame):
            frame = frame.copy()
            frame.insert(0, "fold", fold)
            frame.insert(1, "method", method)
            frame.insert(2, "budget", budget)
            self.cells.append(frame)

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rows = pd.DataFrame(self.rows)
        events = pd.DataFrame(self.events)
        cells = (
            pd.concat(self.cells, ignore_index=True)
            if self.cells
            else pd.DataFrame()
        )
        return rows, events, cells
