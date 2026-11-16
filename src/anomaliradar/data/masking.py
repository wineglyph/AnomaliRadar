"""Label-free data-quality masks M1–M3.

| Mask | Rule |
| --- | --- |
| M1 | maximal runs where **all 7 analog channels** are constant for at least ``frozen_min_rows`` consecutive native rows (default 360 ≈ 1 h) |
| M2 | native timestamp gaps **longer than** ``major_gap`` (default 1 h) |
| M3 | the first ``warmup`` (default 30 min) of the grid after every M1 block or M2 gap |

Every rule is parameter-free with respect to the data and uses **no labels**.
The masks are therefore retrospective quality filters, not causal features, and
they may be computed once on the whole series without leaking label or test
information. This is why L4 is implemented as "mask is a pure function of the
sensor frame", not "mask fitted on train".

Masked rows are excluded from fitting, calibration, and both the numerator and
the denominator of alarm rates. They are never imputed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from anomaliradar.data.grid import MinuteGrid
from anomaliradar.data.labels import EventWindow, FAILURE_EVENTS
from anomaliradar.data.schema import ANALOG, TIMESTAMP, SchemaError
from anomaliradar.timewindow import TimeWindow


@dataclass(frozen=True, slots=True)
class MaskConfig:
    frozen_min_rows: int = 360          # ≈ 1 h at 10 s sampling
    frozen_channels: tuple[str, ...] = ANALOG
    major_gap: pd.Timedelta = pd.Timedelta(hours=1)
    warmup: pd.Timedelta = pd.Timedelta(minutes=30)

    def __post_init__(self) -> None:
        if self.frozen_min_rows < 1:
            raise ValueError("frozen_min_rows must be >= 1")
        if self.major_gap <= pd.Timedelta(0):
            raise ValueError("major_gap must be positive")
        if self.warmup < pd.Timedelta(0):
            raise ValueError("warmup must be non-negative")


@dataclass(frozen=True, slots=True)
class NativeMasks:
    frozen_blocks: pd.DataFrame
    major_gaps: pd.DataFrame
    native_rows: int

    def summary(self) -> dict:
        frozen_rows = int(self.frozen_blocks["rows"].sum()) if len(self.frozen_blocks) else 0
        gap_hours = float(self.major_gaps["hours"].sum()) if len(self.major_gaps) else 0.0
        return {
            "native_rows": self.native_rows,
            "m1_blocks": int(len(self.frozen_blocks)),
            "m1_native_rows": frozen_rows,
            "m1_native_share": frozen_rows / self.native_rows if self.native_rows else 0.0,
            "m1_max_hours": float(self.frozen_blocks["hours"].max()) if len(self.frozen_blocks) else 0.0,
            "m2_gaps": int(len(self.major_gaps)),
            "m2_gap_hours": gap_hours,
        }


@dataclass(frozen=True, slots=True)
class GridMasks:
    m1: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    m3_after_frozen: np.ndarray
    m3_after_gap: np.ndarray
    index: pd.DatetimeIndex

    @property
    def combined(self) -> np.ndarray:
        return self.m1 | self.m2 | self.m3

    def __len__(self) -> int:
        return len(self.m1)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "m1_frozen": self.m1,
                "m2_gap": self.m2,
                "m3_warmup": self.m3,
                "combined": self.combined,
            },
            index=self.index,
        )

    def signature(self) -> str:
        return array_signature(m1=self.m1, m2=self.m2, m3=self.m3)


def array_signature(**arrays: np.ndarray) -> str:
    """Stable SHA-256 over named boolean arrays (used for L4 assertions)."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.asarray(arrays[name], dtype=np.uint8)
        digest.update(name.encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Native-row detection
# --------------------------------------------------------------------------- #
def detect_frozen_blocks(frame: pd.DataFrame, config: MaskConfig | None = None) -> pd.DataFrame:
    """M1: maximal all-analog-constant runs of at least ``frozen_min_rows``."""
    cfg = config or MaskConfig()
    missing = [c for c in cfg.frozen_channels if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    if frame.empty:
        return pd.DataFrame(columns=["start", "end", "rows", "hours"])

    same = np.ones(len(frame), dtype=bool)
    same[0] = False
    for col in cfg.frozen_channels:
        values = frame[col].to_numpy()
        same[1:] &= values[1:] == values[:-1]

    run_start = np.flatnonzero(~same)
    run_id = np.cumsum(~same)
    sizes = np.bincount(run_id)[1:]  # run ids start at 1

    keep = sizes >= cfg.frozen_min_rows
    if not keep.any():
        return pd.DataFrame(columns=["start", "end", "rows", "hours"])

    timestamps = frame[TIMESTAMP].to_numpy()
    rows = []
    for run_no in np.flatnonzero(keep):
        start_pos = int(run_start[run_no])
        size = int(sizes[run_no])
        end_pos = start_pos + size - 1
        rows.append(
            {
                "start": pd.Timestamp(timestamps[start_pos]),
                "end": pd.Timestamp(timestamps[end_pos]),
                "rows": size,
                # A run of N samples at 10 s spans (N-1) intervals, but Day 0
                # reported N rows as N×10 s; keep that convention for continuity.
                "hours": round(size * 10.0 / 3600.0, 4),
            }
        )
    return pd.DataFrame(rows, columns=["start", "end", "rows", "hours"])


def detect_major_gaps(frame: pd.DataFrame, config: MaskConfig | None = None) -> pd.DataFrame:
    """M2: native intervals strictly longer than ``major_gap``."""
    cfg = config or MaskConfig()
    if len(frame) < 2:
        return pd.DataFrame(columns=["start", "end", "hours", "missing_minutes"])

    ts = frame[TIMESTAMP]
    delta = ts.diff()
    hits = np.flatnonzero((delta > cfg.major_gap).to_numpy())
    rows = []
    for pos in hits:
        gap = delta.iloc[pos]
        rows.append(
            {
                "start": pd.Timestamp(ts.iloc[pos - 1]),
                "end": pd.Timestamp(ts.iloc[pos]),
                "hours": round(gap.total_seconds() / 3600.0, 6),
                "missing_minutes": int(gap / pd.Timedelta(minutes=1)) - 1,
            }
        )
    return pd.DataFrame(rows, columns=["start", "end", "hours", "missing_minutes"])


def detect_native_masks(frame: pd.DataFrame, config: MaskConfig | None = None) -> NativeMasks:
    cfg = config or MaskConfig()
    return NativeMasks(
        frozen_blocks=detect_frozen_blocks(frame, cfg),
        major_gaps=detect_major_gaps(frame, cfg),
        native_rows=len(frame),
    )


# --------------------------------------------------------------------------- #
# Grid projection
# --------------------------------------------------------------------------- #
def _rows_between(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Rows with ``start <= ts <= end`` (both timestamps are native samples)."""
    return np.asarray((index >= start) & (index <= end), dtype=bool)


def _rows_after_until(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Rows with ``start <= ts < end``."""
    return np.asarray((index >= start) & (index < end), dtype=bool)


def build_grid_masks(
    grid: MinuteGrid,
    native_masks: NativeMasks,
    config: MaskConfig | None = None,
) -> GridMasks:
    """Project M1/M2 to the minute grid and derive M3 warm-up."""
    cfg = config or MaskConfig()
    index = grid.index
    n = len(index)

    m1 = np.zeros(n, dtype=bool)
    m3_after_frozen = np.zeros(n, dtype=bool)
    for _, block in native_masks.frozen_blocks.iterrows():
        block_start = pd.Timestamp(block["start"]).floor("1min")
        block_end = pd.Timestamp(block["end"]).floor("1min")
        m1 |= _rows_between(index, block_start, block_end)
        # Warm-up begins on the first minute *after* the block's last sample.
        warm_start = pd.Timestamp(block["end"]).floor("1min") + pd.Timedelta(minutes=1)
        m3_after_frozen |= _rows_after_until(
            index, warm_start, warm_start + cfg.warmup
        )

    m2 = np.zeros(n, dtype=bool)
    m3_after_gap = np.zeros(n, dtype=bool)
    for _, gap in native_masks.major_gaps.iterrows():
        gap_start = pd.Timestamp(gap["start"])
        gap_end = pd.Timestamp(gap["end"])
        # Minutes strictly inside the gap are absent from the data; mark them.
        m2 |= np.asarray((index > gap_start) & (index < gap_end), dtype=bool)
        m3_after_gap |= _rows_after_until(index, gap_end, gap_end + cfg.warmup)

    m3 = m3_after_frozen | m3_after_gap
    return GridMasks(
        m1=m1,
        m2=m2,
        m3=m3,
        m3_after_frozen=m3_after_frozen,
        m3_after_gap=m3_after_gap,
        index=index,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _gap_overlap(
    gaps: pd.DataFrame,
    events: Sequence[EventWindow],
) -> dict[int, tuple[float, int]]:
    out: dict[int, tuple[float, int]] = {}
    for event in events:
        hours = 0.0
        count = 0
        for _, gap in gaps.iterrows():
            inter = event.window.intersect(
                TimeWindow(pd.Timestamp(gap["start"]), pd.Timestamp(gap["end"]))
            )
            if inter is not None:
                hours += inter.duration_hours
                count += 1
        out[event.event_id] = (round(hours, 6), count)
    return out


def mask_summary(
    grid: MinuteGrid,
    masks: GridMasks,
    native_masks: NativeMasks,
    events: Iterable[EventWindow] | None = None,
) -> dict:
    """Global mask shares and per-event overlap, M1/M2/M3 reported separately.

    ``share`` denominators are wall-clock minutes in the event window, so a gap
    counts as masked time even though it contributes no grid row values.
    """
    chosen = tuple(FAILURE_EVENTS if events is None else events)
    index = grid.index
    observed = grid.frame["observed"].to_numpy()

    global_stats = {
        **native_masks.summary(),
        "grid_rows": int(len(grid)),
        "grid_observed_rows": int(observed.sum()),
        "grid_has_value_rows": int(grid.frame["has_value"].sum()),
        "grid_wall_clock_hours": round(grid.wall_clock_hours, 3),
        "grid_m1_rows": int(masks.m1.sum()),
        "grid_m1_share": float(masks.m1.mean()),
        "grid_m2_rows": int(masks.m2.sum()),
        "grid_m3_rows": int(masks.m3.sum()),
        "grid_m3_after_frozen_rows": int(masks.m3_after_frozen.sum()),
        "grid_m3_after_gap_rows": int(masks.m3_after_gap.sum()),
        "grid_combined_rows": int(masks.combined.sum()),
        "grid_combined_share": float(masks.combined.mean()),
        "grid_unmasked_observed_rows": int((observed & ~masks.combined).sum()),
        "grid_unmasked_observed_share": float((observed & ~masks.combined).mean()),
    }

    gap_overlap = _gap_overlap(native_masks.major_gaps, chosen)
    event_rows = []
    for event in chosen:
        in_window = event.window.mask(index)
        total = int(in_window.sum())
        row = {
            "event_id": event.event_id,
            "window_hours": round(event.duration_hours, 3),
            "window_minutes": total,
        }
        for name, mask in (
            ("m1", masks.m1),
            ("m2", masks.m2),
            ("m3", masks.m3),
            ("combined", masks.combined),
        ):
            count = int((in_window & mask).sum())
            row[f"{name}_minutes"] = count
            row[f"{name}_share"] = count / total if total else 0.0
        row["m2_gap_hours"] = gap_overlap[event.event_id][0]
        row["m2_gap_count"] = gap_overlap[event.event_id][1]
        row["usable_minutes"] = int(
            (in_window & observed & ~masks.combined).sum()
        )
        event_rows.append(row)

    return {"global": global_stats, "events": event_rows}


def mask_is_label_free(frame: pd.DataFrame, config: MaskConfig | None = None) -> str:
    """Return the mask signature; the frame is the only input by construction."""
    native = detect_native_masks(frame, config)
    digest = hashlib.sha256()
    digest.update(native.frozen_blocks.to_csv(index=False).encode("utf-8"))
    digest.update(native.major_gaps.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()
