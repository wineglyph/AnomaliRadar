"""Synthetic-frame builders shared by the test suite (plain module, not a fixture)."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATE_TO_CURRENT = {
    "off": 0.0,
    "offloaded": 3.5,
    "loaded": 6.5,
    "other": 1.0,
}
STATE_TO_DV = {
    "off": 0,
    "offloaded": 0,
    "loaded": 1,
    "other": 0,
}


def make_native_frame(
    *,
    start: str = "2020-02-01 00:00",
    minutes: int = 1200,
    freq: str = "1min",
    states: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a complete, schema-valid synthetic native frame.

    ``states`` drives ``Motor_current``/``DV_eletric`` so that the raw state is
    exactly the requested label. Analog channels carry small seeded noise, so a
    normal frame never triggers the frozen-block rule.
    """
    if states is None:
        states = ["offloaded"] * minutes
    if len(states) != minutes:
        raise ValueError("states length must equal minutes")

    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=minutes, freq=freq)
    current = np.array([STATE_TO_CURRENT[s] for s in states], dtype=float)
    current = current + rng.normal(0.0, 0.01, minutes)
    dv = np.array([STATE_TO_DV[s] for s in states], dtype=int)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "TP2": rng.normal(8.5, 0.05, minutes).round(3),
            "TP3": rng.normal(8.4, 0.05, minutes).round(3),
            "H1": rng.normal(0.2, 0.02, minutes).round(3),
            "DV_pressure": rng.normal(0.05, 0.01, minutes).round(3),
            "Reservoirs": rng.normal(8.3, 0.05, minutes).round(3),
            "Oil_temperature": rng.normal(45.0, 1.0, minutes).round(3),
            "Motor_current": current.round(3),
            "COMP": dv,
            "DV_eletric": dv,
            "Towers": rng.integers(0, 2, minutes),
            "MPG": dv,
            "LPS": np.zeros(minutes, dtype=int),
            "Pressure_switch": rng.integers(0, 2, minutes),
            "Oil_level": np.zeros(minutes, dtype=int),
            "Caudal_impulses": np.zeros(minutes, dtype=int),
        }
    )


def drop_minutes(frame: pd.DataFrame, minutes: list[int]) -> pd.DataFrame:
    """Remove every native row whose timestamp floors to one of ``minutes``.

    ``minutes`` are offsets from the first timestamp. The result stays sorted and
    valid; it simply has no rows for the requested minutes.
    """
    first = frame["timestamp"].min()
    offset = ((frame["timestamp"] - first) // pd.Timedelta(minutes=1)).astype(int)
    return frame.loc[~offset.isin(minutes)].reset_index(drop=True)


def freeze_analog(frame: pd.DataFrame, start_row: int, end_row: int) -> pd.DataFrame:
    """Make all 7 analog channels constant over ``[start_row, end_row)``."""
    from anomaliradar.data.schema import ANALOG

    out = frame.copy()
    for col in ANALOG:
        out.loc[out.index[start_row:end_row], col] = float(out[col].iloc[0])
    return out


def synthetic_events(start_a: str, start_b: str | None = None):
    """One or two synthetic 2-hour events, used by split/leakage tests."""
    from anomaliradar.data.labels import EventWindow
    from anomaliradar.timewindow import TimeWindow

    events = [
        EventWindow(
            event_id=1,
            pdf_label="synthetic-1",
            kind="test",
            window=TimeWindow(start_a, pd.Timestamp(start_a) + pd.Timedelta(hours=2)),
        )
    ]
    if start_b is not None:
        events.append(
            EventWindow(
                event_id=2,
                pdf_label="synthetic-2",
                kind="test",
                window=TimeWindow(start_b, pd.Timestamp(start_b) + pd.Timedelta(hours=2)),
            )
        )
    return tuple(events)
