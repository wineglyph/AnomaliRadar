"""Shared presentation helpers for the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import altair as alt
import pandas as pd
import streamlit as st

from app_lib import artifacts


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "—"
    return f"{number:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "—"
    return f"{100.0 * number:.{digits}f}%"


def delay(value: object) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "—"
    number = int(round(number))
    if number == 0:
        return "0 min"
    return f"{number:+d} min"


def timestamp(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str, kicker: str | None = None) -> None:
    if kicker:
        st.caption(kicker.upper())
    st.title(title)
    st.markdown(subtitle)


def metric_cards(items: Iterable[tuple[str, str, str | None]]) -> None:
    items = list(items)
    columns = st.columns(len(items))
    for column, (title, value, help_text) in zip(columns, items):
        with column:
            st.metric(title, value, help=help_text, border=True)


def artifact_banner() -> bool:
    missing = artifacts.missing_artifacts()
    if missing:
        st.error(
            "Frozen experiment artefacts are missing from this checkout:\n\n"
            + "\n".join(f"- `{path}`" for path in missing)
            + "\n\nRun `python scripts/run_experiments.py` after downloading the "
            "dataset. The app will not substitute demo values."
        )
        return False
    return True


def dataset_banner() -> bool:
    if artifacts.dataset_available():
        return True
    st.warning(
        "The MetroPT-3 CSV is not present. Metric-only pages still work. For the "
        "data explorer, run:\n\n"
        "```bash\npython scripts/download_data.py\n```\n\n"
        "The dataset is CC BY 4.0 and is never bundled with this repository."
    )
    return False


def provenance_strip() -> None:
    data = artifacts.provenance()
    st.caption(
        "Frozen experiment · "
        f"config `{data['config_hash']}` · "
        f"dataset fingerprint `{data['fingerprint'][:16]}…` · "
        f"Python {data['python']} · NumPy {data['numpy']} · pandas {data['pandas']}"
    )


def sidebar_note() -> None:
    data = artifacts.provenance()
    with st.sidebar:
        st.divider()
        st.caption("**Frozen artefacts**")
        st.caption(f"config `{data['config_hash']}`")
        st.caption(f"fingerprint `{data['fingerprint'][:20]}…`")
        st.caption(
            "This viewer never fits, refits, calibrates or re-scores a model. "
            "All values are read from `reports/experiments`."
        )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def style_chart(chart: alt.Chart, height: int = 320) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#e9ecef", domainColor="#ced4da", labelColor="#495057")
        .configure_legend(labelColor="#495057", titleColor="#212529")
    )


def event_rules(
    events: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> alt.Chart:
    block = events.copy()
    if start is not None:
        block = block[block["end"] >= start]
    if end is not None:
        block = block[block["start"] <= end]
    return (
        alt.Chart(block)
        .mark_rule(color="#c92a2a", strokeDash=[4, 4], opacity=0.75)
        .encode(
            x=alt.X("start:T", title=None),
            tooltip=[
                alt.Tooltip("event_id:O", title="Event"),
                alt.Tooltip("start:T", title="Label start"),
                alt.Tooltip("end:T", title="Label end"),
            ],
        )
        .transform_calculate(event_label='("event " + datum.event_id)')
    )


def state_color_scale() -> alt.Color:
    return alt.Color(
        "state:N",
        title="State",
        scale=alt.Scale(
            domain=["off", "offloaded", "loaded", "other"],
            range=["#868e96", "#4dabf7", "#2b8a3e", "#f08c00"],
        ),
    )


def figure_card(name: str, caption: str) -> None:
    path: Path = artifacts.figure_path(name)
    if not path.exists():
        st.info(f"Figure `{name}` is not available in this checkout.")
        return
    st.image(str(path), caption=caption, width="stretch")


def compact_frame(frame: pd.DataFrame, height: int | None = None) -> None:
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        height=height,
    )
