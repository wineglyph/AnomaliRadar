"""Data Explorer page."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib import artifacts
from app_lib import ui

PRESETS = {
    "Event 1 window": 1,
    "Event 2 window": 2,
    "Event 3 window": 3,
    "Event 4 window": 4,
    "Tail window (Aug 2020)": "T",
    "Full series": "all",
    "Custom range": "custom",
}


def _preset_range(choice: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if choice == "Full series":
        return pd.Timestamp("2020-02-01"), pd.Timestamp("2020-09-02")
    table = artifacts.fold_table()
    fold = PRESETS[choice]
    row = table[table["fold"] == str(fold)].iloc[0]
    start = pd.Timestamp(row["test_start"]).floor("D")
    end = pd.Timestamp(row["test_end"]).ceil("D")
    return start, end


def _auto_rule(start: pd.Timestamp, end: pd.Timestamp) -> str:
    days = max((end - start).total_seconds() / 86400.0, 1e-9)
    if days <= 2:
        return "1min"
    if days <= 7:
        return "5min"
    if days <= 30:
        return "15min"
    return "1h"


def _window_sensors(
    frame: pd.DataFrame,
    channels: list[str],
    rule: str,
) -> pd.DataFrame:
    aggregation = {
        channel: ("mean" if channel in {"TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"} else "last")
        for channel in channels
    }
    series = frame[channels].resample(rule).agg(aggregation)
    series = series.dropna(how="all")
    long = (
        series.reset_index()
        .melt(id_vars="timestamp", var_name="channel", value_name="value")
        .dropna(subset=["value"])
    )
    return long


def _state_segments(state: pd.DataFrame) -> pd.DataFrame:
    if state.empty:
        return pd.DataFrame(columns=["state", "start", "end"])
    block = state.reset_index()[["timestamp", "state"]].copy()
    block["state"] = block["state"].fillna("unknown")
    block["segment"] = block["state"].ne(block["state"].shift()).cumsum()
    grouped = block.groupby("segment", sort=True).agg(
        state=("state", "first"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
    )
    grouped["end"] = grouped["end"] + pd.Timedelta(minutes=1)
    return grouped.reset_index(drop=True)


def _range_mask_summary(mask: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    block = mask.loc[(mask.index >= start) & (mask.index < end)]
    if block.empty:
        return pd.DataFrame(columns=["Mask", "Minutes", "Share"])
    return pd.DataFrame(
        {
            "Mask": ["M1 frozen", "M2 gap", "M3 warm-up", "Combined"],
            "Minutes": [
                int(block["m1_frozen"].sum()),
                int(block["m2_gap"].sum()),
                int(block["m3_warmup"].sum()),
                int(block["combined"].sum()),
            ],
            "Share": [
                float(block["m1_frozen"].mean()),
                float(block["m2_gap"].mean()),
                float(block["m3_warmup"].mean()),
                float(block["combined"].mean()),
            ],
        }
    )


def _render_missing_dataset() -> None:
    ui.page_header(
        "Data Explorer",
        "Sensor traces, operating state, masks, failure windows and data gaps. "
        "This page reads the local MetroPT-3 CSV; metrics-only pages do not need it.",
        kicker="MetroPT-3 · real data only",
    )
    if not ui.dataset_banner():
        st.subheader("What the frozen artefacts already contain")
        coverage = artifacts.load_mask_coverage()
        st.dataframe(
            coverage.rename(
                columns={
                    "event_id": "Event",
                    "window_hours": "Window hours",
                    "window_minutes": "Window minutes",
                    "masked_minutes": "Masked minutes",
                    "masked_hours": "Masked hours",
                    "masked_share": "Masked share",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={"Masked share": st.column_config.NumberColumn(format="%.2f%%")},
        )
        ui.figure_card(
            "mask_coverage.png",
            "Masked share of each labelled event window, computed from the frozen experimental artefacts.",
        )
        st.code(
            "python scripts/download_data.py\npython scripts/download_data.py --verify-only",
            language="bash",
        )
        st.caption(
            "The downloader verifies the official UCI archive against the SHA-256 "
            "recorded in `data/raw/manifest.json`."
        )


def render() -> None:
    if not ui.artifact_banner():
        return

    ui.page_header(
        "Data Explorer",
        "Everything on this page is computed from the local MetroPT-3 CSV with the "
        "same deterministic data layer used by the experiment. No synthetic series "
        "are shown.",
        kicker="MetroPT-3 · real data only",
    )
    if not artifacts.dataset_available():
        _render_missing_dataset()
        return

    manifest = artifacts.load_manifest()
    dataset_summary = manifest["dataset_summary"]
    states_manifest = dataset_summary["state"]
    masks_manifest = dataset_summary["masks"]

    control_a, control_b, control_c = st.columns([1.0, 1.0, 1.6])
    with control_a:
        choice = st.selectbox("Window", list(PRESETS), index=0)
    if choice == "Custom range":
        with control_b:
            custom = st.date_input(
                "Dates",
                value=(pd.Timestamp("2020-04-15").date(), pd.Timestamp("2020-04-21").date()),
                min_value=pd.Timestamp("2020-02-01").date(),
                max_value=pd.Timestamp("2020-09-01").date(),
            )
        if isinstance(custom, tuple) and len(custom) == 2:
            start = pd.Timestamp(custom[0])
            end = pd.Timestamp(custom[1]) + pd.Timedelta(days=1)
        else:
            start, end = pd.Timestamp("2020-04-15"), pd.Timestamp("2020-04-22")
    else:
        start, end = _preset_range(choice)
        with control_b:
            st.caption("Selected window")
            st.write(f"{start:%Y-%m-%d} → {end:%Y-%m-%d}")

    with st.spinner("Loading and preparing the deterministic 1-minute data layer…"):
        dataset = artifacts.load_dataset()
    grid = dataset.grid.frame
    mask = dataset.masks.as_frame()
    state = dataset.state
    events = artifacts.events()

    window = grid.loc[(grid.index >= start) & (grid.index < end)]
    window_state = state.loc[(state.index >= start) & (state.index < end)]
    window_mask = mask.loc[(mask.index >= start) & (mask.index < end)]

    observed_share = float(window["observed"].mean()) if len(window) else float("nan")
    masked_share = float(window_mask["combined"].mean()) if len(window_mask) else float("nan")
    gaps = dataset.grid.breaks
    gaps_window = gaps[
        (pd.to_datetime(gaps["start"]) < end) & (pd.to_datetime(gaps["end"]) >= start)
    ]
    ui.metric_cards(
        [
            ("Grid minutes in window", f"{len(window):,}", "1-minute rows, including missing minutes."),
            ("Observed share", ui.pct(observed_share), "Rows with a real sample in the minute."),
            ("Masked share", ui.pct(masked_share), "M1 + M2 + M3 combined."),
            ("Breaks in window", f"{len(gaps_window):,}", "Gaps longer than the 2-minute forward-fill limit."),
        ]
    )

    tab_sensor, tab_state, tab_masks, tab_events, tab_gaps = st.tabs(
        ["Sensor time series", "Operating state", "Masked regions", "Failure windows", "Data gaps"]
    )

    with tab_sensor:
        sensor_a, sensor_b = st.columns([2.1, 1.0])
        with sensor_a:
            channels = st.multiselect(
                "Channels",
                options=list(window.columns[:15]),
                default=["DV_pressure", "Reservoirs", "Oil_temperature"],
                help="Analog channels are minute means; digital channels are the last sample in the minute.",
            )
        with sensor_b:
            resolution = st.selectbox("Rendering resolution", ["Auto", "1 min", "5 min", "15 min", "1 hour"])
        rule = _auto_rule(start, end) if resolution == "Auto" else {
            "1 min": "1min",
            "5 min": "5min",
            "15 min": "15min",
            "1 hour": "1h",
        }[resolution]
        if channels:
            long = _window_sensors(window, channels, rule)
            if long.empty:
                st.info("No observed sensor values in this window.")
            else:
                line = (
                    alt.Chart(long)
                    .mark_line(strokeWidth=1.15)
                    .encode(
                        x=alt.X("timestamp:T", title=None),
                        y=alt.Y("value:Q", title="Sensor value", scale=alt.Scale(zero=False)),
                        color=alt.Color("channel:N", title=None),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="Time"),
                            alt.Tooltip("channel:N", title="Channel"),
                            alt.Tooltip("value:Q", title="Value", format=".4f"),
                        ],
                    )
                )
                ev = events[(events["end"] >= start) & (events["start"] < end)]
                layers = [line]
                if not ev.empty:
                    shade = (
                        alt.Chart(ev)
                        .mark_rect(color="#c92a2a", opacity=0.09)
                        .encode(x=alt.X("start:T", title=None), x2="end:T")
                    )
                    rules = (
                        alt.Chart(ev)
                        .mark_rule(color="#c92a2a", strokeDash=[4, 4], opacity=0.8)
                        .encode(x="start:T")
                    )
                    layers = [shade, line, rules]
                st.altair_chart(ui.style_chart(alt.layer(*layers), height=420), width="stretch")
                st.caption(
                    f"Rendered at {rule} resolution from {len(window):,} grid minutes. "
                    "Red dashed lines mark labelled failure-window starts; red bands show the full windows. "
                    "The underlying data are not interpolated for display."
                )
        st.markdown("**State distribution in the selected window**")
        state_counts = (
            window_state["state"].fillna("unknown").value_counts().rename_axis("State").reset_index(name="Minutes")
        )
        state_counts["Share"] = state_counts["Minutes"] / max(len(window_state), 1)
        st.dataframe(
            state_counts,
            width="stretch",
            hide_index=True,
            column_config={"Share": st.column_config.NumberColumn(format="%.3f")},
        )

    with tab_state:
        st.markdown(
            "The operating state is confirmed with 3-minute hysteresis. `other` is a "
            "transition artefact and is excluded from fitting and calibration."
        )
        segments = _state_segments(window_state)
        if segments.empty:
            st.info("No state rows in this window.")
        else:
            rect = (
                alt.Chart(segments)
                .mark_rect(height=34)
                .encode(
                    x=alt.X("start:T", title=None),
                    x2="end:T",
                    color=ui.state_color_scale(),
                    tooltip=[
                        alt.Tooltip("state:N", title="State"),
                        alt.Tooltip("start:T", title="From"),
                        alt.Tooltip("end:T", title="To"),
                    ],
                )
            )
            st.altair_chart(ui.style_chart(rect, height=130), width="stretch")
        st.markdown("**Experiment-wide confirmed-state summary**")
        state_summary = pd.DataFrame(
            {
                "State": list(states_manifest["confirmed_counts"].keys()),
                "Minutes": list(states_manifest["confirmed_counts"].values()),
                "Share": [states_manifest["confirmed_share"][k] for k in states_manifest["confirmed_counts"]],
                "Episodes": [states_manifest["episodes"].get(k, 0) for k in states_manifest["confirmed_counts"]],
                "Median run (min)": [
                    states_manifest["run_length_median_min"].get(k, np.nan)
                    for k in states_manifest["confirmed_counts"]
                ],
            }
        )
        st.dataframe(
            state_summary,
            width="stretch",
            hide_index=True,
            column_config={"Share": st.column_config.NumberColumn(format="%.2f%%")},
        )
        st.caption(
            f"{states_manifest['segments']} segments; other-state rows: "
            f"{states_manifest['other_state_rows']:,} "
            f"({states_manifest['other_state_share']:.2%})."
        )

    with tab_masks:
        summary = _range_mask_summary(mask, start, end)
        st.markdown("**Mask composition in the selected window**")
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
            column_config={"Share": st.column_config.NumberColumn(format="%.2f%%")},
        )
        block = mask.loc[(mask.index >= start) & (mask.index < end)]
        if not block.empty:
            hourly = (
                block[["m1_frozen", "m2_gap", "m3_warmup"]]
                .resample("1h")
                .mean()
                .reset_index()
                .melt(id_vars="timestamp", var_name="mask", value_name="share")
            )
            hourly["mask"] = hourly["mask"].map(
                {"m1_frozen": "M1 frozen", "m2_gap": "M2 gap", "m3_warmup": "M3 warm-up"}
            )
            chart = (
                alt.Chart(hourly)
                .mark_area(opacity=0.75)
                .encode(
                    x=alt.X("timestamp:T", title=None),
                    y=alt.Y("share:Q", title="Share of hour masked", stack="zero", axis=alt.Axis(format="%")),
                    color=alt.Color(
                        "mask:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["M1 frozen", "M2 gap", "M3 warm-up"],
                            range=["#5c7cfa", "#f59f00", "#e64980"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Hour"),
                        alt.Tooltip("mask:N", title="Mask"),
                        alt.Tooltip("share:Q", title="Share", format=".1%"),
                    ],
                )
            )
            st.altair_chart(ui.style_chart(chart, height=260), width="stretch")
        st.markdown("**Experiment-wide mask summary**")
        global_masks = pd.DataFrame(
            [
                ("M1 frozen native rows", f"{masks_manifest['global']['m1_native_rows']:,}", ui.pct(masks_manifest["global"]["m1_native_share"])),
                ("M1 blocks", f"{masks_manifest['global']['m1_blocks']:,}", f"max {masks_manifest['global']['m1_max_hours']:.2f} h"),
                ("M2 major gaps", f"{masks_manifest['global']['m2_gaps']:,}", f"{masks_manifest['global']['m2_gap_hours']:.1f} h total"),
                ("M3 warm-up after M1/M2", f"{masks_manifest['global']['grid_m3_rows']:,}", f"{masks_manifest['global']['grid_combined_share']:.2%} combined with M1/M2"),
                ("Unmasked observed rows", f"{masks_manifest['global']['grid_unmasked_observed_rows']:,}", ui.pct(masks_manifest["global"]["grid_unmasked_observed_share"])),
            ],
            columns=["Quantity", "Value", "Note"],
        )
        st.dataframe(global_masks, width="stretch", hide_index=True)
        st.markdown("**Frozen blocks (M1) overlapping the window**")
        frozen = dataset.native_masks.frozen_blocks
        frozen_window = frozen[(frozen["end"] >= start) & (frozen["start"] < end)]
        if frozen_window.empty:
            st.caption("None.")
        else:
            st.dataframe(frozen_window, width="stretch", hide_index=True)
        st.markdown("**Major gaps (M2) overlapping the window**")
        major = dataset.native_masks.major_gaps
        major_window = major[(major["end"] >= start) & (major["start"] < end)]
        if major_window.empty:
            st.caption("None.")
        else:
            st.dataframe(major_window, width="stretch", hide_index=True)

    with tab_events:
        coverage = artifacts.load_mask_coverage()
        table = events.merge(coverage, left_on="event_id", right_on="event_id", how="left")
        table = table.rename(
            columns={
                "event_id": "Event",
                "pdf_label": "PDF label",
                "start": "Start",
                "end": "End",
                "hours": "Hours",
                "window_minutes": "Window minutes",
                "masked_minutes": "Masked minutes",
                "masked_share": "Masked share",
            }
        )
        st.dataframe(
            table[["Event", "PDF label", "Start", "End", "Hours", "Window minutes", "Masked minutes", "Masked share"]],
            width="stretch",
            hide_index=True,
            column_config={
                "Start": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "End": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "Masked share": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.caption(
            "The four labelled failure windows are the only labels in the experiment. "
            "Event 2 is the duplicate '#1' entry documented in the dataset notes. "
            "Masked share here is the event window; detection-credit masking is "
            "reported separately in the limitations."
        )
        ui.figure_card(
            "mask_coverage.png",
            "Masked share of each event window. Event 3 is the highest at 8.51%, below the 10% falsification line.",
        )

    with tab_gaps:
        st.markdown(
            "A break is any interval longer than the 2-minute forward-fill limit. "
            "Breaks longer than one hour also create an M2 mask and a 30-minute M3 warm-up."
        )
        if gaps_window.empty:
            st.info("No breaks in the selected window.")
        else:
            gaps_display = gaps_window.copy()
            gaps_display["start"] = pd.to_datetime(gaps_display["start"])
            gaps_display["end"] = pd.to_datetime(gaps_display["end"])
            gaps_display = gaps_display.rename(
                columns={
                    "start": "Start",
                    "end": "End",
                    "delta_minutes": "Gap minutes",
                    "missing_minutes": "Missing minutes",
                }
            )
            st.metric("Total missing minutes", f"{int(gaps_display['Missing minutes'].sum()):,}")
            st.dataframe(
                gaps_display.sort_values("Gap minutes", ascending=False),
                width="stretch",
                hide_index=True,
                column_config={
                    "Start": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    "End": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    "Gap minutes": st.column_config.NumberColumn(format="%.1f"),
                },
            )
        st.caption(
            f"Experiment-wide: {dataset_summary['breaks_longer_than_ffill']:,} breaks, "
            f"{dataset_summary['segments']:,} segments."
        )

    ui.provenance_strip()


render()
