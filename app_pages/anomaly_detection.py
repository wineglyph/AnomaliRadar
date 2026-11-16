"""Anomaly Detection page."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_lib import artifacts
from app_lib import ui

METHOD_CHOICES = {
    "OR union (recommended)": "P3_or",
    "Score A — hybrid": "P1_A_hybrid",
    "Score B — B1": "P2_B_B1",
}


def _neglog(series: pd.Series) -> pd.Series:
    return -np.log10(series.clip(lower=1e-12, upper=1.0))


def _fold_event(fold: str) -> pd.Series | None:
    table = artifacts.fold_table()
    rows = table[table["fold"] == str(fold)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if pd.isna(row.get("event_start")):
        return None
    return row


def _alarm_table(alarms: pd.DataFrame) -> pd.DataFrame:
    if alarms.empty:
        return alarms
    table = alarms.copy()
    table["Event-related"] = table["event_related"].map({True: "yes", False: "no"})
    table = table.rename(
        columns={
            "timestamp": "Timestamp",
            "p": "p",
            "alpha": "alpha",
            "p_a": "p (A)",
            "p_b": "p (B)",
            "fired_by": "Fired by",
            "state": "State",
            "cell": "Cell",
            "cal_level": "Cal. level",
            "cal_n": "Cal. n",
            "channel": "Channel",
            "value_effect": "Value effect",
            "context_effect": "Context effect",
            "run_length_min": "Run length (min)",
        }
    )
    return table[
        [
            "Timestamp",
            "p",
            "alpha",
            "p (A)",
            "p (B)",
            "Fired by",
            "State",
            "Cell",
            "Cal. level",
            "Cal. n",
            "Channel",
            "Value effect",
            "Context effect",
            "Run length (min)",
            "Event-related",
        ]
    ]


def _render_tail(budget: float, method: str) -> None:
    st.warning(
        "Per-minute prediction files are exported for event folds 1–4 only. "
        "The tail fold is represented by its frozen alarm table and tail metrics. "
        "Nothing is generated on the fly here."
    )
    alarms = artifacts.alarms_for("T", method, budget)
    st.subheader(f"Tail alarms · {artifacts.label(method)} · {budget:.1f}/day")
    st.metric("Alarms", f"{len(alarms):,}", border=True)
    if alarms.empty:
        st.info("No alarms at this budget.")
    else:
        st.dataframe(
            _alarm_table(alarms),
            width="stretch",
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "p": st.column_config.NumberColumn(format="%.6f"),
                "alpha": st.column_config.NumberColumn(format="%.6f"),
                "p (A)": st.column_config.NumberColumn(format="%.6f"),
                "p (B)": st.column_config.NumberColumn(format="%.6f"),
                "Cal. n": st.column_config.NumberColumn(format="%d"),
                "Value effect": st.column_config.NumberColumn(format="%.3f"),
                "Context effect": st.column_config.NumberColumn(format="%.3f"),
                "Run length (min)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
    ui.figure_card(
        "tail_budget_control.png",
        "Realized alarm rate on the unlabelled tail. OR realizes 0.5609/day at a 1/day request; the response is stepwise.",
    )


def _render_sensor_context(fold: str, alarms: pd.DataFrame) -> None:
    st.subheader("Sensor deviations")
    if alarms.empty:
        st.info("Select a fold/budget with at least one alarm to inspect sensor context.")
        return

    effects = alarms[
        ["timestamp", "fired_by", "channel", "state", "value_effect", "context_effect", "p"]
    ].copy()
    effects = effects.rename(
        columns={
            "timestamp": "Timestamp",
            "fired_by": "Fired by",
            "channel": "Channel",
            "state": "State",
            "value_effect": "Value effect",
            "context_effect": "Context effect",
            "p": "p",
        }
    )
    st.markdown(
        "The exported alarm record contains the channel that carried the largest "
        "value effect and the context effect at the alarm minute. These are effect "
        "sizes from the frozen scoring path, not re-computed values."
    )
    st.dataframe(
        effects,
        width="stretch",
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "p": st.column_config.NumberColumn(format="%.6f"),
            "Value effect": st.column_config.NumberColumn(format="%.3f"),
            "Context effect": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    if not artifacts.dataset_available():
        st.caption(
            "Raw sensor snapshots require the local MetroPT-3 CSV. "
            "Run `python scripts/download_data.py` to enable them."
        )
        return

    with st.expander("Load a local sensor snapshot around one alarm (±30 min)"):
        options = list(alarms.index)
        labels = {
            idx: f"{row.timestamp:%Y-%m-%d %H:%M} · {row.fired_by} · {row.channel} · p={row.p:.2e}"
            for idx, row in alarms.iterrows()
        }
        selected = st.selectbox("Alarm", options, format_func=lambda idx: labels[idx])
        row = alarms.loc[selected]
        channel = row["channel"]
        if channel not in {"TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"}:
            st.info(f"Alarm channel `{channel}` is not an analog channel; no snapshot shown.")
            return
        with st.spinner("Loading the deterministic 1-minute grid…"):
            dataset = artifacts.load_dataset()
        grid = dataset.grid.frame
        start = row["timestamp"] - pd.Timedelta(minutes=30)
        end = row["timestamp"] + pd.Timedelta(minutes=30)
        block = grid.loc[(grid.index >= start) & (grid.index <= end), [channel, "observed"]].reset_index()
        if block.empty:
            st.info("No grid rows around this alarm.")
            return
        line = (
            alt.Chart(block)
            .mark_line(color="#1971c2", strokeWidth=1.4)
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y(f"{channel}:Q", title=channel, scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip(f"{channel}:Q", title=channel, format=".4f"),
                    alt.Tooltip("observed:N", title="Observed"),
                ],
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"timestamp": [row["timestamp"]]}))
            .mark_rule(color="#c92a2a", strokeDash=[4, 4])
            .encode(x="timestamp:T")
        )
        st.altair_chart(ui.style_chart(alt.layer(line, rule), height=280), width="stretch")
        st.caption(
            f"Alarm at {row['timestamp']:%Y-%m-%d %H:%M} on `{channel}`. "
            f"State `{row['state']}`, cell `{row['cell']}`, calibration "
            f"`{row['cal_level']}` with n={int(row['cal_n'])}. "
            "The line is the gridded 1-minute mean; missing minutes stay missing."
        )


def render() -> None:
    if not ui.artifact_banner():
        return

    ui.page_header(
        "Anomaly Detection",
        "Calibrated p-values, alarm decisions, calibration support and sensor "
        "effects for the frozen event folds. Score A is value-based; Score B is "
        "state-persistence-based; OR takes the minimum p-value.",
        kicker="Frozen predictions · no refitting",
    )
    ui.provenance_strip()

    budgets = artifacts.available_budgets()
    control_a, control_b, control_c = st.columns(3)
    with control_a:
        fold = st.selectbox(
            "Fold",
            ["1", "2", "3", "4", "T"],
            index=2,
            format_func=lambda f: "Tail (fold T)" if f == "T" else f"Fold {f} · Event {f}",
        )
    with control_b:
        budget = st.selectbox("Alarm budget (per observed day)", budgets, index=budgets.index(1.0))
    with control_c:
        method_label = st.selectbox("Alarm method", list(METHOD_CHOICES), index=0)
    method = METHOD_CHOICES[method_label]

    if fold == "T":
        _render_tail(float(budget), method)
        return

    row = artifacts.method_row(fold, method, float(budget))
    if row is None:
        st.error("No frozen row for this fold/method/budget combination.")
        return
    predictions = artifacts.load_predictions(fold)
    alarms = artifacts.alarms_for(fold, method, float(budget))
    event = _fold_event(fold)

    st.subheader(f"Fold {fold} · {artifacts.label(method)} · {budget:.1f}/day")
    st.caption(
        f"Test window {predictions['timestamp'].min():%Y-%m-%d %H:%M} → "
        f"{predictions['timestamp'].max():%Y-%m-%d %H:%M}. "
        f"{len(predictions):,} scored minutes. Event recall "
        f"`{row['event_recall']}`; {int(row['alarms'])} alarm(s), "
        f"{int(row['false_alarms'])} false."
    )

    alpha = float(row["alpha"])
    st.metric(
        "Selected alpha (calibration-only)",
        f"{alpha:.3e}",
        help="The budget solver selects alpha from the calibration selection split, never from the test window.",
        border=True,
    )

    pdata = predictions.copy()
    pdata["−log10 pA"] = _neglog(pdata["p_a"])
    pdata["−log10 pB"] = _neglog(pdata["p_b"])
    pdata["−log10 pOR"] = _neglog(pdata["p_or"])
    p_long = pdata.melt(
        id_vars="timestamp",
        value_vars=["−log10 pA", "−log10 pB", "−log10 pOR"],
        var_name="score",
        value_name="evidence",
    )

    chart = (
        alt.Chart(p_long)
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("evidence:Q", title="−log10(p)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "score:N",
                title=None,
                scale=alt.Scale(
                    domain=["−log10 pA", "−log10 pB", "−log10 pOR"],
                    range=["#2b8a3e", "#1971c2", "#c92a2a"],
                ),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("score:N", title="Score"),
                alt.Tooltip("evidence:Q", title="−log10(p)", format=".3f"),
            ],
        )
    )
    layers = [chart]
    threshold = -np.log10(max(alpha, 1e-12))
    threshold_frame = pd.DataFrame({"y": [threshold]})
    layers.append(
        alt.Chart(threshold_frame)
        .mark_rule(color="#212529", strokeDash=[6, 4], opacity=0.8)
        .encode(y="y:Q")
    )
    if event is not None:
        ev_start = pd.Timestamp(event["event_start"])
        ev_end = pd.Timestamp(event["event_end"])
        shade = (
            alt.Chart(pd.DataFrame({"start": [ev_start], "end": [ev_end]}))
            .mark_rect(color="#c92a2a", opacity=0.08)
            .encode(x=alt.X("start:T"), x2="end:T")
        )
        layers.insert(0, shade)
    if not alarms.empty:
        alarm_frame = alarms[["timestamp"]].copy()
        layers.append(
            alt.Chart(alarm_frame)
            .mark_rule(color="#e8590c", strokeWidth=1.6)
            .encode(
                x=alt.X("timestamp:T", title=None),
                tooltip=[alt.Tooltip("timestamp:T", title="Alarm")],
            )
        )
    st.altair_chart(ui.style_chart(alt.layer(*layers), height=380), width="stretch")
    st.caption(
        "Dashed black: calibration-selected alpha. Red band: labelled event window "
        "(clipped by the chart). Orange vertical lines: accepted alarms after the "
        "30-minute debounce. Higher values mean stronger evidence."
    )

    score_a, score_b = st.columns(2)
    with score_a:
        chart_a = (
            alt.Chart(predictions)
            .mark_line(color="#2b8a3e", strokeWidth=1.1)
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("a_hybrid:Q", title="Score A (hybrid)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("a_hybrid:Q", title="Score A", format=".4f"),
                ],
            )
        )
        st.altair_chart(ui.style_chart(chart_a, height=220), width="stretch")
        st.caption("Score A: causal ridge residual nonconformity, max over five value channels.")
    with score_b:
        chart_b = (
            alt.Chart(predictions)
            .mark_line(color="#1971c2", strokeWidth=1.1)
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("score_b:Q", title="Score B (B1)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("score_b:Q", title="Score B", format=".4f"),
                ],
            )
        )
        st.altair_chart(ui.style_chart(chart_b, height=220), width="stretch")
        st.caption("Score B: state-run survival nonconformity. It never reads a value channel.")

    st.subheader("Calibrated p-values and alarm timestamps")
    if alarms.empty:
        st.info(
            "No accepted alarms at this budget. An empty alarm set is a real result, "
            "not missing data."
        )
    else:
        st.dataframe(
            _alarm_table(alarms),
            width="stretch",
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "p": st.column_config.NumberColumn(format="%.6f"),
                "alpha": st.column_config.NumberColumn(format="%.6f"),
                "p (A)": st.column_config.NumberColumn(format="%.6f"),
                "p (B)": st.column_config.NumberColumn(format="%.6f"),
                "Cal. n": st.column_config.NumberColumn(format="%d"),
                "Value effect": st.column_config.NumberColumn(format="%.3f"),
                "Context effect": st.column_config.NumberColumn(format="%.3f"),
                "Run length (min)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(
            "Every row is an accepted alarm from `alarms.csv`. The 30-minute debounce "
            "is already applied. `p` is the fused p-value; `p (A)` and `p (B)` are "
            "the component scores. `Fired by` identifies which component supplied "
            "the minimum for OR."
        )

    with st.expander("Calibration diagnostics for this fold and budget", expanded=False):
        diag = artifacts.calibration_diagnostics(method, float(budget))
        st.dataframe(diag, width="stretch", hide_index=True)
        st.caption(
            "`cal_alarm_n_lt_50` and `cal_alarm_n_lt_200` count alarms whose "
            "calibration cell had fewer than 50 or 200 reference rows. "
            "`cal_alarm_max_depth` is the back-off depth used (0 = cell, 2 = global). "
            "`cal_budget_coarse` flags a step-function budget response."
        )
        if isinstance(row.get("cal_alarm_levels"), str):
            st.code(row["cal_alarm_levels"], language="json")

    _render_sensor_context(fold, alarms)

    st.divider()
    st.subheader("Event-level context")
    if event is not None:
        st.markdown(
            f"Labelled event {int(event['event_id'])}: **{pd.Timestamp(event['event_start']):%Y-%m-%d %H:%M}** "
            f"→ **{pd.Timestamp(event['event_end']):%Y-%m-%d %H:%M}**. "
            f"Detection-credit window: {pd.Timestamp(event['detection_start']):%Y-%m-%d %H:%M} "
            f"→ {pd.Timestamp(event['detection_end']):%Y-%m-%d %H:%M}."
        )
        delays = artifacts.load_event_outcomes()
        selected = delays[
            (delays["fold"] == fold) & (delays["method"] == method) & (delays["budget"] == float(budget))
        ]
        if not selected.empty:
            st.dataframe(
                selected[
                    ["event_id", "detected", "delay_minutes", "first_alarm", "alarms_in_detection", "masked_detection_rows"]
                ],
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("This fold has no labelled event (tail fold).")

    ui.provenance_strip()


render()
