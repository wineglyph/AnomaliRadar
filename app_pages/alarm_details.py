"""Alarm Details page."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_lib import artifacts
from app_lib import ui

METHOD_CHOICES = {
    "OR union (recommended)": "P3_or",
    "Score A — hybrid": "P1_A_hybrid",
    "Score B — B1": "P2_B_B1",
    "AND union": "P3_and",
    "Weighted λ=0.5": "P3_weighted",
    "Baseline — global robust z": "B1_global_z",
    "Baseline — IForest + context": "B4_iforest_context",
}

LEVEL_TEXT = {
    "cell": "Mondrian cell `(lagged state, day type)`",
    "state": "state-level back-off",
    "state_low": "state-level back-off flagged low-support (50 ≤ n < 200)",
    "global": "global back-off",
    "global_low": "global back-off flagged low-support",
}


def _explain(row: pd.Series, method: str) -> str:
    if method == "P3_or":
        rule = (
            f"The OR rule accepted this minute because min(pA, pB) = "
            f"**{row['p']:.3e}** is at or below the calibration-selected "
            f"alpha = **{row['alpha']:.3e}**. The minimum came from "
            f"**Score {row['fired_by']}**."
        )
    elif method == "P3_and":
        rule = (
            f"The AND rule accepted this minute because max(pA, pB) = "
            f"**{row['p']:.3e}** is at or below alpha = **{row['alpha']:.3e}**."
        )
    elif method == "P3_weighted":
        rule = (
            f"The weighted rule accepted this minute because the fused p-value = "
            f"**{row['p']:.3e}** is at or below alpha = **{row['alpha']:.3e}**."
        )
    else:
        rule = (
            f"**{artifacts.label(method)}** accepted this minute because its "
            f"calibrated p-value = **{row['p']:.3e}** is at or below alpha = "
            f"**{row['alpha']:.3e}**."
        )
    level = LEVEL_TEXT.get(str(row["cal_level"]), str(row["cal_level"]))
    calibration = (
        f"Calibration used the {level} with **n = {int(row['cal_n'])}** reference "
        f"rows (back-off depth {int(row['cal_depth'])})."
    )
    if int(row["cal_depth"]) == 2:
        calibration += (
            " The loaded-state cell or state pool was too small, so this alarm is "
            "supported by global calibration, not loaded-state calibration."
        )
    elif str(row["cal_level"]).endswith("_low"):
        calibration += (
            " The pool is above the 50-row refusal threshold but below the 200-row "
            "preferred minimum; treat the alarm as low-support."
        )
    sensor = (
        f"The largest value effect is on **{row['channel']}** "
        f"(value effect {row['value_effect']:.3f}, context effect "
        f"{row['context_effect']:.3f}). The confirmed operating state is "
        f"**{row['state']}**, with a run length of "
        f"{float(row['run_length_min']):.1f} min."
    )
    if bool(row["event_related"]):
        verdict = "This alarm falls inside a labelled failure window or its detection-credit window."
    else:
        verdict = "This alarm is not event-related under the pre-registered evaluation rule."
    return " ".join([rule, calibration, sensor, verdict])


def render() -> None:
    if not ui.artifact_banner():
        return

    ui.page_header(
        "Alarm Details",
        "Inspect any accepted alarm from the frozen alarm table: timestamp, the "
        "triggering score, sensor effects, operating state, calibration support "
        "and a plain-language explanation.",
        kicker="Alarm forensics · frozen outputs",
    )
    ui.provenance_strip()

    alarms = artifacts.load_alarms()
    control_a, control_b, control_c = st.columns(3)
    with control_a:
        fold = st.selectbox(
            "Fold",
            ["1", "2", "3", "4", "T"],
            index=2,
            format_func=lambda f: "Tail (fold T)" if f == "T" else f"Fold {f} · Event {f}",
        )
    with control_b:
        method_label = st.selectbox("Method", list(METHOD_CHOICES), index=0)
    method = METHOD_CHOICES[method_label]
    with control_c:
        all_alarms = artifacts.load_alarms()
        available = sorted(
            all_alarms[(all_alarms["fold"] == fold) & (all_alarms["method"] == method)]["budget"]
            .unique()
            .tolist()
        )
        if not available:
            available = artifacts.available_budgets()
        default_index = available.index(1.0) if 1.0 in available else 0
        budget = st.selectbox("Budget (per observed day)", available, index=default_index)

    selected_alarms = artifacts.alarms_for(fold, method, float(budget))
    if selected_alarms.empty:
        st.warning(
            f"No accepted alarms for fold {fold}, {artifacts.label(method)}, "
            f"{budget:.1f}/day. This is a real empty result; no substitute alarm is shown."
        )
        st.caption(
            "Choose another fold, method or budget. The frozen row for this "
            "combination is in `reports/experiments/metrics/fold_results.csv`."
        )
        return

    st.metric("Accepted alarms in this selection", f"{len(selected_alarms):,}", border=True)
    ordered = selected_alarms.sort_values(
        ["event_related", "timestamp"], ascending=[False, True]
    ).reset_index(drop=True)
    options = list(ordered.index)
    labels = {}
    for idx, row in ordered.iterrows():
        relation = "event-related" if row["event_related"] else "not event-related"
        labels[idx] = (
            f"{row['timestamp']:%Y-%m-%d %H:%M} · fired by {row['fired_by']} · "
            f"p={row['p']:.2e} · {relation}"
        )
    selected_index = st.selectbox("Select alarm", options, format_func=lambda idx: labels[idx])
    row = ordered.loc[selected_index]

    st.divider()
    st.subheader(f"Alarm at {row['timestamp']:%Y-%m-%d %H:%M}")
    ui.metric_cards(
        [
            ("Fused p-value", f"{row['p']:.3e}", "Accepted because p ≤ alpha after the 30-minute debounce."),
            ("Alpha", f"{row['alpha']:.3e}", "Selected on calibration rows, not test rows."),
            ("Triggering score", f"Score {row['fired_by']}", "Component that supplied the minimum (OR) or the accepted score."),
            ("Calibration n", f"{int(row['cal_n']):,}", "Reference rows behind the threshold pool."),
        ]
    )

    detail_a, detail_b = st.columns(2, gap="large")
    with detail_a:
        st.markdown("**Alarm identity**")
        identity = pd.DataFrame(
            [
                ("Timestamp", f"{row['timestamp']:%Y-%m-%d %H:%M}"),
                ("Fold", "Tail (T)" if fold == "T" else f"Fold {fold}"),
                ("Method", artifacts.label(method)),
                ("Budget", f"{budget:.1f} alarm / observed day"),
                ("Position", f"{int(row['position']):,}"),
                ("Event-related", "yes" if row["event_related"] else "no"),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(identity, width="stretch", hide_index=True)

        st.markdown("**Triggering scores**")
        scores = pd.DataFrame(
            [
                ("p (A)", f"{row['p_a']:.6e}" if pd.notna(row["p_a"]) else "—"),
                ("p (B)", f"{row['p_b']:.6e}" if pd.notna(row["p_b"]) else "—"),
                ("Fused p", f"{row['p']:.6e}"),
                ("Fired by", row["fired_by"]),
            ],
            columns=["Score", "Value"],
        )
        st.dataframe(scores, width="stretch", hide_index=True)

    with detail_b:
        st.markdown("**Sensor and state**")
        sensor = pd.DataFrame(
            [
                ("Channel", row["channel"]),
                ("Value effect", f"{row['value_effect']:.3f}"),
                ("Context effect", f"{row['context_effect']:.3f}"),
                ("Operating state", row["state"]),
                ("Run length", f"{float(row['run_length_min']):.1f} min"),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(sensor, width="stretch", hide_index=True)

        st.markdown("**Calibration support**")
        calibration = pd.DataFrame(
            [
                ("Cell", str(row["cell"]) if pd.notna(row["cell"]) else "—"),
                ("Level", row["cal_level"]),
                ("Reference n", f"{int(row['cal_n']):,}"),
                ("Back-off depth", f"{int(row['cal_depth'])}"),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(calibration, width="stretch", hide_index=True)

    st.subheader("Explanation")
    st.markdown(_explain(row, method))

    if int(row["cal_depth"]) == 2:
        st.warning(
            "This alarm uses global calibration because the loaded-state support was "
            "below the refusal threshold. Fold 4 has 41 loaded reference rows for "
            "Score A and 30 for Score B; both are below 50."
        )

    st.subheader("All alarms in this selection")
    display = selected_alarms[
        [
            "timestamp",
            "p",
            "alpha",
            "p_a",
            "p_b",
            "fired_by",
            "state",
            "cell",
            "cal_level",
            "cal_n",
            "channel",
            "value_effect",
            "context_effect",
            "event_related",
        ]
    ].rename(
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
            "event_related": "Event-related",
        }
    )
    st.dataframe(
        display,
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
        },
    )

    st.subheader("Sensor snapshot")
    if not artifacts.dataset_available():
        st.info(
            "The local MetroPT-3 CSV is not available, so no sensor snapshot can be "
            "drawn. The alarm record above still contains the frozen channel and "
            "effect sizes. Run `python scripts/download_data.py` to enable snapshots."
        )
    elif row["channel"] in {
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Oil_temperature",
        "Motor_current",
    }:
        with st.spinner("Loading the deterministic 1-minute grid…"):
            dataset = artifacts.load_dataset()
        grid = dataset.grid.frame
        channel = row["channel"]
        start = row["timestamp"] - pd.Timedelta(minutes=30)
        end = row["timestamp"] + pd.Timedelta(minutes=30)
        block = grid.loc[(grid.index >= start) & (grid.index <= end), [channel, "observed"]].reset_index()
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
            .encode(x=alt.X("timestamp:T", title=None))
        )
        st.altair_chart(ui.style_chart(alt.layer(line, rule), height=300), width="stretch")
        st.caption(
            f"±30 minutes around the alarm. Dashed line at {row['timestamp']:%Y-%m-%d %H:%M}. "
            "The series is the gridded 1-minute mean; missing minutes remain gaps."
        )
    else:
        st.info(f"Channel `{row['channel']}` is not an analog channel, so no snapshot is drawn.")

    ui.provenance_strip()


render()
