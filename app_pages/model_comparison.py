"""Model Comparison page."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_lib import artifacts
from app_lib import ui


def _format_comparison(table: pd.DataFrame) -> pd.DataFrame:
    display = table[
        [
            "method_label",
            "recall",
            "alarms",
            "false_alarms",
            "alarms_per_day",
            "false_per_day",
            "mean_abs_budget_error",
            "tail_alarms",
            "tail_per_day",
            "tail_relative_error",
        ]
    ].rename(
        columns={
            "method_label": "Method",
            "recall": "Recall",
            "alarms": "Alarms",
            "false_alarms": "False",
            "alarms_per_day": "Alarms/day",
            "false_per_day": "False/day",
            "mean_abs_budget_error": "Mean |budget error|",
            "tail_alarms": "Tail alarms",
            "tail_per_day": "Tail/day",
            "tail_relative_error": "Tail rel. error",
        }
    )
    return display


def _event_delay_chart(budget: float) -> alt.Chart:
    outcomes = artifacts.load_event_outcomes()
    block = outcomes[
        (outcomes["budget"] == budget)
        & (outcomes["detected"])
        & (outcomes["method"].isin(artifacts.COMPARISON_METHODS))
    ].copy()
    block["method_label"] = block["method"].map(artifacts.label)
    block["event_label"] = "Event " + block["event_id"].astype(int).astype(str)
    points = (
        alt.Chart(block)
        .mark_circle(size=110, opacity=0.9)
        .encode(
            x=alt.X("delay_minutes:Q", title="First-alarm delay (minutes; negative = before label start)"),
            y=alt.Y("method_label:N", title=None, sort=list(block["method_label"].unique())),
            color=alt.Color("event_label:N", title="Event"),
            tooltip=[
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip("event_label:N", title="Event"),
                alt.Tooltip("delay_minutes:Q", title="Delay (min)", format="+.0f"),
                alt.Tooltip("first_alarm:T", title="First alarm"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#868e96").encode(x="x:Q")
    return alt.layer(zero, points)


def render() -> None:
    if not ui.artifact_banner():
        return

    ui.page_header(
        "Model Comparison",
        "Baseline comparison, event-level recall, alarm rates, detection delays "
        "and calibration diagnostics at every requested budget. All methods share "
        "the same alarm layer and folds.",
        kicker="Frozen results · identical evaluation rules",
    )
    ui.provenance_strip()

    budgets = artifacts.available_budgets()
    budget = st.selectbox(
        "Alarm budget (per observed day)",
        budgets,
        index=budgets.index(1.0),
        help="The primary pre-registered budget is 1.0/day.",
    )
    comparison = artifacts.pooled_comparison(float(budget))
    or_row = comparison[comparison["method"] == "P3_or"]
    if not or_row.empty:
        or_row = or_row.iloc[0]
        ui.metric_cards(
            [
                ("OR recall", or_row["recall"], "Out of four labelled event folds."),
                ("OR alarms · event folds", f"{int(or_row['alarms'])}", "Pooled over 7.746 observed days."),
                ("OR false alarms", f"{int(or_row['false_alarms'])}", "False alarms are alarms outside event windows."),
                ("OR tail rate", f"{or_row['tail_per_day']:.4f}/day", "24.958 unlabelled observed days."),
            ]
        )

    st.subheader(f"Pooled comparison at {budget:.1f} alarm / observed day")
    display = _format_comparison(comparison)
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Alarms/day": st.column_config.NumberColumn(format="%.3f"),
            "False/day": st.column_config.NumberColumn(format="%.3f"),
            "Mean |budget error|": st.column_config.NumberColumn(format="%.3f"),
            "Tail/day": st.column_config.NumberColumn(format="%.4f"),
            "Tail rel. error": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    st.caption(
        "Event-fold counts are pooled; recall counts detected events out of four. "
        "Tail alarms are all false by construction (no labels in the tail). "
        "`Tail rel. error` is (realized/day − requested)/requested."
    )

    chart_data = comparison.copy()
    chart_data["Recall count"] = chart_data["recall_count"]
    recall_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            y=alt.Y("method_label:N", title=None, sort="-x"),
            x=alt.X("Recall count:Q", title="Detected events (of 4)"),
            color=alt.Color(
                "method_label:N",
                title=None,
                legend=None,
                scale=alt.Scale(
                    domain=[artifacts.label(m) for m in artifacts.COMPARISON_METHODS],
                    range=[
                        "#2b8a3e",
                        "#1971c2",
                        "#c92a2a",
                        "#f08c00",
                        "#7048e8",
                        "#868e96",
                        "#adb5bd",
                        "#495057",
                        "#0b7285",
                    ],
                ),
            ),
            tooltip=[
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip("Recall count:Q", title="Events detected"),
                alt.Tooltip("alarms:Q", title="Alarms"),
                alt.Tooltip("false_alarms:Q", title="False alarms"),
            ],
        )
    )
    false_chart = (
        alt.Chart(chart_data)
        .mark_bar(color="#e8590c", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            y=alt.Y("method_label:N", title=None, sort="-x"),
            x=alt.X("false_alarms:Q", title="False alarms (event folds)"),
            tooltip=[
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip("false_alarms:Q", title="False alarms"),
            ],
        )
    )
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.altair_chart(ui.style_chart(recall_chart, height=300), width="stretch")
    with chart_b:
        st.altair_chart(ui.style_chart(false_chart, height=300), width="stretch")

    st.divider()
    st.subheader("Event-level recall and detection delay")
    matrix = artifacts.event_matrix(float(budget))
    if not matrix.empty:
        display_matrix = matrix.copy()
        for event_id in (1, 2, 3, 4):
            display_matrix[f"E{event_id}"] = display_matrix.apply(
                lambda row, e=event_id: (
                    f"✓ {ui.delay(row[f'delay_{e}'])}"
                    if row[f"E{e}"] == "detected"
                    else "✗ missed"
                ),
                axis=1,
            )
        st.dataframe(
            display_matrix[["method_label", "E1", "E2", "E3", "E4"]].rename(
                columns={"method_label": "Method"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Cell text is the detected/missed verdict and the first accepted alarm's "
            "delay from the labelled event start. Negative delay means the alarm "
            "came before the start of the labelled event."
        )

    delay_chart = _event_delay_chart(float(budget))
    st.altair_chart(ui.style_chart(delay_chart, height=320), width="stretch")
    st.caption(
        "Only detected method/event pairs are plotted. An empty chart at a budget "
        "means no compared method detected an event there."
    )

    st.divider()
    st.subheader("Calibration diagnostics")
    method_label = st.selectbox(
        "Method for calibration detail",
        [artifacts.label(m) for m in ("P3_or", "P1_A_hybrid", "P2_B_B1", "B1_global_z")],
        index=0,
    )
    method = {artifacts.label(m): m for m in ("P3_or", "P1_A_hybrid", "P2_B_B1", "B1_global_z")}[method_label]
    diagnostics = artifacts.calibration_diagnostics(method, float(budget))
    st.dataframe(
        diagnostics,
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "`cal_alarm_n_lt_50` / `cal_alarm_n_lt_200`: alarms whose calibration pool "
        "had fewer than 50 / 200 reference rows. `cal_alarm_max_depth`: 0 = cell, "
        "1 = state, 2 = global. `cal_budget_coarse`: the realized tail rate is a "
        "step function of the request. `cal_calibration_within_50pct`: the "
        "calibration-split realized rate is within 50% of the target."
    )

    st.subheader("How the scores and baselines relate")
    fig_a, fig_b = st.columns(2)
    with fig_a:
        ui.figure_card(
            "score_contribution.png",
            "Unique detections and which score fired the OR alarms. Event 3 is A-only; event 4 is B-only.",
        )
    with fig_b:
        ui.figure_card(
            "score_scatter.png",
            "−log10 pA against −log10 pB. The scores are only weakly related; detection rows are not on one diagonal.",
        )
    ui.figure_card(
        "calibration_reliability.png",
        "Per-cell nominal alpha against empirical false-positive rate, with calibration support at 1/day. Small cells scatter.",
    )

    st.subheader("Baseline comparison in words")
    global_z = comparison[comparison["method"] == "B1_global_z"]
    iforest = comparison[comparison["method"] == "B4_iforest_context"]
    st.markdown(
        (
            f"- **Global robust z** reaches {global_z.iloc[0]['recall']} at {budget:.1f}/day "
            f"with {int(global_z.iloc[0]['false_alarms'])} false alarms; OR reaches "
            f"{or_row['recall']} with {int(or_row['false_alarms'])} false alarms.\n"
            if not global_z.empty and not or_row.empty
            else ""
        )
        + (
            f"- **IForest + context** reaches {iforest.iloc[0]['recall']} with "
            f"{int(iforest.iloc[0]['alarms'])} alarms and mean absolute budget error "
            f"{iforest.iloc[0]['mean_abs_budget_error']:.3f}.\n"
            if not iforest.empty
            else ""
        )
        + "- Baselines use the same folds, masks, debounce and budget solver; the "
        "comparison is not a comparison against unconstrained thresholding."
    )

    ui.provenance_strip()


render()
