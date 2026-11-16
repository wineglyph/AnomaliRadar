"""Limitations page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_lib import artifacts
from app_lib import ui


def _tail_table() -> pd.DataFrame:
    frame = artifacts.load_fold_results()
    block = frame[(frame["fold"] == "T") & (frame["method"] == "P3_or")].copy()
    block = block.sort_values("budget")
    block["Relative error"] = (block["alarms_per_day"] - block["budget"]) / block["budget"]
    return block[["budget", "alarms", "alarms_per_day", "Relative error"]].rename(
        columns={
            "budget": "Requested / day",
            "alarms": "Tail alarms",
            "alarms_per_day": "Realized / day",
            "Relative error": "Relative error",
        }
    )


def render() -> None:
    ui.page_header(
        "Limitations",
        "The limitations are part of the result. This page states them explicitly, "
        "with the measured numbers, so the release cannot be read as a stronger "
        "claim than the evidence supports.",
        kicker="Read before citing",
    )
    ui.provenance_strip()

    st.error(
        "**CONDITIONAL GO, not unconditional deployment evidence.** The pipeline "
        "is complete and reproducible, but four labelled events, 7.746 observed "
        "event-fold days and a non-exchangeable time series cannot support a tight "
        "false-alarm guarantee or statistical significance.",
        icon=":material/warning:",
    )

    st.subheader("1. Only four labelled events")
    manifest = artifacts.load_manifest()
    events = artifacts.events()
    observed_days = float(
        artifacts.load_fold_results()[
            (artifacts.load_fold_results()["fold"].isin(["1", "2", "3", "4"]))
            & (artifacts.load_fold_results()["method"] == "P3_or")
            & (artifacts.load_fold_results()["budget"] == 1.0)
        ]["observed_days"].sum()
    )
    st.markdown(
        f"The dataset contains **four** labelled failure windows and "
        f"**{observed_days:.3f} observed days** across the four event folds. "
        "Every recall statement is a count out of four. A single miss or detection "
        "moves recall by 25 percentage points. The result is an empirical "
        "observation on one dataset, not a population estimate."
    )
    st.dataframe(
        events.rename(
            columns={
                "event_id": "Event",
                "pdf_label": "PDF label",
                "start": "Start",
                "end": "End",
                "hours": "Hours",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Start": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "End": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )

    st.subheader("2. The recommended method detects two events")
    row3 = artifacts.method_row("3", "P3_or", 1.0)
    row4 = artifacts.method_row("4", "P3_or", 1.0)
    st.markdown(
        f"At the primary 1/day budget, OR detects **event 3** "
        f"({ui.delay(row3['delay_3'])} from the label start) and **event 4** "
        f"({ui.delay(row4['delay_4'])} after the label start). It **misses events "
        "1 and 2**. Score A and Score B also miss both. Global robust z catches "
        "event 2; no primary score does. The two detections are not redundant: "
        "event 3 is A-only and event 4 is B-only, which is exactly why the union "
        "helps on this sample."
    )
    st.warning(
        "Event 1's detection-credit window is **35.2% masked** (169 of 480 "
        "minutes). Only 311 minutes can earn credit, so a miss on event 1 is not a "
        "fair test of the detector. The event-window mask share (3.40%) is a "
        "different number.",
        icon=":material/visibility_off:",
    )

    st.subheader("3. The independent tail result")
    st.markdown(
        "The tail is the final unlabelled window (31 wall days, 24.958 observed "
        "days) evaluated with the frozen fold-4 model. At a **1.0/day request**, OR "
        "realizes **0.5609 alarms/day** (14 alarms, **−43.9%** relative error). "
        "Every tail alarm is false by construction because the tail has no labels."
    )
    st.dataframe(
        _tail_table(),
        width="stretch",
        hide_index=True,
        column_config={
            "Requested / day": st.column_config.NumberColumn(format="%.1f"),
            "Realized / day": st.column_config.NumberColumn(format="%.4f"),
            "Relative error": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    st.caption(
        "This is the independent tail result cited throughout the release. "
        "It is not the event-fold pooled rate."
    )

    st.subheader("4. A strict 1/day alarm rate cannot be guaranteed")
    st.markdown(
        "The budget solver is a **step function**, not a proportional controller. "
        "At 0.5/day and 1.0/day, OR selects the **same 14 tail alarms**. Score B "
        "selects the same 35 tail alarms at 1.0/day and 2.0/day. The pre-registered "
        "F4 criterion passes only for OR, and its margin is narrow (6.1 percentage "
        "points under the 50% limit). Do not present the conformal p-value threshold "
        "as a guaranteed rate controller. If a hard rate is required, implement a "
        "hard quota (for example, at most one alarm per 24 hours per stream) or "
        "report the empirical range."
    )
    st.markdown(
        "| Claim | Status |\n"
        "| --- | --- |\n"
        "| OR is within the pre-registered 50% tail limit at 1/day | Yes (−43.9%) |\n"
        "| OR is within 10% of 1/day | No |\n"
        "| A and AND are within 50% at 1/day | No (+52.3%) |\n"
        "| The rate is a smooth function of the request | No; it is stepwise |"
    )

    st.subheader("5. Sparse loaded-state calibration")
    cell_sizes = artifacts.load_cell_sizes()
    fold4 = cell_sizes[
        (cell_sizes["fold"] == 4)
        & (cell_sizes["score"].isin(["P1_A_hybrid", "P2_B_B1"]))
        & (cell_sizes["key"].astype(str).str.startswith("loaded"))
    ]
    st.markdown(
        "Fold 4 has **41** loaded reference rows for Score A and **30** for Score B, "
        "both below the 50-row refusal threshold. Every fold-4 loaded alarm, "
        "including the event-4 detection, therefore uses **global** calibration "
        "(`n = 13,382`), not loaded-state calibration. The event-4 detection should "
        "be read as a global-calibration detection with a loaded-state label, not "
        "as evidence that loaded-state calibration works at that support level."
    )
    st.dataframe(
        fold4.rename(
            columns={
                "fold": "Fold",
                "score": "Score",
                "level": "Level",
                "key": "Cell / state",
                "n": "Reference rows",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Source: `reports/experiments/metrics/calibration_cell_sizes.csv`. "
        "The state-level loaded rows are the sum of the weekday and weekend cells."
    )

    st.subheader("6. Time-series observations are not exchangeable")
    st.markdown(
        "The calibration p-values are **empirical**, not distribution-free. The "
        "series is chronological, autocorrelated and non-stationary; a Mondrian "
        "cell does not make minutes exchangeable. There is no confidence interval "
        "that would survive multiplicity, and no statistical significance claim. "
        "The correct language is *empirical calibration* and *empirical "
        "false-alarm control*, never *distribution-free guarantee*."
    )

    st.subheader("7. The conclusions are dataset-specific")
    st.markdown(
        "This is one air-production compressor at one site, with one sensor set and "
        "four labelled events. The recommended 1/day configuration was selected by "
        "the pre-registered fallback rule, not by a population-level optimisation. "
        "Do not assume the same alpha, cell structure or fusion choice transfers to "
        "another compressor, another sampling regime or another failure taxonomy."
    )

    st.subheader("8. High budgets are not certified")
    st.markdown(
        "The pre-registered calibration claim is for 1/day. At 4/day, **36.1% of "
        "Score B alarms** (95 of 263) use `state_low` with `n < 200`, and OR recall "
        "(2/4) is **below** Score B alone (3/4). Small cells and `state_low` alarms "
        "break the per-cell claim even when the aggregate rate looks acceptable. "
        "Use 1/day for calibrated claims, or coarsen cells and re-validate before "
        "going higher."
    )

    st.divider()
    st.subheader("What the F1–F9 passes do not mean")
    st.markdown(
        "- **F1/F2 rest on two events.** Event 3 is A-only and event 4 is B-only. "
        "Remove either and the union has no unique detection. The result is real "
        "but fragile.\n"
        "- **F3 has almost no power.** At 1/day only one event is detected by any "
        "context arm, so the 'lose two events' criterion cannot fire. It is not "
        "evidence that state conditioning is beneficial.\n"
        "- **F4 passes only for OR**, and only at the pre-registered limit. A and "
        "AND exceed the 50% tail limit at 1/day.\n"
        "- **F5 tests one budget.** It says nothing about 2/day or 4/day, where "
        "`state_low` and small cells appear.\n"
        "- **F6 measures the event window, not the detection-credit window.** "
        "Event 1's 480-minute credit window is 35.2% masked.\n"
        "- **No statistical significance.** Four events and no exchangeability "
        "mean the design cannot support a distribution-free guarantee."
    )

    st.divider()
    st.subheader("When the decision would be NO-GO")
    st.markdown(
        "The current **CONDITIONAL GO** would become **NO-GO** if any of these is a "
        "hard requirement:\n\n"
        "1. Detect events 1 and 2. A, B and OR miss both.\n"
        "2. Guarantee ≤1.0 alarm/day within 10%. OR is −43.9% on the tail.\n"
        "3. Certify loaded-state calibration. Fold 4 has 41/30 loaded rows, below "
        "the 50-row refusal threshold.\n"
        "4. Certify per-cell false-positive rates above 1/day. Small cells and "
        "`state_low` alarms break the cell-level claim.\n"
        "5. Prove statistical significance or a distribution-free guarantee. The "
        "design cannot.\n\n"
        "None of these is an implementation failure; they are limits of the evidence."
    )

    st.caption(
        "Use the sidebar to open **Model Comparison** for the measured comparison "
        "or **Overview** to return to the summary."
    )

    ui.provenance_strip()


render()
