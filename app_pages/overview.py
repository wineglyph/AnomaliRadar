"""Overview page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_lib import artifacts
from app_lib import ui


def _headline_rows() -> pd.DataFrame:
    table = artifacts.pooled_comparison(1.0)
    wanted = [
        "P1_A_hybrid",
        "P2_B_B1",
        "P3_or",
        "P3_and",
        "P3_weighted",
        "B1_global_z",
        "B3_iforest_raw",
        "B4_iforest_context",
    ]
    table = table[table["method"].isin(wanted)].copy()
    table["method"] = pd.Categorical(table["method"], categories=wanted, ordered=True)
    return table.sort_values("method")


def render() -> None:
    ui.page_header(
        "AnomaliRadar",
        "A state-aware, empirically calibrated anomaly detector for the MetroPT-3 "
        "air-production compressor, presented as a reproducible engineering release.",
        kicker="MetroPT-3 · frozen experiment",
    )
    ui.provenance_strip()

    summary = artifacts.load_summary()
    manifest = artifacts.load_manifest()
    dataset = manifest["dataset_summary"]
    masks = dataset["masks"]
    states = dataset["state"]
    or_row = artifacts.method_row("1", "P3_or", 1.0)
    tail = artifacts.pooled_comparison(1.0)
    or_tail = tail[tail["method"] == "P3_or"].iloc[0]

    st.info(
        "**Status: CONDITIONAL GO.** The recommended OR method detects **2 of 4** "
        "labelled events at the primary 1/day budget with **1 false alarm** across "
        "the four event folds. The result is real but small-sample; the limitations "
        "are part of the result, not fine print.",
        icon=":material/verified:",
    )

    ui.metric_cards(
        [
            ("Event recall · 1/day", "2 / 4", "Events 3 and 4; events 1 and 2 are missed."),
            ("Alarms · event folds", "11", "Nine from Score B, two from Score A; 1 false."),
            ("Tail rate · 1/day", f"{or_tail['tail_per_day']:.4f}/day", "14 alarms in 24.958 observed days; all false by construction."),
            ("Falsification", "9 PASS", "F1–F9, 0 triggered. Several passes are narrow or low-power."),
        ]
    )

    st.divider()

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("What this project is")
        st.markdown(
            "AnomaliRadar is an **auditable composition** of established ideas: "
            "context-aware value scoring (Score A), confirmed-state persistence "
            "scoring (Score B), empirical Mondrian calibration with explicit "
            "back-off and refusal, a calibration-only alarm-budget solver, and a "
            "30-minute debounce. The contribution is the composition and the "
            "evaluation discipline, not a claim of newly invented algorithms."
        )
        st.markdown(
            "The scientific implementation, metrics and F1–F9 verdicts are frozen. "
            "This app reads the exported artefacts and never refits a model. "
            "Every displayed number is traceable to `reports/experiments`, and "
            "the dataset is downloaded separately under CC BY 4.0."
        )
        st.markdown(
            "**Pipeline:** native 10 s samples → 1-minute grid → quality masks "
            "M1–M3 → hysteresis-confirmed state → lagged context → chronological "
            "folds → Score A / Score B → empirical calibration → OR fusion → "
            "budget solver → 30-minute debounce → evaluation."
        )

    with right:
        st.subheader("Recommended configuration")
        recommended = summary["recommended_configuration"]
        st.dataframe(
            pd.DataFrame(
                [
                    ("Score A", recommended["score_a"]),
                    ("Score B", recommended["score_b"]),
                    ("Fusion", recommended["fusion"]),
                    ("Budget", f"{recommended['alarm_budget_per_day']:.1f} alarm / observed day"),
                    ("Debounce", f"{recommended['debounce_minutes']} min"),
                    ("Calibration", "Mondrian (lagged state, day type) → state → global"),
                ],
                columns=["Component", "Setting"],
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Selection rationale: {recommended['score_a_reason']}; "
            f"{recommended['fusion_reason']}. This is the pre-registered fallback, "
            "not a post-hoc optimum."
        )

    st.divider()
    st.subheader("Dataset at a glance")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Native rows", f"{dataset['native_rows']:,}", border=True)
    d2.metric("Grid minutes", f"{dataset['grid_rows']:,}", border=True)
    d3.metric("Observed minutes", f"{dataset['grid_observed_rows']:,}", border=True)
    d4.metric("Labelled events", f"{len(dataset['labels'])}", border=True)

    st.markdown(
        f"MetroPT-3 contains {dataset['native_rows']:,} native samples at a ~10 s "
        f"median interval, gridded to {dataset['grid_rows']:,} minutes. "
        f"{dataset['grid_observed_rows']:,} minutes are observed; "
        f"{masks['global']['grid_combined_share']:.1%} of grid minutes are masked "
        f"(M1 frozen sensors, M2 gaps > 1 h, M3 warm-up). The confirmed state "
        f"distribution is off {states['confirmed_share']['off']:.1%}, "
        f"offloaded {states['confirmed_share']['offloaded']:.1%}, "
        f"loaded {states['confirmed_share']['loaded']:.1%}, and other "
        f"{states['confirmed_share']['other']:.2%}. There are "
        f"{dataset['segments']} data segments."
    )
    st.caption(
        "Dataset: MetroPT-3, UCI Machine Learning Repository ID 791, CC BY 4.0. "
        "The CSV is never redistributed with this repository."
    )

    st.divider()
    st.subheader("Measured results at 1 alarm / observed day")
    rows = _headline_rows()
    display = rows[
        [
            "method_label",
            "recall",
            "alarms",
            "false_alarms",
            "alarms_per_day",
            "false_per_day",
            "mean_abs_budget_error",
            "tail_per_day",
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
            "tail_per_day": "Tail/day",
        }
    )
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Alarms/day": st.column_config.NumberColumn(format="%.3f"),
            "False/day": st.column_config.NumberColumn(format="%.3f"),
            "Mean |budget error|": st.column_config.NumberColumn(format="%.3f"),
            "Tail/day": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "Event-fold counts are pooled over the four chronological folds. "
        "The tail uses the frozen fold-4 model on 24.958 observed days with no "
        "labels; every tail alarm is false by construction. 'Mean |budget error|' "
        "is the mean over event folds of |realized rate − requested rate|."
    )

    fig_a, fig_b = st.columns(2)
    with fig_a:
        ui.figure_card(
            "event_traces.png",
            "Calibrated p-values in each event window. Fold 3 is an A dip; fold 4 is a B p-floor run.",
        )
    with fig_b:
        ui.figure_card(
            "budget_recall.png",
            "Recall and mean absolute budget error across requested budgets. OR leads recall; the rate response is coarse.",
        )

    st.divider()
    st.subheader("Read the result honestly")
    st.markdown(
        f"- **Only four labelled events.** At 1/day, OR finds event 3 (A, "
        f"{ui.delay(artifacts.method_row('3', 'P3_or', 1.0)['delay_3'])} from the "
        f"label start) and event 4 (B, "
        f"{ui.delay(artifacts.method_row('4', 'P3_or', 1.0)['delay_4'])} after it). "
        "Events 1 and 2 are missed by A, B, OR, weighted and AND.\n"
        f"- **The tail rate is {or_tail['tail_per_day']:.4f}/day at a 1/day request "
        f"({or_tail['tail_relative_error']:+.1%}).** 0.5/day and 1/day select the "
        "same 14 tail alarms: the budget response is a step function, not a "
        "proportional controller.\n"
        "- **Fold 4's loaded calibration is thin.** Score A has 41 loaded reference "
        "rows and Score B has 30, both below the 50-row refusal threshold; event 4 "
        "is therefore detected with global calibration, not loaded-state calibration.\n"
        "- **Event 1's detection credit is 35.2% masked.** Only 311 of 480 credit "
        "minutes are usable, so a miss there is not a fair test of the detector.\n"
        "- **No distribution-free guarantee.** Calibration is empirical; the series "
        "is not exchangeable and four events cannot support a tight false-alarm "
        "claim or statistical significance."
    )

    st.caption(
        "Use the sidebar to open **Limitations** for the full disclosure and "
        "**Model Comparison** for every method."
    )

    st.divider()
    st.subheader("Reproduce it")
    st.code(
        "python -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -e \".[app,dev]\"\n"
        "python scripts/download_data.py\n"
        "python scripts/run_experiments.py --output reports/experiments\n"
        "python scripts/make_figures.py\n"
        "pytest -q\n"
        "streamlit run app.py",
        language="bash",
    )
    st.caption(
        f"Frozen identity: config hash `{summary['config_hash']}`, dataset "
        f"fingerprint `{manifest['dataset_fingerprint']}`."
    )


render()
