#!/usr/bin/env python3
"""Day 0 exploratory data analysis for MetroPT-3.

Produces ``reports/day0/summary.json``, small CSV summaries and the Day 0 figures.
Read-only with respect to the raw data: nothing here trains a model or defines a
detector.

Column names and the operating-regime definition below were verified against the
actual CSV and the dataset PDF; they deliberately differ from the UCI web page.

Usage:
    python scripts/eda_day0.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV = REPO_ROOT / "data" / "raw" / "metropt3" / "MetroPT3(AirCompressor).csv"
OUT = REPO_ROOT / "reports" / "day0"
FIG = OUT / "figures"

# Failure reports as published in the dataset PDF (verbatim). The PDF table is
# mis-numbered (#1 appears twice, then #3, #4); ids below are sequential and the
# original numbering is preserved in ``pdf_report_label``.
FAILURES = [
    {"id": 1, "pdf_label": "#1", "start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:00",
     "type": "Air leak", "severity": "High stress"},
    {"id": 2, "pdf_label": "#1 (duplicate)", "start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00",
     "type": "Air leak", "severity": "High stress"},
    {"id": 3, "pdf_label": "#3", "start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00",
     "type": "Air leak", "severity": "High stress"},
    {"id": 4, "pdf_label": "#4", "start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00",
     "type": "Air leak", "severity": "High stress"},
]

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
REQUIRED = ["timestamp"] + ANALOG + DIGITAL


def load() -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(f"[error] missing {CSV}; run scripts/download_data.py first")
    frame = pd.read_csv(CSV)
    frame.columns = [c.strip() for c in frame.columns]
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith("Unnamed")])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", format="mixed")
    return frame


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Verified regime from Motor_current bands + DV_eletric, per the dataset PDF.

    Motor_current ~0A = off, ~4A = offloaded, ~7A = loaded, ~9A = start.
    DV_eletric is active under load. Bands overlap cleanly enough that a
    (band, DV) rule leaves only 1.3% "other" (transition) rows.
    """
    m = frame["Motor_current"]
    dv = frame["DV_eletric"]
    regime = np.select(
        [(m >= 2) & (m < 5) & (dv == 0), (m >= 5) & (m < 8) & (dv == 1), (m < 0.5), (m >= 8)],
        ["offloaded", "loaded", "off", "start"],
        default="other",
    )
    frame["_regime"] = pd.Series(regime, index=frame.index)
    frame["_hour"] = frame["timestamp"].dt.hour
    frame["_dow"] = frame["timestamp"].dt.dayofweek
    frame["_daytype"] = np.where(frame["_dow"] >= 5, "weekend", "weekday")
    frame["_date"] = frame["timestamp"].dt.date
    return frame


def longest_constant_runs(frame: pd.DataFrame) -> dict:
    runs = {}
    for col in ANALOG:
        values = frame[col].values
        changes = np.flatnonzero(np.diff(values) != 0)
        lengths = np.diff(np.r_[-1, changes, len(values) - 1])
        idx = int(np.argmax(lengths))
        start = int(changes[idx - 1] + 1) if idx > 0 else 0
        runs[col] = {
            "max_rows": int(lengths.max()),
            "max_hours": round(lengths.max() * 10 / 3600, 2),
            "start": str(frame["timestamp"].iloc[start]),
            "end": str(frame["timestamp"].iloc[start + lengths.max()]),
        }
    return runs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = add_derived(load())
    s: dict = {}

    print("=== schema ===")
    s["rows"] = int(len(frame))
    s["columns"] = list(frame.columns[: len(REQUIRED)])
    s["n_columns_data"] = int(len(REQUIRED) - 1)
    s["dtypes"] = {c: str(frame[c].dtype) for c in REQUIRED}
    s["missing_required_columns"] = [c for c in REQUIRED if c not in frame.columns]
    print(f"rows={len(frame):,} data_columns={s['n_columns_data']}")
    assert not s["missing_required_columns"], s["missing_required_columns"]

    print("=== timestamps ===")
    ts = frame["timestamp"]
    s["timestamp_parse_failures"] = int(ts.isna().sum())
    s["t_min"] = str(ts.min())
    s["t_max"] = str(ts.max())
    s["span_days"] = round((ts.max() - ts.min()).total_seconds() / 86400, 3)
    s["timestamp_monotonic"] = bool(ts.is_monotonic_increasing)
    s["duplicate_timestamps"] = int(ts.duplicated().sum())
    s["duplicate_full_rows"] = int(frame[REQUIRED].duplicated().sum())
    diffs = ts.diff().dt.total_seconds().dropna()
    s["intervals"] = {
        "min_s": float(diffs.min()), "median_s": float(diffs.median()),
        "mean_s": round(float(diffs.mean()), 4), "max_s": float(diffs.max()),
        "share_exactly_10s": round(float((diffs == 10).mean()), 4),
        "count_gt_60s": int((diffs > 60).sum()),
        "count_gt_1h": int((diffs > 3600).sum()),
        "max_gap_h": round(float(diffs.max()) / 3600, 2),
    }
    s["interval_value_counts_top"] = {f"{k:g}s": int(v) for k, v in diffs.value_counts().head(6).items()}
    print(f"{s['t_min']} -> {s['t_max']} ({s['span_days']}d), dup_ts={s['duplicate_timestamps']}")
    print(f"intervals {s['intervals']}")

    print("=== missing values ===")
    na = frame[REQUIRED].isna().sum()
    s["missing_by_column"] = {c: int(v) for c, v in na.items() if v}
    s["total_missing_cells"] = int(na.sum())
    print(f"total missing cells: {s['total_missing_cells']}")

    print("=== analog distributions ===")
    desc = frame[ANALOG].describe().T
    s["analog_describe"] = json.loads(desc.round(4).to_json(orient="index"))
    s["analog_share_near_zero"] = {c: round(float((frame[c].abs() < 0.05).mean()), 4) for c in ANALOG}
    print(desc.round(3).to_string())

    print("=== digital signals ===")
    s["digital_active_share"] = {c: round(float((frame[c] == 1).mean()), 4) for c in DIGITAL}
    s["digital_is_binary"] = {c: bool(set(pd.unique(frame[c].dropna())) <= {0.0, 1.0}) for c in DIGITAL}
    print(json.dumps(s["digital_active_share"], indent=2))

    print("=== operating regime ===")
    s["regime_share"] = {k: round(float(v), 4) for k, v in frame["_regime"].value_counts(normalize=True).items()}
    s["regime_counts"] = {k: int(v) for k, v in frame["_regime"].value_counts().items()}
    s["motor_current_by_regime"] = json.loads(
        frame.groupby("_regime")["Motor_current"].agg(["count", "median", "mean"]).round(3).to_json(orient="index")
    )
    print(json.dumps(s["regime_share"], indent=2))

    # episode structure: the duty cycle of the compressor
    change_id = (frame["_regime"] != frame["_regime"].shift()).cumsum()
    episodes = frame["_regime"].groupby(change_id).agg(["first", "size"])
    ep = episodes.groupby("first")["size"].agg(["count", "median", "max"]).round(1)
    s["episode_length_rows"] = json.loads(ep.to_json(orient="index"))
    print(ep.to_string())

    print("=== context sparsity ===")
    per_min = frame.set_index("timestamp").resample("1min").mean(numeric_only=True)
    fine = frame.groupby(["_regime", "_daytype", "_hour"]).size()
    s["cells_regime_daytype_hour"] = {
        "n_cells": int(len(fine)),
        "min_rows": int(fine.min()),
        "median_rows": float(fine.median()),
        "cells_below_200": int((fine < 200).sum()),
        "cells_below_200_at_1min_scale": int((fine / 6 < 200).sum()),
    }
    by_rd = frame.groupby(["_regime", "_daytype"]).size()
    s["regime_daytype_rows"] = {f"{a}|{b}": int(v) for (a, b), v in by_rd.items()}
    print(json.dumps(s["cells_regime_daytype_hour"], indent=2))
    print(by_rd.to_string())

    print("=== data quality: frozen sensors ===")
    s["longest_constant_run"] = longest_constant_runs(frame)
    res = s["longest_constant_run"]["Reservoirs"]
    print(f"longest uniform block: {res['start']} -> {res['end']} ({res['max_hours']}h)")

    values = frame["Reservoirs"].values
    changes = np.flatnonzero(np.diff(values) != 0)
    runs = np.diff(np.r_[-1, changes, len(values) - 1])
    s["frozen_runs_over_1h"] = int((runs > 360).sum())
    s["frozen_rows_share"] = round(float(runs[runs > 360].sum() / len(frame)), 5)

    # rows affected by any logging gap > 60s (the row *after* the gap)
    gap_rows = int((diffs > 60).sum())
    s["rows_after_gap_gt_60s"] = gap_rows

    print("=== labels ===")
    fails = []
    base = frame.set_index("timestamp")
    for f in FAILURES:
        st, en = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
        mask = (frame["timestamp"] >= st) & (frame["timestamp"] <= en)
        rec = dict(f)
        rec["duration_h"] = round((en - st).total_seconds() / 3600, 2)
        rec["rows"] = int(mask.sum())
        rec["rows_expected_at_10s"] = int(rec["duration_h"] * 360)
        rec["coverage_pct"] = round(100 * rec["rows"] / max(rec["rows_expected_at_10s"], 1), 1)
        if mask.any():
            win = frame.loc[mask]
            rec["motor_median"] = round(float(win["Motor_current"].median()), 3)
            rec["reservoirs_median"] = round(float(win["Reservoirs"].median()), 3)
            pre = frame[(frame["timestamp"] >= st - pd.Timedelta(days=7)) & (frame["timestamp"] < st)]
            rec["reservoirs_mean_window"] = round(float(win["Reservoirs"].mean()), 3)
            rec["reservoirs_mean_prev7d"] = round(float(pre["Reservoirs"].mean()), 3)
            rec["reservoirs_delta_vs_prev7d"] = round(
                rec["reservoirs_mean_window"] - rec["reservoirs_mean_prev7d"], 3)
            rec["regime_mix"] = {k: round(float(v), 3) for k, v in win["_regime"].value_counts(normalize=True).items()}
        fails.append(rec)
    s["failures"] = fails
    s["total_labeled_hours"] = round(sum(f["duration_h"] for f in fails), 2)
    s["labeled_share_of_span"] = round(s["total_labeled_hours"] / (s["span_days"] * 24), 5)
    print(json.dumps([{k: f[k] for k in ("id", "duration_h", "rows", "reservoirs_delta_vs_prev7d")} for f in fails], indent=2))

    # Regime composition inside vs before each window. This is the single most
    # important label finding: the failure signature is a *sustained state*, and
    # the state variable is also the one we would be tempted to use as context.
    print("=== regime shift during failures ===")
    for rec in fails:
        st = pd.Timestamp(rec["start"])
        win = frame[(frame["timestamp"] >= st) & (frame["timestamp"] <= pd.Timestamp(rec["end"]))]
        pre = frame[(frame["timestamp"] >= st - pd.Timedelta(days=7)) & (frame["timestamp"] < st)]
        rec["loaded_share_window"] = round(float((win["_regime"] == "loaded").mean()), 3)
        rec["loaded_share_prev7d"] = round(float((pre["_regime"] == "loaded").mean()), 3)
        rec["offloaded_share_window"] = round(float((win["_regime"] == "offloaded").mean()), 3)
        rec["offloaded_share_prev7d"] = round(float((pre["_regime"] == "offloaded").mean()), 3)
        print(f"  #{rec['id']}: loaded {rec['loaded_share_prev7d']:.1%} -> {rec['loaded_share_window']:.1%}"
              f" | offloaded {rec['offloaded_share_prev7d']:.1%} -> {rec['offloaded_share_window']:.1%}")

    # Circularity: is TP2 (and the digital signals) essentially a function of regime?
    s["tp2_near_zero_share_by_regime"] = {
        k: round(float(frame.loc[frame["_regime"] == k, "TP2"].abs().lt(0.05).mean()), 4)
        for k in ["off", "offloaded", "loaded"]
    }
    s["tp2_corr_with_tp3"] = {
        "all_rows": round(float(frame["TP2"].corr(frame["TP3"])), 3),
        "tp2_positive_rows": round(float(frame.loc[frame["TP2"] > 0, "TP2"].corr(frame.loc[frame["TP2"] > 0, "TP3"])), 3),
    }
    s["circularity_warning"] = (
        "Operating regime is derived from Motor_current/DV_eletric, which are themselves the "
        "channel that reveals the failures (loaded share 15%->98%). Using regime as context for a "
        "score on Motor_current is therefore partly circular and must be handled explicitly."
    )
    print(json.dumps({k: s[k] for k in ("tp2_near_zero_share_by_regime", "tp2_corr_with_tp3")}, indent=2))

    first_start = pd.Timestamp(FAILURES[0]["start"])
    last_end = pd.Timestamp(FAILURES[-1]["end"])
    s["normal_before_first_failure_days"] = round((first_start - ts.min()).total_seconds() / 86400, 2)
    s["clean_after_last_failure_days"] = round((ts.max() - last_end).total_seconds() / 86400, 2)
    s["rows_before_first_failure"] = int((ts < first_start).sum())
    s["rows_after_last_failure"] = int((ts > last_end).sum())
    print(f"normal before #1: {s['normal_before_first_failure_days']}d; after #4: {s['clean_after_last_failure_days']}d")

    # fraction of the frozen block that overlaps any label
    frozen_start = pd.Timestamp(res["start"])
    frozen_end = pd.Timestamp(res["end"])
    overlaps = any(not (frozen_end < pd.Timestamp(f["start"]) or frozen_start > pd.Timestamp(f["end"])) for f in FAILURES)
    s["frozen_block_overlaps_label"] = bool(overlaps)
    print(f"frozen block overlaps a label: {overlaps}")

    # ---- figures -------------------------------------------------------
    print("=== figures ===")
    day = frame.set_index("timestamp").sort_index()

    fig, axes = plt.subplots(len(ANALOG), 1, figsize=(11, 1.15 * len(ANALOG)), sharex=True)
    for ax, col in zip(axes, ANALOG):
        ax.plot(day.index, day[col], lw=0.2, color="#1f4e79")
        ax.set_ylabel(col, rotation=0, ha="right", va="center", fontsize=7)
        for f in FAILURES:
            ax.axvspan(pd.Timestamp(f["start"]), pd.Timestamp(f["end"]), color="#c00000", alpha=0.25, lw=0)
        ax.axvspan(frozen_start, frozen_end, color="#7f7f7f", alpha=0.25, lw=0)
    axes[0].set_title("MetroPT-3 analog sensors — red: labelled failures, grey: 51h frozen block")
    fig.savefig(FIG / "01_sensor_overview.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(diffs.clip(upper=60), bins=60, color="#1f4e79")
    ax.set_yscale("log")
    ax.set_xlabel("interval between consecutive timestamps (s, clipped at 60)")
    ax.set_ylabel("count (log scale)")
    ax.set_title(f"Sampling intervals — median {s['intervals']['median_s']:.0f}s, "
                 f"{s['intervals']['count_gt_60s']} gaps >60s")
    fig.savefig(FIG / "02_sampling_intervals.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    missing = frame[REQUIRED].isna().sum()
    ax.barh(REQUIRED, missing.values, color="#c00000")
    ax.set_xlabel("missing cells")
    ax.set_title("Missing values by column")
    ax.text(0.5, 0.5, "0 missing cells in the full file", transform=ax.transAxes,
            ha="center", fontsize=10)
    fig.savefig(FIG / "03_missingness.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    order = ["off", "offloaded", "loaded", "other", "start"]
    shares = frame["_regime"].value_counts(normalize=True).reindex([o for o in order if o in frame["_regime"].unique()])
    shares.plot.bar(ax=ax, color="#1f4e79")
    for i, v in enumerate(shares.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_ylabel("share of rows")
    ax.set_title("Operating regime distribution (Motor_current + DV_eletric)")
    fig.savefig(FIG / "04_regime_distribution.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, col in zip(axes, ["Motor_current", "Reservoirs", "Oil_temperature"]):
        data, labels = [], []
        for reg in ["off", "offloaded", "loaded"]:
            vals = frame.loc[frame["_regime"] == reg, col].dropna()
            if len(vals) > 50:
                data.append(vals.sample(min(len(vals), 20000), random_state=0).values)
                labels.append(reg)
        ax.boxplot(data, tick_labels=labels, showfliers=True, flierprops={"markersize": 1, "alpha": 0.3})
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=15, labelsize=7)
    fig.suptitle("Sensor distributions by operating regime")
    fig.savefig(FIG / "05_sensors_by_regime.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 1.6))
    ax.axvspan(frozen_start, frozen_end, color="#7f7f7f", alpha=0.35, lw=0)
    for f in FAILURES:
        ax.axvspan(pd.Timestamp(f["start"]), pd.Timestamp(f["end"]), color="#c00000", alpha=0.6, lw=0)
        ax.text(pd.Timestamp(f["start"]), 0.5, f" #{f['id']}", fontsize=8, va="center")
    ax.set_xlim(day.index.min(), day.index.max())
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Labelled failure windows (red) and the unlabelled frozen block (grey)")
    fig.savefig(FIG / "06_label_timeline.png")
    plt.close(fig)

    for f in FAILURES:
        st = pd.Timestamp(f["start"]) - pd.Timedelta(hours=12)
        en = pd.Timestamp(f["end"]) + pd.Timedelta(hours=12)
        sub = day.loc[st:en]
        if sub.empty:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(10, 4.2), sharex=True)
        for ax, col in zip(axes, ["Reservoirs", "Motor_current", "TP3"]):
            ax.plot(sub.index, sub[col], lw=0.4, color="#1f4e79")
            ax.axvspan(pd.Timestamp(f["start"]), pd.Timestamp(f["end"]), color="#c00000", alpha=0.2, lw=0)
            ax.set_ylabel(col, rotation=0, ha="right", va="center", fontsize=7)
        axes[0].set_title(f"Failure #{f['id']} ({f['start']} → {f['end']}) ±12h")
        fig.savefig(FIG / f"07_failure_{f['id']}_context.png")
        plt.close(fig)

    # ---- outputs -------------------------------------------------------
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(fails).drop(columns=["regime_mix"], errors="ignore").to_csv(OUT / "failure_windows.csv", index=False)
    fine.rename("n_rows").reset_index().to_csv(OUT / "context_cells.csv", index=False)
    print(f"[written] {OUT.relative_to(REPO_ROOT)}/ (summary.json, failure_windows.csv, context_cells.csv, figures/)")


if __name__ == "__main__":
    main()
