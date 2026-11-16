#!/usr/bin/env python3
"""Build every Day 3 figure from frozen experiment artefacts.

The script never fits a model and never reads a threshold from test data. It
only plots the CSVs written by ``scripts/run_experiments.py`` and, when the raw
dataset is available, the channel series for context. Output is deterministic:
fixed ordering, no random sampling, ``Agg`` backend.

Usage:
    .venv/bin/python scripts/make_figures.py
    .venv/bin/python scripts/make_figures.py --reports reports/experiments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PRIMARY = ["P1_A_hybrid", "P2_B_B1", "P3_or", "P3_weighted", "P3_and"]
BASELINES = [
    "B1_global_z",
    "B2_rolling_z",
    "B3_iforest_raw",
    "B4_iforest_context",
]
LABELS = {
    "P1_A_hybrid": "Score A (hybrid)",
    "P2_B_B1": "Score B (B1)",
    "P3_or": "OR (A,B)",
    "P3_weighted": "weighted (λ=0.5)",
    "P3_and": "AND (A,B)",
    "P3_or_bonferroni": "OR Bonferroni",
    "B1_global_z": "global robust z",
    "B2_rolling_z": "rolling robust z",
    "B3_iforest_raw": "IForest raw",
    "B4_iforest_context": "IForest + context",
    "P1_A_none": "A: no context",
    "P1_A_calendar": "A: calendar",
    "P1_A_state": "A: lagged state",
    "P1_A_hybrid": "A: hybrid",
}
FOLD_ORDER = ["1", "2", "3", "4", "T"]
COLORS = {
    "P1_A_hybrid": "#1f77b4",
    "P2_B_B1": "#d62728",
    "P3_or": "#2ca02c",
    "P3_weighted": "#9467bd",
    "P3_and": "#8c564b",
    "B1_global_z": "#7f7f7f",
    "B2_rolling_z": "#bcbd22",
    "B3_iforest_raw": "#17becf",
    "B4_iforest_context": "#e377c2",
    "P1_A_none": "#bbbbbb",
    "P1_A_calendar": "#ff7f0e",
    "P1_A_state": "#2ca02c",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _event_folds(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["fold"].astype(str).isin(("1", "2", "3", "4"))].copy()


def _recall_by_budget(results: pd.DataFrame) -> pd.DataFrame:
    """Event-fold recall count per method and budget (0..4)."""
    event = _event_folds(results)
    rows: list[dict] = []
    for (method, budget), group in event.groupby(["method", "budget"], sort=True):
        recall = 0
        for _, row in group.iterrows():
            fold = str(row["fold"])
            recall += int(bool(row.get(f"event_{fold}", False)))
        rows.append(
            {
                "method": method,
                "budget": float(budget),
                "recall": recall,
                "false_per_day": float(group["false_alarms_per_day"].mean()),
                "budget_error": float(group["budget_error"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": "matplotlib"})
    plt.close(fig)


def _gap_break(timestamps: pd.Series, values: pd.Series, max_gap_minutes: int = 5) -> np.ndarray:
    """Insert NaN after data gaps so lines do not bridge unobserved intervals."""
    times = pd.to_datetime(timestamps)
    out = values.to_numpy(dtype=float).copy()
    gaps = (times.diff().dt.total_seconds() / 60.0).to_numpy(dtype=float)
    out[gaps > max_gap_minutes] = np.nan
    return out


def figure_event_traces(reports: Path, figures: Path) -> None:
    manifest = json.loads((reports / "run_manifest.json").read_text(encoding="utf-8"))
    folds = {str(row["fold_id"]): row for row in manifest["fold_table"]}
    predictions = {
        fold: _read(reports / "predictions" / f"fold_{fold}.csv") for fold in ("1", "2", "3", "4")
    }
    alarms = _read(reports / "alarms" / "alarms.csv")
    or_alarms = alarms[
        (alarms["method"] == "P3_or") & np.isclose(alarms["budget"], 1.0)
    ]

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False)
    for ax, fold in zip(axes, ("1", "2", "3", "4")):
        pred = predictions[fold]
        timestamps = pd.to_datetime(pred["timestamp"])
        info = folds[fold]
        start = pd.Timestamp(info["detection_start"])
        end = pd.Timestamp(info["detection_end"])
        ax.axvspan(start, end, color="#f2c14e", alpha=0.22, label="detection credit")
        ax.axvspan(
            pd.Timestamp(info["event_start"]),
            pd.Timestamp(info["event_end"]),
            color="#d62728",
            alpha=0.12,
            label="labelled event",
        )
        ax.plot(timestamps, _gap_break(timestamps, pred["p_a"]), lw=1.0, color=COLORS["P1_A_hybrid"], label="p (A)")
        ax.plot(timestamps, _gap_break(timestamps, pred["p_b"]), lw=1.0, color=COLORS["P2_B_B1"], label="p (B)")
        ax.plot(timestamps, _gap_break(timestamps, pred["p_or"]), lw=1.2, color=COLORS["P3_or"], label="p (OR)")
        chosen = or_alarms[or_alarms["fold"].astype(str) == fold]
        if len(chosen):
            ax.scatter(
                pd.to_datetime(chosen["timestamp"]),
                chosen["p"],
                marker="v",
                s=42,
                color="#111111",
                zorder=5,
                label="OR alarm @1/day",
            )
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.05)
        ax.set_ylabel("p-value (log)")
        ax.set_title(
            f"Fold {fold} — event {fold} test window "
            f"({pred['timestamp'].iloc[0]} → {pred['timestamp'].iloc[-1]})",
            loc="left",
            fontsize=10,
        )
        ax.grid(alpha=0.25)
        if fold == "1":
            ax.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("time")
    fig.suptitle(
        "Event traces: calibrated p-values, detection windows and 1/day OR alarms",
        fontsize=12,
    )
    _save(fig, figures / "event_traces.png")


def figure_budget_recall(results: pd.DataFrame, figures: Path) -> None:
    recall = _recall_by_budget(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in PRIMARY:
        sub = recall[recall["method"] == method].sort_values("budget")
        if sub.empty:
            continue
        axes[0].plot(
            sub["budget"], sub["recall"], marker="o", lw=1.7,
            color=COLORS.get(method), label=LABELS.get(method, method),
        )
        axes[1].plot(
            sub["budget"], sub["budget_error"], marker="o", lw=1.5,
            color=COLORS.get(method), label=LABELS.get(method, method),
        )
    for ax, title, ylabel in (
        (axes[0], "Event recall across the four event folds", "detected events (of 4)"),
        (axes[1], "Mean absolute realised-budget error", "|realised − requested| / requested"),
    ):
        ax.set_xlabel("requested alarms per observed day")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks(sorted(recall["budget"].unique()))
        ax.grid(alpha=0.25)
    axes[1].axhline(0.5, color="#666666", ls="--", lw=1, label="50% error (F4 limit)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Budget sweep on event folds (no test-set threshold tuning)", fontsize=12)
    _save(fig, figures / "budget_recall.png")


def figure_tail_budget(tail: pd.DataFrame, figures: Path) -> None:
    methods = PRIMARY + BASELINES
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    budgets = sorted(tail["budget"].unique())
    ax.plot(budgets, budgets, color="#333333", ls="--", lw=1, label="perfect control")
    for method in methods:
        sub = tail[tail["method"] == method].sort_values("budget")
        if sub.empty:
            continue
        ax.plot(
            sub["budget"], sub["realized_budget"], marker="o", lw=1.6,
            color=COLORS.get(method, "#999999"), label=LABELS.get(method, method),
        )
    ax.set_xlabel("requested alarms per observed day")
    ax.set_ylabel("realised alarms per observed day")
    ax.set_title("Frozen fold-4 model on the 31-day tail", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, figures / "tail_budget_control.png")


def figure_calibration(
    cells: pd.DataFrame,
    fold_results: pd.DataFrame,
    figures: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    methods = ["P1_A_hybrid", "P2_B_B1", "P3_or"]
    for method in methods:
        sub = cells[cells["method"] == method]
        if sub.empty:
            continue
        axes[0].scatter(
            sub["nominal_alpha"], sub["empirical_fpr"].clip(lower=1e-6),
            s=14, alpha=0.55, color=COLORS.get(method),
            label=LABELS.get(method, method),
        )
    lo, hi = 1e-6, 1.0
    axes[0].plot([lo, hi], [lo, hi], color="#333333", ls="--", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("nominal α used for the budget")
    axes[0].set_ylabel("empirical FPR outside event-related rows")
    axes[0].set_title("Per-cell empirical calibration", loc="left", fontsize=10)
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8, loc="upper left")

    event = _event_folds(fold_results)
    at_one = event[np.isclose(event["budget"], 1.0)]
    labels, values, colors = [], [], []
    for method in methods:
        sub = at_one[at_one["method"] == method]
        if sub.empty:
            continue
        labels.append(LABELS.get(method, method))
        values.append(float(sub["cal_selection_median_n"].median()))
        colors.append(COLORS.get(method))
    axes[1].bar(range(len(labels)), values, color=colors)
    axes[1].axhline(200, color="#d62728", ls="--", lw=1, label="n_min = 200")
    axes[1].axhline(50, color="#ff7f0e", ls=":", lw=1, label="n_refuse = 50")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("median effective reference n")
    axes[1].set_title("Calibration support at 1/day", loc="left", fontsize=10)
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(fontsize=8)
    fig.suptitle("Empirical false-alarm control and back-off support", fontsize=12)
    _save(fig, figures / "calibration_reliability.png")


def figure_context_arms(context: pd.DataFrame, figures: Path) -> None:
    recall = _recall_by_budget(context)
    arms = ["P1_A_none", "P1_A_calendar", "P1_A_state", "P1_A_hybrid"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm in arms:
        sub = recall[recall["method"] == arm].sort_values("budget")
        axes[0].plot(
            sub["budget"], sub["recall"], marker="o", lw=1.6,
            color=COLORS.get(arm, "#999999"), label=LABELS.get(arm, arm),
        )
    axes[0].set_xlabel("requested alarms per observed day")
    axes[0].set_ylabel("detected events (of 4)")
    axes[0].set_title("Recall by Score A context arm", loc="left", fontsize=10)
    axes[0].set_xticks(sorted(recall["budget"].unique()))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    at_one = recall[np.isclose(recall["budget"], 1.0)].set_index("method")
    labels = [LABELS.get(a, a) for a in arms if a in at_one.index]
    values = [float(at_one.loc[a, "false_per_day"]) for a in arms if a in at_one.index]
    axes[1].bar(
        range(len(labels)), values,
        color=[COLORS.get(a, "#999999") for a in arms if a in at_one.index],
    )
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("false alarms per observed day")
    axes[1].set_title("False alarms at 1/day", loc="left", fontsize=10)
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle("Context ablation on Score A (no context → calendar → state → hybrid)", fontsize=12)
    _save(fig, figures / "context_arms.png")


def figure_score_scatter(reports: Path, figures: Path) -> None:
    frames = []
    manifest = json.loads((reports / "run_manifest.json").read_text(encoding="utf-8"))
    folds = {str(row["fold_id"]): row for row in manifest["fold_table"]}
    for fold in ("1", "2", "3", "4"):
        pred = _read(reports / "predictions" / f"fold_{fold}.csv")
        pred["fold"] = fold
        info = folds[fold]
        times = pd.to_datetime(pred["timestamp"])
        pred["in_detection"] = (times >= pd.Timestamp(info["detection_start"])) & (
            times < pd.Timestamp(info["detection_end"])
        )
        frames.append(pred)
    data = pd.concat(frames, ignore_index=True)
    # Deterministic thinning for legibility.
    sample = data.iloc[::5].copy()
    x = -np.log10(np.clip(sample["p_a"].to_numpy(dtype=float), 1e-5, 1.0))
    y = -np.log10(np.clip(sample["p_b"].to_numpy(dtype=float), 1e-5, 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(x, y, gridsize=45, cmap="Blues", mincnt=1)
    axes[0].plot([0, 5], [0, 5], color="#333333", ls="--", lw=1)
    axes[0].set_xlabel("−log10 p (Score A, hybrid)")
    axes[0].set_ylabel("−log10 p (Score B, B1)")
    axes[0].set_title("All test rows, four folds", loc="left", fontsize=10)

    normal = sample[~sample["in_detection"]]
    det = sample[sample["in_detection"]]
    axes[1].scatter(
        -np.log10(np.clip(normal["p_a"], 1e-5, 1.0)),
        -np.log10(np.clip(normal["p_b"], 1e-5, 1.0)),
        s=7, alpha=0.25, color="#9ecae1", label="outside detection",
    )
    axes[1].scatter(
        -np.log10(np.clip(det["p_a"], 1e-5, 1.0)),
        -np.log10(np.clip(det["p_b"], 1e-5, 1.0)),
        s=12, alpha=0.8, color="#d62728", label="detection window",
    )
    relation = _read(reports / "metrics" / "score_relationship.csv")
    rho = float(relation["spearman_rho"].median())
    axes[1].set_xlabel("−log10 p (Score A, hybrid)")
    axes[1].set_ylabel("−log10 p (Score B, B1)")
    axes[1].set_title(f"Detection rows highlighted (median Spearman ρ={rho:.2f})", loc="left", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("Score relationship: A and B are not redundant", fontsize=12)
    _save(fig, figures / "score_scatter.png")


def figure_score_contribution(
    results: pd.DataFrame,
    alarms: pd.DataFrame,
    figures: Path,
) -> None:
    event = _event_folds(results)
    at_one = event[np.isclose(event["budget"], 1.0)]
    categories = {"A only": 0, "B only": 0, "both": 0, "neither": 0}
    for fold in ("1", "2", "3", "4"):
        a = at_one[(at_one["method"] == "P1_A_hybrid") & (at_one["fold"].astype(str) == fold)]
        b = at_one[(at_one["method"] == "P2_B_B1") & (at_one["fold"].astype(str) == fold)]
        if a.empty or b.empty:
            continue
        da = bool(a.iloc[0][f"event_{fold}"])
        db = bool(b.iloc[0][f"event_{fold}"])
        key = "both" if da and db else "A only" if da else "B only" if db else "neither"
        categories[key] += 1

    or_alarms = alarms[(alarms["method"] == "P3_or") & np.isclose(alarms["budget"], 1.0)]
    fired = or_alarms["fired_by"].value_counts().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    keys = ["A only", "B only", "both", "neither"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#bbbbbb"]
    axes[0].bar(keys, [categories[k] for k in keys], color=colors)
    axes[0].set_ylabel("labelled events (of 4)")
    axes[0].set_title("Unique detections at 1/day", loc="left", fontsize=10)
    axes[0].grid(alpha=0.25, axis="y")

    keys2 = ["A", "B"]
    axes[1].bar(keys2, [fired.get(k, 0) for k in keys2], color=["#1f77b4", "#d62728"])
    axes[1].set_ylabel("emitted OR alarms (all folds)")
    axes[1].set_title("Which score fired the union alarm", loc="left", fontsize=10)
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle("Score contribution at the primary 1/day budget", fontsize=12)
    _save(fig, figures / "score_contribution.png")


def figure_weighted_lambda(weighted: pd.DataFrame, figures: Path) -> None:
    recall = _recall_by_budget(weighted)
    recall["lambda"] = (
        recall["method"].str.extract(r"l(\d{3})")[0].astype(float) / 100.0
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for budget in sorted(recall["budget"].unique()):
        sub = recall[np.isclose(recall["budget"], budget)].sort_values("lambda")
        axes[0].plot(sub["lambda"], sub["recall"], marker="o", lw=1.5, label=f"{budget:g}/day")
        axes[1].plot(sub["lambda"], sub["budget_error"], marker="o", lw=1.5, label=f"{budget:g}/day")
    axes[0].set_xlabel("λ (weight on Score A)")
    axes[0].set_ylabel("detected events (of 4)")
    axes[0].set_title("Recall across the weighted-fusion grid", loc="left", fontsize=10)
    axes[1].set_xlabel("λ (weight on Score A)")
    axes[1].set_ylabel("mean absolute budget error")
    axes[1].set_title("Realised-budget error across the grid", loc="left", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Weighted fusion sensitivity (λ fixed at 0.5 for the primary run)", fontsize=12)
    _save(fig, figures / "weighted_lambda.png")


def figure_mask_coverage(mask_table: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [f"event {int(e)}" for e in mask_table["event_id"]]
    ax.bar(labels, mask_table["masked_share"], color="#4c78a8")
    ax.axhline(0.10, color="#d62728", ls="--", lw=1, label="F6 limit 10%")
    for i, value in enumerate(mask_table["masked_share"]):
        ax.text(i, value + 0.002, f"{value:.2%}", ha="center", fontsize=8)
    ax.set_ylabel("share of event window masked")
    ax.set_ylim(0, max(0.12, float(mask_table["masked_share"].max()) * 1.35))
    ax.set_title("Mask coverage inside the labelled event windows", loc="left", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    _save(fig, figures / "mask_coverage.png")


def figure_event_delays(results: pd.DataFrame, figures: Path) -> None:
    event = _event_folds(results)
    at_one = event[np.isclose(event["budget"], 1.0)]
    methods = PRIMARY + BASELINES
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.8 / len(methods)
    for j, method in enumerate(methods):
        sub = at_one[at_one["method"] == method]
        xs, ys = [], []
        for fold in ("1", "2", "3", "4"):
            row = sub[sub["fold"].astype(str) == fold]
            if row.empty:
                continue
            delay = row.iloc[0].get(f"delay_{fold}")
            if pd.notna(delay):
                xs.append(int(fold) - 0.4 + j * width + width / 2)
                ys.append(float(delay))
        ax.bar(xs, ys, width=width, color=COLORS.get(method, "#999999"), label=LABELS.get(method, method))
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([f"event {i}" for i in (1, 2, 3, 4)])
    ax.set_ylabel("first-alarm delay (minutes from event start)")
    ax.set_title("Detection delays at 1/day (only detected events are drawn)", loc="left", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7, ncol=3)
    _save(fig, figures / "event_delays.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default=str(REPO_ROOT / "reports" / "experiments"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    reports = Path(args.reports)
    figures = Path(args.output) if args.output else reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    metrics = reports / "metrics"
    primary = _read(metrics / "primary_results.csv")
    context = _read(metrics / "context_arms.csv")
    tail = _read(metrics / "tail_results.csv")
    cells = _read(metrics / "cell_calibration.csv")
    weighted = _read(metrics / "weighted_lambda.csv")
    mask_table = _read(metrics / "mask_coverage.csv")
    fold_results = _read(metrics / "fold_results.csv")
    alarms = _read(reports / "alarms" / "alarms.csv")

    figure_event_traces(reports, figures)
    figure_budget_recall(primary, figures)
    figure_tail_budget(tail, figures)
    figure_calibration(cells, fold_results, figures)
    figure_context_arms(context, figures)
    figure_score_scatter(reports, figures)
    figure_score_contribution(fold_results, alarms, figures)
    figure_weighted_lambda(weighted, figures)
    figure_mask_coverage(mask_table, figures)
    figure_event_delays(primary, figures)

    files = sorted(path.name for path in figures.glob("*.png"))
    print(f"Wrote {len(files)} figures to {figures}")
    for name in files:
        print(" -", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
