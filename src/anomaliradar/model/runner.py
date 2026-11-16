"""End-to-end Day 3 experiment runner.

Execution order is fixed by ``docs/experimental_protocol.md`` §7:

1. data layer (already verified Day 2);
2. Score A per context arm, fit on train only;
3. Score B survival reference from the calibration reference part only;
4. Mondrian calibration with chronological reference/selection separation (C1);
5. calibration-only budget solver, 30-minute debounce;
6. baselines through the same calibration and alarm layer (C8);
7. frozen fold-4 model on the 31-day tail;
8. ablations, diagnostics, falsification and artefact writing.

The runner never passes labels to fitting, calibration or budget solving. Test
labels are read only after every threshold for that fold has been frozen.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from anomaliradar.data.labels import EventWindow, label_mask
from anomaliradar.data.loader import default_dataset_path, load_native
from anomaliradar.data.pipeline import Dataset, build_dataset
from anomaliradar.data.splits import (
    FoldSpec,
    SplitIndices,
    materialise_split,
)
from anomaliradar.model.alarm import BudgetSolution, emit_alarms, solve_budget
from anomaliradar.model.baselines import (
    build_rolling_z_components,
    fit_global_z,
    fit_iforest,
    iforest_scores,
)
from anomaliradar.model.calibration import (
    CalibrationResult,
    CalibrationSplit,
    ConformalCalibrator,
    chronological_split,
)
from anomaliradar.model.config import (
    CONTEXT_ARMS,
    ExperimentConfig,
    config_json,
)
from anomaliradar.model.evaluation import (
    EvaluationAccumulator,
    alarm_volume,
    budget_error,
    event_related_mask,
    exposure_days,
    spearman_scores,
)
from anomaliradar.model.fusion import FUSION_LAMBDAS, fuse
from anomaliradar.model.score_a import (
    ScoreAModel,
    build_score_a_features,
    fit_score_a,
    score_a_frame,
)
from anomaliradar.model.score_b import (
    RunTable,
    ScoreBReference,
    build_run_table,
    fit_score_b_reference,
    reference_run_rows,
    score_b_series,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "experiments"

A_ARMS = tuple(CONTEXT_ARMS)
B_VARIANTS = ("B1", "B3")
BASELINE_KEYS = (
    "B1_global_z",
    "B2_rolling_z",
    "B3_iforest_raw",
    "B4_iforest_context",
)
PRIMARY_A = "P1_A_hybrid"
PRIMARY_B = "P2_B_B1"
PRIMARY_OR = "P3_or"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = dataset.context["context_cell"].astype(object).to_numpy()
    lag_state = dataset.context["state_lag1"].astype(object).to_numpy()
    state = dataset.state["state"].astype(object).to_numpy()
    return cell, lag_state, state


def _make_split(
    dataset: Dataset,
    fold: FoldSpec,
    *,
    mask_override: np.ndarray | None = None,
) -> SplitIndices:
    if mask_override is None:
        return dataset.split(fold)
    return materialise_split(
        dataset.index,
        fold,
        mask=mask_override,
        usable=dataset.usable,
        transition=dataset.transition,
        context_ready=dataset.context_ready,
        events=dataset.events,
    )


def _hygienic_split(
    dataset: Dataset,
    fold: FoldSpec,
    split: SplitIndices,
    runs: pd.DataFrame,
    *,
    min_loaded_minutes: int = 120,
) -> SplitIndices:
    """Drop loaded runs longer than 2 h that intersect the calibration window."""
    drop = np.zeros(len(dataset.index), dtype=bool)
    for row in runs.itertuples(index=False):
        if row.state != "loaded" or row.length < min_loaded_minutes:
            continue
        start_ts = dataset.index[int(row.start)]
        end_ts = dataset.index[int(row.end)]
        if end_ts < fold.calibration.start or start_ts >= fold.calibration.end:
            continue
        drop[int(row.start) : int(row.end) + 1] = True
    calibration = split.calibration[~drop[split.calibration]]
    return replace(split, calibration=calibration)


@dataclass(slots=True)
class Layer:
    p: np.ndarray
    eligible: np.ndarray
    level: np.ndarray
    n: np.ndarray
    depth: np.ndarray
    fired_by: np.ndarray
    raw: np.ndarray


def _layer_from(
    result: CalibrationResult, fired: str, raw: np.ndarray
) -> Layer:
    return Layer(
        p=np.asarray(result.p, dtype=float),
        eligible=np.asarray(result.eligible, dtype=bool),
        level=np.asarray(result.level, dtype=object),
        n=np.asarray(result.n, dtype=int),
        depth=np.asarray(result.depth, dtype=int),
        fired_by=np.full(result.p.shape[0], fired, dtype=object),
        raw=np.asarray(raw, dtype=float),
    )


def _fuse_layers(a: Layer, b: Layer, rule: str, lambda_: float = 0.5) -> Layer:
    result = fuse(
        a.p,
        b.p,
        rule=rule,
        lambda_=lambda_,
        eligible_a=a.eligible,
        eligible_b=b.eligible,
    )
    if rule in ("or", "or_bonferroni"):
        eligible = a.eligible | b.eligible
    elif rule in ("and", "weighted"):
        eligible = a.eligible & b.eligible
    else:
        eligible = a.eligible if rule == "A" else b.eligible

    choose_a = a.p <= b.p
    choose_a = np.where(a.eligible & ~b.eligible, True, choose_a)
    choose_a = np.where(b.eligible & ~a.eligible, False, choose_a)
    # If neither side is eligible the fused row is refused; provenance is A for
    # a stable report, but the row can never fire.
    choose_a = np.where(~a.eligible & ~b.eligible, True, choose_a)

    level = np.where(choose_a, a.level, b.level)
    n = np.where(choose_a, a.n, b.n)
    depth = np.where(choose_a, a.depth, b.depth)
    raw = np.where(choose_a, a.raw, b.raw)
    fired = np.where(choose_a, "A", "B").astype(object)
    return Layer(
        p=result.p,
        eligible=eligible,
        level=level,
        n=n,
        depth=depth,
        fired_by=fired,
        raw=raw,
    )


def _calibrate(
    score: np.ndarray,
    cell: np.ndarray,
    lag_state: np.ndarray,
    reference_mask: np.ndarray,
    config: ExperimentConfig,
) -> tuple[ConformalCalibrator, CalibrationResult]:
    calibrator = ConformalCalibrator(config.calibration).fit(
        score, cell, lag_state, reference_mask
    )
    result = calibrator.p_values(score, cell, lag_state)
    return calibrator, result


@dataclass(slots=True)
class Bundle:
    fold_id: str
    split: SplitIndices
    cal_split: CalibrationSplit
    selection_positions: np.ndarray
    layers: dict[str, Layer]
    a_scored: dict[str, pd.DataFrame]
    b_scored: dict[str, pd.DataFrame]
    a_models: dict[str, ScoreAModel]
    b_reference: ScoreBReference
    baseline_raw: dict[str, np.ndarray]
    solutions: dict[tuple[str, float], BudgetSolution] = field(default_factory=dict)
    calibration_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Bundle construction
# --------------------------------------------------------------------------- #
def build_bundle(
    dataset: Dataset,
    fold: FoldSpec,
    split: SplitIndices,
    *,
    config: ExperimentConfig,
    features: dict[str, pd.DataFrame],
    rolling_components,
    runs: RunTable,
    arms: Sequence[str] = A_ARMS,
    variants: Sequence[str] = B_VARIANTS,
    with_baselines: bool = True,
    verbose: bool = False,
) -> Bundle:
    cell, lag_state, _ = _arrays(dataset)
    grid = dataset.grid.frame
    context = dataset.context
    index = dataset.index
    cal_split = chronological_split(
        split.calibration,
        index,
        fraction=config.calibration.reference_fraction,
    )
    reference = cal_split.reference
    selection_positions = split.calibration[cal_split.selection[split.calibration]]
    layers: dict[str, Layer] = {}
    a_scored: dict[str, pd.DataFrame] = {}
    b_scored: dict[str, pd.DataFrame] = {}
    a_models: dict[str, ScoreAModel] = {}
    calibration_tables: dict[str, pd.DataFrame] = {}

    for arm in arms:
        if verbose:
            print(f"  Score A arm={arm}", flush=True)
        model = fit_score_a(
            features,
            context,
            grid,
            split.train,
            arm=arm,
            config=config.score_a,
        )
        scored = score_a_frame(
            model,
            features,
            context,
            grid_frame=grid,
            segment=grid["segment_id"],
        )
        a_models[arm] = model
        a_scored[arm] = scored
        calibrator, result = _calibrate(
            scored["a"].to_numpy(dtype=float),
            cell,
            lag_state,
            reference,
            config,
        )
        calibration_tables[f"P1_A_{arm}"] = calibrator.cell_table()
        layers[f"P1_A_{arm}"] = _layer_from(
            result, "A", scored["a"].to_numpy(dtype=float)
        )

    b_reference: ScoreBReference | None = None
    for variant in variants:
        if verbose:
            print(f"  Score B variant={variant}", flush=True)
        bcfg = replace(config.score_b, variant=variant)
        reference_dist = fit_score_b_reference(runs, reference, bcfg)
        if variant == config.primary_score_b:
            b_reference = reference_dist
        b_calibration_rows = reference_run_rows(runs, reference)
        scored = score_b_series(
            dataset.state,
            reference_dist,
            bcfg,
            loo_mask=b_calibration_rows,
            runs=runs,
        )
        b_scored[variant] = scored
        calibrator, result = _calibrate(
            scored["score_b"].to_numpy(dtype=float),
            cell,
            lag_state,
            b_calibration_rows,
            config,
        )
        calibration_tables[f"P2_B_{variant}"] = calibrator.cell_table()
        layers[f"P2_B_{variant}"] = _layer_from(
            result, "B", scored["score_b"].to_numpy(dtype=float)
        )
    if b_reference is None:
        bcfg = replace(config.score_b, variant=config.primary_score_b)
        b_reference = fit_score_b_reference(runs, reference, bcfg)

    baseline_raw: dict[str, np.ndarray] = {}
    if with_baselines:
        if verbose:
            print("  baselines", flush=True)
        gz = fit_global_z(grid, split.train, config=config.baseline)
        baseline_raw["B1_global_z"] = gz.score(grid)
        baseline_raw["B2_rolling_z"] = rolling_components.score(
            grid, split.train
        )
        ifr = fit_iforest(
            grid,
            context,
            split.train,
            with_context=False,
            config=config.baseline,
        )
        baseline_raw["B3_iforest_raw"] = iforest_scores(
            ifr, grid, context, with_context=False
        )
        ifc = fit_iforest(
            grid,
            context,
            split.train,
            with_context=True,
            config=config.baseline,
        )
        baseline_raw["B4_iforest_context"] = iforest_scores(
            ifc, grid, context, with_context=True
        )
        for key, raw in baseline_raw.items():
            calibrator, result = _calibrate(raw, cell, lag_state, reference, config)
            calibration_tables[key] = calibrator.cell_table()
            layers[key] = _layer_from(result, "A", raw)

    # Fusion of the primary scores. Other arms/variants are ablations and are
    # never combined into the primary result.
    a = layers[PRIMARY_A]
    b = layers[PRIMARY_B]
    for rule in ("or", "or_bonferroni", "and", "weighted"):
        layers[f"P3_{rule}"] = _fuse_layers(a, b, rule)
    for lam in FUSION_LAMBDAS:
        key = f"P3_weighted_l{int(round(lam * 100)):03d}"
        layers[key] = _fuse_layers(a, b, "weighted", lambda_=lam)

    return Bundle(
        fold_id=fold.fold_id,
        split=split,
        cal_split=cal_split,
        selection_positions=selection_positions,
        layers=layers,
        a_scored=a_scored,
        b_scored=b_scored,
        a_models=a_models,
        b_reference=b_reference,
        baseline_raw=baseline_raw,
        calibration_tables=calibration_tables,
    )


# --------------------------------------------------------------------------- #
# Alarm records and evaluation
# --------------------------------------------------------------------------- #
def _calibration_info(
    layer: Layer,
    selection: np.ndarray,
    test: np.ndarray,
    alarm_local: np.ndarray,
    bundle: Bundle,
) -> dict:
    sel_elig = layer.eligible[selection]
    n_sel = layer.n[selection][sel_elig]
    levels = pd.Series(layer.level[selection]).value_counts(dropna=False)
    alarm_levels = (
        pd.Series(layer.level[test][alarm_local]).value_counts(dropna=False)
        if alarm_local.size
        else pd.Series(dtype=int)
    )
    return {
        "reference_rows": int(bundle.cal_split.reference_rows),
        "selection_rows": int(selection.size),
        "selection_refused": int((~sel_elig).sum()),
        "selection_min_n": int(n_sel.min()) if n_sel.size else 0,
        "selection_median_n": float(np.median(n_sel)) if n_sel.size else 0.0,
        "selection_levels": json.dumps({str(k): int(v) for k, v in levels.items()}),
        "test_refused": int((~layer.eligible[test]).sum()),
        "alarm_n_lt_50": int((layer.n[test][alarm_local] < 50).sum())
        if alarm_local.size
        else 0,
        "alarm_n_lt_200": int((layer.n[test][alarm_local] < 200).sum())
        if alarm_local.size
        else 0,
        "alarm_max_depth": int(layer.depth[test][alarm_local].max())
        if alarm_local.size
        else -1,
        "alarm_levels": json.dumps(
            {str(k): int(v) for k, v in alarm_levels.items()}
        ),
    }


def _alarm_frame(
    *,
    dataset: Dataset,
    fold: FoldSpec,
    method: str,
    budget: float,
    alarm_positions_local: np.ndarray,
    test_positions: np.ndarray,
    layer: Layer,
    bundle: Bundle,
    target_positions: Sequence[int] | None = None,
) -> pd.DataFrame:
    if alarm_positions_local.size == 0:
        return pd.DataFrame(
            columns=[
                "fold",
                "method",
                "budget",
                "timestamp",
                "position",
                "p",
                "alpha",
                "p_a",
                "p_b",
                "fired_by",
                "cell",
                "cal_level",
                "cal_n",
                "cal_depth",
                "channel",
                "state",
                "run_length_min",
                "context_effect",
                "value_effect",
                "event_related",
            ]
        )
    global_positions = np.asarray(test_positions, dtype=int)[alarm_positions_local]
    timestamps = dataset.index[global_positions]
    cell, _, state = _arrays(dataset)
    p_a = bundle.layers[PRIMARY_A].p
    p_b = bundle.layers[PRIMARY_B].p

    arm = "hybrid"
    if method in bundle.a_scored:
        arm = method.removeprefix("P1_A_")
    scored_a = bundle.a_scored.get(arm)
    if scored_a is not None:
        channel = scored_a["a_channel"].to_numpy(dtype=object)[global_positions]
        context_effect = scored_a["context_effect"].to_numpy(dtype=float)[
            global_positions
        ]
    else:
        channel = np.full(global_positions.size, "", dtype=object)
        context_effect = np.zeros(global_positions.size, dtype=float)

    if method.startswith("P1_A_"):
        method_pa = layer.p[global_positions]
        method_pb = np.ones(global_positions.size)
    elif method.startswith("P2_B_"):
        method_pa = np.ones(global_positions.size)
        method_pb = layer.p[global_positions]
    elif method.startswith("P3_") or method.startswith("Mhyg_") or method.startswith("Mnone_"):
        method_pa = p_a[global_positions]
        method_pb = p_b[global_positions]
    else:
        method_pa = np.full(global_positions.size, np.nan)
        method_pb = np.full(global_positions.size, np.nan)

    events = tuple(e for e in dataset.events if fold.event_id is not None)
    related = (
        event_related_mask(pd.DatetimeIndex(timestamps), events)
        if events
        else np.zeros(global_positions.size, dtype=bool)
    )

    return pd.DataFrame(
        {
            "fold": fold.fold_id,
            "method": method,
            "budget": budget,
            "timestamp": timestamps,
            "position": global_positions,
            "p": layer.p[global_positions],
            "alpha": layer.p[global_positions] * 0 + np.nan,
            "p_a": method_pa,
            "p_b": method_pb,
            "fired_by": layer.fired_by[global_positions],
            "cell": cell[global_positions],
            "cal_level": layer.level[global_positions],
            "cal_n": layer.n[global_positions],
            "cal_depth": layer.depth[global_positions],
            "channel": channel,
            "state": state[global_positions],
            "run_length_min": dataset.state["run_length_min"].to_numpy(dtype=float)[
                global_positions
            ],
            "context_effect": context_effect,
            "value_effect": layer.raw[global_positions],
            "event_related": related,
        }
    )


def evaluate_bundle(
    dataset: Dataset,
    fold: FoldSpec,
    bundle: Bundle,
    *,
    config: ExperimentConfig,
    accumulator: EvaluationAccumulator,
    methods: Sequence[str] | None = None,
    n_events: int | None = 4,
    tag_prefix: str = "",
    alarm_store: list[pd.DataFrame] | None = None,
    solve: bool = True,
    long_intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> None:
    index = dataset.index
    split = bundle.split
    test = split.test
    test_index = index[test]
    events = tuple(dataset.events) if fold.event_id is not None else tuple()
    outside_event = (
        ~event_related_mask(test_index, events)
        if events
        else np.ones(len(test_index), dtype=bool)
    )
    cell, _, _ = _arrays(dataset)
    selected = bundle.selection_positions

    chosen_methods = list(methods) if methods is not None else list(bundle.layers)
    for method in chosen_methods:
        layer = bundle.layers[method]
        for budget in config.alarm.budgets:
            key = (method, float(budget))
            if key in bundle.solutions:
                solution = bundle.solutions[key]
            else:
                solution = solve_budget(
                    index[selected],
                    layer.p[selected],
                    budget,
                    config.alarm,
                    exposure_days=exposure_days(index[selected]),
                    eligible=layer.eligible[selected],
                )
                if solve:
                    bundle.solutions[key] = solution
            alarms = emit_alarms(
                test_index,
                layer.p[test],
                solution,
                config.alarm,
                eligible=layer.eligible[test],
            )
            alarm_df = _alarm_frame(
                dataset=dataset,
                fold=fold,
                method=f"{tag_prefix}{method}",
                budget=budget,
                alarm_positions_local=alarms.positions,
                test_positions=test,
                layer=layer,
                bundle=bundle,
            )
            if len(alarm_df):
                alarm_df["alpha"] = solution.alpha
            if alarm_store is not None and len(alarm_df):
                alarm_store.append(alarm_df)

            info = _calibration_info(
                layer, selected, test, alarms.positions, bundle
            )
            extra = {
                "budget_reachable": solution.reachable,
                "budget_coarse": solution.coarse,
                "calibration_floor_rate": solution.floor_rate,
                "calibration_alarms": solution.calibration_alarms,
                "calibration_rate": solution.calibration_rate,
                "calibration_target_alarms": solution.target_alarms,
                "calibration_absolute_error": solution.absolute_error,
                "calibration_relative_error": (
                    (solution.calibration_rate - budget) / budget if budget else None
                ),
                "calibration_within_50pct": bool(
                    abs(solution.calibration_rate - budget) / budget <= 0.5
                    if budget
                    else False
                ),
                "debiased": True,
            }
            if events:
                intervals = (
                    _long_loaded_intervals(dataset)
                    if long_intervals is None
                    else long_intervals
                )
                in_long = _times_in_intervals(
                    pd.DatetimeIndex(alarm_df["timestamp"]) if len(alarm_df) else pd.DatetimeIndex([]),
                    intervals,
                )
                strict_false = int(
                    (~alarm_df["event_related"]).sum() if len(alarm_df) else 0
                )
                extra.update(
                    {
                        "false_alarms_excl_long_loaded": int(
                            max(strict_false - int(in_long.sum()), 0)
                        ),
                    }
                )
            accumulator.add(
                fold=fold.fold_id,
                method=f"{tag_prefix}{method}",
                budget=budget,
                alarms=alarm_df,
                observed_index=test_index,
                events=events,
                p=layer.p[test],
                cell=cell[test],
                outside_event=outside_event,
                alpha=solution.alpha,
                calibration={**info, **extra},
                n_events=n_events,
            )


def _long_loaded_intervals(
    dataset: Dataset,
    *,
    min_minutes: int = 120,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Long loaded runs outside every labelled window (contamination sensitivity)."""
    labelled = label_mask(dataset.index, dataset.events)
    run_table = build_run_table(dataset.state)
    runs = run_table.frame
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for row in runs.itertuples(index=False):
        if row.state != "loaded" or row.length < min_minutes:
            continue
        start, end = int(row.start), int(row.end)
        if labelled[start : end + 1].any():
            continue
        intervals.append((dataset.index[start], dataset.index[end]))
    return intervals


def _times_in_intervals(
    times: pd.DatetimeIndex,
    intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times <= end)
    return mask


# --------------------------------------------------------------------------- #
# Predictions, context diagnostics, figures data
# --------------------------------------------------------------------------- #
def _save_predictions(
    dataset: Dataset,
    fold: FoldSpec,
    bundle: Bundle,
    *,
    out_dir: Path,
) -> None:
    split = bundle.split
    test = split.test
    cell, _, _ = _arrays(dataset)
    layers = bundle.layers
    pred = pd.DataFrame(
        {
            "timestamp": dataset.index[test],
            "a_hybrid": layers[PRIMARY_A].raw[test],
            "a_none": layers["P1_A_none"].raw[test]
            if "P1_A_none" in layers
            else np.nan,
            "score_b": layers[PRIMARY_B].raw[test],
            "p_a": layers[PRIMARY_A].p[test],
            "p_b": layers[PRIMARY_B].p[test],
            "p_or": layers[PRIMARY_OR].p[test],
            "p_and": layers["P3_and"].p[test],
            "p_weighted": layers["P3_weighted"].p[test],
            "cell": cell[test],
            "cal_level_a": layers[PRIMARY_A].level[test],
            "cal_n_a": layers[PRIMARY_A].n[test],
            "cal_level_b": layers[PRIMARY_B].level[test],
            "cal_n_b": layers[PRIMARY_B].n[test],
            "cal_level_or": layers[PRIMARY_OR].level[test],
            "cal_n_or": layers[PRIMARY_OR].n[test],
        }
    )
    pred.to_csv(out_dir / "predictions" / f"fold_{fold.fold_id}.csv", index=False)


def _context_diagnostics(
    dataset: Dataset,
    fold: FoldSpec,
    bundle: Bundle,
) -> list[dict]:
    rows: list[dict] = []
    if fold.event_id is None:
        return rows
    event = next(e for e in dataset.events if e.event_id == fold.event_id)
    window = event.detection_window()
    in_window = window.mask(dataset.index)
    p_none = bundle.layers.get("P1_A_none")
    for arm in A_ARMS:
        key = f"P1_A_{arm}"
        if key not in bundle.layers:
            continue
        layer = bundle.layers[key]
        scored = bundle.a_scored[arm]
        values = scored["a"].to_numpy(dtype=float)[in_window]
        effect = scored["context_effect"].to_numpy(dtype=float)[in_window]
        suppressed = 0
        if p_none is not None and arm != "none":
            suppressed = int(
                ((p_none.p[in_window] <= 0.05) & (layer.p[in_window] > 0.05)).sum()
            )
        rows.append(
            {
                "fold": fold.fold_id,
                "event_id": event.event_id,
                "arm": arm,
                "detection_rows": int(in_window.sum()),
                "mean_a_in_detection": float(np.nanmean(values)) if values.size else np.nan,
                "max_a_in_detection": float(np.nanmax(values)) if values.size else np.nan,
                "mean_context_effect": float(np.nanmean(effect)) if effect.size else 0.0,
                "max_context_effect": float(np.nanmax(effect)) if effect.size else 0.0,
                "suppressed_rows_p005": suppressed,
                "mean_a_delta_vs_none": (
                    float(np.nanmean(values) - np.nanmean(p_none.raw[in_window]))
                    if p_none is not None and arm != "none"
                    else 0.0
                ),
            }
        )
    return rows


def _score_relationship(
    dataset: Dataset,
    fold: FoldSpec,
    bundle: Bundle,
) -> list[dict]:
    rows: list[dict] = []
    selection = bundle.selection_positions
    if selection.size == 0:
        return rows
    a = bundle.layers[PRIMARY_A].raw[selection]
    b = bundle.layers[PRIMARY_B].raw[selection]
    rho = spearman_scores(a, b)
    # Top-k Jaccard overlap, k=20, methodology §9. The top set is the k
    # smallest calibrated p-values (most anomalous), within the selection rows.
    k = min(20, selection.size)
    p_a = bundle.layers[PRIMARY_A].p[selection]
    p_b = bundle.layers[PRIMARY_B].p[selection]
    top_a = set(np.argsort(p_a, kind="stable")[:k].tolist())
    top_b = set(np.argsort(p_b, kind="stable")[:k].tolist())
    jaccard = len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else 0.0
    rows.append(
        {
            "fold": fold.fold_id,
            "spearman_rho": rho,
            "top20_jaccard": jaccard,
            "selection_rows": int(selection.size),
        }
    )
    return rows


# --------------------------------------------------------------------------- #
# Falsification
# --------------------------------------------------------------------------- #
def _recall(
    results: pd.DataFrame,
    method: str,
    budget: float,
    folds: Iterable[str] = ("1", "2", "3", "4"),
) -> int:
    sub = results[
        (results["method"] == method)
        & (np.isclose(results["budget"], budget))
        & (results["fold"].astype(str).isin(tuple(folds)))
    ]
    total = 0
    for _, row in sub.iterrows():
        fold = str(row["fold"])
        total += int(bool(row.get(f"event_{fold}", False)))
    return total


def _budget_error_mean(
    results: pd.DataFrame,
    method: str,
    folds: Iterable[str] = ("1", "2", "3", "4"),
) -> float:
    sub = results[
        (results["method"] == method)
        & (results["fold"].astype(str).isin(tuple(folds)))
    ]
    if sub.empty:
        return float("nan")
    return float(sub["budget_error"].abs().mean())


def compute_falsification(
    *,
    results: pd.DataFrame,
    alarms: pd.DataFrame,
    mask_table: pd.DataFrame,
    relationship: pd.DataFrame,
) -> dict:
    budgets = sorted(results["budget"].unique())
    primary_budget = 1.0

    def row(method: str, budget: float, fold: str) -> pd.Series | None:
        sub = results[
            (results["method"] == method)
            & (np.isclose(results["budget"], budget))
            & (results["fold"].astype(str) == fold)
        ]
        return None if sub.empty else sub.iloc[0]

    # F1: unique detections at the primary budget (all four events).
    unique = []
    for fold in ("1", "2", "3", "4"):
        ra = row(PRIMARY_A, primary_budget, fold)
        rb = row(PRIMARY_B, primary_budget, fold)
        if ra is None or rb is None:
            continue
        a_det = bool(ra.get(f"event_{fold}", False))
        b_det = bool(rb.get(f"event_{fold}", False))
        if a_det != b_det:
            unique.append(fold)
    f1 = {
        "id": "F1",
        "criterion": "Two-score union has no unique detections",
        "primary_budget": primary_budget,
        "unique_events": unique,
        "triggered": len(unique) == 0,
        "verdict": "FAIL" if len(unique) == 0 else "PASS",
        "by_budget": {
            str(b): [
                f
                for f in ("1", "2", "3", "4")
                if (row(PRIMARY_A, b, f) is not None and row(PRIMARY_B, b, f) is not None)
                and bool(row(PRIMARY_A, b, f).get(f"event_{f}", False))
                != bool(row(PRIMARY_B, b, f).get(f"event_{f}", False))
            ]
            for b in budgets
        },
    }

    # F2: union recall gain over best single score at >= 2 budgets.
    gains = {}
    for b in budgets:
        r_a = _recall(results, PRIMARY_A, b)
        r_b = _recall(results, PRIMARY_B, b)
        r_or = _recall(results, PRIMARY_OR, b)
        gains[str(b)] = {
            "A": r_a,
            "B": r_b,
            "OR": r_or,
            "gain": r_or - max(r_a, r_b),
        }
    gain_budgets = [b for b, v in gains.items() if v["gain"] > 0]
    f2 = {
        "id": "F2",
        "criterion": "Union does not improve recall over the best single score",
        "gains": gains,
        "budgets_with_gain": gain_budgets,
        "triggered": len(gain_budgets) < 2,
        "verdict": "FAIL" if len(gain_budgets) < 2 else "PASS",
    }

    # F3: state conditioning suppresses at least two events at the primary budget.
    a0 = _recall(results, "P1_A_none", primary_budget)
    a2 = _recall(results, "P1_A_state", primary_budget)
    a3 = _recall(results, "P1_A_hybrid", primary_budget)
    loss_state = a0 - a2
    loss_hybrid = a0 - a3
    f3 = {
        "id": "F3",
        "criterion": "State conditioning normalizes the fault",
        "budget": primary_budget,
        "recall_none": a0,
        "recall_state": a2,
        "recall_hybrid": a3,
        "loss_state": loss_state,
        "loss_hybrid": loss_hybrid,
        "triggered": bool(loss_state >= 2 or loss_hybrid >= 2),
        "verdict": "FAIL" if (loss_state >= 2 or loss_hybrid >= 2) else "PASS",
        "by_budget": {
            str(b): {
                "none": _recall(results, "P1_A_none", b),
                "state": _recall(results, "P1_A_state", b),
                "hybrid": _recall(results, "P1_A_hybrid", b),
            }
            for b in budgets
        },
    }

    # F4: tail budget control at B <= 2.
    tail = results[results["fold"].astype(str) == "T"]
    f4_rows = {}
    for b in budgets:
        if b > 2:
            continue
        sub = tail[np.isclose(tail["budget"], b) & (tail["method"] == PRIMARY_OR)]
        if sub.empty:
            continue
        realized = float(sub.iloc[0]["realized_budget"])
        error = float(sub.iloc[0]["budget_error"])
        f4_rows[str(b)] = {
            "realized_per_day": realized,
            "relative_error": error,
            "unreachable": not bool(sub.iloc[0]["cal_budget_reachable"]),
            "failed": abs(error) > 0.5 or not bool(sub.iloc[0]["cal_budget_reachable"]),
        }
    f4 = {
        "id": "F4",
        "criterion": "Budget is not controllable on the 31-day tail",
        "rows": f4_rows,
        "triggered": any(v["failed"] for v in f4_rows.values()),
        "verdict": "FAIL" if any(v["failed"] for v in f4_rows.values()) else "PASS",
    }

    # F5: >25% of emitted primary-OR alarms use n<200 after back-off.
    primary_alarms = alarms[
        (alarms["method"] == PRIMARY_OR)
        & np.isclose(alarms["budget"], primary_budget)
    ]
    n_alarms = int(len(primary_alarms))
    low = int((primary_alarms["cal_n"] < 200).sum()) if n_alarms else 0
    share = low / n_alarms if n_alarms else 0.0
    f5 = {
        "id": "F5",
        "criterion": "Calibration cells unreliable after back-off",
        "budget": primary_budget,
        "alarms": n_alarms,
        "alarms_n_lt_200": low,
        "share": share,
        "threshold": 0.25,
        "triggered": share > 0.25,
        "verdict": "FAIL" if share > 0.25 else "PASS",
    }

    # F6: mask coverage inside any event window > 10%.
    max_share = float(mask_table["masked_share"].max()) if len(mask_table) else 0.0
    f6 = {
        "id": "F6",
        "criterion": "Masks destroy the labels",
        "max_masked_share": max_share,
        "threshold": 0.10,
        "triggered": max_share > 0.10,
        "verdict": "FAIL" if max_share > 0.10 else "PASS",
        "table": mask_table.to_dict(orient="records"),
    }

    # F7: statistical redundancy AND F1.
    rho_values = relationship["spearman_rho"].dropna().tolist() if len(relationship) else []
    median_rho = float(np.median(rho_values)) if rho_values else float("nan")
    f7 = {
        "id": "F7",
        "criterion": "Scores are redundant (Spearman > 0.8) and F1 holds",
        "spearman_by_fold": rho_values,
        "median_spearman": median_rho,
        "f1_holds": f1["triggered"],
        "triggered": bool(f1["triggered"] and np.isfinite(median_rho) and median_rho > 0.8),
        "verdict": "FAIL"
        if (f1["triggered"] and np.isfinite(median_rho) and median_rho > 0.8)
        else "PASS",
    }

    # F8: union loses to naive context-IForest at every budget.
    f8_rows = {}
    for b in budgets:
        r_or = _recall(results, PRIMARY_OR, b)
        r_b4 = _recall(results, "B4_iforest_context", b)
        f8_rows[str(b)] = {"OR": r_or, "B4": r_b4, "OR_lower": r_or < r_b4}
    or_error = _budget_error_mean(results, PRIMARY_OR)
    b4_error = _budget_error_mean(results, "B4_iforest_context")
    all_lower = all(v["OR_lower"] for v in f8_rows.values())
    f8_triggered = bool(all_lower and np.isfinite(or_error) and np.isfinite(b4_error) and or_error < b4_error)
    f8 = {
        "id": "F8",
        "criterion": "Union loses to naive context-IForest at every budget with better budget error",
        "rows": f8_rows,
        "mean_abs_budget_error_OR": or_error,
        "mean_abs_budget_error_B4": b4_error,
        "triggered": f8_triggered,
        "verdict": "FAIL" if f8_triggered else "PASS",
    }

    # F9: calibration hygiene changes recall by >= 2 events.
    diffs = {}
    for b in budgets:
        primary = _recall(results, PRIMARY_OR, b)
        hygienic = _recall(results, "Mhyg_P3_or", b)
        if hygienic == 0 and not (
            results["method"].eq("Mhyg_P3_or") & np.isclose(results["budget"], b)
        ).any():
            diffs[str(b)] = {"primary": primary, "hygienic": None, "difference": None}
        else:
            diffs[str(b)] = {
                "primary": primary,
                "hygienic": hygienic,
                "difference": hygienic - primary,
            }
    finite_diffs = [
        abs(v["difference"]) for v in diffs.values() if v["difference"] is not None
    ]
    f9_triggered = bool(finite_diffs and max(finite_diffs) >= 2)
    f9 = {
        "id": "F9",
        "criterion": "Results unstable under calibration hygiene",
        "by_budget": diffs,
        "triggered": f9_triggered,
        "verdict": "FAIL" if f9_triggered else "PASS",
    }

    return {
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "F4": f4,
        "F5": f5,
        "F6": f6,
        "F7": f7,
        "F8": f8,
        "F9": f9,
        "notes": {
            "positive_recall_means": "detected / 4 labelled events; no significance claim",
            "verdict_semantics": "PASS means the falsification criterion did not hold; FAIL means it did",
        },
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_experiments(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    config: ExperimentConfig | None = None,
    dataset: Dataset | None = None,
    fold_ids: Sequence[str] | None = None,
    with_ablations: bool = True,
    with_baselines: bool = True,
    save_predictions: bool = True,
    verbose: bool = True,
) -> dict:
    cfg = config or ExperimentConfig()
    out = Path(output_dir)
    for sub in ("metrics", "figures", "predictions", "alarms"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    if dataset is None:
        if verbose:
            print("Loading MetroPT-3 and building the Day 2 data layer...", flush=True)
        dataset = build_dataset(load_native())
    folds = dataset.folds(include_tail=cfg.include_tail)
    if fold_ids is not None:
        folds = [f for f in folds if f.fold_id in set(fold_ids)]
    event_folds = [f for f in folds if f.event_id is not None]
    tail_folds = [f for f in folds if f.event_id is None]

    if verbose:
        print("Precomputing causal Score A features and rolling-z components...", flush=True)
    features = build_score_a_features(dataset.grid.frame, cfg.score_a)
    rolling_components = (
        build_rolling_z_components(dataset.grid.frame, config=cfg.baseline)
        if with_baselines
        else None
    )
    run_table = build_run_table(dataset.state)
    runs = run_table.frame

    accumulator = EvaluationAccumulator()
    alarm_store: list[pd.DataFrame] = []
    context_rows: list[dict] = []
    relationship_rows: list[dict] = []
    bundles: dict[str, Bundle] = {}
    mask_override = np.zeros(len(dataset.index), dtype=bool)
    long_intervals = _long_loaded_intervals(dataset)

    for fold in event_folds:
        if verbose:
            print(f"Fold {fold.fold_id}: building primary bundle", flush=True)
        split = _make_split(dataset, fold)
        bundle = build_bundle(
            dataset,
            fold,
            split,
            config=cfg,
            features=features,
            rolling_components=rolling_components,
            runs=run_table,
            with_baselines=with_baselines,
            verbose=verbose,
        )
        bundles[fold.fold_id] = bundle
        evaluate_bundle(
            dataset,
            fold,
            bundle,
            config=cfg,
            accumulator=accumulator,
            alarm_store=alarm_store,
            n_events=4,
            long_intervals=long_intervals,
        )
        context_rows.extend(_context_diagnostics(dataset, fold, bundle))
        relationship_rows.extend(_score_relationship(dataset, fold, bundle))
        if save_predictions:
            _save_predictions(dataset, fold, bundle, out_dir=out)

        if with_ablations:
            # M-none: ignore data-quality masks in split materialisation only (C7).
            if verbose:
                print(f"Fold {fold.fold_id}: M-none ablation", flush=True)
            split_none = _make_split(dataset, fold, mask_override=mask_override)
            bundle_none = build_bundle(
                dataset,
                fold,
                split_none,
                config=cfg,
                features=features,
                rolling_components=rolling_components,
                runs=run_table,
                arms=("hybrid",),
                variants=("B1",),
                with_baselines=False,
            )
            evaluate_bundle(
                dataset,
                fold,
                bundle_none,
                config=cfg,
                accumulator=accumulator,
                methods=(PRIMARY_OR,),
                n_events=4,
                tag_prefix="Mnone_",
                alarm_store=alarm_store,
                solve=False,
            )

            # M-hygienic: drop >2 h loaded runs from the calibration window (C7).
            if verbose:
                print(f"Fold {fold.fold_id}: M-hygienic ablation", flush=True)
            split_hyg = _hygienic_split(dataset, fold, split, run_table.frame)
            bundle_hyg = build_bundle(
                dataset,
                fold,
                split_hyg,
                config=cfg,
                features=features,
                rolling_components=rolling_components,
                runs=run_table,
                arms=("hybrid",),
                variants=("B1",),
                with_baselines=False,
            )
            evaluate_bundle(
                dataset,
                fold,
                bundle_hyg,
                config=cfg,
                accumulator=accumulator,
                methods=(PRIMARY_OR,),
                n_events=4,
                tag_prefix="Mhyg_",
                alarm_store=alarm_store,
                solve=False,
            )

    # Frozen fold-4 model on the 31-day tail: reuse every fold-4 p-value and
    # threshold. Only the test rows change.
    for tail in tail_folds:
        base = bundles.get(str(tail.inherits_from))
        if base is None:
            if verbose:
                print("Tail skipped: inherited fold was not run", flush=True)
            continue
        if verbose:
            print(f"Fold {tail.fold_id}: frozen fold-{tail.inherits_from} model on tail", flush=True)
        split_tail = _make_split(dataset, tail)
        tail_bundle = replace(base, split=split_tail)
        evaluate_bundle(
            dataset,
            tail,
            tail_bundle,
            config=cfg,
            accumulator=accumulator,
            methods=tuple(base.layers),
            n_events=None,
            alarm_store=alarm_store,
        )

    results, events_frame, cells_frame = accumulator.frames()
    alarms_frame = (
        pd.concat(alarm_store, ignore_index=True)
        if alarm_store
        else pd.DataFrame(
            columns=[
                "fold",
                "method",
                "budget",
                "timestamp",
                "position",
                "p",
                "alpha",
                "p_a",
                "p_b",
                "fired_by",
                "cell",
                "cal_level",
                "cal_n",
                "cal_depth",
                "channel",
                "state",
                "run_length_min",
                "context_effect",
                "value_effect",
                "event_related",
            ]
        )
    )

    from anomaliradar.data.labels import event_overlap_table

    mask_table = event_overlap_table(
        dataset.index, dataset.masks.combined, dataset.events
    )
    relationship = pd.DataFrame(relationship_rows)
    context_frame = pd.DataFrame(context_rows)
    cal_size_rows = []
    for fold_id, bundle in bundles.items():
        for key, table in bundle.calibration_tables.items():
            table = table.copy()
            table.insert(0, "fold", fold_id)
            table.insert(1, "score", key)
            cal_size_rows.append(table)
    cal_sizes = (
        pd.concat(cal_size_rows, ignore_index=True)
        if cal_size_rows
        else pd.DataFrame(columns=["fold", "score", "level", "key", "n"])
    )

    results.to_csv(out / "metrics" / "fold_results.csv", index=False)
    events_frame.to_csv(out / "metrics" / "event_outcomes.csv", index=False)
    cells_frame.to_csv(out / "metrics" / "cell_calibration.csv", index=False)
    alarms_frame.to_csv(out / "alarms" / "alarms.csv", index=False)
    mask_table.to_csv(out / "metrics" / "mask_coverage.csv", index=False)
    context_frame.to_csv(out / "metrics" / "context_diagnostics.csv", index=False)
    relationship.to_csv(out / "metrics" / "score_relationship.csv", index=False)
    cal_sizes.to_csv(out / "metrics" / "calibration_cell_sizes.csv", index=False)

    # Focused tables.
    primaries = [
        PRIMARY_A,
        PRIMARY_B,
        PRIMARY_OR,
        "P3_and",
        "P3_weighted",
        "P3_or_bonferroni",
        *BASELINE_KEYS,
    ]
    results[results["method"].isin(primaries)].to_csv(
        out / "metrics" / "primary_results.csv", index=False
    )
    results[results["method"].str.startswith("P1_A_")].to_csv(
        out / "metrics" / "context_arms.csv", index=False
    )
    results[results["method"].isin(("P2_B_B1", "P2_B_B3"))].to_csv(
        out / "metrics" / "score_b_variants.csv", index=False
    )
    results[results["method"].str.match(r"P3_weighted_l")].to_csv(
        out / "metrics" / "weighted_lambda.csv", index=False
    )
    results[results["method"].str.startswith("M")].to_csv(
        out / "metrics" / "mask_ablations.csv", index=False
    )
    tail_frame = results[results["fold"].astype(str) == "T"]
    tail_frame.to_csv(out / "metrics" / "tail_results.csv", index=False)

    falsification = compute_falsification(
        results=results,
        alarms=alarms_frame,
        mask_table=mask_table,
        relationship=relationship,
    )
    with open(out / "metrics" / "falsification.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(falsification), fh, indent=2, sort_keys=True)

    headline = _headline(results, falsification, cfg)
    with open(out / "metrics" / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(headline), fh, indent=2, sort_keys=True)

    manifest = {
        "config": _jsonable(cfg.to_dict()),
        "config_hash": cfg.hash(),
        "dataset_fingerprint": dataset.fingerprint(),
        "dataset_summary": _jsonable(dataset.summary()),
        "fold_table": _jsonable(
            [fold.to_dict() for fold in dataset.folds(include_tail=cfg.include_tail)]
        ),
        "split_table": _jsonable({f.fold_id: dataset.split(f).to_dict() for f in folds}),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "evaluated_methods": sorted(results["method"].unique().tolist())
        if len(results)
        else [],
        "budgets": list(cfg.alarm.budgets),
        "debounce_minutes": cfg.alarm.debounce_minutes,
        "tail_inherits_from": "4",
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(manifest), fh, indent=2, sort_keys=True)

    if verbose:
        print(f"Wrote artefacts to {out}", flush=True)
    return {
        "results": results,
        "events": events_frame,
        "cells": cells_frame,
        "alarms": alarms_frame,
        "falsification": falsification,
        "summary": headline,
        "manifest": manifest,
    }


def _headline(
    results: pd.DataFrame, falsification: dict, cfg: ExperimentConfig
) -> dict:
    rows = {}
    for method in (
        PRIMARY_A,
        PRIMARY_B,
        PRIMARY_OR,
        "P3_and",
        "P3_weighted",
        *BASELINE_KEYS,
    ):
        sub = results[
            (results["method"] == method) & np.isclose(results["budget"], 1.0)
        ]
        if sub.empty:
            continue
        event_rows = sub[sub["fold"].astype(str).isin(("1", "2", "3", "4"))]
        tail_rows = sub[sub["fold"].astype(str) == "T"]
        recall = 0
        for _, row in event_rows.iterrows():
            fold = str(row["fold"])
            recall += int(bool(row.get(f"event_{fold}", False)))
        rows[method] = {
            "budget": 1.0,
            "event_recall": f"{recall}/4",
            "event_recall_count": recall,
            "mean_alarms_per_day_event_folds": float(
                event_rows["alarms_per_day"].mean()
            )
            if len(event_rows)
            else None,
            "mean_false_alarms_per_day_event_folds": float(
                event_rows["false_alarms_per_day"].mean()
            )
            if len(event_rows)
            else None,
            "mean_abs_budget_error_event_folds": float(
                event_rows["budget_error"].abs().mean()
            )
            if len(event_rows)
            else None,
            "tail_realized_per_day": float(tail_rows["realized_budget"].iloc[0])
            if len(tail_rows)
            else None,
            "tail_relative_budget_error": float(tail_rows["budget_error"].iloc[0])
            if len(tail_rows)
            else None,
        }
    verdicts = {
        key: value["verdict"] for key, value in falsification.items() if key.startswith("F")
    }
    n_fail = sum(1 for v in verdicts.values() if v == "FAIL")
    return {
        "config_hash": cfg.hash(),
        "headline_budget": 1.0,
        "methods": rows,
        "falsification_verdicts": verdicts,
        "falsified_criteria": [k for k, v in verdicts.items() if v == "FAIL"],
        "n_falsified": n_fail,
        "recommended_configuration": _recommend(results, falsification),
    }


def _recommend(results: pd.DataFrame, falsification: dict) -> dict:
    """Apply the pre-registered fallbacks mechanically; no post-hoc tuning."""
    if falsification["F3"]["triggered"]:
        score_a = "P1_A_none"
        reason_a = "F3: state conditioning suppressed at least two events"
    else:
        score_a = PRIMARY_A
        reason_a = "hybrid context arm retained"
    if falsification["F1"]["triggered"] or falsification["F2"]["triggered"]:
        # Single score: choose whichever has higher recall at B=1, tie -> A.
        a = _recall(results, PRIMARY_A, 1.0)
        b = _recall(results, PRIMARY_B, 1.0)
        if b > a:
            fusion = PRIMARY_B
            reason_fusion = "F1/F2: Score B alone detected more events"
        else:
            fusion = PRIMARY_A
            reason_fusion = "F1/F2: Score A alone detected as many or more events"
    else:
        fusion = PRIMARY_OR
        reason_fusion = "union retained"
    return {
        "score_a": score_a,
        "score_a_reason": reason_a,
        "score_b": PRIMARY_B,
        "fusion": fusion,
        "fusion_reason": reason_fusion,
        "alarm_budget_per_day": 1.0,
        "debounce_minutes": 30,
    }
