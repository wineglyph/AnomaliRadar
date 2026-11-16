"""Day 3 model, calibration, alarm and experiment package."""

from anomaliradar.model.alarm import (
    AlarmSet,
    BudgetSolution,
    debounce_positions,
    emit_alarms,
    solve_budget,
)
from anomaliradar.model.calibration import (
    CalibrationResult,
    CalibrationSplit,
    ConformalCalibrator,
    chronological_split,
)
from anomaliradar.model.config import (
    BUDGETS,
    CONTEXT_ARMS,
    VALUE_CHANNELS,
    ExperimentConfig,
)
from anomaliradar.model.fusion import FUSION_LAMBDAS, FusionResult, fuse
from anomaliradar.model.score_a import (
    ScoreAModel,
    build_score_a_features,
    fit_score_a,
    score_a_frame,
)
from anomaliradar.model.score_b import (
    ScoreBReference,
    build_run_table,
    fit_score_b_reference,
    score_b_series,
)

__all__ = [
    "AlarmSet",
    "BUDGETS",
    "BudgetSolution",
    "CONTEXT_ARMS",
    "CalibrationResult",
    "CalibrationSplit",
    "ConformalCalibrator",
    "ExperimentConfig",
    "FUSION_LAMBDAS",
    "FusionResult",
    "ScoreAModel",
    "ScoreBReference",
    "VALUE_CHANNELS",
    "build_run_table",
    "build_score_a_features",
    "chronological_split",
    "debounce_positions",
    "emit_alarms",
    "fit_score_a",
    "fit_score_b_reference",
    "fuse",
    "score_a_frame",
    "score_b_series",
    "solve_budget",
]
