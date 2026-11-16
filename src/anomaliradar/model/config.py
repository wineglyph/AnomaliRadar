"""Configuration objects for the Day 3 model, calibration and alarm layers.

Every parameter that can affect a result lives here so that a run can be hashed
and reproduced. The defaults are the pre-registered values from
``docs/experimental_protocol.md`` and ``docs/methodology.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

#: The five approved Score A value channels. TP2 is excluded by R4.
VALUE_CHANNELS: tuple[str, ...] = (
    "Reservoirs",
    "TP3",
    "H1",
    "DV_pressure",
    "Oil_temperature",
)

#: Context arms from protocol §5.1.
CONTEXT_ARMS: tuple[str, ...] = ("none", "calendar", "state", "hybrid")

#: Primary and alternative Score B variants.
SCORE_B_VARIANTS: tuple[str, ...] = ("B1", "B3")

#: Fusion rules from methodology §8.1.
FUSION_RULES: tuple[str, ...] = ("A", "B", "or", "and", "weighted")

#: Budget grid from protocol §6.6.
BUDGETS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True, slots=True)
class ScoreAConfig:
    """Per-channel ridge model with causal features and a causal residual scale."""

    channels: tuple[str, ...] = VALUE_CHANNELS
    lags: tuple[int, ...] = (1, 5)
    rolling_windows: tuple[int, ...] = (15, 60, 360)
    dispersion_window: int = 60
    ridge_alpha: float = 1.0
    scale_halflife_min: float = 60.0
    scale_floor_quantile: float = 0.5
    log1p_scores: bool = True


@dataclass(frozen=True, slots=True)
class ScoreBConfig:
    """Empirical survival of completed run lengths."""

    variant: str = "B1"                 # B1 primary; B3 discrete-time hazard
    min_reference_runs: int = 50
    density_target: int = 200
    hazard_smoothing: float = 1.0       # Laplace smoothing for B3


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Mondrian conformal calibration with hierarchical back-off.

    ``reference_fraction`` is the statistical audit requested by the final
    brief: the earliest part of the calibration window builds the conformal
    reference distribution, and the latest part is used only to select the
    alarm threshold ``alpha``. This prevents the same observations from being
    used twice and removes optimistic in-sample budget estimates.
    """

    cells: str = "state_daytype"        # state_daytype | state | global
    n_min: int = 200
    n_refuse: int = 50
    reference_fraction: float = 2.0 / 3.0
    min_reference_rows: int = 50

    def __post_init__(self) -> None:
        if not 0.0 < self.reference_fraction < 1.0:
            raise ValueError("reference_fraction must lie in (0, 1)")
        if self.n_refuse > self.n_min:
            raise ValueError("n_refuse must not exceed n_min")


@dataclass(frozen=True, slots=True)
class AlarmConfig:
    debounce_minutes: int = 30
    budgets: tuple[float, ...] = BUDGETS
    solver_grid: int = 40
    solver_extra_grid: int = 20
    max_alarm_p: float = 1.0

    def __post_init__(self) -> None:
        if self.debounce_minutes < 0:
            raise ValueError("debounce_minutes must be non-negative")


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    rolling_window: int = 360
    iforest_estimators: int = 200
    random_state: int = 20201116
    mad_floor_quantile: float = 0.5


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    score_a: ScoreAConfig = field(default_factory=ScoreAConfig)
    score_b: ScoreBConfig = field(default_factory=ScoreBConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    primary_context_arm: str = "hybrid"
    primary_score_b: str = "B1"
    include_tail: bool = True
    seed: int = 20201116

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_a": asdict(self.score_a),
            "score_b": asdict(self.score_b),
            "calibration": asdict(self.calibration),
            "alarm": asdict(self.alarm),
            "baseline": asdict(self.baseline),
            "primary_context_arm": self.primary_context_arm,
            "primary_score_b": self.primary_score_b,
            "include_tail": self.include_tail,
            "seed": self.seed,
        }

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def config_json(config: ExperimentConfig) -> str:
    return json.dumps(config.to_dict(), indent=2, sort_keys=True, default=str)


def timestamp_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()
