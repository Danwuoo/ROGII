"""Serializable contracts for the prefix-calibrated inverse probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


SCHEMA_VERSION = "prefix_calibrated_inverse_probe_v1"
CONTROL_CORRECT = "correct"
CONTROL_WRONG_TYPEWELL = "wrong_typewell"
CONTROL_PHASE_DESTROY = "phase_destroy"
CONTROL_GR_ORDER_SHUFFLE = "gr_order_shuffle"
CONTROL_DYNAMICS_ONLY = "dynamics_only"
CONTROL_IDS = (
    CONTROL_CORRECT,
    CONTROL_WRONG_TYPEWELL,
    CONTROL_PHASE_DESTROY,
    CONTROL_GR_ORDER_SHUFFLE,
    CONTROL_DYNAMICS_ONLY,
)


@dataclass(frozen=True)
class PrefixCalibrationConfig:
    """Locked physical-unit settings, fixed before Stage 0 is observed."""

    resample_step_ft: float = 1.0
    maximum_resampled_points: int = 2048
    minimum_prefix_pairs: int = 48
    minimum_evaluation_pairs: int = 20
    lag_values_ft: tuple[float, ...] = (
        -8.0,
        -6.0,
        -4.0,
        -2.0,
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
    )
    stretch_values: tuple[float, ...] = (0.98, 1.0, 1.02)
    smoothing_scales_ft: tuple[float, ...] = (5.0, 15.0)
    calibration_block_ft: float = 20.0
    student_t_df: float = 4.0
    raw_weight: float = 0.30
    smooth_weight: float = 0.30
    derivative_weight: float = 0.18
    contrast_weight: float = 0.14
    polarity_weight: float = 0.08
    amplitude_scale_min: float = 0.25
    amplitude_scale_max: float = 4.0
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20260729
    stage0_correct_win_fraction: float = 0.65

    def __post_init__(self) -> None:
        positive = (
            self.resample_step_ft,
            self.calibration_block_ft,
            self.student_t_df,
            self.amplitude_scale_min,
            self.amplitude_scale_max,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("prefix calibration physical settings must be positive")
        if (
            self.maximum_resampled_points < 64
            or self.minimum_prefix_pairs < 16
            or self.minimum_evaluation_pairs < 8
        ):
            raise ValueError("prefix calibration sample limits are too small")
        lag = tuple(float(value) for value in self.lag_values_ft)
        stretch = tuple(float(value) for value in self.stretch_values)
        scales = tuple(float(value) for value in self.smoothing_scales_ft)
        if (
            not lag
            or not stretch
            or not scales
            or any(not math.isfinite(value) for value in (*lag, *stretch, *scales))
            or any(value <= 0.0 for value in (*stretch, *scales))
        ):
            raise ValueError("lag, stretch, and scale grids must be finite")
        weights = (
            self.raw_weight,
            self.smooth_weight,
            self.derivative_weight,
            self.contrast_weight,
            self.polarity_weight,
        )
        if any(value < 0.0 or not math.isfinite(value) for value in weights):
            raise ValueError("likelihood weights must be finite and nonnegative")
        if abs(sum(weights) - 1.0) > 1.0e-12:
            raise ValueError("likelihood weights must sum to one")
        if not 0.5 < self.stage0_correct_win_fraction <= 1.0:
            raise ValueError("Stage 0 win gate must lie in (0.5, 1]")
        object.__setattr__(self, "lag_values_ft", lag)
        object.__setattr__(self, "stretch_values", stretch)
        object.__setattr__(self, "smoothing_scales_ft", scales)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrefixCalibrationConfig":
        values = dict(payload)
        for field in ("lag_values_ft", "stretch_values", "smoothing_scales_ft"):
            if field in values:
                values[field] = tuple(values[field])
        return cls(**values)


@dataclass(frozen=True)
class PrefixLikelihoodResult:
    """One target-free calibrated prefix likelihood result."""

    well_id: str
    reference_well_id: str
    control_id: str
    cost: float
    coherence: float
    reliability: float
    effective_rows: int
    prefix_rows: int
    prefix_length_ft: float
    gr_missing_fraction: float
    typewell_coverage_fraction: float
    lag_ft: float
    stretch: float
    amplitude_offset: float
    amplitude_scale: float
    raw_cost: float
    smooth_cost: float
    derivative_cost: float
    contrast_cost: float
    polarity_cost: float
    calibration_rows: int
    evaluation_rows: int
    boundary_fraction: float

    def __post_init__(self) -> None:
        if self.control_id not in CONTROL_IDS:
            raise ValueError(f"unknown prefix control: {self.control_id!r}")
        numeric = (
            self.cost,
            self.coherence,
            self.reliability,
            self.prefix_length_ft,
            self.gr_missing_fraction,
            self.typewell_coverage_fraction,
            self.lag_ft,
            self.stretch,
            self.amplitude_offset,
            self.amplitude_scale,
            self.raw_cost,
            self.smooth_cost,
            self.derivative_cost,
            self.contrast_cost,
            self.polarity_cost,
            self.boundary_fraction,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("prefix likelihood result must be finite")
        if (
            self.effective_rows < 1
            or self.prefix_rows < 1
            or self.calibration_rows < 1
            or self.evaluation_rows < 1
        ):
            raise ValueError("prefix likelihood row counts must be positive")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("prefix reliability must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrefixLikelihoodResult":
        return cls(**dict(payload))
