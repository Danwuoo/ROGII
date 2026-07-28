"""Fold-local smooth residual targets and row/grid reconstruction.

Labels enter only through :func:`make_grid_residual_target`, which is called
for outer-training wells.  Prediction reconstruction consumes only a model
output, its inference-safe sequence payload, and the incumbent prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.fft import dct, idct

from .components import (
    ResidualComponents,
    compose_residual_components,
    decompose_residual_target,
    orthogonalize_shape,
)
from .contracts import ResidualTrainingWell, SequenceModelInput


@dataclass(frozen=True)
class SmoothTargetConfig:
    """Fixed low-frequency policy shared by every family."""

    retained_dct_modes: int = 16
    maximum_absolute_correction: float = 80.0

    def __post_init__(self) -> None:
        modes = int(self.retained_dct_modes)
        maximum = float(self.maximum_absolute_correction)
        if modes < 1:
            raise ValueError("retained_dct_modes must be positive")
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_absolute_correction must be finite and positive")
        object.__setattr__(self, "retained_dct_modes", modes)
        object.__setattr__(self, "maximum_absolute_correction", maximum)


DEFAULT_SMOOTH_TARGET_CONFIG = SmoothTargetConfig()


def _finite_vector(values: Any, *, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 1
        or result.size == 0
        or not np.isfinite(result).all()
        or (length is not None and len(result) != int(length))
    ):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def low_pass_shape(
    shape: Any,
    coordinate: Any,
    *,
    retained_modes: int,
) -> np.ndarray:
    """DCT low-pass followed by exact common/slope removal."""

    values = _finite_vector(shape, name="shape")
    x = _finite_vector(coordinate, name="coordinate", length=len(values))
    modes = min(int(retained_modes), len(values))
    if modes < 1:
        raise ValueError("retained_modes must be positive")
    coefficients = dct(values, type=2, norm="ortho")
    coefficients[modes:] = 0.0
    smoothed = idct(coefficients, type=2, norm="ortho")
    result = orthogonalize_shape(smoothed, x)
    if result.shape != values.shape or not np.isfinite(result).all():
        raise ValueError("low-pass shape is invalid")
    return result


def _smooth_components(
    residual: np.ndarray,
    coordinate: np.ndarray,
    *,
    incumbent: np.ndarray,
    config: SmoothTargetConfig,
) -> ResidualComponents:
    exact = decompose_residual_target(
        residual, coordinate, incumbent_prediction=incumbent
    )
    shape = low_pass_shape(
        exact.shape,
        coordinate,
        retained_modes=config.retained_dct_modes,
    )
    result = compose_residual_components(
        incumbent,
        coordinate,
        common_coefficient=float(exact.common[0]),
        slope_coefficient=_slope_coefficient(exact.slope, coordinate),
        raw_shape=shape,
    )
    if float(np.max(np.abs(result.correction))) > config.maximum_absolute_correction:
        # Target clipping is component-preserving: clip the total field, then
        # reproject it into the identifiable decomposition.
        clipped = np.clip(
            result.correction,
            -config.maximum_absolute_correction,
            config.maximum_absolute_correction,
        )
        exact_clipped = decompose_residual_target(
            clipped, coordinate, incumbent_prediction=incumbent
        )
        result = compose_residual_components(
            incumbent,
            coordinate,
            common_coefficient=float(exact_clipped.common[0]),
            slope_coefficient=_slope_coefficient(exact_clipped.slope, coordinate),
            raw_shape=low_pass_shape(
                exact_clipped.shape,
                coordinate,
                retained_modes=config.retained_dct_modes,
            ),
        )
    return result


def _slope_coefficient(slope_field: np.ndarray, coordinate: np.ndarray) -> float:
    basis = coordinate - float(np.mean(coordinate))
    denominator = float(basis @ basis)
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.asarray(slope_field, dtype=np.float64) @ basis / denominator)


def make_grid_residual_target(
    training: ResidualTrainingWell,
    *,
    config: SmoothTargetConfig = DEFAULT_SMOOTH_TARGET_CONFIG,
) -> ResidualComponents:
    """Create a smooth fixed-grid target from one legal training residual."""

    if not isinstance(training, ResidualTrainingWell):
        raise TypeError("training must be a ResidualTrainingWell")
    if not isinstance(config, SmoothTargetConfig):
        raise TypeError("config must be a SmoothTargetConfig")
    inference = training.inference
    row_coordinate = np.asarray(inference.row_coordinate, dtype=np.float64)
    grid = np.asarray(inference.grid_coordinate, dtype=np.float64)
    residual = np.asarray(training.residual_target, dtype=np.float64)
    grid_residual = np.interp(grid, row_coordinate, residual)
    zero = np.zeros(len(grid), dtype=np.float64)
    return _smooth_components(
        grid_residual,
        grid,
        incumbent=zero,
        config=config,
    )


def grid_prediction_to_row_components(
    grid_prediction: Any,
    inference: SequenceModelInput,
    *,
    config: SmoothTargetConfig = DEFAULT_SMOOTH_TARGET_CONFIG,
) -> ResidualComponents:
    """Project a family grid output and reconstruct finite original-row fields."""

    if not isinstance(inference, SequenceModelInput):
        raise TypeError("inference must be a SequenceModelInput")
    if not isinstance(config, SmoothTargetConfig):
        raise TypeError("config must be a SmoothTargetConfig")
    grid = np.asarray(inference.grid_coordinate, dtype=np.float64)
    predicted = _finite_vector(
        grid_prediction, name="grid_prediction", length=len(grid)
    )
    predicted = np.clip(
        predicted,
        -config.maximum_absolute_correction,
        config.maximum_absolute_correction,
    )
    grid_components = _smooth_components(
        predicted,
        grid,
        incumbent=np.zeros(len(grid), dtype=np.float64),
        config=config,
    )
    row_coordinate = np.asarray(inference.row_coordinate, dtype=np.float64)
    raw_row_shape = np.interp(row_coordinate, grid, grid_components.shape)
    return compose_residual_components(
        inference.incumbent_prediction,
        row_coordinate,
        common_coefficient=float(grid_components.common[0]),
        slope_coefficient=_slope_coefficient(grid_components.slope, grid),
        raw_shape=raw_row_shape,
    )


def target_manifest(
    config: SmoothTargetConfig = DEFAULT_SMOOTH_TARGET_CONFIG,
) -> dict[str, object]:
    if not isinstance(config, SmoothTargetConfig):
        raise TypeError("config must be a SmoothTargetConfig")
    return {
        "target": (
            "outer-training truth minus fold-safe heel-registered incumbent; "
            "interpolated to normalized-MD grid"
        ),
        "decomposition": "constant + zero-mean linear + zero-mean slope-orthogonal shape",
        "smoothing": "orthonormal DCT-II low-pass on shape component",
        "retained_dct_modes": config.retained_dct_modes,
        "maximum_absolute_correction": config.maximum_absolute_correction,
        "validation_labels_used_at_prediction": False,
    }


__all__ = [
    "DEFAULT_SMOOTH_TARGET_CONFIG",
    "SmoothTargetConfig",
    "grid_prediction_to_row_components",
    "low_pass_shape",
    "make_grid_residual_target",
    "target_manifest",
]
