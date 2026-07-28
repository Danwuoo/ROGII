"""Inference-safe heel-calibrated distributional GR registration.

The registration deliberately discards suffix order after deterministic GR
preprocessing.  It estimates one constant TVT correction to an existing base
prediction by comparing the suffix GR distribution with the typewell GR
distribution sampled along shifted base trajectories.

Only the competition inference views are selected:

* horizontal: ``MD, X, Y, Z, GR, TVT_input`` (only ``GR`` and
  ``TVT_input`` enter the calculation);
* typewell: ``TVT, GR``.

Targets, formation surfaces, geology annotations, images, and filesystem state
are never inputs to this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from .data import select_inference_view, validate_prefix


HEEL_REGISTRATION_VERSION = "heel_quantile_registration_v1"


@dataclass(frozen=True)
class HeelRegistrationConfig:
    """Fixed numerical policy for heel-calibrated quantile registration."""

    shift_min_ft: float = -40.0
    shift_max_ft: float = 40.0
    shift_step_ft: float = 0.5
    rolling_median_width: int = 7
    max_suffix_rows: int = 1_000
    quantile_count: int = 21
    squared_loss_cap: float = 2_500.0
    minimum_coverage: float = 0.95
    coverage_penalty_scale: float = 1_000.0
    posterior_temperature_multiplier: float = 2.0
    posterior_mse_floor: float = 1.0
    minimum_prefix_pairs: int = 2

    def __post_init__(self) -> None:
        numeric = (
            ("shift_min_ft", self.shift_min_ft),
            ("shift_max_ft", self.shift_max_ft),
            ("shift_step_ft", self.shift_step_ft),
            ("squared_loss_cap", self.squared_loss_cap),
            ("minimum_coverage", self.minimum_coverage),
            ("coverage_penalty_scale", self.coverage_penalty_scale),
            (
                "posterior_temperature_multiplier",
                self.posterior_temperature_multiplier,
            ),
            ("posterior_mse_floor", self.posterior_mse_floor),
        )
        for name, value in numeric:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite real number")
        if self.shift_min_ft >= self.shift_max_ft or self.shift_step_ft <= 0.0:
            raise ValueError("Shift range must be increasing with a positive step")
        steps = (self.shift_max_ft - self.shift_min_ft) / self.shift_step_ft
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("Shift range must contain an integer number of steps")
        if (
            isinstance(self.rolling_median_width, bool)
            or not isinstance(self.rolling_median_width, Integral)
            or self.rolling_median_width < 1
            or self.rolling_median_width % 2 == 0
        ):
            raise ValueError("rolling_median_width must be a positive odd integer")
        for name, value, minimum in (
            ("max_suffix_rows", self.max_suffix_rows, 1),
            ("quantile_count", self.quantile_count, 2),
            ("minimum_prefix_pairs", self.minimum_prefix_pairs, 2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.squared_loss_cap <= 0.0:
            raise ValueError("squared_loss_cap must be positive")
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be between zero and one")
        if self.coverage_penalty_scale < 0.0:
            raise ValueError("coverage_penalty_scale must be non-negative")
        if self.posterior_temperature_multiplier <= 0.0:
            raise ValueError("posterior_temperature_multiplier must be positive")
        if self.posterior_mse_floor <= 0.0:
            raise ValueError("posterior_mse_floor must be positive")

    @property
    def shift_grid(self) -> np.ndarray:
        """Return the inclusive deterministic shift grid."""
        count = int(round((self.shift_max_ft - self.shift_min_ft) / self.shift_step_ft))
        return self.shift_min_ft + self.shift_step_ft * np.arange(
            count + 1, dtype=np.float64
        )

    @property
    def quantile_grid(self) -> np.ndarray:
        """Return inclusive equally spaced quantile probabilities."""
        return np.linspace(0.0, 1.0, self.quantile_count, dtype=np.float64)


DEFAULT_HEEL_REGISTRATION_CONFIG = HeelRegistrationConfig()


@dataclass(frozen=True)
class HeelAffineCalibration:
    """OLS map from preprocessed horizontal GR into typewell-GR units."""

    intercept: float
    slope: float
    pair_count: int
    prefix_coverage: float
    prefix_mse: float


@dataclass(frozen=True)
class HeelRegistrationResult:
    """Posterior shift estimate and complete finite support diagnostics."""

    posterior_mean_shift: float
    map_shift: float
    posterior_entropy: float
    effective_support: float
    maximum_posterior: float
    edge_mass: float
    best_coverage: float
    posterior_mean_coverage: float
    calibration: HeelAffineCalibration
    sampled_suffix_row_index: np.ndarray
    shift_grid: np.ndarray
    candidate_cost: np.ndarray
    candidate_coverage: np.ndarray
    candidate_support_count: np.ndarray
    posterior_weight: np.ndarray


def _as_float_vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinity")
    return array


def _interpolate_and_median(values: np.ndarray, *, width: int, name: str) -> np.ndarray:
    """Linearly fill missing rows, then apply a centered rolling median."""
    vector = _as_float_vector(values, name=name)
    finite = np.isfinite(vector)
    if not finite.any():
        raise ValueError(f"{name} requires at least one finite observation")
    positions = np.arange(len(vector), dtype=np.float64)
    interpolated = np.interp(positions, positions[finite], vector[finite]).astype(
        np.float64
    )
    smoothed = (
        pd.Series(interpolated, copy=False)
        .rolling(window=width, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(smoothed).all():
        raise ValueError(f"{name} preprocessing produced non-finite values")
    return smoothed


def _validated_typewell(
    typewell: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if not isinstance(typewell, pd.DataFrame):
        raise TypeError("typewell must be a pandas DataFrame")
    frame = select_inference_view(typewell, kind="typewell")
    tvt = _as_float_vector(frame["TVT"], name="typewell TVT")
    gr = _as_float_vector(frame["GR"], name="typewell GR")
    if not np.isfinite(tvt).all() or len(tvt) < 2 or np.any(np.diff(tvt) <= 0.0):
        raise ValueError("Typewell TVT must be finite and strictly increasing")
    finite_gr = np.isfinite(gr)
    if int(finite_gr.sum()) < 2:
        raise ValueError("Typewell GR requires at least two finite observations")
    supported_tvt = tvt[finite_gr]
    supported_gr = gr[finite_gr]
    return (
        supported_tvt,
        supported_gr,
        float(supported_tvt[0]),
        float(supported_tvt[-1]),
    )


def _fit_prefix_affine(
    prefix_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    support_min: float,
    support_max: float,
    config: HeelRegistrationConfig,
) -> HeelAffineCalibration:
    supported = (
        np.isfinite(prefix_gr)
        & np.isfinite(prefix_tvt)
        & (prefix_tvt >= support_min)
        & (prefix_tvt <= support_max)
    )
    pair_count = int(supported.sum())
    if pair_count < config.minimum_prefix_pairs:
        raise ValueError(
            "Visible prefix has fewer supported GR/typewell pairs than "
            f"minimum_prefix_pairs={config.minimum_prefix_pairs}"
        )
    horizontal_values = prefix_gr[supported]
    typewell_values = np.interp(
        prefix_tvt[supported], typewell_tvt, typewell_gr
    ).astype(np.float64)

    horizontal_center = float(np.mean(horizontal_values))
    typewell_center = float(np.mean(typewell_values))
    centered_horizontal = horizontal_values - horizontal_center
    denominator = float(centered_horizontal @ centered_horizontal)
    scale = max(float(np.max(np.abs(horizontal_values))), 1.0)
    if denominator <= np.finfo(np.float64).eps * scale * scale * pair_count:
        slope = 0.0
    else:
        slope = float(
            centered_horizontal @ (typewell_values - typewell_center) / denominator
        )
    intercept = float(typewell_center - slope * horizontal_center)
    residual = intercept + slope * horizontal_values - typewell_values
    prefix_mse = float(np.mean(residual * residual))
    prefix_coverage = float(pair_count / len(prefix_tvt))
    values = (intercept, slope, prefix_coverage, prefix_mse)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Prefix OLS calibration produced non-finite values")
    return HeelAffineCalibration(
        intercept=intercept,
        slope=slope,
        pair_count=pair_count,
        prefix_coverage=prefix_coverage,
        prefix_mse=max(prefix_mse, 0.0),
    )


def _equally_spaced_indices(length: int, maximum: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Suffix must contain at least one row")
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    indices = np.rint(
        np.linspace(0.0, float(length - 1), maximum, dtype=np.float64)
    ).astype(np.int64)
    indices = np.unique(indices)
    if len(indices) != maximum:
        raise RuntimeError("Equally spaced suffix sampling produced duplicate rows")
    return indices


def clipped_quantile_loss(
    left: Any,
    right: Any,
    *,
    quantile_count: int = 21,
    squared_loss_cap: float = 2_500.0,
) -> float:
    """Return the order-invariant clipped squared 1-D quantile loss.

    This small public primitive makes the distributional contract explicit and
    permits direct negative-control tests without involving GR preprocessing.
    """
    left_values = _as_float_vector(left, name="left distribution")
    right_values = _as_float_vector(right, name="right distribution")
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("Quantile-loss distributions must be finite")
    if (
        isinstance(quantile_count, bool)
        or not isinstance(quantile_count, Integral)
        or quantile_count < 2
    ):
        raise ValueError("quantile_count must be an integer >= 2")
    if (
        isinstance(squared_loss_cap, bool)
        or not isinstance(squared_loss_cap, Real)
        or not math.isfinite(float(squared_loss_cap))
        or squared_loss_cap <= 0.0
    ):
        raise ValueError("squared_loss_cap must be finite and positive")
    probabilities = np.linspace(0.0, 1.0, quantile_count, dtype=np.float64)
    left_quantiles = np.quantile(left_values, probabilities)
    right_quantiles = np.quantile(right_values, probabilities)
    squared = (left_quantiles - right_quantiles) ** 2
    loss = float(np.mean(np.minimum(squared, squared_loss_cap)))
    if not math.isfinite(loss):
        raise ValueError("Quantile loss is non-finite")
    return loss


def estimate_heel_registration(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    base_prediction: Any,
    *,
    config: HeelRegistrationConfig = DEFAULT_HEEL_REGISTRATION_CONFIG,
) -> HeelRegistrationResult:
    """Estimate a constant TVT shift using only inference-visible inputs.

    ``base_prediction`` must contain exactly the horizontal prediction suffix,
    in row order.  The returned shift can be added to that prediction.
    """
    if not isinstance(config, HeelRegistrationConfig):
        raise TypeError("config must be a HeelRegistrationConfig")
    if not isinstance(horizontal, pd.DataFrame):
        raise TypeError("horizontal must be a pandas DataFrame")
    horizontal_view = select_inference_view(horizontal, kind="horizontal")
    known_count, prediction_count = validate_prefix(horizontal_view["TVT_input"])
    base = _as_float_vector(base_prediction, name="base_prediction")
    if len(base) != prediction_count:
        raise ValueError(
            "base_prediction length must equal the horizontal prediction suffix"
        )
    if not np.isfinite(base).all():
        raise ValueError("base_prediction must be finite")

    horizontal_gr = _as_float_vector(horizontal_view["GR"], name="horizontal GR")
    prefix_gr = _interpolate_and_median(
        horizontal_gr[:known_count],
        width=config.rolling_median_width,
        name="prefix horizontal GR",
    )
    suffix_gr = _interpolate_and_median(
        horizontal_gr[known_count:],
        width=config.rolling_median_width,
        name="suffix horizontal GR",
    )
    prefix_tvt = _as_float_vector(
        horizontal_view["TVT_input"].iloc[:known_count],
        name="visible prefix TVT_input",
    )
    if not np.isfinite(prefix_tvt).all():
        raise ValueError("Visible prefix TVT_input must be finite")

    typewell_tvt, typewell_gr, support_min, support_max = _validated_typewell(typewell)
    calibration = _fit_prefix_affine(
        prefix_gr,
        prefix_tvt,
        typewell_tvt,
        typewell_gr,
        support_min=support_min,
        support_max=support_max,
        config=config,
    )
    calibrated_suffix = (calibration.intercept + calibration.slope * suffix_gr).astype(
        np.float64
    )
    if not np.isfinite(calibrated_suffix).all():
        raise ValueError("Calibrated suffix GR must be finite")

    sampled_index = _equally_spaced_indices(prediction_count, config.max_suffix_rows)
    sampled_horizontal = calibrated_suffix[sampled_index]
    sampled_base = base[sampled_index]
    shifts = config.shift_grid
    costs = np.empty(len(shifts), dtype=np.float64)
    coverage = np.empty(len(shifts), dtype=np.float64)
    support_count = np.empty(len(shifts), dtype=np.int64)

    for candidate_index, shift in enumerate(shifts):
        candidate_tvt = sampled_base + shift
        supported = (candidate_tvt >= support_min) & (candidate_tvt <= support_max)
        count = int(supported.sum())
        candidate_coverage = float(count / len(sampled_index))
        coverage[candidate_index] = candidate_coverage
        support_count[candidate_index] = count
        if count == 0:
            distribution_loss = config.squared_loss_cap
        else:
            candidate_typewell_gr = np.interp(
                candidate_tvt[supported], typewell_tvt, typewell_gr
            )
            distribution_loss = clipped_quantile_loss(
                sampled_horizontal[supported],
                candidate_typewell_gr,
                quantile_count=config.quantile_count,
                squared_loss_cap=config.squared_loss_cap,
            )
        coverage_shortfall = max(0.0, config.minimum_coverage - candidate_coverage)
        costs[candidate_index] = (
            distribution_loss + config.coverage_penalty_scale * coverage_shortfall**2
        )

    if not np.isfinite(costs).all() or not np.isfinite(coverage).all():
        raise ValueError("Heel-registration candidate diagnostics must be finite")
    minimum_cost = float(np.min(costs))
    temperature = config.posterior_temperature_multiplier * max(
        calibration.prefix_mse, config.posterior_mse_floor
    )
    log_weight = -(costs - minimum_cost) / temperature
    posterior = np.exp(log_weight)
    normalizer = float(np.sum(posterior))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("Heel-registration posterior cannot be normalized")
    posterior /= normalizer

    mean_shift = float(posterior @ shifts)
    map_index = int(np.argmin(costs))
    positive = posterior > 0.0
    entropy = float(-np.sum(posterior[positive] * np.log(posterior[positive])))
    effective_support = float(math.exp(entropy))
    maximum_posterior = float(posterior[map_index])
    edge_mass = float(posterior[0] + posterior[-1])
    best_coverage = float(coverage[map_index])
    posterior_mean_coverage = float(posterior @ coverage)
    scalar_diagnostics = (
        mean_shift,
        float(shifts[map_index]),
        entropy,
        effective_support,
        maximum_posterior,
        edge_mass,
        best_coverage,
        posterior_mean_coverage,
    )
    if not all(math.isfinite(value) for value in scalar_diagnostics):
        raise ValueError("Heel-registration posterior diagnostics must be finite")

    return HeelRegistrationResult(
        posterior_mean_shift=mean_shift,
        map_shift=float(shifts[map_index]),
        posterior_entropy=entropy,
        effective_support=effective_support,
        maximum_posterior=maximum_posterior,
        edge_mass=edge_mass,
        best_coverage=best_coverage,
        posterior_mean_coverage=posterior_mean_coverage,
        calibration=calibration,
        sampled_suffix_row_index=sampled_index,
        shift_grid=shifts,
        candidate_cost=costs,
        candidate_coverage=coverage,
        candidate_support_count=support_count,
        posterior_weight=posterior,
    )


def estimate_heel_shift(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    base_prediction: Any,
    *,
    config: HeelRegistrationConfig = DEFAULT_HEEL_REGISTRATION_CONFIG,
) -> float:
    """Return only the posterior-mean shift for production integration."""
    return estimate_heel_registration(
        horizontal, typewell, base_prediction, config=config
    ).posterior_mean_shift


def heel_registration_manifest(
    config: HeelRegistrationConfig = DEFAULT_HEEL_REGISTRATION_CONFIG,
) -> dict[str, Any]:
    """Return the stable inference and algorithm contract."""
    if not isinstance(config, HeelRegistrationConfig):
        raise TypeError("config must be a HeelRegistrationConfig")
    return {
        "version": HEEL_REGISTRATION_VERSION,
        "horizontal_source_columns": [
            "MD",
            "X",
            "Y",
            "Z",
            "GR",
            "TVT_input",
        ],
        "horizontal_columns_used": ["GR", "TVT_input"],
        "typewell_source_columns": ["TVT", "GR"],
        "base_prediction_contract": "finite prediction-suffix TVT in horizontal row order",
        "config": asdict(config),
        "gr_preprocessing": (
            "prefix and suffix independently linearly interpolated by row, then "
            "centered rolling median"
        ),
        "prefix_calibration": (
            "OLS affine map from preprocessed horizontal prefix GR to typewell "
            "GR interpolated at visible TVT_input"
        ),
        "candidate_objective": (
            "clipped squared equally-spaced quantile loss plus coverage penalty"
        ),
        "posterior": (
            "exp(-(cost-min_cost)/(2*max(prefix_mse,1))) under default config"
        ),
    }


__all__ = [
    "DEFAULT_HEEL_REGISTRATION_CONFIG",
    "HEEL_REGISTRATION_VERSION",
    "HeelAffineCalibration",
    "HeelRegistrationConfig",
    "HeelRegistrationResult",
    "clipped_quantile_loss",
    "estimate_heel_registration",
    "estimate_heel_shift",
    "heel_registration_manifest",
]
