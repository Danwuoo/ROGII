"""Identifiable common/slope/shape residual fields and fail-closed validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from .contracts import PREDICTION_STATUS_IDENTITY, PREDICTION_STATUS_OK


class PredictionFailure(RuntimeError):
    """Raised when a family cannot produce a valid, eligible prediction."""


def _vector(values: Any, *, name: str) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _coordinate(values: Any, *, expected: int | None = None) -> np.ndarray:
    coordinate = _vector(values, name="coordinate")
    if expected is not None and coordinate.shape != (expected,):
        raise ValueError("coordinate length mismatch")
    if np.any(np.diff(coordinate) < 0.0):
        raise ValueError("coordinate must be monotone")
    return coordinate


def _linear_basis(coordinate: np.ndarray) -> np.ndarray:
    return coordinate - float(np.mean(coordinate))


def orthogonalize_shape(raw_shape: Any, coordinate: Any) -> np.ndarray:
    """Remove the constant and linear projections from a proposed shape field."""

    shape = _vector(raw_shape, name="raw_shape")
    x = _coordinate(coordinate, expected=len(shape))
    centered = np.asarray(shape - float(np.mean(shape)), dtype=np.float64)
    basis = _linear_basis(x)
    denominator = float(basis @ basis)
    if denominator > np.finfo(np.float64).eps:
        centered -= float(centered @ basis / denominator) * basis
    # A final mean subtraction protects the invariant from accumulated roundoff.
    centered -= float(np.mean(centered))
    centered.setflags(write=False)
    return centered


@dataclass(frozen=True)
class ResidualComponents:
    """A complete residual prediction with explicit eligibility semantics."""

    common: np.ndarray
    slope: np.ndarray
    shape: np.ndarray
    correction: np.ndarray
    final: np.ndarray
    status: str
    eligible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        arrays = {
            name: _vector(getattr(self, name), name=name)
            for name in ("common", "slope", "shape", "correction", "final")
        }
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("all residual component arrays must have equal length")
        status = str(self.status)
        if status not in {PREDICTION_STATUS_OK, PREDICTION_STATUS_IDENTITY}:
            raise ValueError("unknown residual prediction status")
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be bool")
        reason = None if self.reason is None else str(self.reason)
        if self.eligible:
            if status != PREDICTION_STATUS_OK or reason is not None:
                raise ValueError(
                    "eligible predictions must have status='ok' and no reason"
                )
        elif status != PREDICTION_STATUS_IDENTITY or not reason:
            raise ValueError(
                "ineligible predictions must be explicit identity fallbacks with a reason"
            )
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)


def validate_residual_components(
    components: ResidualComponents,
    incumbent_prediction: Any,
    coordinate: Any,
    *,
    atol: float = 1e-8,
) -> ResidualComponents:
    """Validate algebra, identifiability, and explicit identity behavior."""

    if not isinstance(components, ResidualComponents):
        raise TypeError("components must be a ResidualComponents instance")
    incumbent = _vector(incumbent_prediction, name="incumbent_prediction")
    x = _coordinate(coordinate, expected=len(incumbent))
    arrays = (
        components.common,
        components.slope,
        components.shape,
        components.correction,
        components.final,
    )
    if any(value.shape != incumbent.shape for value in arrays):
        raise ValueError("residual components do not align with incumbent rows")
    if isinstance(atol, bool) or not math.isfinite(float(atol)) or atol <= 0.0:
        raise ValueError("atol must be finite and positive")
    scale = max(
        1.0,
        float(np.max(np.abs(incumbent))),
        *(float(np.max(np.abs(value))) for value in arrays),
    )
    tolerance = float(atol) * scale

    if float(np.max(np.abs(components.common - components.common[0]))) > tolerance:
        raise ValueError("common component must be constant within a well")
    basis = _linear_basis(x)
    if abs(float(np.mean(components.slope))) > tolerance:
        raise ValueError("slope component must have zero row mean")
    denominator = float(basis @ basis)
    if denominator <= np.finfo(np.float64).eps:
        if float(np.max(np.abs(components.slope))) > tolerance:
            raise ValueError("slope must be zero for a degenerate coordinate")
    else:
        coefficient = float(components.slope @ basis / denominator)
        if float(np.max(np.abs(components.slope - coefficient * basis))) > tolerance:
            raise ValueError("slope component must be linear in output coordinate")
    if abs(float(np.mean(components.shape))) > tolerance:
        raise ValueError("shape component must have zero row mean")
    if denominator > np.finfo(np.float64).eps:
        shape_projection = abs(float(components.shape @ basis))
        projection_tolerance = tolerance * max(1.0, float(np.sum(np.abs(basis))))
        if shape_projection > projection_tolerance:
            raise ValueError("shape component must be orthogonal to linear slope")

    expected_correction = components.common + components.slope + components.shape
    if not np.allclose(
        components.correction,
        expected_correction,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("correction does not equal common + slope + shape")
    if not np.allclose(
        components.final,
        incumbent + components.correction,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("final does not equal incumbent + correction")

    if not components.eligible:
        zero = max(
            float(np.max(np.abs(components.common))),
            float(np.max(np.abs(components.slope))),
            float(np.max(np.abs(components.shape))),
            float(np.max(np.abs(components.correction))),
        )
        if zero > tolerance or not np.allclose(
            components.final, incumbent, rtol=0.0, atol=tolerance
        ):
            raise ValueError("identity fallback must preserve incumbent exactly")
    return components


def compose_residual_components(
    incumbent_prediction: Any,
    coordinate: Any,
    *,
    common_coefficient: float,
    slope_coefficient: float,
    raw_shape: Any | None = None,
) -> ResidualComponents:
    """Build an eligible residual field with identifiable component projections."""

    incumbent = _vector(incumbent_prediction, name="incumbent_prediction")
    x = _coordinate(coordinate, expected=len(incumbent))
    common_value = float(common_coefficient)
    slope_value = float(slope_coefficient)
    if not math.isfinite(common_value) or not math.isfinite(slope_value):
        raise ValueError("common and slope coefficients must be finite")
    common = np.full(len(incumbent), common_value, dtype=np.float64)
    slope = slope_value * _linear_basis(x)
    shape = (
        np.zeros(len(incumbent), dtype=np.float64)
        if raw_shape is None
        else orthogonalize_shape(raw_shape, x)
    )
    correction = common + slope + shape
    final = incumbent + correction
    components = ResidualComponents(
        common=common,
        slope=slope,
        shape=shape,
        correction=correction,
        final=final,
        status=PREDICTION_STATUS_OK,
        eligible=True,
    )
    return validate_residual_components(components, incumbent, x)


def decompose_residual_target(
    residual: Any,
    coordinate: Any,
    *,
    incumbent_prediction: Any | None = None,
) -> ResidualComponents:
    """Project a residual target exactly into common, slope, and shape fields."""

    target = _vector(residual, name="residual")
    x = _coordinate(coordinate, expected=len(target))
    incumbent = (
        np.zeros(len(target), dtype=np.float64)
        if incumbent_prediction is None
        else _vector(incumbent_prediction, name="incumbent_prediction")
    )
    if incumbent.shape != target.shape:
        raise ValueError("incumbent_prediction length mismatch")
    common = float(np.mean(target))
    basis = _linear_basis(x)
    centered = target - common
    denominator = float(basis @ basis)
    slope = (
        0.0
        if denominator <= np.finfo(np.float64).eps
        else float(centered @ basis / denominator)
    )
    shape = centered - slope * basis
    result = compose_residual_components(
        incumbent,
        x,
        common_coefficient=common,
        slope_coefficient=slope,
        raw_shape=shape,
    )
    # The projection should reconstruct to floating-point precision.
    tolerance = max(1e-10, float(np.max(np.abs(target))) * 1e-12)
    if not np.allclose(result.correction, target, rtol=0.0, atol=tolerance):
        raise ValueError("residual decomposition failed exact reconstruction")
    return result


def identity_residual_components(
    incumbent_prediction: Any,
    coordinate: Any,
    *,
    reason: str,
) -> ResidualComponents:
    """Return an explicit, ineligible identity fallback; never silently eligible."""

    incumbent = _vector(incumbent_prediction, name="incumbent_prediction")
    x = _coordinate(coordinate, expected=len(incumbent))
    message = str(reason).strip()
    if not message:
        raise ValueError("identity fallback requires a non-empty reason")
    zero = np.zeros(len(incumbent), dtype=np.float64)
    result = ResidualComponents(
        common=zero,
        slope=zero,
        shape=zero,
        correction=zero,
        final=incumbent,
        status=PREDICTION_STATUS_IDENTITY,
        eligible=False,
        reason=message,
    )
    return validate_residual_components(result, incumbent, x)


def guarded_residual_prediction(
    producer: Callable[[], ResidualComponents],
    incumbent_prediction: Any,
    coordinate: Any,
    *,
    allow_identity_fallback: bool = False,
    failure_label: str = "external residual family",
) -> ResidualComponents:
    """Validate a family prediction or fail closed with an explicit opt-in fallback."""

    if not callable(producer):
        raise TypeError("producer must be callable")
    incumbent = _vector(incumbent_prediction, name="incumbent_prediction")
    x = _coordinate(coordinate, expected=len(incumbent))
    try:
        result = producer()
        return validate_residual_components(result, incumbent, x)
    except Exception as error:
        if not allow_identity_fallback:
            raise PredictionFailure(
                f"{failure_label} failed validation: {type(error).__name__}: {error}"
            ) from error
        return identity_residual_components(
            incumbent,
            x,
            reason=f"{failure_label}: {type(error).__name__}: {error}",
        )


__all__ = [
    "PredictionFailure",
    "ResidualComponents",
    "compose_residual_components",
    "decompose_residual_target",
    "guarded_residual_prediction",
    "identity_residual_components",
    "orthogonalize_shape",
    "validate_residual_components",
]
