"""Family-balanced constrained and strictly nested OOF ensembling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .portfolio_metrics import (
    DEFAULT_TRUTH_COLUMN,
    OOF_KEY_COLUMNS,
    validate_oof_frame,
)


def _family_specification(
    frame: pd.DataFrame, families: Mapping[str, Sequence[str]]
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    if not isinstance(families, Mapping) or not families:
        raise ValueError("families must be a non-empty family-to-columns mapping")
    names = tuple(map(str, families.keys()))
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Family names must be non-empty and unique")
    normalized: dict[str, tuple[str, ...]] = {}
    claimed: set[str] = set()
    for original_name, name in zip(families.keys(), names, strict=True):
        members = tuple(map(str, families[original_name]))
        if not members or len(set(members)) != len(members):
            raise ValueError(f"Family {name!r} must have unique prediction columns")
        missing = set(members) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Family {name!r} is missing prediction columns: {sorted(missing)}"
            )
        overlap = claimed & set(members)
        if overlap:
            raise ValueError(
                f"A seed/model column may belong to only one family: {sorted(overlap)}"
            )
        claimed.update(members)
        normalized[name] = members
    return names, normalized


def average_family_predictions(
    frame: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Average seeds/models inside each family before cross-family blending."""

    names, normalized = _family_specification(frame, families)
    result: dict[str, np.ndarray] = {}
    for family in names:
        values = (
            frame.loc[:, list(normalized[family])]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Family {family!r} has non-finite predictions")
        result[family] = np.mean(values, axis=1, dtype=np.float64)
    return pd.DataFrame(result, index=frame.index)


def simple_blend_target(
    family_names: Sequence[str],
    *,
    incumbent_family: str | None = None,
    incumbent_weight: float = 0.70,
) -> dict[str, float]:
    """Create either a uniform or configurable incumbent-heavy target."""

    families = tuple(map(str, family_names))
    if not families or len(set(families)) != len(families):
        raise ValueError("family_names must be non-empty and unique")
    if len(families) == 1:
        if incumbent_family is not None and incumbent_family not in families:
            raise ValueError("incumbent_family is not present")
        return {families[0]: 1.0}
    if incumbent_family is None:
        return {family: 1.0 / len(families) for family in families}
    if incumbent_family not in families:
        raise ValueError(f"Unknown incumbent family {incumbent_family!r}")
    if not math.isfinite(incumbent_weight) or not 0.0 <= incumbent_weight <= 1.0:
        raise ValueError("incumbent_weight must be finite and in [0, 1]")
    remainder = (1.0 - incumbent_weight) / (len(families) - 1)
    return {
        family: incumbent_weight if family == incumbent_family else remainder
        for family in families
    }


def _weight_vector(
    value: Mapping[str, float] | Sequence[float] | None,
    families: Sequence[str],
    *,
    incumbent_family: str | None,
    incumbent_weight: float,
) -> np.ndarray:
    names = tuple(families)
    if value is None:
        mapping = simple_blend_target(
            names,
            incumbent_family=incumbent_family,
            incumbent_weight=incumbent_weight,
        )
        vector = np.asarray([mapping[name] for name in names], dtype=np.float64)
    elif isinstance(value, Mapping):
        mapping = {str(key): float(current) for key, current in value.items()}
        if set(mapping) != set(names):
            raise ValueError("Regularization target keys must match family names")
        vector = np.asarray([mapping[name] for name in names], dtype=np.float64)
    else:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (len(names),):
            raise ValueError("Regularization target has the wrong length")
    if (
        not np.isfinite(vector).all()
        or np.any(vector < 0.0)
        or not math.isclose(float(vector.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(
            "Regularization target must be finite, nonnegative, and sum to one"
        )
    return vector


def _maximum_vector(
    family_max_weight: Mapping[str, float] | None, families: Sequence[str]
) -> np.ndarray:
    names = tuple(families)
    maximum = np.ones(len(names), dtype=np.float64)
    if family_max_weight is not None:
        configured = {
            str(key): float(current) for key, current in family_max_weight.items()
        }
        unknown = set(configured) - set(names)
        if unknown:
            raise ValueError(f"Unknown family max-weight keys: {sorted(unknown)}")
        for index, family in enumerate(names):
            if family in configured:
                maximum[index] = configured[family]
    if not np.isfinite(maximum).all() or np.any(maximum < 0.0) or np.any(maximum > 1.0):
        raise ValueError("Family maximum weights must be finite and in [0, 1]")
    if float(maximum.sum()) < 1.0 - 1e-12:
        raise ValueError("Family maximum weights make the simplex infeasible")
    return maximum


def _project_capped_simplex(values: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``sum(w)=1, 0<=w<=maximum``."""

    values = np.asarray(values, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    if values.shape != maximum.shape or float(maximum.sum()) < 1.0 - 1e-12:
        raise ValueError("Cannot project onto an infeasible capped simplex")
    low = float(np.min(values - maximum)) - 1.0
    high = float(np.max(values)) + 1.0
    for _ in range(200):
        midpoint = (low + high) / 2.0
        projected = np.clip(values - midpoint, 0.0, maximum)
        if float(projected.sum()) > 1.0:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(values - (low + high) / 2.0, 0.0, maximum)
    difference = 1.0 - float(projected.sum())
    if abs(difference) > 1e-12:
        if difference > 0.0:
            room = maximum - projected
        else:
            room = projected
        for index in np.argsort(-room):
            adjustment = math.copysign(
                min(abs(difference), float(room[index])), difference
            )
            projected[index] += adjustment
            difference -= adjustment
            if abs(difference) <= 1e-12:
                break
    if (
        abs(float(projected.sum()) - 1.0) > 1e-9
        or np.any(projected < -1e-12)
        or np.any(projected > maximum + 1e-12)
    ):
        raise ValueError("Capped-simplex projection failed")
    return projected


@dataclass(frozen=True)
class BlendFit:
    """Serializable result of one constrained weight fit."""

    family_names: tuple[str, ...]
    weights: tuple[float, ...]
    target_weights: tuple[float, ...]
    maximum_weights: tuple[float, ...]
    regularization_strength: float
    training_rows: int
    training_rmse: float
    objective_value: float
    optimizer_iterations: int
    optimizer_message: str

    @property
    def weight_by_family(self) -> dict[str, float]:
        return dict(zip(self.family_names, self.weights, strict=True))

    @property
    def target_by_family(self) -> dict[str, float]:
        return dict(zip(self.family_names, self.target_weights, strict=True))

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_names": list(self.family_names),
            "weights": self.weight_by_family,
            "target_weights": self.target_by_family,
            "maximum_weights": dict(
                zip(self.family_names, self.maximum_weights, strict=True)
            ),
            "regularization_strength": self.regularization_strength,
            "training_rows": self.training_rows,
            "training_rmse": self.training_rmse,
            "objective_value": self.objective_value,
            "optimizer_iterations": self.optimizer_iterations,
            "optimizer_message": self.optimizer_message,
        }


def fit_constrained_blend(
    truth: Sequence[float] | np.ndarray,
    family_predictions: pd.DataFrame,
    *,
    regularization_strength: float = 0.0,
    regularization_target: Mapping[str, float] | Sequence[float] | None = None,
    incumbent_family: str | None = None,
    incumbent_target_weight: float = 0.70,
    family_max_weight: Mapping[str, float] | None = None,
    max_iterations: int = 1_000,
    tolerance: float = 1e-12,
) -> BlendFit:
    """Fit nonnegative sum-one weights under optional caps and shrinkage."""

    if not isinstance(family_predictions, pd.DataFrame) or family_predictions.empty:
        raise ValueError("family_predictions must be a non-empty DataFrame")
    families = tuple(map(str, family_predictions.columns))
    if len(set(families)) != len(families) or any(not name for name in families):
        raise ValueError("Family prediction columns must be non-empty and unique")
    prediction = family_predictions.apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64
    )
    target_truth = np.asarray(truth, dtype=np.float64)
    if target_truth.ndim != 1 or target_truth.shape[0] != prediction.shape[0]:
        raise ValueError("Truth and family prediction rows differ")
    if (
        not np.isfinite(target_truth).all()
        or not np.isfinite(prediction).all()
        or len(target_truth) == 0
    ):
        raise ValueError("Truth and family predictions must be finite and non-empty")
    if not math.isfinite(regularization_strength) or regularization_strength < 0.0:
        raise ValueError("regularization_strength must be finite and nonnegative")
    if max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("Optimizer controls must be positive")
    target = _weight_vector(
        regularization_target,
        families,
        incumbent_family=incumbent_family,
        incumbent_weight=incumbent_target_weight,
    )
    maximum = _maximum_vector(family_max_weight, families)
    initial = _project_capped_simplex(target, maximum)

    def objective(weight: np.ndarray) -> float:
        error = prediction @ weight - target_truth
        penalty = weight - target
        return float(
            np.mean(error * error) + regularization_strength * float(penalty @ penalty)
        )

    def gradient(weight: np.ndarray) -> np.ndarray:
        error = prediction @ weight - target_truth
        return 2.0 * (prediction.T @ error) / len(
            error
        ) + 2.0 * regularization_strength * (weight - target)

    if len(families) == 1:
        optimized = np.ones(1, dtype=np.float64)
        iterations = 0
        message = "single family; no optimization required"
    else:
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, float(limit)) for limit in maximum],
            constraints={
                "type": "eq",
                "fun": lambda weight: float(np.sum(weight) - 1.0),
                "jac": lambda weight: np.ones_like(weight),
            },
            options={
                "maxiter": int(max_iterations),
                "ftol": float(tolerance),
                "disp": False,
            },
        )
        if not result.success:
            raise RuntimeError(
                f"Constrained blend optimization failed: {result.message}"
            )
        optimized = _project_capped_simplex(
            np.asarray(result.x, dtype=np.float64), maximum
        )
        iterations = int(result.nit)
        message = str(result.message)
    if (
        np.any(optimized < -1e-10)
        or np.any(optimized > maximum + 1e-10)
        or abs(float(optimized.sum()) - 1.0) > 1e-9
    ):
        raise RuntimeError("Optimizer returned infeasible blend weights")
    training_error = prediction @ optimized - target_truth
    training_rmse = math.sqrt(
        float(training_error @ training_error) / len(target_truth)
    )
    return BlendFit(
        family_names=families,
        weights=tuple(map(float, optimized)),
        target_weights=tuple(map(float, target)),
        maximum_weights=tuple(map(float, maximum)),
        regularization_strength=float(regularization_strength),
        training_rows=int(len(target_truth)),
        training_rmse=training_rmse,
        objective_value=objective(optimized),
        optimizer_iterations=iterations,
        optimizer_message=message,
    )


def apply_blend(
    family_predictions: pd.DataFrame,
    weights: BlendFit | Mapping[str, float],
) -> np.ndarray:
    """Apply audited family weights, rejecting schema drift and non-finite output."""

    if isinstance(weights, BlendFit):
        mapping = weights.weight_by_family
    elif isinstance(weights, Mapping):
        mapping = {str(key): float(value) for key, value in weights.items()}
    else:
        raise TypeError("weights must be BlendFit or a family-to-weight mapping")
    families = tuple(map(str, family_predictions.columns))
    if set(mapping) != set(families):
        raise ValueError("Weight families differ from family prediction columns")
    vector = np.asarray([mapping[family] for family in families], dtype=np.float64)
    if (
        not np.isfinite(vector).all()
        or np.any(vector < 0.0)
        or not math.isclose(float(vector.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("Blend weights must be finite, nonnegative, and sum to one")
    values = (
        family_predictions.loc[:, list(families)]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float64)
    )
    prediction = values @ vector
    if not np.isfinite(values).all() or not np.isfinite(prediction).all():
        raise ValueError("Family or blended predictions contain non-finite values")
    return prediction


def fit_family_blend(
    frame: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    **fit_options: Any,
) -> BlendFit:
    """Average each family and fit one cross-family constrained blend."""

    if truth_column not in frame:
        raise ValueError(f"Missing truth column {truth_column!r}")
    averaged = average_family_predictions(frame, families)
    return fit_constrained_blend(
        frame[truth_column].to_numpy(dtype=np.float64),
        averaged,
        **fit_options,
    )


@dataclass(frozen=True)
class NestedBlendProvenance:
    """Required evidence boundary for one outer-fold blend-weight fit."""

    outer_fold: int
    outer_training_well_ids: tuple[str, ...]
    outer_validation_well_ids: tuple[str, ...]
    source_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        training = tuple(sorted(map(str, self.outer_training_well_ids)))
        validation = tuple(sorted(map(str, self.outer_validation_well_ids)))
        artifacts = tuple(sorted(map(str, self.source_artifacts)))
        if self.outer_fold < 0:
            raise ValueError("outer_fold must be nonnegative")
        if (
            not training
            or not validation
            or len(set(training)) != len(training)
            or len(set(validation)) != len(validation)
        ):
            raise ValueError(
                "Outer training/validation provenance must be non-empty and unique"
            )
        overlap = set(training) & set(validation)
        if overlap:
            raise ValueError(
                f"Outer training and validation wells overlap: {sorted(overlap)[:5]}"
            )
        if (
            not artifacts
            or any(not value for value in artifacts)
            or len(set(artifacts)) != len(artifacts)
        ):
            raise ValueError(
                "source artifact IDs must be non-empty, unique, and present"
            )
        object.__setattr__(self, "outer_training_well_ids", training)
        object.__setattr__(self, "outer_validation_well_ids", validation)
        object.__setattr__(self, "source_artifacts", artifacts)

    @property
    def digest(self) -> str:
        payload = {
            "outer_fold": int(self.outer_fold),
            "outer_training_well_ids": list(self.outer_training_well_ids),
            "outer_validation_well_ids": list(self.outer_validation_well_ids),
            "source_artifacts": list(self.source_artifacts),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NestedFoldBlendResult:
    """Cross-fitted predictions and weights for one untouched outer fold."""

    outer_fold: int
    provenance_digest: str
    source_artifacts: tuple[str, ...]
    fit: BlendFit
    inner_fold_count: int
    inner_wells: int
    outer_wells: int
    inner_rmse: float
    outer_rmse: float
    prediction_column: str
    predictions: pd.DataFrame

    @property
    def weights(self) -> dict[str, float]:
        return self.fit.weight_by_family

    def summary(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "provenance_digest": self.provenance_digest,
            "source_artifacts": list(self.source_artifacts),
            "fit": self.fit.as_dict(),
            "inner_fold_count": self.inner_fold_count,
            "inner_wells": self.inner_wells,
            "outer_wells": self.outer_wells,
            "inner_rmse": self.inner_rmse,
            "outer_rmse": self.outer_rmse,
            "prediction_column": self.prediction_column,
        }


def _prediction_members(families: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    return tuple(str(column) for columns in families.values() for column in columns)


def fit_nested_fold_blend(
    inner_oof: pd.DataFrame,
    outer_validation_oof: pd.DataFrame,
    *,
    families: Mapping[str, Sequence[str]],
    provenance: NestedBlendProvenance,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    prediction_column: str = "y_pred_nested_blend",
    minimum_inner_folds: int = 2,
    **fit_options: Any,
) -> NestedFoldBlendResult:
    """Fit weights on outer-training inner OOF and predict untouched outer wells."""

    if not isinstance(provenance, NestedBlendProvenance):
        raise TypeError("A NestedBlendProvenance instance is required")
    if minimum_inner_folds < 2:
        raise ValueError("minimum_inner_folds must be at least two")
    if prediction_column in {*OOF_KEY_COLUMNS, truth_column}:
        raise ValueError("prediction_column conflicts with the OOF schema")
    members = _prediction_members(families)
    _family_specification(inner_oof, families)
    _family_specification(outer_validation_oof, families)
    inner_ids = set(inner_oof["well_id"].astype(str))
    outer_training_ids = set(provenance.outer_training_well_ids)
    outer_validation_ids = set(provenance.outer_validation_well_ids)
    leaked = inner_ids & outer_validation_ids
    if leaked:
        raise ValueError(
            f"Outer-validation IDs appeared in blend-weight fit: {sorted(leaked)[:5]}"
        )
    validate_oof_frame(
        inner_oof,
        members,
        truth_column=truth_column,
        expected_well_ids=provenance.outer_training_well_ids,
    )
    if inner_ids != outer_training_ids:
        raise ValueError(
            "Inner OOF must cover every and only outer-training well: "
            f"missing={len(outer_training_ids - inner_ids)}, "
            f"extra={len(inner_ids - outer_training_ids)}"
        )
    inner_folds = tuple(sorted(map(int, inner_oof["fold"].unique())))
    if len(inner_folds) < minimum_inner_folds:
        raise ValueError(
            f"Inner OOF has {len(inner_folds)} folds; need {minimum_inner_folds}"
        )

    validate_oof_frame(
        outer_validation_oof,
        members,
        truth_column=truth_column,
        expected_well_ids=provenance.outer_validation_well_ids,
        expected_folds=(provenance.outer_fold,),
    )
    actual_outer_ids = set(outer_validation_oof["well_id"].astype(str))
    if actual_outer_ids != outer_validation_ids:
        raise ValueError("Outer validation frame differs from provenance")

    inner_family = average_family_predictions(inner_oof, families)
    outer_family = average_family_predictions(outer_validation_oof, families)
    fit = fit_constrained_blend(
        inner_oof[truth_column].to_numpy(dtype=np.float64),
        inner_family,
        **fit_options,
    )
    inner_prediction = apply_blend(inner_family, fit)
    outer_prediction = apply_blend(outer_family, fit)
    inner_error = inner_prediction - inner_oof[truth_column].to_numpy(dtype=np.float64)
    outer_error = outer_prediction - outer_validation_oof[truth_column].to_numpy(
        dtype=np.float64
    )
    prediction_frame = outer_validation_oof.loc[
        :, [*OOF_KEY_COLUMNS, truth_column]
    ].copy()
    for family in inner_family.columns:
        prediction_frame[f"y_pred_family__{family}"] = outer_family[family].to_numpy(
            dtype=np.float64
        )
    prediction_frame[prediction_column] = outer_prediction
    validate_oof_frame(
        prediction_frame,
        [
            *(f"y_pred_family__{family}" for family in inner_family.columns),
            prediction_column,
        ],
        truth_column=truth_column,
        expected_well_ids=provenance.outer_validation_well_ids,
        expected_folds=(provenance.outer_fold,),
    )
    return NestedFoldBlendResult(
        outer_fold=int(provenance.outer_fold),
        provenance_digest=provenance.digest,
        source_artifacts=provenance.source_artifacts,
        fit=fit,
        inner_fold_count=int(len(inner_folds)),
        inner_wells=int(len(inner_ids)),
        outer_wells=int(len(actual_outer_ids)),
        inner_rmse=math.sqrt(float(inner_error @ inner_error) / len(inner_error)),
        outer_rmse=math.sqrt(float(outer_error @ outer_error) / len(outer_error)),
        prediction_column=prediction_column,
        predictions=prediction_frame,
    )


def combine_nested_fold_predictions(
    results: Sequence[NestedFoldBlendResult],
    *,
    expected_keys: pd.DataFrame | None = None,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
) -> pd.DataFrame:
    """Combine untouched outer-fold outputs into one strict nested OOF artifact."""

    folds = tuple(results)
    if not folds:
        raise ValueError("At least one nested fold result is required")
    if not all(isinstance(result, NestedFoldBlendResult) for result in folds):
        raise TypeError("All results must be NestedFoldBlendResult instances")
    outer_folds = [result.outer_fold for result in folds]
    if len(set(outer_folds)) != len(outer_folds):
        raise ValueError("Nested results contain duplicate outer folds")
    prediction_columns = {result.prediction_column for result in folds}
    if len(prediction_columns) != 1:
        raise ValueError("Nested fold prediction column names differ")
    prediction_column = next(iter(prediction_columns))
    combined = pd.concat([result.predictions for result in folds], ignore_index=True)
    validate_oof_frame(
        combined,
        (prediction_column,),
        truth_column=truth_column,
        expected_keys=expected_keys,
        expected_folds=sorted(outer_folds),
    )
    return combined.sort_values(list(OOF_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def nested_weight_table(
    results: Sequence[NestedFoldBlendResult],
) -> pd.DataFrame:
    """Return fold-by-family weights for stability analysis."""

    rows: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, NestedFoldBlendResult):
            raise TypeError("All results must be NestedFoldBlendResult instances")
        for family, weight in result.weights.items():
            rows.append(
                {
                    "outer_fold": int(result.outer_fold),
                    "family": family,
                    "weight": float(weight),
                    "provenance_digest": result.provenance_digest,
                    "inner_rmse": result.inner_rmse,
                    "outer_rmse": result.outer_rmse,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["outer_fold", "family"], kind="stable")
        .reset_index(drop=True)
    )


__all__ = [
    "BlendFit",
    "NestedBlendProvenance",
    "NestedFoldBlendResult",
    "apply_blend",
    "average_family_predictions",
    "combine_nested_fold_predictions",
    "fit_constrained_blend",
    "fit_family_blend",
    "fit_nested_fold_blend",
    "nested_weight_table",
    "simple_blend_target",
]
