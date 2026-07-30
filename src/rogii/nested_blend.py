"""Fold-local component shrinkage for incumbent plus residual candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .portfolio_metrics import validate_oof_frame


NESTED_COMPONENT_SCHEMA_VERSION = "nested_component_shrinkage_v1"
COMPONENT_NAMES = ("common", "slope", "shape")


@dataclass(frozen=True)
class ComponentShrinkageState:
    """Three bounded residual-component weights learned on outer-training wells."""

    outer_fold: int
    candidate_column: str
    incumbent_column: str
    training_well_ids: tuple[str, ...]
    validation_well_ids: tuple[str, ...]
    weights: tuple[float, float, float]
    training_ids_sha256: str
    ridge: float

    def __post_init__(self) -> None:
        training = tuple(map(str, self.training_well_ids))
        validation = tuple(map(str, self.validation_well_ids))
        if (
            not training
            or not validation
            or len(set(training)) != len(training)
            or len(set(validation)) != len(validation)
            or set(training) & set(validation)
        ):
            raise ValueError("nested training/validation well scopes are invalid")
        weights = tuple(map(float, self.weights))
        if len(weights) != len(COMPONENT_NAMES) or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights
        ):
            raise ValueError("component weights must be finite values in [0, 1]")
        expected = _ids_sha256(training)
        if self.training_ids_sha256 != expected:
            raise ValueError("training_ids_sha256 does not match training_well_ids")
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be finite and nonnegative")
        object.__setattr__(self, "training_well_ids", training)
        object.__setattr__(self, "validation_well_ids", validation)
        object.__setattr__(self, "weights", weights)


def _ids_sha256(well_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for well_id in map(str, well_ids):
        digest.update(well_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def decompose_candidate_correction(
    frame: pd.DataFrame,
    *,
    incumbent_column: str,
    candidate_column: str,
    position_column: str = "row_index",
) -> pd.DataFrame:
    """Decompose candidate-minus-incumbent into orthogonal per-well components."""

    required = {
        "well_id",
        "row_index",
        "fold",
        incumbent_column,
        candidate_column,
        position_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"component decomposition missing columns: {sorted(missing)}")
    validate_oof_frame(
        frame.assign(y_true=np.zeros(len(frame), dtype=np.float64)),
        [incumbent_column, candidate_column],
    )
    result = frame.loc[:, ["well_id", "row_index", "fold"]].copy()
    common = np.empty(len(frame), dtype=np.float64)
    slope = np.empty(len(frame), dtype=np.float64)
    shape = np.empty(len(frame), dtype=np.float64)
    incumbent = frame[incumbent_column].to_numpy(dtype=np.float64)
    candidate = frame[candidate_column].to_numpy(dtype=np.float64)
    difference = candidate - incumbent
    position = frame[position_column].to_numpy(dtype=np.float64)
    for indices in frame.groupby("well_id", sort=False).indices.values():
        selected = np.asarray(indices, dtype=np.int64)
        current = difference[selected]
        current_position = position[selected]
        current_common = float(np.mean(current))
        centered = current - current_common
        x = current_position - float(np.mean(current_position))
        scale = float(np.max(np.abs(x))) if len(x) else 0.0
        normalized_x = x / scale if scale > 0.0 else np.zeros_like(x)
        denominator = float(normalized_x @ normalized_x)
        coefficient = (
            float(normalized_x @ centered) / denominator if denominator > 0.0 else 0.0
        )
        current_slope = coefficient * normalized_x
        common[selected] = current_common
        slope[selected] = current_slope
        shape[selected] = centered - current_slope
    if not (
        np.isfinite(common).all()
        and np.isfinite(slope).all()
        and np.isfinite(shape).all()
    ):
        raise ValueError("component decomposition produced non-finite values")
    reconstruction = incumbent + common + slope + shape
    maximum = float(np.max(np.abs(reconstruction - candidate)))
    if maximum > 1e-9:
        raise ValueError(f"component reconstruction failed: {maximum}")
    result["common"] = common
    result["slope"] = slope
    result["shape"] = shape
    return result


def fit_component_shrinkage(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    incumbent_column: str,
    candidate_column: str,
    truth_column: str = "y_true",
    ridge: float = 1e-6,
) -> ComponentShrinkageState:
    """Fit three bounded weights using only wells outside ``outer_fold``."""

    if truth_column not in frame:
        raise ValueError(f"missing truth column {truth_column!r}")
    outer_fold = int(outer_fold)
    folds = tuple(sorted(map(int, frame["fold"].unique())))
    if outer_fold not in folds:
        raise ValueError("outer_fold is absent from the frame")
    training = frame.loc[~frame["fold"].astype(int).eq(outer_fold)].copy()
    validation = frame.loc[frame["fold"].astype(int).eq(outer_fold)].copy()
    training_ids = tuple(sorted(training["well_id"].astype(str).unique()))
    validation_ids = tuple(sorted(validation["well_id"].astype(str).unique()))
    if (
        not training_ids
        or not validation_ids
        or set(training_ids) & set(validation_ids)
    ):
        raise ValueError("nested split is empty or overlapping")
    components = decompose_candidate_correction(
        training,
        incumbent_column=incumbent_column,
        candidate_column=candidate_column,
    )
    design = components.loc[:, list(COMPONENT_NAMES)].to_numpy(dtype=np.float64)
    residual_target = training[truth_column].to_numpy(dtype=np.float64) - training[
        incumbent_column
    ].to_numpy(dtype=np.float64)
    gram = design.T @ design
    scale = max(float(np.trace(gram)) / len(COMPONENT_NAMES), 1.0)
    regularized = gram + float(ridge) * scale * np.eye(len(COMPONENT_NAMES))
    rhs = design.T @ residual_target
    try:
        weights = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        weights = np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
    weights = np.clip(weights, 0.0, 1.0)
    if not np.isfinite(weights).all():
        weights = np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
    return ComponentShrinkageState(
        outer_fold=outer_fold,
        candidate_column=str(candidate_column),
        incumbent_column=str(incumbent_column),
        training_well_ids=training_ids,
        validation_well_ids=validation_ids,
        weights=tuple(map(float, weights)),
        training_ids_sha256=_ids_sha256(training_ids),
        ridge=float(ridge),
    )


def apply_component_shrinkage(
    frame: pd.DataFrame,
    state: ComponentShrinkageState,
    *,
    prediction_column: str | None = None,
) -> pd.Series:
    """Apply a fitted state only to its declared outer-validation wells."""

    if not isinstance(state, ComponentShrinkageState):
        raise TypeError("state must be ComponentShrinkageState")
    actual = tuple(sorted(frame["well_id"].astype(str).unique()))
    if actual != state.validation_well_ids:
        raise ValueError("frame does not contain exactly the state's validation wells")
    folds = tuple(sorted(map(int, frame["fold"].unique())))
    if folds != (state.outer_fold,):
        raise ValueError("frame fold does not match the state's outer fold")
    components = decompose_candidate_correction(
        frame,
        incumbent_column=state.incumbent_column,
        candidate_column=state.candidate_column,
    )
    design = components.loc[:, list(COMPONENT_NAMES)].to_numpy(dtype=np.float64)
    prediction = frame[state.incumbent_column].to_numpy(
        dtype=np.float64
    ) + design @ np.asarray(state.weights, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("nested component prediction is non-finite")
    return pd.Series(
        prediction,
        index=frame.index,
        name=(
            str(prediction_column)
            if prediction_column is not None
            else f"nested_{state.candidate_column}"
        ),
    )


def cross_fit_component_shrinkage(
    frame: pd.DataFrame,
    *,
    incumbent_column: str,
    candidate_columns: Sequence[str],
    truth_column: str = "y_true",
    ridge: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Return untouched outer-fold predictions and serializable fit provenance."""

    candidates = tuple(map(str, candidate_columns))
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate_columns must be non-empty and unique")
    validate_oof_frame(
        frame,
        [incumbent_column, *candidates],
        truth_column=truth_column,
    )
    result = frame.copy()
    provenance: dict[str, list[dict[str, Any]]] = {}
    folds = tuple(sorted(map(int, frame["fold"].unique())))
    for candidate in candidates:
        output = np.full(len(frame), np.nan, dtype=np.float64)
        records: list[dict[str, Any]] = []
        for outer_fold in folds:
            state = fit_component_shrinkage(
                frame,
                outer_fold=outer_fold,
                incumbent_column=incumbent_column,
                candidate_column=candidate,
                truth_column=truth_column,
                ridge=ridge,
            )
            mask = frame["fold"].astype(int).eq(outer_fold)
            validation = frame.loc[mask]
            output[mask.to_numpy()] = apply_component_shrinkage(
                validation, state
            ).to_numpy(dtype=np.float64)
            records.append(
                {
                    "schema_version": NESTED_COMPONENT_SCHEMA_VERSION,
                    "outer_fold": state.outer_fold,
                    "candidate_column": state.candidate_column,
                    "incumbent_column": state.incumbent_column,
                    "training_well_ids": list(state.training_well_ids),
                    "validation_well_ids": list(state.validation_well_ids),
                    "training_validation_intersection": [],
                    "training_ids_sha256": state.training_ids_sha256,
                    "weights": dict(zip(COMPONENT_NAMES, state.weights, strict=True)),
                    "ridge": state.ridge,
                    "outer_validation_labels_used_for_fit": False,
                }
            )
        if not np.isfinite(output).all():
            raise ValueError(f"cross-fit output is incomplete for {candidate!r}")
        result[f"nested_{candidate}"] = output
        provenance[candidate] = records
    return result, provenance


__all__ = [
    "COMPONENT_NAMES",
    "NESTED_COMPONENT_SCHEMA_VERSION",
    "ComponentShrinkageState",
    "apply_component_shrinkage",
    "cross_fit_component_shrinkage",
    "decompose_candidate_correction",
    "fit_component_shrinkage",
]
