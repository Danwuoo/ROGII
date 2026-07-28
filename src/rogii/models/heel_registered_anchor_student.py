"""Deployable heel-registered sparse-anchor student.

This model is a deliberately thin production wrapper around
``SparseAnchorStudentModel``.  Fitting learns only the base model.  At
inference time, one deterministic common-mode shift is estimated from the
visible horizontal/typewell GR pair and added to both the base row prediction
and its predicted anchors.

The wrapper reads competition files only through the inference-schema loaders
in :mod:`rogii.data`; targets, formation surfaces, geology annotations, and
images are not part of its payload or prediction path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data import (
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    WellFiles,
    load_horizontal_inference,
    load_typewell_inference,
    validate_prefix,
)
from ..heel_registration import (
    DEFAULT_HEEL_REGISTRATION_CONFIG,
    HEEL_REGISTRATION_VERSION,
    HeelRegistrationConfig,
    HeelRegistrationResult,
    estimate_heel_registration,
    heel_registration_manifest,
)
from .base import RunContext, TrainingWell
from .sparse_anchor_student import (
    DEFAULT_RIDGE_ALPHA,
    DEFAULT_STUDENT_KIND,
    DEFAULT_TARGET_COMPONENTS,
    SparseAnchorInferenceWell,
    SparseAnchorPrediction,
    SparseAnchorStudentModel,
    SparseAnchorStudentState,
)


MODEL_ID = "heel_registered_anchor_student"


@dataclass(frozen=True)
class HeelRegisteredAnchorInferenceWell:
    """Inference-safe base payload plus exact safe input views for registration."""

    base: SparseAnchorInferenceWell
    horizontal: pd.DataFrame
    typewell: pd.DataFrame

    @property
    def well_id(self) -> str:
        return self.base.well_id

    @property
    def row_index(self) -> np.ndarray:
        return self.base.row_index

    @property
    def anchor(self) -> float:
        return self.base.anchor

    @property
    def features(self) -> np.ndarray:
        return self.base.features


@dataclass(frozen=True)
class HeelRegisteredAnchorStudentState:
    """Fold-fit base state and immutable registration/provenance contract."""

    base_state: SparseAnchorStudentState
    registration_config: HeelRegistrationConfig
    registration_version: str
    base_model_id: str
    training_well_ids: tuple[str, ...]
    fold: int
    random_seed: int


@dataclass(frozen=True)
class HeelRegisteredAnchorPrediction:
    """Base components, registration posterior, shifted anchors, and final TVT."""

    base: SparseAnchorPrediction
    registration: HeelRegistrationResult
    anchor_tvt: np.ndarray
    final: np.ndarray

    @property
    def posterior_mean_shift(self) -> float:
        return self.registration.posterior_mean_shift


def _validated_base_state(
    base_state: SparseAnchorStudentState,
    *,
    config: HeelRegistrationConfig,
) -> HeelRegisteredAnchorStudentState:
    if not isinstance(base_state, SparseAnchorStudentState):
        raise TypeError("base_state must be a SparseAnchorStudentState")
    if not isinstance(config, HeelRegistrationConfig):
        raise TypeError("config must be a HeelRegistrationConfig")
    ids = tuple(str(value) for value in base_state.training_well_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("base state training well IDs must be non-empty and unique")
    student_ids = tuple(
        str(value) for value in base_state.student_state.training_well_ids
    )
    if ids != student_ids:
        raise ValueError("base and student training-well provenance differ")
    fold = int(base_state.fold)
    seed = int(base_state.student_state.random_seed)
    return HeelRegisteredAnchorStudentState(
        base_state=base_state,
        registration_config=config,
        registration_version=HEEL_REGISTRATION_VERSION,
        base_model_id=SparseAnchorStudentModel.model_id,
        training_well_ids=ids,
        fold=fold,
        random_seed=seed,
    )


def wrap_sparse_anchor_student_state(
    base_state: SparseAnchorStudentState,
    *,
    config: HeelRegistrationConfig = DEFAULT_HEEL_REGISTRATION_CONFIG,
) -> HeelRegisteredAnchorStudentState:
    """Wrap an already-loaded fold state without fitting anything new."""

    return _validated_base_state(base_state, config=config)


def heel_registered_inference_schema() -> dict[str, Any]:
    """Return the exact file and calculation schema used at inference."""

    return {
        "horizontal_source_columns": list(HORIZONTAL_INFERENCE_COLUMNS),
        "horizontal_registration_columns_used": ["GR", "TVT_input"],
        "typewell_source_columns": list(TYPEWELL_INFERENCE_COLUMNS),
        "typewell_registration_columns_used": ["TVT", "GR"],
        "base_prediction_input": (
            "SparseAnchorStudentModel finite prediction suffix in horizontal row order"
        ),
        "target_columns_used": [],
        "formation_surface_columns_used": [],
        "geology_columns_used": [],
        "image_inputs_used": False,
    }


def heel_registered_anchor_student_manifest(
    *,
    config: HeelRegistrationConfig = DEFAULT_HEEL_REGISTRATION_CONFIG,
    student_kind: str = DEFAULT_STUDENT_KIND,
    alpha: float = DEFAULT_RIDGE_ALPHA,
    target_components: int = DEFAULT_TARGET_COMPONENTS,
    extra_trees_estimators: int = 160,
) -> dict[str, Any]:
    """Return a serializable architecture, schema, and fit-scope manifest."""

    if not isinstance(config, HeelRegistrationConfig):
        raise TypeError("config must be a HeelRegistrationConfig")
    return {
        "model_id": MODEL_ID,
        "base_model_id": SparseAnchorStudentModel.model_id,
        "base_model": {
            "student_kind": str(student_kind),
            "alpha": float(alpha),
            "target_components": int(target_components),
            "extra_trees_estimators": int(extra_trees_estimators),
        },
        "registration": heel_registration_manifest(config),
        "inference_schema": heel_registered_inference_schema(),
        "fit_contract": (
            "fit delegates entirely to SparseAnchorStudentModel within the outer "
            "training wells; heel registration has no learned cross-well state"
        ),
        "row_correction": "final = sparse_anchor_student.final + posterior_mean_shift",
        "anchor_correction": (
            "anchor_tvt = sparse_anchor_student.anchor_tvt + posterior_mean_shift"
        ),
    }


def heel_registered_state_provenance(
    state: HeelRegisteredAnchorStudentState,
) -> dict[str, Any]:
    """Return fold-safe provenance suitable for an artifact manifest."""

    if not isinstance(state, HeelRegisteredAnchorStudentState):
        raise TypeError("state must be a HeelRegisteredAnchorStudentState")
    return {
        "model_id": MODEL_ID,
        "base_model_id": state.base_model_id,
        "fold": state.fold,
        "random_seed": state.random_seed,
        "training_well_count": len(state.training_well_ids),
        "training_well_ids": list(state.training_well_ids),
        "registration_version": state.registration_version,
        "registration_config": asdict(state.registration_config),
        "registration_fit_wells": [],
        "registration_uses_cross_well_learned_statistics": False,
    }


class HeelRegisteredAnchorStudentModel:
    """Sparse-anchor student with inference-only heel GR common-mode correction."""

    model_id = MODEL_ID

    def __init__(
        self,
        *,
        student_kind: str = DEFAULT_STUDENT_KIND,
        alpha: float = DEFAULT_RIDGE_ALPHA,
        target_components: int = DEFAULT_TARGET_COMPONENTS,
        extra_trees_estimators: int = 160,
        registration_config: HeelRegistrationConfig = (
            DEFAULT_HEEL_REGISTRATION_CONFIG
        ),
        base_model: SparseAnchorStudentModel | None = None,
    ) -> None:
        if not isinstance(registration_config, HeelRegistrationConfig):
            raise TypeError("registration_config must be a HeelRegistrationConfig")
        if base_model is None:
            base_model = SparseAnchorStudentModel(
                student_kind=student_kind,
                alpha=alpha,
                target_components=target_components,
                extra_trees_estimators=extra_trees_estimators,
            )
        elif not isinstance(base_model, SparseAnchorStudentModel):
            raise TypeError("base_model must be a SparseAnchorStudentModel")
        self.base_model = base_model
        self.registration_config = registration_config

    def prepare(
        self, well: WellFiles, *, split: str, cache_root: Path
    ) -> HeelRegisteredAnchorInferenceWell:
        """Prepare the base payload and retain only defensive safe-column views."""

        horizontal = load_horizontal_inference(well.horizontal_path)
        typewell = load_typewell_inference(well.typewell_path)
        known_count, prediction_count = validate_prefix(horizontal["TVT_input"])
        base = self.base_model.prepare(well, split=split, cache_root=cache_root)
        if not isinstance(base, SparseAnchorInferenceWell):
            raise TypeError("base model prepare returned the wrong payload type")
        expected_rows = np.arange(
            known_count, known_count + prediction_count, dtype=np.int64
        )
        row_index = np.asarray(base.row_index)
        if (
            base.well_id != well.well_id
            or row_index.ndim != 1
            or row_index.shape != expected_rows.shape
            or not np.array_equal(row_index, expected_rows)
        ):
            raise ValueError(f"base prediction-row alignment mismatch: {well.well_id}")
        visible_anchor = float(horizontal["TVT_input"].iloc[known_count - 1])
        if not math.isfinite(float(base.anchor)) or not math.isclose(
            float(base.anchor), visible_anchor, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(f"base visible-anchor mismatch: {well.well_id}")
        if tuple(horizontal.columns) != HORIZONTAL_INFERENCE_COLUMNS:
            raise ValueError("horizontal inference view schema changed unexpectedly")
        if tuple(typewell.columns) != TYPEWELL_INFERENCE_COLUMNS:
            raise ValueError("typewell inference view schema changed unexpectedly")
        return HeelRegisteredAnchorInferenceWell(
            base=base,
            horizontal=horizontal,
            typewell=typewell,
        )

    def from_base_state(
        self, base_state: SparseAnchorStudentState
    ) -> HeelRegisteredAnchorStudentState:
        """Create a deployable wrapper state from a previously loaded base state."""

        return wrap_sparse_anchor_student_state(
            base_state, config=self.registration_config
        )

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> HeelRegisteredAnchorStudentState:
        """Fit only the base model on the supplied outer-training wells."""

        if not training_wells:
            raise ValueError("heel-registered fit requires training wells")
        if not all(
            isinstance(well.inference, HeelRegisteredAnchorInferenceWell)
            for well in training_wells
        ):
            raise TypeError("heel-registered fit received the wrong inference payload")
        base_training = [
            TrainingWell(well.inference.base, well.target) for well in training_wells
        ]
        base_state = self.base_model.fit(base_training, context=context)
        return self.from_base_state(base_state)

    def predict_components(
        self,
        state: HeelRegisteredAnchorStudentState,
        well: HeelRegisteredAnchorInferenceWell,
    ) -> HeelRegisteredAnchorPrediction:
        if not isinstance(state, HeelRegisteredAnchorStudentState):
            raise TypeError("state must be a HeelRegisteredAnchorStudentState")
        if not isinstance(well, HeelRegisteredAnchorInferenceWell):
            raise TypeError("well must be a HeelRegisteredAnchorInferenceWell")
        if (
            state.registration_version != HEEL_REGISTRATION_VERSION
            or state.base_model_id != self.base_model.model_id
        ):
            raise ValueError(
                "state model/version provenance does not match this wrapper"
            )
        base = self.base_model.predict_components(state.base_state, well.base)
        if not isinstance(base, SparseAnchorPrediction):
            raise TypeError("base model returned the wrong prediction payload type")
        base_final = np.asarray(base.final, dtype=np.float64)
        base_anchors = np.asarray(base.anchor_tvt, dtype=np.float64)
        if (
            base_final.ndim != 1
            or base_final.shape != well.row_index.shape
            or not np.isfinite(base_final).all()
            or base_anchors.ndim != 1
            or not np.isfinite(base_anchors).all()
        ):
            raise ValueError(f"invalid base prediction: {well.well_id}")

        registration = estimate_heel_registration(
            well.horizontal,
            well.typewell,
            base_final,
            config=state.registration_config,
        )
        shift = float(registration.posterior_mean_shift)
        if not math.isfinite(shift):
            raise ValueError(f"non-finite heel-registration shift: {well.well_id}")
        final = base_final + shift
        anchor_tvt = base_anchors + shift
        if (
            final.shape != well.row_index.shape
            or not np.isfinite(final).all()
            or anchor_tvt.shape != base_anchors.shape
            or not np.isfinite(anchor_tvt).all()
        ):
            raise ValueError(f"invalid heel-registered prediction: {well.well_id}")
        return HeelRegisteredAnchorPrediction(
            base=base,
            registration=registration,
            anchor_tvt=anchor_tvt,
            final=final,
        )

    def predict(
        self,
        state: HeelRegisteredAnchorStudentState,
        well: HeelRegisteredAnchorInferenceWell,
    ) -> np.ndarray:
        return self.predict_components(state, well).final

    def manifest(self) -> dict[str, Any]:
        """Return this instance's serializable deployment manifest."""

        return heel_registered_anchor_student_manifest(
            config=self.registration_config,
            student_kind=self.base_model.student_kind,
            alpha=self.base_model.alpha,
            target_components=self.base_model.target_components,
            extra_trees_estimators=self.base_model.extra_trees_estimators,
        )


__all__ = [
    "MODEL_ID",
    "HeelRegisteredAnchorInferenceWell",
    "HeelRegisteredAnchorPrediction",
    "HeelRegisteredAnchorStudentModel",
    "HeelRegisteredAnchorStudentState",
    "heel_registered_anchor_student_manifest",
    "heel_registered_inference_schema",
    "heel_registered_state_provenance",
    "wrap_sparse_anchor_student_state",
]
