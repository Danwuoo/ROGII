"""Deployable sparse-anchor student over the fixed P3/Path-PCA blend."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data import WellFiles, load_horizontal_inference, load_typewell_inference
from ..sparse_anchor import (
    ANCHOR_FRACTIONS,
    DESCRIPTOR_VERSION,
    AnchorPlacement,
    build_anchor_delta_targets,
    build_student_descriptor,
    reconstruct_from_predicted_anchors,
    select_anchor_rows,
)
from .base import InferenceWell, RunContext, TrainingWell
from .p3 import P3Model
from .path_pca import PathPCAModel


MODEL_ID = "sparse_anchor_student"
STUDENT_KINDS = (
    "zero",
    "mean",
    "ridge",
    "target_pca_ridge",
    "pls",
    "extra_trees",
)
DEFAULT_STUDENT_KIND = "extra_trees"
DEFAULT_RIDGE_ALPHA = 100.0
DEFAULT_TARGET_COMPONENTS = 3


@dataclass(frozen=True)
class SparseAnchorInferenceWell:
    well_id: str
    row_index: np.ndarray
    anchor: float
    features: np.ndarray
    descriptor_names: tuple[str, ...]
    p3: InferenceWell
    path_pca: InferenceWell
    placement: AnchorPlacement


@dataclass(frozen=True)
class StudentEstimatorState:
    kind: str
    estimator: Any
    target_pca: PCA | None
    training_well_ids: tuple[str, ...]
    descriptor_width: int
    target_components: int | None
    alpha: float | None
    random_seed: int


@dataclass(frozen=True)
class SparseAnchorStudentState:
    p3_state: Any
    path_pca_state: Any
    student_state: StudentEstimatorState
    training_well_ids: tuple[str, ...]
    fold: int


@dataclass(frozen=True)
class SparseAnchorPrediction:
    p3: np.ndarray
    path_pca: np.ndarray
    baseline: np.ndarray
    anchor_delta: np.ndarray
    anchor_tvt: np.ndarray
    final: np.ndarray


class TrainingRangeClipper(TransformerMixin, BaseEstimator):
    """Clip validation descriptors to per-feature outer-training support."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "TrainingRangeClipper":
        del y
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("TrainingRangeClipper fit requires a finite matrix")
        self.lower_ = np.min(values, axis=0)
        self.upper_ = np.max(values, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != self.n_features_in_
            or not np.isfinite(values).all()
        ):
            raise ValueError("TrainingRangeClipper transform shape/values are invalid")
        return np.clip(values, self.lower_, self.upper_)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _descriptor_cache_path(cache_root: Path, split: str, well_id: str) -> Path:
    return Path(cache_root) / DESCRIPTOR_VERSION / split / f"{well_id}.npz"


def _name_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()


def _load_descriptor_cache(
    path: Path,
    *,
    p3: InferenceWell,
) -> tuple[np.ndarray, tuple[str, ...], AnchorPlacement] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            values = cached["descriptor"].astype(np.float64, copy=True)
            names = tuple(str(value) for value in cached["descriptor_names"].tolist())
            local = cached["anchor_local_indices"].astype(np.int64, copy=True)
            rows = cached["anchor_row_indices"].astype(np.int64, copy=True)
            normalized = cached["normalized_positions"].astype(np.float64, copy=True)
            selected = cached["selected_positions"].astype(np.float64, copy=True)
            requested = cached["requested_positions"].astype(np.float64, copy=True)
            cached_row_index = cached["prediction_row_index"].astype(
                np.int64, copy=True
            )
            digest = str(cached["descriptor_name_digest"][0])
        valid = (
            values.ndim == 1
            and len(values) == len(names)
            and len(set(names)) == len(names)
            and not np.isinf(values).any()
            and local.shape == (len(ANCHOR_FRACTIONS),)
            and rows.shape == local.shape
            and normalized.shape == p3.row_index.shape
            and selected.shape == local.shape
            and requested.shape == local.shape
            and np.array_equal(cached_row_index, p3.row_index)
            and np.array_equal(rows, p3.row_index[local])
            and np.all(np.diff(local) > 0)
            and np.allclose(selected, normalized[local], rtol=0.0, atol=0.0)
            and np.allclose(
                requested, np.asarray(ANCHOR_FRACTIONS), rtol=0.0, atol=0.0
            )
            and digest == _name_digest(names)
        )
        if not valid:
            return None
        return values, names, AnchorPlacement(
            local_indices=local,
            row_indices=rows,
            normalized_positions=normalized,
            selected_positions=selected,
            requested_positions=requested,
        )
    except (OSError, ValueError, KeyError, EOFError):
        return None


def fit_student_estimator(
    training_wells: Sequence[TrainingWell],
    *,
    kind: str,
    context: RunContext,
    alpha: float = DEFAULT_RIDGE_ALPHA,
    target_components: int = DEFAULT_TARGET_COMPONENTS,
    extra_trees_estimators: int = 160,
) -> StudentEstimatorState:
    """Fit only the eight-output student from outer-training wells."""
    if kind not in STUDENT_KINDS:
        raise ValueError(f"unknown student kind {kind!r}")
    if not training_wells:
        raise ValueError("student fit requires at least one training well")
    inference = [well.inference for well in training_wells]
    if not all(isinstance(well, SparseAnchorInferenceWell) for well in inference):
        raise TypeError("student fit requires SparseAnchorInferenceWell payloads")
    ids = tuple(str(well.well_id) for well in inference)
    if len(set(ids)) != len(ids):
        raise ValueError("student training well IDs must be unique")
    names = inference[0].descriptor_names
    if any(well.descriptor_names != names for well in inference):
        raise ValueError("student descriptor schema differs across wells")
    x = np.stack([well.features for well in inference])
    y = np.stack(
        [
            build_anchor_delta_targets(
                training.target,
                last_known_tvt=training.inference.anchor,
                placement=training.inference.placement,
            )
            for training in training_wells
        ]
    )
    if x.ndim != 2 or y.shape != (len(x), len(ANCHOR_FRACTIONS)):
        raise ValueError("student design or target shape is invalid")
    if np.isinf(x).any() or not np.isfinite(y).all():
        raise ValueError("student design may contain NaN but target must be finite")

    estimator: Any = None
    target_pca: PCA | None = None
    fitted_components: int | None = None
    fitted_alpha: float | None = None
    if kind == "mean":
        estimator = np.mean(y, axis=0)
    elif kind in {"ridge", "target_pca_ridge"}:
        response = y
        if kind == "target_pca_ridge":
            fitted_components = min(int(target_components), len(y), y.shape[1])
            if fitted_components < 1:
                raise ValueError("target PCA requires at least one component")
            target_pca = PCA(n_components=fitted_components, svd_solver="full")
            response = target_pca.fit_transform(y)
        fitted_alpha = float(alpha)
        estimator = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("clip", TrainingRangeClipper()),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=fitted_alpha)),
            ]
        )
        estimator.fit(x, response)
    elif kind == "pls":
        fitted_components = min(int(target_components), len(y) - 1, y.shape[1])
        if fitted_components < 1:
            raise ValueError("PLS requires at least two training wells")
        estimator = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("clip", TrainingRangeClipper()),
                ("scale", StandardScaler()),
                (
                    "pls",
                    PLSRegression(
                        n_components=fitted_components,
                        scale=False,
                        max_iter=500,
                        tol=1e-06,
                    ),
                ),
            ]
        )
        estimator.fit(x, y)
    elif kind == "extra_trees":
        estimator = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("clip", TrainingRangeClipper()),
                (
                    "trees",
                    ExtraTreesRegressor(
                        n_estimators=int(extra_trees_estimators),
                        max_depth=8,
                        min_samples_leaf=5,
                        max_features=0.5,
                        n_jobs=max(1, int(context.workers)),
                        random_state=int(context.seed),
                    ),
                ),
            ]
        )
        estimator.fit(x, y)
    # ``zero`` deliberately has no learned state.  At reconstruction time it
    # is the exact zero-correction regression control, so the baseline is
    # returned bit-for-bit instead of interpreting zero delta as eight flat
    # TVT observations.
    return StudentEstimatorState(
        kind=kind,
        estimator=estimator,
        target_pca=target_pca,
        training_well_ids=ids,
        descriptor_width=x.shape[1],
        target_components=fitted_components,
        alpha=fitted_alpha,
        random_seed=int(context.seed),
    )


def predict_anchor_delta(
    state: StudentEstimatorState, well: SparseAnchorInferenceWell
) -> np.ndarray:
    if len(well.features) != state.descriptor_width:
        raise ValueError("student descriptor width mismatch")
    if state.kind == "zero":
        return np.zeros(len(ANCHOR_FRACTIONS), dtype=np.float64)
    if state.kind == "mean":
        result = np.asarray(state.estimator, dtype=np.float64)
    else:
        result = np.asarray(state.estimator.predict(well.features[None, :]))
        result = result.reshape(1, -1)[0].astype(np.float64, copy=False)
        if state.target_pca is not None:
            result = np.asarray(
                state.target_pca.inverse_transform(result[None, :])[0],
                dtype=np.float64,
            )
    if result.shape != (len(ANCHOR_FRACTIONS),) or not np.isfinite(result).all():
        raise ValueError(f"student produced invalid anchors for {well.well_id}")
    return result


class SparseAnchorStudentModel:
    """P3/Path-PCA blend corrected by eight inference-safe predicted anchors."""

    model_id = MODEL_ID

    def __init__(
        self,
        *,
        student_kind: str = DEFAULT_STUDENT_KIND,
        alpha: float = DEFAULT_RIDGE_ALPHA,
        target_components: int = DEFAULT_TARGET_COMPONENTS,
        extra_trees_estimators: int = 160,
    ) -> None:
        if student_kind not in STUDENT_KINDS:
            raise ValueError(f"unknown student kind {student_kind!r}")
        self.student_kind = student_kind
        self.alpha = float(alpha)
        self.target_components = int(target_components)
        self.extra_trees_estimators = int(extra_trees_estimators)
        self.p3_model = P3Model()
        self.path_pca_model = PathPCAModel()

    def cache_path(self, well: WellFiles, *, split: str, cache_root: Path) -> Path:
        return _descriptor_cache_path(cache_root, split, well.well_id)

    def prepare(
        self, well: WellFiles, *, split: str, cache_root: Path
    ) -> SparseAnchorInferenceWell:
        p3 = self.p3_model.prepare(well, split=split, cache_root=cache_root)
        path = self.path_pca_model.prepare(well, split=split, cache_root=cache_root)
        if not np.array_equal(p3.row_index, path.row_index) or not np.isclose(
            p3.anchor, path.anchor, rtol=0.0, atol=0.0
        ):
            raise ValueError(f"baseline payload mismatch: {well.well_id}")

        cache_path = _descriptor_cache_path(cache_root, split, well.well_id)
        cached = _load_descriptor_cache(cache_path, p3=p3)
        if cached is None:
            horizontal = load_horizontal_inference(well.horizontal_path)
            typewell = load_typewell_inference(well.typewell_path)
            known_count = int(path.row_index[0])
            md = path.features[:, 0]
            placement = select_anchor_rows(
                md[known_count:],
                float(md[known_count - 1]),
                row_indices=path.row_index,
            )
            descriptor = build_student_descriptor(
                horizontal,
                typewell,
                p3_features=p3.features,
                prediction_row_index=p3.row_index,
                placement=placement,
            )
            _atomic_npz(
                cache_path,
                descriptor=descriptor.values,
                descriptor_names=np.asarray(descriptor.names, dtype="U"),
                descriptor_name_digest=np.asarray([_name_digest(descriptor.names)], dtype="U"),
                anchor_local_indices=placement.local_indices,
                anchor_row_indices=placement.row_indices,
                normalized_positions=placement.normalized_positions,
                selected_positions=placement.selected_positions,
                requested_positions=placement.requested_positions,
                prediction_row_index=p3.row_index,
            )
            values, names = descriptor.values, descriptor.names
        else:
            values, names, placement = cached
        return SparseAnchorInferenceWell(
            well_id=well.well_id,
            row_index=p3.row_index,
            anchor=p3.anchor,
            features=values,
            descriptor_names=names,
            p3=p3,
            path_pca=path,
            placement=placement,
        )

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> SparseAnchorStudentState:
        if not training_wells:
            raise ValueError("sparse-anchor fit requires training wells")
        if not all(
            isinstance(well.inference, SparseAnchorInferenceWell)
            for well in training_wells
        ):
            raise TypeError("sparse-anchor fit received the wrong inference payload")
        ids = tuple(str(well.inference.well_id) for well in training_wells)
        p3_training = [
            TrainingWell(well.inference.p3, well.target) for well in training_wells
        ]
        path_training = [
            TrainingWell(well.inference.path_pca, well.target)
            for well in training_wells
        ]
        p3_state = self.p3_model.fit(p3_training, context=context)
        path_state = self.path_pca_model.fit(path_training, context=context)
        student_state = fit_student_estimator(
            training_wells,
            kind=self.student_kind,
            context=context,
            alpha=self.alpha,
            target_components=self.target_components,
            extra_trees_estimators=self.extra_trees_estimators,
        )
        return SparseAnchorStudentState(
            p3_state=p3_state,
            path_pca_state=path_state,
            student_state=student_state,
            training_well_ids=ids,
            fold=int(context.fold),
        )

    def predict_components(
        self, state: SparseAnchorStudentState, well: SparseAnchorInferenceWell
    ) -> SparseAnchorPrediction:
        p3 = np.asarray(self.p3_model.predict(state.p3_state, well.p3), dtype=np.float64)
        path = np.asarray(
            self.path_pca_model.predict(state.path_pca_state, well.path_pca),
            dtype=np.float64,
        )
        if p3.shape != path.shape or p3.shape != well.row_index.shape:
            raise ValueError(f"baseline prediction shape mismatch: {well.well_id}")
        baseline = 0.5 * p3 + 0.5 * path
        delta = predict_anchor_delta(state.student_state, well)
        if state.student_state.kind == "zero":
            anchor_tvt = baseline[well.placement.local_indices].copy()
            final = baseline.copy()
        else:
            anchor_tvt = well.anchor + delta
            final = reconstruct_from_predicted_anchors(
                baseline, anchor_tvt, well.placement
            )
        if final.shape != well.row_index.shape or not np.isfinite(final).all():
            raise ValueError(f"invalid sparse-anchor prediction: {well.well_id}")
        return SparseAnchorPrediction(p3, path, baseline, delta, anchor_tvt, final)

    def predict(
        self, state: SparseAnchorStudentState, well: SparseAnchorInferenceWell
    ) -> np.ndarray:
        return self.predict_components(state, well).final
