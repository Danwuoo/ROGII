"""Nested-CV variant of the whole-path PCA forecaster.

Every outer-fold fit selects the number of predicted path components and the
Ridge penalty using only that outer fold's training wells. Inner validation is
also split by well, and candidates are ranked by pooled prediction-row RMSE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import InferenceWell, RunContext, TrainingWell
from .path_pca import (
    PATH_POINTS,
    PCA_FIT_COMPONENTS,
    PathPCAModel,
    _MD,
    _prefix_features,
    _strict_coordinate,
    _target_curve,
)


MODEL_ID = "path_pca_nested"
INNER_FOLDS = 3
COMPONENT_GRID = (2, 5, 8)
ALPHA_GRID = (10.0, 100.0, 1_000.0)


@dataclass(frozen=True)
class NestedPathPCAState:
    pca: PCA
    regressor: Pipeline
    predicted_components: int
    selected_alpha: float
    inner_pooled_rmse: float
    inner_candidate_rmse: dict[str, float]


def _make_regressor(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _residual_at_prediction_rows(
    inference: InferenceWell, curve: np.ndarray
) -> np.ndarray:
    known_count = int(inference.row_index[0])
    md = _strict_coordinate(inference.features[:, _MD], name="MD")
    anchor_md = float(md[known_count - 1])
    horizon = float(md[-1] - anchor_md)
    coordinate = (md[inference.row_index] - anchor_md) / horizon
    return np.interp(
        coordinate,
        np.linspace(0.0, 1.0, PATH_POINTS),
        np.asarray(curve, dtype=np.float64),
    )


class NestedPathPCAModel(PathPCAModel):
    """Whole-suffix forecaster with outer-safe inner model selection."""

    model_id = MODEL_ID

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> NestedPathPCAState:
        if len(training_wells) < 6:
            raise ValueError("Nested Path-PCA fit requires at least six training wells")

        ordered = sorted(training_wells, key=lambda well: well.inference.well_id)
        design = np.stack(
            [
                _prefix_features(
                    well.inference.features, int(well.inference.row_index[0])
                )
                for well in ordered
            ]
        )
        paths = np.stack(
            [
                _target_curve(
                    well.inference.features,
                    int(well.inference.row_index[0]),
                    well.target,
                )
                for well in ordered
            ]
        )

        minimum_inner_training_wells = len(ordered) - math.ceil(
            len(ordered) / INNER_FOLDS
        )
        component_grid = tuple(
            components
            for components in COMPONENT_GRID
            if components <= minimum_inner_training_wells
        )
        if not component_grid:
            raise ValueError("No Path-PCA component candidate fits the inner folds")
        candidate_sse = {
            (components, alpha): 0.0
            for components in component_grid
            for alpha in ALPHA_GRID
        }
        candidate_rows = {key: 0 for key in candidate_sse}
        splitter = KFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=context.seed + 10_007 * context.fold,
        )
        all_indices = np.arange(len(ordered), dtype=np.int64)
        for inner_train, inner_validation in splitter.split(all_indices):
            fit_components = min(
                PCA_FIT_COMPONENTS, len(inner_train), paths.shape[1]
            )
            pca = PCA(n_components=fit_components, svd_solver="full")
            coefficients = pca.fit_transform(paths[inner_train])
            max_components = min(max(component_grid), fit_components)
            for alpha in ALPHA_GRID:
                regressor = _make_regressor(alpha)
                regressor.fit(
                    design[inner_train], coefficients[:, :max_components]
                )
                predicted_coefficients = np.asarray(
                    regressor.predict(design[inner_validation]),
                    dtype=np.float64,
                )
                for components in component_grid:
                    used_components = min(components, max_components)
                    curves = (
                        pca.mean_
                        + predicted_coefficients[:, :used_components]
                        @ pca.components_[:used_components]
                    )
                    key = (components, alpha)
                    for local_index, well_index in enumerate(inner_validation):
                        well = ordered[int(well_index)]
                        residual = _residual_at_prediction_rows(
                            well.inference, curves[local_index]
                        )
                        prediction = well.inference.anchor + residual
                        error = well.target - prediction
                        candidate_sse[key] += float(np.dot(error, error))
                        candidate_rows[key] += len(error)

        candidate_rmse = {
            key: float(np.sqrt(candidate_sse[key] / candidate_rows[key]))
            for key in candidate_sse
        }
        selected_components, selected_alpha = min(
            candidate_rmse,
            key=lambda key: (candidate_rmse[key], key[0], key[1]),
        )

        fit_components = min(PCA_FIT_COMPONENTS, len(paths), paths.shape[1])
        pca = PCA(n_components=fit_components, svd_solver="full")
        coefficients = pca.fit_transform(paths)
        selected_components = min(selected_components, fit_components)
        regressor = _make_regressor(selected_alpha)
        regressor.fit(design, coefficients[:, :selected_components])
        serialized_rmse = {
            f"components={components},alpha={alpha:g}": candidate_rmse[
                (components, alpha)
            ]
            for components in component_grid
            for alpha in ALPHA_GRID
        }
        return NestedPathPCAState(
            pca=pca,
            regressor=regressor,
            predicted_components=selected_components,
            selected_alpha=selected_alpha,
            inner_pooled_rmse=candidate_rmse[
                (selected_components, selected_alpha)
            ],
            inner_candidate_rmse=serialized_rmse,
        )

    def predict(
        self, state: NestedPathPCAState, well: InferenceWell
    ) -> np.ndarray:
        return super().predict(state, well)  # type: ignore[arg-type]
