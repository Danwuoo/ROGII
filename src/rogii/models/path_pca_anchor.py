"""Anchor-inclusive whole-path PCA forecast.

This ablation is identical to ``path_pca`` except that every resampled training
path explicitly starts at residual zero at the final visible TVT anchor.  It
tests whether endpoint representation bias in the original resampling matters.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import RunContext, TrainingWell
from .path_pca import (
    PATH_POINTS,
    PCA_FIT_COMPONENTS,
    PCA_PREDICT_COMPONENTS,
    RIDGE_ALPHA,
    PathPCAModel,
    PathPCAState,
    _MD,
    _TVT_INPUT,
    _prefix_features,
    _strict_coordinate,
)


MODEL_ID = "path_pca_anchor"


def _anchor_inclusive_target_curve(well: TrainingWell) -> np.ndarray:
    values = np.asarray(well.inference.features, dtype=np.float64)
    target = np.asarray(well.target, dtype=np.float64)
    known_count = int(well.inference.row_index[0])
    if target.shape != (len(values) - known_count,) or not np.isfinite(target).all():
        raise ValueError("Anchor Path-PCA training target does not cover the suffix")
    md = _strict_coordinate(values[:, _MD], name="MD")
    anchor_md = float(md[known_count - 1])
    horizon = float(md[-1] - anchor_md)
    if horizon <= 0:
        raise ValueError("Anchor Path-PCA prediction horizon must be positive")
    suffix_coordinate = (md[known_count:] - anchor_md) / horizon
    coordinate = np.concatenate(([0.0], suffix_coordinate))
    anchor = float(values[known_count - 1, _TVT_INPUT])
    residual = np.concatenate(([0.0], target - anchor))
    return np.interp(
        np.linspace(0.0, 1.0, PATH_POINTS),
        coordinate,
        residual,
    )


class AnchorPathPCAModel(PathPCAModel):
    """Fixed-hyperparameter path forecast with an explicit zero-residual anchor."""

    model_id = MODEL_ID

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> PathPCAState:
        del context
        if len(training_wells) < 2:
            raise ValueError("Anchor Path-PCA fit requires at least two training wells")
        design = np.stack(
            [
                _prefix_features(
                    well.inference.features, int(well.inference.row_index[0])
                )
                for well in training_wells
            ]
        )
        paths = np.stack(
            [_anchor_inclusive_target_curve(well) for well in training_wells]
        )
        fit_components = min(PCA_FIT_COMPONENTS, len(paths), paths.shape[1])
        pca = PCA(n_components=fit_components, svd_solver="full")
        coefficients = pca.fit_transform(paths)
        # The anchor coordinate has identically zero variance. Numerical
        # null-space components are arbitrary, so enforce the known constraint.
        pca.mean_[0] = 0.0
        pca.components_[:, 0] = 0.0
        predicted_components = min(PCA_PREDICT_COMPONENTS, fit_components)
        regressor = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=RIDGE_ALPHA)),
            ]
        )
        regressor.fit(design, coefficients[:, :predicted_components])
        return PathPCAState(
            pca=pca,
            regressor=regressor,
            predicted_components=predicted_components,
        )
