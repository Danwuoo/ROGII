"""Fold-safe whole-path forecasting with a low-rank residual representation.

The model treats the hidden suffix as one structured object instead of a set
of independent rows.  It learns a PCA codebook of anchor-relative TVT paths
from outer-training wells, then predicts a small number of path coefficients
from the validation well's observed prefix and trajectory context.

No target-bearing statistic is computed in ``prepare``.  PCA, scaling, and
regression are all fitted inside ``fit`` from the current outer fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data import WellFiles, load_horizontal_inference, validate_prefix
from .base import InferenceWell, RunContext, TrainingWell


MODEL_ID = "path_pca"
PREFIX_POINTS = 64
PATH_POINTS = 128
PREFIX_MD_WINDOW_FT = 1_000.0
PCA_FIT_COMPONENTS = 20
PCA_PREDICT_COMPONENTS = 5
RIDGE_ALPHA = 100.0

_MD = 0
_X = 1
_Y = 2
_Z = 3
_GR = 4
_TVT_INPUT = 5


@dataclass(frozen=True)
class PathPCAState:
    pca: PCA
    regressor: Pipeline
    predicted_components: int


def _strict_coordinate(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional coordinate")
    if np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def _prefix_features(features: np.ndarray, known_count: int) -> np.ndarray:
    """Encode only information available at the requested prediction start."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Path-PCA expects six inference-safe horizontal columns")
    if known_count < 2 or known_count >= len(values):
        raise ValueError("Path-PCA requires a non-empty prefix and suffix")

    md = _strict_coordinate(values[:, _MD], name="MD")
    anchor_md = float(md[known_count - 1])
    prefix_start_md = max(float(md[0]), anchor_md - PREFIX_MD_WINDOW_FT)
    prefix_rows = np.flatnonzero(
        (np.arange(len(values)) < known_count) & (md >= prefix_start_md)
    )
    if len(prefix_rows) < 2:
        prefix_rows = np.arange(max(0, known_count - 2), known_count)
    span = max(anchor_md - prefix_start_md, 1.0)
    coordinate = (md[prefix_rows] - anchor_md) / span
    query = np.linspace(-1.0, 0.0, PREFIX_POINTS)

    def sample(column: int) -> np.ndarray:
        source = values[prefix_rows, column]
        if not np.isfinite(source).all():
            raise ValueError("Path-PCA prefix geometry and TVT must be finite")
        return np.interp(query, coordinate, source)

    prefix_tvt = sample(_TVT_INPUT)
    anchor = float(prefix_tvt[-1])
    prefix_z = sample(_Z)
    prefix_x = sample(_X)
    prefix_y = sample(_Y)
    gr = values[:known_count, _GR]
    finite_gr = gr[np.isfinite(gr)]
    if len(finite_gr):
        gr_mean = float(np.mean(finite_gr))
        gr_std = float(np.std(finite_gr))
    else:
        gr_mean = 0.0
        gr_std = 0.0

    return np.concatenate(
        [
            prefix_tvt - anchor,
            prefix_z - prefix_z[-1],
            (prefix_x - prefix_x[-1]) / 1_000.0,
            (prefix_y - prefix_y[-1]) / 1_000.0,
            np.asarray(
                [
                    gr_mean,
                    gr_std,
                    float(np.mean(~np.isfinite(gr))),
                    float(md[-1] - anchor_md) / 1_000.0,
                    float(values[-1, _Z] - values[known_count - 1, _Z])
                    / 100.0,
                ],
                dtype=np.float64,
            ),
        ]
    )


def _target_curve(
    features: np.ndarray, known_count: int, target: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    if labels.shape != (len(values) - known_count,) or not np.isfinite(labels).all():
        raise ValueError("Path-PCA training target does not cover the suffix")
    md = _strict_coordinate(values[:, _MD], name="MD")
    anchor_md = float(md[known_count - 1])
    horizon = float(md[-1] - anchor_md)
    if horizon <= 0:
        raise ValueError("Path-PCA prediction horizon must be positive")
    coordinate = (md[known_count:] - anchor_md) / horizon
    query = np.linspace(0.0, 1.0, PATH_POINTS)
    anchor = float(values[known_count - 1, _TVT_INPUT])
    return np.interp(query, coordinate, labels - anchor)


class PathPCAModel:
    """Low-rank, whole-suffix path forecaster."""

    model_id = MODEL_ID

    def prepare(
        self, well: WellFiles, *, split: str, cache_root: Path
    ) -> InferenceWell:
        del split, cache_root
        horizontal = load_horizontal_inference(well.horizontal_path)
        known_count, _ = validate_prefix(horizontal["TVT_input"])
        row_index = np.arange(known_count, len(horizontal), dtype=np.int64)
        anchor = float(horizontal["TVT_input"].iloc[known_count - 1])
        features = horizontal.loc[
            :, ["MD", "X", "Y", "Z", "GR", "TVT_input"]
        ].to_numpy(dtype=np.float64, copy=True)
        return InferenceWell(
            well_id=well.well_id,
            row_index=row_index,
            anchor=anchor,
            features=features,
        )

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> PathPCAState:
        del context
        if len(training_wells) < 2:
            raise ValueError("Path-PCA fit requires at least two training wells")
        design: list[np.ndarray] = []
        paths: list[np.ndarray] = []
        for well in training_wells:
            known_count = int(well.inference.row_index[0])
            design.append(_prefix_features(well.inference.features, known_count))
            paths.append(
                _target_curve(
                    well.inference.features,
                    known_count,
                    well.target,
                )
            )
        x = np.stack(design)
        y = np.stack(paths)
        fit_components = min(PCA_FIT_COMPONENTS, len(y), y.shape[1])
        pca = PCA(n_components=fit_components, svd_solver="full")
        coefficients = pca.fit_transform(y)
        predicted_components = min(PCA_PREDICT_COMPONENTS, fit_components)
        regressor = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=RIDGE_ALPHA)),
            ]
        )
        regressor.fit(x, coefficients[:, :predicted_components])
        return PathPCAState(
            pca=pca,
            regressor=regressor,
            predicted_components=predicted_components,
        )

    def predict(self, state: PathPCAState, well: InferenceWell) -> np.ndarray:
        known_count = int(well.row_index[0])
        design = _prefix_features(well.features, known_count)
        coefficients = np.asarray(
            state.regressor.predict(design[None, :])[0], dtype=np.float64
        )
        curve = (
            state.pca.mean_
            + coefficients
            @ state.pca.components_[: state.predicted_components]
        )
        md = _strict_coordinate(well.features[:, _MD], name="MD")
        anchor_md = float(md[known_count - 1])
        horizon = float(md[-1] - anchor_md)
        coordinate = (md[well.row_index] - anchor_md) / horizon
        residual = np.interp(
            coordinate,
            np.linspace(0.0, 1.0, PATH_POINTS),
            curve,
        )
        prediction = well.anchor + residual
        if prediction.shape != (len(well.row_index),) or not np.isfinite(
            prediction
        ).all():
            raise ValueError(f"Path-PCA produced invalid predictions for {well.well_id}")
        return prediction.astype(np.float64, copy=False)

