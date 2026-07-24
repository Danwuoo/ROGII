"""P3 physical-TVT alignment residual model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..alignment import (
    ALIGNMENT_FEATURE_COLUMNS,
    ALIGNMENT_VERSION,
    build_alignment_features,
)
from ..data import WellFiles, load_horizontal_inference, load_typewell_inference
from ..features import FEATURE_COLUMNS, FEATURE_VERSION, build_prediction_features
from .base import InferenceWell, RunContext, TrainingWell


MODEL_SEED = 20260711
MODEL_PARAMETERS: dict[str, Any] = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 80,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "max_bins": 63,
    "early_stopping": False,
    "random_state": MODEL_SEED,
    "verbose": 0,
}
CACHE_VERSION = f"{FEATURE_VERSION}+{ALIGNMENT_VERSION}"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


class P3Model:
    model_id = "p3"

    def prepare(
        self, well: WellFiles, *, split: str, cache_root: Path
    ) -> InferenceWell:
        output = Path(cache_root) / CACHE_VERSION / split / f"{well.well_id}.npz"
        if not output.is_file():
            horizontal = load_horizontal_inference(well.horizontal_path)
            typewell = load_typewell_inference(well.typewell_path)
            base = build_prediction_features(horizontal, typewell)
            alignment = build_alignment_features(horizontal, typewell)
            if not np.array_equal(base.row_index, alignment.row_index):
                raise ValueError(f"P3 row-index mismatch: {well.well_id}")
            if not np.isclose(base.anchor, alignment.anchor, rtol=0.0, atol=0.0):
                raise ValueError(f"P3 anchor mismatch: {well.well_id}")
            features = np.concatenate(
                [
                    base.features.to_numpy(dtype=np.float32, copy=False),
                    alignment.features.to_numpy(dtype=np.float32, copy=False),
                ],
                axis=1,
            )
            expected_width = len(FEATURE_COLUMNS) + len(ALIGNMENT_FEATURE_COLUMNS)
            if features.shape != (len(base.row_index), expected_width):
                raise ValueError(f"P3 feature shape mismatch: {well.well_id}")
            _atomic_npz(
                output,
                features=features,
                row_index=base.row_index.astype(np.int64, copy=False),
                anchor=np.asarray([base.anchor], dtype=np.float64),
            )
        with np.load(output, allow_pickle=False) as cached:
            return InferenceWell(
                well_id=well.well_id,
                row_index=cached["row_index"].copy(),
                anchor=float(cached["anchor"][0]),
                features=cached["features"].copy(),
            )

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> HistGradientBoostingRegressor:
        if not training_wells:
            raise ValueError("P3 fit requires at least one training well")
        features = np.concatenate(
            [well.inference.features for well in training_wells], axis=0
        )
        target = np.concatenate(
            [well.target - well.inference.anchor for well in training_wells],
            axis=0,
        )
        parameters = {**MODEL_PARAMETERS, "random_state": context.seed}
        model = HistGradientBoostingRegressor(**parameters)
        model.fit(features, target)
        return model

    def predict(
        self, state: HistGradientBoostingRegressor, well: InferenceWell
    ) -> np.ndarray:
        residual = np.asarray(state.predict(well.features), dtype=np.float64)
        if residual.shape != (len(well.row_index),) or not np.isfinite(residual).all():
            raise ValueError(f"P3 produced invalid predictions for {well.well_id}")
        return residual + well.anchor
