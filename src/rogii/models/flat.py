"""Flat last-known baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data import WellFiles, load_horizontal_inference, validate_prefix
from .base import InferenceWell, RunContext, TrainingWell


class FlatModel:
    model_id = "flat"

    def prepare(
        self, well: WellFiles, *, split: str, cache_root: Path
    ) -> InferenceWell:
        del split, cache_root
        horizontal = load_horizontal_inference(well.horizontal_path)
        known_rows, _ = validate_prefix(horizontal["TVT_input"])
        row_index = np.arange(known_rows, len(horizontal), dtype=np.int64)
        anchor = float(horizontal["TVT_input"].iloc[known_rows - 1])
        return InferenceWell(
            well_id=well.well_id,
            row_index=row_index,
            anchor=anchor,
            features=np.empty((len(row_index), 0), dtype=np.float32),
        )

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> None:
        del context
        if not training_wells:
            raise ValueError("Flat fit requires at least one training well")
        return None

    def predict(self, state: Any, well: InferenceWell) -> np.ndarray:
        del state
        return np.full(len(well.row_index), well.anchor, dtype=np.float64)
