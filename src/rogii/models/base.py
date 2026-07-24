"""Minimal model interface for adding new inference-safe experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from ..data import WellFiles


@dataclass(frozen=True)
class InferenceWell:
    well_id: str
    row_index: np.ndarray
    anchor: float
    features: np.ndarray


@dataclass(frozen=True)
class TrainingWell:
    inference: InferenceWell
    target: np.ndarray


@dataclass(frozen=True)
class RunContext:
    fold: int
    seed: int
    workers: int
    work_dir: Path


@runtime_checkable
class Model(Protocol):
    model_id: str

    def prepare(
        self, well: WellFiles, *, split: str, cache_root: Path
    ) -> InferenceWell: ...

    def fit(
        self, training_wells: Sequence[TrainingWell], *, context: RunContext
    ) -> Any: ...

    def predict(self, state: Any, well: InferenceWell) -> np.ndarray: ...
