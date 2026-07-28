"""Fold-safe access to the immutable heel-registered incumbent states.

The portfolio models are residual models.  Their training residual must be
formed against an incumbent that has not seen the outer validation fold.
This module therefore never substitutes the convenient global OOF column for
outer-training predictions: it loads the exact state for the requested outer
fold and applies that state to both its training and validation wells.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..data import enumerate_wells
from ..folds import FOLD_COUNT, load_fold_mapping, partition_well_ids
from ..models.heel_registered_anchor_student import (
    HeelRegisteredAnchorInferenceWell,
    HeelRegisteredAnchorStudentModel,
    HeelRegisteredAnchorStudentState,
    heel_registered_state_provenance,
)


DEFAULT_INCUMBENT_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)


@dataclass(frozen=True)
class IncumbentPrediction:
    """A finite, row-aligned ensemble prediction with explicit provenance."""

    well_id: str
    row_index: np.ndarray
    prediction: np.ndarray
    fold_states: tuple[int, ...]
    seeds: tuple[int, ...]
    state_sha256: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HeelIncumbentProvider:
    """Load and apply the audited 5-seed heel-registered fold states.

    Parameters are deliberately paths rather than implicit globals so that a
    run manifest can reproduce exactly which immutable artifacts were used.
    """

    def __init__(
        self,
        *,
        data_root: Path,
        fold_config: Path,
        artifact_root: Path,
        cache_root: Path,
        seeds: Sequence[int] = DEFAULT_INCUMBENT_SEEDS,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.fold_config = Path(fold_config).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.seeds = tuple(map(int, seeds))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("incumbent seeds must be non-empty and unique")

        self.model = HeelRegisteredAnchorStudentModel()
        self.train_files = {
            item.well_id: item for item in enumerate_wells(self.data_root, "train")
        }
        self.test_files = {
            item.well_id: item for item in enumerate_wells(self.data_root, "test")
        }
        self.mapping = load_fold_mapping(
            self.fold_config, expected_well_ids=self.train_files
        )
        self.assigned_fold = (
            self.mapping.set_index("well_id")["fold"].astype(int).to_dict()
        )
        self._states: dict[tuple[int, int], HeelRegisteredAnchorStudentState] = {}
        self._state_hashes: dict[tuple[int, int], str] = {}
        self._prepared: dict[tuple[str, str], HeelRegisteredAnchorInferenceWell] = {}

    def _state_path(self, *, seed: int, fold: int) -> Path:
        return self.artifact_root / "models" / f"seed_{seed}" / f"fold_{fold}.pkl"

    def load_state(self, *, seed: int, fold: int) -> HeelRegisteredAnchorStudentState:
        """Load one state and reject any fold/seed/training-ID mismatch."""

        seed = int(seed)
        fold = int(fold)
        if seed not in self.seeds:
            raise ValueError(
                f"seed {seed} is outside the configured incumbent ensemble"
            )
        if fold not in range(FOLD_COUNT):
            raise ValueError("fold must be in 0..4")
        key = (seed, fold)
        if key in self._states:
            return self._states[key]
        path = self._state_path(seed=seed, fold=fold)
        if not path.is_file():
            raise FileNotFoundError(f"incumbent state is missing: {path}")
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, HeelRegisteredAnchorStudentState):
            raise TypeError(f"incumbent state has the wrong type: {path}")
        expected_training, validation = partition_well_ids(self.mapping, fold)
        provenance = heel_registered_state_provenance(state)
        actual_training = tuple(map(str, state.training_well_ids))
        invalid = (
            int(state.fold) != fold
            or int(state.random_seed) != seed
            or actual_training != expected_training
            or set(actual_training) & set(validation)
            or provenance["registration_uses_cross_well_learned_statistics"]
        )
        if invalid:
            raise ValueError(f"incumbent state provenance mismatch: {path}")
        self._states[key] = state
        self._state_hashes[key] = _sha256(path)
        return state

    def load_all_states(self) -> None:
        for fold in range(FOLD_COUNT):
            for seed in self.seeds:
                self.load_state(seed=seed, fold=fold)

    def prepare(
        self, well_id: str, *, split: str = "train"
    ) -> HeelRegisteredAnchorInferenceWell:
        """Prepare an inference-only payload and retain it in memory."""

        split = str(split)
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        well_id = str(well_id)
        files = self.train_files if split == "train" else self.test_files
        if well_id not in files:
            raise KeyError(f"unknown {split} well: {well_id}")
        key = (split, well_id)
        if key not in self._prepared:
            prepared = self.model.prepare(
                files[well_id], split=split, cache_root=self.cache_root
            )
            if prepared.well_id != well_id:
                raise ValueError("prepared incumbent payload changed the well ID")
            self._prepared[key] = prepared
        return self._prepared[key]

    def predict_with_folds(
        self,
        well_id: str,
        *,
        folds: Iterable[int],
        split: str = "train",
    ) -> IncumbentPrediction:
        """Average seeds within every requested fold, then average folds.

        Outer OOF calls pass exactly one fold.  Visible-test smoke calls pass all
        five folds.  This explicit distinction prevents accidentally using the
        25-state test ensemble as an OOF predictor.
        """

        fold_tuple = tuple(map(int, folds))
        if (
            not fold_tuple
            or len(set(fold_tuple)) != len(fold_tuple)
            or any(fold not in range(FOLD_COUNT) for fold in fold_tuple)
        ):
            raise ValueError("folds must be a non-empty unique subset of 0..4")
        prepared = self.prepare(well_id, split=split)
        predictions: list[np.ndarray] = []
        hashes: list[str] = []
        for fold in fold_tuple:
            seed_predictions: list[np.ndarray] = []
            for seed in self.seeds:
                state = self.load_state(seed=seed, fold=fold)
                values = np.asarray(
                    self.model.predict(state, prepared), dtype=np.float64
                )
                if (
                    values.shape != prepared.row_index.shape
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        f"invalid incumbent prediction: {well_id}/{seed}/{fold}"
                    )
                seed_predictions.append(values)
                hashes.append(self._state_hashes[(seed, fold)])
            predictions.append(np.mean(np.stack(seed_predictions), axis=0))
        prediction = np.mean(np.stack(predictions), axis=0)
        if (
            prediction.shape != prepared.row_index.shape
            or not np.isfinite(prediction).all()
        ):
            raise ValueError(f"invalid incumbent ensemble prediction: {well_id}")
        return IncumbentPrediction(
            well_id=str(well_id),
            row_index=np.asarray(prepared.row_index, dtype=np.int64).copy(),
            prediction=prediction,
            fold_states=fold_tuple,
            seeds=self.seeds,
            state_sha256=tuple(hashes),
        )

    def predict_outer_fold(
        self, well_id: str, *, outer_fold: int
    ) -> IncumbentPrediction:
        """Apply only the state trained without ``outer_fold``."""

        return self.predict_with_folds(well_id, folds=(int(outer_fold),), split="train")

    def predict_visible_test(self, well_id: str) -> IncumbentPrediction:
        """Apply the audited 25-state ensemble without creating a submission."""

        return self.predict_with_folds(well_id, folds=range(FOLD_COUNT), split="test")

    def verify_oof_match(
        self,
        frame: pd.DataFrame,
        *,
        prediction_column: str = "y_pred_heel_registered_ensemble",
        atol: float = 1e-9,
    ) -> dict[str, float | int]:
        """Recompute the selected validation wells from states and match disk OOF."""

        required = {"well_id", "row_index", "fold", prediction_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"incumbent OOF verification missing columns: {sorted(missing)}"
            )
        if frame.duplicated(["well_id", "row_index"]).any():
            raise ValueError("incumbent OOF verification contains duplicate keys")
        maximum = 0.0
        rows = 0
        wells = 0
        for well_id, group in frame.groupby("well_id", sort=True):
            fold_values = group["fold"].astype(int).unique()
            if len(fold_values) != 1:
                raise ValueError(f"OOF well spans multiple folds: {well_id}")
            fold = int(fold_values[0])
            if self.assigned_fold.get(str(well_id)) != fold:
                raise ValueError(f"OOF fold mapping mismatch: {well_id}")
            ordered = group.sort_values("row_index", kind="stable")
            result = self.predict_outer_fold(str(well_id), outer_fold=fold)
            if not np.array_equal(
                result.row_index,
                ordered["row_index"].to_numpy(dtype=np.int64),
            ):
                raise ValueError(f"OOF row-index mismatch: {well_id}")
            reference = ordered[prediction_column].to_numpy(dtype=np.float64)
            maximum = max(maximum, float(np.max(np.abs(result.prediction - reference))))
            rows += len(reference)
            wells += 1
        if not np.isfinite(maximum) or maximum > float(atol):
            raise ValueError(
                f"incumbent state prediction differs from OOF: {maximum} > {atol}"
            )
        return {
            "wells": wells,
            "rows": rows,
            "maximum_absolute_delta": maximum,
            "tolerance": float(atol),
        }

    def state_manifest(self) -> dict[str, object]:
        self.load_all_states()
        records = []
        for fold in range(FOLD_COUNT):
            training, validation = partition_well_ids(self.mapping, fold)
            for seed in self.seeds:
                state = self._states[(seed, fold)]
                records.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "sha256": self._state_hashes[(seed, fold)],
                        "training_wells": len(training),
                        "validation_wells": len(validation),
                        "training_validation_intersection": 0,
                        "registration_version": state.registration_version,
                    }
                )
        return {
            "artifact_root": str(self.artifact_root),
            "seeds": list(self.seeds),
            "folds": list(range(FOLD_COUNT)),
            "states": records,
        }


__all__ = [
    "DEFAULT_INCUMBENT_SEEDS",
    "HeelIncumbentProvider",
    "IncumbentPrediction",
]
