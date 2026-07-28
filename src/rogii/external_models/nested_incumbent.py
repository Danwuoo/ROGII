"""Strict inner-refit incumbent predictions for nested blend evaluation.

For an outer fold ``f`` and an original inner-validation fold ``g != f``, the
state fitted here sees labels only from original folds outside ``{f, g}``.
Validation wells are prepared through the incumbent model's inference-only
loader and are never passed to ``fit`` or to the training-target loader.

Checkpoints are fail-closed.  A partial, corrupt, stale, or provenance-mismatched
checkpoint raises instead of being silently reused or replaced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
from typing import Any, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..data import WellFiles, enumerate_wells, validate_prefix
from ..folds import FOLD_COUNT, load_fold_mapping
from ..models.base import RunContext, TrainingWell
from ..models.heel_registered_anchor_student import (
    MODEL_ID,
    HeelRegisteredAnchorStudentModel,
    HeelRegisteredAnchorStudentState,
    heel_registered_state_provenance,
)
from .incumbent import DEFAULT_INCUMBENT_SEEDS


NESTED_INCUMBENT_SCHEMA_VERSION = "nested_heel_incumbent_v1"
PREDICTION_COLUMN = "y_pred_nested_incumbent"


class NestedIncumbentError(RuntimeError):
    """Raised when nested incumbent evidence cannot be produced safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _ids_sha256(well_ids: Sequence[str]) -> str:
    payload = "\0".join(map(str, well_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(payload: Any) -> Any:
    return json.loads(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    )


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validated_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(map(str, values))
    if (
        not result
        or result != tuple(sorted(result))
        or len(set(result)) != len(result)
        or any(not value for value in result)
    ):
        raise ValueError(f"{name} must be non-empty, unique, and sorted")
    return result


@dataclass(frozen=True)
class NestedFoldSpec:
    """Exact original-fold partition for one nested incumbent fit."""

    outer_fold: int
    inner_validation_fold: int
    training_well_ids: tuple[str, ...]
    validation_well_ids: tuple[str, ...]
    excluded_outer_well_ids: tuple[str, ...]
    fold_version: str
    fold_mapping_sha256: str

    def __post_init__(self) -> None:
        outer = int(self.outer_fold)
        inner = int(self.inner_validation_fold)
        if (
            outer not in range(FOLD_COUNT)
            or inner not in range(FOLD_COUNT)
            or inner == outer
        ):
            raise ValueError(
                "outer_fold and inner_validation_fold must be distinct folds in 0..4"
            )
        training = _validated_ids(
            self.training_well_ids,
            name="training_well_ids",
        )
        validation = _validated_ids(
            self.validation_well_ids,
            name="validation_well_ids",
        )
        excluded = _validated_ids(
            self.excluded_outer_well_ids,
            name="excluded_outer_well_ids",
        )
        groups = (set(training), set(validation), set(excluded))
        if any(
            groups[left] & groups[right] for left in range(3) for right in range(left)
        ):
            raise ValueError("nested fold well groups must be pairwise disjoint")
        fold_version = str(self.fold_version)
        mapping_sha = str(self.fold_mapping_sha256)
        if not fold_version:
            raise ValueError("fold_version must not be empty")
        if not _is_sha256(mapping_sha):
            raise ValueError("fold_mapping_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "outer_fold", outer)
        object.__setattr__(self, "inner_validation_fold", inner)
        object.__setattr__(self, "training_well_ids", training)
        object.__setattr__(self, "validation_well_ids", validation)
        object.__setattr__(self, "excluded_outer_well_ids", excluded)
        object.__setattr__(self, "fold_version", fold_version)
        object.__setattr__(self, "fold_mapping_sha256", mapping_sha)

    @property
    def training_ids_sha256(self) -> str:
        return _ids_sha256(self.training_well_ids)


@dataclass(frozen=True)
class NestedCheckpointProvenance:
    """Deterministic fit identity embedded in every seed checkpoint."""

    schema_version: str
    outer_fold: int
    inner_validation_fold: int
    seed: int
    ordered_training_well_ids: tuple[str, ...]
    training_ids_sha256: str
    validation_well_ids: tuple[str, ...]
    excluded_outer_well_ids: tuple[str, ...]
    fold_version: str
    fold_mapping_sha256: str
    model_id: str
    model_manifest_sha256: str
    model_implementation_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != NESTED_INCUMBENT_SCHEMA_VERSION:
            raise ValueError("nested checkpoint schema version mismatch")
        outer = int(self.outer_fold)
        inner = int(self.inner_validation_fold)
        seed = int(self.seed)
        if (
            outer not in range(FOLD_COUNT)
            or inner not in range(FOLD_COUNT)
            or outer == inner
            or seed < 0
        ):
            raise ValueError("invalid nested checkpoint fold or seed")
        training = _validated_ids(
            self.ordered_training_well_ids,
            name="ordered_training_well_ids",
        )
        validation = _validated_ids(
            self.validation_well_ids,
            name="validation_well_ids",
        )
        excluded = _validated_ids(
            self.excluded_outer_well_ids,
            name="excluded_outer_well_ids",
        )
        if self.training_ids_sha256 != _ids_sha256(training):
            raise ValueError("nested checkpoint training-ID digest mismatch")
        for name in (
            "fold_mapping_sha256",
            "model_manifest_sha256",
            "model_implementation_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        if str(self.model_id) != MODEL_ID:
            raise ValueError("nested checkpoint model_id mismatch")
        if not str(self.fold_version):
            raise ValueError("nested checkpoint fold_version must not be empty")
        object.__setattr__(self, "outer_fold", outer)
        object.__setattr__(self, "inner_validation_fold", inner)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "ordered_training_well_ids", training)
        object.__setattr__(self, "validation_well_ids", validation)
        object.__setattr__(self, "excluded_outer_well_ids", excluded)


@dataclass(frozen=True)
class _NestedCheckpointEnvelope:
    schema_version: str
    provenance: NestedCheckpointProvenance
    state: HeelRegisteredAnchorStudentState


@dataclass(frozen=True)
class NestedCheckpointRecord:
    """Small auditable checkpoint record retained with prediction artifacts."""

    seed: int
    checkpoint_path: str
    manifest_path: str
    checkpoint_sha256: str
    reused: bool
    provenance: NestedCheckpointProvenance
    reciprocal_source_checkpoint_path: str | None = None
    reciprocal_source_checkpoint_sha256: str | None = None
    reciprocal_source_manifest_path: str | None = None
    reciprocal_source_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if int(self.seed) != self.provenance.seed:
            raise ValueError("checkpoint record seed differs from provenance")
        if not _is_sha256(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be bool")
        source_path = self.reciprocal_source_checkpoint_path
        source_sha = self.reciprocal_source_checkpoint_sha256
        source_manifest_path = self.reciprocal_source_manifest_path
        source_manifest_sha = self.reciprocal_source_manifest_sha256
        source_fields = (
            source_path,
            source_sha,
            source_manifest_path,
            source_manifest_sha,
        )
        if any(value is None for value in source_fields) and not all(
            value is None for value in source_fields
        ):
            raise ValueError(
                "reciprocal source checkpoint/manifest paths and SHAs must be "
                "all set or all absent"
            )
        if source_path is not None and (
            not str(source_path)
            or not _is_sha256(source_sha)
            or not str(source_manifest_path)
            or not _is_sha256(source_manifest_sha)
        ):
            raise ValueError("reciprocal source artifact provenance is invalid")


@dataclass(frozen=True)
class NestedCheckpointArtifact:
    """Validated loaded state plus its immutable disk record."""

    state: HeelRegisteredAnchorStudentState
    record: NestedCheckpointRecord


def _readonly_vector(
    values: Any,
    *,
    name: str,
    dtype: Any,
    finite: bool = True,
) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NestedIncumbentPrediction:
    """One inner-validation well predicted by a seed-balanced nested ensemble."""

    well_id: str
    row_index: np.ndarray
    prediction: np.ndarray
    outer_fold: int
    inner_validation_fold: int
    seeds: tuple[int, ...]
    checkpoint_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        well_id = str(self.well_id)
        if not well_id:
            raise ValueError("well_id must not be empty")
        row_index = _readonly_vector(
            self.row_index,
            name="row_index",
            dtype=np.int64,
        )
        prediction = _readonly_vector(
            self.prediction,
            name="prediction",
            dtype=np.float64,
        )
        if (
            prediction.shape != row_index.shape
            or np.any(row_index < 0)
            or np.any(np.diff(row_index) != 1)
        ):
            raise ValueError("nested prediction rows are not aligned and contiguous")
        outer = int(self.outer_fold)
        inner = int(self.inner_validation_fold)
        if (
            outer not in range(FOLD_COUNT)
            or inner not in range(FOLD_COUNT)
            or outer == inner
        ):
            raise ValueError("nested prediction folds are invalid")
        seeds = tuple(map(int, self.seeds))
        hashes = tuple(map(str, self.checkpoint_sha256))
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or len(hashes) != len(seeds)
            or any(seed < 0 for seed in seeds)
            or any(not _is_sha256(value) for value in hashes)
        ):
            raise ValueError("nested prediction seed/checkpoint provenance is invalid")
        object.__setattr__(self, "well_id", well_id)
        object.__setattr__(self, "row_index", row_index)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "outer_fold", outer)
        object.__setattr__(self, "inner_validation_fold", inner)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "checkpoint_sha256", hashes)


@dataclass(frozen=True)
class NestedScopedPredictions:
    """Predictions for an explicitly authorized subset of one nested split."""

    spec: NestedFoldSpec
    allowed_scope: str
    requested_well_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    checkpoints: tuple[NestedCheckpointRecord, ...]
    predictions: tuple[NestedIncumbentPrediction, ...]

    def __post_init__(self) -> None:
        allowed_scopes = {
            "inner_training",
            "inner_validation",
            "inner_training_or_validation",
        }
        scope = str(self.allowed_scope)
        if scope not in allowed_scopes:
            raise ValueError(f"allowed_scope must be one of {sorted(allowed_scopes)}")
        requested = tuple(map(str, self.requested_well_ids))
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(not value for value in requested)
        ):
            raise ValueError("requested_well_ids must be non-empty and unique")
        prediction_ids = tuple(prediction.well_id for prediction in self.predictions)
        if prediction_ids != requested:
            raise ValueError("scoped predictions differ from requested well order")
        seeds = tuple(map(int, self.seeds))
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or tuple(record.seed for record in self.checkpoints) != seeds
        ):
            raise ValueError("scoped checkpoint order differs from seeds")
        checkpoint_hashes = tuple(
            record.checkpoint_sha256 for record in self.checkpoints
        )
        for prediction in self.predictions:
            if (
                prediction.outer_fold != self.spec.outer_fold
                or prediction.inner_validation_fold != self.spec.inner_validation_fold
                or prediction.seeds != seeds
                or prediction.checkpoint_sha256 != checkpoint_hashes
            ):
                raise ValueError(
                    "scoped prediction provenance differs from split/checkpoints"
                )
        object.__setattr__(self, "allowed_scope", scope)
        object.__setattr__(self, "requested_well_ids", requested)
        object.__setattr__(self, "seeds", seeds)

    def to_frame(self) -> pd.DataFrame:
        blocks = [
            pd.DataFrame(
                {
                    "well_id": prediction.well_id,
                    "row_index": prediction.row_index,
                    "outer_fold": prediction.outer_fold,
                    "inner_validation_fold": prediction.inner_validation_fold,
                    PREDICTION_COLUMN: prediction.prediction,
                }
            )
            for prediction in self.predictions
        ]
        frame = pd.concat(blocks, ignore_index=True)
        if frame.duplicated(["well_id", "row_index"]).any():
            raise NestedIncumbentError(
                "scoped nested predictions have duplicate row keys"
            )
        if not np.isfinite(frame[PREDICTION_COLUMN]).all():
            raise NestedIncumbentError("scoped nested predictions are non-finite")
        return frame.reset_index(drop=True)


@dataclass(frozen=True)
class NestedFoldPredictions:
    """All wells from one original inner-validation fold."""

    spec: NestedFoldSpec
    seeds: tuple[int, ...]
    checkpoints: tuple[NestedCheckpointRecord, ...]
    predictions: tuple[NestedIncumbentPrediction, ...]

    def __post_init__(self) -> None:
        seeds = tuple(map(int, self.seeds))
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(seed < 0 for seed in seeds)
        ):
            raise ValueError("nested fold seeds must be non-empty and unique")
        if tuple(record.seed for record in self.checkpoints) != seeds:
            raise ValueError("nested checkpoint order must match configured seeds")
        checkpoint_hashes = tuple(
            record.checkpoint_sha256 for record in self.checkpoints
        )
        for record in self.checkpoints:
            provenance = record.provenance
            if (
                provenance.outer_fold != self.spec.outer_fold
                or provenance.inner_validation_fold != self.spec.inner_validation_fold
                or provenance.ordered_training_well_ids != self.spec.training_well_ids
            ):
                raise ValueError("nested checkpoint provenance differs from fold spec")
        prediction_ids = tuple(prediction.well_id for prediction in self.predictions)
        if prediction_ids != self.spec.validation_well_ids:
            raise ValueError(
                "nested predictions do not cover the inner validation fold"
            )
        for prediction in self.predictions:
            if (
                prediction.outer_fold != self.spec.outer_fold
                or prediction.inner_validation_fold != self.spec.inner_validation_fold
                or prediction.seeds != seeds
                or prediction.checkpoint_sha256 != checkpoint_hashes
            ):
                raise ValueError("nested prediction provenance differs from fold spec")
        object.__setattr__(self, "seeds", seeds)

    @property
    def row_count(self) -> int:
        return sum(len(prediction.row_index) for prediction in self.predictions)

    def to_frame(self) -> pd.DataFrame:
        blocks = [
            pd.DataFrame(
                {
                    "well_id": prediction.well_id,
                    "row_index": prediction.row_index,
                    "outer_fold": prediction.outer_fold,
                    "inner_validation_fold": prediction.inner_validation_fold,
                    PREDICTION_COLUMN: prediction.prediction,
                }
            )
            for prediction in self.predictions
        ]
        frame = pd.concat(blocks, ignore_index=True)
        if frame.duplicated(["well_id", "row_index"]).any():
            raise NestedIncumbentError("nested prediction frame has duplicate row keys")
        return frame.sort_values(
            ["well_id", "row_index"],
            kind="stable",
        ).reset_index(drop=True)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_INCUMBENT_SCHEMA_VERSION,
            "outer_fold": self.spec.outer_fold,
            "inner_validation_fold": self.spec.inner_validation_fold,
            "training_well_ids": list(self.spec.training_well_ids),
            "training_ids_sha256": self.spec.training_ids_sha256,
            "validation_well_ids": list(self.spec.validation_well_ids),
            "excluded_outer_well_ids": list(self.spec.excluded_outer_well_ids),
            "seeds": list(self.seeds),
            "rows": self.row_count,
            "checkpoints": [
                {
                    "seed": record.seed,
                    "path": record.checkpoint_path,
                    "manifest_path": record.manifest_path,
                    "sha256": record.checkpoint_sha256,
                    "reused": record.reused,
                    "reciprocal_source_checkpoint_path": (
                        record.reciprocal_source_checkpoint_path
                    ),
                    "reciprocal_source_checkpoint_sha256": (
                        record.reciprocal_source_checkpoint_sha256
                    ),
                    "reciprocal_source_manifest_path": (
                        record.reciprocal_source_manifest_path
                    ),
                    "reciprocal_source_manifest_sha256": (
                        record.reciprocal_source_manifest_sha256
                    ),
                }
                for record in self.checkpoints
            ],
        }


@dataclass(frozen=True)
class NestedOuterOOF:
    """Four inner-fold prediction blocks covering one outer-training population."""

    outer_fold: int
    folds: tuple[NestedFoldPredictions, ...]

    def __post_init__(self) -> None:
        outer = int(self.outer_fold)
        if outer not in range(FOLD_COUNT):
            raise ValueError("outer_fold must be in 0..4")
        expected_inner = tuple(fold for fold in range(FOLD_COUNT) if fold != outer)
        actual_inner = tuple(result.spec.inner_validation_fold for result in self.folds)
        if actual_inner != expected_inner:
            raise ValueError("outer nested OOF must contain all other folds in order")
        if any(result.spec.outer_fold != outer for result in self.folds):
            raise ValueError("outer nested OOF contains a foreign outer fold")
        validation_ids = [
            well_id
            for result in self.folds
            for well_id in result.spec.validation_well_ids
        ]
        if len(validation_ids) != len(set(validation_ids)):
            raise ValueError("outer nested OOF predicts a well more than once")
        object.__setattr__(self, "outer_fold", outer)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.concat(
            [result.to_frame() for result in self.folds],
            ignore_index=True,
        )
        if frame.duplicated(["well_id", "row_index"]).any():
            raise NestedIncumbentError("outer nested OOF has duplicate row keys")
        return frame.sort_values(
            ["inner_validation_fold", "well_id", "row_index"],
            kind="stable",
        ).reset_index(drop=True)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_INCUMBENT_SCHEMA_VERSION,
            "outer_fold": self.outer_fold,
            "inner_folds": [result.manifest() for result in self.folds],
            "well_count": sum(len(result.predictions) for result in self.folds),
            "row_count": sum(result.row_count for result in self.folds),
        }


class NestedHeelIncumbentRefitter:
    """Fit/load fold-safe heel-incumbent states for nested blend evidence."""

    def __init__(
        self,
        *,
        data_root: Path,
        fold_config: Path,
        cache_root: Path,
        checkpoint_root: Path,
        seeds: Sequence[int] = DEFAULT_INCUMBENT_SEEDS,
        workers: int = 1,
        model: HeelRegisteredAnchorStudentModel | None = None,
        reciprocal_state_reuse: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.fold_config = Path(fold_config).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.checkpoint_root = Path(checkpoint_root).resolve()
        for name, path in (
            ("cache_root", self.cache_root),
            ("checkpoint_root", self.checkpoint_root),
        ):
            if path == self.data_root or self.data_root in path.parents:
                raise ValueError(f"{name} cannot be inside competition data")
        workers = int(workers)
        if workers < 1:
            raise ValueError("workers must be positive")
        configured_seeds = tuple(map(int, seeds))
        if (
            not configured_seeds
            or len(set(configured_seeds)) != len(configured_seeds)
            or any(seed < 0 for seed in configured_seeds)
        ):
            raise ValueError("seeds must be non-empty, unique, and non-negative")

        self.workers = workers
        self.seeds = configured_seeds
        self.model = model or HeelRegisteredAnchorStudentModel()
        if str(getattr(self.model, "model_id", "")) != MODEL_ID:
            raise TypeError("model must implement the heel-registered incumbent")
        for method in ("prepare", "fit", "predict", "manifest"):
            if not callable(getattr(self.model, method, None)):
                raise TypeError(f"model is missing required method {method}")
        manifest = self.model.manifest()
        if not isinstance(manifest, dict) or manifest.get("model_id") != MODEL_ID:
            raise ValueError("heel incumbent model manifest is invalid")
        if not isinstance(reciprocal_state_reuse, bool):
            raise TypeError("reciprocal_state_reuse must be bool")
        if (
            reciprocal_state_reuse
            and type(self.model) is not HeelRegisteredAnchorStudentModel
        ):
            raise TypeError(
                "reciprocal state reuse requires the exact heel-registered model"
            )
        self.reciprocal_state_reuse = reciprocal_state_reuse
        self.model_manifest = _jsonable(manifest)
        self.model_manifest_sha256 = _canonical_sha256(self.model_manifest)
        implementation_path = inspect.getsourcefile(type(self.model))
        if implementation_path is None or not Path(implementation_path).is_file():
            raise ValueError("cannot identify heel incumbent implementation source")
        self.model_implementation_sha256 = _sha256_file(Path(implementation_path))

        train_files = enumerate_wells(self.data_root, "train")
        self.train_files = {well.well_id: well for well in train_files}
        self.mapping = load_fold_mapping(
            self.fold_config,
            expected_well_ids=self.train_files,
        )
        self.fold_version = str(self.mapping["fold_version"].iat[0])
        self.fold_mapping_sha256 = _sha256_file(self.fold_config)
        self.assigned_fold = (
            self.mapping.set_index("well_id")["fold"].astype(int).to_dict()
        )
        self._prepared: dict[str, Any] = {}
        # Reciprocal orientations exclude the same unordered pair of folds and
        # therefore use numerically identical states.  Cache only predictions
        # for the shared training wells; validation wells differ by direction.
        # This cache is target-free, process-local, and optional: a resume in a
        # fresh process safely recomputes every missing prediction.
        self._reciprocal_prediction_cache: dict[tuple[int, int, str], np.ndarray] = {}

    def fold_spec(
        self,
        *,
        outer_fold: int,
        inner_validation_fold: int,
    ) -> NestedFoldSpec:
        outer = int(outer_fold)
        inner = int(inner_validation_fold)
        if (
            outer not in range(FOLD_COUNT)
            or inner not in range(FOLD_COUNT)
            or outer == inner
        ):
            raise ValueError(
                "outer_fold and inner_validation_fold must be distinct folds in 0..4"
            )
        training = tuple(
            sorted(
                self.mapping.loc[
                    ~self.mapping["fold"].isin([outer, inner]),
                    "well_id",
                ].astype(str)
            )
        )
        validation = tuple(
            sorted(
                self.mapping.loc[
                    self.mapping["fold"].eq(inner),
                    "well_id",
                ].astype(str)
            )
        )
        excluded = tuple(
            sorted(
                self.mapping.loc[
                    self.mapping["fold"].eq(outer),
                    "well_id",
                ].astype(str)
            )
        )
        spec = NestedFoldSpec(
            outer_fold=outer,
            inner_validation_fold=inner,
            training_well_ids=training,
            validation_well_ids=validation,
            excluded_outer_well_ids=excluded,
            fold_version=self.fold_version,
            fold_mapping_sha256=self.fold_mapping_sha256,
        )
        covered = (
            set(spec.training_well_ids)
            | set(spec.validation_well_ids)
            | set(spec.excluded_outer_well_ids)
        )
        if covered != set(self.train_files):
            raise NestedIncumbentError("nested fold spec does not cover train wells")
        return spec

    def _checkpoint_paths(
        self,
        spec: NestedFoldSpec,
        *,
        seed: int,
    ) -> tuple[Path, Path]:
        directory = (
            self.checkpoint_root
            / NESTED_INCUMBENT_SCHEMA_VERSION
            / f"outer_{spec.outer_fold}"
            / f"inner_{spec.inner_validation_fold}"
        )
        checkpoint = directory / f"seed_{int(seed)}.pkl"
        return checkpoint, checkpoint.with_suffix(".json")

    def _expected_provenance(
        self,
        spec: NestedFoldSpec,
        *,
        seed: int,
    ) -> NestedCheckpointProvenance:
        return NestedCheckpointProvenance(
            schema_version=NESTED_INCUMBENT_SCHEMA_VERSION,
            outer_fold=spec.outer_fold,
            inner_validation_fold=spec.inner_validation_fold,
            seed=int(seed),
            ordered_training_well_ids=spec.training_well_ids,
            training_ids_sha256=spec.training_ids_sha256,
            validation_well_ids=spec.validation_well_ids,
            excluded_outer_well_ids=spec.excluded_outer_well_ids,
            fold_version=spec.fold_version,
            fold_mapping_sha256=spec.fold_mapping_sha256,
            model_id=MODEL_ID,
            model_manifest_sha256=self.model_manifest_sha256,
            model_implementation_sha256=self.model_implementation_sha256,
        )

    def _validate_state(
        self,
        state: HeelRegisteredAnchorStudentState,
        expected: NestedCheckpointProvenance,
    ) -> None:
        if not isinstance(state, HeelRegisteredAnchorStudentState):
            raise TypeError("nested incumbent state has the wrong type")
        training_ids = tuple(map(str, state.training_well_ids))
        base_ids = tuple(map(str, state.base_state.training_well_ids))
        student_ids = tuple(map(str, state.base_state.student_state.training_well_ids))
        invalid = (
            int(state.fold) != expected.inner_validation_fold
            or int(state.base_state.fold) != expected.inner_validation_fold
            or int(state.random_seed) != expected.seed
            or int(state.base_state.student_state.random_seed) != expected.seed
            or training_ids != expected.ordered_training_well_ids
            or base_ids != expected.ordered_training_well_ids
            or student_ids != expected.ordered_training_well_ids
            or state.base_model_id
            != str(
                getattr(
                    getattr(self.model, "base_model", None),
                    "model_id",
                    "",
                )
            )
            or set(training_ids) & set(expected.validation_well_ids)
            or set(training_ids) & set(expected.excluded_outer_well_ids)
        )
        if invalid:
            raise NestedIncumbentError(
                "nested incumbent state fold/seed/training provenance mismatch"
            )
        provenance = heel_registered_state_provenance(state)
        if (
            provenance["registration_uses_cross_well_learned_statistics"]
            or provenance["training_well_ids"]
            != list(expected.ordered_training_well_ids)
            or int(provenance["fold"]) != expected.inner_validation_fold
            or int(provenance["random_seed"]) != expected.seed
        ):
            raise NestedIncumbentError(
                "nested incumbent state registration provenance mismatch"
            )
        registration = self.model_manifest.get("registration", {})
        if provenance["registration_version"] != registration.get(
            "version"
        ) or provenance["registration_config"] != registration.get("config"):
            raise NestedIncumbentError(
                "nested incumbent state differs from configured registration"
            )

    def _prepare(self, well_id: str) -> Any:
        identifier = str(well_id)
        if identifier not in self.train_files:
            raise KeyError(f"unknown train well: {identifier}")
        if identifier not in self._prepared:
            prepared = self.model.prepare(
                self.train_files[identifier],
                split="train",
                cache_root=self.cache_root,
            )
            row_index = np.asarray(getattr(prepared, "row_index", None))
            if (
                str(getattr(prepared, "well_id", "")) != identifier
                or row_index.ndim != 1
                or len(row_index) == 0
                or np.any(row_index < 0)
                or np.any(np.diff(row_index) != 1)
            ):
                raise NestedIncumbentError(
                    f"prepared incumbent payload is misaligned: {identifier}"
                )
            self._prepared[identifier] = prepared
        return self._prepared[identifier]

    @staticmethod
    def _load_training_target(
        well: WellFiles,
        row_index: np.ndarray,
    ) -> np.ndarray:
        frame = pd.read_csv(
            well.horizontal_path,
            usecols=["TVT", "TVT_input"],
        )
        known_count, prediction_count = validate_prefix(
            frame["TVT_input"],
            target=frame["TVT"],
        )
        expected = np.arange(
            known_count,
            known_count + prediction_count,
            dtype=np.int64,
        )
        rows = np.asarray(row_index, dtype=np.int64)
        if not np.array_equal(rows, expected):
            raise NestedIncumbentError(
                f"training target row alignment differs: {well.well_id}"
            )
        target = frame["TVT"].iloc[rows].to_numpy(dtype=np.float64)
        if target.shape != rows.shape or not np.isfinite(target).all():
            raise NestedIncumbentError(f"training target is invalid: {well.well_id}")
        return target

    def _load_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        manifest_path: Path,
        expected: NestedCheckpointProvenance,
        reused: bool,
    ) -> NestedCheckpointArtifact:
        if not checkpoint_path.is_file() or not manifest_path.is_file():
            raise NestedIncumbentError(
                "nested checkpoint is partial; both pickle and manifest are required"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise NestedIncumbentError(
                f"nested checkpoint manifest is unreadable: {manifest_path}"
            ) from error
        checkpoint_sha = _sha256_file(checkpoint_path)
        if (
            manifest.get("schema_version") != NESTED_INCUMBENT_SCHEMA_VERSION
            or manifest.get("checkpoint_sha256") != checkpoint_sha
            or manifest.get("provenance") != _jsonable(asdict(expected))
        ):
            raise NestedIncumbentError(
                f"nested checkpoint manifest provenance mismatch: {manifest_path}"
            )
        try:
            with checkpoint_path.open("rb") as handle:
                envelope = pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError) as error:
            raise NestedIncumbentError(
                f"nested checkpoint is unreadable: {checkpoint_path}"
            ) from error
        if (
            not isinstance(envelope, _NestedCheckpointEnvelope)
            or envelope.schema_version != NESTED_INCUMBENT_SCHEMA_VERSION
            or envelope.provenance != expected
        ):
            raise NestedIncumbentError(
                f"nested checkpoint envelope provenance mismatch: {checkpoint_path}"
            )
        self._validate_state(envelope.state, expected)
        state_provenance = heel_registered_state_provenance(envelope.state)
        if manifest.get("state_provenance") != _jsonable(state_provenance):
            raise NestedIncumbentError(
                f"nested checkpoint state manifest mismatch: {manifest_path}"
            )
        reciprocal = manifest.get("reciprocal_state_reuse")
        source_path: str | None = None
        source_sha: str | None = None
        source_manifest_path: str | None = None
        source_manifest_sha: str | None = None
        if reciprocal is not None:
            if not isinstance(reciprocal, dict):
                raise NestedIncumbentError(
                    f"nested reciprocal reuse provenance is invalid: {manifest_path}"
                )
            source_path = str(reciprocal.get("source_checkpoint_path", ""))
            source_sha = str(reciprocal.get("source_checkpoint_sha256", ""))
            source_manifest_path = str(reciprocal.get("source_manifest_path", ""))
            source_manifest_sha = str(reciprocal.get("source_manifest_sha256", ""))
            source = Path(source_path)
            source_manifest = Path(source_manifest_path)
            valid_reciprocal = (
                bool(source_path)
                and _is_sha256(source_sha)
                and source.is_file()
                and _sha256_file(source) == source_sha
                and bool(source_manifest_path)
                and _is_sha256(source_manifest_sha)
                and source_manifest.is_file()
                and _sha256_file(source_manifest) == source_manifest_sha
                and int(reciprocal.get("source_outer_fold", -1))
                == expected.inner_validation_fold
                and int(reciprocal.get("source_inner_validation_fold", -1))
                == expected.outer_fold
                and int(reciprocal.get("seed", -1)) == expected.seed
            )
            if not valid_reciprocal:
                raise NestedIncumbentError(
                    f"nested reciprocal reuse source is stale: {manifest_path}"
                )
        record = NestedCheckpointRecord(
            seed=expected.seed,
            checkpoint_path=str(checkpoint_path),
            manifest_path=str(manifest_path),
            checkpoint_sha256=checkpoint_sha,
            reused=reused,
            provenance=expected,
            reciprocal_source_checkpoint_path=source_path,
            reciprocal_source_checkpoint_sha256=source_sha,
            reciprocal_source_manifest_path=source_manifest_path,
            reciprocal_source_manifest_sha256=source_manifest_sha,
        )
        return NestedCheckpointArtifact(state=envelope.state, record=record)

    def fit_or_load_seed(
        self,
        *,
        outer_fold: int,
        inner_validation_fold: int,
        seed: int,
        resume: bool = True,
    ) -> NestedCheckpointArtifact:
        """Fit one seed or load its exact matching checkpoint."""

        seed = int(seed)
        if seed not in self.seeds:
            raise ValueError(f"seed {seed} is outside the configured ensemble")
        if not isinstance(resume, bool):
            raise TypeError("resume must be bool")
        spec = self.fold_spec(
            outer_fold=outer_fold,
            inner_validation_fold=inner_validation_fold,
        )
        expected = self._expected_provenance(spec, seed=seed)
        checkpoint_path, manifest_path = self._checkpoint_paths(
            spec,
            seed=seed,
        )
        existing = (checkpoint_path.exists(), manifest_path.exists())
        if any(existing):
            if not all(existing):
                raise NestedIncumbentError(
                    "nested checkpoint is partial and will not be overwritten"
                )
            if not resume:
                raise FileExistsError(
                    f"nested checkpoint already exists: {checkpoint_path}"
                )
            return self._load_checkpoint(
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                expected=expected,
                reused=True,
            )

        if self.reciprocal_state_reuse:
            reciprocal_spec = self.fold_spec(
                outer_fold=spec.inner_validation_fold,
                inner_validation_fold=spec.outer_fold,
            )
            if reciprocal_spec.training_well_ids != spec.training_well_ids:
                raise NestedIncumbentError(
                    "reciprocal nested splits differ in training population"
                )
            reciprocal_expected = self._expected_provenance(
                reciprocal_spec,
                seed=seed,
            )
            reciprocal_checkpoint, reciprocal_manifest = self._checkpoint_paths(
                reciprocal_spec,
                seed=seed,
            )
            reciprocal_existing = (
                reciprocal_checkpoint.exists(),
                reciprocal_manifest.exists(),
            )
            if any(reciprocal_existing) and not all(reciprocal_existing):
                raise NestedIncumbentError(
                    "reciprocal nested checkpoint is partial and cannot be reused"
                )
            if all(reciprocal_existing):
                source = self._load_checkpoint(
                    checkpoint_path=reciprocal_checkpoint,
                    manifest_path=reciprocal_manifest,
                    expected=reciprocal_expected,
                    reused=True,
                )
                cloned_base = replace(
                    source.state.base_state,
                    fold=spec.inner_validation_fold,
                )
                state = replace(
                    source.state,
                    base_state=cloned_base,
                    fold=spec.inner_validation_fold,
                )
                self._validate_state(state, expected)
                envelope = _NestedCheckpointEnvelope(
                    schema_version=NESTED_INCUMBENT_SCHEMA_VERSION,
                    provenance=expected,
                    state=state,
                )
                _atomic_pickle(checkpoint_path, envelope)
                checkpoint_sha = _sha256_file(checkpoint_path)
                _atomic_json(
                    manifest_path,
                    {
                        "schema_version": NESTED_INCUMBENT_SCHEMA_VERSION,
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_path": str(checkpoint_path),
                        "provenance": asdict(expected),
                        "state_provenance": heel_registered_state_provenance(state),
                        "reciprocal_state_reuse": {
                            "source_checkpoint_path": str(reciprocal_checkpoint),
                            "source_checkpoint_sha256": (
                                source.record.checkpoint_sha256
                            ),
                            "source_manifest_path": str(reciprocal_manifest),
                            "source_manifest_sha256": _sha256_file(reciprocal_manifest),
                            "source_outer_fold": reciprocal_spec.outer_fold,
                            "source_inner_validation_fold": (
                                reciprocal_spec.inner_validation_fold
                            ),
                            "seed": seed,
                            "equivalence_basis": (
                                "identical ordered training wells and seed; "
                                "current exact heel model uses context.fold "
                                "only as stored state provenance"
                            ),
                        },
                    },
                )
                return self._load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    manifest_path=manifest_path,
                    expected=expected,
                    reused=False,
                )

        training: list[TrainingWell] = []
        for well_id in spec.training_well_ids:
            inference = self._prepare(well_id)
            target = self._load_training_target(
                self.train_files[well_id],
                np.asarray(inference.row_index, dtype=np.int64),
            )
            training.append(TrainingWell(inference=inference, target=target))
        actual_ids = tuple(str(item.inference.well_id) for item in training)
        if actual_ids != spec.training_well_ids:
            raise NestedIncumbentError("nested fit training order changed")

        work_dir = checkpoint_path.parent / "work" / f"seed_{seed}"
        work_dir.mkdir(parents=True, exist_ok=True)
        context = RunContext(
            fold=spec.inner_validation_fold,
            seed=seed,
            workers=self.workers,
            work_dir=work_dir,
        )
        try:
            with threadpool_limits(limits=self.workers):
                state = self.model.fit(training, context=context)
        except Exception as error:
            raise NestedIncumbentError(
                "nested incumbent fit failed for "
                f"outer={spec.outer_fold}, inner={spec.inner_validation_fold}, "
                f"seed={seed}: {type(error).__name__}: {error}"
            ) from error
        self._validate_state(state, expected)
        envelope = _NestedCheckpointEnvelope(
            schema_version=NESTED_INCUMBENT_SCHEMA_VERSION,
            provenance=expected,
            state=state,
        )
        _atomic_pickle(checkpoint_path, envelope)
        checkpoint_sha = _sha256_file(checkpoint_path)
        _atomic_json(
            manifest_path,
            {
                "schema_version": NESTED_INCUMBENT_SCHEMA_VERSION,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_path": str(checkpoint_path),
                "provenance": asdict(expected),
                "state_provenance": heel_registered_state_provenance(state),
            },
        )
        return self._load_checkpoint(
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            expected=expected,
            reused=False,
        )

    def predict_inner_validation(
        self,
        *,
        outer_fold: int,
        inner_validation_fold: int,
        resume: bool = True,
    ) -> NestedFoldPredictions:
        """Predict every well in original fold ``g`` from states excluding ``f,g``."""

        spec = self.fold_spec(
            outer_fold=outer_fold,
            inner_validation_fold=inner_validation_fold,
        )
        scoped = self.predict_wells(
            outer_fold=spec.outer_fold,
            inner_validation_fold=spec.inner_validation_fold,
            well_ids=spec.validation_well_ids,
            allowed_scope="inner_validation",
            resume=resume,
        )
        return NestedFoldPredictions(
            spec=spec,
            seeds=self.seeds,
            checkpoints=scoped.checkpoints,
            predictions=scoped.predictions,
        )

    def predict_wells(
        self,
        *,
        outer_fold: int,
        inner_validation_fold: int,
        well_ids: Sequence[str],
        allowed_scope: str = "inner_training_or_validation",
        resume: bool = True,
    ) -> NestedScopedPredictions:
        """Predict authorized non-outer wells with the configured seed ensemble.

        The scope check happens before checkpoint loading or fitting.  Original
        outer-validation wells are never a legal request.
        """

        spec = self.fold_spec(
            outer_fold=outer_fold,
            inner_validation_fold=inner_validation_fold,
        )
        requested = tuple(map(str, well_ids))
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(not value for value in requested)
        ):
            raise ValueError("well_ids must be non-empty and unique")
        scopes = {
            "inner_training": set(spec.training_well_ids),
            "inner_validation": set(spec.validation_well_ids),
            "inner_training_or_validation": set(spec.training_well_ids)
            | set(spec.validation_well_ids),
        }
        scope = str(allowed_scope)
        if scope not in scopes:
            raise ValueError(f"allowed_scope must be one of {sorted(scopes)}")
        requested_set = set(requested)
        outer_requested = requested_set & set(spec.excluded_outer_well_ids)
        if outer_requested:
            raise NestedIncumbentError(
                "outer-validation wells are forbidden in nested incumbent "
                f"prediction: {sorted(outer_requested)[:5]}"
            )
        unauthorized = requested_set - scopes[scope]
        if unauthorized:
            raise NestedIncumbentError(
                f"well_ids are outside allowed_scope={scope!r}: "
                f"{sorted(unauthorized)[:5]}"
            )
        artifacts = tuple(
            self.fit_or_load_seed(
                outer_fold=spec.outer_fold,
                inner_validation_fold=spec.inner_validation_fold,
                seed=seed,
                resume=resume,
            )
            for seed in self.seeds
        )
        predictions: list[NestedIncumbentPrediction] = []
        checkpoint_hashes = tuple(
            artifact.record.checkpoint_sha256 for artifact in artifacts
        )
        unordered_folds = tuple(sorted((spec.outer_fold, spec.inner_validation_fold)))
        shared_training = set(spec.training_well_ids)
        for well_id in requested:
            inference = self._prepare(well_id)
            cache_key = (*unordered_folds, well_id)
            cached = (
                self._reciprocal_prediction_cache.get(cache_key)
                if self.reciprocal_state_reuse and well_id in shared_training
                else None
            )
            if cached is None:
                seed_predictions: list[np.ndarray] = []
                for artifact in artifacts:
                    try:
                        values = np.asarray(
                            self.model.predict(artifact.state, inference),
                            dtype=np.float64,
                        )
                    except Exception as error:
                        raise NestedIncumbentError(
                            "nested incumbent prediction failed for "
                            f"{well_id}/outer={spec.outer_fold}/"
                            f"inner={spec.inner_validation_fold}/"
                            f"seed={artifact.record.seed}: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    if (
                        values.ndim != 1
                        or values.shape != np.asarray(inference.row_index).shape
                        or not np.isfinite(values).all()
                    ):
                        raise NestedIncumbentError(
                            f"nested incumbent produced invalid rows: {well_id}"
                        )
                    seed_predictions.append(values)
                ensemble = np.mean(np.stack(seed_predictions), axis=0)
                if self.reciprocal_state_reuse and well_id in shared_training:
                    stored = np.array(ensemble, dtype=np.float64, copy=True)
                    stored.setflags(write=False)
                    self._reciprocal_prediction_cache[cache_key] = stored
            else:
                ensemble = np.array(cached, dtype=np.float64, copy=True)
            if not np.isfinite(ensemble).all():
                raise NestedIncumbentError(
                    f"nested incumbent ensemble is non-finite: {well_id}"
                )
            predictions.append(
                NestedIncumbentPrediction(
                    well_id=well_id,
                    row_index=np.asarray(inference.row_index, dtype=np.int64),
                    prediction=ensemble,
                    outer_fold=spec.outer_fold,
                    inner_validation_fold=spec.inner_validation_fold,
                    seeds=self.seeds,
                    checkpoint_sha256=checkpoint_hashes,
                )
            )
        return NestedScopedPredictions(
            spec=spec,
            allowed_scope=scope,
            requested_well_ids=requested,
            seeds=self.seeds,
            checkpoints=tuple(artifact.record for artifact in artifacts),
            predictions=tuple(predictions),
        )

    def predict_outer_training_oof(
        self,
        *,
        outer_fold: int,
        resume: bool = True,
    ) -> NestedOuterOOF:
        """Cross-fit the incumbent across all four folds outside ``outer_fold``."""

        outer = int(outer_fold)
        if outer not in range(FOLD_COUNT):
            raise ValueError("outer_fold must be in 0..4")
        results = tuple(
            self.predict_inner_validation(
                outer_fold=outer,
                inner_validation_fold=inner,
                resume=resume,
            )
            for inner in range(FOLD_COUNT)
            if inner != outer
        )
        output = NestedOuterOOF(outer_fold=outer, folds=results)
        expected_ids = set(self.train_files) - {
            well_id
            for well_id, fold in self.assigned_fold.items()
            if int(fold) == outer
        }
        actual_ids = {
            prediction.well_id
            for result in output.folds
            for prediction in result.predictions
        }
        if actual_ids != expected_ids:
            raise NestedIncumbentError(
                "outer nested OOF does not cover exactly the outer-training wells"
            )
        return output


__all__ = [
    "NESTED_INCUMBENT_SCHEMA_VERSION",
    "PREDICTION_COLUMN",
    "NestedCheckpointArtifact",
    "NestedCheckpointProvenance",
    "NestedCheckpointRecord",
    "NestedFoldPredictions",
    "NestedFoldSpec",
    "NestedHeelIncumbentRefitter",
    "NestedIncumbentError",
    "NestedIncumbentPrediction",
    "NestedOuterOOF",
    "NestedScopedPredictions",
]
