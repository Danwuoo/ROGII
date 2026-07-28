"""Strictly nested TCN/incumbent ensemble evidence.

This runner deliberately separates two prediction populations:

* inner OOF predictions are produced by refitting both the heel-registered
  incumbent and the TCN after excluding the outer fold and the current inner
  validation fold;
* outer predictions are loaded unchanged from an independently audited full
  external-model artifact bundle.

Only the inner OOF population is allowed to fit blend weights.  Every outer
validation well is rejected by the nested incumbent API before a checkpoint is
loaded or fitted, and TCN checkpoint sidecars bind the exact training IDs,
inputs, targets, row weights, incumbent states, implementation, and data
manifest.  Partial or stale artifacts fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import WellFiles, enumerate_wells, validate_prefix
from ..ensemble import (
    NestedBlendProvenance,
    NestedFoldBlendResult,
    fit_nested_fold_blend,
    nested_weight_table,
)
from ..external_models.artifacts import (
    DEFAULT_BUNDLE_NAME,
    capture_data_manifest,
    independent_reaudit_external_artifacts,
    load_audited_incumbent_reference,
    read_external_artifact_bundle,
    sha256_file,
    well_ids_sha256,
)
from ..external_models.components import validate_residual_components
from ..external_models.contracts import ResidualTrainingWell, SequenceModelInput
from ..external_models.data import (
    MODEL_INPUT_CHANNEL_NAMES,
    FixedGridConfig,
    FixedGridSequenceWell,
    build_sequence_model_input,
    fixed_grid_manifest,
    prepare_fixed_grid_well,
)
from ..external_models.incumbent import DEFAULT_INCUMBENT_SEEDS
from ..external_models.nested_incumbent import (
    NestedHeelIncumbentRefitter,
)
from ..external_models.targets import (
    SmoothTargetConfig,
    grid_prediction_to_row_components,
    make_grid_residual_target,
    target_manifest,
)
from ..external_models.torch_families import (
    FAMILY_TCN,
    TorchFamilyConfig,
    TorchFamilyState,
    fit_torch_family,
    load_torch_family,
)
from ..folds import FOLD_COUNT, load_fold_mapping, partition_well_ids
from ..portfolio_metrics import (
    OOF_KEY_COLUMNS,
    common_slope_shape_metrics,
    paired_well_cluster_bootstrap,
    per_fold_metrics,
    per_well_win_summary,
    pooled_metrics,
    residual_correlation,
    shadow_private_subset_simulation,
    tail_metrics,
    validate_oof_frame,
)


SCHEMA_VERSION = "nested_external_tcn_ensemble_v1"
PRODUCTION_TCN_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
PRODUCTION_GRID = FixedGridConfig(horizontal_points=256, typewell_points=256)
PRODUCTION_TARGET = SmoothTargetConfig(
    retained_dct_modes=16,
    maximum_absolute_correction=80.0,
)
PRODUCTION_EPOCHS = 60
PRODUCTION_HIDDEN_DIM = 16
PRODUCTION_LAYERS = 2
REGULARIZATION_STRENGTH = 25.0
REGULARIZATION_TARGET = {"incumbent": 0.75, "tcn": 0.25}
FAMILY_MAX_WEIGHT = {"tcn": 0.50}
FIXED_INCUMBENT_HEAVY = {"incumbent": 0.75, "tcn": 0.25}
FIXED_SIMPLE_AVERAGE = {"incumbent": 0.50, "tcn": 0.50}

INCUMBENT_COLUMN = "y_pred_incumbent"
TCN_COLUMN = "y_pred_tcn"
NESTED_COLUMN = "y_pred_nested_tcn"
FIXED_075_COLUMN = "y_pred_fixed_075_025"
FIXED_050_COLUMN = "y_pred_simple_050_050"
SOURCE_INCUMBENT_COLUMN = "y_pred_heel_registered_ensemble"
CANONICAL_TCN_COLUMN = "y_pred__tcn"


class NestedExternalEnsembleError(RuntimeError):
    """Raised when nested evidence cannot be constructed safely."""


def _jsonable(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _string_sequence_sha256(value: Sequence[Any]) -> str:
    encoded = json.dumps(
        tuple(map(str, value)),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    compression: str | dict[str, Any] | None = None
    if destination.suffix == ".gz":
        compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        compression=compression,
    )
    os.replace(temporary, destination)


def _safe_identifier(value: str, *, name: str) -> str:
    text = str(value)
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    )
    if not text or any(character not in allowed for character in text):
        raise ValueError(f"{name} must contain only letters, digits, '_' or '-'")
    return text


def _ensure_outside_data(path: Path, data_root: Path, *, name: str) -> Path:
    resolved = Path(path).resolve()
    data = Path(data_root).resolve()
    if resolved == data or data in resolved.parents:
        raise ValueError(f"{name} cannot be inside competition data")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NestedExternalEnsembleError(
            f"unreadable JSON artifact: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise NestedExternalEnsembleError(f"JSON artifact must be an object: {path}")
    return payload


def _read_oof(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            dtype={"well_id": "string", "row_index": "int64", "fold": "int8"},
            float_precision="round_trip",
        )
    except (OSError, ValueError, TypeError) as error:
        raise NestedExternalEnsembleError(f"unreadable OOF artifact: {path}") from error
    return frame


def _implementation_hashes() -> dict[str, str]:
    objects = {
        "nested_runner": run_nested_external_ensemble,
        "nested_incumbent": NestedHeelIncumbentRefitter,
        "fixed_grid": build_sequence_model_input,
        "residual_target": make_grid_residual_target,
        "torch_family": fit_torch_family,
        "blend": fit_nested_fold_blend,
    }
    result: dict[str, str] = {}
    for name, value in objects.items():
        source = inspect.getsourcefile(value)
        if source is None or not Path(source).is_file():
            raise NestedExternalEnsembleError(
                f"cannot resolve implementation source for {name}"
            )
        result[name] = sha256_file(source)
    return result


def _tcn_config(
    *,
    seed: int,
    device: str,
    epochs: int,
    hidden_dim: int,
    layers: int,
) -> TorchFamilyConfig:
    return TorchFamilyConfig(
        family=FAMILY_TCN,
        seed=int(seed),
        device=str(device),
        epochs=int(epochs),
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        gradient_clip=5.0,
        hidden_dim=int(hidden_dim),
        layers=int(layers),
        tcn_kernel_size=3,
    )


@dataclass(frozen=True)
class OuterTcnSource:
    """Validated untouched outer OOF incumbent and family-first TCN mean."""

    frame: pd.DataFrame
    source_root: str
    oof_sha256: str
    manifest_sha256: str
    artifact_ids: tuple[str, ...]
    audit: Mapping[str, Any]
    evidence_eligible: bool

    def __post_init__(self) -> None:
        required = [
            *OOF_KEY_COLUMNS,
            "y_true",
            INCUMBENT_COLUMN,
            TCN_COLUMN,
        ]
        missing = set(required) - set(self.frame)
        if missing:
            raise ValueError(f"outer TCN source is missing columns: {sorted(missing)}")
        frame = self.frame.loc[:, required].copy()
        frame["well_id"] = frame["well_id"].astype(str)
        validate_oof_frame(frame, (INCUMBENT_COLUMN, TCN_COLUMN))
        for name, value in (
            ("oof_sha256", self.oof_sha256),
            ("manifest_sha256", self.manifest_sha256),
        ):
            text = str(value)
            if len(text) != 64 or any(
                character not in "0123456789abcdef" for character in text
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        artifacts = tuple(map(str, self.artifact_ids))
        if not artifacts or len(set(artifacts)) != len(artifacts):
            raise ValueError("artifact_ids must be non-empty and unique")
        if not isinstance(self.evidence_eligible, bool):
            raise TypeError("evidence_eligible must be bool")
        object.__setattr__(
            self,
            "frame",
            frame.sort_values(list(OOF_KEY_COLUMNS), kind="stable").reset_index(
                drop=True
            ),
        )
        object.__setattr__(self, "source_root", str(self.source_root))
        object.__setattr__(self, "artifact_ids", artifacts)
        object.__setattr__(self, "audit", _jsonable(dict(self.audit)))


def make_smoke_outer_tcn_source(
    frame: pd.DataFrame,
    *,
    source_id: str = "synthetic_smoke_source",
) -> OuterTcnSource:
    """Create an explicitly evidence-ineligible source for tests and smoke runs."""

    identifier = _safe_identifier(source_id, name="source_id")
    required = [
        *OOF_KEY_COLUMNS,
        "y_true",
        INCUMBENT_COLUMN,
        TCN_COLUMN,
    ]
    normalized = frame.loc[:, required].copy()
    normalized["well_id"] = normalized["well_id"].astype(str)
    normalized = normalized.sort_values(
        list(OOF_KEY_COLUMNS), kind="stable"
    ).reset_index(drop=True)
    validation = validate_oof_frame(
        normalized,
        (INCUMBENT_COLUMN, TCN_COLUMN),
    )
    digest = _canonical_sha256(
        {
            "source_id": identifier,
            "validation": validation,
            "value_hashes": {
                column: (
                    _string_sequence_sha256(normalized[column].astype(str).to_numpy())
                    if column == "well_id"
                    else _array_sha256(normalized[column].to_numpy())
                )
                for column in required
            },
        }
    )
    return OuterTcnSource(
        frame=normalized,
        source_root=f"smoke://{identifier}",
        oof_sha256=digest,
        manifest_sha256=digest,
        artifact_ids=(f"smoke-source:{identifier}:{digest}",),
        audit={
            "passed": True,
            "evidence_eligible": False,
            "validation": validation,
        },
        evidence_eligible=False,
    )


def _locked_incumbent_reference(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        frame, record = load_audited_incumbent_reference(path)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise NestedExternalEnsembleError(
            "incumbent independent re-audit is absent, stale, or failed"
        ) from exc
    frame["well_id"] = frame["well_id"].astype(str)
    return frame, record


def load_production_outer_tcn_source(
    source_root: Path,
    *,
    incumbent_oof: Path,
    data_root: Path,
    fold_config: Path | None = None,
    bundle_name: str = DEFAULT_BUNDLE_NAME,
) -> OuterTcnSource:
    """Load only a completed, canonical, independently re-audited full TCN OOF."""

    root = Path(source_root).resolve()
    locked, incumbent_record = _locked_incumbent_reference(incumbent_oof)
    audit = independent_reaudit_external_artifacts(
        root,
        bundle_name=bundle_name,
        expected_reference=locked,
        data_root=data_root,
    )
    if audit.get("passed") is not True or audit.get("profile") != "full":
        raise NestedExternalEnsembleError(
            "outer TCN source is not a passed full canonical artifact bundle"
        )
    bundle = read_external_artifact_bundle(
        root,
        bundle_name=bundle_name,
        expected_reference=locked,
    )
    manifest = bundle.manifest
    source_fold_path = (
        Path(manifest["fold_config"]["path"]).resolve()
        if fold_config is None
        else Path(fold_config).resolve()
    )
    if not source_fold_path.is_file() or sha256_file(source_fold_path) != str(
        manifest["fold_config"]["sha256"]
    ):
        raise NestedExternalEnsembleError(
            "requested fold config differs from the canonical TCN source"
        )
    source_mapping = load_fold_mapping(
        source_fold_path,
        expected_well_ids=locked["well_id"].astype(str).unique(),
    )
    candidates = [
        candidate
        for candidate in manifest["oof"]["candidates"]
        if candidate.get("candidate_id") == "tcn"
    ]
    if len(candidates) != 1:
        raise NestedExternalEnsembleError(
            "canonical source must contain exactly one unshrunk TCN candidate"
        )
    candidate = candidates[0]
    model_ids = tuple(map(str, candidate.get("source_model_ids", ())))
    if (
        candidate.get("prediction_column") != CANONICAL_TCN_COLUMN
        or candidate.get("family_id") != FAMILY_TCN
        or candidate.get("control_id") != "none"
        or candidate.get("transfer_mode") != "architecture_only"
        or candidate.get("eligible_for_model_selection") is not True
        or int(candidate.get("identity_fallback_rows", -1)) != 0
        or len(model_ids) != FOLD_COUNT * len(PRODUCTION_TCN_SEEDS)
        or len(set(model_ids)) != len(model_ids)
    ):
        raise NestedExternalEnsembleError(
            "canonical TCN candidate provenance differs from production policy"
        )
    by_id = {
        str(record["model_id"]): record
        for record in manifest["model_artifacts"]
        if str(record.get("model_id", "")) in set(model_ids)
    }
    if set(by_id) != set(model_ids):
        raise NestedExternalEnsembleError(
            "canonical TCN candidate does not resolve every source model"
        )
    expected_pairs = {
        (fold, seed) for fold in range(FOLD_COUNT) for seed in PRODUCTION_TCN_SEEDS
    }
    actual_pairs: set[tuple[int, int]] = set()
    for model_id in model_ids:
        record = by_id[model_id]
        pair = (int(record["outer_fold"]), int(record["seed"]))
        actual_pairs.add(pair)
        outer_training, _ = partition_well_ids(source_mapping, pair[0])
        checkpoint = root / str(record["path"])
        state = load_torch_family(checkpoint)
        expected_config = _tcn_config(
            seed=pair[1],
            device=str(state.config.device),
            epochs=PRODUCTION_EPOCHS,
            hidden_dim=PRODUCTION_HIDDEN_DIM,
            layers=PRODUCTION_LAYERS,
        )
        if (
            record.get("family_id") != FAMILY_TCN
            or record.get("fit_scope") != "outer_training"
            or record.get("inner_fold") is not None
            or tuple(map(str, record.get("fit_well_ids", ()))) != outer_training
            or state.config != expected_config
            or state.channel_names != MODEL_INPUT_CHANNEL_NAMES
            or state.manifest != record.get("state_manifest")
        ):
            raise NestedExternalEnsembleError(
                f"outer TCN checkpoint provenance mismatch: {model_id}"
            )
    if actual_pairs != expected_pairs:
        raise NestedExternalEnsembleError(
            "outer TCN checkpoints do not cover the exact fold/seed product"
        )

    source = bundle.oof.loc[
        :,
        [*OOF_KEY_COLUMNS, "y_true", INCUMBENT_COLUMN, CANONICAL_TCN_COLUMN],
    ].rename(columns={CANONICAL_TCN_COLUMN: TCN_COLUMN})
    validate_oof_frame(
        source,
        (INCUMBENT_COLUMN, TCN_COLUMN),
        expected_keys=locked,
        expected_well_ids=tuple(sorted(locked["well_id"].astype(str).unique())),
        expected_folds=range(FOLD_COUNT),
    )
    manifest_path = bundle.bundle_root / "manifest.json"
    return OuterTcnSource(
        frame=source,
        source_root=str(root),
        oof_sha256=str(manifest["oof"]["sha256"]),
        manifest_sha256=sha256_file(manifest_path),
        artifact_ids=(
            f"outer-tcn-bundle:{manifest['run_id']}:{sha256_file(manifest_path)}",
            f"locked-incumbent:{incumbent_record['sha256']}",
        ),
        audit={
            "bundle_reaudit": audit,
            "incumbent": incumbent_record,
            "candidate": candidate,
            "source_models": len(model_ids),
        },
        evidence_eligible=True,
    )


@dataclass(frozen=True)
class NestedRunPolicy:
    """Immutable training and evaluation policy for one runner invocation."""

    evidence_mode: str = "production"
    outer_folds: tuple[int, ...] = tuple(range(FOLD_COUNT))
    tcn_seeds: tuple[int, ...] = PRODUCTION_TCN_SEEDS
    incumbent_seeds: tuple[int, ...] = DEFAULT_INCUMBENT_SEEDS
    grid_config: FixedGridConfig = PRODUCTION_GRID
    target_config: SmoothTargetConfig = PRODUCTION_TARGET
    epochs: int = PRODUCTION_EPOCHS
    hidden_dim: int = PRODUCTION_HIDDEN_DIM
    layers: int = PRODUCTION_LAYERS
    bootstrap_samples: int = 10_000
    shadow_samples: int = 100_000
    shadow_subset_wells: int = 200

    def __post_init__(self) -> None:
        mode = str(self.evidence_mode)
        if mode not in {"production", "smoke"}:
            raise ValueError("evidence_mode must be 'production' or 'smoke'")
        outer = tuple(map(int, self.outer_folds))
        tcn_seeds = tuple(map(int, self.tcn_seeds))
        incumbent_seeds = tuple(map(int, self.incumbent_seeds))
        if (
            not outer
            or len(set(outer)) != len(outer)
            or any(fold not in range(FOLD_COUNT) for fold in outer)
        ):
            raise ValueError("outer_folds must be unique values from 0..4")
        if (
            not tcn_seeds
            or len(set(tcn_seeds)) != len(tcn_seeds)
            or any(seed < 0 for seed in tcn_seeds)
            or not incumbent_seeds
            or len(set(incumbent_seeds)) != len(incumbent_seeds)
            or any(seed < 0 for seed in incumbent_seeds)
        ):
            raise ValueError("TCN/incumbent seeds must be unique and non-negative")
        for name in (
            "epochs",
            "hidden_dim",
            "layers",
            "bootstrap_samples",
            "shadow_samples",
            "shadow_subset_wells",
        ):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if mode == "production" and (
            outer != tuple(range(FOLD_COUNT))
            or tcn_seeds != PRODUCTION_TCN_SEEDS
            or incumbent_seeds != DEFAULT_INCUMBENT_SEEDS
            or self.grid_config != PRODUCTION_GRID
            or self.target_config != PRODUCTION_TARGET
            or self.epochs != PRODUCTION_EPOCHS
            or self.hidden_dim != PRODUCTION_HIDDEN_DIM
            or self.layers != PRODUCTION_LAYERS
            or self.bootstrap_samples != 10_000
            or self.shadow_samples != 100_000
            or self.shadow_subset_wells != 200
        ):
            raise ValueError(
                "production policy is locked to all folds, five matched seeds, "
                "the full TCN/grid/target configuration, and fixed audit samples"
            )
        object.__setattr__(self, "evidence_mode", mode)
        object.__setattr__(self, "outer_folds", outer)
        object.__setattr__(self, "tcn_seeds", tcn_seeds)
        object.__setattr__(self, "incumbent_seeds", incumbent_seeds)

    @property
    def model_fit_safe(self) -> bool:
        return self.evidence_mode == "production"

    @property
    def weight_fit_safe(self) -> bool:
        return self.evidence_mode == "production"

    @property
    def family_selection_safe(self) -> bool:
        # This runner was added after TCN had already been selected from global
        # quick/full OOF.  It can cross-fit weights safely, but it cannot turn
        # that earlier family selection into nested evidence after the fact.
        return False

    @property
    def evidence_eligible(self) -> bool:
        return (
            self.model_fit_safe and self.weight_fit_safe and self.family_selection_safe
        )


class _SequenceProvider:
    def __init__(
        self,
        train_by_id: Mapping[str, WellFiles],
        *,
        cache_root: Path,
        config: FixedGridConfig,
    ) -> None:
        self.train_by_id = dict(train_by_id)
        self.cache_root = Path(cache_root)
        self.config = config
        self.values: dict[str, FixedGridSequenceWell] = {}

    def get(self, well_id: str) -> FixedGridSequenceWell:
        identifier = str(well_id)
        if identifier not in self.train_by_id:
            raise KeyError(f"unknown train well: {identifier}")
        if identifier not in self.values:
            self.values[identifier] = prepare_fixed_grid_well(
                self.train_by_id[identifier],
                split="train",
                cache_root=self.cache_root,
                config=self.config,
            )
        return self.values[identifier]


def _training_target(well: WellFiles, row_index: np.ndarray) -> np.ndarray:
    """Load labels for an explicitly authorized nested-training well only."""

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
        raise NestedExternalEnsembleError(
            f"training target row alignment differs: {well.well_id}"
        )
    target = frame["TVT"].iloc[rows].to_numpy(dtype=np.float64)
    if target.shape != rows.shape or not np.isfinite(target).all():
        raise NestedExternalEnsembleError(f"training target is invalid: {well.well_id}")
    return target


def _prediction_frame(
    *,
    inference: SequenceModelInput,
    fold: int,
    grid_prediction: np.ndarray,
    target_config: SmoothTargetConfig,
    prediction_column: str,
) -> pd.DataFrame:
    components = grid_prediction_to_row_components(
        grid_prediction,
        inference,
        config=target_config,
    )
    validate_residual_components(
        components,
        inference.incumbent_prediction,
        inference.row_coordinate,
    )
    return pd.DataFrame(
        {
            "well_id": inference.well_id,
            "row_index": inference.row_index,
            "fold": int(fold),
            prediction_column: components.final,
            f"{prediction_column}__common": components.common,
            f"{prediction_column}__slope": components.slope,
            f"{prediction_column}__shape": components.shape,
        }
    )


def _same_key_order(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return all(
        np.array_equal(
            left[column].astype(str).to_numpy()
            if column == "well_id"
            else left[column].to_numpy(dtype=np.int64),
            right[column].astype(str).to_numpy()
            if column == "well_id"
            else right[column].to_numpy(dtype=np.int64),
        )
        for column in OOF_KEY_COLUMNS
    )


@dataclass(frozen=True)
class InnerSplitArtifact:
    outer_fold: int
    inner_fold: int
    frame: pd.DataFrame
    manifest: Mapping[str, Any]
    oof_path: Path
    manifest_path: Path

    @property
    def artifact_id(self) -> str:
        return (
            f"inner-tcn-oof:outer_{self.outer_fold}:inner_{self.inner_fold}:"
            f"{self.manifest['oof_sha256']}"
        )


def _checkpoint_paths(
    run_root: Path,
    *,
    outer_fold: int,
    inner_fold: int,
    seed: int,
) -> tuple[Path, Path]:
    checkpoint = (
        run_root
        / "models"
        / "tcn"
        / f"outer_{outer_fold}"
        / f"inner_{inner_fold}"
        / f"seed_{seed}.pt"
    )
    return checkpoint, checkpoint.with_suffix(".json")


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as error:
        raise NestedExternalEnsembleError(
            f"artifact path is outside the run root: {path}"
        ) from error


def _checkpoint_provenance(
    *,
    outer_fold: int,
    inner_fold: int,
    seed: int,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
    excluded_outer_ids: Sequence[str],
    x: np.ndarray,
    targets: np.ndarray,
    row_counts: np.ndarray,
    config: TorchFamilyConfig,
    incumbent_checkpoint_sha256: Sequence[str],
    data_manifest: Mapping[str, Any],
    implementation_hashes: Mapping[str, str],
    grid_config: FixedGridConfig,
    target_config: SmoothTargetConfig,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY_TCN,
        "fit_scope": "inner_training",
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "seed": int(seed),
        "training_well_ids": list(map(str, training_ids)),
        "training_ids_sha256": well_ids_sha256(tuple(map(str, training_ids))),
        "validation_well_ids": list(map(str, validation_ids)),
        "excluded_outer_well_ids": list(map(str, excluded_outer_ids)),
        "training_validation_intersection": sorted(
            set(map(str, training_ids)) & set(map(str, validation_ids))
        ),
        "training_outer_intersection": sorted(
            set(map(str, training_ids)) & set(map(str, excluded_outer_ids))
        ),
        "config": asdict(config),
        "channel_names": list(MODEL_INPUT_CHANNEL_NAMES),
        "x_sha256": _array_sha256(x),
        "targets_sha256": _array_sha256(targets),
        "row_counts_sha256": _array_sha256(row_counts),
        "row_count_sum": int(np.sum(row_counts)),
        "incumbent_checkpoint_sha256": list(map(str, incumbent_checkpoint_sha256)),
        "data_manifest": _jsonable(data_manifest),
        "implementation_hashes": _jsonable(implementation_hashes),
        "fixed_grid": fixed_grid_manifest(grid_config),
        "residual_target": target_manifest(target_config),
        "labels_loaded_only_for_training_ids": True,
        "outer_validation_labels_used_for_fit": False,
        "inner_validation_labels_used_for_fit": False,
    }


def _validate_tcn_checkpoint_record(
    record: Mapping[str, Any],
    *,
    run_root: Path,
    expected_provenance: Mapping[str, Any] | None = None,
) -> TorchFamilyState:
    required = {
        "checkpoint",
        "sidecar",
        "checkpoint_sha256",
        "sidecar_sha256",
        "provenance",
        "state_manifest",
        "resumed",
    }
    if not required.issubset(record):
        raise NestedExternalEnsembleError(
            f"TCN checkpoint record is incomplete: {sorted(required - set(record))}"
        )
    checkpoint = run_root / str(record["checkpoint"])
    sidecar = run_root / str(record["sidecar"])
    if not checkpoint.is_file() or not sidecar.is_file():
        raise NestedExternalEnsembleError(
            "TCN checkpoint/sidecar pair is partial or missing"
        )
    if sha256_file(checkpoint) != str(record["checkpoint_sha256"]) or sha256_file(
        sidecar
    ) != str(record["sidecar_sha256"]):
        raise NestedExternalEnsembleError(
            "TCN checkpoint or sidecar bytes differ from recorded SHA-256"
        )
    sidecar_payload = _read_json(sidecar)
    if (
        sidecar_payload.get("checkpoint_sha256") != str(record["checkpoint_sha256"])
        or sidecar_payload.get("provenance") != record["provenance"]
        or sidecar_payload.get("state_manifest") != record["state_manifest"]
    ):
        raise NestedExternalEnsembleError(
            "TCN checkpoint sidecar differs from the OOF manifest"
        )
    reciprocal = record.get("reciprocal_state_reuse")
    if sidecar_payload.get("reciprocal_state_reuse") != reciprocal:
        raise NestedExternalEnsembleError(
            "TCN reciprocal reuse provenance differs from its sidecar"
        )
    if reciprocal is not None:
        if not isinstance(reciprocal, Mapping):
            raise NestedExternalEnsembleError(
                "TCN reciprocal reuse provenance must be a mapping"
            )
        source_checkpoint = run_root / str(reciprocal.get("source_checkpoint", ""))
        source_sidecar = run_root / str(reciprocal.get("source_sidecar", ""))
        if (
            not source_checkpoint.is_file()
            or not source_sidecar.is_file()
            or sha256_file(source_checkpoint)
            != str(reciprocal.get("source_checkpoint_sha256", ""))
            or sha256_file(source_sidecar)
            != str(reciprocal.get("source_sidecar_sha256", ""))
        ):
            raise NestedExternalEnsembleError(
                "TCN reciprocal source checkpoint/sidecar is stale"
            )
        source_payload = _read_json(source_sidecar)
        source_provenance = source_payload.get("provenance", {})
        current_provenance = record["provenance"]
        invariant_fields = (
            "schema_version",
            "family",
            "fit_scope",
            "seed",
            "training_well_ids",
            "training_ids_sha256",
            "config",
            "channel_names",
            "x_sha256",
            "targets_sha256",
            "row_counts_sha256",
            "row_count_sum",
            "data_manifest",
            "implementation_hashes",
            "fixed_grid",
            "residual_target",
            "labels_loaded_only_for_training_ids",
            "outer_validation_labels_used_for_fit",
            "inner_validation_labels_used_for_fit",
        )
        valid_reciprocal = (
            all(
                source_provenance.get(field) == current_provenance.get(field)
                for field in invariant_fields
            )
            and int(source_provenance.get("outer_fold", -1))
            == int(current_provenance["inner_fold"])
            and int(source_provenance.get("inner_fold", -1))
            == int(current_provenance["outer_fold"])
            and source_provenance.get("validation_well_ids")
            == current_provenance.get("excluded_outer_well_ids")
            and source_provenance.get("excluded_outer_well_ids")
            == current_provenance.get("validation_well_ids")
            and source_payload.get("checkpoint_sha256")
            == reciprocal.get("source_checkpoint_sha256")
            and source_payload.get("state_manifest") == record["state_manifest"]
        )
        if not valid_reciprocal:
            raise NestedExternalEnsembleError(
                "TCN reciprocal source is not an exact training-equivalent split"
            )
    if expected_provenance is not None and record["provenance"] != _jsonable(
        expected_provenance
    ):
        raise NestedExternalEnsembleError(
            "TCN checkpoint provenance differs from current inputs/config"
        )
    state = load_torch_family(checkpoint)
    provenance = record["provenance"]
    config = TorchFamilyConfig(**dict(provenance["config"]))
    if (
        state.config != config
        or state.channel_names != tuple(provenance["channel_names"])
        or state.manifest != record["state_manifest"]
        or state.manifest.get("fit_ids_sha256") != provenance["training_ids_sha256"]
        or int(state.manifest.get("sample_count", -1))
        != len(provenance["training_well_ids"])
        or int(state.manifest.get("row_count_sum", -1))
        != int(provenance["row_count_sum"])
        or int(state.manifest.get("sequence_length", -1))
        != int(provenance["fixed_grid"]["config"]["horizontal_points"])
        or int(state.manifest.get("target_dim", -1))
        != int(provenance["fixed_grid"]["config"]["horizontal_points"])
        or state.manifest.get("family") != FAMILY_TCN
        or provenance["training_validation_intersection"]
        or provenance["training_outer_intersection"]
    ):
        raise NestedExternalEnsembleError(
            "loaded TCN state differs from strict inner-fit provenance"
        )
    return state


def _fit_or_load_tcn(
    path: Path,
    sidecar_path: Path,
    *,
    run_root: Path,
    x: np.ndarray,
    targets: np.ndarray,
    channel_names: Sequence[str],
    training_ids: Sequence[str],
    row_counts: np.ndarray,
    config: TorchFamilyConfig,
    provenance: Mapping[str, Any],
    resume: bool,
    reciprocal_source: tuple[Path, Path] | None = None,
) -> tuple[TorchFamilyState, dict[str, Any]]:
    existing = (path.exists(), sidecar_path.exists())
    if any(existing):
        if not all(existing):
            raise NestedExternalEnsembleError(
                "TCN checkpoint is partial; checkpoint and sidecar are both required"
            )
        if not resume:
            raise FileExistsError(f"TCN checkpoint already exists: {path}")
        sidecar_payload = _read_json(sidecar_path)
        record = {
            "checkpoint": _relative(path, run_root),
            "sidecar": _relative(sidecar_path, run_root),
            "checkpoint_sha256": sha256_file(path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "provenance": sidecar_payload.get("provenance"),
            "state_manifest": sidecar_payload.get("state_manifest"),
            "reciprocal_state_reuse": sidecar_payload.get("reciprocal_state_reuse"),
            "resumed": True,
        }
        state = _validate_tcn_checkpoint_record(
            record,
            run_root=run_root,
            expected_provenance=provenance,
        )
        return state, record

    if reciprocal_source is not None:
        source_path, source_sidecar_path = reciprocal_source
        reciprocal_existing = (
            source_path.exists(),
            source_sidecar_path.exists(),
        )
        if any(reciprocal_existing) and not all(reciprocal_existing):
            raise NestedExternalEnsembleError(
                "reciprocal TCN checkpoint is partial and cannot be reused"
            )
        if all(reciprocal_existing):
            source_payload = _read_json(source_sidecar_path)
            source_record = {
                "checkpoint": _relative(source_path, run_root),
                "sidecar": _relative(source_sidecar_path, run_root),
                "checkpoint_sha256": sha256_file(source_path),
                "sidecar_sha256": sha256_file(source_sidecar_path),
                "provenance": source_payload.get("provenance"),
                "state_manifest": source_payload.get("state_manifest"),
                "reciprocal_state_reuse": source_payload.get("reciprocal_state_reuse"),
                "resumed": True,
            }
            source_state = _validate_tcn_checkpoint_record(
                source_record,
                run_root=run_root,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, path)
            reciprocal_record = {
                "source_checkpoint": _relative(source_path, run_root),
                "source_sidecar": _relative(source_sidecar_path, run_root),
                "source_checkpoint_sha256": sha256_file(source_path),
                "source_sidecar_sha256": sha256_file(source_sidecar_path),
                "equivalence_basis": (
                    "reciprocal split has identical ordered training IDs, "
                    "model-input bytes, residual-target bytes, row weights, "
                    "seed, config, data manifest, and implementation hashes"
                ),
            }
            sidecar_payload = {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_sha256": sha256_file(path),
                "provenance": _jsonable(provenance),
                "state_manifest": source_state.manifest,
                "reciprocal_state_reuse": reciprocal_record,
            }
            _atomic_json(sidecar_path, sidecar_payload)
            record = {
                "checkpoint": _relative(path, run_root),
                "sidecar": _relative(sidecar_path, run_root),
                "checkpoint_sha256": sha256_file(path),
                "sidecar_sha256": sha256_file(sidecar_path),
                "provenance": _jsonable(provenance),
                "state_manifest": source_state.manifest,
                "reciprocal_state_reuse": reciprocal_record,
                "resumed": False,
            }
            state = _validate_tcn_checkpoint_record(
                record,
                run_root=run_root,
                expected_provenance=provenance,
            )
            return state, record

    state = fit_torch_family(
        x,
        targets,
        channel_names=channel_names,
        ids=training_ids,
        row_counts=row_counts,
        config=config,
    )
    state.save(path)
    restored = load_torch_family(path)
    if restored.manifest != state.manifest or restored.config != state.config:
        raise NestedExternalEnsembleError(
            "TCN state changed during checkpoint round trip"
        )
    sidecar_payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": sha256_file(path),
        "provenance": _jsonable(provenance),
        "state_manifest": restored.manifest,
        "reciprocal_state_reuse": None,
    }
    _atomic_json(sidecar_path, sidecar_payload)
    record = {
        "checkpoint": _relative(path, run_root),
        "sidecar": _relative(sidecar_path, run_root),
        "checkpoint_sha256": sha256_file(path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "provenance": _jsonable(provenance),
        "state_manifest": restored.manifest,
        "reciprocal_state_reuse": None,
        "resumed": False,
    }
    _validate_tcn_checkpoint_record(
        record,
        run_root=run_root,
        expected_provenance=provenance,
    )
    return restored, record


def _validate_incumbent_checkpoint_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    outer_fold: int,
    inner_fold: int,
) -> None:
    if tuple(int(record["seed"]) for record in records) != tuple(
        map(int, expected_seeds)
    ):
        raise NestedExternalEnsembleError(
            "nested incumbent checkpoint seed order differs"
        )
    for record in records:
        checkpoint = Path(str(record["checkpoint_path"]))
        manifest_path = Path(str(record["manifest_path"]))
        if not checkpoint.is_file() or not manifest_path.is_file():
            raise NestedExternalEnsembleError(
                "nested incumbent checkpoint/manifest pair is missing"
            )
        manifest = _read_json(manifest_path)
        provenance = record["provenance"]
        if (
            sha256_file(checkpoint) != str(record["checkpoint_sha256"])
            or manifest.get("checkpoint_sha256") != str(record["checkpoint_sha256"])
            or manifest.get("provenance") != provenance
            or int(provenance["outer_fold"]) != int(outer_fold)
            or int(provenance["inner_validation_fold"]) != int(inner_fold)
        ):
            raise NestedExternalEnsembleError(
                "nested incumbent checkpoint provenance is stale or corrupt"
            )
        source_path = record.get("reciprocal_source_checkpoint_path")
        source_sha = record.get("reciprocal_source_checkpoint_sha256")
        source_manifest_path = record.get("reciprocal_source_manifest_path")
        source_manifest_sha = record.get("reciprocal_source_manifest_sha256")
        source_fields = (
            source_path,
            source_sha,
            source_manifest_path,
            source_manifest_sha,
        )
        if any(value is None for value in source_fields) and not all(
            value is None for value in source_fields
        ):
            raise NestedExternalEnsembleError(
                "nested incumbent reciprocal source artifact record is partial"
            )
        if source_path is not None:
            source = Path(str(source_path))
            source_manifest_path = Path(str(source_manifest_path))
            source_manifest = (
                _read_json(source_manifest_path)
                if source_manifest_path.is_file()
                else {}
            )
            source_provenance = source_manifest.get("provenance", {})
            if (
                not source.is_file()
                or sha256_file(source) != str(source_sha)
                or not source_manifest_path.is_file()
                or sha256_file(source_manifest_path) != str(source_manifest_sha)
                or int(source_provenance.get("outer_fold", -1)) != int(inner_fold)
                or int(source_provenance.get("inner_validation_fold", -1))
                != int(outer_fold)
                or int(source_provenance.get("seed", -1)) != int(record["seed"])
                or source_provenance.get("ordered_training_well_ids")
                != provenance.get("ordered_training_well_ids")
                or int(provenance["outer_fold"]) != int(outer_fold)
                or int(provenance["inner_validation_fold"]) != int(inner_fold)
            ):
                raise NestedExternalEnsembleError(
                    "nested incumbent reciprocal source is stale"
                )


def _inner_paths(
    run_root: Path,
    *,
    outer_fold: int,
    inner_fold: int,
) -> tuple[Path, Path]:
    directory = run_root / "inner_oof" / f"outer_{outer_fold}" / f"inner_{inner_fold}"
    return directory / "oof_predictions.csv.gz", directory / "manifest.json"


def _inner_expected_manifest_core(
    *,
    outer_fold: int,
    inner_fold: int,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
    excluded_outer_ids: Sequence[str],
    policy_digest: str,
    run_config_digest: str,
    source: OuterTcnSource,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "inner_oof",
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "training_well_ids": list(map(str, training_ids)),
        "training_ids_sha256": well_ids_sha256(tuple(map(str, training_ids))),
        "validation_well_ids": list(map(str, validation_ids)),
        "validation_ids_sha256": well_ids_sha256(tuple(map(str, validation_ids))),
        "excluded_outer_well_ids": list(map(str, excluded_outer_ids)),
        "excluded_outer_ids_sha256": well_ids_sha256(
            tuple(map(str, excluded_outer_ids))
        ),
        "training_validation_intersection": sorted(
            set(map(str, training_ids)) & set(map(str, validation_ids))
        ),
        "training_outer_intersection": sorted(
            set(map(str, training_ids)) & set(map(str, excluded_outer_ids))
        ),
        "validation_outer_intersection": sorted(
            set(map(str, validation_ids)) & set(map(str, excluded_outer_ids))
        ),
        "policy_digest": policy_digest,
        "run_config_digest": run_config_digest,
        "source_oof_sha256": source.oof_sha256,
        "truth_attached_only_after_all_seed_predictions": True,
        "outer_validation_labels_used_for_fit": False,
        "inner_validation_labels_used_for_fit": False,
    }


def _validate_family_mean(frame: pd.DataFrame, seeds: Sequence[int]) -> None:
    seed_columns = [f"{TCN_COLUMN}__seed_{seed}" for seed in seeds]
    expected = frame[seed_columns].to_numpy(dtype=np.float64).mean(axis=1)
    actual = frame[TCN_COLUMN].to_numpy(dtype=np.float64)
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-10):
        raise NestedExternalEnsembleError(
            "TCN family prediction is not the row-level mean of every seed"
        )
    for component in ("common", "slope", "shape"):
        members = [f"{TCN_COLUMN}__seed_{seed}__{component}" for seed in seeds]
        expected_component = frame[members].to_numpy(dtype=np.float64).mean(axis=1)
        actual_component = frame[f"{TCN_COLUMN}__{component}"].to_numpy(
            dtype=np.float64
        )
        if not np.allclose(
            actual_component,
            expected_component,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise NestedExternalEnsembleError(
                f"TCN {component} is not the row-level mean of every seed"
            )
    reconstructed = (
        frame[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
        + frame[f"{TCN_COLUMN}__common"].to_numpy(dtype=np.float64)
        + frame[f"{TCN_COLUMN}__slope"].to_numpy(dtype=np.float64)
        + frame[f"{TCN_COLUMN}__shape"].to_numpy(dtype=np.float64)
    )
    if not np.allclose(
        frame[TCN_COLUMN].to_numpy(dtype=np.float64),
        reconstructed,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise NestedExternalEnsembleError(
            "TCN family mean prediction/component algebra is inconsistent"
        )


def _load_inner_split(
    *,
    run_root: Path,
    outer_fold: int,
    inner_fold: int,
    source_truth: pd.DataFrame,
    expected_core: Mapping[str, Any],
    tcn_seeds: Sequence[int],
    incumbent_seeds: Sequence[int],
) -> InnerSplitArtifact:
    oof_path, manifest_path = _inner_paths(
        run_root,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    if not oof_path.is_file() or not manifest_path.is_file():
        raise NestedExternalEnsembleError(
            "inner OOF artifact is partial; CSV and manifest are both required"
        )
    manifest = _read_json(manifest_path)
    for key, value in expected_core.items():
        if manifest.get(key) != _jsonable(value):
            raise NestedExternalEnsembleError(
                f"inner OOF manifest field differs: {key}"
            )
    if (
        manifest.get("completed") is not True
        or manifest.get("oof_sha256") != sha256_file(oof_path)
        or int(manifest.get("rows", -1)) <= 0
    ):
        raise NestedExternalEnsembleError(
            "inner OOF artifact is incomplete or checksum-mismatched"
        )
    frame = _read_oof(oof_path)
    prediction_columns = [
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        *(f"{TCN_COLUMN}__seed_{seed}" for seed in tcn_seeds),
        *(f"{TCN_COLUMN}__{component}" for component in ("common", "slope", "shape")),
        *(
            f"{TCN_COLUMN}__seed_{seed}__{component}"
            for seed in tcn_seeds
            for component in ("common", "slope", "shape")
        ),
    ]
    validation = validate_oof_frame(
        frame,
        prediction_columns,
        expected_keys=source_truth,
        expected_well_ids=expected_core["validation_well_ids"],
        expected_folds=(inner_fold,),
    )
    if manifest.get("validation") != validation:
        raise NestedExternalEnsembleError(
            "inner OOF validation differs from its manifest"
        )
    _validate_family_mean(frame, tcn_seeds)
    incumbent_records = manifest.get("incumbent_checkpoints")
    tcn_records = manifest.get("tcn_checkpoints")
    if not isinstance(incumbent_records, list) or not isinstance(tcn_records, list):
        raise NestedExternalEnsembleError(
            "inner OOF lacks incumbent or TCN checkpoint provenance"
        )
    _validate_incumbent_checkpoint_records(
        incumbent_records,
        expected_seeds=incumbent_seeds,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    if tuple(int(record["provenance"]["seed"]) for record in tcn_records) != tuple(
        map(int, tcn_seeds)
    ):
        raise NestedExternalEnsembleError("inner TCN checkpoint seed order differs")
    for record in tcn_records:
        _validate_tcn_checkpoint_record(record, run_root=run_root)
    return InnerSplitArtifact(
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        frame=frame,
        manifest=manifest,
        oof_path=oof_path,
        manifest_path=manifest_path,
    )


def _build_inner_split(
    *,
    run_root: Path,
    refitter: NestedHeelIncumbentRefitter,
    sequence_provider: _SequenceProvider,
    train_by_id: Mapping[str, WellFiles],
    source: OuterTcnSource,
    outer_fold: int,
    inner_fold: int,
    policy: NestedRunPolicy,
    policy_digest: str,
    run_config_digest: str,
    data_manifest: Mapping[str, Any],
    implementation_hashes: Mapping[str, str],
    device: str,
    resume: bool,
) -> InnerSplitArtifact:
    spec = refitter.fold_spec(
        outer_fold=outer_fold,
        inner_validation_fold=inner_fold,
    )
    expected_core = _inner_expected_manifest_core(
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        training_ids=spec.training_well_ids,
        validation_ids=spec.validation_well_ids,
        excluded_outer_ids=spec.excluded_outer_well_ids,
        policy_digest=policy_digest,
        run_config_digest=run_config_digest,
        source=source,
    )
    if any(
        expected_core[name]
        for name in (
            "training_validation_intersection",
            "training_outer_intersection",
            "validation_outer_intersection",
        )
    ):
        raise NestedExternalEnsembleError("inner split groups are not disjoint")
    oof_path, manifest_path = _inner_paths(
        run_root,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    existing = (oof_path.exists(), manifest_path.exists())
    source_truth = source.frame.loc[
        source.frame["well_id"].isin(spec.validation_well_ids),
        [*OOF_KEY_COLUMNS, "y_true"],
    ].copy()
    if any(existing):
        if not all(existing):
            raise NestedExternalEnsembleError(
                "inner OOF is partial and will not be overwritten"
            )
        if not resume:
            raise FileExistsError(f"inner OOF already exists: {oof_path}")
        return _load_inner_split(
            run_root=run_root,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            source_truth=source_truth,
            expected_core=expected_core,
            tcn_seeds=policy.tcn_seeds,
            incumbent_seeds=policy.incumbent_seeds,
        )

    requested = (*spec.training_well_ids, *spec.validation_well_ids)
    scoped = refitter.predict_wells(
        outer_fold=outer_fold,
        inner_validation_fold=inner_fold,
        well_ids=requested,
        allowed_scope="inner_training_or_validation",
        resume=resume,
    )
    by_prediction = {
        prediction.well_id: prediction for prediction in scoped.predictions
    }
    if tuple(by_prediction) != requested:
        raise NestedExternalEnsembleError(
            "nested incumbent prediction order differs from requested IDs"
        )
    model_inputs: dict[str, SequenceModelInput] = {}
    for well_id in requested:
        sequence = sequence_provider.get(well_id)
        prediction = by_prediction[well_id]
        if not np.array_equal(sequence.row_index, prediction.row_index):
            raise NestedExternalEnsembleError(
                f"fixed-grid/incumbent row keys differ: {well_id}"
            )
        model_inputs[well_id] = build_sequence_model_input(
            sequence,
            prediction.prediction,
            grid_points=policy.grid_config.horizontal_points,
        )

    training_x: list[np.ndarray] = []
    training_targets: list[np.ndarray] = []
    row_counts: list[int] = []
    for well_id in spec.training_well_ids:
        inference = model_inputs[well_id]
        target = _training_target(train_by_id[well_id], inference.row_index)
        training = ResidualTrainingWell(
            inference=inference,
            incumbent_prediction=inference.incumbent_prediction,
            target=target,
        )
        training_x.append(inference.x)
        training_targets.append(
            make_grid_residual_target(
                training,
                config=policy.target_config,
            ).correction
        )
        row_counts.append(len(inference.row_index))
    x = np.stack(training_x).astype(np.float32)
    targets = np.stack(training_targets).astype(np.float32)
    counts = np.asarray(row_counts, dtype=np.int64)
    if (
        x.shape
        != (
            len(spec.training_well_ids),
            len(MODEL_INPUT_CHANNEL_NAMES),
            policy.grid_config.horizontal_points,
        )
        or targets.shape
        != (len(spec.training_well_ids), policy.grid_config.horizontal_points)
        or not np.isfinite(x).all()
        or not np.isfinite(targets).all()
    ):
        raise NestedExternalEnsembleError("nested TCN training arrays are invalid")

    validation_x = np.stack(
        [model_inputs[well_id].x for well_id in spec.validation_well_ids]
    ).astype(np.float32)
    incumbent_checkpoint_sha = tuple(
        record.checkpoint_sha256 for record in scoped.checkpoints
    )
    tcn_records: list[dict[str, Any]] = []
    seed_frames: list[pd.DataFrame] = []
    for seed in policy.tcn_seeds:
        config = _tcn_config(
            seed=seed,
            device=device,
            epochs=policy.epochs,
            hidden_dim=policy.hidden_dim,
            layers=policy.layers,
        )
        provenance = _checkpoint_provenance(
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            seed=seed,
            training_ids=spec.training_well_ids,
            validation_ids=spec.validation_well_ids,
            excluded_outer_ids=spec.excluded_outer_well_ids,
            x=x,
            targets=targets,
            row_counts=counts,
            config=config,
            incumbent_checkpoint_sha256=incumbent_checkpoint_sha,
            data_manifest=data_manifest,
            implementation_hashes=implementation_hashes,
            grid_config=policy.grid_config,
            target_config=policy.target_config,
        )
        checkpoint, sidecar = _checkpoint_paths(
            run_root,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            seed=seed,
        )
        reciprocal_checkpoint = _checkpoint_paths(
            run_root,
            outer_fold=inner_fold,
            inner_fold=outer_fold,
            seed=seed,
        )
        state, record = _fit_or_load_tcn(
            checkpoint,
            sidecar,
            run_root=run_root,
            x=x,
            targets=targets,
            channel_names=MODEL_INPUT_CHANNEL_NAMES,
            training_ids=spec.training_well_ids,
            row_counts=counts,
            config=config,
            provenance=provenance,
            resume=resume,
            reciprocal_source=reciprocal_checkpoint,
        )
        tcn_records.append(record)
        grid_prediction = state.predict(
            validation_x,
            channel_names=MODEL_INPUT_CHANNEL_NAMES,
            ids=spec.validation_well_ids,
        )
        if grid_prediction.shape != (
            len(spec.validation_well_ids),
            policy.grid_config.horizontal_points,
        ):
            raise NestedExternalEnsembleError(
                "nested TCN prediction grid has an invalid shape"
            )
        seed_column = f"{TCN_COLUMN}__seed_{seed}"
        blocks = [
            _prediction_frame(
                inference=model_inputs[well_id],
                fold=inner_fold,
                grid_prediction=grid_prediction[index],
                target_config=policy.target_config,
                prediction_column=seed_column,
            )
            for index, well_id in enumerate(spec.validation_well_ids)
        ]
        current = (
            pd.concat(blocks, ignore_index=True)
            .sort_values(
                list(OOF_KEY_COLUMNS),
                kind="stable",
            )
            .reset_index(drop=True)
        )
        if seed_frames and not _same_key_order(seed_frames[0], current):
            raise NestedExternalEnsembleError("nested TCN seed prediction keys differ")
        seed_frames.append(current)

    base = seed_frames[0].loc[:, list(OOF_KEY_COLUMNS)].copy()
    for current in seed_frames:
        for column in current.columns:
            if column not in OOF_KEY_COLUMNS:
                base[column] = current[column].to_numpy(dtype=np.float64)
    for component in (None, "common", "slope", "shape"):
        members = [
            f"{TCN_COLUMN}__seed_{seed}"
            + ("" if component is None else f"__{component}")
            for seed in policy.tcn_seeds
        ]
        destination = TCN_COLUMN + ("" if component is None else f"__{component}")
        base[destination] = base[members].to_numpy(dtype=np.float64).mean(axis=1)

    incumbent_blocks = [
        pd.DataFrame(
            {
                "well_id": well_id,
                "row_index": by_prediction[well_id].row_index,
                "fold": int(inner_fold),
                INCUMBENT_COLUMN: by_prediction[well_id].prediction,
            }
        )
        for well_id in spec.validation_well_ids
    ]
    incumbent_frame = (
        pd.concat(incumbent_blocks, ignore_index=True)
        .sort_values(
            list(OOF_KEY_COLUMNS),
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if not _same_key_order(base, incumbent_frame):
        raise NestedExternalEnsembleError(
            "nested TCN and incumbent validation keys differ"
        )
    base.insert(
        len(OOF_KEY_COLUMNS),
        INCUMBENT_COLUMN,
        incumbent_frame[INCUMBENT_COLUMN].to_numpy(dtype=np.float64),
    )

    # Truth is attached only now, after every TCN seed has fitted and predicted.
    source_truth = source_truth.sort_values(
        list(OOF_KEY_COLUMNS),
        kind="stable",
    ).reset_index(drop=True)
    if not _same_key_order(base, source_truth):
        raise NestedExternalEnsembleError(
            "inner validation predictions differ from locked truth keys"
        )
    base.insert(
        len(OOF_KEY_COLUMNS),
        "y_true",
        source_truth["y_true"].to_numpy(dtype=np.float64),
    )
    prediction_columns = [
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        *(f"{TCN_COLUMN}__seed_{seed}" for seed in policy.tcn_seeds),
        *(f"{TCN_COLUMN}__{component}" for component in ("common", "slope", "shape")),
        *(
            f"{TCN_COLUMN}__seed_{seed}__{component}"
            for seed in policy.tcn_seeds
            for component in ("common", "slope", "shape")
        ),
    ]
    validation = validate_oof_frame(
        base,
        prediction_columns,
        expected_keys=source_truth,
        expected_well_ids=spec.validation_well_ids,
        expected_folds=(inner_fold,),
    )
    _validate_family_mean(base, policy.tcn_seeds)
    incumbent_records = _jsonable(
        [
            {
                **asdict(record),
                "provenance": asdict(record.provenance),
            }
            for record in scoped.checkpoints
        ]
    )
    _validate_incumbent_checkpoint_records(
        incumbent_records,
        expected_seeds=policy.incumbent_seeds,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    _atomic_csv(oof_path, base)
    manifest = {
        **expected_core,
        "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(base)),
        "wells": int(base["well_id"].nunique()),
        "oof_path": _relative(oof_path, run_root),
        "oof_sha256": sha256_file(oof_path),
        "validation": validation,
        "prediction_columns": prediction_columns,
        "incumbent_checkpoints": incumbent_records,
        "tcn_checkpoints": tcn_records,
    }
    _atomic_json(manifest_path, manifest)
    return _load_inner_split(
        run_root=run_root,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        source_truth=source_truth,
        expected_core=expected_core,
        tcn_seeds=policy.tcn_seeds,
        incumbent_seeds=policy.incumbent_seeds,
    )


def _outer_paths(run_root: Path, outer_fold: int) -> tuple[Path, Path]:
    directory = run_root / "outer_oof" / f"fold_{outer_fold}"
    return directory / "oof_predictions.csv.gz", directory / "blend.json"


def _outer_frame_from_result(
    result: NestedFoldBlendResult,
) -> pd.DataFrame:
    frame = result.predictions.rename(
        columns={
            "y_pred_family__incumbent": INCUMBENT_COLUMN,
            "y_pred_family__tcn": TCN_COLUMN,
            result.prediction_column: NESTED_COLUMN,
        }
    )
    frame[FIXED_075_COLUMN] = FIXED_INCUMBENT_HEAVY["incumbent"] * frame[
        INCUMBENT_COLUMN
    ].to_numpy(dtype=np.float64) + FIXED_INCUMBENT_HEAVY["tcn"] * frame[
        TCN_COLUMN
    ].to_numpy(dtype=np.float64)
    frame[FIXED_050_COLUMN] = FIXED_SIMPLE_AVERAGE["incumbent"] * frame[
        INCUMBENT_COLUMN
    ].to_numpy(dtype=np.float64) + FIXED_SIMPLE_AVERAGE["tcn"] * frame[
        TCN_COLUMN
    ].to_numpy(dtype=np.float64)
    return frame.loc[
        :,
        [
            *OOF_KEY_COLUMNS,
            "y_true",
            INCUMBENT_COLUMN,
            TCN_COLUMN,
            NESTED_COLUMN,
            FIXED_075_COLUMN,
            FIXED_050_COLUMN,
        ],
    ]


def _persist_or_validate_outer(
    *,
    run_root: Path,
    outer_fold: int,
    result: NestedFoldBlendResult,
    frame: pd.DataFrame,
    provenance: NestedBlendProvenance,
    inner_artifacts: Sequence[InnerSplitArtifact],
    source: OuterTcnSource,
    resume: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    oof_path, blend_path = _outer_paths(run_root, outer_fold)
    prediction_columns = (
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        NESTED_COLUMN,
        FIXED_075_COLUMN,
        FIXED_050_COLUMN,
    )
    validation = validate_oof_frame(
        frame,
        prediction_columns,
        expected_keys=source.frame.loc[source.frame["fold"].eq(outer_fold)],
        expected_well_ids=provenance.outer_validation_well_ids,
        expected_folds=(outer_fold,),
    )
    expected_blend = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "outer_nested_blend",
        "completed": True,
        "outer_fold": int(outer_fold),
        "provenance": asdict(provenance),
        "provenance_digest": provenance.digest,
        "blend": result.summary(),
        "regularization_strength": REGULARIZATION_STRENGTH,
        "regularization_target": REGULARIZATION_TARGET,
        "family_max_weight": FAMILY_MAX_WEIGHT,
        "fixed_controls": {
            FIXED_075_COLUMN: FIXED_INCUMBENT_HEAVY,
            FIXED_050_COLUMN: FIXED_SIMPLE_AVERAGE,
        },
        "inner_artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "oof_path": _relative(artifact.oof_path, run_root),
                "oof_sha256": artifact.manifest["oof_sha256"],
                "manifest_path": _relative(artifact.manifest_path, run_root),
                "manifest_sha256": sha256_file(artifact.manifest_path),
            }
            for artifact in inner_artifacts
        ],
        "source_oof_sha256": source.oof_sha256,
        "validation": validation,
    }
    existing = (oof_path.exists(), blend_path.exists())
    if any(existing):
        if not all(existing):
            raise NestedExternalEnsembleError(
                "outer OOF/blend artifact is partial and will not be overwritten"
            )
        if not resume:
            raise FileExistsError(f"outer OOF already exists: {oof_path}")
        persisted = _read_oof(oof_path)
        manifest = _read_json(blend_path)
        if manifest.get("oof_sha256") != sha256_file(oof_path) or {
            key: manifest.get(key) for key in expected_blend
        } != _jsonable(expected_blend):
            raise NestedExternalEnsembleError(
                "persisted outer blend differs from recomputed provenance"
            )
        persisted = persisted.loc[:, frame.columns].copy()
        if not _same_key_order(frame, persisted):
            raise NestedExternalEnsembleError(
                "persisted outer blend keys differ from recomputed keys"
            )
        for column in ("y_true", *prediction_columns):
            if not np.allclose(
                frame[column].to_numpy(dtype=np.float64),
                persisted[column].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise NestedExternalEnsembleError(
                    f"persisted outer blend values differ: {column}"
                )
        validate_oof_frame(
            persisted,
            prediction_columns,
            expected_keys=frame,
            expected_well_ids=provenance.outer_validation_well_ids,
            expected_folds=(outer_fold,),
        )
        return persisted, manifest

    _atomic_csv(oof_path, frame)
    manifest = {
        **expected_blend,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "oof_path": _relative(oof_path, run_root),
        "oof_sha256": sha256_file(oof_path),
    }
    _atomic_json(blend_path, manifest)
    persisted = _read_oof(oof_path)
    validate_oof_frame(
        persisted,
        prediction_columns,
        expected_keys=frame,
        expected_well_ids=provenance.outer_validation_well_ids,
        expected_folds=(outer_fold,),
    )
    return persisted, manifest


def _run_config(
    *,
    run_id: str,
    data_root: Path,
    fold_config: Path,
    cache_root: Path,
    source: OuterTcnSource,
    policy: NestedRunPolicy,
    device: str,
    workers: int,
    data_manifest: Mapping[str, Any],
    implementation_hashes: Mapping[str, str],
    mapping: pd.DataFrame,
) -> dict[str, Any]:
    cohort: dict[str, Any] = {}
    for fold in range(FOLD_COUNT):
        training, validation = partition_well_ids(mapping, fold)
        cohort[str(fold)] = {
            "training_well_ids": list(training),
            "validation_well_ids": list(validation),
            "training_ids_sha256": well_ids_sha256(training),
            "validation_ids_sha256": well_ids_sha256(validation),
            "intersection": sorted(set(training) & set(validation)),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "evidence_mode": policy.evidence_mode,
        "evidence_eligible": bool(
            policy.evidence_eligible and source.evidence_eligible
        ),
        "evidence_contract": {
            "version": 2,
            "scope": "conditional_weight_crossfit",
            "model_fit_safe": policy.model_fit_safe,
            "weight_fit_safe": policy.weight_fit_safe,
            "family_selection_safe": policy.family_selection_safe,
            "overall_promotion_eligible": bool(
                policy.evidence_eligible and source.evidence_eligible
            ),
            "family_selection_mode": "posthoc_global_quick_full_oof",
            "reason": (
                "TCN family/config was selected before this runner from global "
                "quick/full OOF; no outer-fold-safe family-selection artifact exists"
            ),
        },
        "data_root": str(data_root),
        "data_manifest_before": _jsonable(data_manifest),
        "fold_config": str(fold_config),
        "fold_config_sha256": sha256_file(fold_config),
        "fold_version": str(mapping["fold_version"].iat[0]),
        "cache_root": str(cache_root),
        "source": {
            "root": source.source_root,
            "oof_sha256": source.oof_sha256,
            "manifest_sha256": source.manifest_sha256,
            "artifact_ids": list(source.artifact_ids),
            "evidence_eligible": source.evidence_eligible,
        },
        "policy": {
            **asdict(policy),
            "grid_config": asdict(policy.grid_config),
            "target_config": asdict(policy.target_config),
        },
        "tcn_config": {
            "family": FAMILY_TCN,
            "epochs": policy.epochs,
            "hidden_dim": policy.hidden_dim,
            "layers": policy.layers,
            "learning_rate": 3.0e-3,
            "weight_decay": 1.0e-4,
            "gradient_clip": 5.0,
            "tcn_kernel_size": 3,
            "architecture_only": True,
            "external_pretrained_weights_used": [],
            "reciprocal_state_reuse": True,
            "reciprocal_reuse_requires_exact_array_hashes": True,
        },
        "blend_policy": {
            "families": {
                "incumbent": [INCUMBENT_COLUMN],
                "tcn": [TCN_COLUMN],
            },
            "family_first_seed_average": True,
            "regularization_strength": REGULARIZATION_STRENGTH,
            "regularization_target": REGULARIZATION_TARGET,
            "family_max_weight": FAMILY_MAX_WEIGHT,
            "fixed_controls": {
                FIXED_075_COLUMN: FIXED_INCUMBENT_HEAVY,
                FIXED_050_COLUMN: FIXED_SIMPLE_AVERAGE,
            },
            "weight_selection": "inner OOF only",
            "outer_validation_used_for_weight_fit": False,
        },
        "incumbent_runtime_policy": {
            "reciprocal_state_reuse": True,
            "target_free_shared_training_prediction_cache": True,
            "prediction_cache_resume_behavior": (
                "process-local cache miss recomputes from validated checkpoints"
            ),
            "equivalence_basis": (
                "reciprocal (outer, inner) orientations have identical ordered "
                "training wells and seeds; fold is changed only in stored state "
                "provenance for the exact heel-registered implementation"
            ),
        },
        "device": str(device),
        "workers": int(workers),
        "implementation_hashes": _jsonable(implementation_hashes),
        "cohort": cohort,
        "submission_created": False,
    }


def _safe_run_root(
    output_root: Path,
    run_id: str,
    data_root: Path,
    *,
    resume: bool,
    run_config: Mapping[str, Any],
) -> tuple[Path, str]:
    identifier = _safe_identifier(run_id, name="run_id")
    root = _ensure_outside_data(
        Path(output_root).resolve() / identifier,
        data_root,
        name="run root",
    )
    config_path = root / "run_config.json"
    if root.exists():
        if not resume:
            raise FileExistsError(f"run already exists: {root}")
        if (root / "summary.json").is_file():
            raise ValueError("completed nested ensemble run cannot be resumed")
        if not config_path.is_file():
            raise NestedExternalEnsembleError(
                "resume run is missing immutable run_config.json"
            )
        existing = _read_json(config_path)
        if existing != _jsonable(run_config):
            raise NestedExternalEnsembleError(
                "resume run_config differs; use a new run ID"
            )
    else:
        root.mkdir(parents=True)
        _atomic_json(config_path, run_config)
    for directory in ("models", "inner_oof", "outer_oof", "metrics"):
        (root / directory).mkdir(exist_ok=True)
    return root, _canonical_sha256(run_config)


def _model_records_from_inner(
    artifacts: Sequence[InnerSplitArtifact],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for artifact in artifacts:
        for record in artifact.manifest["tcn_checkpoints"]:
            records.append(
                {
                    "outer_fold": artifact.outer_fold,
                    "inner_fold": artifact.inner_fold,
                    "seed": int(record["provenance"]["seed"]),
                    "checkpoint": record["checkpoint"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "sidecar": record["sidecar"],
                    "sidecar_sha256": record["sidecar_sha256"],
                    "fit_scope": "inner_training",
                    "fit_wells": len(record["provenance"]["training_well_ids"]),
                    "fit_ids_sha256": record["provenance"]["training_ids_sha256"],
                    "validation_wells": len(
                        record["provenance"]["validation_well_ids"]
                    ),
                    "excluded_outer_wells": len(
                        record["provenance"]["excluded_outer_well_ids"]
                    ),
                    "reciprocal_state_reuse": record.get("reciprocal_state_reuse"),
                    "state_manifest": record["state_manifest"],
                }
            )
    return sorted(
        records,
        key=lambda record: (
            record["outer_fold"],
            record["inner_fold"],
            record["seed"],
        ),
    )


def _metric_payload(
    frame: pd.DataFrame,
    *,
    policy: NestedRunPolicy,
) -> dict[str, Any]:
    predictions = (
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        NESTED_COLUMN,
        FIXED_075_COLUMN,
        FIXED_050_COLUMN,
    )
    pooled = pooled_metrics(
        frame,
        predictions,
        baseline_column=INCUMBENT_COLUMN,
    )
    fold = per_fold_metrics(
        frame,
        predictions,
        baseline_column=INCUMBENT_COLUMN,
    )
    tail = tail_metrics(frame, predictions)
    decomposition = common_slope_shape_metrics(frame, predictions)
    bootstrap = paired_well_cluster_bootstrap(
        frame,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=predictions[1:],
        samples=policy.bootstrap_samples,
    )
    shadow: dict[str, Any] | None = None
    if policy.shadow_subset_wells <= int(frame["well_id"].nunique()):
        shadow = shadow_private_subset_simulation(
            frame,
            baseline_column=INCUMBENT_COLUMN,
            candidate_columns=predictions[1:],
            samples=policy.shadow_samples,
            subset_wells=policy.shadow_subset_wells,
        )
    correlation = residual_correlation(frame, predictions)
    wins = per_well_win_summary(
        frame,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=predictions[1:],
    )
    return {
        "pooled": pooled.to_dict("records"),
        "fold": fold.to_dict("records"),
        "tail": tail.to_dict("records"),
        "decomposition": decomposition.to_dict("records"),
        "paired_well_cluster_bootstrap": bootstrap,
        "shadow_private_subset": shadow,
        "residual_correlation": {
            "columns": list(correlation.columns),
            "values": correlation.to_numpy(dtype=np.float64).tolist(),
        },
        "per_well_win_loss": wins.to_dict("records"),
    }


def estimate_nested_tcn_runtime(
    *,
    outer_folds: int = FOLD_COUNT,
    inner_folds_per_outer: int = FOLD_COUNT - 1,
    tcn_seeds: int = len(PRODUCTION_TCN_SEEDS),
    incumbent_seeds: int = len(DEFAULT_INCUMBENT_SEEDS),
    seconds_per_tcn_fit: float = 11.0,
    seconds_per_incumbent_fit: float = 55.0,
    overhead_seconds: float = 900.0,
) -> dict[str, Any]:
    """Return an explicit sequential CPU planning estimate, not measured evidence."""

    outer_count = int(outer_folds)
    if outer_count < 1 or outer_count > FOLD_COUNT:
        raise ValueError("outer_folds must be in 1..5")
    orientations = outer_count * int(inner_folds_per_outer)
    untouched_folds = FOLD_COUNT - outer_count
    unique_pairs = (FOLD_COUNT * (FOLD_COUNT - 1) // 2) - (
        untouched_folds * (untouched_folds - 1) // 2
    )
    tcn_checkpoints = orientations * int(tcn_seeds)
    incumbent_checkpoints = orientations * int(incumbent_seeds)
    tcn_fits = unique_pairs * int(tcn_seeds)
    incumbent_fits = unique_pairs * int(incumbent_seeds)
    seconds = (
        tcn_fits * float(seconds_per_tcn_fit)
        + incumbent_fits * float(seconds_per_incumbent_fit)
        + float(overhead_seconds)
    )
    return {
        "estimate_kind": "planning heuristic; calibrate with smoke/runtime artifacts",
        "execution": "sequential CPU",
        "workers_scope": (
            "incumbent fit threadpool limit and ExtraTrees n_jobs only; "
            "outer folds, inner folds, seeds, and TCN fits remain sequential"
        ),
        "torch_cpu_threads": (
            "PyTorch runtime default; intentionally not derived from --workers"
        ),
        "shared_cache_reuse": (
            "versioned P3, sparse-descriptor, and external fixed-grid caches"
        ),
        "target_free_reciprocal_prediction_cache": True,
        "oriented_inner_splits": orientations,
        "unique_unordered_excluded_fold_pairs": unique_pairs,
        "tcn_checkpoints": tcn_checkpoints,
        "tcn_physical_fits": tcn_fits,
        "tcn_reciprocal_aliases": tcn_checkpoints - tcn_fits,
        "incumbent_checkpoints": incumbent_checkpoints,
        "incumbent_physical_fits": incumbent_fits,
        "incumbent_reciprocal_aliases": (incumbent_checkpoints - incumbent_fits),
        "assumed_seconds_per_tcn_fit": float(seconds_per_tcn_fit),
        "assumed_seconds_per_incumbent_fit": float(seconds_per_incumbent_fit),
        "assumed_overhead_seconds": float(overhead_seconds),
        "estimated_seconds": seconds,
        "estimated_hours": seconds / 3600.0,
    }


def run_nested_external_ensemble(
    *,
    data_root: Path,
    fold_config: Path,
    output_root: Path,
    run_id: str,
    cache_root: Path,
    source_root: Path | None = None,
    incumbent_oof: Path | None = None,
    source: OuterTcnSource | None = None,
    policy: NestedRunPolicy = NestedRunPolicy(),
    device: str = "cpu",
    workers: int = 4,
    resume: bool = False,
    incumbent_model: Any | None = None,
) -> dict[str, Any]:
    """Cross-fit TCN blend weights and evaluate untouched outer predictions.

    ``source`` and ``incumbent_model`` are accepted only for explicitly
    evidence-ineligible smoke runs.  Production compute always loads a
    canonical full bundle and the audited incumbent from disk.  Because TCN
    was selected using earlier global OOF, this runner marks family-selection
    safety false and must not advertise promotion-eligible nested evidence.
    """

    started = time.perf_counter()
    if not isinstance(policy, NestedRunPolicy):
        raise TypeError("policy must be NestedRunPolicy")
    data = Path(data_root).resolve()
    folds_path = Path(fold_config).resolve()
    cache = _ensure_outside_data(cache_root, data, name="cache root")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    if policy.evidence_mode == "production":
        if source is not None or incumbent_model is not None:
            raise ValueError(
                "production forbids injected sources or incumbent model overrides"
            )
        if source_root is None or incumbent_oof is None:
            raise ValueError("production requires source_root and incumbent_oof")
        source = load_production_outer_tcn_source(
            Path(source_root),
            incumbent_oof=Path(incumbent_oof),
            data_root=data,
            fold_config=folds_path,
        )
    elif source is None:
        raise ValueError("smoke mode requires an explicit evidence-ineligible source")
    if source is None:  # pragma: no cover - narrowed above
        raise RuntimeError("source resolution failed")
    if policy.model_fit_safe and not source.evidence_eligible:
        raise ValueError("production requires an independently eligible outer source")
    if policy.evidence_mode == "smoke" and source.evidence_eligible:
        raise ValueError("smoke mode requires an evidence-ineligible injected source")

    train_files = enumerate_wells(data, "train")
    train_by_id = {well.well_id: well for well in train_files}
    mapping = load_fold_mapping(
        folds_path,
        expected_well_ids=train_by_id,
    )
    validate_oof_frame(
        source.frame,
        (INCUMBENT_COLUMN, TCN_COLUMN),
        expected_well_ids=tuple(sorted(train_by_id)),
        expected_folds=range(FOLD_COUNT),
    )
    source_fold = (
        source.frame.loc[:, ["well_id", "fold"]]
        .drop_duplicates()
        .set_index("well_id")["fold"]
        .astype(int)
        .to_dict()
    )
    assigned_fold = mapping.set_index("well_id")["fold"].astype(int).to_dict()
    if source_fold != assigned_fold:
        raise NestedExternalEnsembleError(
            "outer source well/fold assignments differ from fold config"
        )

    data_manifest_before = capture_data_manifest(data)
    implementation_hashes = _implementation_hashes()
    run_config = _run_config(
        run_id=_safe_identifier(run_id, name="run_id"),
        data_root=data,
        fold_config=folds_path,
        cache_root=cache,
        source=source,
        policy=policy,
        device=device,
        workers=workers,
        data_manifest=data_manifest_before,
        implementation_hashes=implementation_hashes,
        mapping=mapping,
    )
    run_root, run_config_digest = _safe_run_root(
        output_root,
        run_id,
        data,
        resume=resume,
        run_config=run_config,
    )
    policy_digest = _canonical_sha256(run_config["policy"])
    refitter = NestedHeelIncumbentRefitter(
        data_root=data,
        fold_config=folds_path,
        # P3 and sparse-anchor descriptor caches are target-free and keyed by
        # their implementation versions.  Reuse the incumbent/full-portfolio
        # shared cache instead of rebuilding an identical 773-well copy.
        cache_root=cache,
        checkpoint_root=run_root / "incumbent_checkpoints",
        seeds=policy.incumbent_seeds,
        workers=workers,
        model=incumbent_model,
        reciprocal_state_reuse=True,
    )
    sequence_provider = _SequenceProvider(
        train_by_id,
        # Match the external portfolio runner's parent directory.  The cache
        # payload itself is additionally keyed by the fixed-grid config digest.
        cache_root=cache / "external_sequence_v1",
        config=policy.grid_config,
    )

    all_inner: list[InnerSplitArtifact] = []
    results: list[NestedFoldBlendResult] = []
    outer_frames: list[pd.DataFrame] = []
    outer_manifests: list[dict[str, Any]] = []
    completed_inner: list[dict[str, int]] = []
    completed_outer: list[int] = []
    progress_path = run_root / "progress.json"
    if progress_path.is_file():
        progress = _read_json(progress_path)
        if (
            not resume
            or progress.get("schema_version") != SCHEMA_VERSION
            or progress.get("run_config_digest") != run_config_digest
        ):
            raise NestedExternalEnsembleError(
                "progress artifact differs from the immutable run configuration"
            )

    def write_progress() -> None:
        _atomic_json(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "run_config_digest": run_config_digest,
                "completed_inner_splits": completed_inner,
                "completed_outer_folds": completed_outer,
                "expected_inner_splits": (len(policy.outer_folds) * (FOLD_COUNT - 1)),
                "expected_outer_folds": len(policy.outer_folds),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    for outer_fold in policy.outer_folds:
        inner_artifacts: list[InnerSplitArtifact] = []
        for inner_fold in range(FOLD_COUNT):
            if inner_fold == outer_fold:
                continue
            artifact = _build_inner_split(
                run_root=run_root,
                refitter=refitter,
                sequence_provider=sequence_provider,
                train_by_id=train_by_id,
                source=source,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                policy=policy,
                policy_digest=policy_digest,
                run_config_digest=run_config_digest,
                data_manifest=data_manifest_before,
                implementation_hashes=implementation_hashes,
                device=device,
                resume=resume,
            )
            inner_artifacts.append(artifact)
            all_inner.append(artifact)
            completed_inner.append({"outer_fold": outer_fold, "inner_fold": inner_fold})
            write_progress()
            print(
                "nested TCN inner complete "
                f"outer={outer_fold} inner={inner_fold} "
                f"rows={len(artifact.frame)}",
                flush=True,
            )

        inner_oof = (
            pd.concat(
                [artifact.frame for artifact in inner_artifacts],
                ignore_index=True,
            )
            .sort_values(list(OOF_KEY_COLUMNS), kind="stable")
            .reset_index(drop=True)
        )
        outer_training, outer_validation = partition_well_ids(mapping, outer_fold)
        expected_inner = source.frame.loc[
            source.frame["well_id"].isin(outer_training),
            [*OOF_KEY_COLUMNS, "y_true"],
        ]
        validate_oof_frame(
            inner_oof,
            (INCUMBENT_COLUMN, TCN_COLUMN),
            expected_keys=expected_inner,
            expected_well_ids=outer_training,
            expected_folds=tuple(
                fold for fold in range(FOLD_COUNT) if fold != outer_fold
            ),
        )
        outer_oof = source.frame.loc[
            source.frame["well_id"].isin(outer_validation),
            [*OOF_KEY_COLUMNS, "y_true", INCUMBENT_COLUMN, TCN_COLUMN],
        ].copy()
        provenance = NestedBlendProvenance(
            outer_fold=outer_fold,
            outer_training_well_ids=outer_training,
            outer_validation_well_ids=outer_validation,
            source_artifacts=tuple(
                sorted(
                    (
                        *source.artifact_ids,
                        *(artifact.artifact_id for artifact in inner_artifacts),
                    )
                )
            ),
        )
        result = fit_nested_fold_blend(
            inner_oof,
            outer_oof,
            families={
                "incumbent": (INCUMBENT_COLUMN,),
                "tcn": (TCN_COLUMN,),
            },
            provenance=provenance,
            prediction_column=NESTED_COLUMN,
            minimum_inner_folds=FOLD_COUNT - 1,
            regularization_strength=REGULARIZATION_STRENGTH,
            regularization_target=REGULARIZATION_TARGET,
            family_max_weight=FAMILY_MAX_WEIGHT,
        )
        outer_frame = _outer_frame_from_result(result)
        persisted, outer_manifest = _persist_or_validate_outer(
            run_root=run_root,
            outer_fold=outer_fold,
            result=result,
            frame=outer_frame,
            provenance=provenance,
            inner_artifacts=inner_artifacts,
            source=source,
            resume=resume,
        )
        results.append(result)
        outer_frames.append(persisted)
        outer_manifests.append(outer_manifest)
        completed_outer.append(outer_fold)
        write_progress()
        print(
            "nested blend outer complete "
            f"fold={outer_fold} rows={len(persisted)} "
            f"weights={result.weights}",
            flush=True,
        )

    final = (
        pd.concat(outer_frames, ignore_index=True)
        .sort_values(
            list(OOF_KEY_COLUMNS),
            kind="stable",
        )
        .reset_index(drop=True)
    )
    expected_final = source.frame.loc[
        source.frame["fold"].isin(policy.outer_folds),
        [*OOF_KEY_COLUMNS, "y_true"],
    ]
    prediction_columns = (
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        NESTED_COLUMN,
        FIXED_075_COLUMN,
        FIXED_050_COLUMN,
    )
    final_validation = validate_oof_frame(
        final,
        prediction_columns,
        expected_keys=expected_final,
        expected_well_ids=tuple(
            sorted(
                mapping.loc[
                    mapping["fold"].isin(policy.outer_folds),
                    "well_id",
                ].astype(str)
            )
        ),
        expected_folds=policy.outer_folds,
    )
    data_manifest_after = capture_data_manifest(data)
    if data_manifest_after != data_manifest_before:
        raise NestedExternalEnsembleError(
            "competition data manifest changed during nested ensemble run"
        )

    final_path = run_root / "oof_predictions.csv.gz"
    weights_path = run_root / "nested_weights.csv"
    model_manifest_path = run_root / "model_manifest.json"
    audit_path = run_root / "artifact_audit.json"
    metrics_path = run_root / "metrics" / "summary.json"
    final_targets = (
        final_path,
        weights_path,
        model_manifest_path,
        audit_path,
        metrics_path,
    )
    if any(path.exists() for path in final_targets):
        raise NestedExternalEnsembleError(
            "final artifacts already or partially exist; completed runs are immutable"
        )
    _atomic_csv(final_path, final)
    weights = nested_weight_table(results)
    _atomic_csv(weights_path, weights)
    model_records = _model_records_from_inner(all_inner)
    expected_model_count = (
        len(policy.outer_folds) * (FOLD_COUNT - 1) * len(policy.tcn_seeds)
    )
    if len(model_records) != expected_model_count:
        raise NestedExternalEnsembleError(
            "nested model manifest does not contain every fold/seed checkpoint"
        )
    incumbent_records = [
        record
        for artifact in all_inner
        for record in artifact.manifest["incumbent_checkpoints"]
    ]
    expected_incumbent_count = (
        len(policy.outer_folds) * (FOLD_COUNT - 1) * len(policy.incumbent_seeds)
    )
    if len(incumbent_records) != expected_incumbent_count:
        raise NestedExternalEnsembleError(
            "nested incumbent manifests do not contain every fold/seed checkpoint"
        )
    planning_estimate = estimate_nested_tcn_runtime(
        outer_folds=len(policy.outer_folds),
        tcn_seeds=len(policy.tcn_seeds),
        incumbent_seeds=len(policy.incumbent_seeds),
    )
    tcn_reciprocal_aliases = sum(
        record["reciprocal_state_reuse"] is not None for record in model_records
    )
    incumbent_reciprocal_aliases = sum(
        record.get("reciprocal_source_checkpoint_path") is not None
        for record in incumbent_records
    )
    if (
        tcn_reciprocal_aliases != planning_estimate["tcn_reciprocal_aliases"]
        or incumbent_reciprocal_aliases
        != planning_estimate["incumbent_reciprocal_aliases"]
    ):
        raise NestedExternalEnsembleError(
            "reciprocal checkpoint reuse count differs from the fold-pair plan"
        )
    tcn_physical_fits = len(model_records) - tcn_reciprocal_aliases
    incumbent_physical_fits = len(incumbent_records) - incumbent_reciprocal_aliases
    _atomic_json(
        model_manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "completed": True,
            "models": model_records,
            "model_count": len(model_records),
            "physical_fit_count": tcn_physical_fits,
            "reciprocal_alias_count": tcn_reciprocal_aliases,
            "incumbent_checkpoint_count": len(incumbent_records),
            "incumbent_physical_fit_count": incumbent_physical_fits,
            "incumbent_reciprocal_alias_count": (incumbent_reciprocal_aliases),
            "failures": [],
            "identity_fallbacks_scored": False,
        },
    )
    metrics = _metric_payload(final, policy=policy)
    _atomic_json(metrics_path, metrics)
    artifact_audit = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "evidence_mode": policy.evidence_mode,
        "evidence_eligible": bool(
            policy.evidence_eligible and source.evidence_eligible
        ),
        "evidence_contract": run_config["evidence_contract"],
        "oof_path": str(final_path),
        "oof_sha256": sha256_file(final_path),
        "oof_validation": final_validation,
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "model_manifest_path": str(model_manifest_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "source": _jsonable(source.audit),
        "source_oof_sha256": source.oof_sha256,
        "source_manifest_sha256": source.manifest_sha256,
        "outer_fold_manifests": outer_manifests,
        "checks": {
            "inner_training_excludes_outer_and_inner_validation": True,
            "outer_validation_excluded_from_weight_fit": True,
            "evidence_safety_contract_recorded": True,
            "family_first_seed_average": True,
            "fixed_regularization_strength_25": True,
            "tcn_max_weight_0_5": True,
            "finite": True,
            "exact_keys": True,
            "no_duplicates": True,
            "data_manifest_equal": True,
            "source_outer_predictions_untouched": True,
            "reciprocal_fit_equivalence_verified_by_exact_hashes": True,
            "submission_created": False,
        },
        "data_manifest_before": data_manifest_before,
        "data_manifest_after": data_manifest_after,
        "submission_created": False,
    }
    _atomic_json(audit_path, artifact_audit)
    # Independent disk re-read before completion marker.
    disk = _read_oof(final_path)
    disk_validation = validate_oof_frame(
        disk,
        prediction_columns,
        expected_keys=final,
        expected_well_ids=tuple(sorted(final["well_id"].astype(str).unique())),
        expected_folds=policy.outer_folds,
    )
    if disk_validation != final_validation:
        raise NestedExternalEnsembleError("final OOF disk re-read validation differs")
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": SCHEMA_VERSION,
        "completed": True,
        "run_id": run_id,
        "run_root": str(run_root),
        "evidence_mode": policy.evidence_mode,
        "evidence_eligible": artifact_audit["evidence_eligible"],
        "evidence_contract": run_config["evidence_contract"],
        "outer_folds": list(policy.outer_folds),
        "inner_splits": len(all_inner),
        "tcn_checkpoints": len(model_records),
        "tcn_physical_fits": tcn_physical_fits,
        "tcn_reciprocal_aliases": tcn_reciprocal_aliases,
        "incumbent_checkpoints": len(incumbent_records),
        "incumbent_refits": incumbent_physical_fits,
        "incumbent_physical_fits": incumbent_physical_fits,
        "incumbent_reciprocal_aliases": incumbent_reciprocal_aliases,
        "rows": int(len(final)),
        "wells": int(final["well_id"].nunique()),
        "weights": weights.to_dict("records"),
        "metrics": metrics,
        "artifact_audit_sha256": sha256_file(audit_path),
        "oof_sha256": sha256_file(final_path),
        "data_manifest_equal": True,
        "submission_created": False,
        "elapsed_seconds": elapsed,
        "planning_estimate": planning_estimate,
    }
    _atomic_json(run_root / "summary.json", summary)
    return _read_json(run_root / "summary.json")


__all__ = [
    "FIXED_050_COLUMN",
    "FIXED_075_COLUMN",
    "INCUMBENT_COLUMN",
    "NESTED_COLUMN",
    "NestedExternalEnsembleError",
    "NestedRunPolicy",
    "OuterTcnSource",
    "PRODUCTION_TCN_SEEDS",
    "REGULARIZATION_STRENGTH",
    "TCN_COLUMN",
    "estimate_nested_tcn_runtime",
    "load_production_outer_tcn_source",
    "make_smoke_outer_tcn_source",
    "run_nested_external_ensemble",
]
