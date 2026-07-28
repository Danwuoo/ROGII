"""Fail-closed disk artifacts for external-family OOF experiments.

The bundle written here is the evidence boundary between a model runner and
the portfolio metrics / nested ensemble code.  A completed bundle contains
only three files and is committed with one directory rename::

    external_model_artifacts/
        manifest.json
        manifest.sha256
        oof_predictions.csv.gz

Model checkpoints remain under the run root and are referenced by run-relative
path, byte size, and SHA-256.  Readers never trust an in-memory runner object:
they verify the manifest sidecar, OOF bytes, canonical keys and components,
fold provenance, checkpoint bytes, and the read-only data manifest again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from ..portfolio_metrics import (
    DEFAULT_TRUTH_COLUMN,
    OOF_KEY_COLUMNS,
    canonical_oof_keys,
    validate_oof_frame,
)
from .contracts import PROFILE_NAMES


ARTIFACT_SCHEMA_VERSION = "rogii_external_model_artifacts_v1"
DEFAULT_BUNDLE_NAME = "external_model_artifacts"
DEFAULT_AUDITED_INCUMBENT_OOF = Path(
    ".runs/heel-registered-anchor-full-20260728/oof_predictions.csv.gz"
)
OOF_FILENAME = "oof_predictions.csv.gz"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SHA256_FILENAME = "manifest.sha256"
ROW_COORDINATE_COLUMN = "row_coordinate"
INCUMBENT_COLUMN = "y_pred_incumbent"
AUDITED_INCUMBENT_SOURCE_COLUMN = "y_pred_heel_registered_ensemble"
AUDITED_REFERENCE_ATOL = 1.0e-9
DATA_MANIFEST_FIELDS = (
    "file_count",
    "total_bytes",
    "name_size_mtime_sha256",
)
MODEL_FIT_SCOPES = (
    "outer_training",
    "inner_training",
    "external_pretrained",
)
INDEPENDENT_REAUDIT_CHECKLIST = (
    "manifest_sha256",
    "manifest_schema_and_completion",
    "oof_file_sha256_and_size",
    "canonical_column_order",
    "exact_keys_truth_and_incumbent",
    "finite_values_and_no_duplicates",
    "component_algebra_and_identifiability",
    "fold_validation_coverage_and_disjointness",
    "model_checkpoint_sha256_size_and_fit_scope",
    "fold_config_sha256",
    "data_manifest_before_after_and_current",
    "no_submission_csv",
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_ATOL = 1.0e-8


def _identifier(value: Any, *, name: str) -> str:
    result = str(value)
    if not _IDENTIFIER.fullmatch(result):
        raise ValueError(
            f"{name} must match {_IDENTIFIER.pattern!r}; received {result!r}"
        )
    return result


def _sha256(value: Any, *, name: str) -> str:
    result = str(value).lower()
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return result


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 for one existing regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"artifact file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and JSON serializable") from exc


def _well_ids(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
    sort: bool = False,
) -> tuple[str, ...]:
    result = tuple(map(str, values))
    if (
        (not result and not allow_empty)
        or any(not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain unique non-empty well IDs")
    return tuple(sorted(result)) if sort else result


def well_ids_sha256(well_ids: Sequence[str]) -> str:
    """Hash an ordered well-ID sequence using the Torch-family convention."""

    values = _well_ids(
        well_ids,
        name="well_ids",
        allow_empty=True,
        sort=False,
    )
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _sorted_well_ids_sha256(well_ids: Sequence[str]) -> str:
    return well_ids_sha256(
        _well_ids(well_ids, name="well_ids", allow_empty=True, sort=True)
    )


def validate_data_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the metadata-only read-only data fingerprint."""

    if not isinstance(manifest, Mapping):
        raise TypeError("data manifest must be a mapping")
    missing = set(DATA_MANIFEST_FIELDS) - set(manifest)
    if missing:
        raise ValueError(f"data manifest is missing fields: {sorted(missing)}")
    file_count = int(manifest["file_count"])
    total_bytes = int(manifest["total_bytes"])
    if (
        isinstance(manifest["file_count"], bool)
        or isinstance(manifest["total_bytes"], bool)
        or file_count != manifest["file_count"]
        or total_bytes != manifest["total_bytes"]
        or file_count < 1
        or total_bytes < 1
    ):
        raise ValueError("data manifest counts must be positive integers")
    digest = _sha256(
        manifest["name_size_mtime_sha256"],
        name="data manifest name_size_mtime_sha256",
    )
    result = _json_copy(dict(manifest), name="data manifest")
    result.update(
        {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "name_size_mtime_sha256": digest,
        }
    )
    return result


def capture_data_manifest(data_root: str | Path) -> dict[str, Any]:
    """Fingerprint names, sizes, and mtimes without reading large data bytes."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    rows: list[str] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        rows.append(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}")
        total_bytes += int(stat.st_size)
    payload = "\n".join(rows).encode("utf-8")
    return validate_data_manifest(
        {
            "file_count": len(rows),
            "total_bytes": total_bytes,
            "name_size_mtime_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )


def load_audited_incumbent_reference(
    path: str | Path = DEFAULT_AUDITED_INCUMBENT_OOF,
    *,
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load Heel OOF truth/incumbent only after validating its sibling re-audit."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"audited incumbent OOF does not exist: {source}")
    reaudit_path = source.parent / "independent_reaudit.json"
    if not reaudit_path.is_file():
        raise FileNotFoundError(
            "audited incumbent OOF requires sibling independent_reaudit.json"
        )
    actual_sha256 = sha256_file(source)
    if expected_sha256 is not None and actual_sha256 != _sha256(
        expected_sha256,
        name="expected incumbent OOF sha256",
    ):
        raise ValueError("audited incumbent OOF SHA-256 differs from run provenance")
    reaudit = _read_json(reaudit_path)
    if (
        reaudit.get("passed") is not True
        or reaudit.get("candidate_oof_sha256") != actual_sha256
        or reaudit.get("model_state_provenance_passed") is not True
        or reaudit.get("submission_csv_created") is not False
        or int(reaudit.get("disk_oof_nonfinite_values", -1)) != 0
        or int(reaudit.get("disk_oof_duplicate_or_nonmonotone_keys", -1)) != 0
    ):
        raise ValueError("incumbent independent re-audit is absent, stale, or failed")
    required = [
        *OOF_KEY_COLUMNS,
        DEFAULT_TRUTH_COLUMN,
        AUDITED_INCUMBENT_SOURCE_COLUMN,
    ]
    try:
        frame = pd.read_csv(
            source,
            usecols=required,
            dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
            float_precision="round_trip",
        ).rename(columns={AUDITED_INCUMBENT_SOURCE_COLUMN: INCUMBENT_COLUMN})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"failed to read audited incumbent OOF: {source}") from exc
    validation = validate_oof_frame(frame, (INCUMBENT_COLUMN,))
    # The independently audited 20260728 Heel artifact predates the shared
    # canonical-key helper.  Its audit hashes the immutable CSV row order
    # (fold, well_id, row_index), while ``validate_oof_frame`` hashes the
    # canonical (well_id, row_index, fold) ordering.  The file SHA above locks
    # the exact audited bytes, so accepting either documented key convention
    # preserves the external lock without weakening it.
    legacy_key_digest = hashlib.sha256()
    for well_id, row_index, fold in frame[["well_id", "row_index", "fold"]].itertuples(
        index=False, name=None
    ):
        legacy_key_digest.update(
            f"{int(fold)}\0{well_id}\0{int(row_index)}\n".encode("utf-8")
        )
    audited_key_sha256 = str(reaudit.get("disk_oof_key_sha256", ""))
    if int(reaudit.get("disk_oof_rows", -1)) != len(
        frame
    ) or audited_key_sha256 not in {
        str(validation["key_sha256"]),
        legacy_key_digest.hexdigest(),
    }:
        raise ValueError("incumbent OOF differs from its independent re-audit")
    frame = frame.sort_values(list(OOF_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )
    return frame, {
        "path": str(source),
        "sha256": actual_sha256,
        "bytes": source.stat().st_size,
        "independent_reaudit_path": str(reaudit_path),
        "independent_reaudit_sha256": sha256_file(reaudit_path),
        "validation": validation,
    }


def select_audited_incumbent_reference(
    frame: pd.DataFrame,
    audited_reference: pd.DataFrame,
    *,
    value_atol: float = AUDITED_REFERENCE_ATOL,
) -> pd.DataFrame:
    """Select exact audited rows and reject truth/incumbent drift in ``frame``."""

    if not math.isfinite(value_atol) or value_atol < 0.0:
        raise ValueError("value_atol must be finite and nonnegative")
    validate_oof_frame(frame, (INCUMBENT_COLUMN,))
    validate_oof_frame(audited_reference, (INCUMBENT_COLUMN,))
    keys = canonical_oof_keys(frame)
    reference = audited_reference.loc[
        :,
        [
            *OOF_KEY_COLUMNS,
            DEFAULT_TRUTH_COLUMN,
            INCUMBENT_COLUMN,
        ],
    ].copy()
    reference["well_id"] = reference["well_id"].astype(str)
    selected = keys.merge(
        reference,
        on=list(OOF_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(selected) != len(keys) or not selected["_merge"].eq("both").all():
        raise ValueError(
            "OOF keys are not completely covered by the audited incumbent reference"
        )
    selected = selected.drop(columns="_merge")
    actual = keys.merge(
        frame.loc[
            :,
            [
                *OOF_KEY_COLUMNS,
                DEFAULT_TRUTH_COLUMN,
                INCUMBENT_COLUMN,
            ],
        ].assign(well_id=lambda value: value["well_id"].astype(str)),
        on=list(OOF_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    deltas: dict[str, float] = {}
    for column in (DEFAULT_TRUTH_COLUMN, INCUMBENT_COLUMN):
        left = pd.to_numeric(actual[column], errors="raise").to_numpy(dtype=np.float64)
        right = pd.to_numeric(selected[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("audited incumbent comparison contains non-finite values")
        deltas[column] = float(np.max(np.abs(left - right)))
    if max(deltas.values()) > float(value_atol):
        raise ValueError(
            "OOF truth/incumbent differs from audited incumbent reference; "
            f"maximum_deltas={deltas}, tolerance={value_atol}"
        )
    return selected


@dataclass(frozen=True)
class FoldProvenance:
    """Exact well boundary for one untouched outer validation fold."""

    outer_fold: int
    training_well_ids: tuple[str, ...]
    validation_well_ids: tuple[str, ...]
    inner_oof_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fold = int(self.outer_fold)
        if isinstance(self.outer_fold, bool) or fold != self.outer_fold or fold < 0:
            raise ValueError("outer_fold must be a non-negative integer")
        training = _well_ids(
            self.training_well_ids,
            name="training_well_ids",
            sort=True,
        )
        validation = _well_ids(
            self.validation_well_ids,
            name="validation_well_ids",
            sort=True,
        )
        overlap = set(training) & set(validation)
        if overlap:
            raise ValueError(
                f"outer training and validation wells overlap: {sorted(overlap)[:5]}"
            )
        inner = tuple(
            sorted(
                _identifier(value, name="inner_oof_artifact_id")
                for value in self.inner_oof_artifact_ids
            )
        )
        if len(set(inner)) != len(inner):
            raise ValueError("inner_oof_artifact_ids must be unique")
        object.__setattr__(self, "outer_fold", fold)
        object.__setattr__(self, "training_well_ids", training)
        object.__setattr__(self, "validation_well_ids", validation)
        object.__setattr__(self, "inner_oof_artifact_ids", inner)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "training_well_ids": list(self.training_well_ids),
            "validation_well_ids": list(self.validation_well_ids),
            "training_wells": len(self.training_well_ids),
            "validation_wells": len(self.validation_well_ids),
            "training_well_ids_sha256": _sorted_well_ids_sha256(self.training_well_ids),
            "validation_well_ids_sha256": _sorted_well_ids_sha256(
                self.validation_well_ids
            ),
            "training_validation_intersection": 0,
            "inner_oof_artifact_ids": list(self.inner_oof_artifact_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FoldProvenance":
        if not isinstance(value, Mapping):
            raise TypeError("fold provenance record must be a mapping")
        result = cls(
            outer_fold=int(value["outer_fold"]),
            training_well_ids=tuple(value["training_well_ids"]),
            validation_well_ids=tuple(value["validation_well_ids"]),
            inner_oof_artifact_ids=tuple(value.get("inner_oof_artifact_ids", ())),
        )
        expected = result.as_dict()
        for field in (
            "training_wells",
            "validation_wells",
            "training_well_ids_sha256",
            "validation_well_ids_sha256",
            "training_validation_intersection",
        ):
            if value.get(field) != expected[field]:
                raise ValueError(f"fold provenance field {field!r} is inconsistent")
        return result


@dataclass(frozen=True)
class ModelArtifact:
    """One checkpoint and the exact wells/statistics used to fit it."""

    model_id: str
    family_id: str
    variant_id: str
    path: str
    sha256: str
    bytes: int
    fit_scope: str
    outer_fold: int | None
    inner_fold: int | None
    seed: int | None
    fit_well_ids: tuple[str, ...]
    state_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        model_id = _identifier(self.model_id, name="model_id")
        family_id = _identifier(self.family_id, name="family_id")
        variant_id = _identifier(self.variant_id, name="variant_id")
        relative = Path(str(self.path))
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or str(relative) in {"", "."}
        ):
            raise ValueError("model path must be a safe run-relative path")
        path = relative.as_posix()
        digest = _sha256(self.sha256, name="model sha256")
        size = int(self.bytes)
        if isinstance(self.bytes, bool) or size != self.bytes or size < 1:
            raise ValueError("model bytes must be a positive integer")
        fit_scope = str(self.fit_scope)
        if fit_scope not in MODEL_FIT_SCOPES:
            raise ValueError(f"fit_scope must be one of {MODEL_FIT_SCOPES}")
        outer = None if self.outer_fold is None else int(self.outer_fold)
        inner = None if self.inner_fold is None else int(self.inner_fold)
        seed = None if self.seed is None else int(self.seed)
        if outer is not None and outer < 0:
            raise ValueError("model outer_fold must be non-negative")
        if inner is not None and inner < 0:
            raise ValueError("model inner_fold must be non-negative")
        if seed is not None and seed < 0:
            raise ValueError("model seed must be non-negative")
        fit_ids = _well_ids(
            self.fit_well_ids,
            name="fit_well_ids",
            allow_empty=fit_scope == "external_pretrained",
            sort=False,
        )
        if fit_scope == "external_pretrained":
            if outer is not None or inner is not None or fit_ids:
                raise ValueError(
                    "external_pretrained records have no fold or competition fit IDs"
                )
        elif outer is None:
            raise ValueError("fold-local models require outer_fold")
        if fit_scope == "outer_training" and inner is not None:
            raise ValueError("outer_training model must not declare inner_fold")
        if fit_scope == "inner_training" and inner is None:
            raise ValueError("inner_training model must declare inner_fold")
        state = _json_copy(dict(self.state_manifest), name="state_manifest")
        fit_digest = well_ids_sha256(fit_ids)
        if "fit_ids_sha256" in state and state["fit_ids_sha256"] != fit_digest:
            raise ValueError("state_manifest fit_ids_sha256 differs from fit_well_ids")
        if "parameter_sha256" in state:
            _sha256(state["parameter_sha256"], name="parameter_sha256")
        if "family" in state and str(state["family"]) != family_id:
            raise ValueError("state_manifest family differs from family_id")
        if "variant" in state and str(state["variant"]) != variant_id:
            raise ValueError("state_manifest variant differs from variant_id")
        if "seed" in state and seed is not None and int(state["seed"]) != seed:
            raise ValueError("state_manifest seed differs from model seed")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "variant_id", variant_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "bytes", size)
        object.__setattr__(self, "fit_scope", fit_scope)
        object.__setattr__(self, "outer_fold", outer)
        object.__setattr__(self, "inner_fold", inner)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "fit_well_ids", fit_ids)
        object.__setattr__(self, "state_manifest", state)

    @property
    def fit_well_ids_sha256(self) -> str:
        return well_ids_sha256(self.fit_well_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family_id": self.family_id,
            "variant_id": self.variant_id,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "fit_scope": self.fit_scope,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "seed": self.seed,
            "fit_well_ids": list(self.fit_well_ids),
            "fit_wells": len(self.fit_well_ids),
            "fit_well_ids_sha256": self.fit_well_ids_sha256,
            "state_manifest": _json_copy(self.state_manifest, name="state_manifest"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelArtifact":
        if not isinstance(value, Mapping):
            raise TypeError("model artifact record must be a mapping")
        result = cls(
            model_id=str(value["model_id"]),
            family_id=str(value["family_id"]),
            variant_id=str(value["variant_id"]),
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            bytes=int(value["bytes"]),
            fit_scope=str(value["fit_scope"]),
            outer_fold=value.get("outer_fold"),
            inner_fold=value.get("inner_fold"),
            seed=value.get("seed"),
            fit_well_ids=tuple(value.get("fit_well_ids", ())),
            state_manifest=value.get("state_manifest", {}),
        )
        if value.get("fit_wells") != len(result.fit_well_ids):
            raise ValueError("model artifact fit_wells is inconsistent")
        if value.get("fit_well_ids_sha256") != result.fit_well_ids_sha256:
            raise ValueError("model artifact fit_well_ids_sha256 is inconsistent")
        return result


def make_model_artifact(
    run_root: str | Path,
    model_path: str | Path,
    *,
    model_id: str,
    family_id: str,
    variant_id: str,
    fit_scope: str,
    fit_well_ids: Sequence[str] = (),
    outer_fold: int | None = None,
    inner_fold: int | None = None,
    seed: int | None = None,
    state_manifest: Mapping[str, Any] | None = None,
) -> ModelArtifact:
    """Freeze checkpoint bytes and fold-fit provenance into one record."""

    root = Path(run_root).resolve()
    source = Path(model_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError("model checkpoint must be inside the run root") from exc
    if not source.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {source}")
    return ModelArtifact(
        model_id=model_id,
        family_id=family_id,
        variant_id=variant_id,
        path=relative.as_posix(),
        sha256=sha256_file(source),
        bytes=source.stat().st_size,
        fit_scope=fit_scope,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        seed=seed,
        fit_well_ids=tuple(fit_well_ids),
        state_manifest={} if state_manifest is None else state_manifest,
    )


@dataclass(frozen=True)
class OofCandidateSchema:
    """Canonical common/slope/shape columns for one candidate prediction."""

    candidate_id: str
    family_id: str
    variant_id: str
    transfer_mode: str
    control_id: str
    eligible_for_model_selection: bool
    source_model_ids: tuple[str, ...]
    common_column: str
    slope_column: str
    shape_column: str
    correction_column: str
    prediction_column: str

    def __post_init__(self) -> None:
        candidate = _identifier(self.candidate_id, name="candidate_id")
        family = _identifier(self.family_id, name="family_id")
        variant = _identifier(self.variant_id, name="variant_id")
        transfer = _identifier(self.transfer_mode, name="transfer_mode")
        control = _identifier(self.control_id, name="control_id")
        if not isinstance(self.eligible_for_model_selection, bool):
            raise TypeError("eligible_for_model_selection must be bool")
        if control != "none" and self.eligible_for_model_selection:
            raise ValueError(
                "negative-control predictions cannot be eligible for model selection"
            )
        sources = tuple(
            sorted(
                _identifier(value, name="source_model_id")
                for value in self.source_model_ids
            )
        )
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("source_model_ids must be non-empty and unique")
        expected = {
            "common_column": f"res_common__{candidate}",
            "slope_column": f"res_slope__{candidate}",
            "shape_column": f"res_shape__{candidate}",
            "correction_column": f"res_correction__{candidate}",
            "prediction_column": f"y_pred__{candidate}",
        }
        for field, expected_value in expected.items():
            if str(getattr(self, field)) != expected_value:
                raise ValueError(f"{field} must use canonical name {expected_value!r}")
            object.__setattr__(self, field, expected_value)
        object.__setattr__(self, "candidate_id", candidate)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "variant_id", variant)
        object.__setattr__(self, "transfer_mode", transfer)
        object.__setattr__(self, "control_id", control)
        object.__setattr__(self, "source_model_ids", sources)

    @classmethod
    def canonical(
        cls,
        candidate_id: str,
        *,
        family_id: str,
        variant_id: str,
        transfer_mode: str,
        source_model_ids: Sequence[str],
        control_id: str = "none",
        eligible_for_model_selection: bool | None = None,
    ) -> "OofCandidateSchema":
        candidate = _identifier(candidate_id, name="candidate_id")
        transfer = _identifier(transfer_mode, name="transfer_mode")
        control = _identifier(control_id, name="control_id")
        eligible = (
            control == "none" and transfer not in {"random_frozen", "negative_control"}
            if eligible_for_model_selection is None
            else eligible_for_model_selection
        )
        return cls(
            candidate_id=candidate,
            family_id=family_id,
            variant_id=variant_id,
            transfer_mode=transfer,
            control_id=control,
            eligible_for_model_selection=eligible,
            source_model_ids=tuple(source_model_ids),
            common_column=f"res_common__{candidate}",
            slope_column=f"res_slope__{candidate}",
            shape_column=f"res_shape__{candidate}",
            correction_column=f"res_correction__{candidate}",
            prediction_column=f"y_pred__{candidate}",
        )

    @property
    def component_columns(self) -> tuple[str, ...]:
        return (
            self.common_column,
            self.slope_column,
            self.shape_column,
            self.correction_column,
            self.prediction_column,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "variant_id": self.variant_id,
            "transfer_mode": self.transfer_mode,
            "control_id": self.control_id,
            "eligible_for_model_selection": self.eligible_for_model_selection,
            "source_model_ids": list(self.source_model_ids),
            "common_column": self.common_column,
            "slope_column": self.slope_column,
            "shape_column": self.shape_column,
            "correction_column": self.correction_column,
            "prediction_column": self.prediction_column,
            "prediction_definition": (
                f"{INCUMBENT_COLUMN} + {self.common_column} + "
                f"{self.slope_column} + {self.shape_column}"
            ),
            "identity_fallback_rows": 0,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OofCandidateSchema":
        if not isinstance(value, Mapping):
            raise TypeError("OOF candidate schema must be a mapping")
        result = cls(
            candidate_id=str(value["candidate_id"]),
            family_id=str(value["family_id"]),
            variant_id=str(value["variant_id"]),
            transfer_mode=str(value["transfer_mode"]),
            control_id=str(value["control_id"]),
            eligible_for_model_selection=value["eligible_for_model_selection"],
            source_model_ids=tuple(value["source_model_ids"]),
            common_column=str(value["common_column"]),
            slope_column=str(value["slope_column"]),
            shape_column=str(value["shape_column"]),
            correction_column=str(value["correction_column"]),
            prediction_column=str(value["prediction_column"]),
        )
        if value.get("identity_fallback_rows") != 0:
            raise ValueError(
                "completed OOF candidates cannot contain identity fallback"
            )
        return result


@dataclass(frozen=True)
class ExternalArtifactBundle:
    """A fully re-read and re-audited completed artifact bundle."""

    bundle_root: Path
    manifest: Mapping[str, Any]
    oof: pd.DataFrame
    audit: Mapping[str, Any]


def canonical_oof_columns(
    candidates: Sequence[OofCandidateSchema],
) -> tuple[str, ...]:
    """Return the only accepted on-disk OOF column order."""

    normalized = _candidate_tuple(candidates)
    return (
        *OOF_KEY_COLUMNS,
        ROW_COORDINATE_COLUMN,
        DEFAULT_TRUTH_COLUMN,
        INCUMBENT_COLUMN,
        *(column for candidate in normalized for column in candidate.component_columns),
    )


def _candidate_tuple(
    candidates: Sequence[OofCandidateSchema],
) -> tuple[OofCandidateSchema, ...]:
    result = tuple(candidates)
    if not result or any(
        not isinstance(candidate, OofCandidateSchema) for candidate in result
    ):
        raise TypeError("candidates must contain OofCandidateSchema records")
    identifiers = [candidate.candidate_id for candidate in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate IDs must be unique")
    all_columns = [
        column for candidate in result for column in candidate.component_columns
    ]
    if len(set(all_columns)) != len(all_columns):
        raise ValueError("candidate component columns must be unique")
    return result


def _fold_tuple(
    records: Sequence[FoldProvenance],
    *,
    profile: str,
) -> tuple[FoldProvenance, ...]:
    result = tuple(records)
    if not result or any(not isinstance(item, FoldProvenance) for item in result):
        raise TypeError("fold_provenance must contain FoldProvenance records")
    folds = [item.outer_fold for item in result]
    if len(set(folds)) != len(folds):
        raise ValueError("outer fold provenance records must be unique")
    result = tuple(sorted(result, key=lambda item: item.outer_fold))
    if profile == "full" and tuple(item.outer_fold for item in result) != tuple(
        range(5)
    ):
        raise ValueError("full profile requires exact outer folds 0..4")
    return result


def _model_tuple(
    records: Sequence[ModelArtifact],
) -> tuple[ModelArtifact, ...]:
    result = tuple(records)
    if not result or any(not isinstance(item, ModelArtifact) for item in result):
        raise TypeError("model_artifacts must contain ModelArtifact records")
    identifiers = [item.model_id for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("model artifact IDs must be unique")
    return tuple(sorted(result, key=lambda item: item.model_id))


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        result = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OOF column {column!r} must be numeric") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"OOF column {column!r} must be finite")
    return result


def _column_digest(frame: pd.DataFrame, column: str) -> str:
    keys = canonical_oof_keys(frame)
    indexed = frame.loc[:, [*OOF_KEY_COLUMNS, column]].copy()
    indexed["well_id"] = indexed["well_id"].astype(str)
    indexed["row_index"] = pd.to_numeric(indexed["row_index"], errors="raise").astype(
        np.int64
    )
    indexed["fold"] = pd.to_numeric(indexed["fold"], errors="raise").astype(np.int64)
    indexed = indexed.sort_values(list(OOF_KEY_COLUMNS), kind="stable")
    if not np.array_equal(
        indexed.loc[:, list(OOF_KEY_COLUMNS)].to_numpy(),
        keys.loc[:, list(OOF_KEY_COLUMNS)].to_numpy(),
    ):
        raise ValueError("OOF column digest key alignment failed")
    values = _numeric(indexed, column).astype("<f8", copy=False)
    digest = hashlib.sha256()
    digest.update(column.encode("utf-8"))
    digest.update(b"\0")
    digest.update(values.tobytes())
    return digest.hexdigest()


def _compare_reference_column(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    column: str,
) -> None:
    if column not in reference:
        raise ValueError(f"expected_reference is missing {column!r}")
    left = frame.loc[:, [*OOF_KEY_COLUMNS, column]].copy()
    right = reference.loc[:, [*OOF_KEY_COLUMNS, column]].copy()
    for current in (left, right):
        current["well_id"] = current["well_id"].astype(str)
        current["row_index"] = pd.to_numeric(
            current["row_index"], errors="raise"
        ).astype(np.int64)
        current["fold"] = pd.to_numeric(current["fold"], errors="raise").astype(
            np.int64
        )
    aligned = left.merge(
        right,
        on=list(OOF_KEY_COLUMNS),
        validate="one_to_one",
        how="inner",
        suffixes=("_actual", "_expected"),
    )
    actual = _numeric(aligned, f"{column}_actual")
    expected = _numeric(aligned, f"{column}_expected")
    if not np.array_equal(actual, expected):
        maximum = float(np.max(np.abs(actual - expected)))
        raise ValueError(
            f"OOF {column!r} differs from locked reference; "
            f"maximum_absolute_delta={maximum}"
        )


def _component_report(
    frame: pd.DataFrame,
    candidate: OofCandidateSchema,
) -> dict[str, float]:
    common = _numeric(frame, candidate.common_column)
    slope = _numeric(frame, candidate.slope_column)
    shape = _numeric(frame, candidate.shape_column)
    correction = _numeric(frame, candidate.correction_column)
    prediction = _numeric(frame, candidate.prediction_column)
    incumbent = _numeric(frame, INCUMBENT_COLUMN)
    coordinate = _numeric(frame, ROW_COORDINATE_COLUMN)
    algebra = float(np.max(np.abs(correction - (common + slope + shape))))
    final = float(np.max(np.abs(prediction - (incumbent + correction))))
    if algebra > _COMPONENT_ATOL or final > _COMPONENT_ATOL:
        raise ValueError(
            f"candidate {candidate.candidate_id!r} component algebra failed: "
            f"correction={algebra}, final={final}"
        )

    well_ids = frame["well_id"].astype(str).to_numpy()
    common_span = 0.0
    slope_mean = 0.0
    shape_mean = 0.0
    slope_linearity = 0.0
    shape_orthogonality = 0.0
    groups = frame.groupby("well_id", sort=True).indices
    for indices in groups.values():
        selected = np.asarray(indices, dtype=np.int64)
        x = coordinate[selected]
        order = np.argsort(x, kind="stable")
        x = x[order]
        if np.any(np.diff(x) < 0.0):
            raise ValueError("row_coordinate must be monotone within each well")
        if x[0] < -_COMPONENT_ATOL or x[-1] > 1.0 + _COMPONENT_ATOL:
            raise ValueError("row_coordinate must lie within [0, 1]")
        current_common = common[selected][order]
        current_slope = slope[selected][order]
        current_shape = shape[selected][order]
        centered_x = x - float(np.mean(x))
        common_span = max(
            common_span, float(np.max(current_common) - np.min(current_common))
        )
        slope_mean = max(slope_mean, abs(float(np.mean(current_slope))))
        shape_mean = max(shape_mean, abs(float(np.mean(current_shape))))
        denominator = float(centered_x @ centered_x)
        coefficient = (
            float(centered_x @ current_slope) / denominator
            if denominator > np.finfo(np.float64).eps
            else 0.0
        )
        slope_linearity = max(
            slope_linearity,
            float(np.max(np.abs(current_slope - coefficient * centered_x))),
        )
        scale = max(
            1.0,
            float(np.linalg.norm(current_shape) * np.linalg.norm(centered_x)),
        )
        shape_orthogonality = max(
            shape_orthogonality,
            abs(float(current_shape @ centered_x)) / scale,
        )
    del well_ids
    maximum = max(
        common_span,
        slope_mean,
        shape_mean,
        slope_linearity,
        shape_orthogonality,
    )
    if maximum > _COMPONENT_ATOL:
        raise ValueError(
            f"candidate {candidate.candidate_id!r} component identifiability "
            f"failed; maximum_error={maximum}"
        )
    return {
        "maximum_correction_reconstruction_error": algebra,
        "maximum_final_reconstruction_error": final,
        "maximum_common_span": common_span,
        "maximum_slope_mean": slope_mean,
        "maximum_shape_mean": shape_mean,
        "maximum_slope_linearity_error": slope_linearity,
        "maximum_shape_orthogonality_error": shape_orthogonality,
    }


def _canonicalize_and_validate_oof(
    frame: pd.DataFrame,
    *,
    candidates: tuple[OofCandidateSchema, ...],
    folds: tuple[FoldProvenance, ...],
    expected_reference: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("OOF artifact must be a pandas DataFrame")
    expected_columns = canonical_oof_columns(candidates)
    if tuple(map(str, frame.columns)) != expected_columns:
        missing = set(expected_columns) - set(frame.columns)
        extra = set(frame.columns) - set(expected_columns)
        raise ValueError(
            "OOF columns/order differ from the canonical schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}, "
            f"expected_order={list(expected_columns)}"
        )
    result = frame.copy()
    result["well_id"] = result["well_id"].astype(str)
    result["row_index"] = pd.to_numeric(result["row_index"], errors="raise").astype(
        np.int64
    )
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(np.int64)
    result = result.sort_values(list(OOF_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )
    prediction_columns = (
        INCUMBENT_COLUMN,
        *(candidate.prediction_column for candidate in candidates),
    )
    validation = validate_oof_frame(
        result,
        prediction_columns,
        truth_column=DEFAULT_TRUTH_COLUMN,
        expected_keys=expected_reference,
        expected_folds=[item.outer_fold for item in folds],
    )
    _numeric(result, ROW_COORDINATE_COLUMN)
    for candidate in candidates:
        for column in candidate.component_columns:
            _numeric(result, column)
    if expected_reference is not None:
        _compare_reference_column(result, expected_reference, INCUMBENT_COLUMN)
        if ROW_COORDINATE_COLUMN in expected_reference:
            _compare_reference_column(result, expected_reference, ROW_COORDINATE_COLUMN)

    by_well = (
        result.loc[:, ["well_id", "fold"]]
        .drop_duplicates()
        .set_index("well_id")["fold"]
        .astype(int)
        .to_dict()
    )
    provenance_wells: set[str] = set()
    for record in folds:
        expected = set(record.validation_well_ids)
        actual = {
            str(well_id)
            for well_id, fold in by_well.items()
            if int(fold) == record.outer_fold
        }
        if actual != expected:
            raise ValueError(
                f"fold {record.outer_fold} validation coverage differs from "
                f"provenance: missing={len(expected - actual)}, "
                f"extra={len(actual - expected)}"
            )
        overlap = provenance_wells & expected
        if overlap:
            raise ValueError(
                f"validation wells appear in multiple outer folds: "
                f"{sorted(overlap)[:5]}"
            )
        provenance_wells.update(expected)
    if provenance_wells != set(by_well):
        raise ValueError("fold provenance does not cover the complete OOF cohort")

    component_reports = {
        candidate.candidate_id: _component_report(result, candidate)
        for candidate in candidates
    }
    validation.update(
        {
            "column_order": list(expected_columns),
            "truth_sha256": _column_digest(result, DEFAULT_TRUTH_COLUMN),
            "incumbent_sha256": _column_digest(result, INCUMBENT_COLUMN),
            "component_validation": component_reports,
        }
    )
    return result, validation


def _resolve_model_path(run_root: Path, relative: str) -> Path:
    path = (run_root / relative).resolve()
    try:
        path.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("model artifact path escapes the run root") from exc
    return path


def _validate_models(
    run_root: Path,
    *,
    candidates: tuple[OofCandidateSchema, ...],
    folds: tuple[FoldProvenance, ...],
    models: tuple[ModelArtifact, ...],
) -> dict[str, Any]:
    fold_by_id = {item.outer_fold: item for item in folds}
    model_by_id = {item.model_id: item for item in models}
    checked_parameter_digests = 0
    for model in models:
        path = _resolve_model_path(run_root, model.path)
        if not path.is_file():
            raise FileNotFoundError(f"model checkpoint is missing: {path}")
        if path.stat().st_size != model.bytes:
            raise ValueError(f"model checkpoint size differs: {model.model_id}")
        if sha256_file(path) != model.sha256:
            raise ValueError(f"model checkpoint SHA-256 differs: {model.model_id}")
        if "parameter_sha256" in model.state_manifest:
            _sha256(
                model.state_manifest["parameter_sha256"],
                name="parameter_sha256",
            )
            checked_parameter_digests += 1
        if model.fit_scope == "external_pretrained":
            continue
        if model.outer_fold not in fold_by_id:
            raise ValueError(f"model {model.model_id!r} references unknown outer fold")
        fold = fold_by_id[int(model.outer_fold)]
        fit_ids = set(model.fit_well_ids)
        training = set(fold.training_well_ids)
        validation = set(fold.validation_well_ids)
        if fit_ids & validation:
            raise ValueError(
                f"model {model.model_id!r} fit IDs include outer validation wells"
            )
        if model.fit_scope == "outer_training" and fit_ids != training:
            raise ValueError(
                f"outer model {model.model_id!r} fit IDs differ from the exact "
                "outer-training cohort"
            )
        if model.fit_scope == "inner_training" and not fit_ids.issubset(training):
            raise ValueError(
                f"inner model {model.model_id!r} fit IDs escape outer training"
            )

    for candidate in candidates:
        missing = set(candidate.source_model_ids) - set(model_by_id)
        if missing:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} references unknown models: "
                f"{sorted(missing)}"
            )
        sources = [model_by_id[value] for value in candidate.source_model_ids]
        if any(model.family_id != candidate.family_id for model in sources):
            raise ValueError(f"candidate {candidate.candidate_id!r} crosses family IDs")
        covered = {
            int(model.outer_fold)
            for model in sources
            if model.fit_scope == "outer_training" and model.outer_fold is not None
        }
        expected = set(fold_by_id)
        if covered != expected:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} lacks outer model coverage: "
                f"missing={sorted(expected - covered)}, extra={sorted(covered - expected)}"
            )
    return {
        "model_files": len(models),
        "model_bytes": int(sum(item.bytes for item in models)),
        "checkpoint_sha256_checked": len(models),
        "parameter_sha256_declared": checked_parameter_digests,
        "candidate_outer_fold_coverage_checked": len(candidates),
    }


def _validate_incumbent_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _json_copy(dict(value), name="incumbent_manifest")
    states = manifest.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("incumbent_manifest must contain non-empty states")
    seen: set[tuple[int, int]] = set()
    for state in states:
        if not isinstance(state, dict):
            raise ValueError("incumbent state records must be mappings")
        fold = int(state["fold"])
        seed = int(state["seed"])
        if fold < 0 or seed < 0 or (fold, seed) in seen:
            raise ValueError("incumbent fold/seed state provenance is invalid")
        seen.add((fold, seed))
        _sha256(state["sha256"], name="incumbent state sha256")
        if int(state.get("training_validation_intersection", -1)) != 0:
            raise ValueError("incumbent state has training/validation overlap")
    return manifest


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest_sidecar(path: Path, digest: str) -> None:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{digest}  {MANIFEST_FILENAME}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _read_oof(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
            float_precision="round_trip",
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"failed to read OOF artifact: {path}") from exc


def _validate_fold_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("fold_config must be a mapping")
    path = Path(str(value["path"])).resolve()
    digest = _sha256(value["sha256"], name="fold_config sha256")
    size = int(value["bytes"])
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError("fold config file is missing or has a different size")
    if sha256_file(path) != digest:
        raise ValueError("fold config SHA-256 differs")
    return {"path": str(path), "sha256": digest, "bytes": size}


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "completed",
        "run_id",
        "profile",
        "created_utc",
        "submission_csv_created",
        "data_root",
        "data_read_only_manifest_before",
        "data_read_only_manifest_after",
        "fold_config",
        "incumbent",
        "oof",
        "fold_provenance",
        "model_artifacts",
        "metadata",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"artifact manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("artifact manifest schema version is unsupported")
    if manifest["completed"] is not True:
        raise ValueError("artifact manifest is not completed")
    _identifier(manifest["run_id"], name="run_id")
    if manifest["profile"] not in PROFILE_NAMES:
        raise ValueError(f"profile must be one of {PROFILE_NAMES}")
    if manifest["submission_csv_created"] is not False:
        raise ValueError("external-model evidence bundle must not create a submission")
    _json_copy(manifest["metadata"], name="metadata")


def _submission_csvs(run_root: Path) -> list[str]:
    return sorted(
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
        and "submission" in path.name.lower()
        and path.suffix.lower() == ".csv"
    )


def _audit_bundle(
    bundle_root: Path,
    *,
    run_root: Path,
    expected_reference: pd.DataFrame | None,
    verify_current_data: bool,
    data_root_override: Path | None,
) -> ExternalArtifactBundle:
    manifest_path = bundle_root / MANIFEST_FILENAME
    sidecar_path = bundle_root / MANIFEST_SHA256_FILENAME
    oof_path = bundle_root / OOF_FILENAME
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            "artifact bundle is missing manifest or checksum sidecar"
        )
    sidecar = sidecar_path.read_text(encoding="ascii").strip().split()
    if len(sidecar) != 2 or sidecar[1] != MANIFEST_FILENAME:
        raise ValueError("manifest checksum sidecar has an invalid format")
    expected_manifest_sha = _sha256(sidecar[0], name="manifest sidecar sha256")
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("manifest SHA-256 differs from its sidecar")

    manifest = _read_json(manifest_path)
    _validate_manifest_header(manifest)
    profile = str(manifest["profile"])
    candidates = tuple(
        OofCandidateSchema.from_dict(value) for value in manifest["oof"]["candidates"]
    )
    folds = _fold_tuple(
        tuple(FoldProvenance.from_dict(value) for value in manifest["fold_provenance"]),
        profile=profile,
    )
    models = _model_tuple(
        tuple(ModelArtifact.from_dict(value) for value in manifest["model_artifacts"])
    )
    before = validate_data_manifest(manifest["data_read_only_manifest_before"])
    after = validate_data_manifest(manifest["data_read_only_manifest_after"])
    if before != after:
        raise ValueError("data manifest changed during the experiment")
    fold_config = _validate_fold_config(manifest["fold_config"])
    incumbent = manifest["incumbent"]
    if not isinstance(incumbent, Mapping):
        raise TypeError("incumbent manifest block must be a mapping")
    _validate_incumbent_manifest(incumbent["state_manifest"])

    if not oof_path.is_file():
        raise FileNotFoundError("OOF artifact is missing")
    oof_record = manifest["oof"]
    if oof_record.get("path") != OOF_FILENAME:
        raise ValueError("OOF manifest path differs from the canonical filename")
    if oof_path.stat().st_size != int(oof_record["bytes"]):
        raise ValueError("OOF artifact byte size differs from manifest")
    if sha256_file(oof_path) != _sha256(oof_record["sha256"], name="OOF sha256"):
        raise ValueError("OOF artifact SHA-256 differs from manifest")
    oof = _read_oof(oof_path)
    oof, oof_validation = _canonicalize_and_validate_oof(
        oof,
        candidates=candidates,
        folds=folds,
        expected_reference=expected_reference,
    )
    for field in (
        "rows",
        "wells",
        "folds",
        "prediction_columns",
        "key_sha256",
        "column_order",
        "truth_sha256",
        "incumbent_sha256",
    ):
        if oof_record.get(field) != oof_validation[field]:
            raise ValueError(f"OOF manifest field {field!r} is inconsistent")
    if incumbent.get("column") != INCUMBENT_COLUMN:
        raise ValueError("incumbent column declaration is inconsistent")
    if incumbent.get("prediction_sha256") != oof_validation["incumbent_sha256"]:
        raise ValueError("incumbent prediction digest is inconsistent")

    model_validation = _validate_models(
        run_root,
        candidates=candidates,
        folds=folds,
        models=models,
    )
    data_root = (
        Path(data_root_override).resolve()
        if data_root_override is not None
        else Path(str(manifest["data_root"])).resolve()
    )
    current_data: dict[str, Any] | None = None
    if verify_current_data:
        current_data = capture_data_manifest(data_root)
        if current_data != before:
            raise ValueError("current data manifest differs from the experiment input")
    submissions = _submission_csvs(run_root)
    if submissions:
        raise ValueError(f"submission CSV is present in run root: {submissions}")

    audit = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "passed": True,
        "run_id": manifest["run_id"],
        "profile": profile,
        "bundle_root": str(bundle_root),
        "manifest_sha256": expected_manifest_sha,
        "checks": {name: True for name in INDEPENDENT_REAUDIT_CHECKLIST},
        "oof": oof_validation,
        "models": model_validation,
        "fold_config": fold_config,
        "data_read_only_manifest_before": before,
        "data_read_only_manifest_after": after,
        "data_read_only_manifest_current": current_data,
        "submission_csv_created": False,
    }
    return ExternalArtifactBundle(
        bundle_root=bundle_root,
        manifest=manifest,
        oof=oof,
        audit=audit,
    )


def write_external_artifact_bundle(
    run_root: str | Path,
    oof: pd.DataFrame,
    *,
    run_id: str,
    profile: str,
    candidates: Sequence[OofCandidateSchema],
    fold_provenance: Sequence[FoldProvenance],
    model_artifacts: Sequence[ModelArtifact],
    data_root: str | Path,
    data_manifest_before: Mapping[str, Any],
    fold_config_path: str | Path,
    incumbent_manifest: Mapping[str, Any],
    expected_reference: pd.DataFrame,
    metadata: Mapping[str, Any],
    bundle_name: str = DEFAULT_BUNDLE_NAME,
) -> ExternalArtifactBundle:
    """Validate, stage, disk-re-read, and atomically commit one OOF bundle.

    Existing completed bundles are never overwritten.  The caller must pass a
    locked reference containing exact keys, truth, and the incumbent column.
    Any identity fallback must be handled before publication; completed
    candidate schemas always record zero fallback rows.
    """

    run_id = _identifier(run_id, name="run_id")
    profile = str(profile)
    if profile not in PROFILE_NAMES:
        raise ValueError(f"profile must be one of {PROFILE_NAMES}")
    bundle_name = _identifier(bundle_name, name="bundle_name")
    root = Path(run_root).resolve()
    data = Path(data_root).resolve()
    if root == data or data in root.parents:
        raise ValueError("artifact run root cannot be inside competition data")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / bundle_name
    if destination.exists():
        raise FileExistsError(
            f"completed artifact bundle already exists: {destination}"
        )
    submissions = _submission_csvs(root)
    if submissions:
        raise ValueError(f"run root already contains submission CSV: {submissions}")

    normalized_candidates = _candidate_tuple(candidates)
    normalized_folds = _fold_tuple(fold_provenance, profile=profile)
    normalized_models = _model_tuple(model_artifacts)
    before = validate_data_manifest(data_manifest_before)
    after = capture_data_manifest(data)
    if before != after:
        raise ValueError("data manifest changed before artifact finalization")
    fold_path = Path(fold_config_path).resolve()
    if not fold_path.is_file():
        raise FileNotFoundError(f"fold config does not exist: {fold_path}")
    fold_config = {
        "path": str(fold_path),
        "sha256": sha256_file(fold_path),
        "bytes": fold_path.stat().st_size,
    }
    incumbent = _validate_incumbent_manifest(incumbent_manifest)
    metadata_json = _json_copy(dict(metadata), name="metadata")
    canonical, oof_validation = _canonicalize_and_validate_oof(
        oof,
        candidates=normalized_candidates,
        folds=normalized_folds,
        expected_reference=expected_reference,
    )
    _validate_models(
        root,
        candidates=normalized_candidates,
        folds=normalized_folds,
        models=normalized_models,
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{bundle_name}.tmp-", dir=root)
    ).resolve()
    committed = False
    try:
        staged_oof = temporary / OOF_FILENAME
        canonical.to_csv(
            staged_oof,
            index=False,
            float_format="%.17g",
            compression={"method": "gzip", "mtime": 0},
        )
        with staged_oof.open("rb+") as handle:
            os.fsync(handle.fileno())
        disk_oof = _read_oof(staged_oof)
        _, disk_validation = _canonicalize_and_validate_oof(
            disk_oof,
            candidates=normalized_candidates,
            folds=normalized_folds,
            expected_reference=canonical,
        )
        for field in (
            "rows",
            "wells",
            "folds",
            "prediction_columns",
            "key_sha256",
            "column_order",
            "truth_sha256",
            "incumbent_sha256",
        ):
            if disk_validation[field] != oof_validation[field]:
                raise ValueError(
                    f"OOF disk round trip changed validated field {field!r}"
                )
        oof_record = {
            **disk_validation,
            "path": OOF_FILENAME,
            "sha256": sha256_file(staged_oof),
            "bytes": staged_oof.stat().st_size,
            "truth_column": DEFAULT_TRUTH_COLUMN,
            "incumbent_column": INCUMBENT_COLUMN,
            "row_coordinate_column": ROW_COORDINATE_COLUMN,
            "candidates": [candidate.as_dict() for candidate in normalized_candidates],
        }
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "completed": True,
            "run_id": run_id,
            "profile": profile,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "submission_csv_created": False,
            "data_root": str(data),
            "data_read_only_manifest_before": before,
            "data_read_only_manifest_after": after,
            "fold_config": fold_config,
            "incumbent": {
                "column": INCUMBENT_COLUMN,
                "prediction_sha256": disk_validation["incumbent_sha256"],
                "state_manifest": incumbent,
            },
            "oof": oof_record,
            "fold_provenance": [record.as_dict() for record in normalized_folds],
            "model_artifacts": [record.as_dict() for record in normalized_models],
            "metadata": metadata_json,
        }
        _validate_manifest_header(manifest)
        manifest_path = temporary / MANIFEST_FILENAME
        _write_json(manifest_path, manifest)
        manifest_digest = sha256_file(manifest_path)
        _write_manifest_sidecar(temporary / MANIFEST_SHA256_FILENAME, manifest_digest)
        _audit_bundle(
            temporary,
            run_root=root,
            expected_reference=canonical,
            verify_current_data=True,
            data_root_override=data,
        )
        os.replace(temporary, destination)
        committed = True
        return _audit_bundle(
            destination,
            run_root=root,
            expected_reference=canonical,
            verify_current_data=True,
            data_root_override=data,
        )
    except Exception:
        cleanup = destination if committed else temporary
        if cleanup.exists():
            shutil.rmtree(cleanup)
        raise


def read_external_artifact_bundle(
    run_root: str | Path,
    *,
    bundle_name: str = DEFAULT_BUNDLE_NAME,
    expected_reference: pd.DataFrame | None = None,
) -> ExternalArtifactBundle:
    """Read and validate a bundle without rescanning the current data root."""

    root = Path(run_root).resolve()
    bundle = root / _identifier(bundle_name, name="bundle_name")
    if not bundle.is_dir():
        raise FileNotFoundError(f"artifact bundle does not exist: {bundle}")
    return _audit_bundle(
        bundle,
        run_root=root,
        expected_reference=expected_reference,
        verify_current_data=False,
        data_root_override=None,
    )


def independent_reaudit_external_artifacts(
    run_root: str | Path,
    *,
    bundle_name: str = DEFAULT_BUNDLE_NAME,
    expected_reference: pd.DataFrame | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute all checks from disk, including the current data manifest."""

    root = Path(run_root).resolve()
    bundle = root / _identifier(bundle_name, name="bundle_name")
    if not bundle.is_dir():
        raise FileNotFoundError(f"artifact bundle does not exist: {bundle}")
    result = _audit_bundle(
        bundle,
        run_root=root,
        expected_reference=expected_reference,
        verify_current_data=True,
        data_root_override=None if data_root is None else Path(data_root),
    )
    return dict(result.audit)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DATA_MANIFEST_FIELDS",
    "DEFAULT_AUDITED_INCUMBENT_OOF",
    "DEFAULT_BUNDLE_NAME",
    "ExternalArtifactBundle",
    "FoldProvenance",
    "INCUMBENT_COLUMN",
    "INDEPENDENT_REAUDIT_CHECKLIST",
    "MANIFEST_FILENAME",
    "MANIFEST_SHA256_FILENAME",
    "MODEL_FIT_SCOPES",
    "ModelArtifact",
    "OOF_FILENAME",
    "OofCandidateSchema",
    "ROW_COORDINATE_COLUMN",
    "AUDITED_INCUMBENT_SOURCE_COLUMN",
    "AUDITED_REFERENCE_ATOL",
    "canonical_oof_columns",
    "capture_data_manifest",
    "independent_reaudit_external_artifacts",
    "make_model_artifact",
    "read_external_artifact_bundle",
    "load_audited_incumbent_reference",
    "select_audited_incumbent_reference",
    "sha256_file",
    "validate_data_manifest",
    "well_ids_sha256",
    "write_external_artifact_bundle",
]
