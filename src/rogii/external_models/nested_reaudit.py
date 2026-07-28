"""Independent disk-only re-audit for the strictly nested TCN ensemble.

This module intentionally accepts a completed run directory, not runner-owned
Python objects.  It reconstructs every outer-fold blend from persisted inner
OOF predictions, re-reads the independently locked outer source, recomputes
metrics and weights, and verifies the complete checksum/provenance chain.

Production evidence is anchored to the canonical external-model artifact
bundle and the independently re-audited Heel incumbent.  Evidence-ineligible
smoke runs require an explicit source OOF file so truth and outer predictions
are not accepted from the nested run itself.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..diagnostics.nested_external_ensemble import (
    CANONICAL_TCN_COLUMN,
    FAMILY_MAX_WEIGHT,
    FIXED_050_COLUMN,
    FIXED_075_COLUMN,
    FIXED_INCUMBENT_HEAVY,
    FIXED_SIMPLE_AVERAGE,
    INCUMBENT_COLUMN,
    NESTED_COLUMN,
    REGULARIZATION_STRENGTH,
    REGULARIZATION_TARGET,
    SCHEMA_VERSION as NESTED_RUN_SCHEMA_VERSION,
    TCN_COLUMN,
)
from ..ensemble import (
    NestedBlendProvenance,
    NestedFoldBlendResult,
    fit_nested_fold_blend,
    nested_weight_table,
)
from ..folds import FOLD_COUNT, load_fold_mapping, partition_well_ids
from ..portfolio_metrics import (
    DEFAULT_BOOTSTRAP_SEED,
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
from .artifacts import (
    DEFAULT_AUDITED_INCUMBENT_OOF,
    DEFAULT_BUNDLE_NAME,
    capture_data_manifest,
    independent_reaudit_external_artifacts,
    load_audited_incumbent_reference,
    read_external_artifact_bundle,
    sha256_file,
    validate_data_manifest,
    well_ids_sha256,
)


NESTED_REAUDIT_SCHEMA_VERSION = "nested_external_tcn_independent_reaudit_v1"
DEFAULT_OUTPUT_NAME = "independent_nested_reaudit.json"
DEFAULT_VALUE_ATOL = 1.0e-9
DEFAULT_SUMMARY_ATOL = 5.0e-9
DEFAULT_SUMMARY_RTOL = 5.0e-11

PREDICTION_COLUMNS = (
    INCUMBENT_COLUMN,
    TCN_COLUMN,
    NESTED_COLUMN,
    FIXED_075_COLUMN,
    FIXED_050_COLUMN,
)


class NestedReauditError(RuntimeError):
    """Raised when a completed nested run fails independent disk re-audit."""


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise NestedReauditError(f"{label} is not valid finite JSON: {path}") from error
    if not isinstance(payload, dict):
        raise NestedReauditError(f"{label} must contain a JSON object: {path}")
    return payload, digest


def _read_oof(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        frame = pd.read_csv(
            path,
            dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
            float_precision="round_trip",
        )
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise NestedReauditError(f"{label} is unreadable: {path}") from error
    if "well_id" in frame:
        frame["well_id"] = frame["well_id"].astype(str)
    return frame


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


def _safe_output(root: Path, name: str) -> Path:
    candidate = Path(str(name))
    if (
        not str(name)
        or candidate.is_absolute()
        or candidate.name != str(name)
        or str(name) in {".", ".."}
    ):
        raise ValueError("output_name must be one filename inside the run root")
    return root / candidate


def _run_path(root: Path, value: Any, *, label: str) -> Path:
    declared = Path(str(value))
    resolved = (
        declared.resolve() if declared.is_absolute() else (root / declared).resolve()
    )
    if resolved != root and root not in resolved.parents:
        raise NestedReauditError(f"{label} escapes the nested run root: {value}")
    return resolved


def _jsonable(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=lambda item: item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


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


def _compare_json(
    actual: Any,
    expected: Any,
    *,
    label: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    """Compare finite JSON values recursively and return the largest float delta."""

    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise NestedReauditError(f"{label} JSON types differ")
        if set(actual) != set(expected):
            raise NestedReauditError(
                f"{label} JSON keys differ: "
                f"missing={sorted(set(expected) - set(actual))}, "
                f"extra={sorted(set(actual) - set(expected))}"
            )
        maximum = 0.0
        for key in expected:
            maximum = max(
                maximum,
                _compare_json(
                    actual[key],
                    expected[key],
                    label=f"{label}.{key}",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                ),
            )
        return maximum
    if isinstance(actual, list) or isinstance(expected, list):
        if (
            not isinstance(actual, list)
            or not isinstance(expected, list)
            or len(actual) != len(expected)
        ):
            raise NestedReauditError(f"{label} JSON list shape differs")
        maximum = 0.0
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            maximum = max(
                maximum,
                _compare_json(
                    left,
                    right,
                    label=f"{label}[{index}]",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                ),
            )
        return maximum
    if isinstance(actual, bool) or isinstance(expected, bool):
        if actual is not expected:
            raise NestedReauditError(f"{label} differs: {actual!r} != {expected!r}")
        return 0.0
    numeric = isinstance(actual, (int, float, np.number)) and isinstance(
        expected, (int, float, np.number)
    )
    if numeric:
        left = float(actual)
        right = float(expected)
        if not math.isfinite(left) or not math.isfinite(right):
            raise NestedReauditError(f"{label} contains a non-finite number")
        delta = abs(left - right)
        if not math.isclose(
            left,
            right,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise NestedReauditError(
                f"{label} differs: {left!r} != {right!r}; delta={delta}"
            )
        return delta
    if actual != expected:
        raise NestedReauditError(f"{label} differs: {actual!r} != {expected!r}")
    return 0.0


def _sorted_oof(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["well_id"] = result["well_id"].astype(str)
    return result.sort_values(list(OOF_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def _compare_oof_values(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
    value_atol: float,
) -> dict[str, float]:
    left = _sorted_oof(actual)
    right = _sorted_oof(expected)
    left_keys = left.loc[:, list(OOF_KEY_COLUMNS)].reset_index(drop=True)
    right_keys = right.loc[:, list(OOF_KEY_COLUMNS)].reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise NestedReauditError(f"{label} OOF keys or key order differ")
    deltas: dict[str, float] = {}
    for column in columns:
        if column not in left or column not in right:
            raise NestedReauditError(f"{label} is missing comparison column {column}")
        left_value = pd.to_numeric(left[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        right_value = pd.to_numeric(right[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(left_value).all() or not np.isfinite(right_value).all():
            raise NestedReauditError(f"{label} column {column} is non-finite")
        maximum = float(np.max(np.abs(left_value - right_value)))
        if not np.allclose(
            left_value,
            right_value,
            rtol=0.0,
            atol=value_atol,
        ):
            raise NestedReauditError(
                f"{label} column {column} differs; maximum_absolute_delta={maximum}"
            )
        deltas[column] = maximum
    return deltas


def _submission_csvs(root: Path) -> list[str]:
    matches: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or "submission" not in path.name.lower():
            continue
        lower = path.name.lower()
        if lower.endswith(".csv") or lower.endswith(".csv.gz"):
            matches.append(path.relative_to(root).as_posix())
    return sorted(matches)


def _metric_payload(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    shadow_samples: int,
    shadow_subset_wells: int,
) -> dict[str, Any]:
    predictions = PREDICTION_COLUMNS
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
        samples=bootstrap_samples,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    shadow: dict[str, Any] | None = None
    if shadow_subset_wells <= int(frame["well_id"].nunique()):
        shadow = shadow_private_subset_simulation(
            frame,
            baseline_column=INCUMBENT_COLUMN,
            candidate_columns=predictions[1:],
            samples=shadow_samples,
            subset_wells=shadow_subset_wells,
            seed=DEFAULT_BOOTSTRAP_SEED,
        )
    correlation = residual_correlation(frame, predictions)
    wins = per_well_win_summary(
        frame,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=predictions[1:],
    )
    return _jsonable(
        {
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
    )


def _load_smoke_source(
    path: Path,
    *,
    run_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_oof(path.resolve(), label="smoke source OOF")
    if TCN_COLUMN not in frame and CANONICAL_TCN_COLUMN in frame:
        frame = frame.rename(columns={CANONICAL_TCN_COLUMN: TCN_COLUMN})
    required = [*OOF_KEY_COLUMNS, "y_true", INCUMBENT_COLUMN, TCN_COLUMN]
    missing = set(required) - set(frame)
    if missing:
        raise NestedReauditError(
            f"smoke source OOF is missing columns: {sorted(missing)}"
        )
    frame = _sorted_oof(frame.loc[:, required])
    validation = validate_oof_frame(frame, (INCUMBENT_COLUMN, TCN_COLUMN))
    source = run_config["source"]
    root = str(source.get("root", ""))
    if not root.startswith("smoke://") or len(root) <= len("smoke://"):
        raise NestedReauditError("smoke source identifier is absent from run_config")
    source_id = root[len("smoke://") :]
    digest = _canonical_sha256(
        {
            "source_id": source_id,
            "validation": validation,
            "value_hashes": {
                column: (
                    _string_sequence_sha256(frame[column].astype(str).to_numpy())
                    if column == "well_id"
                    else _array_sha256(frame[column].to_numpy())
                )
                for column in required
            },
        }
    )
    expected_artifacts = [f"smoke-source:{source_id}:{digest}"]
    if (
        source.get("oof_sha256") != digest
        or source.get("manifest_sha256") != digest
        or source.get("artifact_ids") != expected_artifacts
        or source.get("evidence_eligible") is not False
    ):
        raise NestedReauditError(
            "smoke source values differ from immutable run provenance"
        )
    return frame, {
        "mode": "smoke",
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "source_digest": digest,
        "manifest_sha256": digest,
        "artifact_ids": expected_artifacts,
        "validation": validation,
        "independently_locked": False,
    }


def _locked_incumbent_sha(source_artifacts: Sequence[Any]) -> str | None:
    values = [
        str(value).split(":", 1)[1]
        for value in source_artifacts
        if str(value).startswith("locked-incumbent:")
    ]
    if not values:
        return None
    if len(values) != 1 or len(values[0]) != 64:
        raise NestedReauditError("locked-incumbent source artifact is malformed")
    return values[0]


def _load_production_source(
    *,
    run_config: Mapping[str, Any],
    data_root: Path,
    source_root: Path | None,
    incumbent_oof: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_record = run_config["source"]
    root = (
        Path(source_root).resolve()
        if source_root is not None
        else Path(str(source_record["root"])).resolve()
    )
    expected_incumbent_sha = _locked_incumbent_sha(
        source_record.get("artifact_ids", ())
    )
    locked, incumbent_record = load_audited_incumbent_reference(
        incumbent_oof,
        expected_sha256=expected_incumbent_sha,
    )
    source_audit = independent_reaudit_external_artifacts(
        root,
        bundle_name=DEFAULT_BUNDLE_NAME,
        expected_reference=locked,
        data_root=data_root,
    )
    if source_audit.get("passed") is not True or source_audit.get("profile") != "full":
        raise NestedReauditError(
            "production outer source is not an independently re-audited full bundle"
        )
    bundle = read_external_artifact_bundle(
        root,
        bundle_name=DEFAULT_BUNDLE_NAME,
        expected_reference=locked,
    )
    manifest_path = bundle.bundle_root / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    oof_sha = str(bundle.manifest["oof"]["sha256"])
    if (
        source_record.get("oof_sha256") != oof_sha
        or source_record.get("manifest_sha256") != manifest_sha
        or source_record.get("evidence_eligible") is not True
    ):
        raise NestedReauditError(
            "production source hashes differ from immutable run provenance"
        )
    candidates = [
        value
        for value in bundle.manifest["oof"]["candidates"]
        if value.get("candidate_id") == "tcn"
    ]
    if len(candidates) != 1:
        raise NestedReauditError(
            "production source does not contain exactly one TCN candidate"
        )
    candidate = candidates[0]
    if (
        candidate.get("prediction_column") != CANONICAL_TCN_COLUMN
        or candidate.get("family_id") != "tcn"
        or candidate.get("control_id") != "none"
        or candidate.get("transfer_mode") != "architecture_only"
        or candidate.get("eligible_for_model_selection") is not True
        or int(candidate.get("identity_fallback_rows", -1)) != 0
    ):
        raise NestedReauditError(
            "production source TCN candidate provenance is not eligible"
        )
    artifact_ids = [
        f"outer-tcn-bundle:{bundle.manifest['run_id']}:{manifest_sha}",
        f"locked-incumbent:{incumbent_record['sha256']}",
    ]
    if source_record.get("artifact_ids") != artifact_ids:
        raise NestedReauditError(
            "production source artifact IDs differ from audited source files"
        )
    frame = bundle.oof.loc[
        :,
        [
            *OOF_KEY_COLUMNS,
            "y_true",
            INCUMBENT_COLUMN,
            CANONICAL_TCN_COLUMN,
        ],
    ].rename(columns={CANONICAL_TCN_COLUMN: TCN_COLUMN})
    frame = _sorted_oof(frame)
    validation = validate_oof_frame(
        frame,
        (INCUMBENT_COLUMN, TCN_COLUMN),
        expected_keys=locked,
        expected_folds=range(FOLD_COUNT),
    )
    return frame, {
        "mode": "production",
        "root": str(root),
        "oof_path": str(bundle.bundle_root / "oof_predictions.csv.gz"),
        "oof_sha256": oof_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "artifact_ids": artifact_ids,
        "incumbent": incumbent_record,
        "bundle_reaudit": source_audit,
        "candidate": candidate,
        "validation": validation,
        "independently_locked": True,
    }


def _validate_fixed_controls(frame: pd.DataFrame, *, value_atol: float) -> None:
    incumbent = frame[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
    tcn = frame[TCN_COLUMN].to_numpy(dtype=np.float64)
    for column, weights in (
        (FIXED_075_COLUMN, FIXED_INCUMBENT_HEAVY),
        (FIXED_050_COLUMN, FIXED_SIMPLE_AVERAGE),
    ):
        expected = float(weights["incumbent"]) * incumbent + float(weights["tcn"]) * tcn
        actual = frame[column].to_numpy(dtype=np.float64)
        if not np.allclose(actual, expected, rtol=0.0, atol=value_atol):
            maximum = float(np.max(np.abs(actual - expected)))
            raise NestedReauditError(
                f"{column} algebra differs; maximum_absolute_delta={maximum}"
            )


def _validate_family_mean(
    frame: pd.DataFrame,
    *,
    tcn_seeds: Sequence[int],
    value_atol: float,
) -> None:
    seeds = tuple(map(int, tcn_seeds))
    seed_columns = [f"{TCN_COLUMN}__seed_{seed}" for seed in seeds]
    expected = frame[seed_columns].to_numpy(dtype=np.float64).mean(axis=1)
    actual = frame[TCN_COLUMN].to_numpy(dtype=np.float64)
    if not np.allclose(actual, expected, rtol=0.0, atol=value_atol):
        raise NestedReauditError("inner TCN mean is not the family-first seed mean")
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
            atol=value_atol,
        ):
            raise NestedReauditError(
                f"inner TCN {component} is not the family-first seed mean"
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
        atol=max(value_atol, 1.0e-8),
    ):
        raise NestedReauditError("inner TCN residual-component algebra differs")


def _expected_model_record(
    checkpoint: Mapping[str, Any],
    *,
    outer_fold: int,
    inner_fold: int,
) -> dict[str, Any]:
    provenance = checkpoint["provenance"]
    return {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "seed": int(provenance["seed"]),
        "checkpoint": checkpoint["checkpoint"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "sidecar": checkpoint["sidecar"],
        "sidecar_sha256": checkpoint["sidecar_sha256"],
        "fit_scope": "inner_training",
        "fit_wells": len(provenance["training_well_ids"]),
        "fit_ids_sha256": provenance["training_ids_sha256"],
        "validation_wells": len(provenance["validation_well_ids"]),
        "excluded_outer_wells": len(provenance["excluded_outer_well_ids"]),
        "reciprocal_state_reuse": checkpoint.get("reciprocal_state_reuse"),
        "state_manifest": checkpoint["state_manifest"],
    }


def _validate_tcn_checkpoints(
    root: Path,
    records: Any,
    *,
    outer_fold: int,
    inner_fold: int,
    training_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    outer_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    run_config: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[Path]]:
    if not isinstance(records, list) or len(records) != len(seeds):
        raise NestedReauditError("inner TCN checkpoint count differs from policy")
    if tuple(int(value["provenance"]["seed"]) for value in records) != seeds:
        raise NestedReauditError("inner TCN checkpoint seed order differs")
    expected_records: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for record, seed in zip(records, seeds, strict=True):
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise NestedReauditError("TCN checkpoint provenance is missing")
        if (
            provenance.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
            or provenance.get("family") != "tcn"
            or provenance.get("fit_scope") != "inner_training"
            or int(provenance.get("outer_fold", -1)) != outer_fold
            or int(provenance.get("inner_fold", -1)) != inner_fold
            or int(provenance.get("seed", -1)) != seed
            or tuple(map(str, provenance.get("training_well_ids", ()))) != training_ids
            or tuple(map(str, provenance.get("validation_well_ids", ())))
            != validation_ids
            or tuple(map(str, provenance.get("excluded_outer_well_ids", ())))
            != outer_ids
            or provenance.get("training_ids_sha256") != well_ids_sha256(training_ids)
            or provenance.get("training_validation_intersection") != []
            or provenance.get("training_outer_intersection") != []
            or provenance.get("outer_validation_labels_used_for_fit") is not False
            or provenance.get("inner_validation_labels_used_for_fit") is not False
            or provenance.get("labels_loaded_only_for_training_ids") is not True
            or provenance.get("data_manifest") != data_manifest
            or provenance.get("implementation_hashes")
            != run_config["implementation_hashes"]
        ):
            raise NestedReauditError(
                f"TCN checkpoint provenance differs for outer={outer_fold}, "
                f"inner={inner_fold}, seed={seed}"
            )
        config = provenance.get("config")
        if (
            not isinstance(config, Mapping)
            or config.get("family") != "tcn"
            or int(config.get("seed", -1)) != seed
            or int(config.get("epochs", -1)) != int(run_config["policy"]["epochs"])
            or int(config.get("hidden_dim", -1))
            != int(run_config["policy"]["hidden_dim"])
            or int(config.get("layers", -1)) != int(run_config["policy"]["layers"])
        ):
            raise NestedReauditError("TCN checkpoint config differs from run policy")
        checkpoint = _run_path(root, record.get("checkpoint"), label="TCN checkpoint")
        sidecar = _run_path(root, record.get("sidecar"), label="TCN sidecar")
        if sha256_file(checkpoint) != str(
            record.get("checkpoint_sha256")
        ) or sha256_file(sidecar) != str(record.get("sidecar_sha256")):
            raise NestedReauditError("TCN checkpoint or sidecar SHA-256 differs")
        sidecar_payload, _ = _read_json(sidecar, label="TCN sidecar")
        if (
            sidecar_payload.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
            or sidecar_payload.get("checkpoint_sha256")
            != record.get("checkpoint_sha256")
            or sidecar_payload.get("provenance") != provenance
            or sidecar_payload.get("state_manifest") != record.get("state_manifest")
            or sidecar_payload.get("reciprocal_state_reuse")
            != record.get("reciprocal_state_reuse")
        ):
            raise NestedReauditError("TCN sidecar differs from inner OOF manifest")
        reciprocal = record.get("reciprocal_state_reuse")
        if reciprocal is not None:
            if (
                not isinstance(reciprocal, Mapping)
                or set(reciprocal)
                != {
                    "source_checkpoint",
                    "source_sidecar",
                    "source_checkpoint_sha256",
                    "source_sidecar_sha256",
                    "equivalence_basis",
                }
                or not str(reciprocal.get("equivalence_basis", "")).strip()
            ):
                raise NestedReauditError(
                    "TCN reciprocal-state provenance schema differs"
                )
            source_checkpoint = _run_path(
                root,
                reciprocal.get("source_checkpoint"),
                label="TCN reciprocal source checkpoint",
            )
            source_sidecar = _run_path(
                root,
                reciprocal.get("source_sidecar"),
                label="TCN reciprocal source sidecar",
            )
            if sha256_file(source_checkpoint) != reciprocal.get(
                "source_checkpoint_sha256"
            ) or sha256_file(source_sidecar) != reciprocal.get("source_sidecar_sha256"):
                raise NestedReauditError(
                    "TCN reciprocal source checkpoint/sidecar SHA-256 differs"
                )
            source_payload, _ = _read_json(
                source_sidecar,
                label="TCN reciprocal source sidecar",
            )
            source_provenance = source_payload.get("provenance")
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
            if (
                not isinstance(source_provenance, Mapping)
                or any(
                    source_provenance.get(field) != provenance.get(field)
                    for field in invariant_fields
                )
                or int(source_provenance.get("outer_fold", -1)) != inner_fold
                or int(source_provenance.get("inner_fold", -1)) != outer_fold
                or tuple(
                    map(
                        str,
                        source_provenance.get("validation_well_ids", ()),
                    )
                )
                != outer_ids
                or tuple(
                    map(
                        str,
                        source_provenance.get("excluded_outer_well_ids", ()),
                    )
                )
                != validation_ids
                or source_payload.get("checkpoint_sha256")
                != reciprocal.get("source_checkpoint_sha256")
                or source_payload.get("state_manifest") != record.get("state_manifest")
            ):
                raise NestedReauditError(
                    "TCN reciprocal source is not training-equivalent"
                )
        state = record.get("state_manifest")
        if (
            not isinstance(state, Mapping)
            or state.get("family") != "tcn"
            or int(state.get("seed", -1)) != seed
            or state.get("fit_ids_sha256") != provenance.get("training_ids_sha256")
            or int(state.get("sample_count", -1)) != len(training_ids)
            or state.get("architecture_only") is not True
        ):
            raise NestedReauditError("TCN state manifest provenance differs")
        expected_records.append(
            _expected_model_record(
                record,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
        )
        paths.update((checkpoint, sidecar))
    return expected_records, paths


def _validate_incumbent_checkpoints(
    root: Path,
    records: Any,
    *,
    outer_fold: int,
    inner_fold: int,
    training_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    outer_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    fold_config_sha256: str,
    fold_version: str,
) -> tuple[set[Path], int]:
    if not isinstance(records, list) or len(records) != len(seeds):
        raise NestedReauditError("inner incumbent checkpoint count differs from policy")
    if tuple(int(value.get("seed", -1)) for value in records) != seeds:
        raise NestedReauditError("inner incumbent checkpoint seed order differs")
    paths: set[Path] = set()
    reciprocal_aliases = 0
    for record, seed in zip(records, seeds, strict=True):
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise NestedReauditError("incumbent checkpoint provenance is missing")
        if (
            int(provenance.get("outer_fold", -1)) != outer_fold
            or int(provenance.get("inner_validation_fold", -1)) != inner_fold
            or int(provenance.get("seed", -1)) != seed
            or tuple(map(str, provenance.get("ordered_training_well_ids", ())))
            != training_ids
            or tuple(map(str, provenance.get("validation_well_ids", ())))
            != validation_ids
            or tuple(map(str, provenance.get("excluded_outer_well_ids", ())))
            != outer_ids
            or provenance.get("fold_mapping_sha256") != fold_config_sha256
            or provenance.get("fold_version") != fold_version
        ):
            raise NestedReauditError("incumbent checkpoint provenance differs")
        checkpoint = _run_path(
            root, record.get("checkpoint_path"), label="incumbent checkpoint"
        )
        manifest_path = _run_path(
            root, record.get("manifest_path"), label="incumbent sidecar"
        )
        if sha256_file(checkpoint) != str(record.get("checkpoint_sha256")):
            raise NestedReauditError("incumbent checkpoint SHA-256 differs")
        sidecar, _ = _read_json(manifest_path, label="incumbent sidecar")
        if (
            sidecar.get("checkpoint_sha256") != record.get("checkpoint_sha256")
            or sidecar.get("provenance") != provenance
            or Path(str(sidecar.get("checkpoint_path"))).resolve() != checkpoint
        ):
            raise NestedReauditError(
                "incumbent sidecar differs from inner OOF manifest"
            )
        state = sidecar.get("state_provenance")
        if (
            not isinstance(state, Mapping)
            or tuple(map(str, state.get("training_well_ids", ()))) != training_ids
            or int(state.get("training_well_count", -1)) != len(training_ids)
            or int(state.get("random_seed", -1)) != seed
            or int(state.get("fold", -1)) != inner_fold
        ):
            raise NestedReauditError("incumbent checkpoint state provenance differs")
        source_fields = (
            record.get("reciprocal_source_checkpoint_path"),
            record.get("reciprocal_source_checkpoint_sha256"),
            record.get("reciprocal_source_manifest_path"),
            record.get("reciprocal_source_manifest_sha256"),
        )
        if any(value is None for value in source_fields) and not all(
            value is None for value in source_fields
        ):
            raise NestedReauditError(
                "incumbent reciprocal source provenance is partial"
            )
        if source_fields[0] is not None:
            reciprocal_aliases += 1
            source_checkpoint = _run_path(
                root,
                source_fields[0],
                label="incumbent reciprocal source checkpoint",
            )
            source_manifest_path = _run_path(
                root,
                source_fields[2],
                label="incumbent reciprocal source sidecar",
            )
            if (
                sha256_file(source_checkpoint) != source_fields[1]
                or sha256_file(source_manifest_path) != source_fields[3]
            ):
                raise NestedReauditError("incumbent reciprocal source SHA-256 differs")
            source_manifest, _ = _read_json(
                source_manifest_path,
                label="incumbent reciprocal source sidecar",
            )
            source_provenance = source_manifest.get("provenance")
            reciprocal_sidecar = sidecar.get("reciprocal_state_reuse")
            if (
                not isinstance(source_provenance, Mapping)
                or not isinstance(reciprocal_sidecar, Mapping)
                or reciprocal_sidecar.get("source_checkpoint_path") != source_fields[0]
                or reciprocal_sidecar.get("source_checkpoint_sha256")
                != source_fields[1]
                or reciprocal_sidecar.get("source_manifest_path") != source_fields[2]
                or reciprocal_sidecar.get("source_manifest_sha256") != source_fields[3]
                or int(reciprocal_sidecar.get("source_outer_fold", -1)) != inner_fold
                or int(reciprocal_sidecar.get("source_inner_validation_fold", -1))
                != outer_fold
                or int(reciprocal_sidecar.get("seed", -1)) != seed
                or not str(reciprocal_sidecar.get("equivalence_basis", "")).strip()
                or int(source_provenance.get("outer_fold", -1)) != inner_fold
                or int(source_provenance.get("inner_validation_fold", -1)) != outer_fold
                or int(source_provenance.get("seed", -1)) != seed
                or tuple(
                    map(
                        str,
                        source_provenance.get("ordered_training_well_ids", ()),
                    )
                )
                != training_ids
            ):
                raise NestedReauditError(
                    "incumbent reciprocal source is not training-equivalent"
                )
        elif sidecar.get("reciprocal_state_reuse") is not None:
            raise NestedReauditError(
                "incumbent sidecar declares unrecorded reciprocal reuse"
            )
        paths.update((checkpoint, manifest_path))
    return paths, reciprocal_aliases


def _outer_expected_frame(result: NestedFoldBlendResult) -> pd.DataFrame:
    frame = result.predictions.rename(
        columns={
            "y_pred_family__incumbent": INCUMBENT_COLUMN,
            "y_pred_family__tcn": TCN_COLUMN,
            result.prediction_column: NESTED_COLUMN,
        }
    )
    incumbent = frame[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
    tcn = frame[TCN_COLUMN].to_numpy(dtype=np.float64)
    frame[FIXED_075_COLUMN] = (
        FIXED_INCUMBENT_HEAVY["incumbent"] * incumbent
        + FIXED_INCUMBENT_HEAVY["tcn"] * tcn
    )
    frame[FIXED_050_COLUMN] = (
        FIXED_SIMPLE_AVERAGE["incumbent"] * incumbent
        + FIXED_SIMPLE_AVERAGE["tcn"] * tcn
    )
    return frame.loc[
        :,
        [
            *OOF_KEY_COLUMNS,
            "y_true",
            *PREDICTION_COLUMNS,
        ],
    ]


def audit_nested_external_ensemble(
    run_root: str | Path,
    *,
    source_oof: str | Path | None = None,
    source_root: str | Path | None = None,
    incumbent_oof: str | Path = DEFAULT_AUDITED_INCUMBENT_OOF,
    data_root: str | Path | None = None,
    fold_config: str | Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    value_atol: float = DEFAULT_VALUE_ATOL,
    summary_atol: float = DEFAULT_SUMMARY_ATOL,
    summary_rtol: float = DEFAULT_SUMMARY_RTOL,
) -> dict[str, Any]:
    """Reconstruct and audit one completed nested run entirely from disk."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"nested run root does not exist: {root}")
    for name, value in (
        ("value_atol", value_atol),
        ("summary_atol", summary_atol),
        ("summary_rtol", summary_rtol),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    output_path = _safe_output(root, output_name)
    config_path = root / "run_config.json"
    summary_path = root / "summary.json"
    artifact_audit_path = root / "artifact_audit.json"
    metrics_path = root / "metrics" / "summary.json"
    final_oof_path = root / "oof_predictions.csv.gz"
    weights_path = root / "nested_weights.csv"
    model_manifest_path = root / "model_manifest.json"
    progress_path = root / "progress.json"

    run_config, run_config_sha = _read_json(config_path, label="nested run_config")
    summary, summary_sha = _read_json(summary_path, label="nested summary")
    artifact_audit, artifact_audit_sha = _read_json(
        artifact_audit_path, label="nested runner artifact audit"
    )
    stored_metrics, metrics_sha = _read_json(metrics_path, label="nested metrics")
    model_manifest, model_manifest_sha = _read_json(
        model_manifest_path, label="nested model manifest"
    )
    progress, progress_sha = _read_json(progress_path, label="nested progress")
    if (
        run_config.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
        or summary.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
        or artifact_audit.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
        or model_manifest.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
        or progress.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
        or summary.get("completed") is not True
        or artifact_audit.get("passed") is not True
        or model_manifest.get("completed") is not True
    ):
        raise NestedReauditError(
            "nested completion/schema markers are absent or inconsistent"
        )
    if (
        run_config.get("submission_created") is not False
        or summary.get("submission_created") is not False
        or artifact_audit.get("submission_created") is not False
        or model_manifest.get("identity_fallbacks_scored") is not False
        or model_manifest.get("failures") != []
    ):
        raise NestedReauditError(
            "nested run declares submission, fallback, or model failures"
        )
    submissions = _submission_csvs(root)
    if submissions:
        raise NestedReauditError(
            f"submission CSV is present in nested run: {submissions}"
        )

    resolved_data = (
        Path(data_root).resolve()
        if data_root is not None
        else Path(str(run_config["data_root"])).resolve()
    )
    data_before = capture_data_manifest(resolved_data)
    recorded_data = validate_data_manifest(run_config["data_manifest_before"])
    if data_before != recorded_data:
        raise NestedReauditError(
            "current competition data manifest differs from nested run input"
        )
    resolved_folds = (
        Path(fold_config).resolve()
        if fold_config is not None
        else Path(str(run_config["fold_config"])).resolve()
    )
    if not resolved_folds.is_file() or sha256_file(resolved_folds) != run_config.get(
        "fold_config_sha256"
    ):
        raise NestedReauditError("fold config path or SHA-256 differs")

    evidence_mode = str(run_config.get("evidence_mode"))
    if evidence_mode == "production":
        if source_oof is not None:
            raise ValueError("production audit uses the canonical source bundle")
        source_frame, source_record = _load_production_source(
            run_config=run_config,
            data_root=resolved_data,
            source_root=None if source_root is None else Path(source_root),
            incumbent_oof=Path(incumbent_oof),
        )
    elif evidence_mode == "smoke":
        if source_oof is None:
            raise ValueError(
                "smoke audit requires --source-oof as an external disk reference"
            )
        if source_root is not None:
            raise ValueError("source_root is not used for smoke audits")
        source_frame, source_record = _load_smoke_source(
            Path(source_oof),
            run_config=run_config,
        )
    else:
        raise NestedReauditError(f"unknown nested evidence_mode: {evidence_mode}")

    mapping = load_fold_mapping(
        resolved_folds,
        expected_well_ids=tuple(sorted(source_frame["well_id"].astype(str).unique())),
    )
    if str(mapping["fold_version"].iat[0]) != str(run_config.get("fold_version")):
        raise NestedReauditError("fold version differs from run_config")
    source_validation = validate_oof_frame(
        source_frame,
        (INCUMBENT_COLUMN, TCN_COLUMN),
        expected_well_ids=tuple(sorted(mapping["well_id"].astype(str).unique())),
        expected_folds=range(FOLD_COUNT),
    )
    source_folds = (
        source_frame.loc[:, ["well_id", "fold"]]
        .drop_duplicates()
        .set_index("well_id")["fold"]
        .astype(int)
        .to_dict()
    )
    assigned_folds = mapping.set_index("well_id")["fold"].astype(int).to_dict()
    if source_folds != assigned_folds:
        raise NestedReauditError("source OOF folds differ from fold config")
    expected_cohort: dict[str, Any] = {}
    for current_fold in range(FOLD_COUNT):
        training_ids, validation_ids = partition_well_ids(mapping, current_fold)
        expected_cohort[str(current_fold)] = {
            "training_well_ids": list(training_ids),
            "validation_well_ids": list(validation_ids),
            "training_ids_sha256": well_ids_sha256(training_ids),
            "validation_ids_sha256": well_ids_sha256(validation_ids),
            "intersection": [],
        }
    if run_config.get("cohort") != expected_cohort:
        raise NestedReauditError("run_config cohort differs from fold mapping")

    policy = run_config.get("policy")
    blend_policy = run_config.get("blend_policy")
    if not isinstance(policy, Mapping) or not isinstance(blend_policy, Mapping):
        raise NestedReauditError("nested policy blocks are missing")
    outer_folds = tuple(map(int, policy.get("outer_folds", ())))
    tcn_seeds = tuple(map(int, policy.get("tcn_seeds", ())))
    incumbent_seeds = tuple(map(int, policy.get("incumbent_seeds", ())))
    if (
        not outer_folds
        or len(set(outer_folds)) != len(outer_folds)
        or any(value not in range(FOLD_COUNT) for value in outer_folds)
        or not tcn_seeds
        or len(set(tcn_seeds)) != len(tcn_seeds)
        or not incumbent_seeds
        or len(set(incumbent_seeds)) != len(incumbent_seeds)
    ):
        raise NestedReauditError("nested fold/seed policy is invalid")
    expected_blend_policy = {
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
    }
    if blend_policy != expected_blend_policy:
        raise NestedReauditError("nested blend policy differs from locked policy")
    if evidence_mode == "production" and (
        tuple(outer_folds) != tuple(range(FOLD_COUNT))
        or len(tcn_seeds) != 5
        or len(incumbent_seeds) != 5
        or int(policy.get("bootstrap_samples", -1)) != 10_000
        or int(policy.get("shadow_samples", -1)) != 100_000
        or int(policy.get("shadow_subset_wells", -1)) != 200
    ):
        raise NestedReauditError("production nested policy is not full strict OOF")

    run_config_digest = _canonical_sha256(run_config)
    policy_digest = _canonical_sha256(policy)
    expected_progress_inner = [
        {"outer_fold": outer_fold, "inner_fold": inner_fold}
        for outer_fold in outer_folds
        for inner_fold in range(FOLD_COUNT)
        if inner_fold != outer_fold
    ]
    if (
        progress.get("run_config_digest") != run_config_digest
        or progress.get("completed_inner_splits") != expected_progress_inner
        or progress.get("completed_outer_folds") != list(outer_folds)
        or int(progress.get("expected_inner_splits", -1))
        != len(expected_progress_inner)
        or int(progress.get("expected_outer_folds", -1)) != len(outer_folds)
    ):
        raise NestedReauditError("nested progress does not describe a complete run")

    final_oof = _read_oof(final_oof_path, label="nested final OOF")
    expected_final_source = source_frame.loc[
        source_frame["fold"].isin(outer_folds),
        [*OOF_KEY_COLUMNS, "y_true", INCUMBENT_COLUMN, TCN_COLUMN],
    ].copy()
    final_validation = validate_oof_frame(
        final_oof,
        PREDICTION_COLUMNS,
        expected_keys=expected_final_source,
        expected_well_ids=tuple(
            sorted(expected_final_source["well_id"].astype(str).unique())
        ),
        expected_folds=outer_folds,
    )
    source_value_deltas = _compare_oof_values(
        final_oof,
        expected_final_source,
        ("y_true", INCUMBENT_COLUMN, TCN_COLUMN),
        label="final versus locked outer source",
        value_atol=value_atol,
    )
    _validate_fixed_controls(final_oof, value_atol=value_atol)

    model_records: list[dict[str, Any]] = []
    declared_tcn_paths: set[Path] = set()
    declared_incumbent_paths: set[Path] = set()
    tcn_reciprocal_aliases = 0
    incumbent_checkpoint_count = 0
    incumbent_reciprocal_aliases = 0
    recomputed_results: list[NestedFoldBlendResult] = []
    expected_outer_frames: list[pd.DataFrame] = []
    actual_outer_manifests: list[dict[str, Any]] = []
    inner_records: list[dict[str, Any]] = []
    maximum_inner_truth_delta = 0.0
    maximum_outer_prediction_delta = 0.0
    for outer_fold in outer_folds:
        outer_training, outer_validation = partition_well_ids(mapping, outer_fold)
        inner_frames: list[pd.DataFrame] = []
        inner_artifact_records: list[dict[str, Any]] = []
        source_artifact_ids: list[str] = []
        for inner_fold in range(FOLD_COUNT):
            if inner_fold == outer_fold:
                continue
            inner_training = tuple(
                sorted(
                    mapping.loc[
                        ~mapping["fold"].isin((outer_fold, inner_fold)),
                        "well_id",
                    ].astype(str)
                )
            )
            _, inner_validation = partition_well_ids(mapping, inner_fold)
            inner_directory = (
                root / "inner_oof" / f"outer_{outer_fold}" / f"inner_{inner_fold}"
            )
            inner_oof_path = inner_directory / "oof_predictions.csv.gz"
            inner_manifest_path = inner_directory / "manifest.json"
            inner_manifest, inner_manifest_sha = _read_json(
                inner_manifest_path,
                label="inner OOF manifest",
            )
            if (
                inner_manifest.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
                or inner_manifest.get("artifact_kind") != "inner_oof"
                or inner_manifest.get("completed") is not True
                or int(inner_manifest.get("outer_fold", -1)) != outer_fold
                or int(inner_manifest.get("inner_fold", -1)) != inner_fold
                or tuple(map(str, inner_manifest.get("training_well_ids", ())))
                != inner_training
                or tuple(map(str, inner_manifest.get("validation_well_ids", ())))
                != inner_validation
                or tuple(
                    map(
                        str,
                        inner_manifest.get("excluded_outer_well_ids", ()),
                    )
                )
                != outer_validation
                or inner_manifest.get("training_ids_sha256")
                != well_ids_sha256(inner_training)
                or inner_manifest.get("validation_ids_sha256")
                != well_ids_sha256(inner_validation)
                or inner_manifest.get("excluded_outer_ids_sha256")
                != well_ids_sha256(outer_validation)
                or inner_manifest.get("training_validation_intersection") != []
                or inner_manifest.get("training_outer_intersection") != []
                or inner_manifest.get("validation_outer_intersection") != []
                or inner_manifest.get("policy_digest") != policy_digest
                or inner_manifest.get("run_config_digest") != run_config_digest
                or inner_manifest.get("source_oof_sha256")
                != run_config["source"]["oof_sha256"]
                or inner_manifest.get("truth_attached_only_after_all_seed_predictions")
                is not True
                or inner_manifest.get("outer_validation_labels_used_for_fit")
                is not False
                or inner_manifest.get("inner_validation_labels_used_for_fit")
                is not False
                or int(inner_manifest.get("rows", -1)) <= 0
                or _run_path(
                    root,
                    inner_manifest.get("oof_path"),
                    label="inner OOF path",
                )
                != inner_oof_path.resolve()
                or inner_manifest.get("oof_sha256") != sha256_file(inner_oof_path)
            ):
                raise NestedReauditError(
                    f"inner OOF provenance differs for outer={outer_fold}, "
                    f"inner={inner_fold}"
                )
            inner_frame = _read_oof(inner_oof_path, label="inner OOF")
            expected_truth = source_frame.loc[
                source_frame["well_id"].isin(inner_validation),
                [*OOF_KEY_COLUMNS, "y_true"],
            ]
            declared_predictions = tuple(
                map(str, inner_manifest.get("prediction_columns", ()))
            )
            expected_predictions = (
                INCUMBENT_COLUMN,
                TCN_COLUMN,
                *(f"{TCN_COLUMN}__seed_{seed}" for seed in tcn_seeds),
                *(
                    f"{TCN_COLUMN}__{component}"
                    for component in ("common", "slope", "shape")
                ),
                *(
                    f"{TCN_COLUMN}__seed_{seed}__{component}"
                    for seed in tcn_seeds
                    for component in ("common", "slope", "shape")
                ),
            )
            if declared_predictions != expected_predictions:
                raise NestedReauditError(
                    "inner OOF prediction schema differs from seed policy"
                )
            inner_validation_result = validate_oof_frame(
                inner_frame,
                expected_predictions,
                expected_keys=expected_truth,
                expected_well_ids=inner_validation,
                expected_folds=(inner_fold,),
            )
            if inner_manifest.get("validation") != inner_validation_result:
                raise NestedReauditError("inner OOF validation differs from manifest")
            truth_delta = _compare_oof_values(
                inner_frame,
                expected_truth,
                ("y_true",),
                label="inner OOF versus locked truth",
                value_atol=value_atol,
            )["y_true"]
            maximum_inner_truth_delta = max(maximum_inner_truth_delta, truth_delta)
            _validate_family_mean(
                inner_frame,
                tcn_seeds=tcn_seeds,
                value_atol=value_atol,
            )
            current_models, current_tcn_paths = _validate_tcn_checkpoints(
                root,
                inner_manifest.get("tcn_checkpoints"),
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                training_ids=inner_training,
                validation_ids=inner_validation,
                outer_ids=outer_validation,
                seeds=tcn_seeds,
                run_config=run_config,
                data_manifest=recorded_data,
            )
            (
                current_incumbent_paths,
                current_incumbent_aliases,
            ) = _validate_incumbent_checkpoints(
                root,
                inner_manifest.get("incumbent_checkpoints"),
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                training_ids=inner_training,
                validation_ids=inner_validation,
                outer_ids=outer_validation,
                seeds=incumbent_seeds,
                fold_config_sha256=str(run_config["fold_config_sha256"]),
                fold_version=str(run_config["fold_version"]),
            )
            model_records.extend(current_models)
            tcn_reciprocal_aliases += sum(
                record.get("reciprocal_state_reuse") is not None
                for record in inner_manifest["tcn_checkpoints"]
            )
            incumbent_checkpoint_count += len(inner_manifest["incumbent_checkpoints"])
            incumbent_reciprocal_aliases += current_incumbent_aliases
            declared_tcn_paths.update(current_tcn_paths)
            declared_incumbent_paths.update(current_incumbent_paths)
            inner_frames.append(inner_frame)
            artifact_id = (
                f"inner-tcn-oof:outer_{outer_fold}:inner_{inner_fold}:"
                f"{inner_manifest['oof_sha256']}"
            )
            source_artifact_ids.append(artifact_id)
            inner_artifact_record = {
                "artifact_id": artifact_id,
                "oof_path": inner_manifest["oof_path"],
                "oof_sha256": inner_manifest["oof_sha256"],
                "manifest_path": inner_manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": inner_manifest_sha,
            }
            inner_artifact_records.append(inner_artifact_record)
            inner_records.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "rows": len(inner_frame),
                    "wells": int(inner_frame["well_id"].nunique()),
                    "oof_sha256": inner_manifest["oof_sha256"],
                    "manifest_sha256": inner_manifest_sha,
                    "training_wells": len(inner_training),
                    "validation_wells": len(inner_validation),
                    "tcn_checkpoints": len(tcn_seeds),
                    "incumbent_checkpoints": len(incumbent_seeds),
                }
            )

        inner_oof = _sorted_oof(pd.concat(inner_frames, ignore_index=True))
        validate_oof_frame(
            inner_oof,
            (INCUMBENT_COLUMN, TCN_COLUMN),
            expected_keys=source_frame.loc[
                source_frame["well_id"].isin(outer_training),
                [*OOF_KEY_COLUMNS, "y_true"],
            ],
            expected_well_ids=outer_training,
            expected_folds=tuple(
                value for value in range(FOLD_COUNT) if value != outer_fold
            ),
        )
        provenance = NestedBlendProvenance(
            outer_fold=outer_fold,
            outer_training_well_ids=outer_training,
            outer_validation_well_ids=outer_validation,
            source_artifacts=tuple(
                sorted(
                    (
                        *map(str, run_config["source"]["artifact_ids"]),
                        *source_artifact_ids,
                    )
                )
            ),
        )
        outer_source = source_frame.loc[
            source_frame["well_id"].isin(outer_validation),
            [*OOF_KEY_COLUMNS, "y_true", INCUMBENT_COLUMN, TCN_COLUMN],
        ].copy()
        result = fit_nested_fold_blend(
            inner_oof,
            outer_source,
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
        expected_outer = _outer_expected_frame(result)
        outer_directory = root / "outer_oof" / f"fold_{outer_fold}"
        outer_oof_path = outer_directory / "oof_predictions.csv.gz"
        blend_path = outer_directory / "blend.json"
        persisted_outer = _read_oof(outer_oof_path, label="outer nested OOF")
        blend_manifest, _ = _read_json(blend_path, label="outer blend manifest")
        outer_validation_result = validate_oof_frame(
            persisted_outer,
            PREDICTION_COLUMNS,
            expected_keys=outer_source,
            expected_well_ids=outer_validation,
            expected_folds=(outer_fold,),
        )
        deltas = _compare_oof_values(
            persisted_outer,
            expected_outer,
            ("y_true", *PREDICTION_COLUMNS),
            label="outer OOF versus independently recomputed blend",
            value_atol=value_atol,
        )
        maximum_outer_prediction_delta = max(
            maximum_outer_prediction_delta, *deltas.values()
        )
        _validate_fixed_controls(persisted_outer, value_atol=value_atol)
        expected_blend_static = {
            "schema_version": NESTED_RUN_SCHEMA_VERSION,
            "artifact_kind": "outer_nested_blend",
            "completed": True,
            "outer_fold": outer_fold,
            "provenance": asdict(provenance),
            "provenance_digest": provenance.digest,
            "regularization_strength": REGULARIZATION_STRENGTH,
            "regularization_target": REGULARIZATION_TARGET,
            "family_max_weight": FAMILY_MAX_WEIGHT,
            "fixed_controls": {
                FIXED_075_COLUMN: FIXED_INCUMBENT_HEAVY,
                FIXED_050_COLUMN: FIXED_SIMPLE_AVERAGE,
            },
            "inner_artifacts": inner_artifact_records,
            "source_oof_sha256": run_config["source"]["oof_sha256"],
            "validation": outer_validation_result,
            "oof_path": outer_oof_path.relative_to(root).as_posix(),
            "oof_sha256": sha256_file(outer_oof_path),
        }
        for key, value in expected_blend_static.items():
            if blend_manifest.get(key) != _jsonable(value):
                raise NestedReauditError(f"outer blend manifest field differs: {key}")
        _compare_json(
            blend_manifest.get("blend"),
            _jsonable(result.summary()),
            label=f"outer_fold_{outer_fold}.blend",
            absolute_tolerance=summary_atol,
            relative_tolerance=summary_rtol,
        )
        if not isinstance(blend_manifest.get("created_utc"), str):
            raise NestedReauditError("outer blend creation timestamp is absent")
        recomputed_results.append(result)
        expected_outer_frames.append(expected_outer)
        actual_outer_manifests.append(blend_manifest)

    expected_final = _sorted_oof(pd.concat(expected_outer_frames, ignore_index=True))
    final_prediction_deltas = _compare_oof_values(
        final_oof,
        expected_final,
        ("y_true", *PREDICTION_COLUMNS),
        label="final OOF versus recomputed outer folds",
        value_atol=value_atol,
    )
    maximum_outer_prediction_delta = max(
        maximum_outer_prediction_delta, *final_prediction_deltas.values()
    )

    expected_weights = nested_weight_table(recomputed_results)
    try:
        disk_weights = pd.read_csv(weights_path, float_precision="round_trip")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise NestedReauditError("nested weights CSV is unreadable") from error
    expected_weight_columns = list(expected_weights.columns)
    if list(disk_weights.columns) != expected_weight_columns:
        raise NestedReauditError("nested weights CSV columns differ")
    disk_weight_records = _jsonable(disk_weights.to_dict("records"))
    expected_weight_records = _jsonable(expected_weights.to_dict("records"))
    weight_maximum_delta = _compare_json(
        disk_weight_records,
        expected_weight_records,
        label="nested_weights",
        absolute_tolerance=summary_atol,
        relative_tolerance=summary_rtol,
    )
    for outer_fold, group in disk_weights.groupby("outer_fold", sort=True):
        if (
            set(group["family"].astype(str)) != {"incumbent", "tcn"}
            or not math.isclose(
                float(group["weight"].sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or (group["weight"].to_numpy(dtype=np.float64) < 0.0).any()
            or float(group.loc[group["family"].eq("tcn"), "weight"].iat[0])
            > FAMILY_MAX_WEIGHT["tcn"] + 1.0e-9
        ):
            raise NestedReauditError(
                f"nested weights violate constraints in outer fold {outer_fold}"
            )

    model_records = sorted(
        model_records,
        key=lambda value: (
            int(value["outer_fold"]),
            int(value["inner_fold"]),
            int(value["seed"]),
        ),
    )
    oriented_splits = len(outer_folds) * (FOLD_COUNT - 1)
    untouched_folds = FOLD_COUNT - len(outer_folds)
    unique_excluded_pairs = (
        FOLD_COUNT * (FOLD_COUNT - 1) // 2
        - untouched_folds * (untouched_folds - 1) // 2
    )
    expected_tcn_aliases = (oriented_splits - unique_excluded_pairs) * len(tcn_seeds)
    expected_incumbent_aliases = (oriented_splits - unique_excluded_pairs) * len(
        incumbent_seeds
    )
    if (
        tcn_reciprocal_aliases != expected_tcn_aliases
        or incumbent_reciprocal_aliases != expected_incumbent_aliases
    ):
        raise NestedReauditError(
            "reciprocal alias counts differ from unordered fold-pair plan"
        )
    if (
        model_manifest.get("models") != _jsonable(model_records)
        or int(model_manifest.get("model_count", -1)) != len(model_records)
        or int(model_manifest.get("physical_fit_count", -1))
        != len(model_records) - tcn_reciprocal_aliases
        or int(model_manifest.get("reciprocal_alias_count", -1))
        != tcn_reciprocal_aliases
        or int(model_manifest.get("incumbent_checkpoint_count", -1))
        != incumbent_checkpoint_count
        or int(model_manifest.get("incumbent_physical_fit_count", -1))
        != incumbent_checkpoint_count - incumbent_reciprocal_aliases
        or int(model_manifest.get("incumbent_reciprocal_alias_count", -1))
        != incumbent_reciprocal_aliases
    ):
        raise NestedReauditError(
            "model_manifest does not exactly cover inner TCN checkpoints"
        )
    actual_tcn_paths = {
        path.resolve()
        for path in (root / "models" / "tcn").rglob("*")
        if path.is_file() and path.suffix in {".pt", ".json"}
    }
    if actual_tcn_paths != declared_tcn_paths:
        raise NestedReauditError(
            "TCN checkpoint files differ from declared model provenance"
        )
    incumbent_root = root / "incumbent_checkpoints"
    actual_incumbent_paths = {
        path.resolve()
        for path in incumbent_root.rglob("*")
        if path.is_file() and path.suffix in {".pkl", ".json"}
    }
    if actual_incumbent_paths != declared_incumbent_paths:
        raise NestedReauditError(
            "incumbent checkpoint files differ from declared provenance"
        )

    recomputed_metrics = _metric_payload(
        final_oof,
        bootstrap_samples=int(policy["bootstrap_samples"]),
        shadow_samples=int(policy["shadow_samples"]),
        shadow_subset_wells=int(policy["shadow_subset_wells"]),
    )
    metric_maximum_delta = _compare_json(
        stored_metrics,
        recomputed_metrics,
        label="metrics.summary",
        absolute_tolerance=summary_atol,
        relative_tolerance=summary_rtol,
    )
    summary_metric_delta = _compare_json(
        summary.get("metrics"),
        recomputed_metrics,
        label="runner_summary.metrics",
        absolute_tolerance=summary_atol,
        relative_tolerance=summary_rtol,
    )
    summary_weight_delta = _compare_json(
        summary.get("weights"),
        expected_weight_records,
        label="runner_summary.weights",
        absolute_tolerance=summary_atol,
        relative_tolerance=summary_rtol,
    )

    final_oof_sha = sha256_file(final_oof_path)
    weights_sha = sha256_file(weights_path)
    if (
        artifact_audit.get("oof_sha256") != final_oof_sha
        or artifact_audit.get("weights_sha256") != weights_sha
        or artifact_audit.get("model_manifest_sha256") != model_manifest_sha
        or artifact_audit.get("metrics_sha256") != metrics_sha
        or artifact_audit.get("oof_validation") != final_validation
        or artifact_audit.get("source_oof_sha256") != run_config["source"]["oof_sha256"]
        or artifact_audit.get("source_manifest_sha256")
        != run_config["source"]["manifest_sha256"]
        or artifact_audit.get("outer_fold_manifests") != actual_outer_manifests
        or artifact_audit.get("data_manifest_before") != recorded_data
        or artifact_audit.get("data_manifest_after") != recorded_data
        or artifact_audit.get("evidence_mode") != evidence_mode
        or artifact_audit.get("evidence_eligible")
        is not bool(run_config.get("evidence_eligible"))
    ):
        raise NestedReauditError(
            "runner artifact audit differs from independently read artifacts"
        )
    runner_source = artifact_audit.get("source")
    if not isinstance(runner_source, Mapping):
        raise NestedReauditError("runner source audit provenance is missing")
    if evidence_mode == "smoke":
        if (
            runner_source.get("passed") is not True
            or runner_source.get("evidence_eligible") is not False
            or runner_source.get("validation") != source_validation
        ):
            raise NestedReauditError(
                "runner smoke-source audit differs from disk source"
            )
    else:
        runner_incumbent = runner_source.get("incumbent")
        runner_bundle = runner_source.get("bundle_reaudit")
        if (
            not isinstance(runner_incumbent, Mapping)
            or not isinstance(runner_bundle, Mapping)
            or runner_incumbent.get("sha256") != source_record["incumbent"]["sha256"]
            or runner_incumbent.get("validation")
            != source_record["incumbent"]["validation"]
            or runner_bundle.get("passed") is not True
            or runner_bundle.get("profile") != "full"
            or runner_bundle.get("manifest_sha256") != source_record["manifest_sha256"]
            or runner_source.get("candidate") != source_record["candidate"]
        ):
            raise NestedReauditError(
                "runner production-source audit differs from independent lock"
            )
    checks = artifact_audit.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or checks.get("submission_created") is not False
        or not all(
            value is True
            for key, value in checks.items()
            if key != "submission_created"
        )
    ):
        raise NestedReauditError("runner artifact audit contains a failed check")
    for path_key, expected_path in (
        ("oof_path", final_oof_path),
        ("weights_path", weights_path),
        ("model_manifest_path", model_manifest_path),
        ("metrics_path", metrics_path),
    ):
        if (
            _run_path(
                root,
                artifact_audit.get(path_key),
                label=f"artifact_audit.{path_key}",
            )
            != expected_path.resolve()
        ):
            raise NestedReauditError(f"artifact_audit.{path_key} differs")

    # This is the non-tautological runner-audit integrity check: the summary's
    # stored digest is compared with bytes independently read from disk.
    if summary.get("artifact_audit_sha256") != artifact_audit_sha:
        raise NestedReauditError(
            "summary artifact_audit_sha256 differs from artifact_audit bytes"
        )
    if (
        summary.get("oof_sha256") != final_oof_sha
        or summary.get("run_id") != run_config.get("run_id")
        or Path(str(summary.get("run_root"))).resolve() != root
        or summary.get("evidence_mode") != evidence_mode
        or summary.get("evidence_eligible")
        is not bool(run_config.get("evidence_eligible"))
        or summary.get("outer_folds") != list(outer_folds)
        or int(summary.get("inner_splits", -1)) != len(expected_progress_inner)
        or int(summary.get("tcn_checkpoints", -1)) != len(model_records)
        or int(summary.get("tcn_physical_fits", -1))
        != len(model_records) - tcn_reciprocal_aliases
        or int(summary.get("tcn_reciprocal_aliases", -1)) != tcn_reciprocal_aliases
        or int(summary.get("incumbent_checkpoints", -1)) != incumbent_checkpoint_count
        or int(summary.get("incumbent_refits", -1))
        != incumbent_checkpoint_count - incumbent_reciprocal_aliases
        or int(summary.get("incumbent_physical_fits", -1))
        != incumbent_checkpoint_count - incumbent_reciprocal_aliases
        or int(summary.get("incumbent_reciprocal_aliases", -1))
        != incumbent_reciprocal_aliases
        or int(summary.get("rows", -1)) != len(final_oof)
        or int(summary.get("wells", -1)) != int(final_oof["well_id"].nunique())
        or summary.get("data_manifest_equal") is not True
    ):
        raise NestedReauditError(
            "runner summary counts, hashes, or evidence markers differ"
        )

    data_after = capture_data_manifest(resolved_data)
    if data_after != data_before:
        raise NestedReauditError(
            "competition data manifest changed during independent re-audit"
        )
    submissions_after = _submission_csvs(root)
    if submissions_after:
        raise NestedReauditError(
            f"submission CSV appeared during re-audit: {submissions_after}"
        )

    report = {
        "schema_version": NESTED_REAUDIT_SCHEMA_VERSION,
        "passed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_config["run_id"],
        "run_root": str(root),
        "evidence_mode": evidence_mode,
        "evidence_eligible": bool(run_config["evidence_eligible"]),
        "scope": {
            "outer_folds": list(outer_folds),
            "inner_splits": len(expected_progress_inner),
            "rows": int(len(final_oof)),
            "wells": int(final_oof["well_id"].nunique()),
            "tcn_checkpoints": len(model_records),
            "tcn_physical_fits": len(model_records) - tcn_reciprocal_aliases,
            "tcn_reciprocal_aliases": tcn_reciprocal_aliases,
            "incumbent_checkpoints": incumbent_checkpoint_count,
            "incumbent_physical_fits": (
                incumbent_checkpoint_count - incumbent_reciprocal_aliases
            ),
            "incumbent_reciprocal_aliases": incumbent_reciprocal_aliases,
        },
        "checks": {
            "disk_only_source_reference": True,
            "source_independent_lock_required_for_production": True,
            "exact_key_and_truth_coverage": True,
            "source_outer_predictions_untouched": True,
            "inner_family_first_seed_average": True,
            "inner_residual_component_algebra": True,
            "weights_recomputed_from_inner_oof": True,
            "outer_predictions_recomputed_from_weights": True,
            "final_predictions_equal_outer_artifacts": True,
            "metrics_recomputed_from_final_oof": True,
            "runner_summary_recomputed": True,
            "runner_artifact_audit_hash_anchored_by_summary": True,
            "checkpoint_and_sidecar_hashes": True,
            "reciprocal_reuse_equivalence_and_alias_counts": True,
            "fold_and_training_provenance": True,
            "data_manifest_equal_before_after_current": True,
            "submission_created": False,
        },
        "source": {
            **source_record,
            "validation": source_validation,
        },
        "artifacts": {
            "run_config": {
                "path": str(config_path),
                "sha256": run_config_sha,
                "digest": run_config_digest,
            },
            "progress": {
                "path": str(progress_path),
                "sha256": progress_sha,
            },
            "final_oof": {
                "path": str(final_oof_path),
                "sha256": final_oof_sha,
                "validation": final_validation,
            },
            "weights": {
                "path": str(weights_path),
                "sha256": weights_sha,
            },
            "model_manifest": {
                "path": str(model_manifest_path),
                "sha256": model_manifest_sha,
            },
            "metrics": {
                "path": str(metrics_path),
                "sha256": metrics_sha,
            },
            "runner_artifact_audit": {
                "path": str(artifact_audit_path),
                "sha256": artifact_audit_sha,
                "summary_declared_sha256": summary["artifact_audit_sha256"],
            },
            "runner_summary": {
                "path": str(summary_path),
                "sha256": summary_sha,
            },
        },
        "data_read_only_manifest_before": data_before,
        "data_read_only_manifest_after": data_after,
        "fold_config": {
            "path": str(resolved_folds),
            "sha256": sha256_file(resolved_folds),
            "fold_version": str(mapping["fold_version"].iat[0]),
        },
        "inner_splits": inner_records,
        "weights": expected_weight_records,
        "metrics": recomputed_metrics,
        "numerical_comparison": {
            "value_atol": value_atol,
            "summary_atol": summary_atol,
            "summary_rtol": summary_rtol,
            "source_value_maximum_absolute_delta": source_value_deltas,
            "inner_truth_maximum_absolute_delta": maximum_inner_truth_delta,
            "outer_prediction_maximum_absolute_delta": (maximum_outer_prediction_delta),
            "weight_maximum_absolute_delta": weight_maximum_delta,
            "metric_maximum_absolute_delta": metric_maximum_delta,
            "summary_metric_maximum_absolute_delta": summary_metric_delta,
            "summary_weight_maximum_absolute_delta": summary_weight_delta,
        },
        "submission_created": False,
    }
    _atomic_json(output_path, report)
    written, _ = _read_json(output_path, label="independent nested re-audit")
    if written != _jsonable(report) or written.get("passed") is not True:
        raise NestedReauditError("independent nested re-audit disk write failed")
    return written


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "DEFAULT_SUMMARY_ATOL",
    "DEFAULT_SUMMARY_RTOL",
    "DEFAULT_VALUE_ATOL",
    "NESTED_REAUDIT_SCHEMA_VERSION",
    "NestedReauditError",
    "audit_nested_external_ensemble",
]
