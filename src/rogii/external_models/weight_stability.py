"""Well-cluster bootstrap stability for the locked two-family nested blend.

Only persisted inner-OOF labels are parsed.  The final/outer OOF is treated as
an opaque byte artifact and is used only as a contextual SHA-256 lock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
import scipy

from ..diagnostics.nested_external_ensemble import (
    FAMILY_MAX_WEIGHT,
    FOLD_COUNT,
    INCUMBENT_COLUMN,
    REGULARIZATION_STRENGTH,
    REGULARIZATION_TARGET,
    SCHEMA_VERSION as NESTED_RUN_SCHEMA_VERSION,
    TCN_COLUMN,
)
from ..portfolio_metrics import DEFAULT_BOOTSTRAP_SEED, OOF_KEY_COLUMNS
from .artifacts import sha256_file, well_ids_sha256
from .nested_reaudit import NESTED_REAUDIT_SCHEMA_VERSION


WEIGHT_STABILITY_SCHEMA_VERSION = "nested_blend_weight_stability_v1"
DEFAULT_OUTPUT_NAME = "weight_stability"
SUMMARY_FILENAME = "summary.json"
STATISTICS_FILENAME = "weight_stability.csv"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = DEFAULT_BOOTSTRAP_SEED
BOOTSTRAP_BATCH_SIZE = 256
QUANTILE_METHOD = "linear"
STATISTICS_DIGEST_SCHEMA = (
    "outer_fold:<i8,inner_fold:<i8,well_id:utf8-length-prefixed,"
    "n:<i8,sum_d2:<f8,sum_e_d:<f8,sum_e2:<f8"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_OUTPUT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REQUIRED_AUDIT_CHECKS = (
    "checkpoint_and_sidecar_hashes",
    "data_manifest_equal_before_after_current",
    "disk_only_source_reference",
    "exact_key_and_truth_coverage",
    "final_predictions_equal_outer_artifacts",
    "fold_and_training_provenance",
    "inner_family_first_seed_average",
    "inner_residual_component_algebra",
    "metrics_recomputed_from_final_oof",
    "outer_predictions_recomputed_from_weights",
    "reciprocal_reuse_equivalence_and_alias_counts",
    "runner_artifact_audit_hash_anchored_by_summary",
    "runner_summary_recomputed",
    "source_independent_lock_required_for_production",
    "source_outer_predictions_untouched",
    "weights_recomputed_from_inner_oof",
)


class WeightStabilityError(RuntimeError):
    """Raised when locked nested evidence or bootstrap invariants differ."""


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise WeightStabilityError(
            f"{label} is not valid finite JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise WeightStabilityError(f"{label} must be a JSON object: {path}")
    return value, digest


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
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


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value).lower()
    if not _SHA256.fullmatch(digest):
        raise WeightStabilityError(f"{label} must be a lowercase SHA-256")
    return digest


def _run_path(root: Path, value: Any, *, expected: Path, label: str) -> Path:
    declared = Path(str(value))
    resolved = (
        declared.resolve() if declared.is_absolute() else (root / declared).resolve()
    )
    if resolved != expected.resolve():
        raise WeightStabilityError(f"{label} path differs from the locked artifact")
    if resolved != root and root not in resolved.parents:
        raise WeightStabilityError(f"{label} escapes the nested run root")
    return resolved


def _submission_csvs(root: Path) -> list[str]:
    result: list[str] = []
    for path in root.rglob("*"):
        if (
            path.is_file()
            and "submission" in path.name.lower()
            and "".join(path.suffixes).lower() in {".csv", ".csv.gz"}
        ):
            result.append(path.relative_to(root).as_posix())
    return sorted(result)


def _expected_blend_policy() -> dict[str, Any]:
    return {
        "families": {
            "incumbent": [INCUMBENT_COLUMN],
            "tcn": [TCN_COLUMN],
        },
        "family_first_seed_average": True,
        "regularization_strength": REGULARIZATION_STRENGTH,
        "regularization_target": REGULARIZATION_TARGET,
        "family_max_weight": FAMILY_MAX_WEIGHT,
        "fixed_controls": {
            "y_pred_fixed_075_025": {"incumbent": 0.75, "tcn": 0.25},
            "y_pred_simple_050_050": {"incumbent": 0.50, "tcn": 0.50},
        },
        "weight_selection": "inner OOF only",
        "outer_validation_used_for_weight_fit": False,
    }


def _validated_cohort(
    run_config: Mapping[str, Any],
    outer_folds: Sequence[int],
) -> tuple[
    tuple[str, ...],
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
]:
    cohort = run_config.get("cohort")
    if not isinstance(cohort, Mapping) or set(cohort) != {
        str(fold) for fold in outer_folds
    }:
        raise WeightStabilityError("run config cohort fold keys differ")

    validation_by_fold: dict[int, tuple[str, ...]] = {}
    training_by_fold: dict[int, tuple[str, ...]] = {}
    for outer_fold in outer_folds:
        record = cohort[str(outer_fold)]
        if not isinstance(record, Mapping):
            raise WeightStabilityError("run config cohort record is invalid")
        training = tuple(map(str, record.get("training_well_ids", ())))
        validation = tuple(map(str, record.get("validation_well_ids", ())))
        if (
            not training
            or not validation
            or training != tuple(sorted(set(training)))
            or validation != tuple(sorted(set(validation)))
            or set(training) & set(validation)
            or record.get("intersection") != []
            or record.get("training_ids_sha256") != well_ids_sha256(training)
            or record.get("validation_ids_sha256") != well_ids_sha256(validation)
        ):
            raise WeightStabilityError(
                f"run config cohort differs for outer fold {outer_fold}"
            )
        training_by_fold[outer_fold] = training
        validation_by_fold[outer_fold] = validation

    validation_sets = [set(validation_by_fold[fold]) for fold in outer_folds]
    for index, left in enumerate(validation_sets):
        for right in validation_sets[index + 1 :]:
            if left & right:
                raise WeightStabilityError("outer validation cohorts overlap")
    universe = tuple(sorted(set().union(*validation_sets)))
    for outer_fold in outer_folds:
        if set(training_by_fold[outer_fold]) != (
            set(universe) - set(validation_by_fold[outer_fold])
        ):
            raise WeightStabilityError(
                f"outer training cohort is not the complement for fold {outer_fold}"
            )
    return universe, training_by_fold, validation_by_fold


def _validate_source_locks(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[int, ...],
    tuple[int, ...],
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
]:
    run_config_path = root / "run_config.json"
    reaudit_path = root / "independent_nested_reaudit.json"
    final_oof_path = root / "oof_predictions.csv.gz"
    run_config, run_config_sha = _read_json(
        run_config_path,
        label="nested run config",
    )
    reaudit, reaudit_sha = _read_json(
        reaudit_path,
        label="independent nested re-audit",
    )
    if (
        run_config.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
        or run_config.get("evidence_mode") != "production"
        or run_config.get("evidence_eligible") is not True
        or run_config.get("submission_created") is not False
        or run_config.get("blend_policy") != _expected_blend_policy()
    ):
        raise WeightStabilityError(
            "nested run config is not locked production evidence"
        )
    policy = run_config.get("policy")
    if not isinstance(policy, Mapping):
        raise WeightStabilityError("nested run policy is missing")
    outer_folds = tuple(map(int, policy.get("outer_folds", ())))
    tcn_seeds = tuple(map(int, policy.get("tcn_seeds", ())))
    incumbent_seeds = tuple(map(int, policy.get("incumbent_seeds", ())))
    if (
        outer_folds != tuple(range(FOLD_COUNT))
        or len(tcn_seeds) != 5
        or len(set(tcn_seeds)) != len(tcn_seeds)
        or len(incumbent_seeds) != 5
        or len(set(incumbent_seeds)) != len(incumbent_seeds)
        or int(policy.get("bootstrap_samples", -1)) != BOOTSTRAP_SAMPLES
    ):
        raise WeightStabilityError(
            "nested production fold/seed/bootstrap policy differs"
        )
    _, training_by_fold, validation_by_fold = _validated_cohort(
        run_config,
        outer_folds,
    )

    checks = reaudit.get("checks")
    if (
        reaudit.get("schema_version") != NESTED_REAUDIT_SCHEMA_VERSION
        or reaudit.get("passed") is not True
        or reaudit.get("evidence_mode") != "production"
        or reaudit.get("evidence_eligible") is not True
        or reaudit.get("submission_created") is not False
        or Path(str(reaudit.get("run_root", ""))).resolve() != root
        or not isinstance(checks, Mapping)
        or any(checks.get(name) is not True for name in _REQUIRED_AUDIT_CHECKS)
        or checks.get("submission_created") is not False
    ):
        raise WeightStabilityError(
            "independent nested re-audit is not passing production evidence"
        )
    if _submission_csvs(root):
        raise WeightStabilityError("nested run contains a submission CSV")

    artifacts = reaudit.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise WeightStabilityError("independent re-audit artifact locks are missing")
    config_lock = artifacts.get("run_config")
    final_lock = artifacts.get("final_oof")
    if not isinstance(config_lock, Mapping) or not isinstance(final_lock, Mapping):
        raise WeightStabilityError("run-config/final-OOF locks are missing")
    _run_path(
        root,
        config_lock.get("path"),
        expected=run_config_path,
        label="re-audit run config",
    )
    run_config_digest = _canonical_sha256(run_config)
    if (
        _require_sha256(
            config_lock.get("sha256"),
            label="re-audit run-config SHA-256",
        )
        != run_config_sha
        or _require_sha256(
            config_lock.get("digest"),
            label="re-audit run-config canonical digest",
        )
        != run_config_digest
    ):
        raise WeightStabilityError("nested run config differs from re-audit lock")
    _run_path(
        root,
        final_lock.get("path"),
        expected=final_oof_path,
        label="re-audit final OOF",
    )
    final_oof_sha = sha256_file(final_oof_path)
    if (
        _require_sha256(
            final_lock.get("sha256"),
            label="re-audit final-OOF SHA-256",
        )
        != final_oof_sha
    ):
        raise WeightStabilityError("final OOF bytes differ from re-audit lock")

    source = reaudit.get("source")
    config_source = run_config.get("source")
    if (
        not isinstance(source, Mapping)
        or not isinstance(config_source, Mapping)
        or source.get("independently_locked") is not True
        or source.get("oof_sha256") != config_source.get("oof_sha256")
        or source.get("manifest_sha256") != config_source.get("manifest_sha256")
    ):
        raise WeightStabilityError("nested external source lock differs")

    lock_record = {
        "run_config": {
            "path": run_config_path.name,
            "sha256": run_config_sha,
            "canonical_digest": run_config_digest,
        },
        "final_oof": {
            "path": final_oof_path.name,
            "sha256": final_oof_sha,
            "bytes": int(final_oof_path.stat().st_size),
            "access": "opaque bytes hashed only; CSV was not parsed",
        },
        "independent_nested_reaudit": {
            "path": reaudit_path.name,
            "sha256": reaudit_sha,
        },
    }
    return (
        run_config,
        reaudit,
        lock_record,
        outer_folds,
        tcn_seeds,
        training_by_fold,
        validation_by_fold,
    )


def _audit_inner_records(
    reaudit: Mapping[str, Any],
    outer_folds: Sequence[int],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    records = reaudit.get("inner_splits")
    if not isinstance(records, list):
        raise WeightStabilityError("independent re-audit inner split locks are missing")
    expected = {
        (outer_fold, inner_fold)
        for outer_fold in outer_folds
        for inner_fold in outer_folds
        if inner_fold != outer_fold
    }
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for value in records:
        if not isinstance(value, Mapping):
            raise WeightStabilityError("invalid independent re-audit inner record")
        key = (int(value.get("outer_fold", -1)), int(value.get("inner_fold", -1)))
        if key in result:
            raise WeightStabilityError("duplicate independent re-audit inner record")
        result[key] = value
    if set(result) != expected:
        raise WeightStabilityError("independent re-audit inner split coverage differs")
    return result


def _expected_prediction_columns(tcn_seeds: Sequence[int]) -> tuple[str, ...]:
    return (
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        *(f"{TCN_COLUMN}__seed_{seed}" for seed in tcn_seeds),
        *(f"{TCN_COLUMN}__{component}" for component in ("common", "slope", "shape")),
        *(
            f"{TCN_COLUMN}__seed_{seed}__{component}"
            for seed in tcn_seeds
            for component in ("common", "slope", "shape")
        ),
    )


def _read_inner_oof(
    path: Path,
    *,
    tcn_seeds: Sequence[int],
) -> pd.DataFrame:
    seed_columns = [f"{TCN_COLUMN}__seed_{seed}" for seed in tcn_seeds]
    columns = [
        *OOF_KEY_COLUMNS,
        "y_true",
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        *seed_columns,
    ]
    try:
        frame = pd.read_csv(
            path,
            usecols=columns,
            dtype={
                "well_id": "string",
                "row_index": "int64",
                "fold": "int64",
            },
            float_precision="round_trip",
        )
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise WeightStabilityError(f"inner OOF is unreadable: {path}") from error
    frame["well_id"] = frame["well_id"].astype(str)
    return frame.loc[:, columns]


def two_family_well_sufficient_statistics(
    frame: pd.DataFrame,
    *,
    outer_fold: int = -1,
    inner_fold: int = -1,
    truth_column: str = "y_true",
    incumbent_column: str = INCUMBENT_COLUMN,
    tcn_column: str = TCN_COLUMN,
) -> pd.DataFrame:
    """Return exact per-well quadratic statistics for the two-family objective."""

    required = {"well_id", truth_column, incumbent_column, tcn_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing sufficient-stat columns: {sorted(missing)}")
    well_ids = frame["well_id"].astype(str)
    if well_ids.empty or (well_ids.str.len() == 0).any():
        raise ValueError("well IDs must be non-empty")
    numeric = frame.loc[
        :,
        [truth_column, incumbent_column, tcn_column],
    ].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("truth and two-family predictions must be finite")
    truth = values[:, 0]
    incumbent = values[:, 1]
    tcn = values[:, 2]
    error = incumbent - truth
    delta = tcn - incumbent
    working = pd.DataFrame(
        {
            "well_id": well_ids.to_numpy(),
            "n": np.ones(len(frame), dtype=np.int64),
            "sum_d2": delta * delta,
            "sum_e_d": error * delta,
            "sum_e2": error * error,
        }
    )
    result = (
        working.groupby("well_id", sort=True, as_index=False)
        .agg(
            n=("n", "sum"),
            sum_d2=("sum_d2", "sum"),
            sum_e_d=("sum_e_d", "sum"),
            sum_e2=("sum_e2", "sum"),
        )
        .reset_index(drop=True)
    )
    result.insert(0, "inner_fold", int(inner_fold))
    result.insert(0, "outer_fold", int(outer_fold))
    statistic_values = result.loc[
        :,
        ["sum_d2", "sum_e_d", "sum_e2"],
    ].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(statistic_values).all()
        or (result["n"].to_numpy(dtype=np.int64) <= 0).any()
    ):
        raise ValueError("non-finite or empty sufficient statistics")
    return result


def fit_two_family_sufficient_statistics(
    statistics: pd.DataFrame,
    *,
    multiplicities: Sequence[int] | np.ndarray | None = None,
    regularization_strength: float = REGULARIZATION_STRENGTH,
    regularization_target: Mapping[str, float] = REGULARIZATION_TARGET,
    family_max_weight: Mapping[str, float] = FAMILY_MAX_WEIGHT,
) -> dict[str, float | int]:
    """Fit the current constrained objective exactly from cluster statistics."""

    required = {"n", "sum_d2", "sum_e_d", "sum_e2"}
    missing = required - set(statistics.columns)
    if statistics.empty or missing:
        raise ValueError(f"invalid sufficient-stat frame; missing={sorted(missing)}")
    matrix = statistics.loc[
        :,
        ["n", "sum_d2", "sum_e_d", "sum_e2"],
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all() or np.any(matrix[:, 0] <= 0.0):
        raise ValueError("sufficient statistics must be finite with positive rows")
    if multiplicities is None:
        counts = np.ones(len(statistics), dtype=np.float64)
    else:
        raw_counts = np.asarray(multiplicities)
        if (
            raw_counts.shape != (len(statistics),)
            or not np.issubdtype(raw_counts.dtype, np.integer)
            or np.any(raw_counts < 0)
            or int(raw_counts.sum()) <= 0
        ):
            raise ValueError(
                "multiplicities must be nonnegative integer cluster counts"
            )
        counts = raw_counts.astype(np.float64)
    if not math.isfinite(regularization_strength) or regularization_strength < 0.0:
        raise ValueError("regularization strength must be finite and nonnegative")
    if set(regularization_target) != {"incumbent", "tcn"}:
        raise ValueError("two-family regularization target keys differ")
    incumbent_target = float(regularization_target["incumbent"])
    tcn_target = float(regularization_target["tcn"])
    if (
        not math.isfinite(incumbent_target)
        or not math.isfinite(tcn_target)
        or incumbent_target < 0.0
        or tcn_target < 0.0
        or not math.isclose(
            incumbent_target + tcn_target,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("two-family regularization target is invalid")
    unknown_caps = set(family_max_weight) - {"incumbent", "tcn"}
    if unknown_caps:
        raise ValueError(f"unknown two-family caps: {sorted(unknown_caps)}")
    incumbent_cap = float(family_max_weight.get("incumbent", 1.0))
    tcn_cap = float(family_max_weight.get("tcn", 1.0))
    if (
        not math.isfinite(incumbent_cap)
        or not math.isfinite(tcn_cap)
        or not 0.0 <= incumbent_cap <= 1.0
        or not 0.0 <= tcn_cap <= 1.0
    ):
        raise ValueError("two-family caps must be finite and in [0, 1]")
    lower = max(0.0, 1.0 - incumbent_cap)
    upper = min(1.0, tcn_cap)
    if lower > upper:
        raise ValueError("two-family caps make the simplex infeasible")

    totals = counts @ matrix
    rows, sum_d2, sum_e_d, sum_e2 = map(float, totals)
    denominator = sum_d2 + 2.0 * regularization_strength * rows
    if rows <= 0.0 or not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("degenerate two-family sufficient-stat objective")
    raw_tcn_weight = (
        2.0 * regularization_strength * rows * tcn_target - sum_e_d
    ) / denominator
    tcn_weight = float(np.clip(raw_tcn_weight, lower, upper))
    incumbent_weight = 1.0 - tcn_weight
    squared_error = (
        sum_e2 + 2.0 * tcn_weight * sum_e_d + tcn_weight * tcn_weight * sum_d2
    )
    if squared_error < -1.0e-8 * max(1.0, sum_e2):
        raise ValueError("sufficient-stat SSE is negative")
    squared_error = max(0.0, squared_error)
    penalty = regularization_strength * (
        (incumbent_weight - incumbent_target) ** 2 + (tcn_weight - tcn_target) ** 2
    )
    return {
        "training_rows": int(round(rows)),
        "incumbent_weight": incumbent_weight,
        "tcn_weight": tcn_weight,
        "raw_tcn_weight": float(raw_tcn_weight),
        "training_rmse": math.sqrt(squared_error / rows),
        "objective_value": squared_error / rows + penalty,
        "tcn_lower_bound": lower,
        "tcn_upper_bound": upper,
    }


def bootstrap_two_family_weights(
    statistics: pd.DataFrame,
    *,
    samples: int,
    rng: np.random.Generator,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
    regularization_strength: float = REGULARIZATION_STRENGTH,
    regularization_target: Mapping[str, float] = REGULARIZATION_TARGET,
    family_max_weight: Mapping[str, float] = FAMILY_MAX_WEIGHT,
) -> np.ndarray:
    """Cluster-resample wells and return exact TCN weights without row expansion."""

    if samples <= 0 or batch_size <= 0:
        raise ValueError("bootstrap samples and batch size must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    required = ["n", "sum_d2", "sum_e_d", "sum_e2"]
    if statistics.empty or set(required) - set(statistics.columns):
        raise ValueError("bootstrap statistics are empty or incomplete")
    matrix = statistics.loc[:, required].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all() or np.any(matrix[:, 0] <= 0.0):
        raise ValueError("bootstrap statistics must be finite with positive rows")
    cluster_count = len(matrix)
    probabilities = np.full(cluster_count, 1.0 / cluster_count, dtype=np.float64)
    fit = fit_two_family_sufficient_statistics(
        statistics,
        regularization_strength=regularization_strength,
        regularization_target=regularization_target,
        family_max_weight=family_max_weight,
    )
    lower = float(fit["tcn_lower_bound"])
    upper = float(fit["tcn_upper_bound"])
    tcn_target = float(regularization_target["tcn"])
    output = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        counts = rng.multinomial(
            cluster_count,
            probabilities,
            size=stop - start,
        )
        totals = counts @ matrix
        rows = totals[:, 0]
        denominator = totals[:, 1] + 2.0 * float(regularization_strength) * rows
        if (
            not np.isfinite(totals).all()
            or not np.isfinite(denominator).all()
            or np.any(rows <= 0.0)
            or np.any(denominator <= 0.0)
        ):
            raise WeightStabilityError("bootstrap produced a degenerate objective")
        raw = (
            2.0 * float(regularization_strength) * rows * tcn_target - totals[:, 2]
        ) / denominator
        output[start:stop] = np.clip(raw, lower, upper)
    if not np.isfinite(output).all():
        raise WeightStabilityError("bootstrap produced non-finite weights")
    return output


def _statistics_sha256(statistics: pd.DataFrame) -> str:
    required = [
        "outer_fold",
        "inner_fold",
        "well_id",
        "n",
        "sum_d2",
        "sum_e_d",
        "sum_e2",
    ]
    if set(required) - set(statistics.columns):
        raise ValueError("canonical sufficient-stat columns are missing")
    ordered = statistics.loc[:, required].sort_values(
        ["outer_fold", "inner_fold", "well_id"],
        kind="stable",
    )
    digest = hashlib.sha256()
    digest.update(b"rogii-two-family-well-statistics-v1\0")
    for record in ordered.itertuples(index=False, name=None):
        outer_fold, inner_fold, well_id, rows, sum_d2, sum_e_d, sum_e2 = record
        encoded_id = str(well_id).encode("utf-8")
        digest.update(
            np.asarray([int(outer_fold), int(inner_fold)], dtype="<i8").tobytes()
        )
        digest.update(np.asarray([len(encoded_id)], dtype="<u4").tobytes())
        digest.update(encoded_id)
        digest.update(np.asarray([int(rows)], dtype="<i8").tobytes())
        digest.update(np.asarray([sum_d2, sum_e_d, sum_e2], dtype="<f8").tobytes())
    return digest.hexdigest()


def _validate_and_summarize_inner_split(
    *,
    root: Path,
    outer_fold: int,
    inner_fold: int,
    run_config: Mapping[str, Any],
    audit_record: Mapping[str, Any],
    expected_training: tuple[str, ...],
    expected_validation: tuple[str, ...],
    excluded_outer: tuple[str, ...],
    tcn_seeds: Sequence[int],
    incumbent_seed_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = root / "inner_oof" / f"outer_{outer_fold}" / f"inner_{inner_fold}"
    manifest_path = directory / "manifest.json"
    oof_path = directory / "oof_predictions.csv.gz"
    manifest, manifest_sha = _read_json(
        manifest_path,
        label=f"inner manifest outer={outer_fold} inner={inner_fold}",
    )
    run_config_digest = _canonical_sha256(run_config)
    policy_digest = _canonical_sha256(run_config["policy"])
    expected_predictions = _expected_prediction_columns(tcn_seeds)
    if (
        manifest.get("schema_version") != NESTED_RUN_SCHEMA_VERSION
        or manifest.get("artifact_kind") != "inner_oof"
        or manifest.get("completed") is not True
        or int(manifest.get("outer_fold", -1)) != outer_fold
        or int(manifest.get("inner_fold", -1)) != inner_fold
        or tuple(map(str, manifest.get("training_well_ids", ()))) != expected_training
        or tuple(map(str, manifest.get("validation_well_ids", ())))
        != expected_validation
        or tuple(map(str, manifest.get("excluded_outer_well_ids", ())))
        != excluded_outer
        or manifest.get("training_ids_sha256") != well_ids_sha256(expected_training)
        or manifest.get("validation_ids_sha256") != well_ids_sha256(expected_validation)
        or manifest.get("excluded_outer_ids_sha256") != well_ids_sha256(excluded_outer)
        or manifest.get("training_validation_intersection") != []
        or manifest.get("training_outer_intersection") != []
        or manifest.get("validation_outer_intersection") != []
        or manifest.get("policy_digest") != policy_digest
        or manifest.get("run_config_digest") != run_config_digest
        or manifest.get("source_oof_sha256") != run_config["source"]["oof_sha256"]
        or manifest.get("truth_attached_only_after_all_seed_predictions") is not True
        or manifest.get("outer_validation_labels_used_for_fit") is not False
        or manifest.get("inner_validation_labels_used_for_fit") is not False
        or tuple(map(str, manifest.get("prediction_columns", ())))
        != expected_predictions
    ):
        raise WeightStabilityError(
            f"inner provenance differs for outer={outer_fold}, inner={inner_fold}"
        )
    _run_path(
        root,
        manifest.get("oof_path"),
        expected=oof_path,
        label=f"inner OOF outer={outer_fold} inner={inner_fold}",
    )
    if (
        _require_sha256(
            audit_record.get("manifest_sha256"),
            label="re-audit inner manifest SHA-256",
        )
        != manifest_sha
        or int(audit_record.get("outer_fold", -1)) != outer_fold
        or int(audit_record.get("inner_fold", -1)) != inner_fold
        or int(audit_record.get("training_wells", -1)) != len(expected_training)
        or int(audit_record.get("validation_wells", -1)) != len(expected_validation)
        or int(audit_record.get("incumbent_checkpoints", -1)) != incumbent_seed_count
        or int(audit_record.get("tcn_checkpoints", -1)) != len(tcn_seeds)
        or not isinstance(manifest.get("incumbent_checkpoints"), list)
        or len(manifest["incumbent_checkpoints"]) != incumbent_seed_count
        or not isinstance(manifest.get("tcn_checkpoints"), list)
        or len(manifest["tcn_checkpoints"]) != len(tcn_seeds)
    ):
        raise WeightStabilityError(
            f"inner manifest differs from re-audit outer={outer_fold}, "
            f"inner={inner_fold}"
        )

    frame = _read_inner_oof(oof_path, tcn_seeds=tcn_seeds)
    oof_sha = sha256_file(oof_path)
    if (
        _require_sha256(
            manifest.get("oof_sha256"),
            label="inner manifest OOF SHA-256",
        )
        != oof_sha
        or _require_sha256(
            audit_record.get("oof_sha256"),
            label="re-audit inner OOF SHA-256",
        )
        != oof_sha
        or int(manifest.get("rows", -1)) != len(frame)
        or int(audit_record.get("rows", -1)) != len(frame)
        or int(manifest.get("wells", -1)) != frame["well_id"].nunique()
        or int(audit_record.get("wells", -1)) != frame["well_id"].nunique()
    ):
        raise WeightStabilityError(
            f"inner OOF bytes/scope differ outer={outer_fold}, inner={inner_fold}"
        )
    if frame.duplicated(list(OOF_KEY_COLUMNS), keep=False).any():
        raise WeightStabilityError("inner OOF contains duplicate OOF keys")
    actual_wells = tuple(sorted(frame["well_id"].unique()))
    if actual_wells != expected_validation:
        raise WeightStabilityError("inner OOF well coverage differs from its split")
    if tuple(sorted(frame["fold"].unique())) != (inner_fold,):
        raise WeightStabilityError("inner OOF fold IDs differ from the original folds")
    numeric_columns = [
        "y_true",
        INCUMBENT_COLUMN,
        TCN_COLUMN,
        *(f"{TCN_COLUMN}__seed_{seed}" for seed in tcn_seeds),
    ]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise WeightStabilityError("inner OOF contains non-finite fit values")
    expected_tcn = (
        numeric.loc[
            :,
            [f"{TCN_COLUMN}__seed_{seed}" for seed in tcn_seeds],
        ]
        .to_numpy(dtype=np.float64)
        .mean(axis=1)
    )
    if not np.allclose(
        numeric[TCN_COLUMN].to_numpy(dtype=np.float64),
        expected_tcn,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise WeightStabilityError("inner TCN is not the row-level family seed mean")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("coverage_checked") is not True
        or validation.get("truth_checked") is not True
        or int(validation.get("rows", -1)) != len(frame)
        or int(validation.get("wells", -1)) != len(actual_wells)
        or tuple(map(int, validation.get("folds", ()))) != (inner_fold,)
        or int(validation.get("duplicate_well_row_keys", -1)) != 0
        or int(validation.get("nonfinite_numeric_values", -1)) != 0
        or tuple(map(str, validation.get("prediction_columns", ())))
        != expected_predictions
    ):
        raise WeightStabilityError("inner OOF validation record differs")

    statistics = two_family_well_sufficient_statistics(
        frame,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    record = {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "rows": int(len(frame)),
        "wells": int(len(actual_wells)),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha,
        "oof_path": oof_path.relative_to(root).as_posix(),
        "oof_sha256": oof_sha,
        "fit_columns_parsed": [
            *OOF_KEY_COLUMNS,
            "y_true",
            INCUMBENT_COLUMN,
            TCN_COLUMN,
            *(f"{TCN_COLUMN}__seed_{seed}" for seed in tcn_seeds),
        ],
        "outer_validation_labels_parsed": False,
        "tcn_family_mean_recomputed": True,
    }
    return statistics, record


def _weight_summary(
    values: np.ndarray,
    *,
    family: str,
    target: float,
    configured_cap: float,
    effective_lower_bound: float,
    effective_upper_bound: float,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.isfinite(vector).all():
        raise ValueError("weight summary requires a finite non-empty vector")
    quantiles = np.quantile(
        vector,
        [0.01, 0.05, 0.50, 0.95, 0.99],
        method=QUANTILE_METHOD,
    )
    lower_hits = np.isclose(
        vector,
        effective_lower_bound,
        rtol=0.0,
        atol=1.0e-12,
    )
    upper_hits = np.isclose(
        vector,
        effective_upper_bound,
        rtol=0.0,
        atol=1.0e-12,
    )
    cap_hits = np.isclose(
        vector,
        configured_cap,
        rtol=0.0,
        atol=1.0e-12,
    )
    return {
        "family": family,
        "samples": int(len(vector)),
        "mean": float(np.mean(vector)),
        "std": float(np.std(vector, ddof=1)) if len(vector) > 1 else 0.0,
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
        "target": float(target),
        "mean_absolute_target_distance": float(np.mean(np.abs(vector - target))),
        "configured_cap": float(configured_cap),
        "effective_lower_bound": float(effective_lower_bound),
        "effective_upper_bound": float(effective_upper_bound),
        "lower_boundary_hit_count": int(lower_hits.sum()),
        "lower_boundary_hit_rate": float(np.mean(lower_hits)),
        "upper_boundary_hit_count": int(upper_hits.sum()),
        "upper_boundary_hit_rate": float(np.mean(upper_hits)),
        "cap_hit_count": int(cap_hits.sum()),
        "cap_hit_rate": float(np.mean(cap_hits)),
    }


def _scope_records(
    tcn_values: np.ndarray,
    *,
    scope: str,
    outer_fold: int | None,
) -> list[dict[str, Any]]:
    tcn = _weight_summary(
        tcn_values,
        family="tcn",
        target=float(REGULARIZATION_TARGET["tcn"]),
        configured_cap=float(FAMILY_MAX_WEIGHT["tcn"]),
        effective_lower_bound=0.0,
        effective_upper_bound=float(FAMILY_MAX_WEIGHT["tcn"]),
    )
    incumbent = _weight_summary(
        1.0 - np.asarray(tcn_values, dtype=np.float64),
        family="incumbent",
        target=float(REGULARIZATION_TARGET["incumbent"]),
        configured_cap=1.0,
        effective_lower_bound=1.0 - float(FAMILY_MAX_WEIGHT["tcn"]),
        effective_upper_bound=1.0,
    )
    result: list[dict[str, Any]] = []
    for summary in (incumbent, tcn):
        result.append(
            {
                "scope": scope,
                "outer_fold": outer_fold,
                **summary,
            }
        )
    return result


def analyze_nested_blend_weight_stability(
    run_root: Path,
    *,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Run the fixed 10k well-cluster bootstrap and atomically persist results."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"nested run root does not exist: {root}")
    if not _SAFE_OUTPUT.fullmatch(str(output_name)):
        raise ValueError("output_name must be one safe directory name")
    output = (root / output_name).resolve()
    if output.parent != root:
        raise ValueError("weight-stability output must be inside the run root")
    if output.exists():
        raise FileExistsError(f"weight-stability output already exists: {output}")

    (
        run_config,
        reaudit,
        source_locks,
        outer_folds,
        tcn_seeds,
        training_by_fold,
        validation_by_fold,
    ) = _validate_source_locks(root)
    audit_records = _audit_inner_records(reaudit, outer_folds)
    incumbent_seeds = tuple(map(int, run_config["policy"]["incumbent_seeds"]))
    artifact_records: list[dict[str, Any]] = []
    statistics_by_outer: dict[int, pd.DataFrame] = {}
    all_statistics: list[pd.DataFrame] = []
    for outer_fold in outer_folds:
        split_statistics: list[pd.DataFrame] = []
        observed_wells: set[str] = set()
        for inner_fold in outer_folds:
            if inner_fold == outer_fold:
                continue
            expected_validation = validation_by_fold[inner_fold]
            expected_training = tuple(
                sorted(set(training_by_fold[outer_fold]) - set(expected_validation))
            )
            statistics, record = _validate_and_summarize_inner_split(
                root=root,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                run_config=run_config,
                audit_record=audit_records[(outer_fold, inner_fold)],
                expected_training=expected_training,
                expected_validation=expected_validation,
                excluded_outer=validation_by_fold[outer_fold],
                tcn_seeds=tcn_seeds,
                incumbent_seed_count=len(incumbent_seeds),
            )
            current_wells = set(statistics["well_id"].astype(str))
            if observed_wells & current_wells:
                raise WeightStabilityError(
                    f"inner validation wells repeat for outer fold {outer_fold}"
                )
            observed_wells.update(current_wells)
            split_statistics.append(statistics)
            all_statistics.append(statistics)
            artifact_records.append(record)
        if observed_wells != set(training_by_fold[outer_fold]):
            raise WeightStabilityError(
                f"four inner OOFs do not exactly cover outer-training fold {outer_fold}"
            )
        outer_statistics = (
            pd.concat(split_statistics, ignore_index=True)
            .sort_values(["inner_fold", "well_id"], kind="stable")
            .reset_index(drop=True)
        )
        if (
            len(outer_statistics) != len(training_by_fold[outer_fold])
            or int(outer_statistics["n"].sum()) <= 0
        ):
            raise WeightStabilityError("outer-fold sufficient-stat scope differs")
        statistics_by_outer[outer_fold] = outer_statistics

    canonical_statistics = pd.concat(all_statistics, ignore_index=True)
    statistics_sha = _statistics_sha256(canonical_statistics)
    audit_weights = reaudit.get("weights")
    if not isinstance(audit_weights, list):
        raise WeightStabilityError("independent re-audit weights are missing")
    persisted_weights: dict[tuple[int, str], float] = {}
    for record in audit_weights:
        if not isinstance(record, Mapping):
            raise WeightStabilityError("invalid independent re-audit weight record")
        key = (int(record.get("outer_fold", -1)), str(record.get("family", "")))
        if key in persisted_weights:
            raise WeightStabilityError("duplicate independent re-audit weight record")
        persisted_weights[key] = float(record.get("weight", math.nan))
    expected_weight_keys = {
        (outer_fold, family)
        for outer_fold in outer_folds
        for family in ("incumbent", "tcn")
    }
    if set(persisted_weights) != expected_weight_keys or not all(
        math.isfinite(value) for value in persisted_weights.values()
    ):
        raise WeightStabilityError("independent re-audit weight coverage differs")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fold_bootstrap: dict[int, np.ndarray] = {}
    fold_results: list[dict[str, Any]] = []
    maximum_point_weight_delta = 0.0
    for outer_fold in outer_folds:
        statistics = statistics_by_outer[outer_fold]
        point_fit = fit_two_family_sufficient_statistics(statistics)
        persisted_tcn = persisted_weights[(outer_fold, "tcn")]
        persisted_incumbent = persisted_weights[(outer_fold, "incumbent")]
        point_delta = max(
            abs(float(point_fit["tcn_weight"]) - persisted_tcn),
            abs(float(point_fit["incumbent_weight"]) - persisted_incumbent),
        )
        maximum_point_weight_delta = max(maximum_point_weight_delta, point_delta)
        if point_delta > 1.0e-8:
            raise WeightStabilityError(
                "sufficient-stat point fit differs from independently re-audited "
                f"weight for outer fold {outer_fold}: {point_delta}"
            )
        tcn_weights = bootstrap_two_family_weights(
            statistics,
            samples=BOOTSTRAP_SAMPLES,
            rng=rng,
            batch_size=BOOTSTRAP_BATCH_SIZE,
        )
        fold_bootstrap[outer_fold] = tcn_weights
        fold_tcn_summary = _weight_summary(
            tcn_weights,
            family="tcn",
            target=float(REGULARIZATION_TARGET["tcn"]),
            configured_cap=float(FAMILY_MAX_WEIGHT["tcn"]),
            effective_lower_bound=0.0,
            effective_upper_bound=float(FAMILY_MAX_WEIGHT["tcn"]),
        )
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "inner_wells": int(len(statistics)),
                "inner_rows": int(statistics["n"].sum()),
                "point_fit": point_fit,
                "persisted_reaudit_weights": {
                    "incumbent": persisted_incumbent,
                    "tcn": persisted_tcn,
                },
                "point_fit_maximum_absolute_weight_delta": point_delta,
                "bootstrap_tcn": fold_tcn_summary,
            }
        )

    fold_matrix = np.vstack([fold_bootstrap[fold] for fold in outer_folds])
    pooled_all_fold_draws = fold_matrix.reshape(-1)
    pooled_fold_mean = np.mean(fold_matrix, axis=0)
    summary_records: list[dict[str, Any]] = []
    for outer_fold in outer_folds:
        summary_records.extend(
            _scope_records(
                fold_bootstrap[outer_fold],
                scope=f"outer_fold_{outer_fold}",
                outer_fold=outer_fold,
            )
        )
    summary_records.extend(
        _scope_records(
            pooled_all_fold_draws,
            scope="pooled_all_fold_draws",
            outer_fold=None,
        )
    )
    summary_records.extend(
        _scope_records(
            pooled_fold_mean,
            scope="pooled_fold_mean",
            outer_fold=None,
        )
    )
    statistics_frame = pd.DataFrame(summary_records)
    summary_numeric = statistics_frame.select_dtypes(include=[np.number]).drop(
        columns=["outer_fold"],
        errors="ignore",
    )
    if not np.isfinite(summary_numeric.to_numpy(dtype=np.float64)).all():
        raise WeightStabilityError(
            "weight-stability summary contains non-finite values"
        )

    module_path = Path(__file__).resolve()
    ensemble_path = module_path.parents[1] / "ensemble.py"
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_name}.tmp-",
            dir=root,
        )
    )
    try:
        csv_path = temporary / STATISTICS_FILENAME
        statistics_frame.to_csv(
            csv_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        locks = {
            **source_locks,
            "inner_artifacts": artifact_records,
            "inner_artifact_set_sha256": _canonical_sha256(artifact_records),
        }
        payload: dict[str, Any] = {
            "schema_version": WEIGHT_STABILITY_SCHEMA_VERSION,
            "completed": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_config.get("run_id"),
            "run_root": str(root),
            "evidence_mode": "conditional_weight_crossfit",
            "evidence_eligible": False,
            "source_artifact_legacy_evidence_eligible": True,
            "evidence_contract": {
                "model_fit_safe": True,
                "weight_fit_safe": True,
                "family_selection_safe": False,
                "overall_promotion_eligible": False,
                "reason": (
                    "TCN family/config was selected from global quick/full OOF "
                    "before the outer-fold weight cross-fit"
                ),
            },
            "analysis": {
                "objective": ("pooled row-MSE + 25 * ||w - (0.75, 0.25)||^2"),
                "families": ["incumbent", "tcn"],
                "family_columns": {
                    "incumbent": [INCUMBENT_COLUMN],
                    "tcn": [TCN_COLUMN],
                },
                "family_first_seed_average": True,
                "regularization_strength": REGULARIZATION_STRENGTH,
                "regularization_target": REGULARIZATION_TARGET,
                "family_max_weight": FAMILY_MAX_WEIGHT,
                "closed_form": (
                    "alpha_tcn = clip((2*lambda*N*0.25 - B) / (A + 2*lambda*N), 0, 0.5)"
                ),
                "sufficient_statistics": {
                    "definitions": {
                        "e": "y_pred_incumbent - y_true",
                        "d": "y_pred_tcn - y_pred_incumbent",
                        "n": "row count",
                        "A": "sum(d^2)",
                        "B": "sum(e*d)",
                        "C": "sum(e^2)",
                    },
                    "canonical_binary_schema": STATISTICS_DIGEST_SCHEMA,
                    "sha256": statistics_sha,
                    "rows": int(len(canonical_statistics)),
                    "not_persisted": True,
                },
                "bootstrap": {
                    "samples_per_outer_fold": BOOTSTRAP_SAMPLES,
                    "outer_folds": list(outer_folds),
                    "seed": BOOTSTRAP_SEED,
                    "bit_generator": type(rng.bit_generator).__name__,
                    "numpy_version": np.__version__,
                    "scipy_version": scipy.__version__,
                    "resampling_unit": "well",
                    "draws_per_replicate": (
                        "number of outer-training wells in that outer fold"
                    ),
                    "cluster_multiplicity_sampler": (
                        "multinomial; distribution-equivalent to uniform sampling "
                        "wells with replacement"
                    ),
                    "stratification": "none",
                    "all_rows_retained_per_selected_well": True,
                    "batch_size": BOOTSTRAP_BATCH_SIZE,
                    "quantile_method": QUANTILE_METHOD,
                    "std_ddof": 1,
                    "pooled_all_fold_draws_definition": (
                        "descriptive mixture of the 5 x 10,000 fold-specific fits"
                    ),
                    "pooled_fold_mean_definition": (
                        "sample-index-wise mean of five independently drawn "
                        "fold-specific weights; 10,000 values"
                    ),
                },
                "limitations": [
                    (
                        "Measures empirical well-composition stability conditional "
                        "on fixed persisted inner predictions."
                    ),
                    (
                        "Does not measure model-refit, representation, optimizer, "
                        "or domain-shift uncertainty."
                    ),
                    (
                        "Pooled fold summaries are descriptive; each outer fold "
                        "retains its own independently fitted deployment weight."
                    ),
                    (
                        "Family-selection uncertainty is not nested: TCN was "
                        "selected from earlier global quick/full OOF."
                    ),
                ],
            },
            "scope": {
                "outer_folds": list(outer_folds),
                "inner_splits": len(artifact_records),
                "unique_training_wells_by_outer_fold": {
                    str(fold): int(len(statistics_by_outer[fold]))
                    for fold in outer_folds
                },
                "inner_rows_by_outer_fold": {
                    str(fold): int(statistics_by_outer[fold]["n"].sum())
                    for fold in outer_folds
                },
                "bootstrap_samples_per_outer_fold": BOOTSTRAP_SAMPLES,
                "pooled_all_fold_draws": int(len(pooled_all_fold_draws)),
                "pooled_fold_mean_samples": int(len(pooled_fold_mean)),
            },
            "locks": locks,
            "point_fit_equivalence": {
                "reference": "independent_nested_reaudit.json weights",
                "maximum_absolute_weight_delta": maximum_point_weight_delta,
                "tolerance": 1.0e-8,
                "passed": True,
            },
            "fold_results": fold_results,
            "pooled_all_fold_draws": {
                "incumbent": _weight_summary(
                    1.0 - pooled_all_fold_draws,
                    family="incumbent",
                    target=float(REGULARIZATION_TARGET["incumbent"]),
                    configured_cap=1.0,
                    effective_lower_bound=0.5,
                    effective_upper_bound=1.0,
                ),
                "tcn": _weight_summary(
                    pooled_all_fold_draws,
                    family="tcn",
                    target=float(REGULARIZATION_TARGET["tcn"]),
                    configured_cap=0.5,
                    effective_lower_bound=0.0,
                    effective_upper_bound=0.5,
                ),
            },
            "pooled_fold_mean": {
                "incumbent": _weight_summary(
                    1.0 - pooled_fold_mean,
                    family="incumbent",
                    target=float(REGULARIZATION_TARGET["incumbent"]),
                    configured_cap=1.0,
                    effective_lower_bound=0.5,
                    effective_upper_bound=1.0,
                ),
                "tcn": _weight_summary(
                    pooled_fold_mean,
                    family="tcn",
                    target=float(REGULARIZATION_TARGET["tcn"]),
                    configured_cap=0.5,
                    effective_lower_bound=0.0,
                    effective_upper_bound=0.5,
                ),
            },
            "checks": {
                "only_inner_oof_labels_parsed": True,
                "outer_validation_labels_used_for_fit": False,
                "final_oof_csv_parsed": False,
                "final_oof_bytes_sha256_locked": True,
                "run_config_sha256_and_canonical_digest_locked": True,
                "independent_reaudit_sha256_locked": True,
                "all_inner_manifest_and_oof_sha256_locked": True,
                "inner_oof_exact_outer_training_coverage": True,
                "inner_outer_well_disjointness": True,
                "inner_oof_keys_unique": True,
                "inner_oof_values_finite": True,
                "tcn_family_seed_mean_recomputed": True,
                "point_fit_matches_independent_reaudit": True,
                "bootstrap_weights_finite": True,
                "data_directory_accessed": False,
                "submission_created": False,
            },
            "implementation": {
                "module_path": str(module_path),
                "module_sha256": sha256_file(module_path),
                "ensemble_objective_path": str(ensemble_path),
                "ensemble_objective_sha256": sha256_file(ensemble_path),
                "locked_run_implementation_hashes": run_config.get(
                    "implementation_hashes"
                ),
            },
            "files": {
                "statistics": {
                    "path": STATISTICS_FILENAME,
                    "sha256": sha256_file(csv_path),
                    "rows": int(len(statistics_frame)),
                },
                "summary": {"path": SUMMARY_FILENAME},
            },
            "submission_created": False,
        }
        _write_json(temporary / SUMMARY_FILENAME, payload)

        if (
            sha256_file(root / "run_config.json")
            != source_locks["run_config"]["sha256"]
            or sha256_file(root / "independent_nested_reaudit.json")
            != source_locks["independent_nested_reaudit"]["sha256"]
            or sha256_file(root / "oof_predictions.csv.gz")
            != source_locks["final_oof"]["sha256"]
        ):
            raise WeightStabilityError("locked source bytes changed during analysis")
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return payload


__all__ = [
    "BOOTSTRAP_BATCH_SIZE",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "DEFAULT_OUTPUT_NAME",
    "STATISTICS_FILENAME",
    "SUMMARY_FILENAME",
    "WEIGHT_STABILITY_SCHEMA_VERSION",
    "WeightStabilityError",
    "analyze_nested_blend_weight_stability",
    "bootstrap_two_family_weights",
    "fit_two_family_sufficient_statistics",
    "two_family_well_sufficient_statistics",
]
