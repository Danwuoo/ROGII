"""Disk-only target-free stress analysis for an independently audited nested run."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from ..diagnostics.nested_external_ensemble import (
    FIXED_050_COLUMN,
    FIXED_075_COLUMN,
    INCUMBENT_COLUMN,
    NESTED_COLUMN,
    TCN_COLUMN,
)
from ..portfolio_metrics import (
    DEFAULT_TRUTH_COLUMN,
    make_target_free_strata,
    target_free_stratified_metrics,
)
from .artifacts import sha256_file
from .nested_reaudit import (
    DEFAULT_OUTPUT_NAME as DEFAULT_NESTED_REAUDIT_NAME,
    NESTED_REAUDIT_SCHEMA_VERSION,
)
from .stress import (
    EXTERNAL_STRESS_SCHEMA_VERSION,
    TARGET_FREE_FEATURE_COLUMNS,
)


NESTED_STRESS_SCHEMA_VERSION = "rogii_nested_target_free_stress_v1"
DEFAULT_OUTPUT_NAME = "stress"
FEATURE_FILENAME = "target_free_well_features.csv"
STRATA_FILENAME = "target_free_strata.csv"
METRICS_FILENAME = "target_free_stratified_metrics.csv"
SUMMARY_FILENAME = "summary.json"
PREDICTION_COLUMNS = (
    INCUMBENT_COLUMN,
    TCN_COLUMN,
    NESTED_COLUMN,
    FIXED_075_COLUMN,
    FIXED_050_COLUMN,
)
STRATA_COLUMNS = (
    "well_id",
    "stratum_feature",
    "stratum",
    "feature_value",
    "lower_bound",
    "upper_bound",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_OUTPUT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_STRATA_NAMES = frozenset(
    {
        "y_true",
        "target",
        "label",
        "prediction",
        "residual",
        "error",
        "rmse",
        "sse",
    }
)


class NestedStressError(RuntimeError):
    """Raised when a frozen stress source or nested evidence lock is invalid."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NestedStressError(f"unreadable {label}: {path}") from error
    if not isinstance(payload, dict):
        raise NestedStressError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(value: Any, *, label: str) -> str:
    result = str(value).lower()
    if not _SHA256.fullmatch(result):
        raise NestedStressError(f"{label} must be a lowercase SHA-256")
    return result


def _resolved_declared_path(
    value: Any,
    *,
    root: Path,
    expected: Path,
    label: str,
) -> Path:
    declared = Path(str(value))
    resolved = (
        (root / declared).resolve()
        if not declared.is_absolute()
        else declared.resolve()
    )
    if resolved != expected.resolve():
        raise NestedStressError(f"{label} path differs from the required artifact")
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


def _source_stress_inputs(
    external_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, str],
]:
    """Load and freeze target-free source files before any nested OOF access."""

    root = Path(external_root).resolve()
    source_summary_path = root / "summary.json"
    stress_root = root / "stress"
    stress_summary_path = stress_root / SUMMARY_FILENAME
    feature_path = stress_root / FEATURE_FILENAME
    strata_path = stress_root / STRATA_FILENAME
    for path, label in (
        (source_summary_path, "full external run summary"),
        (stress_summary_path, "external stress summary"),
        (feature_path, "target-free feature file"),
        (strata_path, "target-free strata file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    source_summary = _read_json(
        source_summary_path,
        label="full external run summary",
    )
    if (
        source_summary.get("profile") != "full"
        or int(source_summary.get("failure_count", -1)) != 0
        or source_summary.get("incomplete_family_seeds") != []
        or source_summary.get("data_manifest_equal") is not True
        or source_summary.get("submission_created") is not False
        or "y_pred__tcn"
        not in tuple(map(str, source_summary.get("eligible_family_columns", ())))
    ):
        raise NestedStressError("external source is not a clean completed full run")
    source_scope = source_summary.get("scope")
    if not isinstance(source_scope, Mapping):
        raise NestedStressError("external full summary lacks scope")
    if tuple(map(int, source_scope.get("folds", ()))) != (0, 1, 2, 3, 4):
        raise NestedStressError("external full summary fold scope differs")

    stress_summary = _read_json(
        stress_summary_path,
        label="external stress summary",
    )
    stress_schema = stress_summary.get("schema_version")
    if (
        stress_schema is not None and stress_schema != EXTERNAL_STRESS_SCHEMA_VERSION
    ) or (
        "completed" in stress_summary and stress_summary.get("completed") is not True
    ):
        raise NestedStressError("external stress summary schema/completion differs")
    if (
        stress_summary.get("target_columns_used_for_strata") != []
        or stress_summary.get("prediction_columns_used_for_strata") != []
    ):
        raise NestedStressError(
            "external stress strata declare target or prediction inputs"
        )
    if Path(str(stress_summary.get("run_root", ""))).resolve() != root:
        raise NestedStressError("external stress summary run root differs")
    feature_columns = tuple(map(str, stress_summary.get("feature_columns", ())))
    if feature_columns != TARGET_FREE_FEATURE_COLUMNS:
        raise NestedStressError("external stress feature schema differs")
    if any(
        name.lower() in _FORBIDDEN_STRATA_NAMES or name.lower().startswith("y_pred")
        for name in feature_columns
    ):
        raise NestedStressError("external stress feature schema is target-derived")
    bins = int(stress_summary.get("bins", 0))
    if bins < 1:
        raise NestedStressError("external stress bin count is invalid")

    feature_sha = sha256_file(feature_path)
    if feature_sha != _sha256(
        stress_summary.get("feature_sha256"),
        label="external stress feature sha256",
    ):
        raise NestedStressError("target-free feature bytes differ from summary")
    strata_sha = sha256_file(strata_path)
    declared_strata_sha = stress_summary.get("strata_sha256")
    if declared_strata_sha is not None and strata_sha != _sha256(
        declared_strata_sha,
        label="external stress strata sha256",
    ):
        raise NestedStressError("target-free strata bytes differ from summary")
    registration_path = Path(
        str(stress_summary.get("registration_diagnostics", ""))
    ).resolve()
    if not registration_path.is_file():
        raise FileNotFoundError(
            f"stress registration diagnostics do not exist: {registration_path}"
        )
    registration_sha = sha256_file(registration_path)
    if registration_sha != _sha256(
        stress_summary.get("registration_diagnostics_sha256"),
        label="stress registration diagnostics sha256",
    ):
        raise NestedStressError(
            "registration diagnostics bytes differ from stress summary"
        )

    try:
        features = pd.read_csv(feature_path, dtype={"well_id": "string"})
        strata = pd.read_csv(
            strata_path,
            dtype={
                "well_id": "string",
                "stratum_feature": "string",
                "stratum": "string",
            },
        )
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise NestedStressError("failed to read frozen target-free inputs") from error

    if (
        tuple(map(str, features.columns)) != ("well_id", *TARGET_FREE_FEATURE_COLUMNS)
        or features.empty
        or features["well_id"].isna().any()
        or features["well_id"].astype(str).str.len().eq(0).any()
        or features["well_id"].astype(str).duplicated().any()
        or not np.isfinite(
            features.loc[
                :,
                list(TARGET_FREE_FEATURE_COLUMNS),
            ].to_numpy(dtype=np.float64)
        ).all()
    ):
        raise NestedStressError("target-free feature contents are invalid")
    if tuple(map(str, strata.columns)) != STRATA_COLUMNS:
        raise NestedStressError(
            "target-free strata schema contains unexpected target/prediction columns"
        )
    if (
        strata.empty
        or strata[["well_id", "stratum_feature", "stratum"]].isna().any().any()
        or strata["well_id"].astype(str).str.len().eq(0).any()
        or strata["stratum_feature"].astype(str).str.len().eq(0).any()
        or strata["stratum"].astype(str).str.len().eq(0).any()
        or strata.duplicated(["well_id", "stratum_feature"]).any()
        or set(strata["stratum_feature"].astype(str))
        != set(TARGET_FREE_FEATURE_COLUMNS)
        or not np.isfinite(
            strata[["feature_value", "lower_bound", "upper_bound"]].to_numpy(
                dtype=np.float64
            )
        ).all()
    ):
        raise NestedStressError("target-free strata contents are invalid")
    feature_wells = set(features["well_id"].astype(str))
    for feature, group in strata.groupby("stratum_feature", sort=False):
        if (
            str(feature) not in TARGET_FREE_FEATURE_COLUMNS
            or set(group["well_id"].astype(str)) != feature_wells
        ):
            raise NestedStressError("target-free strata well coverage differs")

    # The production source summary predates its explicit strata hash.  Exact
    # deterministic reconstruction from the hash-locked feature file is the
    # compatibility lock for that immutable run; newer summaries additionally
    # provide ``strata_sha256`` and are checked above.
    reconstructed = make_target_free_strata(
        features,
        TARGET_FREE_FEATURE_COLUMNS,
        bins=bins,
    )
    try:
        pd.testing.assert_frame_equal(
            strata.reset_index(drop=True),
            reconstructed.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1.0e-12,
        )
    except AssertionError as error:
        raise NestedStressError(
            "target-free strata differ from deterministic frozen features"
        ) from error

    expected_wells = int(stress_summary.get("wells", -1))
    expected_rows = int(stress_summary.get("rows", -1))
    if (
        len(features) != expected_wells
        or len(strata) != int(stress_summary.get("strata_rows", -1))
        or int(source_scope.get("wells", -1)) != expected_wells
        or int(source_scope.get("rows", -1)) != expected_rows
    ):
        raise NestedStressError("external stress/full scope differs")

    hashes = {
        "source_run_summary_sha256": sha256_file(source_summary_path),
        "source_stress_summary_sha256": sha256_file(stress_summary_path),
        "feature_sha256": feature_sha,
        "strata_sha256": strata_sha,
        "registration_diagnostics_sha256": registration_sha,
    }
    # Re-hash the small frozen inputs after parsing to reject concurrent drift.
    if (
        sha256_file(feature_path) != feature_sha
        or sha256_file(strata_path) != strata_sha
        or sha256_file(stress_summary_path) != hashes["source_stress_summary_sha256"]
        or sha256_file(source_summary_path) != hashes["source_run_summary_sha256"]
        or sha256_file(registration_path) != hashes["registration_diagnostics_sha256"]
    ):
        raise NestedStressError("target-free source changed while being frozen")
    return features, strata, stress_summary, hashes


def _audited_nested_oof(
    nested_root: Path,
    *,
    external_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    root = Path(nested_root).resolve()
    audit_path = root / DEFAULT_NESTED_REAUDIT_NAME
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"independent nested re-audit does not exist: {audit_path}"
        )
    audit_sha = sha256_file(audit_path)
    audit = _read_json(audit_path, label="independent nested re-audit")
    checks = audit.get("checks")
    artifacts = audit.get("artifacts")
    final_record = (
        artifacts.get("final_oof") if isinstance(artifacts, Mapping) else None
    )
    if (
        audit.get("schema_version") != NESTED_REAUDIT_SCHEMA_VERSION
        or audit.get("passed") is not True
        or audit.get("evidence_mode") != "production"
        or audit.get("evidence_eligible") is not True
        or audit.get("submission_created") is not False
        or not isinstance(checks, Mapping)
        or checks.get("submission_created") is not False
        or not isinstance(final_record, Mapping)
    ):
        raise NestedStressError(
            "independent nested re-audit is failed or evidence-ineligible"
        )
    if Path(str(audit.get("run_root", ""))).resolve() != root:
        raise NestedStressError("independent nested re-audit run root differs")
    audited_source = audit.get("source")
    if (
        not isinstance(audited_source, Mapping)
        or audited_source.get("mode") != "production"
        or audited_source.get("independently_locked") is not True
        or Path(str(audited_source.get("root", ""))).resolve()
        != Path(external_root).resolve()
    ):
        raise NestedStressError(
            "nested re-audit source differs from frozen external stress run"
        )
    _sha256(
        audited_source.get("oof_sha256"),
        label="nested audited source OOF sha256",
    )
    _sha256(
        audited_source.get("manifest_sha256"),
        label="nested audited source manifest sha256",
    )
    if _submission_csvs(root):
        raise NestedStressError("nested run contains a submission CSV")

    final_path = root / "oof_predictions.csv.gz"
    _resolved_declared_path(
        final_record.get("path"),
        root=root,
        expected=final_path,
        label="independent nested final OOF",
    )
    if not final_path.is_file():
        raise FileNotFoundError(f"nested final OOF does not exist: {final_path}")
    declared_oof_sha = _sha256(
        final_record.get("sha256"),
        label="independent nested final OOF sha256",
    )
    final_stat = final_path.stat()
    actual_oof_sha = sha256_file(final_path)
    if actual_oof_sha != declared_oof_sha:
        raise NestedStressError(
            "nested final OOF bytes differ from independent re-audit"
        )
    required = [
        "well_id",
        "row_index",
        "fold",
        DEFAULT_TRUTH_COLUMN,
        *PREDICTION_COLUMNS,
    ]
    try:
        frame = pd.read_csv(
            final_path,
            usecols=required,
            dtype={"well_id": "string", "row_index": "int64", "fold": "int64"},
            float_precision="round_trip",
        )
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise NestedStressError("failed to read audited nested final OOF") from error
    after_stat = final_path.stat()
    if (
        final_stat.st_size != after_stat.st_size
        or final_stat.st_mtime_ns != after_stat.st_mtime_ns
        or sha256_file(audit_path) != audit_sha
    ):
        raise NestedStressError("nested audit or final OOF changed during read")
    scope = audit.get("scope")
    if (
        not isinstance(scope, Mapping)
        or len(frame) != int(scope.get("rows", -1))
        or int(frame["well_id"].nunique()) != int(scope.get("wells", -1))
    ):
        raise NestedStressError("nested final OOF scope differs from re-audit")
    return (
        frame,
        audit,
        {
            "independent_nested_reaudit_sha256": audit_sha,
            "final_oof_sha256": actual_oof_sha,
        },
    )


def analyze_nested_target_free_stress(
    nested_run_root: Path,
    *,
    external_run_root: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Apply frozen full-source strata to an audited production nested OOF."""

    nested_root = Path(nested_run_root).resolve()
    external_root = Path(external_run_root).resolve()
    if not nested_root.is_dir():
        raise FileNotFoundError(f"nested run root does not exist: {nested_root}")
    if not external_root.is_dir():
        raise FileNotFoundError(
            f"external full run root does not exist: {external_root}"
        )
    identifier = str(output_name)
    if not _SAFE_OUTPUT.fullmatch(identifier):
        raise ValueError("output_name must be a path-safe identifier")
    output_root = nested_root / identifier
    if output_root.exists():
        raise FileExistsError(f"nested stress output already exists: {output_root}")

    # This call fully validates and freezes source features/strata before the
    # nested audit or final OOF is opened.
    features, strata, stress_summary, source_hashes = _source_stress_inputs(
        external_root
    )
    final_oof, audit, nested_hashes = _audited_nested_oof(
        nested_root,
        external_root=external_root,
    )
    audited_source = audit.get("source")
    if not isinstance(audited_source, Mapping):
        raise NestedStressError("nested re-audit source record is invalid")
    well_ids = set(features["well_id"].astype(str))
    if (
        set(final_oof["well_id"].astype(str)) != well_ids
        or len(final_oof) != int(stress_summary["rows"])
        or int(final_oof["well_id"].nunique()) != int(stress_summary["wells"])
    ):
        raise NestedStressError(
            "nested final OOF differs from frozen target-free well scope"
        )
    incumbent = final_oof[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
    raw_tcn = final_oof[TCN_COLUMN].to_numpy(dtype=np.float64)
    if not np.allclose(
        final_oof[FIXED_075_COLUMN].to_numpy(dtype=np.float64),
        0.75 * incumbent + 0.25 * raw_tcn,
        rtol=0.0,
        atol=1.0e-10,
    ) or not np.allclose(
        final_oof[FIXED_050_COLUMN].to_numpy(dtype=np.float64),
        0.50 * incumbent + 0.50 * raw_tcn,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise NestedStressError("nested fixed-control predictions differ")

    try:
        metrics = target_free_stratified_metrics(
            final_oof,
            strata,
            PREDICTION_COLUMNS,
            baseline_column=INCUMBENT_COLUMN,
        )
    except (TypeError, ValueError, ArithmeticError) as error:
        raise NestedStressError(
            "target-free stratified metric computation failed"
        ) from error
    numeric_metric_columns = (
        "n_wells",
        "n_rows",
        "sse",
        "pooled_row_rmse",
        "baseline_rmse",
        "delta_rmse_vs_baseline",
    )
    if (
        metrics.empty
        or set(metrics["candidate"].astype(str)) != set(PREDICTION_COLUMNS)
        or not np.isfinite(
            metrics.loc[:, list(numeric_metric_columns)].to_numpy(dtype=np.float64)
        ).all()
    ):
        raise NestedStressError("target-free stratified metrics are invalid")
    for _, feature_metrics in metrics.groupby("stratum_feature", sort=False):
        row_totals = feature_metrics.groupby("candidate", sort=False)["n_rows"].sum()
        if (
            set(row_totals.index.astype(str)) != set(PREDICTION_COLUMNS)
            or not (row_totals.to_numpy(dtype=np.int64) == len(final_oof)).all()
        ):
            raise NestedStressError("target-free stratum join dropped nested OOF rows")
    candidate_metrics = metrics.loc[~metrics["candidate"].eq(INCUMBENT_COLUMN)]
    best = {
        str(candidate): float(group["delta_rmse_vs_baseline"].min())
        for candidate, group in candidate_metrics.groupby("candidate", sort=True)
    }
    worst = {
        str(candidate): float(group["delta_rmse_vs_baseline"].max())
        for candidate, group in candidate_metrics.groupby("candidate", sort=True)
    }
    locked_files = (
        (
            external_root / "summary.json",
            source_hashes["source_run_summary_sha256"],
        ),
        (
            external_root / "stress" / SUMMARY_FILENAME,
            source_hashes["source_stress_summary_sha256"],
        ),
        (
            external_root / "stress" / FEATURE_FILENAME,
            source_hashes["feature_sha256"],
        ),
        (
            external_root / "stress" / STRATA_FILENAME,
            source_hashes["strata_sha256"],
        ),
        (
            Path(str(stress_summary["registration_diagnostics"])),
            source_hashes["registration_diagnostics_sha256"],
        ),
        (
            nested_root / DEFAULT_NESTED_REAUDIT_NAME,
            nested_hashes["independent_nested_reaudit_sha256"],
        ),
        (
            nested_root / "oof_predictions.csv.gz",
            nested_hashes["final_oof_sha256"],
        ),
    )
    if any(sha256_file(path) != digest for path, digest in locked_files):
        raise NestedStressError(
            "a locked source artifact changed during stress analysis"
        )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{identifier}.tmp-",
            dir=nested_root,
        )
    )
    try:
        metrics_path = temporary / METRICS_FILENAME
        metrics.to_csv(
            metrics_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        metrics_sha = sha256_file(metrics_path)
        summary = {
            "schema_version": NESTED_STRESS_SCHEMA_VERSION,
            "completed": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "nested_run_root": str(nested_root),
            "external_run_root": str(external_root),
            "output_root": str(output_root),
            "evidence_eligible": True,
            "rows": int(len(final_oof)),
            "wells": int(final_oof["well_id"].nunique()),
            "bins": int(stress_summary["bins"]),
            "feature_columns": list(TARGET_FREE_FEATURE_COLUMNS),
            "prediction_columns": list(PREDICTION_COLUMNS),
            "baseline_column": INCUMBENT_COLUMN,
            "feature_rows": int(len(features)),
            "strata_rows": int(len(strata)),
            "metric_rows": int(len(metrics)),
            "candidate_best_delta": best,
            "candidate_worst_delta": worst,
            "feature_path": str(external_root / "stress" / FEATURE_FILENAME),
            "feature_sha256": source_hashes["feature_sha256"],
            "strata_path": str(external_root / "stress" / STRATA_FILENAME),
            "strata_sha256": source_hashes["strata_sha256"],
            "registration_diagnostics": stress_summary["registration_diagnostics"],
            "registration_diagnostics_sha256": source_hashes[
                "registration_diagnostics_sha256"
            ],
            "independent_nested_reaudit_path": str(
                nested_root / DEFAULT_NESTED_REAUDIT_NAME
            ),
            "independent_nested_reaudit_sha256": nested_hashes[
                "independent_nested_reaudit_sha256"
            ],
            "final_oof_path": str(nested_root / "oof_predictions.csv.gz"),
            "final_oof_sha256": nested_hashes["final_oof_sha256"],
            "metrics_path": str(output_root / METRICS_FILENAME),
            "metrics_sha256": metrics_sha,
            "source_run_summary_sha256": source_hashes["source_run_summary_sha256"],
            "source_stress_summary_sha256": source_hashes[
                "source_stress_summary_sha256"
            ],
            "source_stress_declared_feature_sha256": stress_summary["feature_sha256"],
            "source_stress_declared_strata_sha256": stress_summary.get("strata_sha256"),
            "source_oof_sha256": _sha256(
                audited_source.get("oof_sha256"),
                label="nested audited source OOF sha256",
            ),
            "source_manifest_sha256": _sha256(
                audited_source.get("manifest_sha256"),
                label="nested audited source manifest sha256",
            ),
            "independent_nested_reaudit": {
                "path": str(nested_root / DEFAULT_NESTED_REAUDIT_NAME),
                "schema_version": audit["schema_version"],
                "passed": True,
                "evidence_eligible": True,
                "submission_created": False,
            },
            "target_columns_used_for_strata": [],
            "prediction_columns_used_for_strata": [],
            "submission_created": False,
        }
        _write_json(temporary / SUMMARY_FILENAME, summary)
        if (
            _read_json(
                temporary / SUMMARY_FILENAME,
                label="nested stress summary",
            )
            != summary
            or sha256_file(metrics_path) != metrics_sha
        ):
            raise NestedStressError("nested stress temporary artifacts differ")
        if _submission_csvs(nested_root):
            raise NestedStressError("submission CSV appeared during stress analysis")
        if output_root.exists():
            raise FileExistsError(
                f"nested stress output appeared during analysis: {output_root}"
            )
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "FEATURE_FILENAME",
    "METRICS_FILENAME",
    "NESTED_STRESS_SCHEMA_VERSION",
    "NestedStressError",
    "PREDICTION_COLUMNS",
    "STRATA_FILENAME",
    "SUMMARY_FILENAME",
    "analyze_nested_target_free_stress",
]
