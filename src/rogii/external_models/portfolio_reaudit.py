"""Independent disk-only re-audit of canonical external-model portfolios.

The canonical artifact bundle is the only prediction input accepted here.
The re-audit deliberately does not reuse a runner's in-memory frames:

1. :func:`independent_reaudit_external_artifacts` verifies the manifest,
   checkpoints, fold provenance, current data manifest, and absence of a
   submission.
2. :func:`read_external_artifact_bundle` reads the canonical OOF from disk
   again.
3. Portfolio metrics are recomputed for the incumbent and every published
   candidate.
4. Same-name family-mean and shrink candidates in ``summary.json`` are
   compared under explicit numerical tolerances.

Only the small JSON report is written, by atomic replacement in the run root.
No competition-data file is opened for writing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..portfolio_metrics import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_TRUTH_COLUMN,
    common_slope_shape_metrics,
    paired_well_cluster_bootstrap,
    per_fold_metrics,
    per_well_win_summary,
    pooled_metrics,
    shadow_private_subset_simulation,
    tail_metrics,
)
from .artifacts import (
    DEFAULT_AUDITED_INCUMBENT_OOF,
    DEFAULT_BUNDLE_NAME,
    INCUMBENT_COLUMN,
    ROW_COORDINATE_COLUMN,
    OofCandidateSchema,
    capture_data_manifest,
    independent_reaudit_external_artifacts,
    load_audited_incumbent_reference,
    read_external_artifact_bundle,
    sha256_file,
)


REAUDIT_SCHEMA_VERSION = "rogii_external_portfolio_independent_reaudit_v2"
DEFAULT_OUTPUT_NAME = "independent_portfolio_reaudit.json"
DEFAULT_RUNNER_SUMMARY_NAME = "summary.json"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_SHADOW_SAMPLES = 100_000
SHADOW_SUBSET_WELLS = 200
SUMMARY_ABSOLUTE_TOLERANCE = 5.0e-9
SUMMARY_RELATIVE_TOLERANCE = 5.0e-11
_CORRELATION_BLOCK_ROWS = 100_000


def _safe_run_file(root: Path, name: str, *, label: str) -> Path:
    value = Path(str(name))
    if value.name != str(name) or value.is_absolute() or str(value) in {"", "."}:
        raise ValueError(f"{label} must be one filename inside the run root")
    path = (root / value).resolve()
    if path.parent != root:
        raise ValueError(f"{label} escapes the run root")
    return path


def _read_finite_json(path: Path, *, label: str) -> tuple[dict[str, Any], str, int]:
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
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid finite JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, digest, len(raw)


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metric output contains a non-finite float")
        return value
    raise TypeError(f"metric output contains unsupported scalar {type(value)!r}")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = tuple(map(str, frame.columns))
    return [
        {
            column: _json_scalar(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in frame.itertuples(index=False, name=None)
    ]


def _correlation_from_sufficient_statistics(
    count: int,
    sums: np.ndarray,
    cross: np.ndarray,
    columns: Sequence[str],
) -> pd.DataFrame:
    if count < 1:
        raise ValueError("residual correlation requires at least one row")
    centered_gram = cross - np.outer(sums, sums) / count
    diagonal = np.maximum(np.diag(centered_gram), 0.0)
    scale = np.sqrt(diagonal)
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        centered_gram,
        denominator,
        out=np.zeros_like(centered_gram),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    if not np.isfinite(correlation).all():
        raise ValueError("residual correlation is non-finite")
    names = tuple(map(str, columns))
    return pd.DataFrame(correlation, index=names, columns=names)


def _row_residual_correlation(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    block_rows: int = _CORRELATION_BLOCK_ROWS,
) -> pd.DataFrame:
    """Compute a row-residual correlation matrix without a full N x K copy."""

    columns = tuple(map(str, prediction_columns))
    if not columns or block_rows <= 0:
        raise ValueError("correlation columns and block_rows must be positive")
    missing = {truth_column, *columns} - set(frame.columns)
    if missing:
        raise ValueError(f"correlation input is missing columns: {sorted(missing)}")
    sums = np.zeros(len(columns), dtype=np.float64)
    cross = np.zeros((len(columns), len(columns)), dtype=np.float64)
    count = 0
    for start in range(0, len(frame), block_rows):
        stop = min(start + block_rows, len(frame))
        truth = pd.to_numeric(
            frame[truth_column].iloc[start:stop], errors="raise"
        ).to_numpy(dtype=np.float64)
        prediction = (
            frame.iloc[start:stop].loc[:, list(columns)].to_numpy(dtype=np.float64)
        )
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
            raise ValueError("correlation input contains non-finite values")
        residual = prediction - truth[:, None]
        sums += np.sum(residual, axis=0)
        cross += residual.T @ residual
        count += len(residual)
    return _correlation_from_sufficient_statistics(count, sums, cross, columns)


def _well_mean_residual_correlation(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
) -> pd.DataFrame:
    columns = tuple(map(str, prediction_columns))
    truth = pd.to_numeric(frame[truth_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    residual = pd.DataFrame(
        {
            column: pd.to_numeric(frame[column], errors="raise").to_numpy(
                dtype=np.float64
            )
            - truth
            for column in columns
        }
    )
    residual.insert(0, "well_id", frame["well_id"].astype(str).to_numpy())
    means = (
        residual.groupby("well_id", sort=True)[list(columns)]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(means).all():
        raise ValueError("well-mean residual correlation input is non-finite")
    return _correlation_from_sufficient_statistics(
        len(means),
        np.sum(means, axis=0),
        means.T @ means,
        columns,
    )


def _matrix_dict(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(column): {
            str(index): float(frame.loc[index, column]) for index in frame.index
        }
        for column in frame.columns
    }


def _component_field_metrics(
    frame: pd.DataFrame,
    candidates: Sequence[OofCandidateSchema],
) -> list[dict[str, Any]]:
    incumbent = frame[INCUMBENT_COLUMN].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        common = frame[candidate.common_column].to_numpy(dtype=np.float64)
        slope = frame[candidate.slope_column].to_numpy(dtype=np.float64)
        shape = frame[candidate.shape_column].to_numpy(dtype=np.float64)
        correction = frame[candidate.correction_column].to_numpy(dtype=np.float64)
        prediction = frame[candidate.prediction_column].to_numpy(dtype=np.float64)
        algebra_error = np.max(np.abs(correction - (common + slope + shape)))
        prediction_error = np.max(np.abs(prediction - (incumbent + correction)))
        rows.append(
            {
                "candidate": candidate.prediction_column,
                "candidate_id": candidate.candidate_id,
                "common_rms": float(np.sqrt(np.mean(np.square(common)))),
                "slope_rms": float(np.sqrt(np.mean(np.square(slope)))),
                "shape_rms": float(np.sqrt(np.mean(np.square(shape)))),
                "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
                "component_algebra_max_abs_error": float(algebra_error),
                "prediction_algebra_max_abs_error": float(prediction_error),
            }
        )
    return rows


def _metric_frames(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
) -> dict[str, pd.DataFrame]:
    return {
        "pooled": pooled_metrics(
            frame,
            prediction_columns,
            baseline_column=INCUMBENT_COLUMN,
        ),
        "per_fold": per_fold_metrics(
            frame,
            prediction_columns,
            baseline_column=INCUMBENT_COLUMN,
        ),
        "tail": tail_metrics(frame, prediction_columns),
        # The runner used row_index.  Retain it for exact historical comparison.
        "decomposition_row_index": common_slope_shape_metrics(
            frame,
            prediction_columns,
            position_column="row_index",
        ),
        # The canonical evidence coordinate follows inference-safe physical MD.
        "decomposition_row_coordinate": common_slope_shape_metrics(
            frame,
            prediction_columns,
            position_column=ROW_COORDINATE_COLUMN,
        ),
    }


_TABLE_COMPARISONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "pooled": (
        "pooled",
        ("n_rows", "n_wells", "sse", "pooled_row_rmse", "delta_rmse_vs_baseline"),
    ),
    "per_fold": (
        "per_fold",
        (
            "n_rows",
            "n_wells",
            "sse",
            "pooled_row_rmse",
            "baseline_rmse",
            "delta_rmse_vs_baseline",
        ),
    ),
    "tail": (
        "tail",
        (
            "n_wells",
            "p50_rmse",
            "p90_rmse",
            "p95_rmse",
            "max_rmse",
            "wells_rmse_ge_40",
            "wells_rmse_ge_60",
            "worst_10pct_sse_share",
        ),
    ),
    "decomposition": (
        "decomposition_row_index",
        (
            "n_rows",
            "n_wells",
            "total_rmse",
            "common_mode_rmse",
            "common_mode_bias",
            "centered_rmse",
            "slope_component_rmse",
            "zero_mean_shape_rmse",
            "terminal_rmse",
            "well_slope_coefficient_rmse",
            "relative_mse_reconstruction_error",
        ),
    ),
    "per_well_wins": (
        "per_well_wins",
        (
            "wells",
            "wins",
            "ties",
            "losses",
            "win_fraction",
            "nonloss_fraction",
            "mean_well_rmse_delta",
            "median_well_rmse_delta",
        ),
    ),
}
_EXACT_INTEGER_FIELDS = frozenset(
    {
        "fold",
        "n_rows",
        "n_wells",
        "wells",
        "wins",
        "ties",
        "losses",
        "wells_rmse_ge_40",
        "wells_rmse_ge_60",
        "samples",
        "seed",
        "subset_wells",
        "available_wells",
    }
)


def _key_token(values: Sequence[Any]) -> str:
    return "|".join(map(str, values))


def _numeric_difference(
    actual: Any,
    expected: Any,
    *,
    field: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[float, float, bool]:
    if field in _EXACT_INTEGER_FIELDS:
        if isinstance(actual, bool) or isinstance(expected, bool):
            return 0.0, 0.0, actual == expected
        left = int(actual)
        right = int(expected)
        return float(abs(left - right)), 0.0, left == right
    left = float(actual)
    right = float(expected)
    if not math.isfinite(left) or not math.isfinite(right):
        return math.inf, math.inf, False
    difference = abs(left - right)
    scale = max(abs(left), abs(right))
    relative = difference / scale if scale > 0.0 else 0.0
    passed = difference <= absolute_tolerance + relative_tolerance * scale
    return difference, relative, passed


def _compare_table_section(
    runner_metrics: Mapping[str, Any],
    *,
    runner_section: str,
    local_records: Sequence[Mapping[str, Any]],
    prediction_columns: Sequence[str],
    fields: Sequence[str],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = runner_metrics.get(runner_section)
    if raw is None:
        return (
            {
                "status": "not_present",
                "passed": True,
                "matched_rows": 0,
                "fields_compared": [],
            },
            [],
        )
    if not isinstance(raw, list):
        raise ValueError(f"runner summary metrics.{runner_section} must be a list")
    allowed = set(map(str, prediction_columns))
    key_fields = (
        ("candidate", "fold") if runner_section == "per_fold" else ("candidate",)
    )

    def index_rows(
        rows: Sequence[Mapping[str, Any]],
        *,
        label: str,
    ) -> dict[tuple[Any, ...], Mapping[str, Any]]:
        result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} rows must be mappings")
            if str(row.get("candidate", "")) not in allowed:
                continue
            key = tuple(row.get(field) for field in key_fields)
            if any(value is None for value in key):
                raise ValueError(f"{label} row lacks key fields {key_fields}")
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key}")
            result[key] = row
        return result

    local = index_rows(local_records, label=f"local {runner_section}")
    runner = index_rows(raw, label=f"runner {runner_section}")
    common = sorted(set(local) & set(runner), key=lambda key: tuple(map(str, key)))
    comparisons = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    failures: list[dict[str, Any]] = []
    compared_fields: set[str] = set()
    missing_fields: set[str] = set()
    for key in common:
        left = local[key]
        right = runner[key]
        for field in fields:
            if field not in left or field not in right:
                missing_fields.add(field)
                continue
            difference, relative, passed = _numeric_difference(
                left[field],
                right[field],
                field=field,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            comparisons += 1
            compared_fields.add(field)
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, relative)
            if not passed and len(failures) < 20:
                failures.append(
                    {
                        "key": _key_token(key),
                        "field": field,
                        "recomputed": _json_scalar(left[field]),
                        "runner": _json_scalar(right[field]),
                        "absolute_difference": difference,
                        "relative_difference": relative,
                    }
                )
    result = {
        "status": "compared" if common else "no_same_name_rows",
        "passed": not failures,
        "local_rows": len(local),
        "runner_same_name_rows": len(runner),
        "matched_rows": len(common),
        "values_compared": comparisons,
        "fields_compared": sorted(compared_fields),
        "fields_not_present_on_both_sides": sorted(missing_fields),
        "missing_runner_keys": sorted(
            _key_token(key) for key in set(local) - set(runner)
        ),
        "runner_only_keys": sorted(_key_token(key) for key in set(runner) - set(local)),
        "max_absolute_difference": maximum_absolute,
        "max_relative_difference": maximum_relative,
    }
    return result, failures


def _compare_bootstrap(
    runner_metrics: Mapping[str, Any],
    local: Mapping[str, Any],
    *,
    prediction_columns: Sequence[str],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = runner_metrics.get("paired_bootstrap")
    if raw is None:
        return {"status": "not_present", "passed": True, "values_compared": 0}, []
    if not isinstance(raw, Mapping):
        raise ValueError("runner summary paired_bootstrap must be a mapping")
    parameter_fields = ("baseline", "samples", "seed", "wells")
    parameter_mismatch = {
        field: {"recomputed": local.get(field), "runner": raw.get(field)}
        for field in parameter_fields
        if local.get(field) != raw.get(field)
    }
    if parameter_mismatch:
        return (
            {
                "status": "parameter_mismatch",
                "passed": True,
                "values_compared": 0,
                "parameter_mismatch": parameter_mismatch,
            },
            [],
        )
    local_candidates = local.get("candidates")
    runner_candidates = raw.get("candidates")
    if not isinstance(local_candidates, Mapping) or not isinstance(
        runner_candidates, Mapping
    ):
        raise ValueError("bootstrap candidate blocks must be mappings")
    allowed = set(map(str, prediction_columns)) - {INCUMBENT_COLUMN}
    fields = (
        "point_delta_rmse",
        "delta_rmse_q2.5",
        "delta_rmse_q50",
        "delta_rmse_q97.5",
    )
    failures: list[dict[str, Any]] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    comparisons = 0
    common = sorted(
        allowed & set(map(str, local_candidates)) & set(map(str, runner_candidates))
    )
    for candidate in common:
        left = local_candidates[candidate]
        right = runner_candidates[candidate]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError("bootstrap candidate metric must be a mapping")
        for field in fields:
            if field not in left or field not in right:
                continue
            difference, relative, passed = _numeric_difference(
                left[field],
                right[field],
                field=field,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            comparisons += 1
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, relative)
            if not passed and len(failures) < 20:
                failures.append(
                    {
                        "candidate": candidate,
                        "field": field,
                        "recomputed": float(left[field]),
                        "runner": float(right[field]),
                        "absolute_difference": difference,
                        "relative_difference": relative,
                    }
                )
    return (
        {
            "status": "compared",
            "passed": not failures,
            "matched_candidates": len(common),
            "values_compared": comparisons,
            "missing_runner_candidates": sorted(allowed - set(runner_candidates)),
            "max_absolute_difference": maximum_absolute,
            "max_relative_difference": maximum_relative,
        },
        failures,
    )


def _compare_correlation(
    runner_metrics: Mapping[str, Any],
    local: Mapping[str, Mapping[str, float]],
    *,
    prediction_columns: Sequence[str],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = runner_metrics.get("residual_correlation")
    if raw is None:
        return {"status": "not_present", "passed": True, "values_compared": 0}, []
    if not isinstance(raw, Mapping):
        raise ValueError("runner summary residual_correlation must be a mapping")
    columns = tuple(map(str, prediction_columns))
    failures: list[dict[str, Any]] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    comparisons = 0
    missing: list[str] = []
    for column in columns:
        left_column = local.get(column)
        right_column = raw.get(column)
        if not isinstance(left_column, Mapping) or not isinstance(
            right_column, Mapping
        ):
            missing.append(column)
            continue
        for index in columns:
            if index not in left_column or index not in right_column:
                missing.append(f"{column}/{index}")
                continue
            difference, relative, passed = _numeric_difference(
                left_column[index],
                right_column[index],
                field="correlation",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            comparisons += 1
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, relative)
            if not passed and len(failures) < 20:
                failures.append(
                    {
                        "column": column,
                        "index": index,
                        "recomputed": float(left_column[index]),
                        "runner": float(right_column[index]),
                        "absolute_difference": difference,
                        "relative_difference": relative,
                    }
                )
    return (
        {
            "status": "compared",
            "passed": not failures,
            "values_compared": comparisons,
            "missing_entries": sorted(set(missing)),
            "max_absolute_difference": maximum_absolute,
            "max_relative_difference": maximum_relative,
        },
        failures,
    )


def _compare_shadow_private(
    runner_metrics: Mapping[str, Any],
    local: Mapping[str, Any],
    *,
    prediction_columns: Sequence[str],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = runner_metrics.get("shadow_private")
    if raw is None:
        return {"status": "not_present", "passed": True, "values_compared": 0}, []
    if not isinstance(raw, Mapping):
        raise ValueError("runner summary shadow_private must be a mapping")
    local_performed = local.get("performed") is True
    runner_performed = raw.get("performed") is not False and isinstance(
        raw.get("candidates"), Mapping
    )
    if not local_performed or not runner_performed:
        same = local_performed == runner_performed
        return (
            {
                "status": "both_skipped" if same else "performed_status_mismatch",
                "passed": same,
                "values_compared": 0,
                "recomputed_performed": local_performed,
                "runner_performed": runner_performed,
                "recomputed_reason": local.get("reason"),
                "runner_reason": raw.get("reason"),
            },
            []
            if same
            else [
                {
                    "field": "performed",
                    "recomputed": local_performed,
                    "runner": runner_performed,
                }
            ],
        )
    parameter_fields = (
        "baseline",
        "samples",
        "subset_wells",
        "seed",
        "available_wells",
    )
    mismatch = {
        field: {"recomputed": local.get(field), "runner": raw.get(field)}
        for field in parameter_fields
        if local.get(field) != raw.get(field)
    }
    if mismatch:
        return (
            {
                "status": "parameter_mismatch",
                "passed": True,
                "values_compared": 0,
                "parameter_mismatch": mismatch,
            },
            [],
        )
    allowed = set(map(str, prediction_columns)) - {INCUMBENT_COLUMN}
    local_candidates = local["candidates"]
    runner_candidates = raw["candidates"]
    fields = (
        "mean_delta",
        "p_worse",
        "delta_q01",
        "delta_q05",
        "delta_q50",
        "delta_q95",
        "delta_q99",
    )
    failures: list[dict[str, Any]] = []
    comparisons = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    common = sorted(allowed & set(runner_candidates) & set(local_candidates))
    for candidate in common:
        for field in fields:
            if (
                field not in local_candidates[candidate]
                or field not in runner_candidates[candidate]
            ):
                continue
            difference, relative, passed = _numeric_difference(
                local_candidates[candidate][field],
                runner_candidates[candidate][field],
                field=field,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            comparisons += 1
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, relative)
            if not passed and len(failures) < 20:
                failures.append(
                    {
                        "candidate": candidate,
                        "field": field,
                        "recomputed": float(local_candidates[candidate][field]),
                        "runner": float(runner_candidates[candidate][field]),
                        "absolute_difference": difference,
                        "relative_difference": relative,
                    }
                )
    return (
        {
            "status": "compared",
            "passed": not failures,
            "matched_candidates": len(common),
            "values_compared": comparisons,
            "missing_runner_candidates": sorted(allowed - set(runner_candidates)),
            "max_absolute_difference": maximum_absolute,
            "max_relative_difference": maximum_relative,
        },
        failures,
    )


def _compare_runner_summary(
    summary: Mapping[str, Any] | None,
    *,
    manifest: Mapping[str, Any],
    metric_frames: Mapping[str, pd.DataFrame],
    wins: pd.DataFrame,
    bootstrap: Mapping[str, Any],
    row_correlation: Mapping[str, Mapping[str, float]],
    shadow_private: Mapping[str, Any],
    prediction_columns: Sequence[str],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    tolerance = {
        "rule": (
            "abs(recomputed-runner) <= absolute_tolerance + "
            "relative_tolerance * max(abs(recomputed), abs(runner)); "
            "integer counts are exact"
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }
    if summary is None:
        return {
            "performed": False,
            "passed": True,
            "reason": "runner summary is absent",
            "tolerance": tolerance,
            "sections": {},
            "failures": [],
        }
    if str(summary.get("run_id")) != str(manifest["run_id"]):
        raise ValueError("runner summary run_id differs from canonical manifest")
    if str(summary.get("profile")) != str(manifest["profile"]):
        raise ValueError("runner summary profile differs from canonical manifest")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("runner summary metrics must be a mapping")

    sections: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    local_records = {name: _records(frame) for name, frame in metric_frames.items()}
    local_records["per_well_wins"] = _records(wins)
    for runner_name, (local_name, fields) in _TABLE_COMPARISONS.items():
        result, current_failures = _compare_table_section(
            metrics,
            runner_section=runner_name,
            local_records=local_records[local_name],
            prediction_columns=prediction_columns,
            fields=fields,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        sections[runner_name] = result
        failures.extend(
            {"section": runner_name, **failure} for failure in current_failures
        )

    bootstrap_result, bootstrap_failures = _compare_bootstrap(
        metrics,
        bootstrap,
        prediction_columns=prediction_columns,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    sections["paired_bootstrap"] = bootstrap_result
    failures.extend(
        {"section": "paired_bootstrap", **failure} for failure in bootstrap_failures
    )
    correlation_result, correlation_failures = _compare_correlation(
        metrics,
        row_correlation,
        prediction_columns=prediction_columns,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    sections["residual_correlation"] = correlation_result
    failures.extend(
        {"section": "residual_correlation", **failure}
        for failure in correlation_failures
    )
    shadow_result, shadow_failures = _compare_shadow_private(
        metrics,
        shadow_private,
        prediction_columns=prediction_columns,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    sections["shadow_private"] = shadow_result
    failures.extend(
        {"section": "shadow_private", **failure} for failure in shadow_failures
    )
    return {
        "performed": True,
        "passed": not failures,
        "same_name_prediction_columns": list(map(str, prediction_columns)),
        "tolerance": tolerance,
        "sections": sections,
        "failures": failures[:50],
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        encoded = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("re-audit report is not finite JSON") from exc
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name).resolve()
    if temporary.parent != path.parent.resolve():
        os.close(descriptor)
        raise ValueError("atomic report temporary escaped the run root")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise


def audit_external_model_portfolio(
    run_root: str | Path,
    *,
    incumbent_oof: str | Path = DEFAULT_AUDITED_INCUMBENT_OOF,
    bundle_name: str = DEFAULT_BUNDLE_NAME,
    runner_summary_name: str = DEFAULT_RUNNER_SUMMARY_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    shadow_samples: int = DEFAULT_SHADOW_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    summary_absolute_tolerance: float = SUMMARY_ABSOLUTE_TOLERANCE,
    summary_relative_tolerance: float = SUMMARY_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Re-audit one completed canonical bundle and atomically write its report."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    if (
        isinstance(bootstrap_samples, bool)
        or isinstance(shadow_samples, bool)
        or isinstance(seed, bool)
        or int(bootstrap_samples) != bootstrap_samples
        or int(shadow_samples) != shadow_samples
        or int(seed) != seed
        or bootstrap_samples <= 0
        or shadow_samples <= 0
        or seed < 0
    ):
        raise ValueError(
            "sample counts must be positive integers and seed non-negative"
        )
    if (
        not math.isfinite(summary_absolute_tolerance)
        or not math.isfinite(summary_relative_tolerance)
        or summary_absolute_tolerance < 0.0
        or summary_relative_tolerance < 0.0
    ):
        raise ValueError("summary comparison tolerances must be finite and nonnegative")
    output_path = _safe_run_file(root, output_name, label="output_name")
    summary_path = _safe_run_file(
        root, runner_summary_name, label="runner_summary_name"
    )

    bundle_component = Path(str(bundle_name))
    if (
        bundle_component.name != str(bundle_name)
        or bundle_component.is_absolute()
        or str(bundle_component) in {"", "."}
    ):
        raise ValueError("bundle_name must be one directory inside the run root")
    manifest_path = root / bundle_component / "manifest.json"
    discovery_manifest, _, _ = _read_finite_json(
        manifest_path,
        label="canonical artifact manifest",
    )
    discovery_metadata = discovery_manifest.get("metadata")
    if not isinstance(discovery_metadata, Mapping):
        raise ValueError("canonical manifest metadata must be a mapping")
    declared_reference = discovery_metadata.get(
        "locked_incumbent_reference",
        {},
    )
    expected_incumbent_sha = (
        declared_reference.get("sha256")
        if isinstance(declared_reference, Mapping)
        else None
    )
    audited_incumbent, audited_incumbent_record = load_audited_incumbent_reference(
        incumbent_oof,
        expected_sha256=expected_incumbent_sha,
    )
    fold_records = discovery_manifest.get("fold_provenance")
    if not isinstance(fold_records, list) or not fold_records:
        raise ValueError("canonical manifest lacks fold provenance")
    validation_ids = tuple(
        str(well_id)
        for record in fold_records
        if isinstance(record, Mapping)
        for well_id in record.get("validation_well_ids", ())
    )
    if (
        not validation_ids
        or len(set(validation_ids)) != len(validation_ids)
        or any(not well_id for well_id in validation_ids)
    ):
        raise ValueError("canonical manifest validation-well provenance is invalid")
    locked_reference = audited_incumbent.loc[
        audited_incumbent["well_id"].astype(str).isin(validation_ids)
    ].copy()
    if set(locked_reference["well_id"].astype(str)) != set(validation_ids):
        raise ValueError(
            "canonical validation wells are not covered by audited incumbent OOF"
        )

    # Required first evidence boundary: validate every source byte, the current
    # data, and exact truth/incumbent values against the external lock.
    try:
        artifact_audit = independent_reaudit_external_artifacts(
            root,
            bundle_name=bundle_name,
            expected_reference=locked_reference,
        )
    except ValueError as exc:
        raise ValueError(
            "canonical artifact validation against the audited incumbent "
            f"reference failed: {exc}"
        ) from exc
    if artifact_audit.get("passed") is not True:
        raise ValueError("canonical external artifact independent audit did not pass")
    gc.collect()

    # Required second disk read: metrics never use a runner-owned in-memory frame.
    bundle = read_external_artifact_bundle(
        root,
        bundle_name=bundle_name,
        expected_reference=locked_reference,
    )
    if bundle.audit.get("passed") is not True:
        raise ValueError("canonical bundle disk reread did not pass")
    if bundle.audit.get("manifest_sha256") != artifact_audit.get("manifest_sha256"):
        raise ValueError("canonical manifest changed between independent disk reads")
    if bundle.audit["oof"].get("key_sha256") != artifact_audit["oof"].get("key_sha256"):
        raise ValueError("canonical OOF keys changed between independent disk reads")

    candidates = tuple(
        OofCandidateSchema.from_dict(value)
        for value in bundle.manifest["oof"]["candidates"]
    )
    candidate_columns = tuple(candidate.prediction_column for candidate in candidates)
    prediction_columns = (INCUMBENT_COLUMN, *candidate_columns)
    frame = bundle.oof
    metric_frames = _metric_frames(frame, prediction_columns)
    bootstrap = paired_well_cluster_bootstrap(
        frame,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=candidate_columns,
        samples=int(bootstrap_samples),
        seed=int(seed),
    )
    wins = per_well_win_summary(
        frame,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=candidate_columns,
    )
    row_correlation_frame = _row_residual_correlation(frame, prediction_columns)
    well_correlation_frame = _well_mean_residual_correlation(frame, prediction_columns)
    row_correlation = _matrix_dict(row_correlation_frame)
    well_correlation = _matrix_dict(well_correlation_frame)
    available_wells = int(frame["well_id"].nunique())
    if available_wells >= SHADOW_SUBSET_WELLS:
        shadow_private = {
            "performed": True,
            **shadow_private_subset_simulation(
                frame,
                baseline_column=INCUMBENT_COLUMN,
                candidate_columns=candidate_columns,
                samples=int(shadow_samples),
                subset_wells=SHADOW_SUBSET_WELLS,
                seed=int(seed),
            ),
        }
    else:
        shadow_private = {
            "performed": False,
            "reason": (
                f"subset_wells={SHADOW_SUBSET_WELLS} exceeds "
                f"available_wells={available_wells}"
            ),
            "available_wells": available_wells,
            "subset_wells": SHADOW_SUBSET_WELLS,
            "samples_requested": int(shadow_samples),
            "seed": int(seed),
        }

    summary: dict[str, Any] | None
    summary_record: dict[str, Any]
    if summary_path.is_file():
        summary, summary_sha256, summary_bytes = _read_finite_json(
            summary_path, label="runner summary"
        )
        summary_record = {
            "present": True,
            "path": str(summary_path),
            "sha256": summary_sha256,
            "bytes": summary_bytes,
        }
    else:
        summary = None
        summary_record = {
            "present": False,
            "path": str(summary_path),
        }
    comparison = _compare_runner_summary(
        summary,
        manifest=bundle.manifest,
        metric_frames=metric_frames,
        wins=wins,
        bootstrap=bootstrap,
        row_correlation=row_correlation,
        shadow_private=shadow_private,
        prediction_columns=prediction_columns,
        absolute_tolerance=float(summary_absolute_tolerance),
        relative_tolerance=float(summary_relative_tolerance),
    )
    if not comparison["passed"]:
        first = comparison["failures"][0]
        raise ValueError(
            "runner summary comparison failed; "
            f"first_mismatch={json.dumps(first, sort_keys=True)}"
        )
    if summary_path.is_file() and sha256_file(summary_path) != summary_record["sha256"]:
        raise ValueError("runner summary changed during independent re-audit")

    data_root = Path(str(bundle.manifest["data_root"])).resolve()
    ending_data_manifest = capture_data_manifest(data_root)
    starting_data_manifest = artifact_audit["data_read_only_manifest_current"]
    if ending_data_manifest != starting_data_manifest:
        raise ValueError("competition data manifest changed during portfolio re-audit")

    report: dict[str, Any] = {
        "schema_version": REAUDIT_SCHEMA_VERSION,
        "completed": True,
        "passed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": bundle.manifest["run_id"],
        "profile": bundle.manifest["profile"],
        "scope": {
            "rows": int(len(frame)),
            "wells": available_wells,
            "folds": sorted(map(int, frame["fold"].unique())),
            "candidate_count": len(candidates),
            "eligible_candidate_count": sum(
                candidate.eligible_for_model_selection for candidate in candidates
            ),
            "ineligible_candidate_count": sum(
                not candidate.eligible_for_model_selection for candidate in candidates
            ),
        },
        "parameters": {
            "bootstrap_samples": int(bootstrap_samples),
            "shadow_samples": int(shadow_samples),
            "shadow_subset_wells": SHADOW_SUBSET_WELLS,
            "seed": int(seed),
            "formal_default_requirements_used": (
                int(bootstrap_samples) == DEFAULT_BOOTSTRAP_SAMPLES
                and int(shadow_samples) == DEFAULT_SHADOW_SAMPLES
            ),
        },
        "sources": {
            "bundle_root": str(bundle.bundle_root),
            "manifest_sha256": artifact_audit["manifest_sha256"],
            "oof_sha256": bundle.manifest["oof"]["sha256"],
            "oof_bytes": bundle.manifest["oof"]["bytes"],
            "audited_incumbent_reference": audited_incumbent_record,
            "runner_summary": summary_record,
        },
        "artifact_independent_reaudit": artifact_audit,
        "second_disk_read_audit": bundle.audit,
        "data_manifest_unchanged_during_reaudit": True,
        "candidate_registry": [candidate.as_dict() for candidate in candidates],
        "metrics": {
            "pooled": _records(metric_frames["pooled"]),
            "per_fold": _records(metric_frames["per_fold"]),
            "tail": _records(metric_frames["tail"]),
            "decomposition_row_index": _records(
                metric_frames["decomposition_row_index"]
            ),
            "decomposition_row_coordinate": _records(
                metric_frames["decomposition_row_coordinate"]
            ),
            "published_component_fields": _component_field_metrics(frame, candidates),
            "paired_bootstrap": bootstrap,
            "per_well_wins": _records(wins),
            "residual_correlation": {
                "row": row_correlation,
                "well_mean": well_correlation,
            },
            "shadow_private": shadow_private,
        },
        "runner_summary_comparison": comparison,
        "submission_created": False,
    }
    _atomic_json(output_path, report)
    return report


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_OUTPUT_NAME",
    "DEFAULT_RUNNER_SUMMARY_NAME",
    "DEFAULT_SHADOW_SAMPLES",
    "REAUDIT_SCHEMA_VERSION",
    "SHADOW_SUBSET_WELLS",
    "SUMMARY_ABSOLUTE_TOLERANCE",
    "SUMMARY_RELATIVE_TOLERANCE",
    "audit_external_model_portfolio",
]
