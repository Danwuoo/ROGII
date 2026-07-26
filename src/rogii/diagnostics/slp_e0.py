"""SLP-E0 target-bearing representation-oracle diagnostics.

This module deliberately reads training targets and formation surfaces. It is
non-deployable, submission-ineligible, and physically separate from every
production model and inference loader.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..data import WellFiles, enumerate_wells, validate_prefix


SLP_E0_AUDIT_ID = "SLP-E0"
SLP_E0_USES_TARGET = True
SLP_E0_DEPLOYABLE = False
SLP_E0_SUBMISSION_ELIGIBLE = False
SLP_E0_POLYNOMIAL_DEGREES = tuple(range(7))
SLP_E0_SEGMENT_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
SLP_E0_FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
SLP_E0_DIAGNOSTIC_COLUMNS = (
    "MD",
    "Z",
    "GR",
    "TVT_input",
    "TVT",
    *SLP_E0_FORMATION_COLUMNS,
)
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RECURRENCE_THRESHOLDS = (0.25, 0.50, 1.00, 2.00, 4.00)
_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


class UnsupportedOracleComplexity(ValueError):
    """A diagnostic basis is unsupported by the available rows."""


@dataclass(frozen=True)
class OracleFit:
    """One deterministic target-bearing oracle fit."""

    predictions: np.ndarray
    rows: int
    sse: float
    rmse: float
    segment_row_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class PooledOracleMetric:
    """Stable pooled row metric accumulated across wells."""

    wells: int
    rows: int
    sse: float
    pooled_rmse: float


@dataclass(frozen=True)
class SurfaceIdentityWellStats:
    """Within-well structural-identity diagnostic statistics."""

    finite_rows: int
    mean_constant: float
    centered_sse: float
    population_std: float


@dataclass(frozen=True)
class _WellAuditResult:
    well_id: str
    prediction_rows: int
    oracle_rows: tuple[dict[str, Any], ...]
    recurrence: dict[str, Any]
    surfaces: tuple[dict[str, Any], ...]


def _as_float64_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite float64 values")
    return array


def _validated_oracle_vectors(
    normalized_md: Iterable[float], target: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    t = _as_float64_vector(normalized_md, name="normalized_md")
    y = _as_float64_vector(target, name="oracle target")
    if t.size != y.size:
        raise ValueError("normalized_md and oracle target lengths differ")
    if t.size < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("normalized_md must be finite and strictly increasing")
    if t[0] < 0.0 or t[-1] > 1.0 or not t[-1] > t[0]:
        raise ValueError("normalized_md must have positive span inside [0, 1]")
    return t, y


def _fit_from_design(
    design: np.ndarray,
    target: np.ndarray,
    *,
    minimum_rank: int,
    segment_row_counts: tuple[int, ...] = (),
) -> OracleFit:
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if int(rank) < minimum_rank:
        raise UnsupportedOracleComplexity(
            f"Oracle design is rank deficient: rank={rank}, required={minimum_rank}"
        )
    prediction = np.asarray(design @ coefficients, dtype=np.float64)
    residual = np.asarray(target - prediction, dtype=np.float64)
    sse = float(residual @ residual)
    if not np.isfinite(prediction).all() or not math.isfinite(sse):
        raise ValueError("Oracle fit produced non-finite values")
    return OracleFit(
        predictions=prediction,
        rows=int(target.size),
        sse=sse,
        rmse=math.sqrt(sse / target.size),
        segment_row_counts=segment_row_counts,
    )


def nearest_prefix_distance(
    prefix_tvt: Iterable[float], future_tvt: Iterable[float]
) -> np.ndarray:
    """Return exact nearest prefix-TVT distance without a dense distance matrix."""
    prefix = _as_float64_vector(prefix_tvt, name="prefix_tvt")
    future = _as_float64_vector(future_tvt, name="future_tvt")
    support = np.sort(prefix, kind="stable")
    positions = np.searchsorted(support, future, side="left")
    left_indices = np.maximum(positions - 1, 0)
    right_indices = np.minimum(positions, support.size - 1)
    left_distance = np.abs(future - support[left_indices])
    right_distance = np.abs(future - support[right_indices])
    distance = np.minimum(left_distance, right_distance)
    if not np.isfinite(distance).all():
        raise ValueError("Nearest-prefix diagnostic produced non-finite distances")
    return np.asarray(distance, dtype=np.float64)


def fit_polynomial_oracle(
    normalized_md: Iterable[float], target: Iterable[float], *, degree: int
) -> OracleFit:
    """Fit one per-well float64 polynomial target oracle."""
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("Polynomial oracle degree must be a non-negative integer")
    t, y = _validated_oracle_vectors(normalized_md, target)
    columns = degree + 1
    if t.size < columns:
        raise UnsupportedOracleComplexity(
            f"Polynomial degree {degree} needs at least {columns} rows"
        )
    design = np.vander(t, N=columns, increasing=True)
    return _fit_from_design(design, y, minimum_rank=columns)


def _segment_assignments(normalized_md: np.ndarray, segments: int) -> np.ndarray:
    assignment = np.floor(normalized_md * segments).astype(np.int64)
    return np.minimum(assignment, segments - 1)


def _segment_row_counts(normalized_md: np.ndarray, segments: int) -> tuple[int, ...]:
    assignment = _segment_assignments(normalized_md, segments)
    return tuple(
        int(value)
        for value in np.bincount(assignment, minlength=segments).tolist()
    )


def fit_continuous_piecewise(
    normalized_md: Iterable[float], target: Iterable[float], *, segments: int
) -> OracleFit:
    """Fit a continuous equal-segment hinge-basis target oracle."""
    if not isinstance(segments, int) or segments < 1:
        raise ValueError("Continuous piecewise oracle segments must be positive")
    t, y = _validated_oracle_vectors(normalized_md, target)
    columns = segments + 1
    if t.size < columns:
        raise UnsupportedOracleComplexity(
            f"Continuous K={segments} needs at least {columns} rows"
        )
    if segments == 1:
        design = np.column_stack((np.ones(t.size, dtype=np.float64), t))
    else:
        knots = np.arange(1, segments, dtype=np.float64) / float(segments)
        hinges = np.maximum(t[:, None] - knots[None, :], 0.0)
        design = np.column_stack((np.ones(t.size, dtype=np.float64), t, hinges))
    return _fit_from_design(
        design,
        y,
        minimum_rank=columns,
        segment_row_counts=_segment_row_counts(t, segments),
    )


def fit_discontinuous_piecewise(
    normalized_md: Iterable[float], target: Iterable[float], *, segments: int
) -> OracleFit:
    """Fit independent equal-segment linear target oracles with optional jumps."""
    if not isinstance(segments, int) or segments < 1:
        raise ValueError("Discontinuous piecewise oracle segments must be positive")
    t, y = _validated_oracle_vectors(normalized_md, target)
    assignment = _segment_assignments(t, segments)
    counts = tuple(
        int(value)
        for value in np.bincount(assignment, minlength=segments).tolist()
    )
    short = [index for index, count in enumerate(counts) if count < 2]
    if short:
        raise UnsupportedOracleComplexity(
            f"Discontinuous K={segments} has segments with fewer than two rows: {short}"
        )
    prediction = np.empty_like(y)
    for segment in range(segments):
        mask = assignment == segment
        design = np.column_stack(
            (np.ones(int(mask.sum()), dtype=np.float64), t[mask])
        )
        fit = _fit_from_design(design, y[mask], minimum_rank=2)
        prediction[mask] = fit.predictions
    residual = np.asarray(y - prediction, dtype=np.float64)
    sse = float(residual @ residual)
    if not np.isfinite(prediction).all() or not math.isfinite(sse):
        raise ValueError("Discontinuous oracle fit produced non-finite values")
    return OracleFit(
        predictions=prediction,
        rows=int(y.size),
        sse=sse,
        rmse=math.sqrt(sse / y.size),
        segment_row_counts=counts,
    )


def accumulate_pooled_metric(
    metrics: Iterable[OracleFit | tuple[float, int]]
) -> PooledOracleMetric:
    """Accumulate well SSE values in stable order into pooled row-RMSE."""
    sse_values: list[float] = []
    rows = 0
    wells = 0
    for metric in metrics:
        if isinstance(metric, OracleFit):
            sse = metric.sse
            count = metric.rows
        else:
            sse, count = metric
        sse = float(sse)
        count = int(count)
        if not math.isfinite(sse) or sse < 0.0 or count <= 0:
            raise ValueError("Pooled oracle metrics require finite non-negative SSE and rows")
        sse_values.append(sse)
        rows += count
        wells += 1
    if wells == 0 or rows == 0:
        raise ValueError("Cannot accumulate an empty pooled oracle metric")
    total_sse = math.fsum(sse_values)
    return PooledOracleMetric(
        wells=wells,
        rows=rows,
        sse=total_sse,
        pooled_rmse=math.sqrt(total_sse / rows),
    )


def surface_identity_stats(
    tvt: Iterable[float], z: Iterable[float], surface: Iterable[float]
) -> SurfaceIdentityWellStats:
    """Compute one well/formation target-bearing structural identity diagnostic."""
    target = _as_float64_vector(tvt, name="surface diagnostic TVT")
    vertical = _as_float64_vector(z, name="surface diagnostic Z")
    formation = np.asarray(surface, dtype=np.float64)
    if formation.ndim != 1 or not (
        target.size == vertical.size == formation.size
    ):
        raise ValueError("Surface identity diagnostic columns must be aligned vectors")
    finite = np.isfinite(formation)
    if not finite.any():
        raise ValueError("Surface identity diagnostic has no finite formation rows")
    residual = target[finite] + vertical[finite] - formation[finite]
    mean_constant = float(np.mean(residual, dtype=np.float64))
    centered = np.asarray(residual - mean_constant, dtype=np.float64)
    centered_sse = float(centered @ centered)
    if not math.isfinite(mean_constant) or not math.isfinite(centered_sse):
        raise ValueError("Surface identity diagnostic produced non-finite statistics")
    count = int(residual.size)
    return SurfaceIdentityWellStats(
        finite_rows=count,
        mean_constant=mean_constant,
        centered_sse=centered_sse,
        population_std=math.sqrt(centered_sse / count),
    )


def _load_slp_e0_diagnostic_frame(path: Path) -> pd.DataFrame:
    """Read the target-bearing SLP-E0 training-only diagnostic schema.

    This loader must never be imported by production model or prediction code.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SLP-E0 horizontal training CSV is missing: {source}")
    header = pd.read_csv(source, nrows=0)
    missing = set(SLP_E0_DIAGNOSTIC_COLUMNS) - set(header.columns)
    if missing:
        raise ValueError(
            f"SLP-E0 diagnostic columns missing from {source}: {sorted(missing)}"
        )
    frame = pd.read_csv(
        source,
        usecols=list(SLP_E0_DIAGNOSTIC_COLUMNS),
        dtype={column: np.float64 for column in SLP_E0_DIAGNOSTIC_COLUMNS},
    )
    if frame.empty:
        raise ValueError(f"SLP-E0 horizontal training CSV is empty: {source}")
    return frame.loc[:, list(SLP_E0_DIAGNOSTIC_COLUMNS)]


def _oracle_row(
    *,
    well_id: str,
    representation: str,
    method: str,
    complexity_type: str,
    complexity: int,
    fit: OracleFit | None,
    unsupported_reason: str = "",
) -> dict[str, Any]:
    supported = fit is not None
    return {
        "well_id": well_id,
        "representation": representation,
        "method": method,
        "complexity_type": complexity_type,
        "complexity": complexity,
        "supported": supported,
        "valid_k": supported if complexity_type == "segments" else "",
        "rows": fit.rows if fit is not None else 0,
        "sse": fit.sse if fit is not None else None,
        "rmse": fit.rmse if fit is not None else None,
        "segment_row_counts": (
            json.dumps(fit.segment_row_counts, separators=(",", ":"))
            if fit is not None and fit.segment_row_counts
            else ""
        ),
        "unsupported_reason": unsupported_reason,
        "uses_target": SLP_E0_USES_TARGET,
        "deployable": SLP_E0_DEPLOYABLE,
        "submission_eligible": SLP_E0_SUBMISSION_ELIGIBLE,
    }


def _fit_or_unsupported(
    fit_function: Any,
    *,
    normalized_md: np.ndarray,
    target: np.ndarray,
    complexity_name: str,
    complexity: int,
) -> tuple[OracleFit | None, str]:
    try:
        return (
            fit_function(
                normalized_md,
                target,
                **{complexity_name: complexity},
            ),
            "",
        )
    except UnsupportedOracleComplexity as error:
        return None, str(error)


def _audit_one_well(well: WellFiles) -> _WellAuditResult:
    frame = _load_slp_e0_diagnostic_frame(well.horizontal_path)
    known_count, prediction_count = validate_prefix(
        frame["TVT_input"], target=frame["TVT"]
    )
    md = frame["MD"].to_numpy(dtype=np.float64, copy=False)
    target_all = frame["TVT"].to_numpy(dtype=np.float64, copy=False)
    z_all = frame["Z"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(md).all() or not np.all(np.diff(md) > 0.0):
        raise ValueError(f"SLP-E0 MD must be finite and strictly increasing: {well.well_id}")
    if not np.isfinite(target_all).all() or not np.isfinite(z_all).all():
        raise ValueError(f"SLP-E0 TVT and Z must be finite: {well.well_id}")
    row_index = np.arange(known_count, len(frame), dtype=np.int64)
    expected = np.arange(known_count, len(frame), dtype=np.int64)
    if (
        row_index.size != prediction_count
        or np.unique(row_index).size != row_index.size
        or not np.array_equal(row_index, expected)
    ):
        raise ValueError(f"SLP-E0 prediction-row coverage mismatch: {well.well_id}")
    post_md = md[row_index]
    md_span = float(post_md[-1] - post_md[0])
    if post_md.size < 2 or not math.isfinite(md_span) or md_span <= 0.0:
        raise ValueError(f"SLP-E0 post-PS MD span must be positive: {well.well_id}")
    normalized_md = np.asarray((post_md - post_md[0]) / md_span, dtype=np.float64)
    target_tvt = np.asarray(target_all[row_index], dtype=np.float64)
    target_h = np.asarray(target_tvt + z_all[row_index], dtype=np.float64)

    oracle_rows: list[dict[str, Any]] = []
    flat_prediction = np.full(
        prediction_count,
        float(frame["TVT_input"].iloc[known_count - 1]),
        dtype=np.float64,
    )
    flat_residual = target_tvt - flat_prediction
    flat_sse = float(flat_residual @ flat_residual)
    flat_fit = OracleFit(
        predictions=flat_prediction,
        rows=prediction_count,
        sse=flat_sse,
        rmse=math.sqrt(flat_sse / prediction_count),
    )
    oracle_rows.append(
        _oracle_row(
            well_id=well.well_id,
            representation="TVT",
            method="flat_last_known_reference",
            complexity_type="reference",
            complexity=0,
            fit=flat_fit,
        )
    )

    for degree in SLP_E0_POLYNOMIAL_DEGREES:
        fit, reason = _fit_or_unsupported(
            fit_polynomial_oracle,
            normalized_md=normalized_md,
            target=target_tvt,
            complexity_name="degree",
            complexity=degree,
        )
        oracle_rows.append(
            _oracle_row(
                well_id=well.well_id,
                representation="TVT",
                method="polynomial_oracle",
                complexity_type="degree",
                complexity=degree,
                fit=fit,
                unsupported_reason=reason,
            )
        )

    representations = (("TVT", target_tvt), ("H", target_h))
    methods = (
        ("piecewise_continuous_oracle", fit_continuous_piecewise),
        ("piecewise_discontinuous_oracle", fit_discontinuous_piecewise),
    )
    for representation, target in representations:
        for method, fit_function in methods:
            for segments in SLP_E0_SEGMENT_GRID:
                fit, reason = _fit_or_unsupported(
                    fit_function,
                    normalized_md=normalized_md,
                    target=target,
                    complexity_name="segments",
                    complexity=segments,
                )
                oracle_rows.append(
                    _oracle_row(
                        well_id=well.well_id,
                        representation=representation,
                        method=method,
                        complexity_type="segments",
                        complexity=segments,
                        fit=fit,
                        unsupported_reason=reason,
                    )
                )

    prefix = frame["TVT_input"].iloc[:known_count].to_numpy(dtype=np.float64)
    nearest = nearest_prefix_distance(prefix, target_tvt)
    inside = (target_tvt >= float(prefix.min())) & (
        target_tvt <= float(prefix.max())
    )
    gr_observed = np.isfinite(
        frame["GR"].iloc[row_index].to_numpy(dtype=np.float64)
    )
    recurrence: dict[str, Any] = {
        "well_id": well.well_id,
        "rows": prediction_count,
        "inside_prefix_range_rows": int(inside.sum()),
        "inside_prefix_range_fraction": float(inside.mean()),
        "gr_observed_rows": int(gr_observed.sum()),
        "gr_observed_fraction": float(gr_observed.mean()),
    }
    for threshold in _RECURRENCE_THRESHOLDS:
        label = f"{threshold:.2f}".replace(".", "_")
        within = nearest <= threshold
        recurrence[f"nearest_distance_le_{label}_rows"] = int(within.sum())
        recurrence[f"nearest_distance_le_{label}_fraction"] = float(within.mean())
    within_one_and_gr = (nearest <= 1.0) & gr_observed
    recurrence["nearest_distance_le_1_00_and_gr_observed_rows"] = int(
        within_one_and_gr.sum()
    )
    recurrence["nearest_distance_le_1_00_and_gr_observed_fraction"] = float(
        within_one_and_gr.mean()
    )

    surfaces: list[dict[str, Any]] = []
    for formation in SLP_E0_FORMATION_COLUMNS:
        formation_values = frame[formation].iloc[row_index].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(formation_values).any():
            continue
        stats = surface_identity_stats(
            target_tvt,
            z_all[row_index],
            formation_values,
        )
        surfaces.append(
            {
                "well_id": well.well_id,
                "formation": formation,
                "finite_rows": stats.finite_rows,
                "mean_constant": stats.mean_constant,
                "centered_sse": stats.centered_sse,
                "population_std": stats.population_std,
            }
        )
    return _WellAuditResult(
        well_id=well.well_id,
        prediction_rows=prediction_count,
        oracle_rows=tuple(oracle_rows),
        recurrence=recurrence,
        surfaces=tuple(surfaces),
    )


def _safe_diagnostic_run_root(
    output_root: Path, *, run_id: str, data_root: Path
) -> Path:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "SLP-E0 run_id may contain only letters, numbers, dot, underscore, and dash"
        )
    output = Path(output_root).resolve()
    data = Path(data_root).resolve()
    run_root = output / run_id
    if output == data or data in output.parents or run_root == data or data in run_root.parents:
        raise ValueError("SLP-E0 diagnostic output cannot be inside competition data")
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(
    frame: pd.DataFrame, path: Path, *, gzip_compressed: bool = False
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    compression: str | Mapping[str, Any] | None = (
        {"method": "gzip", "mtime": 0} if gzip_compressed else None
    )
    frame.to_csv(
        temporary,
        index=False,
        lineterminator="\n",
        compression=compression,
    )
    os.replace(temporary, path)


def _quantile_mapping(values: Iterable[float]) -> dict[str, float]:
    array = _as_float64_vector(values, name="per-well diagnostic fractions")
    quantiles = np.quantile(array, _QUANTILES)
    return {
        f"q{int(q * 100):02d}": float(value)
        for q, value in zip(_QUANTILES, quantiles, strict=True)
    }


def _aggregate_oracle_metrics(
    per_well: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], dict[str, Any]]]:
    supported = per_well.loc[per_well["supported"]].copy()
    metric_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    method_order = {
        "flat_last_known_reference": 0,
        "polynomial_oracle": 1,
        "piecewise_continuous_oracle": 2,
        "piecewise_discontinuous_oracle": 3,
    }
    representation_order = {"TVT": 0, "H": 1}
    grouped = supported.groupby(
        ["representation", "method", "complexity_type", "complexity"],
        sort=False,
        dropna=False,
    )
    groups = sorted(
        grouped.groups,
        key=lambda key: (
            method_order[str(key[1])],
            representation_order[str(key[0])],
            int(key[3]),
        ),
    )
    for representation, method, complexity_type, complexity in groups:
        group = grouped.get_group(
            (representation, method, complexity_type, complexity)
        ).sort_values("well_id", kind="stable")
        metric = accumulate_pooled_metric(
            (
                (float(row.sse), int(row.rows))
                for row in group.itertuples(index=False)
            )
        )
        row = {
            "representation": str(representation),
            "method": str(method),
            "complexity_type": str(complexity_type),
            "complexity": int(complexity),
            "wells": metric.wells,
            "rows": metric.rows,
            "sse": metric.sse,
            "pooled_rmse": metric.pooled_rmse,
            "uses_target": SLP_E0_USES_TARGET,
            "deployable": SLP_E0_DEPLOYABLE,
            "submission_eligible": SLP_E0_SUBMISSION_ELIGIBLE,
        }
        metric_rows.append(row)
        lookup[(str(representation), str(method), int(complexity))] = row
    return pd.DataFrame(metric_rows), lookup


def _metric_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "wells": int(row["wells"]),
        "rows": int(row["rows"]),
        "sse": float(row["sse"]),
        "pooled_rmse": float(row["pooled_rmse"]),
    }


def accumulate_recurrence_diagnostic(
    per_well: pd.DataFrame,
) -> dict[str, Any]:
    """Pool recurrence diagnostic counts over rows, then summarize wells."""
    if per_well.empty or "rows" not in per_well:
        raise ValueError("Recurrence diagnostic requires non-empty per-well rows")
    ordered = per_well.sort_values("well_id", kind="stable")
    rows = int(ordered["rows"].sum())
    summary: dict[str, Any] = {
        "rows": rows,
        "uses_target": SLP_E0_USES_TARGET,
        "deployable": SLP_E0_DEPLOYABLE,
        "submission_eligible": SLP_E0_SUBMISSION_ELIGIBLE,
    }
    count_columns = [
        column for column in ordered.columns if column.endswith("_rows") and column != "rows"
    ]
    for column in count_columns:
        count = int(ordered[column].sum())
        summary[column] = count
        summary[column.removesuffix("_rows") + "_fraction"] = count / rows
    fraction_columns = sorted(
        column for column in ordered.columns if column.endswith("_fraction")
    )
    summary["per_well_fraction_quantiles"] = {
        column.removesuffix("_fraction"): _quantile_mapping(ordered[column])
        for column in fraction_columns
    }
    return summary


def _aggregate_surfaces(surface_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for formation in SLP_E0_FORMATION_COLUMNS:
        group = surface_rows.loc[
            surface_rows["formation"] == formation
        ].sort_values("well_id", kind="stable")
        if group.empty:
            continue
        finite_rows = int(group["finite_rows"].sum())
        centered_sse = math.fsum(group["centered_sse"].astype(float).tolist())
        per_well_std = group["population_std"].to_numpy(dtype=np.float64)
        constants = group["mean_constant"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "formation": formation,
                "wells": len(group),
                "finite_rows": finite_rows,
                "pooled_within_well_std": math.sqrt(centered_sse / finite_rows),
                "per_well_std_q50": float(np.quantile(per_well_std, 0.50)),
                "per_well_std_q90": float(np.quantile(per_well_std, 0.90)),
                "per_well_std_q95": float(np.quantile(per_well_std, 0.95)),
                "per_well_std_max": float(per_well_std.max()),
                "well_constant_q10": float(np.quantile(constants, 0.10)),
                "well_constant_q50": float(np.quantile(constants, 0.50)),
                "well_constant_q90": float(np.quantile(constants, 0.90)),
                "diagnostic_type": "structural_identity_only",
                "uses_target": SLP_E0_USES_TARGET,
                "deployable": SLP_E0_DEPLOYABLE,
                "submission_eligible": SLP_E0_SUBMISSION_ELIGIBLE,
            }
        )
    return pd.DataFrame(rows)


def run_slp_e0_diagnostic(
    *,
    data_root: Path,
    output_root: Path = Path(".runs"),
    run_id: str,
    workers: int = 4,
) -> dict[str, Any]:
    """Run the full-train, target-bearing, non-deployable SLP-E0 audit."""
    if workers < 1:
        raise ValueError("SLP-E0 workers must be positive")
    data = Path(data_root).resolve()
    train_wells = enumerate_wells(data, "train")
    run_root = _safe_diagnostic_run_root(
        output_root, run_id=run_id, data_root=data
    )

    results: list[_WellAuditResult] = []
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="rogii-slp-e0"
        ) as executor:
            for count, result in enumerate(
                executor.map(_audit_one_well, train_wells), start=1
            ):
                results.append(result)
                if count % 25 == 0 or count == len(train_wells):
                    print(
                        f"slp-e0 diagnostic wells={count}/{len(train_wells)}",
                        file=sys.stderr,
                        flush=True,
                    )

    if tuple(result.well_id for result in results) != tuple(
        well.well_id for well in train_wells
    ):
        raise ValueError("SLP-E0 well result ordering or coverage mismatch")
    prediction_rows = sum(result.prediction_rows for result in results)
    per_well_oracles = pd.DataFrame(
        row for result in results for row in result.oracle_rows
    )
    recurrence_per_well = pd.DataFrame(result.recurrence for result in results)
    surface_per_well = pd.DataFrame(
        row for result in results for row in result.surfaces
    )
    oracle_metrics, metric_lookup = _aggregate_oracle_metrics(per_well_oracles)
    recurrence_summary = accumulate_recurrence_diagnostic(
        recurrence_per_well
    )
    surface_identity = _aggregate_surfaces(surface_per_well)

    flat = metric_lookup[("TVT", "flat_last_known_reference", 0)]
    polynomial = {
        str(degree): _metric_summary(
            metric_lookup[("TVT", "polynomial_oracle", degree)]
        )
        for degree in SLP_E0_POLYNOMIAL_DEGREES
    }

    def piecewise_summary(representation: str, method: str) -> dict[str, Any]:
        return {
            str(segments): _metric_summary(
                metric_lookup[(representation, method, segments)]
            )
            for segments in SLP_E0_SEGMENT_GRID
            if (representation, method, segments) in metric_lookup
        }

    piecewise_tvt = {
        "continuous": piecewise_summary("TVT", "piecewise_continuous_oracle"),
        "discontinuous": piecewise_summary(
            "TVT", "piecewise_discontinuous_oracle"
        ),
    }
    piecewise_h = {
        "continuous": piecewise_summary("H", "piecewise_continuous_oracle"),
        "discontinuous": piecewise_summary(
            "H", "piecewise_discontinuous_oracle"
        ),
    }
    surface_summary = {
        str(row.formation): {
            "wells": int(row.wells),
            "finite_rows": int(row.finite_rows),
            "pooled_within_well_std": float(row.pooled_within_well_std),
            "per_well_std_q50": float(row.per_well_std_q50),
            "per_well_std_q90": float(row.per_well_std_q90),
            "per_well_std_q95": float(row.per_well_std_q95),
            "per_well_std_max": float(row.per_well_std_max),
            "well_constant_q10": float(row.well_constant_q10),
            "well_constant_q50": float(row.well_constant_q50),
            "well_constant_q90": float(row.well_constant_q90),
            "diagnostic_type": "structural_identity_only",
        }
        for row in surface_identity.itertuples(index=False)
    }
    summary = {
        "run_id": run_id,
        "audit_id": SLP_E0_AUDIT_ID,
        "uses_target": SLP_E0_USES_TARGET,
        "deployable": SLP_E0_DEPLOYABLE,
        "submission_eligible": SLP_E0_SUBMISSION_ELIGIBLE,
        "scope": "full_train_diagnostic",
        "data_root": str(data),
        "train_wells": len(train_wells),
        "prediction_rows": prediction_rows,
        "flat_last_known_rmse": float(flat["pooled_rmse"]),
        "polynomial_tvt": polynomial,
        "piecewise_tvt": piecewise_tvt,
        "piecewise_h_tvt_plus_z": piecewise_h,
        "recurrence": recurrence_summary,
        "surface_identity": surface_summary,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    config = {
        "run_id": run_id,
        "audit_id": SLP_E0_AUDIT_ID,
        "uses_target": SLP_E0_USES_TARGET,
        "deployable": SLP_E0_DEPLOYABLE,
        "submission_eligible": SLP_E0_SUBMISSION_ELIGIBLE,
        "input_split": "train",
        "data_root": str(data),
        "output_root": str(Path(output_root).resolve()),
        "workers": workers,
        "float_dtype": "float64",
        "polynomial_degrees": list(SLP_E0_POLYNOMIAL_DEGREES),
        "piecewise_segment_grid": list(SLP_E0_SEGMENT_GRID),
        "diagnostic_columns": list(SLP_E0_DIAGNOSTIC_COLUMNS),
        "formation_surfaces_usage": "structural_identity_diagnostic_only",
    }

    per_well_oracles = per_well_oracles.sort_values(
        ["well_id", "method", "representation", "complexity"],
        kind="stable",
    )
    recurrence_per_well = recurrence_per_well.sort_values(
        "well_id", kind="stable"
    )
    _write_json_atomic(run_root / "summary.json", summary)
    _write_json_atomic(run_root / "run_config.json", config)
    _write_csv_atomic(oracle_metrics, run_root / "oracle_metrics.csv")
    _write_csv_atomic(
        per_well_oracles,
        run_root / "per_well_oracles.csv.gz",
        gzip_compressed=True,
    )
    _write_json_atomic(
        run_root / "recurrence_summary.json", recurrence_summary
    )
    _write_csv_atomic(
        recurrence_per_well,
        run_root / "recurrence_per_well.csv.gz",
        gzip_compressed=True,
    )
    _write_csv_atomic(surface_identity, run_root / "surface_identity.csv")

    return {
        "run_root": run_root,
        "run_id": run_id,
        "audit_id": SLP_E0_AUDIT_ID,
        "uses_target": SLP_E0_USES_TARGET,
        "deployable": SLP_E0_DEPLOYABLE,
        "submission_eligible": SLP_E0_SUBMISSION_ELIGIBLE,
        "train_wells": len(train_wells),
        "prediction_rows": prediction_rows,
        "flat_last_known_rmse": float(flat["pooled_rmse"]),
    }
