"""Auditable metrics for wide, well-level out-of-fold prediction artifacts.

The functions in this module deliberately operate on immutable-looking wide
frames: one row per ``(well_id, row_index, fold)``, one truth column, and one
or more prediction columns.  Validation is always well grouped and the
competition metric is always pooled across rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd


OOF_KEY_COLUMNS = ("well_id", "row_index", "fold")
DEFAULT_TRUTH_COLUMN = "y_true"
DEFAULT_BOOTSTRAP_SEED = 20260728


def _prediction_columns(
    frame: pd.DataFrame, prediction_columns: Sequence[str]
) -> tuple[str, ...]:
    columns = tuple(map(str, prediction_columns))
    if not columns:
        raise ValueError("At least one prediction column is required")
    if len(set(columns)) != len(columns):
        raise ValueError("Prediction columns must be unique")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    return columns


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values: list[np.ndarray] = []
    for column in columns:
        try:
            current = pd.to_numeric(frame[column], errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {column!r} must be numeric") from exc
        if current.ndim != 1 or not np.isfinite(current).all():
            raise ValueError(f"Column {column!r} contains null or non-finite values")
        values.append(current)
    return np.column_stack(values)


def _integer_key(frame: pd.DataFrame, column: str, *, nonnegative: bool) -> np.ndarray:
    try:
        numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OOF key column {column!r} must be integer-valued") from exc
    if (
        not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
        or (nonnegative and np.any(numeric < 0.0))
    ):
        raise ValueError(f"OOF key column {column!r} must be finite integers")
    return numeric.astype(np.int64)


def canonical_oof_keys(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = OOF_KEY_COLUMNS,
) -> pd.DataFrame:
    """Return canonical sorted key values after strict schema validation."""

    keys = tuple(map(str, key_columns))
    if len(keys) != 3:
        raise ValueError("OOF keys must be (well_id, row_index, fold)")
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f"OOF frame is missing key columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("OOF frame must not be empty")
    well_column, row_column, fold_column = keys
    if frame[well_column].isna().any():
        raise ValueError("OOF well IDs must not be null")
    well = frame[well_column].astype(str)
    if (well.str.len() == 0).any():
        raise ValueError("OOF well IDs must not be empty")
    if int(well.nunique()) != int(frame[well_column].nunique(dropna=False)):
        raise ValueError("OOF well IDs collide after canonical string conversion")
    row = _integer_key(frame, row_column, nonnegative=True)
    fold = _integer_key(frame, fold_column, nonnegative=True)
    result = pd.DataFrame(
        {well_column: well.to_numpy(), row_column: row, fold_column: fold}
    )
    duplicate_well_rows = result.duplicated([well_column, row_column], keep=False)
    if duplicate_well_rows.any():
        example = result.loc[duplicate_well_rows].head(3).to_dict("records")
        raise ValueError(f"OOF contains duplicate well/row keys: {example}")
    folds_per_well = result.groupby(well_column, sort=False)[fold_column].nunique()
    if (folds_per_well != 1).any():
        bad = folds_per_well[folds_per_well != 1].index.astype(str).tolist()[:5]
        raise ValueError(f"OOF wells occur in multiple folds: {bad}")
    return result.sort_values(list(keys), kind="stable").reset_index(drop=True)


def _key_digest(keys: pd.DataFrame, key_columns: Sequence[str]) -> str:
    well_column, row_column, fold_column = tuple(key_columns)
    digest = hashlib.sha256()
    for well_id, row_index, fold in keys[
        [well_column, row_column, fold_column]
    ].itertuples(index=False, name=None):
        digest.update(f"{fold}\0{well_id}\0{row_index}\n".encode("utf-8"))
    return digest.hexdigest()


def validate_oof_frame(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    expected_keys: pd.DataFrame | None = None,
    expected_well_ids: Sequence[str] | None = None,
    expected_folds: Sequence[int] | None = None,
    check_expected_truth: bool = True,
    key_columns: Sequence[str] = OOF_KEY_COLUMNS,
) -> dict[str, Any]:
    """Validate exact keys, coverage, fold isolation, and finite values.

    Coverage comparison is order independent.  When ``expected_keys`` also
    contains ``truth_column``, truth values are required to match exactly by
    default; callers can disable that additional check explicitly.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("OOF artifact must be a pandas DataFrame")
    columns = _prediction_columns(frame, prediction_columns)
    if truth_column not in frame:
        raise ValueError(f"OOF frame is missing truth column {truth_column!r}")
    keys = canonical_oof_keys(frame, key_columns=key_columns)
    for numeric_column in (truth_column, *columns):
        _finite_numeric(frame, (numeric_column,))
    well_column, _, fold_column = tuple(map(str, key_columns))

    if expected_keys is not None:
        reference = canonical_oof_keys(expected_keys, key_columns=key_columns)
        actual_index = pd.MultiIndex.from_frame(keys[list(key_columns)])
        expected_index = pd.MultiIndex.from_frame(reference[list(key_columns)])
        missing = expected_index.difference(actual_index)
        extra = actual_index.difference(expected_index)
        if len(missing) or len(extra):
            raise ValueError(
                "OOF key coverage differs from reference: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        if check_expected_truth and truth_column in expected_keys:
            actual_values = frame.loc[:, [*key_columns, truth_column]].copy()
            expected_values = expected_keys.loc[:, [*key_columns, truth_column]].copy()
            well_key, row_key, fold_key = tuple(key_columns)
            for current in (actual_values, expected_values):
                current[well_key] = current[well_key].astype(str)
                current[row_key] = _integer_key(current, row_key, nonnegative=True)
                current[fold_key] = _integer_key(current, fold_key, nonnegative=True)
            left = actual_values.merge(
                expected_values,
                on=list(key_columns),
                how="inner",
                validate="one_to_one",
                suffixes=("_actual", "_expected"),
                sort=False,
            )
            actual_truth = pd.to_numeric(
                left[f"{truth_column}_actual"], errors="raise"
            ).to_numpy(dtype=np.float64)
            expected_truth = pd.to_numeric(
                left[f"{truth_column}_expected"], errors="raise"
            ).to_numpy(dtype=np.float64)
            if not np.array_equal(actual_truth, expected_truth):
                maximum = float(np.max(np.abs(actual_truth - expected_truth)))
                raise ValueError(
                    "OOF truth differs from reference; "
                    f"maximum_absolute_delta={maximum}"
                )

    actual_wells = set(keys[well_column].astype(str))
    if expected_well_ids is not None:
        expected_wells = set(map(str, expected_well_ids))
        if actual_wells != expected_wells:
            raise ValueError(
                "OOF well coverage differs: "
                f"missing={len(expected_wells - actual_wells)}, "
                f"extra={len(actual_wells - expected_wells)}"
            )
    actual_folds = tuple(sorted(map(int, keys[fold_column].unique())))
    if expected_folds is not None:
        required_folds = tuple(sorted(set(map(int, expected_folds))))
        if actual_folds != required_folds:
            raise ValueError(
                f"OOF fold coverage differs: {actual_folds} != {required_folds}"
            )
    return {
        "rows": int(len(frame)),
        "wells": int(len(actual_wells)),
        "folds": list(actual_folds),
        "prediction_columns": list(columns),
        "duplicate_well_row_keys": 0,
        "nonfinite_numeric_values": 0,
        "key_sha256": _key_digest(keys, key_columns),
        "coverage_checked": expected_keys is not None,
        "truth_checked": bool(
            expected_keys is not None
            and check_expected_truth
            and truth_column in expected_keys
        ),
    }


def pooled_metrics(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    baseline_column: str | None = None,
) -> pd.DataFrame:
    """Compute pooled row metrics once for each prediction column."""

    columns = _prediction_columns(frame, prediction_columns)
    validate_oof_frame(frame, columns, truth_column=truth_column)
    truth = frame[truth_column].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    rmse_by_candidate: dict[str, float] = {}
    for candidate in columns:
        error = frame[candidate].to_numpy(dtype=np.float64) - truth
        sse = float(error @ error)
        rmse = math.sqrt(sse / len(error))
        rmse_by_candidate[candidate] = rmse
        rows.append(
            {
                "candidate": candidate,
                "n_rows": int(len(frame)),
                "n_wells": int(frame["well_id"].nunique()),
                "sse": sse,
                "pooled_row_rmse": rmse,
            }
        )
    result = pd.DataFrame(rows)
    if baseline_column is not None:
        if baseline_column not in rmse_by_candidate:
            raise ValueError("Baseline column must be one of prediction_columns")
        result["delta_rmse_vs_baseline"] = (
            result["pooled_row_rmse"] - rmse_by_candidate[baseline_column]
        )
    return result


def per_well_metrics(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    baseline_column: str | None = None,
) -> pd.DataFrame:
    """Return one auditable SSE/RMSE row per well and candidate."""

    columns = _prediction_columns(frame, prediction_columns)
    validate_oof_frame(frame, columns, truth_column=truth_column)
    output: list[pd.DataFrame] = []
    for candidate in columns:
        current = frame.loc[:, ["well_id", "fold"]].copy()
        error = frame[candidate].to_numpy(dtype=np.float64) - frame[
            truth_column
        ].to_numpy(dtype=np.float64)
        current["_squared_error"] = error * error
        aggregate = current.groupby(["well_id", "fold"], sort=True, as_index=False).agg(
            n_rows=("_squared_error", "size"), sse=("_squared_error", "sum")
        )
        aggregate["rmse"] = np.sqrt(aggregate["sse"] / aggregate["n_rows"])
        aggregate.insert(0, "candidate", candidate)
        output.append(aggregate)
    result = pd.concat(output, ignore_index=True)
    if baseline_column is not None:
        if baseline_column not in columns:
            raise ValueError("Baseline column must be one of prediction_columns")
        baseline = result.loc[
            result["candidate"].eq(baseline_column), ["well_id", "rmse", "sse"]
        ].rename(columns={"rmse": "baseline_rmse", "sse": "baseline_sse"})
        result = result.merge(
            baseline, on="well_id", how="left", validate="many_to_one"
        )
        result["rmse_delta_vs_baseline"] = result["rmse"] - result["baseline_rmse"]
        result["sse_delta_vs_baseline"] = result["sse"] - result["baseline_sse"]
    return result


def per_fold_metrics(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    baseline_column: str | None = None,
) -> pd.DataFrame:
    """Return pooled row metrics separately for every outer fold."""

    columns = _prediction_columns(frame, prediction_columns)
    validate_oof_frame(frame, columns, truth_column=truth_column)
    rows: list[dict[str, Any]] = []
    for fold, group in frame.groupby("fold", sort=True):
        truth = group[truth_column].to_numpy(dtype=np.float64)
        fold_metrics: dict[str, float] = {}
        for candidate in columns:
            error = group[candidate].to_numpy(dtype=np.float64) - truth
            sse = float(error @ error)
            rmse = math.sqrt(sse / len(group))
            fold_metrics[candidate] = rmse
            rows.append(
                {
                    "fold": int(fold),
                    "candidate": candidate,
                    "n_wells": int(group["well_id"].nunique()),
                    "n_rows": int(len(group)),
                    "sse": sse,
                    "pooled_row_rmse": rmse,
                }
            )
        if baseline_column is not None and baseline_column not in fold_metrics:
            raise ValueError("Baseline column must be one of prediction_columns")
    result = pd.DataFrame(rows)
    if baseline_column is not None:
        baseline = result.loc[
            result["candidate"].eq(baseline_column),
            ["fold", "pooled_row_rmse"],
        ].rename(columns={"pooled_row_rmse": "baseline_rmse"})
        result = result.merge(baseline, on="fold", how="left", validate="many_to_one")
        result["delta_rmse_vs_baseline"] = (
            result["pooled_row_rmse"] - result["baseline_rmse"]
        )
    return result


def tail_metrics(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
) -> pd.DataFrame:
    """Summarize per-well tails, ranking the worst decile by SSE."""

    wells = per_well_metrics(frame, prediction_columns, truth_column=truth_column)
    rows: list[dict[str, Any]] = []
    for candidate, group in wells.groupby("candidate", sort=False):
        rmse = group["rmse"].to_numpy(dtype=np.float64)
        sse = group["sse"].to_numpy(dtype=np.float64)
        total_sse = float(sse.sum())
        if total_sse <= 0.0:
            worst_share = 0.0
        else:
            worst_count = max(1, math.ceil(0.10 * len(group)))
            worst_share = float(np.sort(sse)[-worst_count:].sum() / total_sse)
        rows.append(
            {
                "candidate": candidate,
                "n_wells": int(len(group)),
                "p50_rmse": float(np.quantile(rmse, 0.50)),
                "p90_rmse": float(np.quantile(rmse, 0.90)),
                "p95_rmse": float(np.quantile(rmse, 0.95)),
                "max_rmse": float(np.max(rmse)),
                "wells_rmse_ge_40": int(np.sum(rmse >= 40.0)),
                "wells_rmse_ge_60": int(np.sum(rmse >= 60.0)),
                "worst_10pct_sse_share": worst_share,
            }
        )
    return pd.DataFrame(rows)


def common_slope_shape_metrics(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    position_column: str = "row_index",
) -> pd.DataFrame:
    """Orthogonally decompose row residuals into common, slope, and shape.

    Residual means ``prediction - truth``.  For each well, the common component
    is its mean residual; the slope component is the least-squares projection
    of the centered residual onto centered row position; the remaining
    component is therefore zero-mean and orthogonal to the linear slope.
    Component RMSEs are pooled over rows, while terminal RMSE has one value per
    well.
    """

    columns = _prediction_columns(frame, prediction_columns)
    if position_column not in frame:
        raise ValueError(f"Missing position column {position_column!r}")
    validate_oof_frame(frame, columns, truth_column=truth_column)
    position = _finite_numeric(frame, (position_column,))[:, 0]
    truth = frame[truth_column].to_numpy(dtype=np.float64)
    groups = frame.groupby("well_id", sort=True).indices
    rows: list[dict[str, Any]] = []
    for candidate in columns:
        error = frame[candidate].to_numpy(dtype=np.float64) - truth
        total_sse = 0.0
        common_sse = 0.0
        centered_sse = 0.0
        slope_sse = 0.0
        shape_sse = 0.0
        error_sum = 0.0
        terminal: list[float] = []
        slope_coefficients: list[float] = []
        for indices in groups.values():
            selected = np.asarray(indices, dtype=np.int64)
            current_error = error[selected]
            current_position = position[selected]
            order = np.argsort(current_position, kind="stable")
            current_error = current_error[order]
            current_position = current_position[order]
            common = float(np.mean(current_error))
            centered = current_error - common
            x = current_position - float(np.mean(current_position))
            scale = float(np.max(np.abs(x))) if len(x) else 0.0
            normalized_x = x / scale if scale > 0.0 else np.zeros_like(x)
            denominator = float(normalized_x @ normalized_x)
            coefficient = (
                float(normalized_x @ centered) / denominator
                if denominator > 0.0
                else 0.0
            )
            slope = coefficient * normalized_x
            shape = centered - slope
            total_sse += float(current_error @ current_error)
            common_sse += len(current_error) * common * common
            centered_sse += float(centered @ centered)
            slope_sse += float(slope @ slope)
            shape_sse += float(shape @ shape)
            error_sum += float(np.sum(current_error))
            terminal.append(float(current_error[-1]))
            slope_coefficients.append(coefficient)
        n_rows = len(frame)
        reconstruction_error = abs(
            total_sse - (common_sse + slope_sse + shape_sse)
        ) / max(total_sse, 1.0)
        rows.append(
            {
                "candidate": candidate,
                "n_rows": int(n_rows),
                "n_wells": int(len(groups)),
                "total_rmse": math.sqrt(total_sse / n_rows),
                "common_mode_rmse": math.sqrt(common_sse / n_rows),
                "common_mode_bias": error_sum / n_rows,
                "centered_rmse": math.sqrt(centered_sse / n_rows),
                "slope_component_rmse": math.sqrt(slope_sse / n_rows),
                "zero_mean_shape_rmse": math.sqrt(shape_sse / n_rows),
                "terminal_rmse": math.sqrt(
                    float(np.mean(np.square(terminal, dtype=np.float64)))
                ),
                "well_slope_coefficient_rmse": math.sqrt(
                    float(np.mean(np.square(slope_coefficients, dtype=np.float64)))
                ),
                "relative_mse_reconstruction_error": reconstruction_error,
            }
        )
    return pd.DataFrame(rows)


def _well_sse_arrays(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    columns = _prediction_columns(frame, prediction_columns)
    validate_oof_frame(frame, columns, truth_column=truth_column)
    well_ids = np.asarray(sorted(frame["well_id"].astype(str).unique()))
    truth = frame[truth_column].to_numpy(dtype=np.float64)
    group = frame["well_id"].astype(str).to_numpy()
    row_counts = (
        pd.Series(group)
        .value_counts(sort=False)
        .reindex(well_ids)
        .to_numpy(dtype=np.int64)
    )
    sse: dict[str, np.ndarray] = {}
    for candidate in columns:
        error = frame[candidate].to_numpy(dtype=np.float64) - truth
        squared = (
            pd.DataFrame({"well_id": group, "squared_error": error * error})
            .groupby("well_id", sort=False)["squared_error"]
            .sum()
        )
        sse[candidate] = squared.reindex(well_ids).to_numpy(dtype=np.float64)
    return well_ids, sse, row_counts


def paired_well_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    baseline_column: str,
    candidate_columns: Sequence[str],
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    samples: int = 10_000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    block_size: int = 256,
) -> dict[str, Any]:
    """Paired bootstrap wells with replacement, retaining every sampled row."""

    candidates = tuple(map(str, candidate_columns))
    if not candidates:
        raise ValueError("At least one bootstrap candidate is required")
    if baseline_column in candidates:
        raise ValueError("Baseline must not also appear in candidate_columns")
    if samples <= 0 or block_size <= 0:
        raise ValueError("Bootstrap samples and block_size must be positive")
    well_ids, sse, row_counts = _well_sse_arrays(
        frame,
        (baseline_column, *candidates),
        truth_column=truth_column,
    )
    random = np.random.default_rng(seed)
    deltas = {
        candidate: np.empty(samples, dtype=np.float64) for candidate in candidates
    }
    offset = 0
    while offset < samples:
        size = min(block_size, samples - offset)
        indices = random.integers(
            0, len(well_ids), size=(size, len(well_ids)), endpoint=False
        )
        denominator = np.sum(row_counts[indices], axis=1)
        baseline = np.sqrt(np.sum(sse[baseline_column][indices], axis=1) / denominator)
        for candidate in candidates:
            current = np.sqrt(np.sum(sse[candidate][indices], axis=1) / denominator)
            deltas[candidate][offset : offset + size] = current - baseline
        offset += size
    point_baseline = math.sqrt(
        float(np.sum(sse[baseline_column])) / int(np.sum(row_counts))
    )
    return {
        "method": (
            "paired well-cluster bootstrap; wells sampled with replacement; "
            "all rows retained"
        ),
        "baseline": baseline_column,
        "samples": int(samples),
        "seed": int(seed),
        "wells": int(len(well_ids)),
        "candidates": {
            candidate: {
                "point_delta_rmse": math.sqrt(
                    float(np.sum(sse[candidate])) / int(np.sum(row_counts))
                )
                - point_baseline,
                "delta_rmse_q2.5": float(np.quantile(delta, 0.025)),
                "delta_rmse_q50": float(np.quantile(delta, 0.50)),
                "delta_rmse_q97.5": float(np.quantile(delta, 0.975)),
            }
            for candidate, delta in deltas.items()
        },
    }


def residual_correlation(
    frame: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    aggregation: str = "row",
) -> pd.DataFrame:
    """Return a labeled residual-correlation matrix.

    ``aggregation='well_mean'`` first averages residuals within each well.
    Constant residual vectors receive correlation zero with other vectors and
    one on their own diagonal, an explicit finite convention.
    """

    columns = _prediction_columns(frame, prediction_columns)
    validate_oof_frame(frame, columns, truth_column=truth_column)
    truth = frame[truth_column].to_numpy(dtype=np.float64)
    residual = np.column_stack(
        [frame[column].to_numpy(dtype=np.float64) - truth for column in columns]
    )
    if aggregation == "well_mean":
        residual_frame = pd.DataFrame(residual, columns=columns)
        residual_frame.insert(0, "well_id", frame["well_id"].astype(str).to_numpy())
        residual = (
            residual_frame.groupby("well_id", sort=True)[list(columns)]
            .mean()
            .to_numpy(dtype=np.float64)
        )
    elif aggregation != "row":
        raise ValueError("aggregation must be 'row' or 'well_mean'")
    centered = residual - np.mean(residual, axis=0, keepdims=True)
    gram = centered.T @ centered
    scale = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    return pd.DataFrame(correlation, index=columns, columns=columns)


def per_well_win_summary(
    frame: pd.DataFrame,
    *,
    baseline_column: str,
    candidate_columns: Sequence[str],
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Count candidate well-level wins, ties, and losses versus a baseline."""

    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be nonnegative")
    candidates = tuple(map(str, candidate_columns))
    if not candidates:
        raise ValueError("At least one win/loss candidate is required")
    metrics = per_well_metrics(
        frame,
        (baseline_column, *candidates),
        truth_column=truth_column,
    )
    pivot = metrics.pivot(index="well_id", columns="candidate", values="rmse")
    rows: list[dict[str, Any]] = []
    baseline = pivot[baseline_column].to_numpy(dtype=np.float64)
    for candidate in candidates:
        delta = pivot[candidate].to_numpy(dtype=np.float64) - baseline
        wins = int(np.sum(delta < -tie_tolerance))
        losses = int(np.sum(delta > tie_tolerance))
        ties = int(len(delta) - wins - losses)
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline_column,
                "wells": int(len(delta)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_fraction": wins / len(delta),
                "nonloss_fraction": (wins + ties) / len(delta),
                "mean_well_rmse_delta": float(np.mean(delta)),
                "median_well_rmse_delta": float(np.median(delta)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_family_out(
    frame: pd.DataFrame,
    *,
    family_columns: Mapping[str, str],
    weights: Mapping[str, float],
    truth_column: str = DEFAULT_TRUTH_COLUMN,
) -> pd.DataFrame:
    """Measure each family's marginal contribution by renormalized removal."""

    families = tuple(map(str, family_columns.keys()))
    if not families:
        raise ValueError("At least one family is required")
    if set(map(str, weights.keys())) != set(families):
        raise ValueError("Weight keys must exactly match family names")
    columns = tuple(str(family_columns[family]) for family in families)
    _prediction_columns(frame, columns)
    validate_oof_frame(frame, columns, truth_column=truth_column)
    weight = np.asarray([float(weights[family]) for family in families])
    if (
        not np.isfinite(weight).all()
        or np.any(weight < 0.0)
        or not math.isclose(float(weight.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("Family weights must be finite, nonnegative, and sum to one")
    prediction = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    truth = frame[truth_column].to_numpy(dtype=np.float64)
    full_error = prediction @ weight - truth
    full_sse = float(full_error @ full_error)
    full_rmse = math.sqrt(full_sse / len(frame))
    rows: list[dict[str, Any]] = []
    for index, family in enumerate(families):
        remaining = 1.0 - weight[index]
        if remaining <= 1e-15:
            raise ValueError(f"Cannot remove sole positive-weight family {family!r}")
        without_weight = weight.copy()
        without_weight[index] = 0.0
        without_weight /= remaining
        without_error = prediction @ without_weight - truth
        without_sse = float(without_error @ without_error)
        without_rmse = math.sqrt(without_sse / len(frame))
        rows.append(
            {
                "family": family,
                "weight": float(weight[index]),
                "full_rmse": full_rmse,
                "rmse_without_family": without_rmse,
                "marginal_rmse_increase": without_rmse - full_rmse,
                "full_sse": full_sse,
                "sse_without_family": without_sse,
            }
        )
    return pd.DataFrame(rows)


def shadow_private_subset_simulation(
    frame: pd.DataFrame,
    *,
    baseline_column: str,
    candidate_columns: Sequence[str],
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    samples: int = 100_000,
    subset_wells: int = 200,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    block_size: int = 1_000,
) -> dict[str, Any]:
    """Simulate uniformly sampled fixed-size hidden-well subsets.

    Sampling is without replacement inside each subset.  Random-key
    ``argpartition`` makes the result deterministic for a fixed seed and
    independent of well ordering.
    """

    candidates = tuple(map(str, candidate_columns))
    if not candidates:
        raise ValueError("At least one shadow-private candidate is required")
    if baseline_column in candidates:
        raise ValueError("Baseline must not also appear in candidate_columns")
    if samples <= 0 or block_size <= 0 or subset_wells <= 0:
        raise ValueError("samples, subset_wells, and block_size must be positive")
    well_ids, sse, row_counts = _well_sse_arrays(
        frame,
        (baseline_column, *candidates),
        truth_column=truth_column,
    )
    if subset_wells > len(well_ids):
        raise ValueError(
            f"subset_wells={subset_wells} exceeds available wells={len(well_ids)}"
        )
    random = np.random.default_rng(seed)
    deltas = {
        candidate: np.empty(samples, dtype=np.float64) for candidate in candidates
    }
    offset = 0
    while offset < samples:
        size = min(block_size, samples - offset)
        if subset_wells == len(well_ids):
            indices = np.broadcast_to(
                np.arange(len(well_ids), dtype=np.int64), (size, len(well_ids))
            )
        else:
            random_keys = random.random((size, len(well_ids)))
            indices = np.argpartition(random_keys, subset_wells - 1, axis=1)[
                :, :subset_wells
            ]
        denominator = np.sum(row_counts[indices], axis=1)
        base = np.sqrt(np.sum(sse[baseline_column][indices], axis=1) / denominator)
        for candidate in candidates:
            current = np.sqrt(np.sum(sse[candidate][indices], axis=1) / denominator)
            deltas[candidate][offset : offset + size] = current - base
        offset += size
    return {
        "method": "uniform fixed-size well subsets sampled without replacement",
        "baseline": baseline_column,
        "samples": int(samples),
        "subset_wells": int(subset_wells),
        "seed": int(seed),
        "available_wells": int(len(well_ids)),
        "candidates": {
            candidate: {
                "mean_delta": float(np.mean(delta)),
                "p_worse": float(np.mean(delta > 0.0)),
                "delta_q01": float(np.quantile(delta, 0.01)),
                "delta_q05": float(np.quantile(delta, 0.05)),
                "delta_q50": float(np.quantile(delta, 0.50)),
                "delta_q95": float(np.quantile(delta, 0.95)),
                "delta_q99": float(np.quantile(delta, 0.99)),
            }
            for candidate, delta in deltas.items()
        },
    }


_DEFAULT_FORBIDDEN_STRATA_COLUMNS = frozenset(
    {
        "y_true",
        "target",
        "label",
        "rmse",
        "sse",
        "error",
        "residual",
    }
)


def make_target_free_strata(
    well_features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    well_id_column: str = "well_id",
    bins: int = 5,
    categorical_columns: Sequence[str] = (),
    forbidden_columns: Sequence[str] = tuple(_DEFAULT_FORBIDDEN_STRATA_COLUMNS),
) -> pd.DataFrame:
    """Create strata using only one-row-per-well pre-prediction features.

    Numeric columns use deterministic empirical-quantile edges.  Explicit
    categorical columns (and nonnumeric columns) retain their observed
    categories.  The long result can be frozen before any target is joined.
    """

    if bins < 1:
        raise ValueError("bins must be positive")
    features = tuple(map(str, feature_columns))
    if not features or len(set(features)) != len(features):
        raise ValueError("feature_columns must be non-empty and unique")
    missing = {well_id_column, *features} - set(well_features.columns)
    if missing:
        raise ValueError(f"Missing target-free stratum columns: {sorted(missing)}")
    forbidden = set(map(str, forbidden_columns))
    bad = set(features) & forbidden
    bad.update(column for column in features if column.startswith("y_pred"))
    if bad:
        raise ValueError(f"Target-derived stratum columns are forbidden: {sorted(bad)}")
    if well_features.empty or well_features[well_id_column].isna().any():
        raise ValueError("Target-free well features must be non-empty with valid IDs")
    ids = well_features[well_id_column].astype(str)
    if ids.duplicated().any():
        raise ValueError("Target-free features must contain exactly one row per well")
    categorical = set(map(str, categorical_columns))
    rows: list[dict[str, Any]] = []
    for feature in features:
        series = well_features[feature]
        is_categorical = feature in categorical or not pd.api.types.is_numeric_dtype(
            series
        )
        if is_categorical:
            if series.isna().any():
                raise ValueError(f"Categorical stratum feature {feature!r} has nulls")
            for well_id, value in zip(ids, series.astype(str), strict=True):
                rows.append(
                    {
                        "well_id": well_id,
                        "stratum_feature": feature,
                        "stratum": f"category={value}",
                        "feature_value": value,
                        "lower_bound": np.nan,
                        "upper_bound": np.nan,
                    }
                )
            continue
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Numeric stratum feature {feature!r} must be finite")
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        actual_bins = max(1, len(edges) - 1)
        if len(edges) == 1:
            assignments = np.zeros(len(values), dtype=np.int64)
            lower = upper = np.repeat(edges[0], len(values))
        else:
            assignments = np.searchsorted(edges[1:-1], values, side="right")
            lower = edges[assignments]
            upper = edges[assignments + 1]
        for well_id, value, assignment, lo, hi in zip(
            ids, values, assignments, lower, upper, strict=True
        ):
            rows.append(
                {
                    "well_id": well_id,
                    "stratum_feature": feature,
                    "stratum": f"q{int(assignment) + 1:02d}_of_{actual_bins:02d}",
                    "feature_value": float(value),
                    "lower_bound": float(lo),
                    "upper_bound": float(hi),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["stratum_feature", "stratum", "well_id"], kind="stable")
        .reset_index(drop=True)
    )


def target_free_stratified_metrics(
    frame: pd.DataFrame,
    strata: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = DEFAULT_TRUTH_COLUMN,
    baseline_column: str | None = None,
) -> pd.DataFrame:
    """Evaluate predictions after joining a previously frozen stratum map."""

    columns = _prediction_columns(frame, prediction_columns)
    validate_oof_frame(frame, columns, truth_column=truth_column)
    required = {"well_id", "stratum_feature", "stratum"}
    missing = required - set(strata.columns)
    if missing:
        raise ValueError(f"Strata frame is missing columns: {sorted(missing)}")
    if strata.duplicated(["well_id", "stratum_feature"]).any():
        raise ValueError("Each well may have only one stratum per feature")
    oof_wells = set(frame["well_id"].astype(str))
    stratum_groups = tuple(strata.groupby("stratum_feature", sort=True))
    for feature, group in stratum_groups:
        stratum_wells = set(group["well_id"].astype(str))
        if stratum_wells != oof_wells:
            raise ValueError(
                f"Stratum feature {feature!r} has incomplete well coverage"
            )
    rows: list[dict[str, Any]] = []
    metric_frame = frame.loc[:, ["well_id", truth_column, *columns]].copy()
    metric_frame["well_id"] = metric_frame["well_id"].astype(str)
    for feature, mapping in stratum_groups:
        current_mapping = mapping.loc[:, ["well_id", "stratum"]].copy()
        current_mapping["well_id"] = current_mapping["well_id"].astype(str)
        joined = metric_frame.merge(
            current_mapping,
            on="well_id",
            how="inner",
            validate="many_to_one",
            sort=False,
        )
        for stratum, group in joined.groupby("stratum", sort=True):
            truth = group[truth_column].to_numpy(dtype=np.float64)
            current_rmse: dict[str, float] = {}
            for candidate in columns:
                error = group[candidate].to_numpy(dtype=np.float64) - truth
                sse = float(error @ error)
                rmse = math.sqrt(sse / len(group))
                current_rmse[candidate] = rmse
                rows.append(
                    {
                        "stratum_feature": feature,
                        "stratum": stratum,
                        "candidate": candidate,
                        "n_wells": int(group["well_id"].nunique()),
                        "n_rows": int(len(group)),
                        "sse": sse,
                        "pooled_row_rmse": rmse,
                    }
                )
            if baseline_column is not None and baseline_column not in current_rmse:
                raise ValueError("Baseline column must be one of prediction_columns")
    result = pd.DataFrame(rows)
    if baseline_column is not None:
        base = result.loc[
            result["candidate"].eq(baseline_column),
            ["stratum_feature", "stratum", "pooled_row_rmse"],
        ].rename(columns={"pooled_row_rmse": "baseline_rmse"})
        result = result.merge(
            base,
            on=["stratum_feature", "stratum"],
            how="left",
            validate="many_to_one",
        )
        result["delta_rmse_vs_baseline"] = (
            result["pooled_row_rmse"] - result["baseline_rmse"]
        )
    return result


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_TRUTH_COLUMN",
    "OOF_KEY_COLUMNS",
    "canonical_oof_keys",
    "common_slope_shape_metrics",
    "leave_one_family_out",
    "make_target_free_strata",
    "paired_well_cluster_bootstrap",
    "per_fold_metrics",
    "per_well_metrics",
    "per_well_win_summary",
    "pooled_metrics",
    "residual_correlation",
    "shadow_private_subset_simulation",
    "tail_metrics",
    "target_free_stratified_metrics",
    "validate_oof_frame",
]
