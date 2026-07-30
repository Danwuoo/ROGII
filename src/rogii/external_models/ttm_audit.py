"""Disk-recomputable metrics and the immutable TTM quick transfer gate."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..portfolio_metrics import (
    common_slope_shape_metrics,
    paired_well_cluster_bootstrap,
    per_fold_metrics,
    per_well_metrics,
    per_well_win_summary,
    pooled_metrics,
    residual_correlation,
    tail_metrics,
    validate_oof_frame,
)
from .ttm_embeddings import FROZEN_VARIANTS


INCUMBENT_COLUMN = "y_pred_incumbent"
RANDOM_COLUMN = "y_pred__ttm_random_frozen"
PRETRAINED_COLUMN = "y_pred__ttm_pretrained_frozen"
CONTROL_COLUMNS = (
    "y_pred__ttm_pretrained_wrong_typewell",
    "y_pred__ttm_pretrained_pair_shuffle",
    "y_pred__ttm_pretrained_gr_order_shuffle",
    "y_pred__ttm_pretrained_phase_destroy",
)


def prediction_column(variant_id: str) -> str:
    return f"y_pred__{variant_id}"


def seed_prediction_column(variant_id: str, seed: int) -> str:
    return f"y_pred__{variant_id}__seed_{int(seed)}"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def _well_sse(
    frame: pd.DataFrame,
    column: str,
    *,
    well_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error = frame[column].to_numpy(dtype=np.float64) - frame["y_true"].to_numpy(
        dtype=np.float64
    )
    values = pd.DataFrame(
        {
            "well_id": frame["well_id"].astype(str),
            "squared_error": error * error,
        }
    )
    grouped = values.groupby("well_id", sort=True)["squared_error"]
    sse = grouped.sum().reindex(well_ids).to_numpy(dtype=np.float64)
    rows = grouped.size().reindex(well_ids).to_numpy(dtype=np.int64)
    return sse, rows


def paired_transfer_bootstrap(
    frame: pd.DataFrame,
    *,
    random_column: str = RANDOM_COLUMN,
    pretrained_column: str = PRETRAINED_COLUMN,
    samples: int = 10_000,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Bootstrap the fixed pretrained-minus-random transfer contrast by well."""

    validate_oof_frame(frame, (random_column, pretrained_column))
    well_ids = np.asarray(sorted(frame["well_id"].astype(str).unique()))
    random_sse, rows = _well_sse(frame, random_column, well_ids=well_ids)
    pretrained_sse, rows_check = _well_sse(
        frame,
        pretrained_column,
        well_ids=well_ids,
    )
    if not np.array_equal(rows, rows_check):
        raise ValueError("paired transfer bootstrap row counts differ")
    random = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    offset = 0
    block = 256
    while offset < samples:
        size = min(block, samples - offset)
        indices = random.integers(
            0,
            len(well_ids),
            size=(size, len(well_ids)),
            endpoint=False,
        )
        denominator = np.sum(rows[indices], axis=1)
        random_rmse = np.sqrt(np.sum(random_sse[indices], axis=1) / denominator)
        pretrained_rmse = np.sqrt(
            np.sum(pretrained_sse[indices], axis=1) / denominator
        )
        deltas[offset : offset + size] = pretrained_rmse - random_rmse
        offset += size
    total_rows = int(np.sum(rows))
    point = math.sqrt(float(np.sum(pretrained_sse)) / total_rows) - math.sqrt(
        float(np.sum(random_sse)) / total_rows
    )
    return {
        "method": "paired well-cluster bootstrap; all rows retained",
        "contrast": "pretrained_frozen_minus_random_frozen",
        "samples": int(samples),
        "seed": int(seed),
        "wells": int(len(well_ids)),
        "point_delta_rmse": point,
        "delta_rmse_q2.5": float(np.quantile(deltas, 0.025)),
        "delta_rmse_q50": float(np.quantile(deltas, 0.50)),
        "delta_rmse_q97.5": float(np.quantile(deltas, 0.975)),
        "probability_pretrained_better": float(np.mean(deltas < 0.0)),
        "probability_pretrained_worse": float(np.mean(deltas > 0.0)),
    }


def evaluate_ttm_oof(
    oof: pd.DataFrame,
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    bootstrap_samples: int,
) -> dict[str, Any]:
    family_columns = [prediction_column(variant) for variant in variants]
    seed_columns = [
        seed_prediction_column(variant, seed)
        for variant in variants
        for seed in seeds
    ]
    prediction_columns = [INCUMBENT_COLUMN, *family_columns, *seed_columns]
    validation = validate_oof_frame(oof, prediction_columns)
    pooled = pooled_metrics(
        oof,
        prediction_columns,
        baseline_column=INCUMBENT_COLUMN,
    )
    folds = per_fold_metrics(
        oof,
        [INCUMBENT_COLUMN, *family_columns],
        baseline_column=INCUMBENT_COLUMN,
    )
    wells = per_well_metrics(
        oof,
        [INCUMBENT_COLUMN, *family_columns],
        baseline_column=INCUMBENT_COLUMN,
    )
    tails = tail_metrics(oof, [INCUMBENT_COLUMN, *family_columns])
    decomposition = common_slope_shape_metrics(
        oof,
        [INCUMBENT_COLUMN, *family_columns],
    )
    bootstrap_vs_incumbent = paired_well_cluster_bootstrap(
        oof,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=family_columns,
        samples=bootstrap_samples,
        seed=20260729,
    )
    transfer = paired_transfer_bootstrap(
        oof,
        samples=bootstrap_samples,
        seed=20260729,
    )
    wins = per_well_win_summary(
        oof,
        baseline_column=INCUMBENT_COLUMN,
        candidate_columns=family_columns,
    )
    correlation = residual_correlation(
        oof,
        [INCUMBENT_COLUMN, *family_columns],
        aggregation="row",
    )
    return {
        "validation": validation,
        "prediction_columns": prediction_columns,
        "family_columns": family_columns,
        "seed_columns": seed_columns,
        "pooled_frame": pooled,
        "per_fold_frame": folds,
        "per_well_frame": wells,
        "tail_frame": tails,
        "decomposition_frame": decomposition,
        "wins_frame": wins,
        "correlation_frame": correlation,
        "summary": {
            "pooled": _records(pooled),
            "per_fold": _records(folds),
            "tail": _records(tails),
            "decomposition": _records(decomposition),
            "paired_bootstrap_vs_incumbent": bootstrap_vs_incumbent,
            "paired_transfer_bootstrap": transfer,
            "per_well_wins": _records(wins),
            "residual_correlation": correlation.to_dict(),
        },
    }


def locked_quick_gate(
    metrics: Mapping[str, Any],
    *,
    gate_config: Mapping[str, Any],
    safety_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the config-locked AND gate without changing thresholds post hoc."""

    summary = metrics["summary"]
    pooled = pd.DataFrame(summary["pooled"]).set_index("candidate")
    folds = pd.DataFrame(summary["per_fold"])
    tails = pd.DataFrame(summary["tail"]).set_index("candidate")
    decomposition = pd.DataFrame(summary["decomposition"]).set_index("candidate")
    transfer = dict(summary["paired_transfer_bootstrap"])
    random_rmse = float(pooled.loc[RANDOM_COLUMN, "pooled_row_rmse"])
    pretrained_rmse = float(pooled.loc[PRETRAINED_COLUMN, "pooled_row_rmse"])
    incumbent_rmse = float(pooled.loc[INCUMBENT_COLUMN, "pooled_row_rmse"])
    transfer_improvement = random_rmse - pretrained_rmse
    pretrained_delta_incumbent = pretrained_rmse - incumbent_rmse

    fold_pivot = folds.pivot(
        index="fold",
        columns="candidate",
        values="pooled_row_rmse",
    )
    fold_deltas = (
        fold_pivot[PRETRAINED_COLUMN] - fold_pivot[RANDOM_COLUMN]
    ).sort_index()
    consistent_folds = int(np.sum(fold_deltas.to_numpy(dtype=np.float64) < 0.0))

    slope_improvement = float(
        decomposition.loc[RANDOM_COLUMN, "slope_component_rmse"]
        - decomposition.loc[PRETRAINED_COLUMN, "slope_component_rmse"]
    )
    shape_improvement = float(
        decomposition.loc[RANDOM_COLUMN, "zero_mean_shape_rmse"]
        - decomposition.loc[PRETRAINED_COLUMN, "zero_mean_shape_rmse"]
    )
    control_degradation = {
        column: float(
            pooled.loc[column, "pooled_row_rmse"]
            - pooled.loc[PRETRAINED_COLUMN, "pooled_row_rmse"]
        )
        for column in CONTROL_COLUMNS
    }
    removed_fraction = {
        column: (
            degradation / transfer_improvement
            if transfer_improvement > 0.0
            else float("-inf")
        )
        for column, degradation in control_degradation.items()
    }
    best_removed_fraction = max(removed_fraction.values())

    incumbent_tail = tails.loc[INCUMBENT_COLUMN]
    pretrained_tail = tails.loc[PRETRAINED_COLUMN]
    tail_checks = {
        "p95": float(pretrained_tail["p95_rmse"] - incumbent_tail["p95_rmse"])
        <= float(gate_config["maximum_p95_degradation_ft"]),
        "max": float(pretrained_tail["max_rmse"] - incumbent_tail["max_rmse"])
        <= float(gate_config["maximum_max_well_degradation_ft"]),
        "ge40": int(pretrained_tail["wells_rmse_ge_40"])
        - int(incumbent_tail["wells_rmse_ge_40"])
        <= int(gate_config["maximum_catastrophic_count_increase"]),
        "ge60": int(pretrained_tail["wells_rmse_ge_60"])
        - int(incumbent_tail["wells_rmse_ge_60"])
        <= int(gate_config["maximum_catastrophic_count_increase"]),
        "worst_10pct_share": float(
            pretrained_tail["worst_10pct_sse_share"]
            - incumbent_tail["worst_10pct_sse_share"]
        )
        <= float(gate_config["maximum_worst_10pct_sse_share_increase"]),
    }
    checks = {
        "minimum_transfer_improvement": transfer_improvement
        >= float(gate_config["minimum_transfer_improvement_ft"]),
        "bootstrap_direction": (
            float(transfer["delta_rmse_q50"]) < 0.0
            and float(transfer["probability_pretrained_better"])
            >= float(
                gate_config[
                    "bootstrap_probability_pretrained_better_minimum"
                ]
            )
        ),
        "fold_direction": consistent_folds
        >= int(gate_config["minimum_directionally_consistent_folds"]),
        "standalone_vs_incumbent": pretrained_delta_incumbent
        <= float(gate_config["maximum_pretrained_delta_vs_incumbent_ft"]),
        "slope_or_shape": max(slope_improvement, shape_improvement)
        >= float(gate_config["minimum_slope_or_shape_improvement_ft"]),
        "tail": all(tail_checks.values()),
        "negative_control_sensitivity": best_removed_fraction
        >= float(gate_config["minimum_removed_transfer_gain_fraction"]),
        "safety": bool(safety_checks) and all(safety_checks.values()),
    }
    passed = all(checks.values())
    return {
        "schema_version": "ttm_quick_gate_v1",
        "locked_before_quick": True,
        "passed": passed,
        "decision": (
            "freeze_full_candidate_configuration"
            if passed
            else "stop_ttm_no_full_no_adapter_no_retune"
        ),
        "checks": checks,
        "thresholds": dict(gate_config),
        "diagnostics": {
            "incumbent_rmse": incumbent_rmse,
            "random_frozen_rmse": random_rmse,
            "pretrained_frozen_rmse": pretrained_rmse,
            "pretrained_minus_random_delta_rmse": pretrained_rmse - random_rmse,
            "transfer_improvement_ft": transfer_improvement,
            "pretrained_delta_vs_incumbent_ft": pretrained_delta_incumbent,
            "bootstrap": transfer,
            "fold_transfer_deltas": {
                str(int(fold)): float(value)
                for fold, value in fold_deltas.items()
            },
            "directionally_consistent_folds": consistent_folds,
            "slope_improvement_ft": slope_improvement,
            "shape_improvement_ft": shape_improvement,
            "control_degradation_vs_correct_pretrained": control_degradation,
            "removed_transfer_gain_fraction": removed_fraction,
            "tail_checks": tail_checks,
            "safety_checks": dict(safety_checks),
        },
        "adapter_allowed_for_future_new_protocol": passed,
        "adapter_executed_this_round": False,
        "full_executed_by_agent": False,
    }


__all__ = [
    "CONTROL_COLUMNS",
    "INCUMBENT_COLUMN",
    "PRETRAINED_COLUMN",
    "RANDOM_COLUMN",
    "evaluate_ttm_oof",
    "locked_quick_gate",
    "paired_transfer_bootstrap",
    "prediction_column",
    "seed_prediction_column",
]
