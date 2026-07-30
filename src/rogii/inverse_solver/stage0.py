"""Stage 0 target-free observability metrics and locked gate."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import PrefixCalibrationConfig


CONTROL_COST_COLUMNS = {
    "wrong": "wrong_cost",
    "phase": "phase_cost",
    "shuffle": "shuffle_cost",
}


def paired_mean_bootstrap(
    delta: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires a finite one-dimensional vector")
    if replicates < 100:
        raise ValueError("paired bootstrap requires at least 100 replicates")
    random = np.random.default_rng(int(seed))
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 512):
        stop = min(start + 512, replicates)
        indices = random.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = np.mean(values[indices], axis=1)
    lower, median, upper = np.quantile(estimates, (0.025, 0.5, 0.975))
    return {
        "n_wells": len(values),
        "replicates": int(replicates),
        "seed": int(seed),
        "point_mean": float(np.mean(values)),
        "ci_lower": float(lower),
        "ci_median": float(median),
        "ci_upper": float(upper),
        "probability_below_zero": float(np.mean(estimates < 0.0)),
    }


def _residualized_intercept(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    denominator = float(x_centered @ x_centered)
    slope = float(x_centered @ (y - float(np.mean(y))) / denominator) if denominator > 1.0e-15 else 0.0
    return float(np.mean(y) - slope * np.mean(x))


def nuisance_residualized_bootstrap(
    nuisance_delta: np.ndarray,
    likelihood_delta: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    x = np.asarray(nuisance_delta, dtype=np.float64)
    y = np.asarray(likelihood_delta, dtype=np.float64)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or len(x) < 3
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("nuisance residualization inputs are invalid")
    random = np.random.default_rng(int(seed))
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = random.integers(0, len(x), size=len(x))
        estimates[index] = _residualized_intercept(x[selected], y[selected])
    lower, median, upper = np.quantile(estimates, (0.025, 0.5, 0.975))
    correlation = (
        float(np.corrcoef(x, y)[0, 1])
        if float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0
        else 0.0
    )
    return {
        "n_wells": len(x),
        "replicates": int(replicates),
        "seed": int(seed),
        "point_intercept_at_matched_nuisance": _residualized_intercept(x, y),
        "ci_lower": float(lower),
        "ci_median": float(median),
        "ci_upper": float(upper),
        "nuisance_likelihood_delta_correlation": correlation,
    }


def audit_stage0_frame(
    frame: pd.DataFrame,
    *,
    config: PrefixCalibrationConfig,
    expected_wells: int,
    failure_count: int = 0,
) -> dict[str, Any]:
    required = {
        "well_id",
        "fold",
        "correct_cost",
        "wrong_cost",
        "phase_cost",
        "shuffle_cost",
        "correct_coherence",
        "correct_effective_rows",
        "correct_reliability",
        "correct_nuisance_cost",
        "wrong_nuisance_cost",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Stage 0 frame is missing columns: {sorted(missing)}")
    if (
        frame["well_id"].astype(str).duplicated().any()
        or len(frame) == 0
        or frame[list(required - {"well_id"})].select_dtypes(include=[np.number]).isna().any().any()
    ):
        raise ValueError("Stage 0 frame contains duplicate IDs or missing numerics")
    numeric_columns = [
        "correct_cost",
        "wrong_cost",
        "phase_cost",
        "shuffle_cost",
        "correct_coherence",
        "correct_effective_rows",
        "correct_reliability",
        "correct_nuisance_cost",
        "wrong_nuisance_cost",
    ]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Stage 0 frame contains nonfinite values")

    correct = frame["correct_cost"].to_numpy(dtype=np.float64)
    comparisons: dict[str, Any] = {}
    for index, (name, column) in enumerate(CONTROL_COST_COLUMNS.items()):
        delta = correct - frame[column].to_numpy(dtype=np.float64)
        comparisons[name] = {
            "correct_wins": int(np.sum(delta < 0.0)),
            "correct_ties": int(np.sum(delta == 0.0)),
            "correct_losses": int(np.sum(delta > 0.0)),
            "correct_win_fraction": float(np.mean(delta < 0.0)),
            "bootstrap": paired_mean_bootstrap(
                delta,
                replicates=config.bootstrap_replicates,
                seed=config.bootstrap_seed + index,
            ),
        }
    nuisance_delta = (
        frame["correct_nuisance_cost"].to_numpy(dtype=np.float64)
        - frame["wrong_nuisance_cost"].to_numpy(dtype=np.float64)
    )
    likelihood_delta = correct - frame["wrong_cost"].to_numpy(dtype=np.float64)
    nuisance = {
        "correct_preferred_fraction": float(np.mean(nuisance_delta < 0.0)),
        "mean_correct_minus_wrong_nuisance_cost": float(np.mean(nuisance_delta)),
        "residualized": nuisance_residualized_bootstrap(
            nuisance_delta,
            likelihood_delta,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed + 17,
        ),
    }
    stacked = np.column_stack(
        [
            correct,
            frame["wrong_cost"].to_numpy(dtype=np.float64),
            frame["phase_cost"].to_numpy(dtype=np.float64),
            frame["shuffle_cost"].to_numpy(dtype=np.float64),
        ]
    )
    correct_rank = 1 + np.sum(stacked[:, 1:] < stacked[:, [0]], axis=1)
    checks = {
        "all_training_wells_audited": bool(
            len(frame) == int(expected_wells) and int(failure_count) == 0
        ),
        "correct_beats_wrong_at_least_65pct": bool(
            comparisons["wrong"]["correct_win_fraction"]
            >= config.stage0_correct_win_fraction
        ),
        "correct_minus_wrong_ci_upper_below_zero": bool(
            comparisons["wrong"]["bootstrap"]["ci_upper"] < 0.0
        ),
        "correct_minus_phase_ci_upper_below_zero": bool(
            comparisons["phase"]["bootstrap"]["ci_upper"] < 0.0
        ),
        "correct_minus_shuffle_ci_upper_below_zero": bool(
            comparisons["shuffle"]["bootstrap"]["ci_upper"] < 0.0
        ),
        "nuisance_residualized_ci_upper_below_zero": bool(
            nuisance["residualized"]["ci_upper"] < 0.0
        ),
        "nuisance_only_preference_below_65pct": bool(
            nuisance["correct_preferred_fraction"]
            < config.stage0_correct_win_fraction
        ),
    }
    return {
        "schema_version": "prefix_inverse_stage0_audit_v1",
        "population": {
            "expected_wells": int(expected_wells),
            "audited_wells": len(frame),
            "failed_wells": int(failure_count),
            "fold_counts": {
                str(int(key)): int(value)
                for key, value in frame.groupby("fold", sort=True).size().items()
            },
        },
        "cost_direction": "lower_is_better",
        "comparisons": comparisons,
        "nuisance_only_baseline": nuisance,
        "correct_rank": {
            "mean": float(np.mean(correct_rank)),
            "median": float(np.median(correct_rank)),
            "rank_1_fraction": float(np.mean(correct_rank == 1)),
        },
        "correct_observability": {
            "mean_cost": float(np.mean(correct)),
            "mean_coherence": float(np.mean(frame["correct_coherence"])),
            "mean_reliability": float(np.mean(frame["correct_reliability"])),
            "median_effective_rows": float(np.median(frame["correct_effective_rows"])),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": (
            "proceed_to_canonical_quick"
            if all(checks.values())
            else "stop_rogii_modeling_no_competitive_signal"
        ),
    }
