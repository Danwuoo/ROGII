"""Locked sparse-anchor local-spike QC and outlier-rejection diagnostic.

This is a target-bearing, acausal counterfactual experiment.  Anchor placement,
the fixed P3/Path-PCA blend, error realizations, threshold, reject limit, and
piecewise reconstruction are all locked.  The operational QC function never
accepts true targets, simulated outlier indicators, or realization seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import enumerate_wells
from ..folds import load_fold_mapping
from .anchor_error_realism import (
    AnchorLatents,
    ErrorScenario,
    METRIC_TOLERANCE,
    PER_WELL_SSE_TOLERANCE,
    PERSISTENT_FOLD_FAILURE_RATE,
    PERSISTENT_FOLD_RMSE_THRESHOLD,
    ROWWISE_TOLERANCE,
    _mask_coefficients,
    _scenario_metadata,
    scenario_anchor_errors,
    stable_anchor_latents,
)
from .anchor_noise_robustness import (
    DEFAULT_NOISE_SEEDS,
    LOCKED_CONFIGS,
    LOCKED_EXACT_METRICS,
    NoiseWellContext,
    _build_noise_contexts,
    _read_blend_oof,
    _selected_wells,
    _validate_locked_selection,
    _validate_seed,
    condition_noisy_piecewise,
    piecewise_noise_weights,
)
from .sparse_anchor_voi import (
    AnchorSelection,
    FOLD_SHA256,
    FOLD_VERSION,
    _DeterministicGzipCsvWriter,
    _atomic_json,
    _markdown_table,
    anchor_mask,
    safe_output_directory,
)
from .terminal_anchor_frontier import (
    _boundary_md,
    frontier_segment_regions,
    select_terminal_anchors,
)


QC_THRESHOLD = 7.5
QC_MAX_REJECTS = 2
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260727
SEGMENTS = ("pre_first", "between", "post_last")
METHODS = (
    "no_qc",
    "local_spike_reject_7p5",
    "oracle_indicator_reject",
    "oracle_clean_values",
)
PRIMARY_METHOD = "local_spike_reject_7p5"
CONFIGS = tuple(
    config for config in LOCKED_CONFIGS if config.name in {"near_k6", "td_k6"}
)


def locked_qc_scenarios() -> tuple[ErrorScenario, ...]:
    """Return the complete fixed 11-scenario QC matrix."""
    return (
        ErrorScenario("control", "exact_clean", 0.0, 0.0, 0.0, 0.0, 0.0),
        ErrorScenario(
            "control", "gaussian_1_clean", 0.0, 1.0, 0.0, 0.0, 1.0
        ),
        ErrorScenario(
            "control",
            "moderate_background_no_outlier",
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
        ),
        ErrorScenario(
            "control", "moderate_m5_control", 1.0, 1.0, 0.10, 5.0, 1.0
        ),
        ErrorScenario(
            "outlier_only", "outlier_p05_m10", 0.0, 0.0, 0.05, 10.0, 0.0
        ),
        ErrorScenario(
            "outlier_only", "outlier_p10_m10", 0.0, 0.0, 0.10, 10.0, 0.0
        ),
        ErrorScenario(
            "outlier_only", "outlier_p20_m10", 0.0, 0.0, 0.20, 10.0, 0.0
        ),
        ErrorScenario(
            "operational",
            "moderate_background_p05_m10",
            1.0,
            1.0,
            0.05,
            10.0,
            1.0,
        ),
        ErrorScenario(
            "operational",
            "moderate_background_p10_m10",
            1.0,
            1.0,
            0.10,
            10.0,
            1.0,
        ),
        ErrorScenario(
            "stress",
            "severe_background_no_outlier",
            2.0,
            2.0,
            0.0,
            0.0,
            2.0,
        ),
        ErrorScenario(
            "stress", "severe", 2.0, 2.0, 0.10, 10.0, 2.0
        ),
    )


QC_SCENARIOS = locked_qc_scenarios()


def select_qc_scenarios(
    names: Sequence[str] | None,
) -> tuple[ErrorScenario, ...]:
    if names is None:
        return QC_SCENARIOS
    requested = tuple(dict.fromkeys(map(str, names)))
    by_name = {scenario.name: scenario for scenario in QC_SCENARIOS}
    if not requested or any(name not in by_name for name in requested):
        raise ValueError(
            "scenario names must be a non-empty subset of the locked QC matrix"
        )
    return tuple(by_name[name] for name in requested)


def validate_qc_contract(threshold: float, maximum_rejects: int) -> None:
    """Fail closed if the pre-fixed operational constants are changed."""
    if (
        isinstance(threshold, (bool, np.bool_))
        or not np.isfinite(float(threshold))
        or float(threshold) != QC_THRESHOLD
    ):
        raise ValueError("QC threshold is locked at 7.5 ft")
    if (
        isinstance(maximum_rejects, (bool, np.bool_))
        or not isinstance(maximum_rejects, (int, np.integer))
        or int(maximum_rejects) != QC_MAX_REJECTS
    ):
        raise ValueError("QC maximum reject count is locked at 2")


def _validate_anchor_qc_inputs(
    normalized_anchor_positions: np.ndarray,
    anchor_residuals: np.ndarray,
    accepted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(normalized_anchor_positions, dtype=np.float64)
    residuals = np.asarray(anchor_residuals, dtype=np.float64)
    retained = np.asarray(accepted, dtype=bool)
    if (
        positions.ndim != 1
        or residuals.shape != positions.shape
        or retained.shape != positions.shape
        or len(positions) < 3
        or not np.isfinite(positions).all()
        or not np.isfinite(residuals).all()
        or np.any(np.diff(positions) <= 0)
        or int(retained.sum()) < 3
    ):
        raise ValueError(
            "local-spike QC requires aligned finite increasing positions and "
            "at least three accepted anchors"
        )
    return positions, residuals, retained


def local_leave_one_out_scores(
    normalized_anchor_positions: np.ndarray,
    anchor_residuals: np.ndarray,
    accepted: np.ndarray | None = None,
) -> np.ndarray:
    """Compute local leave-one-out innovation scores without the PS boundary."""
    positions = np.asarray(normalized_anchor_positions, dtype=np.float64)
    retained = (
        np.ones(len(positions), dtype=bool)
        if accepted is None
        else np.asarray(accepted, dtype=bool)
    )
    positions, residuals, retained = _validate_anchor_qc_inputs(
        positions, anchor_residuals, retained
    )
    kept = np.flatnonzero(retained)
    scores = np.full(len(positions), np.nan, dtype=np.float64)
    for order, anchor in enumerate(kept):
        if order == 0:
            left, right = int(kept[1]), int(kept[2])
        elif order == len(kept) - 1:
            left, right = int(kept[-3]), int(kept[-2])
        else:
            left, right = int(kept[order - 1]), int(kept[order + 1])
        fraction = (
            positions[anchor] - positions[left]
        ) / (positions[right] - positions[left])
        predicted = (
            (1.0 - fraction) * residuals[left]
            + fraction * residuals[right]
        )
        scores[anchor] = abs(residuals[anchor] - predicted)
    return scores


@dataclass(frozen=True)
class LocalSpikeDecision:
    accepted: np.ndarray
    rejected: np.ndarray
    score_at_decision: np.ndarray
    initial_score: np.ndarray
    decision_iteration: np.ndarray

    @property
    def accepted_count(self) -> int:
        return int(self.accepted.sum())


def local_spike_reject_7p5(
    observed_anchor_tvt: np.ndarray,
    base_anchor_prediction: np.ndarray,
    normalized_anchor_positions: np.ndarray,
) -> LocalSpikeDecision:
    """Apply the one locked operational QC rule.

    The signature deliberately contains no true TVT, indicator, sign, seed, or
    scored per-well error.
    """
    validate_qc_contract(QC_THRESHOLD, QC_MAX_REJECTS)
    observed = np.asarray(observed_anchor_tvt, dtype=np.float64)
    base = np.asarray(base_anchor_prediction, dtype=np.float64)
    positions = np.asarray(normalized_anchor_positions, dtype=np.float64)
    if (
        observed.shape != base.shape
        or positions.shape != observed.shape
        or observed.ndim != 1
        or len(observed) < 3
        or not np.isfinite(observed).all()
        or not np.isfinite(base).all()
    ):
        raise ValueError("QC anchor observations/base/positions must align")
    residuals = observed - base
    accepted = np.ones(len(observed), dtype=bool)
    initial = local_leave_one_out_scores(positions, residuals, accepted)
    score_at_decision = np.full(len(observed), np.nan, dtype=np.float64)
    iteration_at_decision = np.zeros(len(observed), dtype=np.int8)
    for iteration in range(1, QC_MAX_REJECTS + 1):
        scores = local_leave_one_out_scores(positions, residuals, accepted)
        maximum = float(np.nanmax(scores))
        if maximum <= QC_THRESHOLD:
            score_at_decision[accepted] = scores[accepted]
            iteration_at_decision[accepted] = iteration
            break
        # np.nanargmax returns the first anchor order on an exact tie.
        rejected_anchor = int(np.nanargmax(scores))
        score_at_decision[rejected_anchor] = scores[rejected_anchor]
        iteration_at_decision[rejected_anchor] = iteration
        accepted[rejected_anchor] = False
    if int(accepted.sum()) < 3:
        raise ValueError("QC failed closed with fewer than three anchors")
    remaining = np.isnan(score_at_decision) & accepted
    if np.any(remaining):
        final_scores = local_leave_one_out_scores(
            positions, residuals, accepted
        )
        score_at_decision[remaining] = final_scores[remaining]
        iteration_at_decision[remaining] = QC_MAX_REJECTS + 1
    return LocalSpikeDecision(
        accepted=accepted,
        rejected=~accepted,
        score_at_decision=score_at_decision,
        initial_score=initial,
        decision_iteration=iteration_at_decision,
    )


def _subset_selection(
    selection: AnchorSelection, accepted: np.ndarray
) -> AnchorSelection:
    retained = np.asarray(accepted, dtype=bool)
    if retained.shape != (selection.achieved_k,):
        raise ValueError("accepted mask does not align to selected anchors")
    return AnchorSelection(
        policy=selection.policy,
        requested_k=int(retained.sum()),
        local_indices=selection.local_indices[retained].copy(),
        normalized_positions=selection.normalized_positions[retained].copy(),
        requested_positions=selection.requested_positions[retained].copy(),
        acquisition_scores=selection.acquisition_scores[retained].copy(),
        minimum_spacing=selection.minimum_spacing,
    )


def condition_with_accepted_anchors(
    base_prediction: np.ndarray,
    observed_anchor_tvt: np.ndarray,
    normalized_positions: np.ndarray,
    selection: AnchorSelection,
    accepted: np.ndarray,
    *,
    fail_below_three: bool = True,
) -> np.ndarray:
    """Reconstruct using accepted knots; rejected rows are not forced."""
    retained = np.asarray(accepted, dtype=bool)
    if retained.shape != (selection.achieved_k,):
        raise ValueError("accepted mask does not align to selected anchors")
    if fail_below_three and int(retained.sum()) < 3:
        raise ValueError("conditioner failed closed below three anchors")
    subset = _subset_selection(selection, retained)
    observed = np.asarray(observed_anchor_tvt, dtype=np.float64)
    if observed.shape != (selection.achieved_k,):
        raise ValueError("observed anchor values do not align")
    if not subset.achieved_k:
        return np.asarray(base_prediction, dtype=np.float64).copy()
    return condition_noisy_piecewise(
        base_prediction,
        observed[retained],
        normalized_positions,
        subset,
    )


def anchor_confusion_counts(
    rejected: np.ndarray, true_outlier_indicator: np.ndarray
) -> dict[str, int]:
    """Return an anchor-level confusion matrix for post-decision scoring."""
    decision = np.asarray(rejected, dtype=bool)
    truth = np.asarray(true_outlier_indicator, dtype=bool)
    if decision.shape != truth.shape:
        raise ValueError("decision and indicator arrays must align")
    return {
        "tp": int(np.sum(decision & truth)),
        "fp": int(np.sum(decision & ~truth)),
        "fn": int(np.sum(~decision & truth)),
        "tn": int(np.sum(~decision & ~truth)),
    }


def reconstruct_pooled_metrics(
    per_well: pd.DataFrame,
) -> tuple[float, float]:
    rows = int(per_well["rows"].sum())
    non_rows = int(per_well["non_anchor_rows"].sum())
    if rows <= 0 or non_rows <= 0:
        raise ValueError("per-well row counts must be positive")
    return (
        float(np.sqrt(per_well["sse"].sum() / rows)),
        float(np.sqrt(per_well["non_anchor_sse"].sum() / non_rows)),
    )


def _score_coefficient_matrix(
    positions: np.ndarray, accepted_mask: int
) -> np.ndarray:
    accepted = np.asarray(
        [bool(accepted_mask & (1 << order)) for order in range(len(positions))]
    )
    if int(accepted.sum()) < 3:
        raise ValueError("score coefficients require at least three anchors")
    kept = np.flatnonzero(accepted)
    coefficients = np.zeros((len(positions), len(positions)), dtype=np.float64)
    for order, anchor in enumerate(kept):
        if order == 0:
            left, right = int(kept[1]), int(kept[2])
        elif order == len(kept) - 1:
            left, right = int(kept[-3]), int(kept[-2])
        else:
            left, right = int(kept[order - 1]), int(kept[order + 1])
        fraction = (
            positions[anchor] - positions[left]
        ) / (positions[right] - positions[left])
        coefficients[anchor, anchor] = 1.0
        coefficients[anchor, left] = -(1.0 - fraction)
        coefficients[anchor, right] = -fraction
    coefficients[~accepted] = np.nan
    return coefficients


def _batch_local_spike_decisions(
    positions: np.ndarray, residuals: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorized locked QC; indicators are intentionally not arguments."""
    values = np.asarray(residuals, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(positions)
        or values.shape[1] != 6
        or not np.isfinite(values).all()
    ):
        raise ValueError("batch QC requires finite (group, 6) residuals")
    full_mask = (1 << values.shape[1]) - 1
    masks = np.full(len(values), full_mask, dtype=np.uint8)
    rejected = np.zeros_like(values, dtype=bool)
    score_at = np.full_like(values, np.nan)
    decision_iteration = np.zeros_like(values, dtype=np.int8)
    score_matrices = {
        mask: _score_coefficient_matrix(positions, mask)
        for mask in range(1 << values.shape[1])
        if int(mask).bit_count() >= 3
    }
    initial = np.abs(values @ score_matrices[full_mask].T)
    for iteration in range(1, QC_MAX_REJECTS + 1):
        for mask in np.unique(masks):
            group = np.flatnonzero(masks == mask)
            coefficients = score_matrices[int(mask)]
            scores = np.abs(values[group] @ coefficients.T)
            score_for_argmax = np.where(np.isnan(scores), -np.inf, scores)
            maxima = np.max(score_for_argmax, axis=1)
            anchors = np.argmax(score_for_argmax, axis=1)
            to_reject = maxima > QC_THRESHOLD
            if np.any(~to_reject):
                stopped = group[~to_reject]
                score_at[stopped] = np.where(
                    np.isnan(score_at[stopped]),
                    scores[~to_reject],
                    score_at[stopped],
                )
                undecided = decision_iteration[stopped] == 0
                decision_iteration[stopped] = np.where(
                    undecided, iteration, decision_iteration[stopped]
                )
            if np.any(to_reject):
                selected_groups = group[to_reject]
                selected_anchors = anchors[to_reject]
                score_at[selected_groups, selected_anchors] = maxima[to_reject]
                decision_iteration[
                    selected_groups, selected_anchors
                ] = iteration
                rejected[selected_groups, selected_anchors] = True
                clear_bits = np.bitwise_not(
                    np.left_shift(
                        np.uint8(1), selected_anchors.astype(np.uint8)
                    )
                )
                masks[selected_groups] &= clear_bits
    for mask in np.unique(masks):
        group = np.flatnonzero(masks == mask)
        final_scores = np.abs(values[group] @ score_matrices[int(mask)].T)
        missing = np.isnan(score_at[group])
        score_at[group] = np.where(missing, final_scores, score_at[group])
        iteration_block = decision_iteration[group]
        iteration_block[missing] = QC_MAX_REJECTS + 1
        decision_iteration[group] = iteration_block
    accepted_count = np.asarray(
        [int(mask).bit_count() for mask in masks], dtype=np.int8
    )
    if np.any(accepted_count < 3):
        raise ValueError("batch QC failed closed below three anchors")
    return {
        "accepted_mask": masks,
        "rejected": rejected,
        "score_at_decision": score_at,
        "initial_score": initial,
        "decision_iteration": decision_iteration,
        "accepted_count": accepted_count,
    }


def _selection_from_mask(
    selection: AnchorSelection, accepted_mask: int
) -> tuple[AnchorSelection, np.ndarray]:
    accepted = np.asarray(
        [
            bool(accepted_mask & (1 << order))
            for order in range(selection.achieved_k)
        ]
    )
    return _subset_selection(selection, accepted), accepted


def _coefficients_for_subset(
    context: NoiseWellContext,
    selection: AnchorSelection,
    true_anchors: np.ndarray,
    accepted_mask: int,
    masks: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    subset, accepted = _selection_from_mask(selection, accepted_mask)
    if subset.achieved_k:
        exact_prediction = condition_noisy_piecewise(
            context.base_prediction,
            true_anchors[accepted],
            context.normalized_md,
            subset,
        )
        subset_weights = piecewise_noise_weights(
            context.normalized_md, subset
        )
        weights = np.zeros(
            (len(context.row_index), selection.achieved_k), dtype=np.float64
        )
        weights[:, accepted] = subset_weights
    else:
        exact_prediction = context.base_prediction.copy()
        weights = np.zeros(
            (len(context.row_index), selection.achieved_k), dtype=np.float64
        )
    coefficients = _mask_coefficients(
        exact_prediction - context.y_true, weights, masks
    )
    return coefficients, exact_prediction, weights


def _evaluate_subsets(
    subset_masks: np.ndarray,
    anchor_errors: np.ndarray,
    coefficient_cache: Mapping[
        int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> np.ndarray:
    masks = np.asarray(subset_masks, dtype=np.uint8)
    errors = np.asarray(anchor_errors, dtype=np.float64)
    if errors.ndim != 2 or len(masks) != len(errors):
        raise ValueError("subset masks and error rows must align")
    mask_metric_count = next(iter(coefficient_cache.values()))[0].shape[0]
    result = np.empty((len(errors), mask_metric_count), dtype=np.float64)
    for subset in np.unique(masks):
        group = np.flatnonzero(masks == subset)
        exact, linear, gram, _ = coefficient_cache[int(subset)]
        selected_errors = errors[group]
        cross = selected_errors @ linear.T
        quadratic = np.einsum(
            "gi,mij,gj->gm",
            selected_errors,
            gram,
            selected_errors,
            optimize=True,
        )
        values = exact[None, :] + 2.0 * cross + quadratic
        tolerance = (
            np.finfo(np.float64).eps
            * max(1.0, float(np.max(np.abs(exact))))
            * 64
        )
        if float(np.min(values)) < -tolerance:
            raise ValueError("subset SSE propagation became negative")
        result[group] = np.maximum(values, 0.0)
    return result


def hierarchical_qc_bootstrap(
    full_sse: np.ndarray,
    non_anchor_sse: np.ndarray,
    rows: np.ndarray,
    non_anchor_rows: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Paired hierarchical well/error bootstrap across all methods/scenarios."""
    full = np.asarray(full_sse, dtype=np.float64)
    non = np.asarray(non_anchor_sse, dtype=np.float64)
    row_counts = np.asarray(rows, dtype=np.int64)
    non_counts = np.asarray(non_anchor_rows, dtype=np.int64)
    bootstrap_seed = _validate_seed(seed)
    if (
        full.ndim != 5
        or non.shape != full.shape
        or row_counts.shape != (full.shape[4],)
        or non_counts.shape != (full.shape[0], full.shape[4])
        or samples < 1
        or np.any(row_counts <= 0)
        or np.any(non_counts <= 0)
        or not np.isfinite(full).all()
        or not np.isfinite(non).all()
    ):
        raise ValueError("invalid hierarchical QC bootstrap inputs")
    config_count, method_count, scenario_count, seed_count, well_count = (
        full.shape
    )
    rng = np.random.default_rng(bootstrap_seed)
    well_draws = rng.integers(
        0, well_count, size=(samples, well_count), dtype=np.int32
    )
    realization_draws = rng.integers(
        0, seed_count, size=(samples, well_count), dtype=np.int16
    )
    sampled_rows = row_counts[well_draws].sum(axis=1, dtype=np.int64)
    output: list[dict[str, Any]] = []

    def quantiles(values: np.ndarray) -> dict[str, float]:
        return {
            "q025": float(np.quantile(values, 0.025)),
            "q50": float(np.quantile(values, 0.50)),
            "q975": float(np.quantile(values, 0.975)),
        }

    for config_index in range(config_count):
        sampled_non_rows = non_counts[config_index][well_draws].sum(
            axis=1, dtype=np.int64
        )
        for method_index in range(method_count):
            for scenario_index in range(scenario_count):
                group_full = full[config_index, method_index, scenario_index]
                group_non = non[config_index, method_index, scenario_index]
                hierarchical_full = np.sqrt(
                    group_full[realization_draws, well_draws].sum(axis=1)
                    / sampled_rows
                )
                hierarchical_non = np.sqrt(
                    group_non[realization_draws, well_draws].sum(axis=1)
                    / sampled_non_rows
                )
                fixed_full = np.sqrt(
                    group_full.sum(axis=1) / row_counts.sum()
                )
                fixed_non = np.sqrt(
                    group_non.sum(axis=1)
                    / non_counts[config_index].sum()
                )
                output.append(
                    {
                        "config_index": config_index,
                        "method_index": method_index,
                        "scenario_index": scenario_index,
                        "fixed_well_error_realization": {
                            "full_rmse": quantiles(fixed_full),
                            "non_anchor_rmse": quantiles(fixed_non),
                        },
                        "hierarchical": {
                            "full_rmse": quantiles(hierarchical_full),
                            "non_anchor_rmse": quantiles(hierarchical_non),
                        },
                    }
                )
    return output


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return (
        float(numerator) / float(denominator)
        if float(denominator) != 0.0
        else float("nan")
    )


def _detection_rows_from_decisions(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate post-decision truth scoring and score distributions."""
    group_keys = ["config", "scenario"]
    records: list[dict[str, Any]] = []
    for (config, scenario), group in decisions.groupby(group_keys, sort=True):
        truth = group["true_outlier_indicator"].to_numpy(dtype=bool)
        rejected = group["rejected"].to_numpy(dtype=bool)
        counts = anchor_confusion_counts(rejected, truth)
        tp, fp, fn, tn = (
            counts["tp"],
            counts["fp"],
            counts["fn"],
            counts["tn"],
        )
        well_truth = group.groupby(
            ["realization_seed", "well_id"], sort=False
        )["true_outlier_indicator"].any()
        well_decision = group.groupby(
            ["realization_seed", "well_id"], sort=False
        )["rejected"].any()
        well_tp = int((well_truth & well_decision).sum())
        well_fp = int((~well_truth & well_decision).sum())
        well_fn = int((well_truth & ~well_decision).sum())
        records.append(
            {
                "config": config,
                "scenario": scenario,
                "scope": "overall",
                "subset": "all_anchors",
                **counts,
                "precision": _safe_ratio(tp, tp + fp),
                "recall": _safe_ratio(tp, tp + fn),
                "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
                "false_positive_rate": _safe_ratio(fp, fp + tn),
                "specificity": _safe_ratio(tn, fp + tn),
                "well_detection_precision": _safe_ratio(
                    well_tp, well_tp + well_fp
                ),
                "well_detection_recall": _safe_ratio(
                    well_tp, well_tp + well_fn
                ),
                "anchors": len(group),
                "score_q05": float(
                    group["qc_score_at_decision"].quantile(0.05)
                ),
                "score_q50": float(
                    group["qc_score_at_decision"].quantile(0.50)
                ),
                "score_q95": float(
                    group["qc_score_at_decision"].quantile(0.95)
                ),
            }
        )

        subset_specs: list[tuple[str, str, pd.Series]] = []
        for order in range(1, 7):
            subset_specs.append(
                (
                    "anchor_order",
                    str(order),
                    group["anchor_order"].eq(order),
                )
            )
        subset_specs.extend(
            (
                (
                    "anchor_region",
                    "endpoint",
                    group["anchor_order"].isin([1, 6]),
                ),
                (
                    "anchor_region",
                    "interior",
                    group["anchor_order"].between(2, 5),
                ),
                (
                    "shock_sign",
                    "positive",
                    group["true_outlier_sign"].eq(1)
                    & group["true_outlier_indicator"],
                ),
                (
                    "shock_sign",
                    "negative",
                    group["true_outlier_sign"].eq(-1)
                    & group["true_outlier_indicator"],
                ),
                (
                    "outlier_multiplicity",
                    "isolated",
                    group["well_outlier_count"].eq(1),
                ),
                (
                    "outlier_multiplicity",
                    "multiple",
                    group["well_outlier_count"].ge(2),
                ),
            )
        )
        for scope, subset, selected in subset_specs:
            block = group[selected]
            if not len(block):
                continue
            block_truth = block["true_outlier_indicator"].to_numpy(dtype=bool)
            block_rejected = block["rejected"].to_numpy(dtype=bool)
            block_counts = anchor_confusion_counts(
                block_rejected, block_truth
            )
            btp, bfp, bfn, btn = (
                block_counts["tp"],
                block_counts["fp"],
                block_counts["fn"],
                block_counts["tn"],
            )
            records.append(
                {
                    "config": config,
                    "scenario": scenario,
                    "scope": scope,
                    "subset": subset,
                    **block_counts,
                    "precision": _safe_ratio(btp, btp + bfp),
                    "recall": _safe_ratio(btp, btp + bfn),
                    "f1": _safe_ratio(2 * btp, 2 * btp + bfp + bfn),
                    "false_positive_rate": _safe_ratio(bfp, bfp + btn),
                    "specificity": _safe_ratio(btn, bfp + btn),
                    "well_detection_precision": float("nan"),
                    "well_detection_recall": float("nan"),
                    "anchors": len(block),
                    "score_q05": float(
                        block["qc_score_at_decision"].quantile(0.05)
                    ),
                    "score_q50": float(
                        block["qc_score_at_decision"].quantile(0.50)
                    ),
                    "score_q95": float(
                        block["qc_score_at_decision"].quantile(0.95)
                    ),
                }
            )

        contribution = group["true_outlier_contribution"].abs()
        score_classes = (
            ("clean", contribution.eq(0.0)),
            ("m5", contribution.eq(5.0)),
            ("m10", contribution.eq(10.0)),
        )
        for label, selected in score_classes:
            block = group[selected]
            if not len(block):
                continue
            records.append(
                {
                    "config": config,
                    "scenario": scenario,
                    "scope": "score_distribution",
                    "subset": label,
                    "tp": 0,
                    "fp": int(block["rejected"].sum()),
                    "fn": 0,
                    "tn": int((~block["rejected"]).sum()),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "false_positive_rate": float(block["rejected"].mean()),
                    "specificity": float((~block["rejected"]).mean()),
                    "well_detection_precision": float("nan"),
                    "well_detection_recall": float("nan"),
                    "anchors": len(block),
                    "score_q05": float(
                        block["qc_score_at_decision"].quantile(0.05)
                    ),
                    "score_q50": float(
                        block["qc_score_at_decision"].quantile(0.50)
                    ),
                    "score_q95": float(
                        block["qc_score_at_decision"].quantile(0.95)
                    ),
                }
            )
    return pd.DataFrame(records)


def _tail_metrics(full_sse: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
    rmse = np.sqrt(full_sse / rows[None, None, None, None, :])
    worst_count = max(1, int(math.ceil(0.10 * len(rows))))
    return {
        "well_rmse": rmse,
        "p50": np.quantile(rmse, 0.50, axis=4),
        "p90": np.quantile(rmse, 0.90, axis=4),
        "p95": np.quantile(rmse, 0.95, axis=4),
        "max": np.max(rmse, axis=4),
        "wells_ge_40": np.sum(rmse >= 40.0, axis=4),
        "wells_ge_60": np.sum(rmse >= 60.0, axis=4),
        "worst10_sse_share": (
            np.sort(full_sse, axis=4)[..., -worst_count:].sum(axis=4)
            / full_sse.sum(axis=4)
        ),
    }


def _regression_reproduction(
    prior_artifact: Path,
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    well_ids: Sequence[str],
    full_sse: np.ndarray,
    non_sse: np.ndarray,
    rows: np.ndarray,
    non_rows: np.ndarray,
) -> dict[str, Any]:
    """Compare no-QC selected scenarios to the prior realism artifact."""
    mapping = {
        "exact_clean": "bias_tau_0",
        "moderate_m5_control": "moderate",
        "outlier_p10_m10": "outlier_p10_m10",
        "severe": "severe",
    }
    available = {
        scenario.name: index for index, scenario in enumerate(scenarios)
    }
    selected = {
        new: old for new, old in mapping.items() if new in available
    }
    old_seed = pd.read_csv(Path(prior_artifact) / "per_seed_metrics.csv")
    config_index = {config.name: index for index, config in enumerate(configs)}
    seed_index = {int(seed): index for index, seed in enumerate(seeds)}
    full_delta = 0.0
    non_delta = 0.0
    metric_groups = 0
    for new_name, old_name in selected.items():
        si = available[new_name]
        prior = old_seed[
            old_seed["scenario"].eq(old_name)
            & old_seed["config"].isin(config_index)
            & old_seed["realization_seed"].isin(seed_index)
        ]
        for row in prior.itertuples(index=False):
            ci = config_index[str(row.config)]
            ri = seed_index[int(row.realization_seed)]
            actual_full = float(
                np.sqrt(full_sse[ci, 0, si, ri].sum() / rows.sum())
            )
            actual_non = float(
                np.sqrt(
                    non_sse[ci, 0, si, ri].sum() / non_rows[ci].sum()
                )
            )
            full_delta = max(full_delta, abs(actual_full - float(row.full_rmse)))
            non_delta = max(
                non_delta, abs(actual_non - float(row.non_anchor_rmse))
            )
            metric_groups += 1
    expected_groups = len(selected) * len(configs) * len(seeds)

    well_position = {well_id: index for index, well_id in enumerate(well_ids)}
    old_to_new = {old: new for new, old in selected.items()}
    well_full_delta = 0.0
    well_non_delta = 0.0
    per_well_rows = 0
    usecols = [
        "config",
        "scenario",
        "realization_seed",
        "well_id",
        "sse",
        "non_anchor_sse",
    ]
    for chunk in pd.read_csv(
        Path(prior_artifact) / "per_well_seed_metrics.csv.gz",
        usecols=usecols,
        dtype={"well_id": "string"},
        chunksize=200_000,
    ):
        block = chunk[
            chunk["scenario"].isin(old_to_new)
            & chunk["config"].isin(config_index)
            & chunk["realization_seed"].isin(seed_index)
            & chunk["well_id"].isin(well_position)
        ]
        if not len(block):
            continue
        ci = block["config"].map(config_index).to_numpy(dtype=np.int64)
        si = (
            block["scenario"]
            .map(old_to_new)
            .map(available)
            .to_numpy(dtype=np.int64)
        )
        ri = block["realization_seed"].map(seed_index).to_numpy(dtype=np.int64)
        wi = block["well_id"].map(well_position).to_numpy(dtype=np.int64)
        actual_full = full_sse[ci, 0, si, ri, wi]
        actual_non = non_sse[ci, 0, si, ri, wi]
        well_full_delta = max(
            well_full_delta,
            float(
                np.max(
                    np.abs(actual_full - block["sse"].to_numpy(dtype=np.float64))
                )
            ),
        )
        well_non_delta = max(
            well_non_delta,
            float(
                np.max(
                    np.abs(
                        actual_non
                        - block["non_anchor_sse"].to_numpy(dtype=np.float64)
                    )
                )
            ),
        )
        per_well_rows += len(block)
    expected_per_well = expected_groups * len(well_ids)
    passed = bool(
        metric_groups == expected_groups
        and per_well_rows == expected_per_well
        and full_delta <= METRIC_TOLERANCE
        and non_delta <= METRIC_TOLERANCE
        and well_full_delta <= PER_WELL_SSE_TOLERANCE
        and well_non_delta <= PER_WELL_SSE_TOLERANCE
    )
    return {
        "passed": passed,
        "scenario_mapping": selected,
        "metric_groups": metric_groups,
        "expected_metric_groups": expected_groups,
        "per_well_rows": per_well_rows,
        "expected_per_well_rows": expected_per_well,
        "max_full_rmse_delta": full_delta,
        "max_non_anchor_rmse_delta": non_delta,
        "max_full_sse_delta": well_full_delta,
        "max_non_anchor_sse_delta": well_non_delta,
    }


def _information_curve(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    full_sse: np.ndarray,
    non_sse: np.ndarray,
    rows: np.ndarray,
    non_rows: np.ndarray,
    fold_sse: np.ndarray,
    fold_non_sse: np.ndarray,
    fold_rows: np.ndarray,
    fold_non_rows: np.ndarray,
    segment_sse: np.ndarray,
    segment_rows: np.ndarray,
    hierarchy: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    total_rows = int(rows.sum())
    full_rmse = np.sqrt(full_sse.sum(axis=4) / total_rows)
    non_rmse = np.sqrt(
        non_sse.sum(axis=4) / non_rows.sum(axis=1)[:, None, None, None]
    )
    fold_rmse = np.sqrt(
        fold_sse / fold_rows[None, None, None, None, :]
    )
    fold_non_rmse = np.sqrt(
        fold_non_sse
        / fold_non_rows[:, None, None, None, :]
    )
    tail = _tail_metrics(full_sse, rows)
    hierarchy_by_index = {
        (
            int(item["config_index"]),
            int(item["method_index"]),
            int(item["scenario_index"]),
        ): item
        for item in hierarchy
    }
    records: list[dict[str, Any]] = []
    no_qc_index = METHODS.index("no_qc")
    for ci, config in enumerate(configs):
        for mi, method in enumerate(METHODS):
            for si, scenario in enumerate(scenarios):
                full = full_rmse[ci, mi, si]
                non = non_rmse[ci, mi, si]
                fold_values = fold_rmse[ci, mi, si]
                fold_failure_rates = np.mean(
                    fold_values >= PERSISTENT_FOLD_RMSE_THRESHOLD, axis=0
                )
                persistent = bool(
                    np.any(
                        fold_failure_rates >= PERSISTENT_FOLD_FAILURE_RATE
                    )
                )
                bootstrap = hierarchy_by_index[(ci, mi, si)]["hierarchical"]
                hierarchical_full = bootstrap["full_rmse"]
                hierarchical_non = bootstrap["non_anchor_rmse"]
                wells_40 = tail["wells_ge_40"][ci, mi, si]
                wells_60 = tail["wells_ge_60"][ci, mi, si]
                catastrophic = bool(
                    int(np.max(wells_40)) > 0
                    or int(np.max(wells_60)) > 0
                )
                robust = bool(
                    np.quantile(full, 0.95) < 4.0
                    and np.quantile(non, 0.95) < 4.0
                    and hierarchical_full["q975"] < 4.0
                    and hierarchical_non["q975"] < 4.0
                    and not persistent
                    and not catastrophic
                )
                segment_quantiles: dict[str, float] = {}
                for segment_index, segment in enumerate(SEGMENTS):
                    count = int(segment_rows[ci, segment_index])
                    segment_quantiles[segment] = (
                        float(
                            np.quantile(
                                np.sqrt(
                                    segment_sse[
                                        ci, mi, si, :, segment_index
                                    ]
                                    / count
                                ),
                                0.50,
                            )
                        )
                        if count
                        else float("nan")
                    )
                no_qc_full = full_rmse[ci, no_qc_index, si]
                no_qc_non = non_rmse[ci, no_qc_index, si]
                records.append(
                    {
                        "config": config.name,
                        "policy": config.policy,
                        "k": config.k,
                        "fractions": config.fractions_text,
                        "method": method,
                        **_scenario_metadata(scenario),
                        "realizations": len(seeds),
                        "full_rmse_mean": float(np.mean(full)),
                        "full_rmse_q05": float(np.quantile(full, 0.05)),
                        "full_rmse_q50": float(np.quantile(full, 0.50)),
                        "full_rmse_q95": float(np.quantile(full, 0.95)),
                        "non_anchor_rmse_mean": float(np.mean(non)),
                        "non_anchor_rmse_q05": float(
                            np.quantile(non, 0.05)
                        ),
                        "non_anchor_rmse_q50": float(
                            np.quantile(non, 0.50)
                        ),
                        "non_anchor_rmse_q95": float(
                            np.quantile(non, 0.95)
                        ),
                        "full_q95_delta_vs_no_qc": float(
                            np.quantile(full, 0.95)
                            - np.quantile(no_qc_full, 0.95)
                        ),
                        "non_anchor_q95_delta_vs_no_qc": float(
                            np.quantile(non, 0.95)
                            - np.quantile(no_qc_non, 0.95)
                        ),
                        "paired_full_delta_vs_no_qc_q50": float(
                            np.quantile(full - no_qc_full, 0.50)
                        ),
                        "paired_full_delta_vs_no_qc_q95": float(
                            np.quantile(full - no_qc_full, 0.95)
                        ),
                        "paired_non_anchor_delta_vs_no_qc_q50": float(
                            np.quantile(non - no_qc_non, 0.50)
                        ),
                        "hierarchical_full_q025": float(
                            hierarchical_full["q025"]
                        ),
                        "hierarchical_full_q50": float(
                            hierarchical_full["q50"]
                        ),
                        "hierarchical_full_q975": float(
                            hierarchical_full["q975"]
                        ),
                        "hierarchical_non_anchor_q025": float(
                            hierarchical_non["q025"]
                        ),
                        "hierarchical_non_anchor_q50": float(
                            hierarchical_non["q50"]
                        ),
                        "hierarchical_non_anchor_q975": float(
                            hierarchical_non["q975"]
                        ),
                        "fold_max_q50": float(
                            np.quantile(fold_values.max(axis=1), 0.50)
                        ),
                        "fold_max_q95": float(
                            np.quantile(fold_values.max(axis=1), 0.95)
                        ),
                        "non_anchor_fold_max_q95": float(
                            np.quantile(
                                fold_non_rmse[ci, mi, si].max(axis=1), 0.95
                            )
                        ),
                        "well_rmse_p50_q50": float(
                            np.quantile(tail["p50"][ci, mi, si], 0.50)
                        ),
                        "well_rmse_p90_q50": float(
                            np.quantile(tail["p90"][ci, mi, si], 0.50)
                        ),
                        "well_rmse_p95_q50": float(
                            np.quantile(tail["p95"][ci, mi, si], 0.50)
                        ),
                        "max_well_rmse_q95": float(
                            np.quantile(tail["max"][ci, mi, si], 0.95)
                        ),
                        "worst10_sse_share_q50": float(
                            np.quantile(
                                tail["worst10_sse_share"][ci, mi, si], 0.50
                            )
                        ),
                        "pre_first_rmse_q50": segment_quantiles["pre_first"],
                        "between_rmse_q50": segment_quantiles["between"],
                        "post_last_rmse_q50": segment_quantiles["post_last"],
                        "persistent_fold_failure": persistent,
                        "maximum_fold_failure_rate": float(
                            np.max(fold_failure_rates)
                        ),
                        "maximum_wells_ge_40": int(np.max(wells_40)),
                        "maximum_wells_ge_60": int(np.max(wells_60)),
                        "catastrophic_tail_failure": catastrophic,
                        "robust_lt_4": robust,
                    }
                )
    arrays = {
        "full_rmse": full_rmse,
        "non_anchor_rmse": non_rmse,
        "fold_rmse": fold_rmse,
        **tail,
    }
    return pd.DataFrame(records), arrays


def _per_seed_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    arrays: Mapping[str, np.ndarray],
    accepted_count: np.ndarray,
    changed_rows: np.ndarray,
    outlier_count: np.ndarray,
    full_sse: np.ndarray,
    rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    method_index = {method: index for index, method in enumerate(METHODS)}
    for ci, config in enumerate(configs):
        for mi, method in enumerate(METHODS):
            for si, scenario in enumerate(scenarios):
                for ri, realization_seed in enumerate(seeds):
                    accepted = accepted_count[ci, mi, si, ri]
                    rejects = config.k - accepted
                    exposure = outlier_count[ci, si, ri]
                    record = {
                        "config": config.name,
                        "policy": config.policy,
                        "k": config.k,
                        "fractions": config.fractions_text,
                        "method": method,
                        **_scenario_metadata(scenario),
                        "realization_seed": int(realization_seed),
                        "full_rows": int(rows.sum()),
                        "non_anchor_rows": int(rows.sum() - config.k * len(rows)),
                        "full_rmse": float(
                            arrays["full_rmse"][ci, mi, si, ri]
                        ),
                        "non_anchor_rmse": float(
                            arrays["non_anchor_rmse"][ci, mi, si, ri]
                        ),
                        "fold_max_rmse": float(
                            arrays["fold_rmse"][ci, mi, si, ri].max()
                        ),
                        "well_rmse_p50": float(
                            arrays["p50"][ci, mi, si, ri]
                        ),
                        "well_rmse_p90": float(
                            arrays["p90"][ci, mi, si, ri]
                        ),
                        "well_rmse_p95": float(
                            arrays["p95"][ci, mi, si, ri]
                        ),
                        "well_rmse_max": float(
                            arrays["max"][ci, mi, si, ri]
                        ),
                        "wells_ge_40": int(
                            arrays["wells_ge_40"][ci, mi, si, ri]
                        ),
                        "wells_ge_60": int(
                            arrays["wells_ge_60"][ci, mi, si, ri]
                        ),
                        "worst10_sse_share": float(
                            arrays["worst10_sse_share"][ci, mi, si, ri]
                        ),
                        "accepted_anchor_count": int(accepted.sum()),
                        "rejected_anchor_count": int(rejects.sum()),
                        "wells_with_0_rejects": int(np.sum(rejects == 0)),
                        "wells_with_1_reject": int(np.sum(rejects == 1)),
                        "wells_with_2_rejects": int(np.sum(rejects == 2)),
                        "wells_with_gt2_rejects": int(np.sum(rejects > 2)),
                        "rows_changed_by_qc": int(
                            changed_rows[ci, mi, si, ri].sum()
                        ),
                        "true_outlier_anchors": int(exposure.sum()),
                        "wells_with_true_outlier": int(
                            np.sum(exposure > 0)
                        ),
                    }
                    for comparator in METHODS:
                        comparator_index = method_index[comparator]
                        record[
                            f"full_delta_vs_{comparator}"
                        ] = float(
                            arrays["full_rmse"][ci, mi, si, ri]
                            - arrays["full_rmse"][
                                ci, comparator_index, si, ri
                            ]
                        )
                        record[
                            f"non_anchor_delta_vs_{comparator}"
                        ] = float(
                            arrays["non_anchor_rmse"][ci, mi, si, ri]
                            - arrays["non_anchor_rmse"][
                                ci, comparator_index, si, ri
                            ]
                        )
                    no_qc = full_sse[
                        ci, method_index["no_qc"], si, ri
                    ]
                    candidate = full_sse[ci, mi, si, ri]
                    subset_specs = (
                        ("outlier_exposed", exposure > 0),
                        ("no_true_outlier", exposure == 0),
                        ("isolated_outlier", exposure == 1),
                        ("multiple_outlier", exposure >= 2),
                    )
                    for label, selected in subset_specs:
                        count = int(np.sum(selected))
                        record[f"{label}_wells"] = count
                        record[f"{label}_sse_improvement_vs_no_qc"] = (
                            float(np.sum(no_qc[selected] - candidate[selected]))
                            if count
                            else 0.0
                        )
                        record[f"{label}_harmed_wells"] = (
                            int(np.sum(candidate[selected] > no_qc[selected]))
                            if count
                            else 0
                        )
                    records.append(record)
    return pd.DataFrame(records)


def _per_fold_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    fold_sse: np.ndarray,
    fold_non_sse: np.ndarray,
    fold_rows: np.ndarray,
    fold_non_rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for ci, config in enumerate(configs):
        for mi, method in enumerate(METHODS):
            for si, scenario in enumerate(scenarios):
                for ri, realization_seed in enumerate(seeds):
                    for fold in range(5):
                        records.append(
                            {
                                "config": config.name,
                                "method": method,
                                **_scenario_metadata(scenario),
                                "realization_seed": int(realization_seed),
                                "fold": fold,
                                "rows": int(fold_rows[fold]),
                                "non_anchor_rows": int(
                                    fold_non_rows[ci, fold]
                                ),
                                "full_rmse": float(
                                    np.sqrt(
                                        fold_sse[ci, mi, si, ri, fold]
                                        / fold_rows[fold]
                                    )
                                ),
                                "non_anchor_rmse": float(
                                    np.sqrt(
                                        fold_non_sse[
                                            ci, mi, si, ri, fold
                                        ]
                                        / fold_non_rows[ci, fold]
                                    )
                                ),
                            }
                        )
    return pd.DataFrame(records)


def _aggregate_mask_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    values: np.ndarray,
    counts: np.ndarray,
    labels: Sequence[str | int],
    label_column: str,
    non_values: np.ndarray | None = None,
    non_counts: np.ndarray | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for ci, config in enumerate(configs):
        for mi, method in enumerate(METHODS):
            for si, scenario in enumerate(scenarios):
                for ri, realization_seed in enumerate(seeds):
                    for li, label in enumerate(labels):
                        count = int(counts[ci, li])
                        if count <= 0:
                            continue
                        sse = float(values[ci, mi, si, ri, li])
                        record = {
                            "config": config.name,
                            "method": method,
                            **_scenario_metadata(scenario),
                            "realization_seed": int(realization_seed),
                            label_column: label,
                            "rows": count,
                            "sse": sse,
                            "full_rmse": float(np.sqrt(sse / count)),
                        }
                        if non_values is not None and non_counts is not None:
                            non_count = int(non_counts[ci, li])
                            non_sse = float(
                                non_values[ci, mi, si, ri, li]
                            )
                            record.update(
                                {
                                    "non_anchor_rows": non_count,
                                    "non_anchor_sse": non_sse,
                                    "non_anchor_rmse": float(
                                        np.sqrt(non_sse / non_count)
                                    ),
                                }
                            )
                        records.append(record)
    return pd.DataFrame(records)


def _worst_wells_frame(
    configs: Sequence[Any],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    well_ids: Sequence[str],
    folds: np.ndarray,
    rows: np.ndarray,
    full_sse: np.ndarray,
) -> pd.DataFrame:
    rmse = np.sqrt(full_sse / rows[None, None, None, None, :])
    records: list[dict[str, Any]] = []
    identifiers = np.asarray(well_ids)
    for ci, config in enumerate(configs):
        for mi, method in enumerate(METHODS):
            for si, scenario in enumerate(scenarios):
                group = rmse[ci, mi, si]
                q95 = np.quantile(group, 0.95, axis=0)
                mean = group.mean(axis=0)
                maximum = group.max(axis=0)
                max_seed = np.argmax(group, axis=0)
                order = np.lexsort((identifiers, -q95))[:20]
                for rank, wi in enumerate(order, start=1):
                    records.append(
                        {
                            "config": config.name,
                            "method": method,
                            **_scenario_metadata(scenario),
                            "rank_by_realization_q95": rank,
                            "well_id": str(well_ids[wi]),
                            "fold": int(folds[wi]),
                            "rows": int(rows[wi]),
                            "rmse_mean": float(mean[wi]),
                            "rmse_q95": float(q95[wi]),
                            "rmse_max": float(maximum[wi]),
                            "realization_seed_at_max": int(
                                seeds[max_seed[wi]]
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _render_report(
    curve: pd.DataFrame,
    detection: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    primary = curve[
        curve["config"].eq("near_k6")
        & curve["method"].isin(
            [
                "no_qc",
                "local_spike_reject_7p5",
                "oracle_indicator_reject",
                "oracle_clean_values",
            ]
        )
        & curve["scenario"].isin(
            [
                "exact_clean",
                "gaussian_1_clean",
                "moderate_m5_control",
                "moderate_background_p10_m10",
                "severe_background_no_outlier",
                "severe",
            ]
        )
    ]
    overall_detection = detection[
        detection["config"].eq("near_k6")
        & detection["scope"].eq("overall")
        & detection["scenario"].isin(
            [
                "exact_clean",
                "moderate_m5_control",
                "moderate_background_p10_m10",
            ]
        )
    ]
    lines = [
        "# Locked near-terminal k=6 robust anchor QC / outlier rejection",
        "",
        "- `deployable: false`",
        "- `submission_eligible: false`",
        "- `counterfactual_information_experiment: true`",
        "- `uses_validation_target_as_revealed_anchor: true`",
        "- `kaggle_submission_created: false`",
        "- `kaggle_submission_uploaded: false`",
        "",
        "The only operational candidate is `local_spike_reject_7p5`: local "
        "leave-one-out residual innovation, strict score >7.5 ft, at most two "
        "rejections. Indicator/sign values are appended only after decisions.",
        "",
        "## Primary decision",
        "",
        f"- Overall primary success: `{str(summary['primary_success']).lower()}`.",
        f"- Adoption decision: **{summary['adoption_decision']}**.",
        f"- O5 candidate robust `<4`: "
        f"`{str(summary['primary_gate']['A_o5_robust']).lower()}`.",
        f"- Control safety: "
        f"`{str(summary['primary_gate']['B_control_safety']).lower()}`.",
        f"- O5 detection gate: "
        f"`{str(summary['primary_gate']['C_o5_detection']).lower()}`.",
        f"- Tail/fold safety: "
        f"`{str(summary['primary_gate']['D_tail_fold_safety']).lower()}`.",
        "",
        "## Primary robustness rows",
        "",
        _markdown_table(
            primary[
                [
                    "config",
                    "method",
                    "scenario",
                    "full_rmse_q95",
                    "non_anchor_rmse_q95",
                    "hierarchical_full_q975",
                    "hierarchical_non_anchor_q975",
                    "maximum_wells_ge_40",
                    "maximum_wells_ge_60",
                    "robust_lt_4",
                ]
            ]
        ),
        "",
        "## Detection controls and O5",
        "",
        _markdown_table(
            overall_detection[
                [
                    "config",
                    "scenario",
                    "precision",
                    "recall",
                    "false_positive_rate",
                    "specificity",
                    "well_detection_precision",
                    "well_detection_recall",
                ]
            ]
        ),
        "",
        "See `detection_metrics.csv` for anchor order, endpoint/interior, sign, "
        "isolated/multiple, false-reject, and score-distribution breakdowns.",
        "",
    ]
    return "\n".join(lines)


def run_anchor_outlier_qc(
    *,
    data_root: Path,
    fold_config: Path,
    blend_oof: Path,
    terminal_frontier: Path,
    prior_realism_artifact: Path,
    output_root: Path,
    run_id: str,
    scenario_names: Sequence[str] | None = None,
    realization_seeds: Sequence[int] = DEFAULT_NOISE_SEEDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    selection_seed: int = 20260711,
    max_wells_per_fold: int | None = None,
) -> dict[str, Any]:
    """Run the complete locked QC comparison."""
    validate_qc_contract(QC_THRESHOLD, QC_MAX_REJECTS)
    scenarios = select_qc_scenarios(scenario_names)
    seeds = tuple(_validate_seed(seed) for seed in realization_seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or bootstrap_samples < 1
        or (max_wells_per_fold is not None and max_wells_per_fold < 1)
    ):
        raise ValueError("invalid seed/bootstrap/smoke-well configuration")
    _validate_seed(bootstrap_seed)
    _validate_seed(selection_seed)

    data = Path(data_root).resolve()
    run_root = safe_output_directory(output_root, run_id, data)
    fold_hash = hashlib.sha256(Path(fold_config).read_bytes()).hexdigest().upper()
    train_wells = enumerate_wells(data, "train")
    mapping = load_fold_mapping(
        fold_config, expected_well_ids=[well.well_id for well in train_wells]
    )
    if (
        str(mapping["fold_version"].iat[0]) != FOLD_VERSION
        or fold_hash != FOLD_SHA256
        or len(mapping) != 773
    ):
        raise ValueError("locked fold version/hash/well contract mismatch")
    baseline = _read_blend_oof(blend_oof)
    if len(baseline) != 3_783_989 or baseline["well_id"].nunique() != 773:
        raise ValueError("locked blend OOF row/well contract mismatch")
    selected_wells = _selected_wells(
        mapping,
        max_wells_per_fold=max_wells_per_fold,
        seed=selection_seed,
    )
    contexts = _build_noise_contexts(
        baseline, data, mapping, selected_wells
    )
    del baseline
    well_ids = tuple(sorted(contexts))
    well_count = len(well_ids)
    rows_by_well = np.asarray(
        [len(contexts[well_id].row_index) for well_id in well_ids],
        dtype=np.int64,
    )
    folds = np.asarray(
        [contexts[well_id].fold for well_id in well_ids], dtype=np.int8
    )
    total_rows = int(rows_by_well.sum())
    strict_full = max_wells_per_fold is None
    if strict_full and (well_count != 773 or total_rows != 3_783_989):
        raise ValueError("strict full QC run row/well contract mismatch")

    configs = CONFIGS
    config_count = len(configs)
    method_count = len(METHODS)
    scenario_count = len(scenarios)
    seed_count = len(seeds)
    group_count = scenario_count * seed_count
    shape = (
        config_count,
        method_count,
        scenario_count,
        seed_count,
        well_count,
    )
    full_sse = np.zeros(shape, dtype=np.float64)
    non_sse = np.zeros(shape, dtype=np.float64)
    accepted_count = np.zeros(shape, dtype=np.int8)
    changed_rows = np.zeros(shape, dtype=np.int32)
    outlier_count = np.zeros(
        (config_count, scenario_count, seed_count, well_count),
        dtype=np.int8,
    )
    endpoint_outlier = np.zeros_like(outlier_count, dtype=bool)
    interior_outlier = np.zeros_like(outlier_count, dtype=bool)
    decision_scores = np.zeros(
        (
            config_count,
            scenario_count,
            seed_count,
            well_count,
            6,
        ),
        dtype=np.float64,
    )
    decision_rejected = np.zeros_like(decision_scores, dtype=bool)
    decision_truth = np.zeros_like(decision_scores, dtype=bool)
    decision_sign = np.zeros_like(decision_scores, dtype=np.int8)
    fold_rows = np.zeros(5, dtype=np.int64)
    non_rows = np.zeros((config_count, well_count), dtype=np.int64)
    fold_non_rows = np.zeros((config_count, 5), dtype=np.int64)
    fold_sse = np.zeros(
        (config_count, method_count, scenario_count, seed_count, 5),
        dtype=np.float64,
    )
    fold_non_sse = np.zeros_like(fold_sse)
    segment_sse = np.zeros(
        (
            config_count,
            method_count,
            scenario_count,
            seed_count,
            len(SEGMENTS),
        ),
        dtype=np.float64,
    )
    segment_rows = np.zeros(
        (config_count, len(SEGMENTS)), dtype=np.int64
    )
    distance_sse = np.zeros(
        (
            config_count,
            method_count,
            scenario_count,
            seed_count,
            10,
        ),
        dtype=np.float64,
    )
    distance_non_sse = np.zeros_like(distance_sse)
    distance_rows = np.zeros((config_count, 10), dtype=np.int64)
    distance_non_rows = np.zeros((config_count, 10), dtype=np.int64)

    scenario_family = np.asarray(
        [scenario.error_family for scenario in scenarios], dtype=object
    )
    scenario_name = np.asarray(
        [scenario.name for scenario in scenarios], dtype=object
    )
    scenario_tau = np.asarray(
        [scenario.tau for scenario in scenarios], dtype=np.float64
    )
    scenario_sigma = np.asarray(
        [scenario.sigma for scenario in scenarios], dtype=np.float64
    )
    scenario_probability = np.asarray(
        [scenario.outlier_probability for scenario in scenarios],
        dtype=np.float64,
    )
    scenario_magnitude = np.asarray(
        [scenario.outlier_magnitude for scenario in scenarios],
        dtype=np.float64,
    )
    flat_scenario = np.repeat(np.arange(scenario_count), seed_count)
    flat_seed = np.tile(np.arange(seed_count), scenario_count)
    flat_seed_value = np.asarray(seeds, dtype=np.int64)[flat_seed]
    full_mask = (1 << 6) - 1

    placement_digest = hashlib.sha256()
    maximum_common_bias_score_delta = 0.0
    maximum_accepted_anchor_delta = 0.0
    maximum_rejected_anchor_observation_lock_delta = 0.0
    maximum_exact_row_delta = 0.0
    latent_by_physical: dict[
        tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    maximum_shared_latent_delta = 0.0

    per_well_path = run_root / "per_well_seed_metrics.csv.gz"
    decision_path = run_root / "per_anchor_decisions.csv.gz"
    with (
        _DeterministicGzipCsvWriter(per_well_path) as well_writer,
        _DeterministicGzipCsvWriter(decision_path) as decision_writer,
    ):
        for well_position, well_id in enumerate(well_ids):
            context = contexts[well_id]
            fold_rows[context.fold] += len(context.row_index)
            boundary = _boundary_md(context)  # type: ignore[arg-type]
            per_well_frames: list[pd.DataFrame] = []
            for config_index, config in enumerate(configs):
                selection = select_terminal_anchors(
                    context.prediction_md,
                    boundary,
                    policy=config.policy,
                    k=config.k,
                )
                _validate_locked_selection(config, selection)
                physical_rows = context.row_index[selection.local_indices]
                anchor_positions = context.normalized_md[
                    selection.local_indices
                ]
                true_anchors = context.y_true[selection.local_indices]
                base_anchors = context.base_prediction[
                    selection.local_indices
                ]
                clean_residuals = true_anchors - base_anchors
                latents: AnchorLatents = stable_anchor_latents(
                    seeds, well_id, physical_rows
                )
                for anchor_column, physical_row in enumerate(physical_rows):
                    key = (well_id, int(physical_row))
                    value = (
                        latents.independent[:, anchor_column],
                        latents.outlier_indicator_uniform[:, anchor_column],
                        latents.outlier_sign[:, anchor_column],
                    )
                    if key in latent_by_physical:
                        prior = latent_by_physical[key]
                        maximum_shared_latent_delta = max(
                            maximum_shared_latent_delta,
                            *(float(np.max(np.abs(a - b))) for a, b in zip(value, prior)),
                        )
                    else:
                        latent_by_physical[key] = tuple(
                            np.asarray(item).copy() for item in value
                        )  # type: ignore[assignment]

                errors_by_scenario: list[np.ndarray] = []
                flags_by_scenario: list[np.ndarray] = []
                for scenario in scenarios:
                    errors, flags = scenario_anchor_errors(
                        scenario, latents
                    )
                    errors_by_scenario.append(errors)
                    flags_by_scenario.append(flags)
                error_cube = np.stack(errors_by_scenario)
                flag_cube = np.stack(flags_by_scenario)
                flat_errors = error_cube.reshape(group_count, config.k)
                flat_flags = flag_cube.reshape(group_count, config.k)
                observed_residuals = clean_residuals[None, :] + flat_errors

                # Operational decisions are complete before any indicator is
                # passed to post-decision scoring or artifacts.
                qc = _batch_local_spike_decisions(
                    anchor_positions, observed_residuals
                )
                biased_qc = _batch_local_spike_decisions(
                    anchor_positions, observed_residuals + 13.25
                )
                maximum_common_bias_score_delta = max(
                    maximum_common_bias_score_delta,
                    float(
                        np.nanmax(
                            np.abs(
                                qc["initial_score"]
                                - biased_qc["initial_score"]
                            )
                        )
                    ),
                )
                if not np.array_equal(
                    qc["accepted_mask"], biased_qc["accepted_mask"]
                ):
                    raise ValueError("common bias changed operational QC decisions")

                powers = 1 << np.arange(config.k, dtype=np.uint8)
                oracle_masks = np.sum(
                    (~flat_flags).astype(np.uint8) * powers[None, :],
                    axis=1,
                    dtype=np.uint8,
                )
                method_masks = {
                    "no_qc": np.full(group_count, full_mask, dtype=np.uint8),
                    PRIMARY_METHOD: qc["accepted_mask"],
                    "oracle_indicator_reject": oracle_masks,
                    "oracle_clean_values": np.full(
                        group_count, full_mask, dtype=np.uint8
                    ),
                }
                method_errors = {
                    "no_qc": flat_errors,
                    PRIMARY_METHOD: flat_errors,
                    "oracle_indicator_reject": flat_errors,
                    "oracle_clean_values": np.zeros_like(flat_errors),
                }

                is_anchor = anchor_mask(len(context.row_index), selection)
                is_non_anchor = ~is_anchor
                regions = frontier_segment_regions(
                    len(context.row_index), selection
                )
                deciles = np.minimum(
                    (context.normalized_md * 10).astype(np.int8), 9
                )
                metric_masks = np.stack(
                    [
                        np.ones(len(context.row_index), dtype=bool),
                        is_non_anchor,
                        *(regions == segment for segment in SEGMENTS),
                        *(deciles == decile for decile in range(10)),
                        *(
                            (deciles == decile) & is_non_anchor
                            for decile in range(10)
                        ),
                    ]
                )
                used_subsets = set()
                for masks_for_method in method_masks.values():
                    used_subsets.update(map(int, np.unique(masks_for_method)))
                coefficient_cache: dict[
                    int,
                    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                ] = {}
                predictions_by_subset: dict[int, np.ndarray] = {}
                weights_by_subset: dict[int, np.ndarray] = {}
                for subset_mask in sorted(used_subsets):
                    coefficients, exact_prediction, weights = (
                        _coefficients_for_subset(
                            context,
                            selection,
                            true_anchors,
                            subset_mask,
                            metric_masks,
                        )
                    )
                    coefficient_cache[subset_mask] = coefficients
                    predictions_by_subset[subset_mask] = exact_prediction
                    weights_by_subset[subset_mask] = weights
                mask_counts = coefficient_cache[full_mask][3]
                non_rows[config_index, well_position] = int(mask_counts[1])
                fold_non_rows[config_index, context.fold] += int(
                    mask_counts[1]
                )
                segment_rows[config_index] += mask_counts[2:5]
                distance_rows[config_index] += mask_counts[5:15]
                distance_non_rows[config_index] += mask_counts[15:25]

                exact_reference = condition_noisy_piecewise(
                    context.base_prediction,
                    true_anchors,
                    context.normalized_md,
                    selection,
                )
                maximum_exact_row_delta = max(
                    maximum_exact_row_delta,
                    float(
                        np.max(
                            np.abs(
                                exact_reference
                                - predictions_by_subset[full_mask]
                            )
                        )
                    ),
                )
                full_exact = predictions_by_subset[full_mask]
                full_weights = weights_by_subset[full_mask]
                structural_changed_by_subset: dict[int, int] = {}
                for subset_mask in used_subsets:
                    if subset_mask == full_mask:
                        structural_changed_by_subset[subset_mask] = 0
                        continue
                    structural = (
                        np.abs(
                            predictions_by_subset[subset_mask] - full_exact
                        )
                        > ROWWISE_TOLERANCE
                    ) | np.any(
                        np.abs(
                            weights_by_subset[subset_mask] - full_weights
                        )
                        > ROWWISE_TOLERANCE,
                        axis=1,
                    )
                    structural_changed_by_subset[subset_mask] = int(
                        structural.sum()
                    )

                method_results: dict[str, np.ndarray] = {}
                method_accepted: dict[str, np.ndarray] = {}
                method_changed: dict[str, np.ndarray] = {}
                for method_index, method in enumerate(METHODS):
                    subset_masks = method_masks[method]
                    results = _evaluate_subsets(
                        subset_masks,
                        method_errors[method],
                        coefficient_cache,
                    )
                    method_results[method] = results
                    accepted_for_group = np.asarray(
                        [int(mask).bit_count() for mask in subset_masks],
                        dtype=np.int8,
                    )
                    changed_for_group = np.asarray(
                        [
                            structural_changed_by_subset[int(mask)]
                            for mask in subset_masks
                        ],
                        dtype=np.int32,
                    )
                    if method == "oracle_clean_values":
                        changed_for_group = np.where(
                            np.any(
                                np.abs(flat_errors) > ROWWISE_TOLERANCE,
                                axis=1,
                            ),
                            len(context.row_index),
                            0,
                        ).astype(np.int32)
                    reshaped = results.reshape(
                        scenario_count, seed_count, -1
                    )
                    full_sse[
                        config_index,
                        method_index,
                        :,
                        :,
                        well_position,
                    ] = reshaped[:, :, 0]
                    non_sse[
                        config_index,
                        method_index,
                        :,
                        :,
                        well_position,
                    ] = reshaped[:, :, 1]
                    accepted_count[
                        config_index,
                        method_index,
                        :,
                        :,
                        well_position,
                    ] = accepted_for_group.reshape(
                        scenario_count, seed_count
                    )
                    changed_rows[
                        config_index,
                        method_index,
                        :,
                        :,
                        well_position,
                    ] = changed_for_group.reshape(
                        scenario_count, seed_count
                    )
                    fold_sse[
                        config_index, method_index, :, :, context.fold
                    ] += reshaped[:, :, 0]
                    fold_non_sse[
                        config_index, method_index, :, :, context.fold
                    ] += reshaped[:, :, 1]
                    segment_sse[
                        config_index, method_index
                    ] += reshaped[:, :, 2:5]
                    distance_sse[
                        config_index, method_index
                    ] += reshaped[:, :, 5:15]
                    distance_non_sse[
                        config_index, method_index
                    ] += reshaped[:, :, 15:25]
                    method_accepted[method] = accepted_for_group
                    method_changed[method] = changed_for_group

                flag_count = flat_flags.sum(axis=1, dtype=np.int8)
                outlier_count[
                    config_index, :, :, well_position
                ] = flag_count.reshape(scenario_count, seed_count)
                endpoint_outlier[
                    config_index, :, :, well_position
                ] = np.any(flat_flags[:, [0, 5]], axis=1).reshape(
                    scenario_count, seed_count
                )
                interior_outlier[
                    config_index, :, :, well_position
                ] = np.any(flat_flags[:, 1:5], axis=1).reshape(
                    scenario_count, seed_count
                )
                decision_scores[
                    config_index, :, :, well_position
                ] = qc["score_at_decision"].reshape(
                    scenario_count, seed_count, config.k
                )
                decision_rejected[
                    config_index, :, :, well_position
                ] = qc["rejected"].reshape(
                    scenario_count, seed_count, config.k
                )
                decision_truth[
                    config_index, :, :, well_position
                ] = flat_flags.reshape(
                    scenario_count, seed_count, config.k
                )
                decision_sign[
                    config_index, :, :, well_position
                ] = np.broadcast_to(
                    latents.outlier_sign[None, :, :],
                    (scenario_count, seed_count, config.k),
                )

                placement_digest.update(well_id.encode("ascii"))
                placement_digest.update(config.name.encode("ascii"))
                placement_digest.update(
                    np.ascontiguousarray(physical_rows).view(np.uint8)
                )

                # Append truth fields only after the QC function returned.
                repeated_anchor = np.tile(np.arange(config.k), group_count)
                repeated_group = np.repeat(np.arange(group_count), config.k)
                observed_anchor = (
                    true_anchors[None, :] + flat_errors
                ).reshape(-1)
                rejected_long = qc["rejected"].reshape(-1)
                decision_frame = pd.DataFrame(
                    {
                        "config": config.name,
                        "method": PRIMARY_METHOD,
                        "scenario": scenario_name[
                            flat_scenario[repeated_group]
                        ],
                        "realization_seed": flat_seed_value[repeated_group],
                        "well_id": well_id,
                        "fold": context.fold,
                        "anchor_order": repeated_anchor + 1,
                        "row_index": physical_rows[repeated_anchor],
                        "normalized_md_position": anchor_positions[
                            repeated_anchor
                        ],
                        "observed_anchor_tvt": observed_anchor,
                        "base_anchor_prediction": base_anchors[
                            repeated_anchor
                        ],
                        "anchor_residual": observed_residuals.reshape(-1),
                        "qc_score_at_decision": qc[
                            "score_at_decision"
                        ].reshape(-1),
                        "initial_qc_score": qc["initial_score"].reshape(-1),
                        "decision_iteration": qc[
                            "decision_iteration"
                        ].reshape(-1),
                        "rejected": rejected_long,
                        "true_outlier_indicator": flat_flags.reshape(-1),
                        "true_outlier_sign": latents.outlier_sign[
                            flat_seed[repeated_group],
                            repeated_anchor,
                        ],
                        "true_outlier_contribution": (
                            flat_flags
                            * np.stack(
                                [
                                    latents.outlier_sign
                                    * scenario.outlier_magnitude
                                    for scenario in scenarios
                                ]
                            ).reshape(group_count, config.k)
                        ).reshape(-1),
                        "well_outlier_count": flag_count[repeated_group],
                        "accepted_anchor_count_final": qc[
                            "accepted_count"
                        ][repeated_group],
                    }
                )
                decision_writer.write(decision_frame)

                for method_index, method in enumerate(METHODS):
                    values = method_results[method]
                    current_full = values[:, 0]
                    current_non = values[:, 1]
                    frames = {
                        "config": config.name,
                        "policy": config.policy,
                        "k": config.k,
                        "fractions": config.fractions_text,
                        "method": method,
                        "error_family": scenario_family[flat_scenario],
                        "scenario": scenario_name[flat_scenario],
                        "tau": scenario_tau[flat_scenario],
                        "sigma": scenario_sigma[flat_scenario],
                        "outlier_probability": scenario_probability[
                            flat_scenario
                        ],
                        "outlier_magnitude": scenario_magnitude[
                            flat_scenario
                        ],
                        "realization_seed": flat_seed_value,
                        "well_id": well_id,
                        "fold": context.fold,
                        "rows": len(context.row_index),
                        "non_anchor_rows": int(mask_counts[1]),
                        "sse": current_full,
                        "non_anchor_sse": current_non,
                        "rmse": np.sqrt(
                            current_full / len(context.row_index)
                        ),
                        "non_anchor_rmse": np.sqrt(
                            current_non / int(mask_counts[1])
                        ),
                        "accepted_anchor_count": method_accepted[method],
                        "rejected_anchor_count": (
                            config.k - method_accepted[method]
                        ),
                        "rows_changed_by_qc": method_changed[method],
                        "true_outlier_anchors": flag_count,
                        "well_has_true_outlier": flag_count > 0,
                        "well_has_isolated_outlier": flag_count == 1,
                        "well_has_multiple_outliers": flag_count >= 2,
                    }
                    for comparator in METHODS:
                        comparator_values = method_results[comparator]
                        frames[f"sse_delta_vs_{comparator}"] = (
                            current_full - comparator_values[:, 0]
                        )
                        frames[
                            f"non_anchor_sse_delta_vs_{comparator}"
                        ] = current_non - comparator_values[:, 1]
                    per_well_frames.append(pd.DataFrame(frames))

                # Spot-check accepted/rejected row semantics with the first
                # candidate group that actually rejects an anchor.
                rejecting = np.flatnonzero(qc["accepted_count"] < config.k)
                if len(rejecting):
                    group = int(rejecting[0])
                    retained = np.asarray(
                        [
                            bool(
                                int(qc["accepted_mask"][group])
                                & (1 << order)
                            )
                            for order in range(config.k)
                        ]
                    )
                    observed = true_anchors + flat_errors[group]
                    prediction = condition_with_accepted_anchors(
                        context.base_prediction,
                        observed,
                        context.normalized_md,
                        selection,
                        retained,
                    )
                    maximum_accepted_anchor_delta = max(
                        maximum_accepted_anchor_delta,
                        float(
                            np.max(
                                np.abs(
                                    prediction[
                                        selection.local_indices[retained]
                                    ]
                                    - observed[retained]
                                )
                            )
                        ),
                    )
                    rejected_delta = np.abs(
                        prediction[selection.local_indices[~retained]]
                        - observed[~retained]
                    )
                    if len(rejected_delta):
                        maximum_rejected_anchor_observation_lock_delta = max(
                            maximum_rejected_anchor_observation_lock_delta,
                            float(np.max(rejected_delta)),
                        )

            well_writer.write(pd.concat(per_well_frames, ignore_index=True))
            if (well_position + 1) % 25 == 0 or well_position + 1 == well_count:
                print(
                    f"anchor-outlier-qc wells={well_position + 1}/{well_count}",
                    flush=True,
                )

    if maximum_exact_row_delta > ROWWISE_TOLERANCE:
        raise ValueError("exact clean reconstruction changed")
    if maximum_common_bias_score_delta > 1e-12:
        raise ValueError("QC scores are not common-bias invariant")
    if maximum_accepted_anchor_delta != 0.0:
        raise ValueError("accepted anchor prediction is not observation-exact")

    regression = (
        _regression_reproduction(
            prior_realism_artifact,
            configs,
            scenarios,
            seeds,
            well_ids,
            full_sse,
            non_sse,
            rows_by_well,
            non_rows,
        )
        if strict_full
        else {
            "passed": True,
            "skipped_for_smoke_subset": True,
            "scenario_mapping": {},
        }
    )
    if not regression["passed"]:
        raise ValueError("no-QC prior-artifact reproduction failed")

    hierarchy = hierarchical_qc_bootstrap(
        full_sse,
        non_sse,
        rows_by_well,
        non_rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    for item in hierarchy:
        config = configs[int(item["config_index"])]
        method = METHODS[int(item["method_index"])]
        scenario = scenarios[int(item["scenario_index"])]
        item.update(
            {
                "config": config.name,
                "method": method,
                **_scenario_metadata(scenario),
                "samples": int(bootstrap_samples),
                "bootstrap_seed": int(bootstrap_seed),
            }
        )
    curve, metric_arrays = _information_curve(
        configs,
        scenarios,
        seeds,
        full_sse,
        non_sse,
        rows_by_well,
        non_rows,
        fold_sse,
        fold_non_sse,
        fold_rows,
        fold_non_rows,
        segment_sse,
        segment_rows,
        hierarchy,
    )
    per_seed = _per_seed_frame(
        configs,
        scenarios,
        seeds,
        metric_arrays,
        accepted_count,
        changed_rows,
        outlier_count,
        full_sse,
        rows_by_well,
    )
    per_fold = _per_fold_frame(
        configs,
        scenarios,
        seeds,
        fold_sse,
        fold_non_sse,
        fold_rows,
        fold_non_rows,
    )
    per_segment = _aggregate_mask_frame(
        configs,
        scenarios,
        seeds,
        segment_sse,
        segment_rows,
        SEGMENTS,
        "segment",
    )
    distance = _aggregate_mask_frame(
        configs,
        scenarios,
        seeds,
        distance_sse,
        distance_rows,
        tuple(range(10)),
        "distance_decile",
        distance_non_sse,
        distance_non_rows,
    )
    worst = _worst_wells_frame(
        configs,
        scenarios,
        seeds,
        well_ids,
        folds,
        rows_by_well,
        full_sse,
    )

    # Build one config/scenario scoring block at a time.  QC decisions are
    # already immutable; indicators enter only in these post-decision blocks.
    detection_blocks: list[pd.DataFrame] = []
    anchor_orders = np.arange(1, 7, dtype=np.int8)
    for config_index, config in enumerate(configs):
        for scenario_index, scenario in enumerate(scenarios):
            score = decision_scores[config_index, scenario_index]
            rejected = decision_rejected[config_index, scenario_index]
            truth = decision_truth[config_index, scenario_index]
            signs = decision_sign[config_index, scenario_index]
            well_counts = truth.sum(axis=2, dtype=np.int8)
            scoring = pd.DataFrame(
                {
                    "config": config.name,
                    "scenario": scenario.name,
                    "realization_seed": np.repeat(
                        np.asarray(seeds, dtype=np.int64),
                        well_count * config.k,
                    ),
                    "well_id": np.tile(
                        np.repeat(np.asarray(well_ids), config.k),
                        seed_count,
                    ),
                    "anchor_order": np.tile(
                        anchor_orders, seed_count * well_count
                    ),
                    "rejected": rejected.reshape(-1),
                    "qc_score_at_decision": score.reshape(-1),
                    "true_outlier_indicator": truth.reshape(-1),
                    "true_outlier_sign": signs.reshape(-1),
                    "true_outlier_contribution": (
                        truth.astype(np.float64)
                        * signs.astype(np.float64)
                        * scenario.outlier_magnitude
                    ).reshape(-1),
                    "well_outlier_count": np.repeat(
                        well_counts.reshape(-1), config.k
                    ),
                }
            )
            detection_blocks.append(
                _detection_rows_from_decisions(scoring)
            )
    detection = pd.concat(detection_blocks, ignore_index=True)

    curve.to_csv(run_root / "qc_robustness_curve.csv", index=False)
    detection.to_csv(run_root / "detection_metrics.csv", index=False)
    per_seed.to_csv(run_root / "per_seed_metrics.csv", index=False)
    per_fold.to_csv(run_root / "per_fold_seed_metrics.csv", index=False)
    per_segment.to_csv(
        run_root / "per_segment_seed_metrics.csv", index=False
    )
    distance.to_csv(
        run_root / "prediction_distance_deciles.csv", index=False
    )
    worst.to_csv(run_root / "worst_wells.csv", index=False)
    _atomic_json(
        run_root / "hierarchical_bootstrap.json",
        {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "paired_well_and_error_draws_across_methods_scenarios": True,
            "comparisons": hierarchy,
        },
    )

    config_payload = {
        "run_id": run_id,
        "profile": (
            "full"
            if strict_full
            and len(scenarios) == len(QC_SCENARIOS)
            and len(seeds) == 100
            else "regression"
            if strict_full
            else "smoke"
        ),
        "data_root": str(data),
        "blend_oof": str(Path(blend_oof).resolve()),
        "terminal_frontier": str(Path(terminal_frontier).resolve()),
        "prior_realism_artifact": str(Path(prior_realism_artifact).resolve()),
        "fold_version": FOLD_VERSION,
        "fold_sha256": fold_hash,
        "wells": well_count,
        "rows": total_rows,
        "max_wells_per_fold": max_wells_per_fold,
        "selection_seed": int(selection_seed),
        "locked_configs": [
            {
                "config": config.name,
                "policy": config.policy,
                "k": config.k,
                "fractions": list(config.fractions),
            }
            for config in configs
        ],
        "methods": list(METHODS),
        "operational_candidate": PRIMARY_METHOD,
        "qc_threshold_ft": QC_THRESHOLD,
        "qc_threshold_comparison": "strictly_greater_than",
        "qc_maximum_rejects": QC_MAX_REJECTS,
        "score_boundary": "post-PS anchors only; PS boundary excluded",
        "scenarios": [_scenario_metadata(scenario) for scenario in scenarios],
        "realization_seeds": list(seeds),
        "stable_random_keys": {
            "common_bias": "(seed, well_id, 'common_bias')",
            "independent": "(seed, well_id, anchor_row, 'independent')",
            "outlier_indicator": (
                "(seed, well_id, anchor_row, 'outlier_indicator')"
            ),
            "outlier_sign": "(seed, well_id, anchor_row, 'outlier_sign')",
        },
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "robust_lt_4_definition": [
            "full fixed-well error-realization q95 < 4",
            "non-anchor fixed-well error-realization q95 < 4",
            "hierarchical full q97.5 < 4",
            "hierarchical non-anchor q97.5 < 4",
            "no persistent fold failure",
            "maximum wells_ge_40 == 0 and maximum wells_ge_60 == 0",
        ],
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "deployable": False,
        "submission_eligible": False,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    _atomic_json(run_root / "run_config.json", config_payload)

    def curve_row(config: str, method: str, scenario: str) -> pd.Series:
        match = curve[
            curve["config"].eq(config)
            & curve["method"].eq(method)
            & curve["scenario"].eq(scenario)
        ]
        if len(match) != 1:
            raise ValueError(
                f"missing unique curve row: {config}/{method}/{scenario}"
            )
        return match.iloc[0]

    primary_gate_available = all(
        name in {scenario.name for scenario in scenarios}
        for name in (
            "exact_clean",
            "gaussian_1_clean",
            "moderate_m5_control",
            "moderate_background_p10_m10",
        )
    )
    if primary_gate_available:
        o5 = curve_row(
            "near_k6", PRIMARY_METHOD, "moderate_background_p10_m10"
        )
        control_names = (
            "exact_clean",
            "gaussian_1_clean",
            "moderate_m5_control",
        )
        control_safety = True
        for name in control_names:
            candidate = curve_row("near_k6", PRIMARY_METHOD, name)
            no_qc = curve_row("near_k6", "no_qc", name)
            control_safety = bool(
                control_safety
                and candidate["full_rmse_q95"]
                - no_qc["full_rmse_q95"]
                <= 0.05
                and candidate["non_anchor_rmse_q95"]
                - no_qc["non_anchor_rmse_q95"]
                <= 0.05
                and (
                    not bool(no_qc["robust_lt_4"])
                    or bool(candidate["robust_lt_4"])
                )
            )
        detection_o5 = detection[
            detection["config"].eq("near_k6")
            & detection["scenario"].eq("moderate_background_p10_m10")
            & detection["scope"].eq("overall")
        ].iloc[0]
        detection_gate = bool(
            float(detection_o5["recall"]) >= 0.70
            and float(detection_o5["false_positive_rate"]) <= 0.02
            and float(detection_o5["precision"]) >= 0.75
        )
        tail_rows = curve[
            curve["config"].eq("near_k6")
            & curve["method"].eq(PRIMARY_METHOD)
            & curve["scenario"].isin(
                [*control_names, "moderate_background_p10_m10"]
            )
        ]
        tail_fold_gate = bool(
            tail_rows["maximum_wells_ge_40"].eq(0).all()
            and tail_rows["maximum_wells_ge_60"].eq(0).all()
            and ~tail_rows["persistent_fold_failure"].any()
        )
        primary_gate = {
            "A_o5_robust": bool(o5["robust_lt_4"]),
            "B_control_safety": control_safety,
            "C_o5_detection": detection_gate,
            "D_tail_fold_safety": tail_fold_gate,
        }
    else:
        primary_gate = {
            "A_o5_robust": None,
            "B_control_safety": None,
            "C_o5_detection": None,
            "D_tail_fold_safety": None,
        }
    primary_success = bool(
        primary_gate_available and all(primary_gate.values())
    )

    oracle_interpretation = "not_available"
    if "moderate_background_p10_m10" in {
        scenario.name for scenario in scenarios
    }:
        candidate_o5 = curve_row(
            "near_k6", PRIMARY_METHOD, "moderate_background_p10_m10"
        )
        oracle_o5 = curve_row(
            "near_k6",
            "oracle_indicator_reject",
            "moderate_background_p10_m10",
        )
        if bool(oracle_o5["robust_lt_4"]) and not bool(
            candidate_o5["robust_lt_4"]
        ):
            oracle_interpretation = (
                "identification_failure_geometry_sufficient; next repeat "
                "measurement or QC metadata; do not retune threshold"
            )
        elif not bool(oracle_o5["robust_lt_4"]):
            oracle_interpretation = (
                "replacement_geometry_insufficient; next redundant anchors "
                "or repeat measurements"
            )
        elif primary_success:
            oracle_interpretation = (
                "candidate_success_with_remaining_oracle_gap_reported"
            )
        else:
            oracle_interpretation = (
                "candidate_improves_but_fails_non-O5 operational gate"
            )

    adoption_decision = (
        "retain this QC in the retrospective M10 specification"
        if primary_success
        else (
            "research repeat measurement / quality metadata"
            if "identification_failure" in oracle_interpretation
            else (
                "increase redundant anchors or repeat measurements"
                if "replacement_geometry" in oracle_interpretation
                else "do not adopt this QC"
            )
        )
    )
    summary = {
        "profile": config_payload["profile"],
        "wells": well_count,
        "rows": total_rows,
        "configs": len(configs),
        "methods": len(METHODS),
        "scenarios": len(scenarios),
        "realizations": len(seeds),
        "hierarchical_bootstrap_samples": int(bootstrap_samples),
        "no_qc_regression_reproduction": regression,
        "primary_gate": primary_gate,
        "primary_success": primary_success,
        "oracle_interpretation": oracle_interpretation,
        "adoption_decision": adoption_decision,
        "catastrophic_wells_reappeared": bool(
            curve["maximum_wells_ge_40"].gt(0).any()
            or curve["maximum_wells_ge_60"].gt(0).any()
        ),
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }

    leakage = {
        "passed": True,
        "qc_decisions_completed_before_indicator_scoring": True,
        "qc_function_signature_excludes_true_indicator_target_seed": True,
        "qc_inputs": [
            "normalized MD",
            "base prediction at anchors",
            "corrupted observed anchor TVT",
            "anchor order",
            "accepted/rejected state",
        ],
        "qc_forbidden_inputs_used": [],
        "score_excludes_ps_boundary": True,
        "maximum_common_bias_score_delta": maximum_common_bias_score_delta,
        "maximum_accepted_anchor_prediction_delta": (
            maximum_accepted_anchor_delta
        ),
        "maximum_rejected_anchor_observation_difference": (
            maximum_rejected_anchor_observation_lock_delta
        ),
        "maximum_exact_row_reproduction_delta": maximum_exact_row_delta,
        "maximum_shared_physical_anchor_latent_delta": (
            maximum_shared_latent_delta
        ),
        "placement_positions_sha256": placement_digest.hexdigest().upper(),
        "placement_target_independent": True,
        "placement_target_permutation_invariant": True,
        "error_does_not_change_anchor_snapping_or_fractions": True,
        "random_draws_cross_process_deterministic": True,
        "same_latents_for_shared_physical_anchor_rows": (
            maximum_shared_latent_delta == 0.0
        ),
        "no_qc_regression_reproduction": regression,
        "horizontal_inference_columns": [
            "MD",
            "X",
            "Y",
            "Z",
            "GR",
            "TVT_input",
        ],
        "typewell_columns_read": [],
        "formation_surfaces_read": False,
        "geology_read": False,
        "png_read": False,
        "data_write_attempted": False,
        "row_key_alignment": True,
        "well_level_folds": True,
        "fold_version": FOLD_VERSION,
        "fold_sha256": fold_hash,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
    }
    leakage["passed"] = bool(
        regression["passed"]
        and maximum_common_bias_score_delta <= 1e-12
        and maximum_accepted_anchor_delta == 0.0
        and maximum_exact_row_delta <= ROWWISE_TOLERANCE
        and maximum_shared_latent_delta == 0.0
    )
    _atomic_json(run_root / "leakage_check.json", leakage)
    _atomic_json(run_root / "summary.json", summary)
    (run_root / "report.md").write_text(
        _render_report(curve, detection, summary), encoding="utf-8"
    )
    artifact_audit = audit_anchor_outlier_qc_artifacts(run_root)
    leakage["artifact_validation"] = artifact_audit
    leakage["passed"] = bool(leakage["passed"] and artifact_audit["passed"])
    summary["artifact_validation"] = artifact_audit
    _atomic_json(run_root / "leakage_check.json", leakage)
    _atomic_json(run_root / "summary.json", summary)
    if not leakage["passed"]:
        raise ValueError("QC leakage/reproduction audit failed")
    return summary


def audit_anchor_outlier_qc_artifacts(run_root: Path) -> dict[str, Any]:
    """Reconstruct pooled metrics and validate all compact artifact keys."""
    root = Path(run_root).resolve()
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    per_seed = pd.read_csv(root / "per_seed_metrics.csv")
    curve = pd.read_csv(root / "qc_robustness_curve.csv")
    per_fold = pd.read_csv(root / "per_fold_seed_metrics.csv")
    per_segment = pd.read_csv(root / "per_segment_seed_metrics.csv")
    distance = pd.read_csv(root / "prediction_distance_deciles.csv")
    detection = pd.read_csv(root / "detection_metrics.csv")
    keys = ["config", "method", "scenario", "realization_seed"]
    partials: list[pd.DataFrame] = []
    duplicate_well_keys = 0
    per_well_rows = 0
    unexpected_nan = 0
    usecols = keys + [
        "well_id",
        "rows",
        "non_anchor_rows",
        "sse",
        "non_anchor_sse",
    ]
    for chunk in pd.read_csv(
        root / "per_well_seed_metrics.csv.gz",
        usecols=usecols,
        dtype={"well_id": "string"},
        chunksize=200_000,
    ):
        per_well_rows += len(chunk)
        duplicate_well_keys += int(
            chunk.duplicated(keys + ["well_id"]).sum()
        )
        unexpected_nan += int(
            chunk.select_dtypes(include=[np.number]).isna().sum().sum()
        )
        partials.append(
            chunk.groupby(keys, sort=False)
            .agg(
                wells=("well_id", "size"),
                rows=("rows", "sum"),
                non_anchor_rows=("non_anchor_rows", "sum"),
                sse=("sse", "sum"),
                non_anchor_sse=("non_anchor_sse", "sum"),
            )
            .reset_index()
        )
    aggregate = (
        pd.concat(partials, ignore_index=True)
        .groupby(keys, sort=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    aggregate["full_rmse"] = np.sqrt(
        aggregate["sse"] / aggregate["rows"]
    )
    aggregate["non_anchor_rmse"] = np.sqrt(
        aggregate["non_anchor_sse"] / aggregate["non_anchor_rows"]
    )
    joined = aggregate.merge(
        per_seed[keys + ["full_rmse", "non_anchor_rmse"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_reconstructed", "_reported"),
        indicator=True,
    )
    full_delta = np.abs(
        joined["full_rmse_reconstructed"] - joined["full_rmse_reported"]
    )
    non_delta = np.abs(
        joined["non_anchor_rmse_reconstructed"]
        - joined["non_anchor_rmse_reported"]
    )
    expected_groups = (
        len(config["locked_configs"])
        * len(config["methods"])
        * len(config["scenarios"])
        * len(config["realization_seeds"])
    )
    expected_well_rows = expected_groups * int(config["wells"])
    expected_decisions = (
        len(config["locked_configs"])
        * len(config["scenarios"])
        * len(config["realization_seeds"])
        * int(config["wells"])
        * 6
    )
    decision_rows = 0
    decision_duplicates = 0
    decision_numeric_nan = 0
    decision_keys = [
        "config",
        "method",
        "scenario",
        "realization_seed",
        "well_id",
        "anchor_order",
    ]
    for chunk in pd.read_csv(
        root / "per_anchor_decisions.csv.gz",
        chunksize=300_000,
        dtype={"well_id": "string"},
    ):
        decision_rows += len(chunk)
        decision_duplicates += int(chunk.duplicated(decision_keys).sum())
        decision_numeric_nan += int(
            chunk.select_dtypes(include=[np.number]).isna().sum().sum()
        )
    table_duplicates = sum(
        (
            int(per_seed.duplicated(keys).sum()),
            int(curve.duplicated(["config", "method", "scenario"]).sum()),
            int(per_fold.duplicated(keys + ["fold"]).sum()),
            int(per_segment.duplicated(keys + ["segment"]).sum()),
            int(distance.duplicated(keys + ["distance_decile"]).sum()),
            decision_duplicates,
            duplicate_well_keys,
        )
    )
    structural_nan = int(
        curve["post_last_rmse_q50"].isna().sum()
    )
    expected_structural_nan = int(
        np.sum(curve["config"].eq("td_k6"))
    )
    passed = bool(
        len(aggregate) == expected_groups
        and per_well_rows == expected_well_rows
        and decision_rows == expected_decisions
        and joined["_merge"].eq("both").all()
        and aggregate["wells"].eq(int(config["wells"])).all()
        and aggregate["rows"].eq(int(config["rows"])).all()
        and float(full_delta.max()) <= METRIC_TOLERANCE
        and float(non_delta.max()) <= METRIC_TOLERANCE
        and table_duplicates == 0
        and unexpected_nan == 0
        and decision_numeric_nan == 0
        and structural_nan == expected_structural_nan
        and len(detection) > 0
    )
    return {
        "passed": passed,
        "expected_metric_groups": expected_groups,
        "actual_metric_groups": int(len(aggregate)),
        "per_well_rows": per_well_rows,
        "expected_per_well_rows": expected_well_rows,
        "per_anchor_decision_rows": decision_rows,
        "expected_per_anchor_decision_rows": expected_decisions,
        "max_full_rmse_reconstruction_delta": float(full_delta.max()),
        "max_non_anchor_rmse_reconstruction_delta": float(non_delta.max()),
        "duplicate_key_count": table_duplicates,
        "unexpected_numeric_nan_count": (
            unexpected_nan + decision_numeric_nan
        ),
        "structural_curve_nan_count": structural_nan,
        "expected_structural_curve_nan_count": expected_structural_nan,
    }
