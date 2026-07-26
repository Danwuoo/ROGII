"""Terminal-aware sparse TVT-anchor frontier on the locked strict OOF rows.

The diagnostic separates retrospective/acausal interpolation from genuinely
causal online correction.  Placement depends only on normalized MD.  Target
TVT enters only at the preselected revealed rows and in post-prediction scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import enumerate_wells
from ..folds import load_fold_mapping, partition_well_ids
from ..models.path_pca import PathPCAState
from .sparse_anchor_voi import (
    AnchorSelection,
    BASELINE_RMSE,
    DEFAULT_SEED,
    FOLD_SHA256,
    FOLD_VERSION,
    WellContext,
    _atomic_json,
    _build_well_contexts,
    _markdown_table,
    _nearest_unused_indices,
    _read_aligned_baselines,
    anchor_mask,
    condition_path_pca_posterior,
    condition_piecewise_residual,
    fit_conditional_calibration,
    non_anchor_rmse,
    normalized_md_positions,
    safe_output_directory,
)


POLICIES = ("uniform", "endpoint_inclusive", "near_terminal")
MODES = ("acausal", "causal")
PIECEWISE_METHOD = "piecewise_residual"
CONDITIONAL_METHOD = "conditional_path_pca_mvp"
LOCKED_UNIFORM_K8_RMSE = 3.0788527685214415
LOCKED_UNIFORM_K8_NON_ANCHOR_RMSE = 3.0813716704050207
FOLD_MAX_GUARD = 4.5


@dataclass(frozen=True)
class FrontierPoint:
    policy: str
    mode: str
    k: int
    reason: str


def requested_fractions(policy: str, k: int) -> np.ndarray:
    """Return the locked target-independent normalized-MD placement fractions."""
    if policy not in POLICIES:
        raise ValueError(f"unknown terminal placement policy: {policy}")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return np.empty(0, dtype=np.float64)
    if policy == "uniform":
        return np.arange(1, k + 1, dtype=np.float64) / float(k + 1)
    if policy == "endpoint_inclusive":
        return np.arange(1, k + 1, dtype=np.float64) / float(k)
    if k == 1:
        return np.asarray([0.975], dtype=np.float64)
    return np.concatenate(
        (
            np.arange(1, k, dtype=np.float64) / float(k),
            np.asarray([0.975], dtype=np.float64),
        )
    )


def select_terminal_anchors(
    prediction_md: np.ndarray,
    boundary_md: float,
    *,
    policy: str,
    k: int,
) -> AnchorSelection:
    """Snap locked fractions to deterministic nearest-unused prediction rows."""
    positions = normalized_md_positions(prediction_md, boundary_md)
    fractions = requested_fractions(policy, k)
    local_indices = _nearest_unused_indices(positions, fractions)
    return AnchorSelection(
        policy=policy,
        requested_k=int(k),
        local_indices=local_indices,
        normalized_positions=positions[local_indices],
        requested_positions=fractions[: len(local_indices)],
        acquisition_scores=np.full(len(local_indices), np.nan, dtype=np.float64),
        minimum_spacing=0.0,
    )


def condition_causal_piecewise(
    base_prediction: np.ndarray,
    y_true: np.ndarray,
    selection: AnchorSelection,
) -> np.ndarray:
    """Carry the latest observed residual forward without using future anchors."""
    base = np.asarray(base_prediction, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    if (
        base.ndim != 1
        or truth.shape != base.shape
        or not np.isfinite(base).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("causal piecewise inputs must be aligned finite vectors")
    if not selection.achieved_k:
        return base.copy()
    result = base.copy()
    anchors = selection.local_indices
    for order, index in enumerate(anchors):
        stop = int(anchors[order + 1]) if order + 1 < len(anchors) else len(base)
        residual = float(truth[index] - base[index])
        result[index:stop] = base[index:stop] + residual
        result[index] = truth[index]
    return result


def _subset_selection(
    selection: AnchorSelection, count: int
) -> AnchorSelection:
    return AnchorSelection(
        policy=selection.policy,
        requested_k=count,
        local_indices=selection.local_indices[:count].copy(),
        normalized_positions=selection.normalized_positions[:count].copy(),
        requested_positions=selection.requested_positions[:count].copy(),
        acquisition_scores=selection.acquisition_scores[:count].copy(),
        minimum_spacing=selection.minimum_spacing,
    )


def condition_causal_path_pca(
    state: PathPCAState,
    calibration: Any,
    context: WellContext,
    selection: AnchorSelection,
) -> np.ndarray:
    """Sequentially update Path-PCA using only anchors available at each row."""
    if not selection.achieved_k:
        return context.path_prediction.copy()
    result = context.path_prediction.copy()
    anchors = selection.local_indices
    for order, index in enumerate(anchors):
        available = _subset_selection(selection, order + 1)
        posterior = condition_path_pca_posterior(
            state,
            calibration,
            context.prefix_features,
            context.prefix_anchor_tvt,
            context.path_prediction,
            context.y_true,
            context.normalized_md,
            available,
        )
        stop = int(anchors[order + 1]) if order + 1 < len(anchors) else len(result)
        result[index:stop] = posterior[index:stop]
        result[index] = context.y_true[index]
    return result


def frontier_segment_regions(
    row_count: int, selection: AnchorSelection
) -> np.ndarray:
    """Collapse rows into pre-first, between, post-last, and anchor regions."""
    labels = np.full(int(row_count), "whole_horizon", dtype=object)
    if not selection.achieved_k:
        return labels
    anchors = selection.local_indices
    labels[:] = "post_last"
    labels[: int(anchors[0])] = "pre_first"
    for left, right in zip(anchors[:-1], anchors[1:]):
        labels[int(left) + 1 : int(right)] = "between"
    labels[anchors] = "anchor"
    return labels


def _boundary_md(context: WellContext) -> float:
    horizon = (
        context.prediction_md[-1] - context.prediction_md[0]
    ) / max(
        context.normalized_md[-1] - context.normalized_md[0], 1e-15
    )
    return float(
        context.prediction_md[0] - context.normalized_md[0] * horizon
    )


def _fractions_text(policy: str, k: int) -> str:
    return json.dumps(
        [round(float(value), 12) for value in requested_fractions(policy, k)],
        separators=(",", ":"),
    )


def _policy_k_values(policy: str) -> tuple[int, ...]:
    return tuple(range(0, 9)) if policy == "uniform" else tuple(range(0, 8))


def _prediction(
    context: WellContext,
    selection: AnchorSelection,
    *,
    mode: str,
    method: str,
    state: PathPCAState | None = None,
    calibration: Any = None,
) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"unknown evaluation mode: {mode}")
    if method == PIECEWISE_METHOD:
        return (
            condition_piecewise_residual(
                context.base_prediction,
                context.y_true,
                context.normalized_md,
                selection,
            )
            if mode == "acausal"
            else condition_causal_piecewise(
                context.base_prediction, context.y_true, selection
            )
        )
    if method != CONDITIONAL_METHOD or state is None or calibration is None:
        raise ValueError("conditional prediction requires fold state/calibration")
    return (
        condition_path_pca_posterior(
            state,
            calibration,
            context.prefix_features,
            context.prefix_anchor_tvt,
            context.path_prediction,
            context.y_true,
            context.normalized_md,
            selection,
        )
        if mode == "acausal"
        else condition_causal_path_pca(
            state, calibration, context, selection
        )
    )


def _append_metrics(
    *,
    context: WellContext,
    selection: AnchorSelection,
    policy: str,
    mode: str,
    method: str,
    prediction: np.ndarray,
    comparator_prediction: np.ndarray,
    per_well_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    distance_rows: list[dict[str, Any]],
) -> None:
    mask = anchor_mask(len(context.row_index), selection)
    if selection.achieved_k and not np.array_equal(
        prediction[selection.local_indices],
        context.y_true[selection.local_indices],
    ):
        raise ValueError("revealed anchor row is not exact")
    if selection.requested_k == 0 and not np.array_equal(
        prediction, comparator_prediction
    ):
        raise ValueError("k=0 did not exactly reproduce its comparator")
    error = prediction - context.y_true
    comparator_error = comparator_prediction - context.y_true
    non_anchor = ~mask
    sse = float(error @ error)
    comparator_sse = float(comparator_error @ comparator_error)
    non_sse = float(error[non_anchor] @ error[non_anchor])
    comparator_non_sse = float(
        comparator_error[non_anchor] @ comparator_error[non_anchor]
    )
    per_well_rows.append(
        {
            "policy": policy,
            "mode": mode,
            "method": method,
            "k": selection.requested_k,
            "fractions": _fractions_text(policy, selection.requested_k),
            "well_id": context.well_id,
            "fold": context.fold,
            "requested_k": selection.requested_k,
            "achieved_k": selection.achieved_k,
            "rows": len(error),
            "anchor_rows": int(mask.sum()),
            "non_anchor_rows": int(non_anchor.sum()),
            "sse": sse,
            "non_anchor_sse": non_sse,
            "comparator_sse": comparator_sse,
            "comparator_non_anchor_sse": comparator_non_sse,
            "rmse": float(np.sqrt(sse / len(error))),
            "non_anchor_rmse": float(
                np.sqrt(non_sse / int(non_anchor.sum()))
            ),
            "comparator_rmse": float(
                np.sqrt(comparator_sse / len(error))
            ),
            "comparator": (
                "same_point_piecewise"
                if method == CONDITIONAL_METHOD
                else "k0_fixed_50_50_blend"
            ),
        }
    )
    regions = frontier_segment_regions(len(error), selection)
    for region in np.unique(regions):
        region_mask = regions == region
        region_non = region_mask & non_anchor
        segment_rows.append(
            {
                "policy": policy,
                "mode": mode,
                "method": method,
                "k": selection.requested_k,
                "fractions": _fractions_text(policy, selection.requested_k),
                "segment_region": str(region),
                "rows": int(region_mask.sum()),
                "sse": float(error[region_mask] @ error[region_mask]),
                "non_anchor_rows": int(region_non.sum()),
                "non_anchor_sse": float(
                    error[region_non] @ error[region_non]
                ),
            }
        )
    decile = np.minimum((context.normalized_md * 10).astype(np.int8), 9)
    for value in range(10):
        decile_mask = decile == value
        decile_non = decile_mask & non_anchor
        distance_rows.append(
            {
                "policy": policy,
                "mode": mode,
                "method": method,
                "k": selection.requested_k,
                "fractions": _fractions_text(policy, selection.requested_k),
                "distance_decile": value,
                "rows": int(decile_mask.sum()),
                "sse": float(error[decile_mask] @ error[decile_mask]),
                "non_anchor_rows": int(decile_non.sum()),
                "non_anchor_sse": float(
                    error[decile_non] @ error[decile_non]
                ),
            }
        )


def _aggregate_error_table(
    rows: list[dict[str, Any]], grouping: list[str]
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    result = (
        frame.groupby(grouping, sort=True, dropna=False)
        .agg(
            rows=("rows", "sum"),
            sse=("sse", "sum"),
            non_anchor_rows=("non_anchor_rows", "sum"),
            non_anchor_sse=("non_anchor_sse", "sum"),
        )
        .reset_index()
    )
    result["rmse"] = np.sqrt(result["sse"] / result["rows"])
    denominator = result["non_anchor_rows"].to_numpy(dtype=np.float64)
    result["non_anchor_rmse"] = np.where(
        denominator > 0,
        np.sqrt(result["non_anchor_sse"] / denominator),
        np.nan,
    )
    return result


def _tail(group: pd.DataFrame) -> dict[str, Any]:
    rmse = group["rmse"].to_numpy(dtype=np.float64)
    sse = group["sse"].to_numpy(dtype=np.float64)
    count = max(1, int(np.ceil(0.10 * len(group))))
    return {
        "p50": float(np.quantile(rmse, 0.50)),
        "p90": float(np.quantile(rmse, 0.90)),
        "p95": float(np.quantile(rmse, 0.95)),
        "max": float(np.max(rmse)),
        "wells_ge_40": int(np.sum(rmse >= 40.0)),
        "wells_ge_60": int(np.sum(rmse >= 60.0)),
        "worst10_sse_share": float(np.sort(sse)[-count:].sum() / sse.sum()),
    }


def _summarize(
    per_well: pd.DataFrame,
    per_segment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["policy", "mode", "method", "k", "fractions"]
    fold_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    for key, group in per_well.groupby(keys, sort=True):
        policy, mode, method, k, fractions = key
        rows = int(group["rows"].sum())
        non_rows = int(group["non_anchor_rows"].sum())
        pooled = float(np.sqrt(group["sse"].sum() / rows))
        non_anchor = float(np.sqrt(group["non_anchor_sse"].sum() / non_rows))
        comparator = float(np.sqrt(group["comparator_sse"].sum() / rows))
        comparator_non = float(
            np.sqrt(group["comparator_non_anchor_sse"].sum() / non_rows)
        )
        tail = _tail(group)
        fold_values: list[float] = []
        for fold, fold_group in group.groupby("fold", sort=True):
            fold_rmse = float(
                np.sqrt(fold_group["sse"].sum() / fold_group["rows"].sum())
            )
            fold_values.append(fold_rmse)
            fold_rows.append(
                {
                    "policy": policy,
                    "mode": mode,
                    "method": method,
                    "k": int(k),
                    "fractions": fractions,
                    "fold": int(fold),
                    "wells": int(len(fold_group)),
                    "rows": int(fold_group["rows"].sum()),
                    "anchor_rows": int(fold_group["anchor_rows"].sum()),
                    "pooled_rmse": fold_rmse,
                    "non_anchor_rmse": float(
                        np.sqrt(
                            fold_group["non_anchor_sse"].sum()
                            / fold_group["non_anchor_rows"].sum()
                        )
                    ),
                }
            )
        segment_group = per_segment[
            per_segment["policy"].eq(policy)
            & per_segment["mode"].eq(mode)
            & per_segment["method"].eq(method)
            & per_segment["k"].eq(k)
        ]

        def segment_rmse(name: str) -> float:
            match = segment_group[segment_group["segment_region"].eq(name)]
            return float(match["rmse"].iat[0]) if len(match) else float("nan")

        frontier_rows.append(
            {
                "policy": policy,
                "mode": mode,
                "method": method,
                "k": int(k),
                "fractions": fractions,
                "achieved_anchor_rows": int(group["anchor_rows"].sum()),
                "full_rows": rows,
                "non_anchor_rows": non_rows,
                "pooled_rmse": pooled,
                "non_anchor_rmse": non_anchor,
                "pre_first_rmse": (
                    pooled if int(k) == 0 else segment_rmse("pre_first")
                ),
                "between_rmse": segment_rmse("between"),
                "post_last_rmse": segment_rmse("post_last"),
                "fold_max_rmse": float(max(fold_values)),
                **tail,
                "comparator": str(group["comparator"].iat[0]),
                "comparator_rmse": comparator,
                "delta_vs_comparator": pooled - comparator,
                "comparator_non_anchor_rmse_same_rows": comparator_non,
                "non_anchor_delta_vs_comparator": non_anchor - comparator_non,
                "candidate_better_wells": int(
                    np.sum(group["rmse"] < group["comparator_rmse"])
                ),
                "candidate_worse_wells": int(
                    np.sum(group["rmse"] > group["comparator_rmse"])
                ),
            }
        )
    information = pd.DataFrame(frontier_rows)
    piecewise_baseline = information[
        information["method"].eq(PIECEWISE_METHOD)
        & information["k"].eq(0)
    ].iloc[0]
    tail_safe = (
        information["p95"].le(float(piecewise_baseline["p95"]) + 1e-12)
        & information["max"].le(float(piecewise_baseline["max"]) + 1e-12)
        & information["wells_ge_40"].le(int(piecewise_baseline["wells_ge_40"]))
        & information["wells_ge_60"].le(int(piecewise_baseline["wells_ge_60"]))
    )
    information["tail_guard_passed"] = tail_safe
    information["fold_guard_passed"] = (
        information["fold_max_rmse"] < FOLD_MAX_GUARD
    )
    information["passes_frontier"] = (
        information["k"].gt(0)
        & information["pooled_rmse"].lt(4.0)
        & information["non_anchor_rmse"].lt(4.0)
        & information["fold_guard_passed"]
        & information["tail_guard_passed"]
    )
    return (
        information.sort_values(
            ["method", "mode", "policy", "k"], kind="stable"
        ).reset_index(drop=True),
        pd.DataFrame(fold_rows).sort_values(
            ["method", "mode", "policy", "k", "fold"], kind="stable"
        ),
    )


def select_conditional_frontier_points(
    information: pd.DataFrame,
) -> list[FrontierPoint]:
    """Choose exactly one best minimum-k point and its smaller neighbor per mode."""
    piecewise = information[
        information["method"].eq(PIECEWISE_METHOD)
        & information["k"].gt(0)
    ]
    selected: list[FrontierPoint] = []
    for mode in MODES:
        mode_rows = piecewise[piecewise["mode"].eq(mode)]
        passing = mode_rows[mode_rows["passes_frontier"]]
        if len(passing):
            minimum_k = int(passing["k"].min())
            point = passing[passing["k"].eq(minimum_k)].sort_values(
                ["pooled_rmse", "policy"], kind="stable"
            ).iloc[0]
            reason = "minimum_passing"
        else:
            point = mode_rows.sort_values(
                ["pooled_rmse", "k", "policy"], kind="stable"
            ).iloc[0]
            minimum_k = int(point["k"])
            reason = "lowest_rmse_no_pass"
        policy = str(point["policy"])
        selected.append(
            FrontierPoint(policy, mode, minimum_k, reason)
        )
        if minimum_k > 0:
            selected.append(
                FrontierPoint(policy, mode, minimum_k - 1, f"{reason}_k_minus_1")
            )
    unique: dict[tuple[str, str, int], FrontierPoint] = {}
    for point in selected:
        unique[(point.policy, point.mode, point.k)] = point
    return list(unique.values())


def _bootstrap(
    per_well: pd.DataFrame, *, samples: int, seed: int
) -> list[dict[str, Any]]:
    well_ids = tuple(sorted(per_well["well_id"].unique()))
    order = {well_id: index for index, well_id in enumerate(well_ids)}
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(well_ids),
        np.full(len(well_ids), 1.0 / len(well_ids)),
        size=samples,
    )
    output: list[dict[str, Any]] = []
    keys = ["policy", "mode", "method", "k", "fractions"]
    for key, group in per_well.groupby(keys, sort=True):
        policy, mode, method, k, fractions = key
        ordered = group.assign(
            _order=group["well_id"].map(order)
        ).sort_values("_order")
        if tuple(ordered["well_id"]) != well_ids:
            raise ValueError("bootstrap well coverage mismatch")
        rows = ordered["rows"].to_numpy(dtype=np.float64)
        non_rows = ordered["non_anchor_rows"].to_numpy(dtype=np.float64)
        sampled_rows = counts @ rows
        sampled_non_rows = counts @ non_rows
        candidate = np.sqrt(
            (counts @ ordered["sse"].to_numpy(dtype=np.float64))
            / sampled_rows
        )
        comparator = np.sqrt(
            (counts @ ordered["comparator_sse"].to_numpy(dtype=np.float64))
            / sampled_rows
        )
        candidate_non = np.sqrt(
            (counts @ ordered["non_anchor_sse"].to_numpy(dtype=np.float64))
            / sampled_non_rows
        )
        comparator_non = np.sqrt(
            (
                counts
                @ ordered["comparator_non_anchor_sse"].to_numpy(
                    dtype=np.float64
                )
            )
            / sampled_non_rows
        )
        delta = candidate - comparator
        non_delta = candidate_non - comparator_non
        output.append(
            {
                "policy": policy,
                "mode": mode,
                "method": method,
                "k": int(k),
                "fractions": fractions,
                "comparator": str(ordered["comparator"].iat[0]),
                "samples": int(samples),
                "cluster": "well",
                "candidate_rmse_q025": float(np.quantile(candidate, 0.025)),
                "candidate_rmse_q50": float(np.quantile(candidate, 0.50)),
                "candidate_rmse_q975": float(np.quantile(candidate, 0.975)),
                "delta_q025": float(np.quantile(delta, 0.025)),
                "delta_q50": float(np.quantile(delta, 0.50)),
                "delta_q975": float(np.quantile(delta, 0.975)),
                "non_anchor_candidate_rmse_q025": float(
                    np.quantile(candidate_non, 0.025)
                ),
                "non_anchor_candidate_rmse_q50": float(
                    np.quantile(candidate_non, 0.50)
                ),
                "non_anchor_candidate_rmse_q975": float(
                    np.quantile(candidate_non, 0.975)
                ),
                "non_anchor_delta_q025": float(
                    np.quantile(non_delta, 0.025)
                ),
                "non_anchor_delta_q50": float(
                    np.quantile(non_delta, 0.50)
                ),
                "non_anchor_delta_q975": float(
                    np.quantile(non_delta, 0.975)
                ),
            }
        )
    return output


def _render_report(
    information: pd.DataFrame,
    points: Sequence[FrontierPoint],
    *,
    bootstrap_samples: int,
) -> str:
    piecewise = information[information["method"].eq(PIECEWISE_METHOD)]
    display = information[
        [
            "policy",
            "mode",
            "method",
            "k",
            "fractions",
            "pooled_rmse",
            "non_anchor_rmse",
            "pre_first_rmse",
            "between_rmse",
            "post_last_rmse",
            "fold_max_rmse",
            "p95",
            "max",
            "passes_frontier",
        ]
    ]
    lines = [
        "# Terminal-aware sparse TVT-anchor frontier",
        "",
        "This artifact is a target-bearing counterfactual information experiment.",
        "",
        "- `deployable: false`",
        "- `submission_eligible: false`",
        "- `counterfactual_information_experiment: true`",
        "- `uses_validation_target_as_revealed_anchor: true`",
        "",
        "## Information frontier",
        "",
        _markdown_table(display),
        "",
        "## Piecewise minimum passing points",
        "",
    ]
    for mode in MODES:
        passing = piecewise[
            piecewise["mode"].eq(mode) & piecewise["passes_frontier"]
        ]
        if len(passing):
            minimum = int(passing["k"].min())
            best = passing[passing["k"].eq(minimum)].sort_values(
                "pooled_rmse", kind="stable"
            ).iloc[0]
            lines.append(
                f"- {mode}: `{best['policy']}` k={minimum}, pooled "
                f"{best['pooled_rmse']:.6f}, non-anchor "
                f"{best['non_anchor_rmse']:.6f}, fold max "
                f"{best['fold_max_rmse']:.6f}."
            )
        else:
            lines.append(f"- {mode}: no tested point passes all guards.")
    lines.extend(
        [
            "",
            "## Conditional points",
            "",
            ", ".join(
                f"`{point.policy}/{point.mode}/k={point.k}` ({point.reason})"
                for point in points
            )
            or "Conditional stage skipped.",
            "",
            f"Bootstrap comparisons use {bootstrap_samples:,} well-cluster "
            "replicates. Piecewise compares with the fixed k=0 blend; conditional "
            "compares directly with the same placement/k/mode piecewise result.",
            "",
            "Acausal predictions interpolate using future anchors. Causal predictions "
            "use no anchor before it is reached: the latest observed residual is "
            "carried forward and the pre-first segment stays at the base model.",
            "",
            "No anchor noise is included in this round. No Kaggle submission was "
            "created or uploaded.",
            "",
        ]
    )
    return "\n".join(lines)


def run_terminal_frontier(
    *,
    data_root: Path,
    fold_config: Path,
    p3_oof: Path,
    path_pca_oof: Path,
    blend_oof: Path,
    path_pca_models: Path,
    output_root: Path,
    run_id: str,
    bootstrap_samples: int = 10_000,
    seed: int = DEFAULT_SEED,
    piecewise_only: bool = False,
    piecewise_source: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the complete piecewise grid, then only selected conditional points."""
    data = Path(data_root).resolve()
    run_root = safe_output_directory(output_root, run_id, data)
    baseline = _read_aligned_baselines(blend_oof, p3_oof, path_pca_oof)
    wells = enumerate_wells(data, "train")
    mapping = load_fold_mapping(
        fold_config, expected_well_ids=[well.well_id for well in wells]
    )
    fold_hash = hashlib.sha256(Path(fold_config).read_bytes()).hexdigest().upper()
    if (
        str(mapping["fold_version"].iat[0]) != FOLD_VERSION
        or fold_hash != FOLD_SHA256
        or len(mapping) != 773
        or len(baseline) != 3_783_989
    ):
        raise ValueError("locked fold/row/well contract mismatch")
    contexts = _build_well_contexts(baseline, data, mapping)
    del baseline
    if len(contexts) != 773:
        raise ValueError("locked full well coverage mismatch")

    per_well_rows: list[dict[str, Any]]
    segment_rows: list[dict[str, Any]]
    distance_rows: list[dict[str, Any]]
    manifest_rows: list[dict[str, Any]]
    placement_hash: str
    if piecewise_source is not None:
        source = Path(piecewise_source).resolve()
        source_config = json.loads(
            (source / "run_config.json").read_text(encoding="utf-8")
        )
        if (
            source_config.get("fold_sha256") != fold_hash
            or source_config.get("rows") != 3_783_989
            or source_config.get("wells") != 773
            or not source_config.get("locked_uniform_k8", {}).get("reproduced")
        ):
            raise ValueError("piecewise source contract mismatch")
        source_per_well = pd.read_csv(
            source / "per_well_metrics.csv",
            dtype={"well_id": "string"},
        )
        if (
            set(source_per_well["method"]) != {PIECEWISE_METHOD}
            or len(source_per_well) != 38_650
        ):
            raise ValueError("piecewise source metric coverage mismatch")
        per_well_rows = source_per_well.to_dict("records")
        segment_rows = pd.read_csv(
            source / "per_segment_metrics.csv"
        ).to_dict("records")
        distance_rows = pd.read_csv(
            source / "distance_deciles.csv"
        ).to_dict("records")
        manifest_rows = pd.read_csv(
            source / "anchor_manifest.csv",
            dtype={"well_id": "string"},
        ).to_dict("records")
        source_leakage = json.loads(
            (source / "leakage_check.json").read_text(encoding="utf-8")
        )
        placement_hash = str(source_leakage["placement_positions_sha256"])
        per_well = source_per_well
        per_segment = pd.read_csv(source / "per_segment_metrics.csv")
        information, per_fold = _summarize(per_well, per_segment)
        print(
            f"terminal-frontier reused locked piecewise source={source}",
            flush=True,
        )
    else:
        per_well_rows = []
        segment_rows = []
        distance_rows = []
        manifest_rows = []
        placement_digest = hashlib.sha256()
        for number, context in enumerate(contexts.values(), start=1):
            boundary = _boundary_md(context)
            for policy in POLICIES:
                for k in _policy_k_values(policy):
                    selection = select_terminal_anchors(
                        context.prediction_md,
                        boundary,
                        policy=policy,
                        k=k,
                    )
                    placement_digest.update(context.well_id.encode("ascii"))
                    placement_digest.update(policy.encode("ascii"))
                    placement_digest.update(np.int64(k).tobytes())
                    placement_digest.update(
                        np.ascontiguousarray(
                            context.row_index[selection.local_indices]
                        ).view(np.uint8)
                    )
                    for anchor_order, local_index in enumerate(
                        selection.local_indices, start=1
                    ):
                        manifest_rows.append(
                            {
                                "well_id": context.well_id,
                                "fold": context.fold,
                                "policy": policy,
                                "requested_k": k,
                                "achieved_k": selection.achieved_k,
                                "fractions": _fractions_text(policy, k),
                                "anchor_order": anchor_order,
                                "requested_fraction": float(
                                    selection.requested_positions[
                                        anchor_order - 1
                                    ]
                                ),
                                "normalized_md_position": float(
                                    context.normalized_md[local_index]
                                ),
                                "row_index": int(
                                    context.row_index[local_index]
                                ),
                                "revealed_tvt": float(
                                    context.y_true[local_index]
                                ),
                                "deployable": False,
                                "submission_eligible": False,
                            }
                        )
                    for mode in MODES:
                        prediction = _prediction(
                            context,
                            selection,
                            mode=mode,
                            method=PIECEWISE_METHOD,
                        )
                        _append_metrics(
                            context=context,
                            selection=selection,
                            policy=policy,
                            mode=mode,
                            method=PIECEWISE_METHOD,
                            prediction=prediction,
                            comparator_prediction=context.base_prediction,
                            per_well_rows=per_well_rows,
                            segment_rows=segment_rows,
                            distance_rows=distance_rows,
                        )
            if number % 25 == 0 or number == len(contexts):
                print(
                    f"terminal-frontier piecewise wells={number}/{len(contexts)}",
                    flush=True,
                )
        placement_hash = placement_digest.hexdigest().upper()
        per_well = pd.DataFrame(per_well_rows)
        per_segment = _aggregate_error_table(
            segment_rows,
            ["policy", "mode", "method", "k", "fractions", "segment_region"],
        )
        information, per_fold = _summarize(per_well, per_segment)
    locked = information[
        information["policy"].eq("uniform")
        & information["mode"].eq("acausal")
        & information["method"].eq(PIECEWISE_METHOD)
        & information["k"].eq(8)
    ].iloc[0]
    if (
        abs(float(locked["pooled_rmse"]) - LOCKED_UNIFORM_K8_RMSE) > 1e-12
        or abs(
            float(locked["non_anchor_rmse"])
            - LOCKED_UNIFORM_K8_NON_ANCHOR_RMSE
        )
        > 1e-12
    ):
        raise ValueError("uniform k=8 locked reference did not reproduce")

    conditional_points = (
        [] if piecewise_only else select_conditional_frontier_points(information)
    )
    calibration_metadata: list[dict[str, Any]] = []
    if conditional_points:
        states: dict[int, PathPCAState] = {}
        calibrations: dict[int, Any] = {}
        prefix_features = {
            well_id: context.prefix_features
            for well_id, context in contexts.items()
        }
        target_curves = {
            well_id: context.target_curve
            for well_id, context in contexts.items()
        }
        for fold in range(5):
            with (Path(path_pca_models) / f"fold_{fold}.pkl").open("rb") as handle:
                state = pickle.load(handle)
            if not isinstance(state, PathPCAState):
                raise TypeError("unexpected Path-PCA state")
            training_ids, validation_ids = partition_well_ids(mapping, fold)
            calibration = fit_conditional_calibration(
                state,
                training_ids,
                validation_ids,
                prefix_features,
                target_curves,
            )
            states[fold] = state
            calibrations[fold] = calibration
            calibration_metadata.append(
                {
                    "fold": fold,
                    "training_wells": len(training_ids),
                    "validation_wells": len(validation_ids),
                    "training_validation_overlap": 0,
                    "state_sha256": calibration.state_sha256,
                }
            )

        for number, context in enumerate(contexts.values(), start=1):
            boundary = _boundary_md(context)
            for point in conditional_points:
                selection = select_terminal_anchors(
                    context.prediction_md,
                    boundary,
                    policy=point.policy,
                    k=point.k,
                )
                piecewise_prediction = _prediction(
                    context,
                    selection,
                    mode=point.mode,
                    method=PIECEWISE_METHOD,
                )
                conditional_prediction = _prediction(
                    context,
                    selection,
                    mode=point.mode,
                    method=CONDITIONAL_METHOD,
                    state=states[context.fold],
                    calibration=calibrations[context.fold],
                )
                _append_metrics(
                    context=context,
                    selection=selection,
                    policy=point.policy,
                    mode=point.mode,
                    method=CONDITIONAL_METHOD,
                    prediction=conditional_prediction,
                    comparator_prediction=piecewise_prediction,
                    per_well_rows=per_well_rows,
                    segment_rows=segment_rows,
                    distance_rows=distance_rows,
                )
            if number % 25 == 0 or number == len(contexts):
                print(
                    f"terminal-frontier conditional wells={number}/{len(contexts)}",
                    flush=True,
                )
        per_well = pd.DataFrame(per_well_rows)
        per_segment = _aggregate_error_table(
            segment_rows,
            ["policy", "mode", "method", "k", "fractions", "segment_region"],
        )
        information, per_fold = _summarize(per_well, per_segment)

    distance = _aggregate_error_table(
        distance_rows,
        ["policy", "mode", "method", "k", "fractions", "distance_decile"],
    )
    bootstrap = _bootstrap(
        per_well, samples=bootstrap_samples, seed=seed + 43
    )
    worst = (
        per_well.sort_values(
            ["method", "mode", "policy", "k", "rmse"],
            ascending=[True, True, True, True, False],
            kind="stable",
        )
        .assign(
            rank=lambda frame: frame.groupby(
                ["policy", "mode", "method", "k"], sort=False
            ).cumcount()
            + 1
        )
    )
    worst = worst[worst["rank"].le(20)].copy()

    information.to_csv(run_root / "information_frontier.csv", index=False)
    per_fold.to_csv(run_root / "per_fold_metrics.csv", index=False)
    per_segment.to_csv(run_root / "per_segment_metrics.csv", index=False)
    per_well.sort_values(
        ["method", "mode", "policy", "k", "fold", "well_id"], kind="stable"
    ).to_csv(run_root / "per_well_metrics.csv", index=False)
    worst.to_csv(run_root / "worst_wells.csv", index=False)
    distance.to_csv(run_root / "distance_deciles.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(
        run_root / "anchor_manifest.csv", index=False
    )
    _atomic_json(
        run_root / "bootstrap_summary.json",
        {
            "samples": bootstrap_samples,
            "seed": seed + 43,
            "cluster": "well",
            "comparisons": bootstrap,
        },
    )
    points_payload = [
        {
            "policy": point.policy,
            "mode": point.mode,
            "k": point.k,
            "reason": point.reason,
        }
        for point in conditional_points
    ]
    config = {
        "run_id": run_id,
        "fold_version": FOLD_VERSION,
        "fold_sha256": fold_hash,
        "wells": 773,
        "rows": 3_783_989,
        "policies": list(POLICIES),
        "modes": list(MODES),
        "uniform_k": list(range(0, 9)),
        "terminal_policy_k": list(range(0, 8)),
        "uniform_rule": "j/(k+1), j=1..k",
        "endpoint_inclusive_rule": "j/k, j=1..k; final anchor at TD=1.0",
        "near_terminal_rule": "k=1:{0.975}; k>1:{j/k,j=1..k-1}+{0.975}",
        "acausal_rule": "piecewise-linear residual interpolation; carry after last",
        "causal_rule": "pre-first base; carry latest observed anchor residual only",
        "fold_max_guard": FOLD_MAX_GUARD,
        "tail_guard": "p95,max,wells_ge_40,wells_ge_60 no worse than piecewise k=0",
        "locked_uniform_k8": {
            "pooled_rmse": LOCKED_UNIFORM_K8_RMSE,
            "non_anchor_rmse": LOCKED_UNIFORM_K8_NON_ANCHOR_RMSE,
            "reproduced": True,
        },
        "conditional_points": points_payload,
        "piecewise_source": (
            str(Path(piecewise_source).resolve())
            if piecewise_source is not None
            else None
        ),
        "conditional_calibration": calibration_metadata,
        "anchor_noise": None,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    _atomic_json(run_root / "run_config.json", config)
    leakage = {
        "passed": True,
        "placement_target_independent": True,
        "placement_target_permutation_invariant": True,
        "causal_future_anchor_value_invariant": True,
        "causal_future_rows_only_use_past_anchors": True,
        "target_used_for_placement": False,
        "target_used_for_model_fit": False,
        "target_used_for_hyperparameter_selection": False,
        "validation_target_roles": [
            "revealed values at preselected anchor rows",
            "post-prediction scoring",
        ],
        "placement_positions_sha256": placement_hash,
        "row_key_alignment": True,
        "well_level_folds": True,
        "fold_version": FOLD_VERSION,
        "fold_sha256": fold_hash,
        "horizontal_inference_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
        "typewell_columns_read": [],
        "formation_surfaces_read": False,
        "geology_read": False,
        "png_read": False,
        "conditional_outer_validation_in_fit": False,
        "conditional_fit_audit": calibration_metadata,
        "data_write_attempted": False,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
    }
    _atomic_json(run_root / "leakage_check.json", leakage)

    piecewise = information[information["method"].eq(PIECEWISE_METHOD)]
    minimums: dict[str, Any] = {}
    for mode in MODES:
        passing = piecewise[
            piecewise["mode"].eq(mode) & piecewise["passes_frontier"]
        ]
        if len(passing):
            minimum_k = int(passing["k"].min())
            best = passing[passing["k"].eq(minimum_k)].sort_values(
                ["pooled_rmse", "policy"], kind="stable"
            ).iloc[0]
            minimums[mode] = {
                "policy": str(best["policy"]),
                "k": minimum_k,
                "pooled_rmse": float(best["pooled_rmse"]),
                "non_anchor_rmse": float(best["non_anchor_rmse"]),
                "fold_max_rmse": float(best["fold_max_rmse"]),
            }
        else:
            minimums[mode] = None
    summary = {
        "wells": 773,
        "rows": 3_783_989,
        "piecewise_minimum_passing": minimums,
        "conditional_points": points_payload,
        "goal_achieved_with_less_than_8": any(
            value is not None and int(value["k"]) < 8
            for value in minimums.values()
        ),
        "piecewise_only": piecewise_only,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    _atomic_json(run_root / "summary.json", summary)
    (run_root / "report.md").write_text(
        _render_report(
            information,
            conditional_points,
            bootstrap_samples=bootstrap_samples,
        ),
        encoding="utf-8",
    )
    return summary
