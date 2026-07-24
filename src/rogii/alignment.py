"""Deterministic test-safe horizontal-GR to typewell-GR alignment features.

The typewell is always resampled on its physical TVT coordinate.  Candidate
selection is per prediction row and uses only inference-visible GR plus the
known-prefix anchor; there is deliberately no path continuity or target-aware
selection in P3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .data import select_inference_view, validate_prefix


ALIGNMENT_VERSION = "p3_physical_tvt_alignment_v1"


@dataclass(frozen=True)
class AlignmentConfig:
    scales_ft: tuple[float, ...] = (15.0, 45.0, 135.0)
    grid_step_ft: float = 1.0
    search_radius_ft: float = 128.0
    top_k: int = 3
    entropy_temperature: float = 1.0
    missing_cost: float = 4.0
    coverage_penalty: float = 0.5
    chunk_rows: int = 2048

    def __post_init__(self) -> None:
        if len(self.scales_ft) < 2 or any(scale <= 0 for scale in self.scales_ft):
            raise ValueError("Alignment requires at least two positive physical scales")
        if tuple(sorted(self.scales_ft)) != self.scales_ft or len(set(self.scales_ft)) != len(self.scales_ft):
            raise ValueError("Alignment physical scales must be unique and increasing")
        if self.grid_step_ft <= 0 or self.search_radius_ft <= 0 or self.top_k < 2:
            raise ValueError("Invalid alignment grid, search radius, or top-k")
        if self.entropy_temperature <= 0 or self.missing_cost <= 0 or self.chunk_rows <= 0:
            raise ValueError("Invalid alignment numerical configuration")


def _scale_label(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def _feature_columns(config: AlignmentConfig) -> tuple[str, ...]:
    columns: list[str] = [
        "alignment_candidate_count",
        "alignment_search_lower_tvt",
        "alignment_search_upper_tvt",
        "alignment_search_lower_clipped",
        "alignment_search_upper_clipped",
    ]
    for scale in config.scales_ft:
        label = _scale_label(scale)
        columns.extend(
            [
                f"alignment_horizontal_observation_count_{label}ft",
                f"alignment_horizontal_coverage_{label}ft",
                f"alignment_best_cost_{label}ft",
                f"alignment_selected_cost_{label}ft",
                f"alignment_selected_typewell_coverage_{label}ft",
            ]
        )
    for rank in range(1, config.top_k + 1):
        columns.extend(
            [
                f"alignment_candidate_{rank}_tvt",
                f"alignment_candidate_{rank}_offset_from_anchor",
                f"alignment_candidate_{rank}_cost",
                f"alignment_candidate_{rank}_raw_cost",
                f"alignment_candidate_{rank}_local_normalized_cost",
                f"alignment_candidate_{rank}_shape_cost",
                f"alignment_candidate_{rank}_coverage",
                f"alignment_candidate_{rank}_edge_flag",
            ]
        )
    columns.extend(
        [
            "alignment_cost_margin_top1_top2",
            "alignment_candidate_separation_top1_top2",
            "alignment_entropy",
            "alignment_effective_candidate_count",
            "alignment_best_is_search_edge",
        ]
    )
    return tuple(columns)


DEFAULT_ALIGNMENT_CONFIG = AlignmentConfig()
ALIGNMENT_FEATURE_COLUMNS = _feature_columns(DEFAULT_ALIGNMENT_CONFIG)


@dataclass(frozen=True)
class AlignmentFeatureFrame:
    features: pd.DataFrame
    row_index: np.ndarray
    candidate_tvt: np.ndarray
    candidate_cost: np.ndarray
    anchor: float
    metadata: dict[str, float | int]


def resample_typewell_physical(
    typewell: pd.DataFrame, *, grid_step_ft: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly resample GR on a fixed physical-TVT grid.

    The output coverage is interpolation support derived only from observed GR;
    it is one for the competition typewells, whose GR is complete.
    """
    if grid_step_ft <= 0:
        raise ValueError("grid_step_ft must be positive")
    frame = select_inference_view(typewell, kind="typewell")
    tvt = frame["TVT"].to_numpy(dtype=np.float64)
    gr = frame["GR"].to_numpy(dtype=np.float64)
    if len(tvt) < 2 or not np.isfinite(tvt).all() or np.any(np.diff(tvt) <= 0):
        raise ValueError("Typewell TVT must be finite and strictly increasing")
    finite = np.isfinite(gr)
    if finite.sum() < 2:
        raise ValueError("Typewell GR requires at least two finite observations")
    start = math.ceil(tvt[0] / grid_step_ft - 1e-10) * grid_step_ft
    stop = math.floor(tvt[-1] / grid_step_ft + 1e-10) * grid_step_ft
    grid = np.arange(start, stop + grid_step_ft * 0.5, grid_step_ft, dtype=np.float64)
    if len(grid) < 2:
        raise ValueError("Typewell physical TVT range is smaller than one grid interval")
    interpolated = np.interp(grid, tvt[finite], gr[finite]).astype(np.float64)
    support = np.interp(grid, tvt, finite.astype(np.float64)).astype(np.float32)
    support[(grid < tvt[finite][0]) | (grid > tvt[finite][-1])] = 0.0
    return grid, interpolated, np.clip(support, 0.0, 1.0)


def _physical_window_stats(
    coordinate: np.ndarray, values: np.ndarray, width_ft: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = width_ft / 2.0
    left = np.searchsorted(coordinate, coordinate - half, side="left")
    right = np.searchsorted(coordinate, coordinate + half, side="right")
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    cumulative_count = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    cumulative_sum = np.concatenate(([0.0], np.cumsum(safe, dtype=np.float64)))
    cumulative_square = np.concatenate(([0.0], np.cumsum(safe * safe, dtype=np.float64)))
    count = cumulative_count[right] - cumulative_count[left]
    total = right - left
    summed = cumulative_sum[right] - cumulative_sum[left]
    square = cumulative_square[right] - cumulative_square[left]
    mean = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
    variance = np.divide(square, count, out=np.zeros_like(square), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))
    coverage = np.divide(count, total, out=np.zeros_like(mean), where=total > 0)
    return mean, std, count.astype(np.float32), coverage.astype(np.float32)


def _global_standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=np.float32), finite
    center = float(np.mean(values[finite]))
    scale = max(float(np.std(values[finite])), 1e-6)
    result = np.zeros(len(values), dtype=np.float32)
    result[finite] = ((values[finite] - center) / scale).astype(np.float32)
    return result, finite


def _descriptors(
    coordinate: np.ndarray, values: np.ndarray, scales: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
    raw, raw_valid = _global_standardize(values)
    finite_values = values[np.isfinite(values)]
    fallback = float(np.mean(finite_values)) if len(finite_values) else 0.0
    global_scale = max(float(np.std(finite_values)) if len(finite_values) else 0.0, 1e-6)
    descriptors: list[dict[str, np.ndarray]] = []
    for scale in scales:
        mean, std, count, coverage = _physical_window_stats(coordinate, values, scale)
        local_valid = raw_valid & (count >= 2) & (std > 1e-6)
        local = np.zeros(len(values), dtype=np.float32)
        local[local_valid] = ((values[local_valid] - mean[local_valid]) / std[local_valid]).astype(np.float32)
        mean_valid = count > 0
        mean_level = ((mean - fallback) / global_scale).astype(np.float32)
        smooth = np.where(mean_valid, mean, fallback)
        shape = (np.gradient(smooth, coordinate) / global_scale).astype(np.float32)
        shape_valid = count >= 2
        descriptors.append(
            {
                "local": local,
                "local_valid": local_valid,
                "mean": mean_level,
                "mean_valid": mean_valid,
                "shape": shape,
                "shape_valid": shape_valid,
                "count": count,
                "coverage": coverage,
            }
        )
    return raw, raw_valid, descriptors


def _component_cost(
    left: np.ndarray,
    left_valid: np.ndarray,
    right: np.ndarray,
    right_valid: np.ndarray,
    missing_cost: float,
) -> np.ndarray:
    valid = left_valid[:, None] & right_valid[None, :]
    difference = left[:, None] - right[None, :]
    return np.where(valid, np.minimum(difference * difference, 100.0), missing_cost).astype(np.float32)


def build_alignment_features(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG,
) -> AlignmentFeatureFrame:
    """Build deterministic top-k alignment candidates and ambiguity features."""
    h = select_inference_view(horizontal, kind="horizontal")
    known_count, prediction_count = validate_prefix(h["TVT_input"])
    md = h["MD"].to_numpy(dtype=np.float64)
    gr = h["GR"].to_numpy(dtype=np.float64)
    if not np.isfinite(md).all() or np.any(np.diff(md) <= 0):
        raise ValueError("Horizontal MD must be finite and strictly increasing")
    anchor = float(h["TVT_input"].iloc[known_count - 1])
    if not np.isfinite(anchor):
        raise ValueError("Last-known TVT anchor must be finite")
    row_index = np.arange(known_count, len(h), dtype=np.int64)

    tvt_grid, typewell_gr, typewell_point_coverage = resample_typewell_physical(
        typewell, grid_step_ft=config.grid_step_ft
    )
    lower_requested = anchor - config.search_radius_ft
    upper_requested = anchor + config.search_radius_ft
    candidate_mask = (tvt_grid >= lower_requested - 1e-9) & (tvt_grid <= upper_requested + 1e-9)
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) < config.top_k:
        raise ValueError("Typewell range contains fewer candidates than top_k around anchor")
    candidate_tvt_grid = tvt_grid[candidate_indices]

    h_raw, h_raw_valid, h_desc = _descriptors(md, gr, config.scales_ft)
    t_raw, t_raw_valid, t_desc = _descriptors(tvt_grid, typewell_gr, config.scales_ft)
    t_raw = t_raw[candidate_indices]
    t_raw_valid = t_raw_valid[candidate_indices]
    for descriptor in t_desc:
        for key in descriptor:
            descriptor[key] = descriptor[key][candidate_indices]

    columns = _feature_columns(config)
    output = {column: np.zeros(prediction_count, dtype=np.float32) for column in columns}
    candidate_tvt = np.zeros((prediction_count, config.top_k), dtype=np.float64)
    candidate_cost = np.zeros((prediction_count, config.top_k), dtype=np.float32)
    pred_positions = row_index

    output["alignment_candidate_count"].fill(float(len(candidate_indices)))
    output["alignment_search_lower_tvt"].fill(float(candidate_tvt_grid[0]))
    output["alignment_search_upper_tvt"].fill(float(candidate_tvt_grid[-1]))
    output["alignment_search_lower_clipped"].fill(float(tvt_grid[0] > lower_requested + 1e-9))
    output["alignment_search_upper_clipped"].fill(float(tvt_grid[-1] < upper_requested - 1e-9))

    raw_t = t_raw
    raw_valid_t = t_raw_valid
    for begin in range(0, prediction_count, config.chunk_rows):
        end = min(begin + config.chunk_rows, prediction_count)
        horizontal_indices = pred_positions[begin:end]
        raw_cost = _component_cost(
            h_raw[horizontal_indices], h_raw_valid[horizontal_indices], raw_t, raw_valid_t, config.missing_cost
        )
        local_total = np.zeros_like(raw_cost)
        shape_total = np.zeros_like(raw_cost)
        coverage_total = np.zeros_like(raw_cost)
        scale_costs: list[np.ndarray] = []
        for scale_index, scale in enumerate(config.scales_ft):
            hd = h_desc[scale_index]
            td = t_desc[scale_index]
            local_cost = _component_cost(
                hd["local"][horizontal_indices], hd["local_valid"][horizontal_indices],
                td["local"], td["local_valid"], config.missing_cost,
            )
            mean_cost = _component_cost(
                hd["mean"][horizontal_indices], hd["mean_valid"][horizontal_indices],
                td["mean"], td["mean_valid"], config.missing_cost,
            )
            shape_cost = _component_cost(
                hd["shape"][horizontal_indices], hd["shape_valid"][horizontal_indices],
                td["shape"], td["shape_valid"], config.missing_cost,
            )
            pair_coverage = np.minimum(
                hd["coverage"][horizontal_indices, None],
                td["coverage"][None, :] * typewell_point_coverage[candidate_indices][None, :],
            ).astype(np.float32)
            scale_cost = (
                0.25 * raw_cost + 0.35 * local_cost + 0.15 * mean_cost + 0.25 * shape_cost
                + config.coverage_penalty * (1.0 - pair_coverage)
            ).astype(np.float32)
            scale_costs.append(scale_cost)
            local_total += local_cost
            shape_total += shape_cost
            coverage_total += pair_coverage
            label = _scale_label(scale)
            output[f"alignment_horizontal_observation_count_{label}ft"][begin:end] = hd["count"][horizontal_indices]
            output[f"alignment_horizontal_coverage_{label}ft"][begin:end] = hd["coverage"][horizontal_indices]
            output[f"alignment_best_cost_{label}ft"][begin:end] = np.min(scale_cost, axis=1)

        combined = np.mean(scale_costs, axis=0, dtype=np.float32)
        tie_break = np.arange(combined.shape[1], dtype=np.float32) * np.float32(1e-7)
        order = np.argsort(combined + tie_break[None, :], axis=1, kind="stable")[:, : config.top_k]
        top_cost = np.take_along_axis(combined, order, axis=1)
        top_tvt = candidate_tvt_grid[order]
        candidate_tvt[begin:end] = top_tvt
        candidate_cost[begin:end] = top_cost
        best = order[:, 0]
        rows = np.arange(end - begin)
        for scale_index, scale in enumerate(config.scales_ft):
            label = _scale_label(scale)
            output[f"alignment_selected_cost_{label}ft"][begin:end] = scale_costs[scale_index][rows, best]
            output[f"alignment_selected_typewell_coverage_{label}ft"][begin:end] = t_desc[scale_index]["coverage"][best]

        local_average = local_total / np.float32(len(config.scales_ft))
        shape_average = shape_total / np.float32(len(config.scales_ft))
        coverage_average = coverage_total / np.float32(len(config.scales_ft))
        for rank in range(config.top_k):
            selected = order[:, rank]
            number = rank + 1
            output[f"alignment_candidate_{number}_tvt"][begin:end] = top_tvt[:, rank]
            output[f"alignment_candidate_{number}_offset_from_anchor"][begin:end] = top_tvt[:, rank] - anchor
            output[f"alignment_candidate_{number}_cost"][begin:end] = top_cost[:, rank]
            output[f"alignment_candidate_{number}_raw_cost"][begin:end] = raw_cost[rows, selected]
            output[f"alignment_candidate_{number}_local_normalized_cost"][begin:end] = local_average[rows, selected]
            output[f"alignment_candidate_{number}_shape_cost"][begin:end] = shape_average[rows, selected]
            output[f"alignment_candidate_{number}_coverage"][begin:end] = coverage_average[rows, selected]
            output[f"alignment_candidate_{number}_edge_flag"][begin:end] = (
                (selected == 0) | (selected == len(candidate_tvt_grid) - 1)
            )

        shifted = (combined - np.min(combined, axis=1, keepdims=True)) / config.entropy_temperature
        weights = np.exp(-np.minimum(shifted, 80.0), dtype=np.float64)
        probabilities = weights / np.sum(weights, axis=1, keepdims=True)
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
        output["alignment_cost_margin_top1_top2"][begin:end] = top_cost[:, 1] - top_cost[:, 0]
        output["alignment_candidate_separation_top1_top2"][begin:end] = np.abs(top_tvt[:, 1] - top_tvt[:, 0])
        output["alignment_entropy"][begin:end] = entropy.astype(np.float32)
        output["alignment_effective_candidate_count"][begin:end] = np.exp(entropy).astype(np.float32)
        output["alignment_best_is_search_edge"][begin:end] = (
            (best == 0) | (best == len(candidate_tvt_grid) - 1)
        )

    features = pd.DataFrame({column: output[column] for column in columns}, columns=columns)
    if not np.isfinite(features.to_numpy(dtype=np.float32, copy=False)).all():
        raise ValueError("Alignment features must be finite after explicit missing-data handling")
    metadata = {
        "known_rows": known_count,
        "prediction_rows": prediction_count,
        "candidate_count": len(candidate_indices),
        "typewell_grid_rows": len(tvt_grid),
        "typewell_grid_step_ft": config.grid_step_ft,
    }
    return AlignmentFeatureFrame(features, row_index, candidate_tvt, candidate_cost, anchor, metadata)


def alignment_manifest(config: AlignmentConfig = DEFAULT_ALIGNMENT_CONFIG) -> dict[str, Any]:
    return {
        "alignment_version": ALIGNMENT_VERSION,
        "feature_columns": list(_feature_columns(config)),
        "dtype": "float32",
        "horizontal_source_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
        "typewell_source_columns": ["TVT", "GR"],
        "config": asdict(config),
        "candidate_policy": "per-row top-k by deterministic multiscale emission cost; no target or path continuity",
        "coordinate_policy": "typewell linearly resampled on physical TVT grid; horizontal windows use physical MD",
        "cost_components": ["globally-standardized raw GR", "local-normalized GR", "local mean level", "first derivative of local mean"],
    }
