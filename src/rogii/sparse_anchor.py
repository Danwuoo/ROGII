"""Inference-safe sparse-anchor placement, descriptors, and reconstruction.

This module contains no target-file loading and has no dependency on the
target-bearing retrospective diagnostics.  The only target-aware helper,
``build_anchor_delta_targets``, accepts an explicit training-only target array
and is called from the model's outer-training path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .alignment import ALIGNMENT_FEATURE_COLUMNS
from .data import select_inference_view, validate_prefix
from .features import FEATURE_COLUMNS


ANCHOR_FRACTIONS = (
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    0.975,
)
DESCRIPTOR_VERSION = "sparse_anchor_student_descriptor_v1"
PREFIX_RESAMPLE_POINTS = 32

AGGREGATE_FEATURE_COLUMNS = (
    "md_from_ps",
    "normalized_distance_from_ps",
    "dx_from_ps",
    "dy_from_ps",
    "dz_from_ps",
    "displacement_3d_from_ps",
    "tangent_x",
    "tangent_y",
    "tangent_z",
    "inclination_proxy",
    "dogleg_proxy",
    "gr_raw",
    "gr_is_missing",
    "gr_nearest_observed_distance",
    "gr_first_difference",
    "gr_roll_mean_51",
    "gr_roll_std_51",
    "gr_roll_observed_fraction_51",
    "alignment_candidate_1_offset_from_anchor",
    "alignment_candidate_1_cost",
    "alignment_candidate_1_coverage",
    "alignment_cost_margin_top1_top2",
    "alignment_entropy",
    "alignment_best_is_search_edge",
)


@dataclass(frozen=True)
class AnchorPlacement:
    """The fixed eight snapped rows for one prediction suffix."""

    local_indices: np.ndarray
    row_indices: np.ndarray
    normalized_positions: np.ndarray
    selected_positions: np.ndarray
    requested_positions: np.ndarray


@dataclass(frozen=True)
class StudentDescriptor:
    values: np.ndarray
    names: tuple[str, ...]


def normalized_suffix_md(
    prediction_md: Sequence[float] | np.ndarray, boundary_md: float
) -> np.ndarray:
    md = np.asarray(prediction_md, dtype=np.float64)
    if (
        md.ndim != 1
        or not len(md)
        or not np.isfinite(md).all()
        or np.any(np.diff(md) <= 0)
        or not np.isfinite(boundary_md)
        or md[0] <= float(boundary_md)
    ):
        raise ValueError("prediction MD must be finite, increasing, and after PS")
    horizon = float(md[-1] - float(boundary_md))
    if horizon <= 0:
        raise ValueError("prediction horizon must be positive")
    return (md - float(boundary_md)) / horizon


def _nearest_unused(
    positions: np.ndarray, requested: np.ndarray
) -> np.ndarray:
    """Legacy-equivalent deterministic nearest-unused selection."""
    available = set(range(len(positions)))
    selected: list[int] = []
    for value in requested:
        if not available:
            break
        index = min(
            available,
            key=lambda candidate: (
                abs(float(positions[candidate]) - float(value)),
                candidate,
            ),
        )
        selected.append(index)
        available.remove(index)
    return np.asarray(selected, dtype=np.int64)


def select_anchor_rows(
    prediction_md: Sequence[float] | np.ndarray,
    boundary_md: float,
    *,
    row_indices: Sequence[int] | np.ndarray | None = None,
) -> AnchorPlacement:
    """Snap the locked eight fractions to distinct increasing suffix rows."""
    md = np.asarray(prediction_md, dtype=np.float64)
    positions = normalized_suffix_md(md, boundary_md)
    requested = np.asarray(ANCHOR_FRACTIONS, dtype=np.float64)
    if len(positions) < len(requested):
        raise ValueError("prediction suffix has fewer than eight anchorable rows")

    # Use the fast collision-free choice first, with the legacy greedy rule as
    # the exact fallback.  Ties always select the smaller row index.
    right = np.searchsorted(positions, requested, side="left")
    right = np.minimum(right, len(positions) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(positions[left] - requested) <= np.abs(
        positions[right] - requested
    )
    selected = np.where(choose_left, left, right).astype(np.int64)
    if len(np.unique(selected)) != len(requested):
        selected = _nearest_unused(positions, requested)
    if (
        selected.shape != requested.shape
        or len(np.unique(selected)) != len(selected)
        or np.any(np.diff(selected) <= 0)
    ):
        raise ValueError("snapped anchor rows must be distinct and increasing")

    physical = (
        np.asarray(row_indices, dtype=np.int64)
        if row_indices is not None
        else np.arange(len(md), dtype=np.int64)
    )
    if physical.shape != md.shape or np.any(np.diff(physical) <= 0):
        raise ValueError("row indices must align and be strictly increasing")
    return AnchorPlacement(
        local_indices=selected,
        row_indices=physical[selected],
        normalized_positions=positions,
        selected_positions=positions[selected],
        requested_positions=requested.copy(),
    )


def reconstruct_from_predicted_anchors(
    baseline_prediction: Sequence[float] | np.ndarray,
    predicted_anchor_tvt: Sequence[float] | np.ndarray,
    placement: AnchorPlacement,
) -> np.ndarray:
    """Apply fixed PS-zero, linear, then final-carry baseline corrections."""
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    anchors = np.asarray(predicted_anchor_tvt, dtype=np.float64)
    positions = np.asarray(placement.normalized_positions, dtype=np.float64)
    indices = np.asarray(placement.local_indices, dtype=np.int64)
    if (
        baseline.ndim != 1
        or positions.shape != baseline.shape
        or anchors.shape != (len(ANCHOR_FRACTIONS),)
        or indices.shape != anchors.shape
        or not np.isfinite(baseline).all()
        or not np.isfinite(anchors).all()
        or not np.isfinite(positions).all()
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("piecewise reconstruction inputs are invalid")
    corrections = anchors - baseline[indices]
    knots_x = np.concatenate(([0.0], positions[indices]))
    knots_y = np.concatenate(([0.0], corrections))
    correction = np.interp(
        positions, knots_x, knots_y, right=float(corrections[-1])
    )
    result = baseline + correction
    result[indices] = anchors
    if not np.isfinite(result).all():
        raise ValueError("piecewise reconstruction produced non-finite values")
    return result


def build_anchor_delta_targets(
    target: Sequence[float] | np.ndarray,
    *,
    last_known_tvt: float,
    placement: AnchorPlacement,
) -> np.ndarray:
    """Build the eight training-only targets: true anchor minus visible PS TVT."""
    labels = np.asarray(target, dtype=np.float64)
    if (
        labels.ndim != 1
        or labels.shape != placement.normalized_positions.shape
        or not np.isfinite(labels).all()
        or not np.isfinite(last_known_tvt)
    ):
        raise ValueError("training target must cover the finite prediction suffix")
    values = labels[placement.local_indices] - float(last_known_tvt)
    if values.shape != (len(ANCHOR_FRACTIONS),) or not np.isfinite(values).all():
        raise ValueError("invalid sparse-anchor training target")
    return values


def _finite_stats(values: np.ndarray) -> tuple[float, ...]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0.0)
    q10, q50, q90 = np.quantile(finite, [0.1, 0.5, 0.9])
    return (
        float(np.mean(finite)),
        float(np.std(finite)),
        float(q10),
        float(q50),
        float(q90),
        float(len(finite)),
    )


def _sample_prefix(
    md: np.ndarray, values: np.ndarray, known_count: int
) -> tuple[list[float], list[str]]:
    ps = known_count - 1
    start_md = max(float(md[0]), float(md[ps]) - 1_000.0)
    rows = np.flatnonzero((np.arange(len(md)) <= ps) & (md >= start_md))
    if len(rows) < 2:
        rows = np.arange(max(0, known_count - 2), known_count)
    coordinate = (md[rows] - md[ps]) / max(float(md[ps] - start_md), 1.0)
    query = np.linspace(-1.0, 0.0, PREFIX_RESAMPLE_POINTS)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for column, label, scale in (
        (5, "prefix_tvt_rel", 1.0),
        (1, "prefix_x_rel_kft", 1_000.0),
        (2, "prefix_y_rel_kft", 1_000.0),
        (3, "prefix_z_rel_hft", 100.0),
    ):
        source = values[rows, column]
        if not np.isfinite(source).all():
            raise ValueError(f"{label} source must be finite")
        sampled = np.interp(query, coordinate, source)
        sampled = (sampled - sampled[-1]) / scale
        blocks.append(sampled)
        names.extend(f"{label}_{index:02d}" for index in range(len(query)))
    return np.concatenate(blocks).astype(float).tolist(), names


def build_student_descriptor(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    p3_features: np.ndarray,
    prediction_row_index: np.ndarray,
    placement: AnchorPlacement,
) -> StudentDescriptor:
    """Build one deterministic fixed-width descriptor from inference-safe data."""
    h = select_inference_view(horizontal, kind="horizontal")
    # Selecting the typewell inference view is an explicit schema guard even
    # though the P3 row features already contain its summaries.
    select_inference_view(typewell, kind="typewell")
    known_count, prediction_count = validate_prefix(h["TVT_input"])
    rows = np.asarray(prediction_row_index, dtype=np.int64)
    matrix = np.asarray(p3_features, dtype=np.float64)
    expected_width = len(FEATURE_COLUMNS) + len(ALIGNMENT_FEATURE_COLUMNS)
    if (
        rows.shape != (prediction_count,)
        or not np.array_equal(rows, np.arange(known_count, len(h)))
        or matrix.shape != (prediction_count, expected_width)
        or placement.normalized_positions.shape != (prediction_count,)
    ):
        raise ValueError("descriptor inputs do not cover the prediction suffix")
    if np.isinf(matrix).any():
        raise ValueError("P3 descriptor features may contain NaN but never infinity")

    raw = h.loc[:, ["MD", "X", "Y", "Z", "GR", "TVT_input"]].to_numpy(
        dtype=np.float64, copy=True
    )
    md, x, y, z, gr = (raw[:, index] for index in range(5))
    ps = known_count - 1
    suffix = slice(known_count, len(h))
    delta_xyz = np.asarray(
        [x[-1] - x[ps], y[-1] - y[ps], z[-1] - z[ps]], dtype=np.float64
    )
    md_step = np.diff(md)
    tangent_x = np.gradient(x, md)
    tangent_y = np.gradient(y, md)
    tangent_z = np.gradient(z, md)
    values: list[float] = [
        float(len(h)),
        float(known_count),
        float(prediction_count),
        float(md[ps] - md[0]),
        float(md[-1] - md[ps]),
        float(md[-1] - md[0]),
        float(h["TVT_input"].iloc[ps]),
        float(np.mean(md_step)),
        float(np.std(md_step)),
        float(delta_xyz[0] / 1_000.0),
        float(delta_xyz[1] / 1_000.0),
        float(delta_xyz[2] / 100.0),
        float(np.linalg.norm(delta_xyz) / 1_000.0),
        float(tangent_x[ps]),
        float(tangent_y[ps]),
        float(tangent_z[ps]),
        float(tangent_x[-1]),
        float(tangent_y[-1]),
        float(tangent_z[-1]),
    ]
    names = [
        "row_count",
        "known_row_count",
        "prediction_row_count",
        "prefix_md_span",
        "suffix_md_horizon",
        "total_md_span",
        "last_known_tvt",
        "md_step_mean",
        "md_step_std",
        "suffix_dx_kft",
        "suffix_dy_kft",
        "suffix_dz_hft",
        "suffix_displacement_kft",
        "ps_tangent_x",
        "ps_tangent_y",
        "ps_tangent_z",
        "end_tangent_x",
        "end_tangent_y",
        "end_tangent_z",
    ]
    for label, source in (
        ("all_gr", gr),
        ("prefix_gr", gr[:known_count]),
        ("suffix_gr", gr[suffix]),
    ):
        stats = _finite_stats(source)
        values.extend(stats[:-1])
        values.append(1.0 - stats[-1] / len(source))
        names.extend(
            [
                f"{label}_mean",
                f"{label}_std",
                f"{label}_q10",
                f"{label}_q50",
                f"{label}_q90",
                f"{label}_missing_fraction",
            ]
        )
    sampled_values, sampled_names = _sample_prefix(md, raw, known_count)
    values.extend(sampled_values)
    names.extend(sampled_names)

    p3_columns = tuple(FEATURE_COLUMNS) + tuple(ALIGNMENT_FEATURE_COLUMNS)
    for anchor_order, local_index in enumerate(placement.local_indices, start=1):
        values.extend(matrix[int(local_index)].tolist())
        names.extend(
            f"a{anchor_order}_{column}" for column in p3_columns
        )

    column_index = {name: index for index, name in enumerate(p3_columns)}
    for column in AGGREGATE_FEATURE_COLUMNS:
        source = matrix[:, column_index[column]]
        stats = _finite_stats(source)
        values.extend(stats[:5])
        names.extend(
            f"suffix_{column}_{stat}"
            for stat in ("mean", "std", "q10", "q50", "q90")
        )

    descriptor = np.asarray(values, dtype=np.float64)
    if descriptor.ndim != 1 or np.isinf(descriptor).any():
        raise ValueError("student descriptor must be one-dimensional without infinity")
    if len(names) != len(descriptor) or len(set(names)) != len(names):
        raise ValueError("student descriptor names must be aligned and unique")
    return StudentDescriptor(descriptor, tuple(names))


def descriptor_manifest() -> dict[str, object]:
    return {
        "descriptor_version": DESCRIPTOR_VERSION,
        "anchor_fractions": list(ANCHOR_FRACTIONS),
        "prefix_resample_points": PREFIX_RESAMPLE_POINTS,
        "local_row_feature_columns": list(FEATURE_COLUMNS)
        + list(ALIGNMENT_FEATURE_COLUMNS),
        "aggregate_feature_columns": list(AGGREGATE_FEATURE_COLUMNS),
        "horizontal_source_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
        "typewell_source_columns": ["TVT", "GR"],
        "missing_policy": "NaN retained; outer-training median imputer, training-range clip, then scaling/model",
        "uses_learned_baseline_prediction": False,
        "uses_target": False,
    }
