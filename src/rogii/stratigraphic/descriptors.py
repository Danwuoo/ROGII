"""Deterministic, target-free multi-scale GR window descriptors.

The public functions in this module accept coordinate and GR vectors only.
Horizontal callers are responsible for supplying MD; typewell callers supply
the inference-time typewell coordinate. No learned statistic is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DescriptorConfig:
    """Configuration shared by horizontal and typewell descriptor builders."""

    points: int = 33
    minimum_observed_fraction: float = 0.80
    flat_epsilon: float = 1e-6
    maximum_gap_fraction: float = 0.25
    smoothing_widths: tuple[int, ...] = (3, 5, 9)

    def __post_init__(self) -> None:
        if self.points < 9 or self.points % 2 != 1:
            raise ValueError("Descriptor points must be an odd integer >= 9")
        if not 0.0 < self.minimum_observed_fraction <= 1.0:
            raise ValueError("minimum_observed_fraction must be in (0, 1]")
        if self.flat_epsilon <= 0.0:
            raise ValueError("flat_epsilon must be positive")
        if not 0.0 < self.maximum_gap_fraction <= 1.0:
            raise ValueError("maximum_gap_fraction must be in (0, 1]")
        if not self.smoothing_widths or any(
            width < 1 or width % 2 != 1 for width in self.smoothing_widths
        ):
            raise ValueError("smoothing_widths must contain positive odd widths")


@dataclass(frozen=True)
class ResampledWindow:
    """One coordinate-aware GR window and its observation metadata."""

    eligible: bool
    values: np.ndarray
    observed_mask: np.ndarray
    observed_fraction: float
    low_information: bool
    reason: str = ""


@dataclass(frozen=True)
class DescriptorBatch:
    """Descriptors for the eligible subset of requested center rows."""

    rows: np.ndarray
    coordinates: np.ndarray
    center_gr: np.ndarray
    raw_shape: np.ndarray
    derivative: np.ndarray
    rank: np.ndarray
    barcode: np.ndarray
    signature: np.ndarray
    observed_mask: np.ndarray
    observed_fraction: np.ndarray
    low_information: np.ndarray
    requested_rows: int
    eligible_rows: int
    half_span: float

    def take(self, indices: np.ndarray) -> "DescriptorBatch":
        selected = np.asarray(indices, dtype=np.int64)
        return DescriptorBatch(
            rows=self.rows[selected],
            coordinates=self.coordinates[selected],
            center_gr=self.center_gr[selected],
            raw_shape=self.raw_shape[selected],
            derivative=self.derivative[selected],
            rank=self.rank[selected],
            barcode=self.barcode[selected],
            signature=self.signature[selected],
            observed_mask=self.observed_mask[selected],
            observed_fraction=self.observed_fraction[selected],
            low_information=self.low_information[selected],
            requested_rows=int(selected.size),
            eligible_rows=int(selected.size),
            half_span=self.half_span,
        )


def _empty_batch(
    *,
    requested_rows: int,
    half_span: float,
    config: DescriptorConfig,
) -> DescriptorBatch:
    barcode_width = (
        len(config.smoothing_widths) * (4 * config.points - 2)
        + config.points
        + 2
    )
    return DescriptorBatch(
        rows=np.empty((0,), dtype=np.int64),
        coordinates=np.empty((0,), dtype=np.float64),
        center_gr=np.empty((0,), dtype=np.float64),
        raw_shape=np.empty((0, config.points), dtype=np.float64),
        derivative=np.empty((0, 2 * config.points - 3), dtype=np.float64),
        rank=np.empty((0, config.points), dtype=np.float64),
        barcode=np.empty((0, barcode_width), dtype=np.float64),
        signature=np.empty((0, 18), dtype=np.float64),
        observed_mask=np.empty((0, config.points), dtype=bool),
        observed_fraction=np.empty((0,), dtype=np.float64),
        low_information=np.empty((0,), dtype=bool),
        requested_rows=int(requested_rows),
        eligible_rows=0,
        half_span=float(half_span),
    )


def _validated_source(
    coordinate: Iterable[float], gr: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(coordinate, dtype=np.float64)
    values = np.asarray(gr, dtype=np.float64)
    if x.ndim != 1 or values.ndim != 1 or x.size != values.size or x.size < 3:
        raise ValueError("coordinate and GR must be aligned one-dimensional vectors")
    if not np.isfinite(x).all() or not np.all(np.diff(x) > 0.0):
        raise ValueError("descriptor coordinate must be finite and strictly increasing")
    return x, values


def _maximum_interpolation_gap(
    coordinate: np.ndarray, half_span: float, config: DescriptorConfig
) -> float:
    step = float(np.median(np.diff(coordinate)))
    return max(4.0 * step, config.maximum_gap_fraction * half_span)


def _window_observed_fraction(
    coordinate: np.ndarray,
    finite: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    starts = np.searchsorted(coordinate, left, side="left")
    stops = np.searchsorted(coordinate, right, side="right")
    prefix = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(finite, dtype=np.int64))
    )
    counts = stops - starts
    observed = prefix[stops] - prefix[starts]
    return np.divide(
        observed,
        counts,
        out=np.zeros_like(observed, dtype=np.float64),
        where=counts > 0,
    )


def _resample_many(
    coordinate: np.ndarray,
    gr: np.ndarray,
    rows: np.ndarray,
    *,
    half_span: float,
    region_start: int,
    region_stop: int,
    config: DescriptorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return resampled values, observed masks, fractions, and eligibility."""
    if half_span <= 0.0 or not math.isfinite(half_span):
        raise ValueError("half_span must be finite and positive")
    if region_start < 0 or region_stop > coordinate.size or region_start >= region_stop:
        raise ValueError("invalid descriptor region")
    if rows.ndim != 1 or np.any(rows < region_start) or np.any(rows >= region_stop):
        raise ValueError("center rows must lie inside the descriptor region")

    centers = coordinate[rows]
    left = centers - half_span
    right = centers + half_span
    boundary_ok = (left >= coordinate[region_start]) & (
        right <= coordinate[region_stop - 1]
    )
    finite = np.isfinite(gr)
    observed_fraction = _window_observed_fraction(
        coordinate, finite, left, right
    )
    fraction_ok = observed_fraction >= config.minimum_observed_fraction

    offsets = np.linspace(-half_span, half_span, config.points, dtype=np.float64)
    grid = centers[:, None] + offsets[None, :]
    finite_coordinate = coordinate[finite]
    finite_gr = gr[finite]
    values = np.full(grid.shape, np.nan, dtype=np.float64)
    observed_mask = np.zeros(grid.shape, dtype=bool)
    gap_ok = np.zeros(grid.shape, dtype=bool)
    if finite_coordinate.size >= 2:
        flat_grid = grid.ravel()
        interpolated = np.interp(flat_grid, finite_coordinate, finite_gr)
        values[:] = interpolated.reshape(grid.shape)
        positions = np.searchsorted(finite_coordinate, flat_grid, side="left")
        exact_position = np.minimum(positions, finite_coordinate.size - 1)
        exact = np.isclose(
            finite_coordinate[exact_position],
            flat_grid,
            rtol=0.0,
            atol=max(1e-10, float(np.median(np.diff(coordinate))) * 1e-8),
        )
        left_position = np.maximum(positions - 1, 0)
        right_position = np.minimum(positions, finite_coordinate.size - 1)
        bracket_gap = finite_coordinate[right_position] - finite_coordinate[left_position]
        inside = (flat_grid >= finite_coordinate[0]) & (
            flat_grid <= finite_coordinate[-1]
        )
        maximum_gap = _maximum_interpolation_gap(coordinate, half_span, config)
        flat_gap_ok = inside & (exact | (bracket_gap <= maximum_gap))
        gap_ok[:] = flat_gap_ok.reshape(grid.shape)
        nearest_distance = np.minimum(
            np.abs(flat_grid - finite_coordinate[left_position]),
            np.abs(finite_coordinate[right_position] - flat_grid),
        )
        observed_tolerance = 0.75 * float(np.median(np.diff(coordinate)))
        observed_mask[:] = (nearest_distance <= observed_tolerance).reshape(grid.shape)

    eligible = boundary_ok & fraction_ok & gap_ok.all(axis=1)
    return values, observed_mask, observed_fraction, eligible


def _robust_normalize(
    values: np.ndarray, *, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=1, keepdims=True)
    centered = values - median
    mad = np.median(np.abs(centered), axis=1, keepdims=True)
    robust_scale = 1.4826 * mad
    low_information = robust_scale[:, 0] < epsilon
    safe_scale = np.where(robust_scale < epsilon, 1.0, robust_scale)
    normalized = centered / safe_scale
    normalized[low_information] = 0.0
    return normalized, low_information


def _rank_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.argsort(order, axis=1, kind="stable").astype(np.float64)
    denominator = max(1, values.shape[1] - 1)
    return ranks / float(denominator)


def _normalize_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    scale = np.sqrt(np.mean(values * values, axis=1, keepdims=True))
    return values / np.where(scale < epsilon, 1.0, scale)


def _smooth_rows(values: np.ndarray, width: int) -> np.ndarray:
    if width == 1:
        return values.copy()
    radius = width // 2
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    cumulative = np.cumsum(padded, axis=1, dtype=np.float64)
    cumulative = np.pad(cumulative, ((0, 0), (1, 0)), mode="constant")
    return (cumulative[:, width:] - cumulative[:, :-width]) / float(width)


def _barcode_features(
    raw: np.ndarray,
    observed_mask: np.ndarray,
    observed_fraction: np.ndarray,
    low_information: np.ndarray,
    config: DescriptorConfig,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for width in config.smoothing_widths:
        smooth = _smooth_rows(raw, width)
        gradient = np.diff(smooth, axis=1)
        sign = np.sign(gradient)
        left = gradient[:, :-1]
        right = gradient[:, 1:]
        extrema = np.zeros_like(left)
        extrema[(left > 0.0) & (right <= 0.0)] = 1.0
        extrema[(left < 0.0) & (right >= 0.0)] = -1.0
        prominence = np.abs(
            smooth[:, 1:-1]
            - 0.5 * (smooth[:, :-2] + smooth[:, 2:])
        )
        prominence *= np.abs(extrema)
        level = np.sign(smooth - np.median(smooth, axis=1, keepdims=True))
        crossing = np.count_nonzero(level[:, 1:] != level[:, :-1], axis=1)
        positive = np.count_nonzero(extrema > 0.0, axis=1)
        negative = np.count_nonzero(extrema < 0.0, axis=1)
        counts = np.column_stack((positive, negative, crossing)).astype(np.float64)
        counts /= float(config.points)
        blocks.extend((sign, extrema, prominence, level, counts))
    blocks.extend(
        (
            observed_mask.astype(np.float64),
            observed_fraction[:, None],
            low_information.astype(np.float64)[:, None],
        )
    )
    return np.concatenate(blocks, axis=1)


def _compact_signature(
    raw: np.ndarray,
    derivative: np.ndarray,
    barcode: np.ndarray,
    config: DescriptorConfig,
) -> np.ndarray:
    raw_positions = np.linspace(0, config.points - 1, 9).round().astype(np.int64)
    first_difference = derivative[:, : config.points - 1]
    derivative_positions = np.linspace(
        0, first_difference.shape[1] - 1, 7
    ).round().astype(np.int64)
    event_tail = barcode[:, -2:]
    signature = np.concatenate(
        (
            raw[:, raw_positions],
            first_difference[:, derivative_positions],
            event_tail,
        ),
        axis=1,
    )
    return _normalize_rows(signature, epsilon=config.flat_epsilon)


def _descriptors_from_resampled(
    values: np.ndarray,
    observed_mask: np.ndarray,
    observed_fraction: np.ndarray,
    config: DescriptorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw, low_information = _robust_normalize(
        values, epsilon=config.flat_epsilon
    )
    first = _normalize_rows(
        np.diff(raw, axis=1), epsilon=config.flat_epsilon
    )
    second = _normalize_rows(
        np.diff(raw, n=2, axis=1), epsilon=config.flat_epsilon
    )
    derivative = np.concatenate((first, second), axis=1)
    rank = _rank_rows(raw)
    barcode = _barcode_features(
        raw,
        observed_mask,
        observed_fraction,
        low_information,
        config,
    )
    signature = _compact_signature(raw, derivative, barcode, config)
    return raw, derivative, rank, barcode, signature, low_information


def resample_gr_window(
    coordinate: Iterable[float],
    gr: Iterable[float],
    *,
    center_row: int,
    half_span: float,
    region_start: int = 0,
    region_stop: int | None = None,
    config: DescriptorConfig | None = None,
) -> ResampledWindow:
    """Resample one GR window using real coordinates and short-gap interpolation."""
    descriptor_config = config or DescriptorConfig()
    x, values = _validated_source(coordinate, gr)
    stop = x.size if region_stop is None else int(region_stop)
    rows = np.array([center_row], dtype=np.int64)
    sampled, observed_mask, observed_fraction, eligible = _resample_many(
        x,
        values,
        rows,
        half_span=half_span,
        region_start=region_start,
        region_stop=stop,
        config=descriptor_config,
    )
    if not bool(eligible[0]):
        reason = (
            "insufficient_observation_or_gap"
            if observed_fraction[0] < descriptor_config.minimum_observed_fraction
            or not np.isfinite(sampled[0]).all()
            else "boundary_or_long_gap"
        )
        return ResampledWindow(
            eligible=False,
            values=sampled[0],
            observed_mask=observed_mask[0],
            observed_fraction=float(observed_fraction[0]),
            low_information=False,
            reason=reason,
        )
    raw, low_information = _robust_normalize(
        sampled, epsilon=descriptor_config.flat_epsilon
    )
    return ResampledWindow(
        eligible=True,
        values=raw[0],
        observed_mask=observed_mask[0],
        observed_fraction=float(observed_fraction[0]),
        low_information=bool(low_information[0]),
    )


def build_descriptor_batch(
    coordinate: Iterable[float],
    gr: Iterable[float],
    *,
    center_rows: Iterable[int],
    half_span: float,
    region_start: int = 0,
    region_stop: int | None = None,
    config: DescriptorConfig | None = None,
) -> DescriptorBatch:
    """Build all three handcrafted descriptor families for eligible centers."""
    descriptor_config = config or DescriptorConfig()
    x, values = _validated_source(coordinate, gr)
    rows = np.asarray(tuple(center_rows), dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("center_rows must be one-dimensional")
    if rows.size == 0:
        return _empty_batch(
            requested_rows=0,
            half_span=half_span,
            config=descriptor_config,
        )
    if np.unique(rows).size != rows.size:
        raise ValueError("center_rows must not contain duplicates")
    rows = np.sort(rows, kind="stable")
    stop = x.size if region_stop is None else int(region_stop)
    sampled, observed_mask, observed_fraction, eligible = _resample_many(
        x,
        values,
        rows,
        half_span=half_span,
        region_start=region_start,
        region_stop=stop,
        config=descriptor_config,
    )
    selected_rows = rows[eligible]
    selected_values = sampled[eligible]
    selected_mask = observed_mask[eligible]
    selected_fraction = observed_fraction[eligible]
    if selected_rows.size == 0:
        return _empty_batch(
            requested_rows=int(rows.size),
            half_span=half_span,
            config=descriptor_config,
        )
    raw, derivative, rank, barcode, signature, low_information = (
        _descriptors_from_resampled(
            selected_values,
            selected_mask,
            selected_fraction,
            descriptor_config,
        )
    )
    return DescriptorBatch(
        rows=selected_rows,
        coordinates=x[selected_rows],
        center_gr=values[selected_rows],
        raw_shape=raw,
        derivative=derivative,
        rank=rank,
        barcode=barcode,
        signature=signature,
        observed_mask=selected_mask,
        observed_fraction=selected_fraction,
        low_information=low_information,
        requested_rows=int(rows.size),
        eligible_rows=int(selected_rows.size),
        half_span=float(half_span),
    )
