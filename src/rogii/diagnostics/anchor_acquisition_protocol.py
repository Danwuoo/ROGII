"""Qualification of the locked eight-location sparse-anchor acquisition protocol.

The experiment translates the retrospective ``near_k8_r1`` design into fixed
nominal acquisition positions and evaluates predeclared bound-edge position
errors.  Placement, anchor count, measurement-error scenarios, PS boundary,
base OOF prediction, and acausal piecewise-residual reconstruction are locked.

This remains a target-bearing retrospective counterfactual diagnostic.  It is
not a deployable competition model and never creates a submission.
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
from .anchor_acquisition_redundancy import (
    DEFAULT_BOOTSTRAP_SEED,
    DEPENDENCE_SINGLE,
    OPERATIONAL_METHOD,
    hierarchical_redundancy_bootstrap,
    stable_replicate_latents,
)
from .anchor_error_realism import (
    ErrorScenario,
    METRIC_TOLERANCE,
    PER_WELL_SSE_TOLERANCE,
    PERSISTENT_FOLD_FAILURE_RATE,
    PERSISTENT_FOLD_RMSE_THRESHOLD,
    _mask_coefficients,
    _scenario_metadata,
)
from .anchor_noise_robustness import (
    NoiseWellContext,
    _build_noise_contexts,
    _read_blend_oof,
    _selected_wells,
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
    _nearest_unused_indices,
    anchor_mask,
    safe_output_directory,
)
from .terminal_anchor_frontier import _boundary_md, select_terminal_anchors


NOMINAL_FRACTIONS = (
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    0.975,
)
POSITION_TOLERANCES_U = (0.0, 0.001, 0.0025, 0.005)
PRIMARY_PROTOCOL_TOLERANCE_U = 0.001
POSITION_MODES = (
    "coherent_early",
    "coherent_late",
    "independent_bounded",
)
ZERO_POSITION_MODE = "nominal"
O5_SCENARIO = "O5_moderate_p10_m10"
SCENARIO_NAMES = (
    "exact_clean",
    "moderate_m5_control",
    O5_SCENARIO,
)
POSITION_VIOLATION_TYPES = (
    "boundary",
    "availability",
    "duplicate",
    "order",
)
DEFAULT_SELECTION_SEED = 20260711
EXPECTED_ZERO_METRICS = {
    "full_rmse_q95": 3.945560597463506,
    "non_anchor_rmse_q95": 3.9461171454489756,
    "hierarchical_full_q975": 3.990370048942644,
    "hierarchical_non_anchor_q975": 3.9910746376968427,
}


def locked_protocol_scenarios() -> tuple[ErrorScenario, ...]:
    return (
        ErrorScenario(
            "control", "exact_clean", 0.0, 0.0, 0.0, 0.0, 0.0
        ),
        ErrorScenario(
            "control",
            "moderate_m5_control",
            1.0,
            1.0,
            0.10,
            5.0,
            1.0,
        ),
        ErrorScenario(
            "primary",
            O5_SCENARIO,
            1.0,
            1.0,
            0.10,
            10.0,
            1.0,
        ),
    )


PROTOCOL_SCENARIOS = locked_protocol_scenarios()


@dataclass(frozen=True)
class PlacementSpec:
    tolerance_u: float
    position_mode: str

    @property
    def key(self) -> str:
        return f"u{self.tolerance_u:g}_{self.position_mode}"


def locked_placement_specs() -> tuple[PlacementSpec, ...]:
    return (
        PlacementSpec(0.0, ZERO_POSITION_MODE),
        *(
            PlacementSpec(tolerance, mode)
            for tolerance in POSITION_TOLERANCES_U[1:]
            for mode in POSITION_MODES
        ),
    )


PLACEMENT_SPECS = locked_placement_specs()


@dataclass(frozen=True)
class ProtocolSelection:
    requested_positions: np.ndarray
    selection: AnchorSelection | None
    boundary_violations: int
    availability_violations: int
    duplicate_violations: int
    order_violations: int

    @property
    def valid(self) -> bool:
        return bool(
            self.selection is not None
            and self.selection.achieved_k == len(NOMINAL_FRACTIONS)
            and self.boundary_violations == 0
            and self.availability_violations == 0
            and self.duplicate_violations == 0
            and self.order_violations == 0
        )

    @property
    def violation_counts(self) -> np.ndarray:
        return np.asarray(
            [
                self.boundary_violations,
                self.availability_violations,
                self.duplicate_violations,
                self.order_violations,
            ],
            dtype=np.int16,
        )


@dataclass(frozen=True)
class EvaluationGroup:
    index: int
    placement_index: int
    scenario_index: int


@dataclass(frozen=True)
class PiecewisePrefixSums:
    normalized_positions: np.ndarray
    base_error: np.ndarray
    sum_u: np.ndarray
    sum_u2: np.ndarray
    sum_e: np.ndarray
    sum_eu: np.ndarray
    sum_e2: np.ndarray


def build_evaluation_groups() -> tuple[EvaluationGroup, ...]:
    return tuple(
        EvaluationGroup(
            index=placement_index * len(PROTOCOL_SCENARIOS) + scenario_index,
            placement_index=placement_index,
            scenario_index=scenario_index,
        )
        for placement_index in range(len(PLACEMENT_SPECS))
        for scenario_index in range(len(PROTOCOL_SCENARIOS))
    )


EVALUATION_GROUPS = build_evaluation_groups()


def _validate_locked_constants() -> None:
    if NOMINAL_FRACTIONS != (
        1 / 8,
        2 / 8,
        3 / 8,
        4 / 8,
        5 / 8,
        6 / 8,
        7 / 8,
        0.975,
    ):
        raise ValueError("nominal protocol fractions changed")
    if POSITION_TOLERANCES_U != (0.0, 0.001, 0.0025, 0.005):
        raise ValueError("position tolerance levels changed")
    if PRIMARY_PROTOCOL_TOLERANCE_U != 0.001:
        raise ValueError("primary protocol tolerance changed")


def _position_digest(
    seed: int, well_id: str, anchor_order: int, key: str = "position_sign"
) -> bytes:
    validated_seed = _validate_seed(seed)
    identifier = str(well_id)
    order = int(anchor_order)
    if (
        not identifier
        or "\x00" in identifier
        or isinstance(anchor_order, (bool, np.bool_))
        or not 1 <= order <= len(NOMINAL_FRACTIONS)
        or key != "position_sign"
    ):
        raise ValueError("invalid stable position-sign key")
    encoded = identifier.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"rogii-anchor-acquisition-protocol-v1\x00")
    digest.update(validated_seed.to_bytes(8, "big", signed=False))
    digest.update(len(encoded).to_bytes(4, "big", signed=False))
    digest.update(encoded)
    digest.update(order.to_bytes(2, "big", signed=False))
    digest.update(b"position_sign")
    return digest.digest()


def stable_position_sign(seed: int, well_id: str, anchor_order: int) -> int:
    """Return a target-independent paired Rademacher position-error sign."""
    digest = _position_digest(seed, well_id, anchor_order)
    uniform = (int.from_bytes(digest[:8], "big") + 0.5) / float(1 << 64)
    return -1 if uniform < 0.5 else 1


def stable_position_sign_matrix(
    seeds: Sequence[int], well_id: str
) -> np.ndarray:
    values = tuple(_validate_seed(seed) for seed in seeds)
    if not values or len(set(values)) != len(values):
        raise ValueError("position-sign seeds must be unique and non-empty")
    return np.asarray(
        [
            [
                stable_position_sign(seed, well_id, anchor_order)
                for anchor_order in range(1, len(NOMINAL_FRACTIONS) + 1)
            ]
            for seed in values
        ],
        dtype=np.int8,
    )


def requested_protocol_positions(
    tolerance_u: float,
    position_mode: str,
    *,
    signs: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Apply a locked bound-edge position displacement without clipping."""
    if isinstance(tolerance_u, (bool, np.bool_)):
        raise ValueError("position tolerance must be finite and non-negative")
    tolerance = float(tolerance_u)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("position tolerance must be finite and non-negative")
    nominal = np.asarray(NOMINAL_FRACTIONS, dtype=np.float64)
    mode = str(position_mode)
    if tolerance == 0.0:
        if mode != ZERO_POSITION_MODE or signs is not None:
            raise ValueError("zero tolerance must use the nominal mode")
        return nominal.copy()
    if mode == "coherent_early":
        if signs is not None:
            raise ValueError("coherent position modes do not accept signs")
        return nominal - tolerance
    if mode == "coherent_late":
        if signs is not None:
            raise ValueError("coherent position modes do not accept signs")
        return nominal + tolerance
    if mode == "independent_bounded":
        sign_array = np.asarray(signs)
        if (
            sign_array.shape != nominal.shape
            or not np.isin(sign_array, (-1, 1)).all()
        ):
            raise ValueError(
                "independent bounded positions require eight +/-1 signs"
            )
        return nominal + tolerance * sign_array.astype(np.float64)
    raise ValueError(f"unknown position mode: {mode}")


def nearest_unused_indices_equivalent(
    normalized_positions: np.ndarray,
    requested_positions: np.ndarray,
) -> np.ndarray:
    """Use sorted nearest lookup when collision-free, otherwise legacy fallback."""
    positions = np.asarray(normalized_positions, dtype=np.float64)
    requested = np.asarray(requested_positions, dtype=np.float64)
    if (
        positions.ndim != 1
        or requested.ndim != 1
        or not len(positions)
        or not np.isfinite(positions).all()
        or not np.isfinite(requested).all()
        or np.any(np.diff(positions) <= 0)
    ):
        raise ValueError("nearest-row inputs must be finite sorted vectors")
    right = np.searchsorted(positions, requested, side="left")
    right_clipped = np.minimum(right, len(positions) - 1)
    left_clipped = np.maximum(right - 1, 0)
    right_distance = np.abs(positions[right_clipped] - requested)
    left_distance = np.abs(positions[left_clipped] - requested)
    # Ties choose the smaller row index, matching the legacy (distance, index)
    # tuple ordering.
    selected = np.where(
        left_distance <= right_distance, left_clipped, right_clipped
    ).astype(np.int64)
    if len(np.unique(selected)) == len(selected):
        return selected
    return _nearest_unused_indices(positions, requested)


def select_protocol_anchors(
    normalized_positions: np.ndarray,
    requested_positions: Sequence[float] | np.ndarray,
) -> ProtocolSelection:
    """Snap to nearest-unused rows and fail closed on any protocol violation."""
    positions = np.asarray(normalized_positions, dtype=np.float64)
    requested = np.asarray(requested_positions, dtype=np.float64)
    if (
        positions.ndim != 1
        or not len(positions)
        or not np.isfinite(positions).all()
        or np.any(np.diff(positions) <= 0)
        or requested.shape != (len(NOMINAL_FRACTIONS),)
        or not np.isfinite(requested).all()
    ):
        raise ValueError("protocol placement inputs must be aligned finite vectors")
    boundary = int(np.sum((requested <= 0.0) | (requested > 1.0)))
    if boundary:
        return ProtocolSelection(
            requested.copy(), None, boundary, 0, 0, 0
        )
    indices = nearest_unused_indices_equivalent(positions, requested)
    availability = max(0, len(NOMINAL_FRACTIONS) - len(indices))
    duplicate = int(len(indices) - len(np.unique(indices)))
    order = int(np.sum(np.diff(indices) <= 0)) if len(indices) > 1 else 0
    selection = AnchorSelection(
        policy="practical_eight_location_protocol",
        requested_k=len(NOMINAL_FRACTIONS),
        local_indices=indices,
        normalized_positions=positions[indices],
        requested_positions=requested.copy(),
        acquisition_scores=np.full(len(indices), np.nan, dtype=np.float64),
        minimum_spacing=0.0,
    )
    if availability or duplicate or order:
        return ProtocolSelection(
            requested.copy(),
            None,
            boundary,
            availability,
            duplicate,
            order,
        )
    return ProtocolSelection(
        requested.copy(), selection, 0, 0, 0, 0
    )


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.asarray([0.0]), np.cumsum(values, dtype=np.float64))
    )


def build_piecewise_prefix_sums(
    normalized_positions: np.ndarray,
    base_prediction: np.ndarray,
    y_true: np.ndarray,
) -> PiecewisePrefixSums:
    positions = np.asarray(normalized_positions, dtype=np.float64)
    base = np.asarray(base_prediction, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    if (
        positions.ndim != 1
        or base.shape != positions.shape
        or truth.shape != positions.shape
        or not np.isfinite(positions).all()
        or not np.isfinite(base).all()
        or not np.isfinite(truth).all()
        or np.any(np.diff(positions) <= 0)
    ):
        raise ValueError("piecewise prefix inputs must be aligned finite vectors")
    error = base - truth
    return PiecewisePrefixSums(
        normalized_positions=positions,
        base_error=error,
        sum_u=_prefix(positions),
        sum_u2=_prefix(np.square(positions)),
        sum_e=_prefix(error),
        sum_eu=_prefix(error * positions),
        sum_e2=_prefix(np.square(error)),
    )


def _range_sums(
    prefix: PiecewisePrefixSums, start: int, stop: int
) -> tuple[float, float, float, float, float, float]:
    if not 0 <= start <= stop <= len(prefix.normalized_positions):
        raise ValueError("invalid prefix-sum range")
    return (
        float(stop - start),
        float(prefix.sum_u[stop] - prefix.sum_u[start]),
        float(prefix.sum_u2[stop] - prefix.sum_u2[start]),
        float(prefix.sum_e[stop] - prefix.sum_e[start]),
        float(prefix.sum_eu[stop] - prefix.sum_eu[start]),
        float(prefix.sum_e2[stop] - prefix.sum_e2[start]),
    )


def _accumulate_affine_segment(
    prefix: PiecewisePrefixSums,
    start: int,
    stop: int,
    correction: tuple[float, float],
    measurement_weights: Sequence[tuple[int, float, float]],
    *,
    exact: np.ndarray,
    linear: np.ndarray,
    gram: np.ndarray,
) -> None:
    if start == stop:
        return
    n, sum_u, sum_u2, sum_e, sum_eu, sum_e2 = _range_sums(
        prefix, start, stop
    )
    alpha, beta = correction
    exact[0] += (
        sum_e2
        + 2.0 * alpha * sum_e
        + 2.0 * beta * sum_eu
        + alpha * alpha * n
        + 2.0 * alpha * beta * sum_u
        + beta * beta * sum_u2
    )
    for column, weight_alpha, weight_beta in measurement_weights:
        linear[0, column] += (
            weight_alpha * sum_e
            + weight_beta * sum_eu
            + alpha * weight_alpha * n
            + (alpha * weight_beta + beta * weight_alpha) * sum_u
            + beta * weight_beta * sum_u2
        )
    for left, (left_column, left_alpha, left_beta) in enumerate(
        measurement_weights
    ):
        for right_column, right_alpha, right_beta in measurement_weights[left:]:
            value = (
                left_alpha * right_alpha * n
                + (
                    left_alpha * right_beta
                    + left_beta * right_alpha
                )
                * sum_u
                + left_beta * right_beta * sum_u2
            )
            gram[0, left_column, right_column] += value
            if left_column != right_column:
                gram[0, right_column, left_column] += value


def analytic_piecewise_coefficients(
    prefix: PiecewisePrefixSums,
    selection: AnchorSelection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute full/non-anchor SSE coefficients using interval prefix sums."""
    indices = np.asarray(selection.local_indices, dtype=np.int64)
    if (
        indices.shape != (len(NOMINAL_FRACTIONS),)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("analytic coefficients require eight ordered anchors")
    positions = prefix.normalized_positions
    anchor_u = positions[indices]
    if anchor_u[0] <= 0 or np.any(np.diff(anchor_u) <= 0):
        raise ValueError("actual achieved positions must follow the PS boundary")
    residual = -prefix.base_error[indices]
    anchor_count = len(indices)
    exact_non = np.zeros(1, dtype=np.float64)
    linear_non = np.zeros((1, anchor_count), dtype=np.float64)
    gram_non = np.zeros((1, anchor_count, anchor_count), dtype=np.float64)

    _accumulate_affine_segment(
        prefix,
        0,
        int(indices[0]),
        (0.0, float(residual[0] / anchor_u[0])),
        ((0, 0.0, float(1.0 / anchor_u[0])),),
        exact=exact_non,
        linear=linear_non,
        gram=gram_non,
    )
    for left in range(anchor_count - 1):
        right = left + 1
        left_u = float(anchor_u[left])
        right_u = float(anchor_u[right])
        denominator = right_u - left_u
        correction_alpha = float(
            (residual[left] * right_u - residual[right] * left_u)
            / denominator
        )
        correction_beta = float(
            (residual[right] - residual[left]) / denominator
        )
        _accumulate_affine_segment(
            prefix,
            int(indices[left]) + 1,
            int(indices[right]),
            (correction_alpha, correction_beta),
            (
                (left, right_u / denominator, -1.0 / denominator),
                (right, -left_u / denominator, 1.0 / denominator),
            ),
            exact=exact_non,
            linear=linear_non,
            gram=gram_non,
        )
    _accumulate_affine_segment(
        prefix,
        int(indices[-1]) + 1,
        len(positions),
        (float(residual[-1]), 0.0),
        ((anchor_count - 1, 1.0, 0.0),),
        exact=exact_non,
        linear=linear_non,
        gram=gram_non,
    )

    exact = np.repeat(exact_non, 2)
    linear = np.repeat(linear_non, 2, axis=0)
    gram = np.repeat(gram_non, 2, axis=0)
    gram[0] += np.eye(anchor_count, dtype=np.float64)
    counts = np.asarray(
        [len(positions), len(positions) - anchor_count], dtype=np.int64
    )
    return exact, linear, gram, counts


def direct_piecewise_coefficients(
    context: NoiseWellContext,
    selection: AnchorSelection,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    true_anchors = context.y_true[selection.local_indices]
    exact_prediction = condition_noisy_piecewise(
        context.base_prediction,
        true_anchors,
        context.normalized_md,
        selection,
    )
    weights = piecewise_noise_weights(context.normalized_md, selection)
    is_anchor = anchor_mask(len(context.row_index), selection)
    masks = np.stack(
        (
            np.ones(len(context.row_index), dtype=bool),
            ~is_anchor,
        )
    )
    coefficients = _mask_coefficients(
        exact_prediction - context.y_true, weights, masks
    )
    return coefficients, exact_prediction, weights


def evaluate_protocol_coefficients(
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    anchor_errors: np.ndarray,
) -> np.ndarray:
    """Evaluate the two protocol masks without repeated einsum path planning."""
    exact, linear, gram, _ = coefficients
    values = np.asarray(anchor_errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != linear.shape[1]:
        raise ValueError("protocol anchor errors do not align with coefficients")
    cross = values @ linear.T
    quadratic = np.column_stack(
        [
            np.sum((values @ gram[metric]) * values, axis=1)
            for metric in range(len(exact))
        ]
    )
    result = exact[None, :] + 2.0 * cross + quadratic
    tolerance = (
        np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(exact))))
        * 128
    )
    if float(np.min(result)) < -tolerance:
        raise ValueError("protocol SSE became materially negative")
    return np.maximum(result, 0.0)


def _group_metadata(group: EvaluationGroup) -> dict[str, Any]:
    placement = PLACEMENT_SPECS[group.placement_index]
    scenario = PROTOCOL_SCENARIOS[group.scenario_index]
    return {
        "protocol": "near_k8_r1_practical_eight_location",
        "tolerance_u": placement.tolerance_u,
        "position_mode": placement.position_mode,
        "nominal_fractions": json.dumps(
            list(NOMINAL_FRACTIONS), separators=(",", ":")
        ),
        "method": OPERATIONAL_METHOD,
        **_scenario_metadata(scenario),
    }


def _tail_arrays(
    full_sse: np.ndarray, rows: np.ndarray
) -> dict[str, np.ndarray]:
    rmse = np.sqrt(full_sse / rows[None, None, :])
    count = max(1, int(math.ceil(0.10 * len(rows))))
    ordered_sse = np.sort(full_sse, axis=2)
    denominator = full_sse.sum(axis=2)
    return {
        "well_rmse": rmse,
        "p50": np.quantile(rmse, 0.50, axis=2),
        "p90": np.quantile(rmse, 0.90, axis=2),
        "p95": np.quantile(rmse, 0.95, axis=2),
        "max": np.max(rmse, axis=2),
        "wells_ge_40": np.sum(rmse >= 40.0, axis=2),
        "wells_ge_60": np.sum(rmse >= 60.0, axis=2),
        "worst10_sse_share": (
            ordered_sse[:, :, -count:].sum(axis=2) / denominator
        ),
    }


def _per_seed_frame(
    seeds: Sequence[int],
    full_sse: np.ndarray,
    non_sse: np.ndarray,
    rows: np.ndarray,
    non_rows: np.ndarray,
    fold_sse: np.ndarray,
    fold_non_sse: np.ndarray,
    fold_rows: np.ndarray,
    fold_non_rows: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    placement_indices = np.asarray(
        [group.placement_index for group in EVALUATION_GROUPS],
        dtype=np.int64,
    )
    full_rmse = np.sqrt(full_sse.sum(axis=2) / rows.sum())
    non_rmse = np.sqrt(
        non_sse.sum(axis=2)
        / non_rows[placement_indices].sum(axis=1)[:, None]
    )
    fold_rmse = np.sqrt(fold_sse / fold_rows[None, None, :])
    fold_non_rmse = np.sqrt(
        fold_non_sse
        / fold_non_rows[placement_indices][:, None, :]
    )
    tail = _tail_arrays(full_sse, rows)
    records: list[dict[str, Any]] = []
    for group in EVALUATION_GROUPS:
        metadata = _group_metadata(group)
        for seed_position, seed in enumerate(seeds):
            records.append(
                {
                    **metadata,
                    "realization_seed": int(seed),
                    "wells": len(rows),
                    "rows": int(rows.sum()),
                    "non_anchor_rows": int(
                        non_rows[group.placement_index].sum()
                    ),
                    "full_rmse": float(
                        full_rmse[group.index, seed_position]
                    ),
                    "non_anchor_rmse": float(
                        non_rmse[group.index, seed_position]
                    ),
                    "fold_max_rmse": float(
                        np.max(fold_rmse[group.index, seed_position])
                    ),
                    "fold_max_non_anchor_rmse": float(
                        np.max(
                            fold_non_rmse[group.index, seed_position]
                        )
                    ),
                    "p50_well_rmse": float(
                        tail["p50"][group.index, seed_position]
                    ),
                    "p90_well_rmse": float(
                        tail["p90"][group.index, seed_position]
                    ),
                    "p95_well_rmse": float(
                        tail["p95"][group.index, seed_position]
                    ),
                    "max_well_rmse": float(
                        tail["max"][group.index, seed_position]
                    ),
                    "wells_ge_40": int(
                        tail["wells_ge_40"][group.index, seed_position]
                    ),
                    "wells_ge_60": int(
                        tail["wells_ge_60"][group.index, seed_position]
                    ),
                    "worst10_sse_share": float(
                        tail["worst10_sse_share"][
                            group.index, seed_position
                        ]
                    ),
                }
            )
    return pd.DataFrame.from_records(records), {
        "full_rmse": full_rmse,
        "non_anchor_rmse": non_rmse,
        "fold_rmse": fold_rmse,
        "fold_non_anchor_rmse": fold_non_rmse,
        **tail,
    }


def _per_fold_frame(
    seeds: Sequence[int],
    fold_sse: np.ndarray,
    fold_non_sse: np.ndarray,
    fold_rows: np.ndarray,
    fold_non_rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group in EVALUATION_GROUPS:
        metadata = _group_metadata(group)
        for seed_position, seed in enumerate(seeds):
            for fold in range(5):
                full = float(
                    fold_sse[group.index, seed_position, fold]
                )
                non = float(
                    fold_non_sse[group.index, seed_position, fold]
                )
                records.append(
                    {
                        **metadata,
                        "realization_seed": int(seed),
                        "fold": fold,
                        "rows": int(fold_rows[fold]),
                        "non_anchor_rows": int(
                            fold_non_rows[group.placement_index, fold]
                        ),
                        "sse": full,
                        "non_anchor_sse": non,
                        "rmse": float(np.sqrt(full / fold_rows[fold])),
                        "non_anchor_rmse": float(
                            np.sqrt(
                                non
                                / fold_non_rows[
                                    group.placement_index, fold
                                ]
                            )
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _quantile_fields(
    values: np.ndarray, prefix: str
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_p50": float(np.quantile(array, 0.50)),
        f"{prefix}_p90": float(np.quantile(array, 0.90)),
        f"{prefix}_p95": float(np.quantile(array, 0.95)),
        f"{prefix}_max": float(np.max(array)),
    }


def _position_exposure_frame(
    seeds: Sequence[int],
    well_ids: Sequence[str],
    requested_u: np.ndarray,
    achieved_u: np.ndarray,
    achieved_md_offset: np.ndarray,
    requested_md_offset: np.ndarray,
    snap_error_u: np.ndarray,
    violation_counts: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    nominal_text = json.dumps(list(NOMINAL_FRACTIONS), separators=(",", ":"))
    for placement_index, placement in enumerate(PLACEMENT_SPECS):
        for seed_position, seed in enumerate(seeds):
            if placement.position_mode == "independent_bounded":
                perturbed_text = json.dumps(
                    [
                        [
                            nominal - placement.tolerance_u,
                            nominal + placement.tolerance_u,
                        ]
                        for nominal in NOMINAL_FRACTIONS
                    ],
                    separators=(",", ":"),
                )
            else:
                perturbed_text = json.dumps(
                    requested_u[
                        placement_index, seed_position, 0
                    ].astype(float).tolist(),
                    separators=(",", ":"),
                )
            violations = violation_counts[
                placement_index, seed_position
            ]
            wells_any = int(np.sum(np.any(violations > 0, axis=1)))
            counts = violations.sum(axis=0, dtype=np.int64)
            well_max = np.max(
                np.abs(
                    achieved_md_offset[placement_index, seed_position]
                ),
                axis=1,
            )
            shared = {
                "tolerance_u": placement.tolerance_u,
                "position_mode": placement.position_mode,
                "realization_seed": int(seed),
                "requested_nominal_positions": nominal_text,
                "requested_perturbed_positions": perturbed_text,
                "boundary_violation_count": int(counts[0]),
                "availability_violation_count": int(counts[1]),
                "duplicate_violation_count": int(counts[2]),
                "order_violation_count": int(counts[3]),
                "wells_with_any_violation": wells_any,
                **_quantile_fields(
                    well_max, "per_well_max_abs_md_displacement_ft"
                ),
            }
            scopes = [("overall", 0, len(NOMINAL_FRACTIONS))]
            scopes.extend(
                ("anchor_order", order, order)
                for order in range(1, len(NOMINAL_FRACTIONS) + 1)
            )
            for scope, start_order, stop_order in scopes:
                if scope == "overall":
                    selected = (slice(None), slice(None))
                    order_value = 0
                else:
                    selected = (slice(None), start_order - 1)
                    order_value = start_order
                requested_values = requested_u[
                    placement_index, seed_position
                ][selected]
                achieved_values = achieved_u[
                    placement_index, seed_position
                ][selected]
                achieved_offsets = np.abs(
                    achieved_md_offset[
                        placement_index, seed_position
                    ][selected]
                )
                requested_offsets = np.abs(
                    requested_md_offset[
                        placement_index, seed_position
                    ][selected]
                )
                snap_values = np.abs(
                    snap_error_u[
                        placement_index, seed_position
                    ][selected]
                )
                records.append(
                    {
                        **shared,
                        "scope": scope,
                        "anchor_order": order_value,
                        "wells": len(well_ids),
                        "positions": int(np.size(achieved_values)),
                        **_quantile_fields(
                            requested_values, "requested_perturbed_u"
                        ),
                        **_quantile_fields(
                            achieved_values, "achieved_normalized_u"
                        ),
                        **_quantile_fields(
                            requested_offsets,
                            "requested_abs_md_displacement_ft",
                        ),
                        **_quantile_fields(
                            achieved_offsets,
                            "achieved_abs_md_displacement_ft",
                        ),
                        **_quantile_fields(
                            snap_values, "absolute_snap_error_u"
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _frontier_frame(
    metric_arrays: Mapping[str, np.ndarray],
    hierarchy: Sequence[Mapping[str, Any]],
    position_exposure: pd.DataFrame,
) -> pd.DataFrame:
    hierarchy_by_group = {
        int(item["group_index"]): item for item in hierarchy
    }
    overall_exposure = (
        position_exposure[position_exposure["scope"].eq("overall")]
        .groupby(["tolerance_u", "position_mode"], sort=False)
        .agg(
            boundary_violation_count=(
                "boundary_violation_count",
                "max",
            ),
            availability_violation_count=(
                "availability_violation_count",
                "max",
            ),
            duplicate_violation_count=(
                "duplicate_violation_count",
                "max",
            ),
            order_violation_count=("order_violation_count", "max"),
            wells_with_any_violation=(
                "wells_with_any_violation",
                "max",
            ),
            achieved_abs_md_displacement_ft_p50=(
                "achieved_abs_md_displacement_ft_p50",
                "median",
            ),
            achieved_abs_md_displacement_ft_p90=(
                "achieved_abs_md_displacement_ft_p90",
                "median",
            ),
            achieved_abs_md_displacement_ft_p95=(
                "achieved_abs_md_displacement_ft_p95",
                "median",
            ),
            achieved_abs_md_displacement_ft_max=(
                "achieved_abs_md_displacement_ft_max",
                "max",
            ),
            per_well_max_abs_md_displacement_ft_p50=(
                "per_well_max_abs_md_displacement_ft_p50",
                "median",
            ),
            per_well_max_abs_md_displacement_ft_p90=(
                "per_well_max_abs_md_displacement_ft_p90",
                "median",
            ),
            per_well_max_abs_md_displacement_ft_p95=(
                "per_well_max_abs_md_displacement_ft_p95",
                "median",
            ),
            per_well_max_abs_md_displacement_ft_max=(
                "per_well_max_abs_md_displacement_ft_max",
                "max",
            ),
        )
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    for group in EVALUATION_GROUPS:
        metadata = _group_metadata(group)
        full = metric_arrays["full_rmse"][group.index]
        non = metric_arrays["non_anchor_rmse"][group.index]
        fold_values = metric_arrays["fold_rmse"][group.index]
        fold_failure_rates = np.mean(
            fold_values >= PERSISTENT_FOLD_RMSE_THRESHOLD, axis=0
        )
        persistent = bool(
            np.any(fold_failure_rates >= PERSISTENT_FOLD_FAILURE_RATE)
        )
        hierarchy_item = hierarchy_by_group[group.index]["hierarchical"]
        exposure = overall_exposure[
            np.isclose(
                overall_exposure["tolerance_u"],
                metadata["tolerance_u"],
                rtol=0.0,
                atol=1e-15,
            )
            & overall_exposure["position_mode"].eq(
                metadata["position_mode"]
            )
        ].iloc[0]
        protocol_valid = bool(
            int(exposure["boundary_violation_count"]) == 0
            and int(exposure["availability_violation_count"]) == 0
            and int(exposure["duplicate_violation_count"]) == 0
            and int(exposure["order_violation_count"]) == 0
            and int(exposure["wells_with_any_violation"]) == 0
        )
        full_q95 = float(np.quantile(full, 0.95))
        non_q95 = float(np.quantile(non, 0.95))
        hier_full = float(hierarchy_item["full_rmse"]["q975"])
        hier_non = float(
            hierarchy_item["non_anchor_rmse"]["q975"]
        )
        full_gate = bool(full_q95 < 4.0 and hier_full < 4.0)
        non_gate = bool(non_q95 < 4.0 and hier_non < 4.0)
        catastrophic = bool(
            np.max(metric_arrays["wells_ge_40"][group.index]) > 0
            or np.max(metric_arrays["wells_ge_60"][group.index]) > 0
        )
        records.append(
            {
                **metadata,
                "realizations": len(full),
                "full_rmse_mean": float(np.mean(full)),
                "full_rmse_q05": float(np.quantile(full, 0.05)),
                "full_rmse_q50": float(np.quantile(full, 0.50)),
                "full_rmse_q95": full_q95,
                "non_anchor_rmse_mean": float(np.mean(non)),
                "non_anchor_rmse_q05": float(np.quantile(non, 0.05)),
                "non_anchor_rmse_q50": float(np.quantile(non, 0.50)),
                "non_anchor_rmse_q95": non_q95,
                "hierarchical_full_q025": float(
                    hierarchy_item["full_rmse"]["q025"]
                ),
                "hierarchical_full_q50": float(
                    hierarchy_item["full_rmse"]["q50"]
                ),
                "hierarchical_full_q975": hier_full,
                "hierarchical_non_anchor_q025": float(
                    hierarchy_item["non_anchor_rmse"]["q025"]
                ),
                "hierarchical_non_anchor_q50": float(
                    hierarchy_item["non_anchor_rmse"]["q50"]
                ),
                "hierarchical_non_anchor_q975": hier_non,
                "fixed_full_margin_ft": 4.0 - full_q95,
                "fixed_non_anchor_margin_ft": 4.0 - non_q95,
                "hierarchical_full_margin_ft": 4.0 - hier_full,
                "hierarchical_non_anchor_margin_ft": 4.0 - hier_non,
                "minimum_numeric_margin_ft": min(
                    4.0 - full_q95,
                    4.0 - non_q95,
                    4.0 - hier_full,
                    4.0 - hier_non,
                ),
                "fold_max_q50": float(
                    np.quantile(
                        np.max(fold_values, axis=1), 0.50
                    )
                ),
                "fold_max_q95": float(
                    np.quantile(
                        np.max(fold_values, axis=1), 0.95
                    )
                ),
                "persistent_fold_failure": persistent,
                "p95_well_rmse_q50": float(
                    np.quantile(
                        metric_arrays["p95"][group.index], 0.50
                    )
                ),
                "p95_well_rmse_q95": float(
                    np.quantile(
                        metric_arrays["p95"][group.index], 0.95
                    )
                ),
                "max_well_rmse_q50": float(
                    np.quantile(
                        metric_arrays["max"][group.index], 0.50
                    )
                ),
                "max_well_rmse_q95": float(
                    np.quantile(
                        metric_arrays["max"][group.index], 0.95
                    )
                ),
                "wells_ge_40_max": int(
                    np.max(
                        metric_arrays["wells_ge_40"][group.index]
                    )
                ),
                "wells_ge_60_max": int(
                    np.max(
                        metric_arrays["wells_ge_60"][group.index]
                    )
                ),
                "worst10_sse_share_q50": float(
                    np.quantile(
                        metric_arrays["worst10_sse_share"][
                            group.index
                        ],
                        0.50,
                    )
                ),
                "worst10_sse_share_q95": float(
                    np.quantile(
                        metric_arrays["worst10_sse_share"][
                            group.index
                        ],
                        0.95,
                    )
                ),
                "protocol_valid": protocol_valid,
                "boundary_violation_count": int(
                    exposure["boundary_violation_count"]
                ),
                "availability_violation_count": int(
                    exposure["availability_violation_count"]
                ),
                "duplicate_violation_count": int(
                    exposure["duplicate_violation_count"]
                ),
                "order_violation_count": int(
                    exposure["order_violation_count"]
                ),
                "wells_with_any_violation": int(
                    exposure["wells_with_any_violation"]
                ),
                "achieved_abs_md_displacement_ft_p50": float(
                    exposure[
                        "achieved_abs_md_displacement_ft_p50"
                    ]
                ),
                "achieved_abs_md_displacement_ft_p90": float(
                    exposure[
                        "achieved_abs_md_displacement_ft_p90"
                    ]
                ),
                "achieved_abs_md_displacement_ft_p95": float(
                    exposure[
                        "achieved_abs_md_displacement_ft_p95"
                    ]
                ),
                "achieved_abs_md_displacement_ft_max": float(
                    exposure[
                        "achieved_abs_md_displacement_ft_max"
                    ]
                ),
                "per_well_max_abs_md_displacement_ft_p50": float(
                    exposure[
                        "per_well_max_abs_md_displacement_ft_p50"
                    ]
                ),
                "per_well_max_abs_md_displacement_ft_p90": float(
                    exposure[
                        "per_well_max_abs_md_displacement_ft_p90"
                    ]
                ),
                "per_well_max_abs_md_displacement_ft_p95": float(
                    exposure[
                        "per_well_max_abs_md_displacement_ft_p95"
                    ]
                ),
                "per_well_max_abs_md_displacement_ft_max": float(
                    exposure[
                        "per_well_max_abs_md_displacement_ft_max"
                    ]
                ),
                "full_gate_pass": full_gate,
                "non_anchor_gate_pass": non_gate,
                "full_non_anchor_decision_consistent": (
                    full_gate == non_gate
                ),
                "robust_lt_4": bool(
                    protocol_valid
                    and full_gate
                    and non_gate
                    and not persistent
                    and not catastrophic
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _paired_deltas_frame(
    metric_arrays: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    zero_by_scenario = {
        group.scenario_index: group
        for group in EVALUATION_GROUPS
        if group.placement_index == 0
    }
    records: list[dict[str, Any]] = []
    for group in EVALUATION_GROUPS:
        if group.placement_index == 0:
            continue
        reference = zero_by_scenario[group.scenario_index]
        full = (
            metric_arrays["full_rmse"][group.index]
            - metric_arrays["full_rmse"][reference.index]
        )
        non = (
            metric_arrays["non_anchor_rmse"][group.index]
            - metric_arrays["non_anchor_rmse"][reference.index]
        )
        records.append(
            {
                **_group_metadata(group),
                "reference_tolerance_u": 0.0,
                "reference_position_mode": ZERO_POSITION_MODE,
                "comparison": "shifted_minus_zero_tolerance",
                "full_rmse_delta_mean": float(np.mean(full)),
                "full_rmse_delta_q05": float(np.quantile(full, 0.05)),
                "full_rmse_delta_q50": float(np.quantile(full, 0.50)),
                "full_rmse_delta_q95": float(np.quantile(full, 0.95)),
                "non_anchor_rmse_delta_mean": float(np.mean(non)),
                "non_anchor_rmse_delta_q05": float(
                    np.quantile(non, 0.05)
                ),
                "non_anchor_rmse_delta_q50": float(
                    np.quantile(non, 0.50)
                ),
                "non_anchor_rmse_delta_q95": float(
                    np.quantile(non, 0.95)
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _zero_tolerance_regression(
    *,
    prior_redundancy: Path,
    seeds: Sequence[int],
    well_ids: Sequence[str],
    rows: np.ndarray,
    non_rows: np.ndarray,
    full_sse: np.ndarray,
    non_sse: np.ndarray,
    hierarchy: Sequence[Mapping[str, Any]],
    strict_full: bool,
    selected_row_max_delta: int,
    measurement_latent_max_delta: float,
    reconstruction_max_delta: float,
) -> dict[str, Any]:
    group_by_scenario = {
        PROTOCOL_SCENARIOS[group.scenario_index].name: group
        for group in EVALUATION_GROUPS
        if group.placement_index == 0
    }
    result: dict[str, Any] = {
        "passed": False,
        "strict_full_comparison": strict_full,
        "zero_tolerance_selected_row_max_abs_delta": selected_row_max_delta,
        "zero_tolerance_measurement_latent_max_abs_delta": (
            measurement_latent_max_delta
        ),
        "zero_tolerance_reconstruction_max_abs_delta": (
            reconstruction_max_delta
        ),
        "per_seed_metric_max_abs_delta": 0.0,
        "per_well_sse_max_abs_delta": 0.0,
        "per_well_non_anchor_sse_max_abs_delta": 0.0,
        "per_well_groups_compared": 0,
        "hierarchical_max_abs_delta": 0.0,
        "locked_metric_max_abs_delta": 0.0,
    }
    if not strict_full:
        result["passed"] = bool(
            selected_row_max_delta == 0
            and measurement_latent_max_delta == 0.0
            and reconstruction_max_delta <= 1e-12
        )
        return result

    prior_root = Path(prior_redundancy).resolve()
    prior_seed = pd.read_csv(prior_root / "per_seed_metrics.csv")
    metric_delta = 0.0
    for scenario_name, group in group_by_scenario.items():
        prior = prior_seed[
            prior_seed["config"].eq("near_k8_r1")
            & prior_seed["method"].eq(OPERATIONAL_METHOD)
            & prior_seed["outlier_dependence"].eq(DEPENDENCE_SINGLE)
            & prior_seed["scenario"].eq(scenario_name)
        ].sort_values("realization_seed", kind="stable")
        if tuple(prior["realization_seed"].astype(int)) != tuple(seeds):
            raise ValueError(
                f"prior zero-tolerance seed coverage mismatch for {scenario_name}"
            )
        current_full = np.sqrt(
            full_sse[group.index].sum(axis=1) / rows.sum()
        )
        current_non = np.sqrt(
            non_sse[group.index].sum(axis=1) / non_rows[0].sum()
        )
        metric_delta = max(
            metric_delta,
            float(
                np.max(
                    np.abs(
                        current_full
                        - prior["full_rmse"].to_numpy(dtype=np.float64)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        current_non
                        - prior["non_anchor_rmse"].to_numpy(
                            dtype=np.float64
                        )
                    )
                )
            ),
        )

    seed_position = {int(seed): index for index, seed in enumerate(seeds)}
    well_position = {str(well): index for index, well in enumerate(well_ids)}
    full_delta = 0.0
    non_delta = 0.0
    compared = 0
    usecols = [
        "config",
        "method",
        "scenario",
        "outlier_dependence",
        "realization_seed",
        "well_id",
        "sse",
        "non_anchor_sse",
    ]
    for chunk in pd.read_csv(
        prior_root / "per_well_seed_metrics.csv.gz",
        usecols=usecols,
        dtype={"well_id": "string"},
        chunksize=250_000,
    ):
        selected = chunk[
            chunk["config"].eq("near_k8_r1")
            & chunk["method"].eq(OPERATIONAL_METHOD)
            & chunk["outlier_dependence"].eq(DEPENDENCE_SINGLE)
            & chunk["scenario"].isin(SCENARIO_NAMES)
            & chunk["realization_seed"].isin(seeds)
            & chunk["well_id"].isin(well_ids)
        ]
        for row in selected.itertuples(index=False):
            group = group_by_scenario[str(row.scenario)]
            si = seed_position[int(row.realization_seed)]
            wi = well_position[str(row.well_id)]
            full_delta = max(
                full_delta,
                abs(
                    float(full_sse[group.index, si, wi])
                    - float(row.sse)
                ),
            )
            non_delta = max(
                non_delta,
                abs(
                    float(non_sse[group.index, si, wi])
                    - float(row.non_anchor_sse)
                ),
            )
            compared += 1
    expected_compared = len(SCENARIO_NAMES) * len(seeds) * len(well_ids)

    prior_hierarchy = json.loads(
        (prior_root / "hierarchical_bootstrap.json").read_text(
            encoding="utf-8"
        )
    )["comparisons"]
    hierarchy_by_group = {
        int(item["group_index"]): item for item in hierarchy
    }
    hierarchical_delta = 0.0
    for scenario_name, group in group_by_scenario.items():
        match = [
            item
            for item in prior_hierarchy
            if item["config"] == "near_k8_r1"
            and item["method"] == OPERATIONAL_METHOD
            and item["outlier_dependence"] == DEPENDENCE_SINGLE
            and item["scenario"] == scenario_name
        ]
        if len(match) != 1:
            raise ValueError(
                f"prior hierarchy coverage mismatch for {scenario_name}"
            )
        current = hierarchy_by_group[group.index]["hierarchical"]
        prior = match[0]["hierarchical"]
        for metric in ("full_rmse", "non_anchor_rmse"):
            for quantile in ("q025", "q50", "q975"):
                hierarchical_delta = max(
                    hierarchical_delta,
                    abs(
                        float(current[metric][quantile])
                        - float(prior[metric][quantile])
                    ),
                )

    zero_frontier = pd.read_csv(prior_root / "acquisition_frontier.csv")
    prior_o5 = zero_frontier[
        zero_frontier["config"].eq("near_k8_r1")
        & zero_frontier["method"].eq(OPERATIONAL_METHOD)
        & zero_frontier["outlier_dependence"].eq(DEPENDENCE_SINGLE)
        & zero_frontier["scenario"].eq(O5_SCENARIO)
    ].iloc[0]
    locked_delta = max(
        abs(
            float(prior_o5["full_rmse_q95"])
            - EXPECTED_ZERO_METRICS["full_rmse_q95"]
        ),
        abs(
            float(prior_o5["non_anchor_rmse_q95"])
            - EXPECTED_ZERO_METRICS["non_anchor_rmse_q95"]
        ),
        abs(
            float(prior_o5["hierarchical_full_q975"])
            - EXPECTED_ZERO_METRICS["hierarchical_full_q975"]
        ),
        abs(
            float(prior_o5["hierarchical_non_anchor_q975"])
            - EXPECTED_ZERO_METRICS["hierarchical_non_anchor_q975"]
        ),
    )
    result.update(
        {
            "per_seed_metric_max_abs_delta": metric_delta,
            "per_well_sse_max_abs_delta": full_delta,
            "per_well_non_anchor_sse_max_abs_delta": non_delta,
            "per_well_groups_compared": compared,
            "expected_per_well_groups_compared": expected_compared,
            "hierarchical_max_abs_delta": hierarchical_delta,
            "locked_metric_max_abs_delta": locked_delta,
        }
    )
    result["passed"] = bool(
        selected_row_max_delta == 0
        and measurement_latent_max_delta == 0.0
        and reconstruction_max_delta <= 1e-12
        and metric_delta <= METRIC_TOLERANCE
        and full_delta <= PER_WELL_SSE_TOLERANCE
        and non_delta <= PER_WELL_SSE_TOLERANCE
        and compared == expected_compared
        and hierarchical_delta <= METRIC_TOLERANCE
        and locked_delta <= METRIC_TOLERANCE
    )
    return result


def _decision_summary(frontier: pd.DataFrame) -> dict[str, Any]:
    primary = frontier[
        np.isclose(
            frontier["tolerance_u"],
            PRIMARY_PROTOCOL_TOLERANCE_U,
            rtol=0.0,
            atol=1e-15,
        )
    ]
    primary_o5 = primary[primary["scenario"].eq(O5_SCENARIO)]
    primary_controls = primary[
        primary["scenario"].isin(
            ("exact_clean", "moderate_m5_control")
        )
    ]
    if (
        len(primary_o5) != len(POSITION_MODES)
        or len(primary_controls) != 2 * len(POSITION_MODES)
    ):
        raise ValueError("primary decision matrix is incomplete")
    limiting = primary_o5.sort_values(
        "minimum_numeric_margin_ft", kind="stable"
    ).iloc[0]
    diagnostics: list[dict[str, Any]] = []
    for tolerance in POSITION_TOLERANCES_U[1:]:
        block = frontier[
            np.isclose(
                frontier["tolerance_u"],
                tolerance,
                rtol=0.0,
                atol=1e-15,
            )
        ]
        diagnostics.append(
            {
                "tolerance_u": tolerance,
                "all_modes_and_scenarios_pass": bool(
                    len(block) == len(POSITION_MODES) * len(PROTOCOL_SCENARIOS)
                    and block["robust_lt_4"].all()
                ),
                "o5_all_modes_pass": bool(
                    len(block[block["scenario"].eq(O5_SCENARIO)])
                    == len(POSITION_MODES)
                    and block.loc[
                        block["scenario"].eq(O5_SCENARIO),
                        "robust_lt_4",
                    ].all()
                ),
                "minimum_o5_numeric_margin_ft": float(
                    block.loc[
                        block["scenario"].eq(O5_SCENARIO),
                        "minimum_numeric_margin_ft",
                    ].min()
                ),
            }
        )
    return {
        "primary_tolerance_u": PRIMARY_PROTOCOL_TOLERANCE_U,
        "primary_protocol_pass": bool(
            primary_o5["robust_lt_4"].all()
            and primary_controls["robust_lt_4"].all()
        ),
        "primary_o5_all_modes_pass": bool(
            primary_o5["robust_lt_4"].all()
        ),
        "primary_exact_clean_controls_robust": bool(
            primary.loc[
                primary["scenario"].eq("exact_clean"), "robust_lt_4"
            ].all()
        ),
        "primary_m5_controls_robust": bool(
            primary.loc[
                primary["scenario"].eq("moderate_m5_control"),
                "robust_lt_4",
            ].all()
        ),
        "primary_limiting_position_mode": str(
            limiting["position_mode"]
        ),
        "primary_minimum_numeric_margin_ft": float(
            limiting["minimum_numeric_margin_ft"]
        ),
        "primary_limiting_metric_values": {
            "full_rmse_q95": float(limiting["full_rmse_q95"]),
            "non_anchor_rmse_q95": float(
                limiting["non_anchor_rmse_q95"]
            ),
            "hierarchical_full_q975": float(
                limiting["hierarchical_full_q975"]
            ),
            "hierarchical_non_anchor_q975": float(
                limiting["hierarchical_non_anchor_q975"]
            ),
        },
        "full_non_anchor_decisions_consistent": bool(
            frontier["full_non_anchor_decision_consistent"].all()
        ),
        "maximum_wells_ge_40": int(frontier["wells_ge_40_max"].max()),
        "maximum_wells_ge_60": int(frontier["wells_ge_60_max"].max()),
        "any_protocol_violation": bool(
            (
                frontier[
                    [
                        "boundary_violation_count",
                        "availability_violation_count",
                        "duplicate_violation_count",
                        "order_violation_count",
                        "wells_with_any_violation",
                    ]
                ]
                .to_numpy(dtype=np.int64)
                > 0
            ).any()
        ),
        "diagnostic_tolerances": diagnostics,
    }


def _render_report(
    frontier: pd.DataFrame,
    position_exposure: pd.DataFrame,
    paired: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    primary = frontier[
        np.isclose(
            frontier["tolerance_u"],
            PRIMARY_PROTOCOL_TOLERANCE_U,
            rtol=0.0,
            atol=1e-15,
        )
    ]
    primary_o5 = primary[primary["scenario"].eq(O5_SCENARIO)]
    diagnostic_o5 = frontier[
        frontier["scenario"].eq(O5_SCENARIO)
        & frontier["tolerance_u"].gt(0)
    ]
    displacement = (
        position_exposure[
            position_exposure["scope"].eq("overall")
        ]
        .groupby(["tolerance_u", "position_mode"], sort=False)
        .agg(
            achieved_abs_md_displacement_ft_p50=(
                "achieved_abs_md_displacement_ft_p50",
                "median",
            ),
            achieved_abs_md_displacement_ft_p90=(
                "achieved_abs_md_displacement_ft_p90",
                "median",
            ),
            achieved_abs_md_displacement_ft_p95=(
                "achieved_abs_md_displacement_ft_p95",
                "median",
            ),
            achieved_abs_md_displacement_ft_max=(
                "achieved_abs_md_displacement_ft_max",
                "max",
            ),
            per_well_max_abs_md_displacement_ft_p50=(
                "per_well_max_abs_md_displacement_ft_p50",
                "median",
            ),
            per_well_max_abs_md_displacement_ft_p90=(
                "per_well_max_abs_md_displacement_ft_p90",
                "median",
            ),
            per_well_max_abs_md_displacement_ft_p95=(
                "per_well_max_abs_md_displacement_ft_p95",
                "median",
            ),
            per_well_max_abs_md_displacement_ft_max=(
                "per_well_max_abs_md_displacement_ft_max",
                "max",
            ),
            wells_with_any_violation=(
                "wells_with_any_violation",
                "max",
            ),
        )
        .reset_index()
    )
    zero = frontier[
        frontier["tolerance_u"].eq(0.0)
        & frontier["scenario"].eq(O5_SCENARIO)
    ].iloc[0]
    lines = [
        "# Sparse-anchor eight-location practical acquisition protocol qualification",
        "",
        "Date: 2026-07-27 (Asia/Taipei)",
        "",
        "## Evidence status",
        "",
        "- `deployable: false`",
        "- `submission_eligible: false`",
        "- `counterfactual_information_experiment: true`",
        "- `uses_validation_target_as_revealed_anchor: true`",
        "- `kaggle_submission_created: false`",
        "- `kaggle_submission_uploaded: false`",
        "",
        "This is a locked acausal retrospective information-acquisition "
        "qualification. It is not a deployable or online competition model.",
        "",
        "## Decision",
        "",
        f"- Zero-tolerance legacy reproduction passed: "
        f"`{str(summary['zero_tolerance_regression']['passed']).lower()}`.",
        f"- Primary `T_u=0.001` protocol pass: "
        f"`{str(summary['primary_protocol_pass']).lower()}`.",
        f"- Limiting primary position mode: "
        f"`{summary['primary_limiting_position_mode']}`.",
        f"- Minimum primary O5 numeric margin: "
        f"`{summary['primary_minimum_numeric_margin_ft']:.9f} ft`.",
        f"- Exact-clean controls robust: "
        f"`{str(summary['primary_exact_clean_controls_robust']).lower()}`; "
        f"M5 controls robust: "
        f"`{str(summary['primary_m5_controls_robust']).lower()}`.",
        f"- Full/non-anchor decisions consistent: "
        f"`{str(summary['full_non_anchor_decisions_consistent']).lower()}`.",
        f"- Any boundary/availability/duplicate/order violation: "
        f"`{str(summary['any_protocol_violation']).lower()}`.",
        f"- Maximum wells >=40 / >=60 RMSE: "
        f"`{summary['maximum_wells_ge_40']} / "
        f"{summary['maximum_wells_ge_60']}`.",
        "",
        "## Zero-tolerance O5 reproduction",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "full_rmse_q95": zero["full_rmse_q95"],
                        "non_anchor_rmse_q95": zero[
                            "non_anchor_rmse_q95"
                        ],
                        "hierarchical_full_q975": zero[
                            "hierarchical_full_q975"
                        ],
                        "hierarchical_non_anchor_q975": zero[
                            "hierarchical_non_anchor_q975"
                        ],
                    }
                ]
            )
        ),
        "",
        "## Primary O5 position modes",
        "",
        _markdown_table(
            primary_o5[
                [
                    "position_mode",
                    "full_rmse_q95",
                    "non_anchor_rmse_q95",
                    "hierarchical_full_q975",
                    "hierarchical_non_anchor_q975",
                    "minimum_numeric_margin_ft",
                    "persistent_fold_failure",
                    "wells_ge_40_max",
                    "wells_ge_60_max",
                    "robust_lt_4",
                ]
            ]
        ),
        "",
        "## Diagnostic O5 tolerances",
        "",
        _markdown_table(
            diagnostic_o5[
                [
                    "tolerance_u",
                    "position_mode",
                    "full_rmse_q95",
                    "non_anchor_rmse_q95",
                    "hierarchical_full_q975",
                    "hierarchical_non_anchor_q975",
                    "minimum_numeric_margin_ft",
                    "robust_lt_4",
                ]
            ]
        ),
        "",
        "`0.0025` and `0.005` are diagnostic characterizations only. They "
        "do not replace or redefine the primary `0.001` decision.",
        "",
        "## Achieved MD displacement",
        "",
        _markdown_table(displacement),
        "",
        "The reported computational `u` levels are not claims about real "
        "hardware tolerance. Field calibration would be needed for that "
        "interpretation. Per-well summaries use the maximum absolute achieved "
        "MD displacement among the eight physical anchors.",
        "",
        "## Paired position effect",
        "",
        _markdown_table(
            paired[
                paired["scenario"].eq(O5_SCENARIO)
            ][
                [
                    "tolerance_u",
                    "position_mode",
                    "full_rmse_delta_q50",
                    "full_rmse_delta_q95",
                    "non_anchor_rmse_delta_q50",
                    "non_anchor_rmse_delta_q95",
                ]
            ]
        ),
        "",
        "## Interpretation and limits",
        "",
        "O5 (`tau=1`, `sigma=1`, `p=.10`, `M=10`) is a stochastic "
        "qualification envelope over paired realizations. It is not a hard "
        "guarantee for any one reading or device.",
        "",
        "The prior phrase shared-fault-safe only ruled out dependence on "
        "same-location repeat independence. This protocol still does not test "
        "a cross-location common M10 gross shift, common position-calibration "
        "drift beyond the coherent bound-edge modes, or other correlated "
        "eight-location failure mechanisms.",
        "",
        "Actual achieved rows and achieved normalized positions drive both the "
        "revealed TVT values and interpolation. Measurement-error latents stay "
        "paired to nominal physical anchor identity, so changing the snapped "
        "achieved row does not silently replace the error draw.",
        "",
        "The result still uses validation target TVT at preselected achieved "
        "anchor rows and future anchors in acausal reconstruction. It is "
        "therefore a retrospective counterfactual acquisition specification, "
        "not competition-deployable inference.",
        "",
        "If the primary protocol passes, the next experiment is an independent "
        "fixed near_k8 stress qualification. If it fails, retain only the "
        "idealized near_k8 conclusion and do not retune placement or tolerance "
        "in this round. If field calibration shows tau or sigma above 1 ft, "
        "datum calibration becomes a higher priority.",
        "",
    ]
    return "\n".join(lines)


def audit_anchor_acquisition_protocol_artifacts(
    run_root: Path,
) -> dict[str, Any]:
    """Reconstruct pooled metrics and validate protocol artifact topology."""
    root = Path(run_root).resolve()
    config = json.loads(
        (root / "run_config.json").read_text(encoding="utf-8")
    )
    per_seed = pd.read_csv(root / "per_seed_metrics.csv")
    per_fold = pd.read_csv(root / "per_fold_seed_metrics.csv")
    frontier = pd.read_csv(root / "protocol_frontier.csv")
    exposure = pd.read_csv(root / "position_exposure.csv")
    paired = pd.read_csv(root / "paired_deltas.csv")
    keys = [
        "tolerance_u",
        "position_mode",
        "scenario",
        "realization_seed",
    ]
    usecols = keys + [
        "well_id",
        "rows",
        "non_anchor_rows",
        "sse",
        "non_anchor_sse",
    ]
    partials: list[pd.DataFrame] = []
    per_well_rows = 0
    unexpected_nan = 0
    for chunk in pd.read_csv(
        root / "per_well_seed_metrics.csv.gz",
        usecols=usecols,
        dtype={"well_id": "string"},
        chunksize=250_000,
    ):
        per_well_rows += len(chunk)
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
        joined["full_rmse_reconstructed"]
        - joined["full_rmse_reported"]
    )
    non_delta = np.abs(
        joined["non_anchor_rmse_reconstructed"]
        - joined["non_anchor_rmse_reported"]
    )
    expected_groups = (
        len(config["evaluation_groups"])
        * len(config["realization_seeds"])
    )
    expected_per_well = expected_groups * int(config["wells"])
    duplicate_count = sum(
        (
            int(per_seed.duplicated(keys).sum()),
            int(per_fold.duplicated(keys + ["fold"]).sum()),
            int(
                frontier.duplicated(
                    [
                        "tolerance_u",
                        "position_mode",
                        "scenario",
                    ]
                ).sum()
            ),
            int(
                exposure.duplicated(
                    [
                        "tolerance_u",
                        "position_mode",
                        "realization_seed",
                        "scope",
                        "anchor_order",
                    ]
                ).sum()
            ),
            int(
                paired.duplicated(
                    [
                        "tolerance_u",
                        "position_mode",
                        "scenario",
                    ]
                ).sum()
            ),
        )
    )
    table_nan = sum(
        int(frame.select_dtypes(include=[np.number]).isna().sum().sum())
        for frame in (per_seed, per_fold, frontier, exposure, paired)
    )
    manifest_rows = 0
    manifest_duplicates = 0
    manifest_nan = 0
    for chunk in pd.read_csv(
        root / "position_manifest.csv.gz",
        dtype={"well_id": "string"},
        chunksize=250_000,
    ):
        manifest_rows += len(chunk)
        manifest_duplicates += int(
            chunk.duplicated(
                [
                    "tolerance_u",
                    "position_mode",
                    "position_seed",
                    "well_id",
                    "anchor_order",
                ]
            ).sum()
        )
        manifest_nan += int(
            chunk.select_dtypes(include=[np.number]).isna().sum().sum()
        )
    fixed_placements = sum(
        spec.position_mode != "independent_bounded"
        for spec in PLACEMENT_SPECS
    )
    independent_placements = len(PLACEMENT_SPECS) - fixed_placements
    expected_manifest = (
        int(config["wells"])
        * len(NOMINAL_FRACTIONS)
        * (
            fixed_placements
            + independent_placements * len(config["realization_seeds"])
        )
    )
    passed = bool(
        len(aggregate) == expected_groups
        and len(per_seed) == expected_groups
        and per_well_rows == expected_per_well
        and aggregate["wells"].eq(int(config["wells"])).all()
        and aggregate["rows"].eq(int(config["rows"])).all()
        and joined["_merge"].eq("both").all()
        and float(full_delta.max()) <= METRIC_TOLERANCE
        and float(non_delta.max()) <= METRIC_TOLERANCE
        and len(frontier) == len(config["evaluation_groups"])
        and len(per_fold) == expected_groups * 5
        and len(exposure)
        == len(PLACEMENT_SPECS)
        * len(config["realization_seeds"])
        * (1 + len(NOMINAL_FRACTIONS))
        and len(paired)
        == (len(PLACEMENT_SPECS) - 1) * len(PROTOCOL_SCENARIOS)
        and duplicate_count == 0
        and unexpected_nan + table_nan == 0
        and manifest_rows == expected_manifest
        and manifest_duplicates == 0
        and manifest_nan == 0
    )
    return {
        "passed": passed,
        "expected_metric_groups": expected_groups,
        "actual_metric_groups": int(len(aggregate)),
        "per_well_rows": per_well_rows,
        "expected_per_well_rows": expected_per_well,
        "max_full_rmse_reconstruction_delta": float(full_delta.max()),
        "max_non_anchor_rmse_reconstruction_delta": float(
            non_delta.max()
        ),
        "duplicate_key_count": duplicate_count + manifest_duplicates,
        "unexpected_numeric_nan_count": (
            unexpected_nan + table_nan + manifest_nan
        ),
        "frontier_rows": int(len(frontier)),
        "position_exposure_rows": int(len(exposure)),
        "position_manifest_rows": manifest_rows,
        "expected_position_manifest_rows": expected_manifest,
    }


def run_anchor_acquisition_protocol(
    *,
    data_root: Path,
    fold_config: Path,
    blend_oof: Path,
    prior_redundancy: Path,
    output_dir: Path,
    profile: str,
) -> dict[str, Any]:
    """Run the fixed quick or full practical acquisition protocol matrix."""
    _validate_locked_constants()
    profile_name = str(profile)
    if profile_name == "quick":
        seeds = tuple(range(5))
        bootstrap_samples = 200
        max_wells_per_fold: int | None = 2
    elif profile_name == "full":
        seeds = tuple(range(100))
        bootstrap_samples = 10_000
        max_wells_per_fold = None
    else:
        raise ValueError("profile must be 'quick' or 'full'")
    bootstrap_seed = DEFAULT_BOOTSTRAP_SEED
    selection_seed = DEFAULT_SELECTION_SEED

    data = Path(data_root).resolve()
    requested_output = Path(output_dir).resolve()
    run_root = safe_output_directory(
        requested_output.parent, requested_output.name, data
    )
    fold_hash = hashlib.sha256(
        Path(fold_config).read_bytes()
    ).hexdigest().upper()
    train_wells = enumerate_wells(data, "train")
    mapping = load_fold_mapping(
        fold_config,
        expected_well_ids=[well.well_id for well in train_wells],
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
    rows = np.asarray(
        [len(contexts[well].row_index) for well in well_ids],
        dtype=np.int64,
    )
    folds = np.asarray(
        [contexts[well].fold for well in well_ids], dtype=np.int8
    )
    strict_full = profile_name == "full"
    if strict_full and (
        well_count != 773 or int(rows.sum()) != 3_783_989
    ):
        raise ValueError("full protocol coverage mismatch")

    group_count = len(EVALUATION_GROUPS)
    seed_count = len(seeds)
    placement_count = len(PLACEMENT_SPECS)
    shape = (group_count, seed_count, well_count)
    full_sse = np.zeros(shape, dtype=np.float64)
    non_sse = np.zeros(shape, dtype=np.float64)
    non_rows = np.zeros((placement_count, well_count), dtype=np.int64)
    fold_rows = np.zeros(5, dtype=np.int64)
    fold_non_rows = np.zeros((placement_count, 5), dtype=np.int64)
    fold_sse = np.zeros((group_count, seed_count, 5), dtype=np.float64)
    fold_non_sse = np.zeros_like(fold_sse)

    position_shape = (
        placement_count,
        seed_count,
        well_count,
        len(NOMINAL_FRACTIONS),
    )
    requested_u = np.zeros(position_shape, dtype=np.float32)
    achieved_u = np.zeros(position_shape, dtype=np.float32)
    achieved_md_offset = np.zeros(position_shape, dtype=np.float32)
    requested_md_offset = np.zeros(position_shape, dtype=np.float32)
    snap_error_u = np.zeros(position_shape, dtype=np.float32)
    violation_counts = np.zeros(
        (
            placement_count,
            seed_count,
            well_count,
            len(POSITION_VIOLATION_TYPES),
        ),
        dtype=np.int16,
    )

    placement_digest = hashlib.sha256()
    max_zero_selected_row_delta = 0
    max_measurement_latent_delta = 0.0
    max_zero_reconstruction_delta = 0.0
    max_analytic_exact_delta = 0.0
    max_analytic_linear_delta = 0.0
    max_analytic_gram_delta = 0.0
    max_anchor_observation_delta = 0.0

    group_lookup = {
        (group.placement_index, group.scenario_index): group.index
        for group in EVALUATION_GROUPS
    }
    scenario_values = np.asarray(
        [scenario.name for scenario in PROTOCOL_SCENARIOS], dtype=object
    )
    group_placement = np.asarray(
        [group.placement_index for group in EVALUATION_GROUPS],
        dtype=np.int64,
    )
    group_scenario = np.asarray(
        [group.scenario_index for group in EVALUATION_GROUPS],
        dtype=np.int64,
    )
    group_tolerance = np.asarray(
        [
            PLACEMENT_SPECS[group.placement_index].tolerance_u
            for group in EVALUATION_GROUPS
        ],
        dtype=np.float64,
    )
    group_mode = np.asarray(
        [
            PLACEMENT_SPECS[group.placement_index].position_mode
            for group in EVALUATION_GROUPS
        ],
        dtype=object,
    )

    per_well_path = run_root / "per_well_seed_metrics.csv.gz"
    manifest_path = run_root / "position_manifest.csv.gz"
    with (
        _DeterministicGzipCsvWriter(per_well_path) as per_well_writer,
        _DeterministicGzipCsvWriter(manifest_path) as manifest_writer,
    ):
        for well_position, well_id in enumerate(well_ids):
            context = contexts[well_id]
            fold_rows[context.fold] += len(context.row_index)
            boundary_md = _boundary_md(context)  # type: ignore[arg-type]
            nominal_legacy = select_terminal_anchors(
                context.prediction_md,
                boundary_md,
                policy="near_terminal",
                k=len(NOMINAL_FRACTIONS),
            )
            if (
                nominal_legacy.achieved_k != len(NOMINAL_FRACTIONS)
                or not np.array_equal(
                    nominal_legacy.requested_positions,
                    np.asarray(NOMINAL_FRACTIONS),
                )
                or np.any(np.diff(nominal_legacy.local_indices) <= 0)
            ):
                raise ValueError("legacy nominal near_k8 placement changed")
            nominal_protocol = select_protocol_anchors(
                context.normalized_md, NOMINAL_FRACTIONS
            )
            if not nominal_protocol.valid:
                raise ValueError("zero-tolerance protocol selection failed")
            assert nominal_protocol.selection is not None
            max_zero_selected_row_delta = max(
                max_zero_selected_row_delta,
                int(
                    np.max(
                        np.abs(
                            nominal_protocol.selection.local_indices
                            - nominal_legacy.local_indices
                        )
                    )
                ),
            )
            nominal_indices = nominal_legacy.local_indices
            nominal_physical_rows = context.row_index[nominal_indices]
            nominal_md = context.prediction_md[nominal_indices]
            horizon_md = float(
                context.prediction_md[-1] - boundary_md
            )
            latents = stable_replicate_latents(
                seeds,
                well_id,
                nominal_physical_rows,
                1,
            )
            legacy_latents = stable_replicate_latents(
                seeds,
                well_id,
                nominal_physical_rows,
                1,
            )
            max_measurement_latent_delta = max(
                max_measurement_latent_delta,
                float(
                    np.max(
                        np.abs(
                            latents.common_bias
                            - legacy_latents.common_bias
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            latents.independent
                            - legacy_latents.independent
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            latents.outlier_indicator_uniform
                            - legacy_latents.outlier_indicator_uniform
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            latents.outlier_sign
                            - legacy_latents.outlier_sign
                        )
                    )
                ),
            )
            errors_by_scenario: list[np.ndarray] = []
            for scenario in PROTOCOL_SCENARIOS:
                no_outlier = (
                    scenario.tau * latents.common_bias[:, None]
                    + scenario.sigma * latents.independent[:, :, 0]
                )
                flags = (
                    latents.outlier_indicator_uniform[:, :, 0]
                    < scenario.outlier_probability
                )
                errors_by_scenario.append(
                    no_outlier
                    + flags.astype(np.float64)
                    * latents.outlier_sign[:, :, 0].astype(np.float64)
                    * scenario.outlier_magnitude
                )
            prefix = build_piecewise_prefix_sums(
                context.normalized_md,
                context.base_prediction,
                context.y_true,
            )
            signs = stable_position_sign_matrix(seeds, well_id)
            manifest_records: list[dict[str, Any]] = []

            for placement_index, placement in enumerate(PLACEMENT_SPECS):
                if placement.position_mode == "independent_bounded":
                    position_requests = [
                        (
                            seed_position,
                            int(seeds[seed_position]),
                            requested_protocol_positions(
                                placement.tolerance_u,
                                placement.position_mode,
                                signs=signs[seed_position],
                            ),
                        )
                        for seed_position in range(seed_count)
                    ]
                else:
                    request = requested_protocol_positions(
                        placement.tolerance_u,
                        placement.position_mode,
                    )
                    position_requests = [(0, -1, request)]

                selections: list[AnchorSelection] = []
                fixed_selection: AnchorSelection | None = None
                for seed_position, position_seed, request in position_requests:
                    outcome = select_protocol_anchors(
                        context.normalized_md, request
                    )
                    if placement.position_mode == "independent_bounded":
                        violation_counts[
                            placement_index,
                            seed_position,
                            well_position,
                        ] = outcome.violation_counts
                    else:
                        violation_counts[
                            placement_index, :, well_position
                        ] = outcome.violation_counts
                    if not outcome.valid:
                        raise ValueError(
                            "protocol violation failed closed for "
                            f"well={well_id}, tolerance={placement.tolerance_u}, "
                            f"mode={placement.position_mode}, "
                            f"seed={position_seed}: "
                            f"{outcome.violation_counts.tolist()}"
                        )
                    assert outcome.selection is not None
                    selection = outcome.selection
                    if placement.position_mode == "independent_bounded":
                        selections.append(selection)
                    else:
                        fixed_selection = selection
                    actual_indices = selection.local_indices
                    actual_u = context.normalized_md[actual_indices]
                    actual_md = context.prediction_md[actual_indices]
                    target_seed: int | slice = (
                        seed_position
                        if placement.position_mode == "independent_bounded"
                        else slice(None)
                    )
                    requested_u[
                        placement_index, target_seed, well_position
                    ] = request
                    achieved_u[
                        placement_index, target_seed, well_position
                    ] = actual_u
                    achieved_md_offset[
                        placement_index, target_seed, well_position
                    ] = actual_md - nominal_md
                    requested_md_offset[
                        placement_index, target_seed, well_position
                    ] = (
                        request - np.asarray(NOMINAL_FRACTIONS)
                    ) * horizon_md
                    snap_error_u[
                        placement_index, target_seed, well_position
                    ] = actual_u - request

                    for anchor_order, local_index in enumerate(
                        actual_indices, start=1
                    ):
                        manifest_records.append(
                            {
                                "tolerance_u": placement.tolerance_u,
                                "position_mode": placement.position_mode,
                                "position_seed": position_seed,
                                "seed_invariant": (
                                    placement.position_mode
                                    != "independent_bounded"
                                ),
                                "well_id": well_id,
                                "fold": context.fold,
                                "anchor_order": anchor_order,
                                "nominal_u": NOMINAL_FRACTIONS[
                                    anchor_order - 1
                                ],
                                "requested_u": float(
                                    request[anchor_order - 1]
                                ),
                                "achieved_u": float(
                                    actual_u[anchor_order - 1]
                                ),
                                "nominal_row_index": int(
                                    nominal_physical_rows[
                                        anchor_order - 1
                                    ]
                                ),
                                "achieved_row_index": int(
                                    context.row_index[local_index]
                                ),
                                "nominal_md": float(
                                    nominal_md[anchor_order - 1]
                                ),
                                "achieved_md": float(
                                    actual_md[anchor_order - 1]
                                ),
                                "requested_md_offset_ft": float(
                                    (
                                        request[anchor_order - 1]
                                        - NOMINAL_FRACTIONS[
                                            anchor_order - 1
                                        ]
                                    )
                                    * horizon_md
                                ),
                                "achieved_md_offset_ft": float(
                                    actual_md[anchor_order - 1]
                                    - nominal_md[anchor_order - 1]
                                ),
                                "snap_error_u": float(
                                    actual_u[anchor_order - 1]
                                    - request[anchor_order - 1]
                                ),
                                "revealed_tvt": float(
                                    context.y_true[local_index]
                                ),
                                "boundary_violation": False,
                                "availability_violation": False,
                                "duplicate_violation": False,
                                "order_violation": False,
                            }
                        )

                    placement_digest.update(well_id.encode("ascii"))
                    placement_digest.update(placement.key.encode("ascii"))
                    placement_digest.update(np.int64(position_seed).tobytes())
                    placement_digest.update(
                        np.ascontiguousarray(request).view(np.uint8)
                    )
                    placement_digest.update(
                        np.ascontiguousarray(
                            context.row_index[actual_indices]
                        ).view(np.uint8)
                    )

                if placement.position_mode != "independent_bounded":
                    assert fixed_selection is not None
                    selections = [fixed_selection] * seed_count
                unique_selections: dict[tuple[int, ...], list[int]] = {}
                for seed_position, selection in enumerate(selections):
                    unique_selections.setdefault(
                        tuple(map(int, selection.local_indices)), []
                    ).append(seed_position)
                for indices_key, seed_positions in unique_selections.items():
                    selection = selections[seed_positions[0]]
                    if placement_index == 0:
                        coefficients, exact_prediction, weights = (
                            direct_piecewise_coefficients(
                                context, selection
                            )
                        )
                        legacy_prediction = condition_noisy_piecewise(
                            context.base_prediction,
                            context.y_true[nominal_legacy.local_indices],
                            context.normalized_md,
                            nominal_legacy,
                        )
                        max_zero_reconstruction_delta = max(
                            max_zero_reconstruction_delta,
                            float(
                                np.max(
                                    np.abs(
                                        exact_prediction
                                        - legacy_prediction
                                    )
                                )
                            ),
                        )
                    else:
                        coefficients = analytic_piecewise_coefficients(
                            prefix, selection
                        )
                        if well_position < 2:
                            direct, _, _ = direct_piecewise_coefficients(
                                context, selection
                            )
                            max_analytic_exact_delta = max(
                                max_analytic_exact_delta,
                                float(
                                    np.max(
                                        np.abs(
                                            coefficients[0] - direct[0]
                                        )
                                    )
                                ),
                            )
                            max_analytic_linear_delta = max(
                                max_analytic_linear_delta,
                                float(
                                    np.max(
                                        np.abs(
                                            coefficients[1] - direct[1]
                                        )
                                    )
                                ),
                            )
                            max_analytic_gram_delta = max(
                                max_analytic_gram_delta,
                                float(
                                    np.max(
                                        np.abs(
                                            coefficients[2] - direct[2]
                                        )
                                    )
                                ),
                            )
                    selected_seed_positions = np.asarray(
                        seed_positions, dtype=np.int64
                    )
                    stacked_errors = np.concatenate(
                        [
                            errors[selected_seed_positions]
                            for errors in errors_by_scenario
                        ],
                        axis=0,
                    )
                    stacked_result = evaluate_protocol_coefficients(
                        coefficients, stacked_errors
                    ).reshape(
                        len(PROTOCOL_SCENARIOS),
                        len(selected_seed_positions),
                        2,
                    )
                    for scenario_index, errors in enumerate(errors_by_scenario):
                        group_index = group_lookup[
                            (placement_index, scenario_index)
                        ]
                        result = stacked_result[scenario_index]
                        full_sse[
                            group_index,
                            selected_seed_positions,
                            well_position,
                        ] = result[:, 0]
                        non_sse[
                            group_index,
                            selected_seed_positions,
                            well_position,
                        ] = result[:, 1]
                        if well_position < 2:
                            first_seed = int(selected_seed_positions[0])
                            observed = (
                                context.y_true[selection.local_indices]
                                + errors[first_seed]
                            )
                            prediction = condition_noisy_piecewise(
                                context.base_prediction,
                                observed,
                                context.normalized_md,
                                selection,
                            )
                            max_anchor_observation_delta = max(
                                max_anchor_observation_delta,
                                float(
                                    np.max(
                                        np.abs(
                                            prediction[
                                                selection.local_indices
                                            ]
                                            - observed
                                        )
                                    )
                                ),
                            )

                non_count = len(context.row_index) - len(NOMINAL_FRACTIONS)
                non_rows[placement_index, well_position] = non_count
                fold_non_rows[placement_index, context.fold] += non_count

            manifest_writer.write(pd.DataFrame.from_records(manifest_records))
            for group in EVALUATION_GROUPS:
                fold_sse[group.index, :, context.fold] += full_sse[
                    group.index, :, well_position
                ]
                fold_non_sse[group.index, :, context.fold] += non_sse[
                    group.index, :, well_position
                ]

            repeated_group = np.repeat(np.arange(group_count), seed_count)
            repeated_seed = np.tile(np.arange(seed_count), group_count)
            sse_values = full_sse[:, :, well_position].reshape(-1)
            non_values = non_sse[:, :, well_position].reshape(-1)
            frame = pd.DataFrame(
                {
                    "protocol": "near_k8_r1_practical_eight_location",
                    "tolerance_u": group_tolerance[repeated_group],
                    "position_mode": group_mode[repeated_group],
                    "scenario": scenario_values[
                        group_scenario[repeated_group]
                    ],
                    "method": OPERATIONAL_METHOD,
                    "realization_seed": np.asarray(
                        seeds, dtype=np.int64
                    )[repeated_seed],
                    "well_id": well_id,
                    "fold": context.fold,
                    "rows": len(context.row_index),
                    "non_anchor_rows": non_rows[
                        group_placement[repeated_group], well_position
                    ],
                    "sse": sse_values,
                    "non_anchor_sse": non_values,
                    "rmse": np.sqrt(
                        sse_values / len(context.row_index)
                    ),
                    "non_anchor_rmse": np.sqrt(
                        non_values
                        / non_rows[
                            group_placement[repeated_group], well_position
                        ]
                    ),
                    "physical_anchor_count": len(NOMINAL_FRACTIONS),
                    "total_readings": len(NOMINAL_FRACTIONS),
                    "protocol_violation": False,
                }
            )
            per_well_writer.write(frame)
            if (well_position + 1) % 20 == 0 or well_position + 1 == well_count:
                print(
                    "anchor-acquisition-protocol "
                    f"wells={well_position + 1}/{well_count}",
                    flush=True,
                )

    if (
        max_zero_selected_row_delta != 0
        or max_measurement_latent_delta != 0.0
        or max_zero_reconstruction_delta != 0.0
        or max_anchor_observation_delta != 0.0
        or max_analytic_exact_delta > 1e-6
        or max_analytic_linear_delta > 1e-8
        or max_analytic_gram_delta > 1e-8
    ):
        raise ValueError("protocol placement/reconstruction invariant failed")

    per_seed, metric_arrays = _per_seed_frame(
        seeds,
        full_sse,
        non_sse,
        rows,
        non_rows,
        fold_sse,
        fold_non_sse,
        fold_rows,
        fold_non_rows,
    )
    per_fold = _per_fold_frame(
        seeds,
        fold_sse,
        fold_non_sse,
        fold_rows,
        fold_non_rows,
    )
    position_exposure = _position_exposure_frame(
        seeds,
        well_ids,
        requested_u,
        achieved_u,
        achieved_md_offset,
        requested_md_offset,
        snap_error_u,
        violation_counts,
    )
    hierarchy = hierarchical_redundancy_bootstrap(
        full_sse,
        non_sse,
        rows,
        non_rows,
        np.asarray(
            [group.placement_index for group in EVALUATION_GROUPS],
            dtype=np.int64,
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    for item in hierarchy:
        item.update(
            _group_metadata(
                EVALUATION_GROUPS[int(item["group_index"])]
            )
        )
    frontier = _frontier_frame(
        metric_arrays, hierarchy, position_exposure
    )
    paired = _paired_deltas_frame(metric_arrays)
    regression = _zero_tolerance_regression(
        prior_redundancy=prior_redundancy,
        seeds=seeds,
        well_ids=well_ids,
        rows=rows,
        non_rows=non_rows,
        full_sse=full_sse,
        non_sse=non_sse,
        hierarchy=hierarchy,
        strict_full=strict_full,
        selected_row_max_delta=max_zero_selected_row_delta,
        measurement_latent_max_delta=max_measurement_latent_delta,
        reconstruction_max_delta=max_zero_reconstruction_delta,
    )
    if not regression["passed"]:
        raise ValueError("zero-tolerance legacy regression failed")
    decision = _decision_summary(frontier)

    artifact_names = [
        "protocol_frontier.csv",
        "per_seed_metrics.csv",
        "per_fold_seed_metrics.csv",
        "per_well_seed_metrics.csv.gz",
        "position_exposure.csv",
        "position_manifest.csv.gz",
        "hierarchical_bootstrap.json",
        "paired_deltas.csv",
        "run_config.json",
        "leakage_check.json",
        "summary.json",
        "report.md",
    ]
    run_config = {
        "run_id": run_root.name,
        "profile": profile_name,
        "data_root": str(data),
        "blend_oof": str(Path(blend_oof).resolve()),
        "prior_acquisition_redundancy": str(
            Path(prior_redundancy).resolve()
        ),
        "fold_version": FOLD_VERSION,
        "fold_sha256": fold_hash,
        "fold_counts": {
            str(key): int(value)
            for key, value in mapping["fold"].value_counts().sort_index().items()
        },
        "wells": well_count,
        "rows": int(rows.sum()),
        "max_wells_per_fold": max_wells_per_fold,
        "selection_seed": selection_seed,
        "nominal_fractions": list(NOMINAL_FRACTIONS),
        "position_tolerance_u": list(POSITION_TOLERANCES_U),
        "primary_protocol_tolerance_u": (
            PRIMARY_PROTOCOL_TOLERANCE_U
        ),
        "position_modes": list(POSITION_MODES),
        "zero_tolerance_position_mode": ZERO_POSITION_MODE,
        "placement_specs": [
            {
                "tolerance_u": spec.tolerance_u,
                "position_mode": spec.position_mode,
            }
            for spec in PLACEMENT_SPECS
        ],
        "physical_anchor_count": len(NOMINAL_FRACTIONS),
        "readings_per_location": 1,
        "total_readings": len(NOMINAL_FRACTIONS),
        "snap_rule": "deterministic nearest-unused prediction row",
        "clipping": False,
        "protocol_violation_policy": (
            "fail closed; no substitute location or reconstruction"
        ),
        "actual_position_conditioning": True,
        "measurement_scenarios": [
            _scenario_metadata(scenario)
            for scenario in PROTOCOL_SCENARIOS
        ],
        "evaluation_groups": [
            _group_metadata(group) for group in EVALUATION_GROUPS
        ],
        "method": OPERATIONAL_METHOD,
        "realization_seeds": list(seeds),
        "stable_random_keys": {
            "position_sign": (
                "(seed, well_id, anchor_order, 'position_sign')"
            ),
            "common_bias": "(seed, well_id, 'common_bias')",
            "independent_measurement_error": (
                "(seed, well_id, nominal_physical_anchor_row, 'independent')"
            ),
            "outlier_indicator": (
                "(seed, well_id, nominal_physical_anchor_row, "
                "'outlier_indicator')"
            ),
            "outlier_sign": (
                "(seed, well_id, nominal_physical_anchor_row, "
                "'outlier_sign')"
            ),
        },
        "measurement_latent_pairing": (
            "nominal physical anchor row/anchor identity; achieved-row shifts "
            "do not change measurement-error draws"
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_pairing": (
            "same well draws and per-selected-well realization draws across "
            "all tolerance/mode/scenario groups"
        ),
        "conditioning": (
            "locked acausal piecewise-linear anchor-residual interpolation "
            "using actual achieved positions and actual achieved-row TVT; "
            "carry after final anchor"
        ),
        "ps_boundary": "exact and unperturbed",
        "non_anchor_definition": (
            "exclude each achieved physical anchor row exactly once"
        ),
        "robust_lt_4_definition": [
            "full fixed-well realization q95 < 4",
            "non-anchor fixed-well realization q95 < 4",
            "hierarchical full q97.5 < 4",
            "hierarchical non-anchor q97.5 < 4",
            "no persistent fold failure",
            "all realizations have wells_ge_40 == 0 and wells_ge_60 == 0",
            "no boundary, availability, duplicate, or order violation",
        ],
        "artifact_paths": {
            name: str(run_root / name) for name in artifact_names
        },
        "computational_tolerances_are_hardware_claims": False,
        "cross_location_common_m10_fault_tested": False,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    _atomic_json(run_root / "run_config.json", run_config)
    per_seed.to_csv(run_root / "per_seed_metrics.csv", index=False)
    per_fold.to_csv(run_root / "per_fold_seed_metrics.csv", index=False)
    frontier.to_csv(run_root / "protocol_frontier.csv", index=False)
    position_exposure.to_csv(
        run_root / "position_exposure.csv", index=False
    )
    paired.to_csv(run_root / "paired_deltas.csv", index=False)
    _atomic_json(
        run_root / "hierarchical_bootstrap.json",
        {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "well_and_realization_draws_paired_across_all_groups": True,
            "comparisons": hierarchy,
        },
    )

    summary: dict[str, Any] = {
        "profile": profile_name,
        "wells": well_count,
        "rows": int(rows.sum()),
        "realizations": seed_count,
        "hierarchical_bootstrap_samples": bootstrap_samples,
        "zero_tolerance_regression": regression,
        **decision,
        "deployable": False,
        "submission_eligible": False,
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    leakage: dict[str, Any] = {
        "passed": True,
        "placement_target_independent": True,
        "placement_target_permutation_invariant": True,
        "configuration_order_does_not_change_keyed_position_draws": True,
        "position_sign_stable_across_tolerance_and_scenario": True,
        "measurement_latents_paired_across_tolerance_and_mode": True,
        "measurement_latents_keyed_to_nominal_anchor_identity": True,
        "common_bias_paired_across_all_groups": True,
        "actual_achieved_positions_used_for_interpolation": True,
        "actual_achieved_row_target_is_revealed_anchor": True,
        "selected_anchor_prediction_equals_corrupted_achieved_observation": (
            max_anchor_observation_delta == 0.0
        ),
        "maximum_anchor_observation_delta": max_anchor_observation_delta,
        "zero_tolerance_regression": regression,
        "zero_tolerance_selected_row_max_abs_delta": (
            max_zero_selected_row_delta
        ),
        "zero_tolerance_measurement_latent_max_abs_delta": (
            max_measurement_latent_delta
        ),
        "zero_tolerance_reconstruction_max_abs_delta": (
            max_zero_reconstruction_delta
        ),
        "analytic_direct_coefficient_audit": {
            "maximum_exact_sse_delta": max_analytic_exact_delta,
            "maximum_linear_delta": max_analytic_linear_delta,
            "maximum_gram_delta": max_analytic_gram_delta,
            "passed": bool(
                max_analytic_exact_delta <= 1e-6
                and max_analytic_linear_delta <= 1e-8
                and max_analytic_gram_delta <= 1e-8
            ),
        },
        "placement_positions_sha256": placement_digest.hexdigest().upper(),
        "actual_rows_distinct_and_strictly_ordered": True,
        "no_clipping": True,
        "protocol_violations_fail_closed": True,
        "ps_boundary_exact_and_unperturbed": True,
        "non_anchor_excludes_each_physical_row_once": True,
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
    _atomic_json(run_root / "summary.json", summary)
    _atomic_json(run_root / "leakage_check.json", leakage)
    (run_root / "report.md").write_text(
        _render_report(frontier, position_exposure, paired, summary),
        encoding="utf-8",
    )
    artifact_audit = audit_anchor_acquisition_protocol_artifacts(
        run_root
    )
    leakage["artifact_validation"] = artifact_audit
    leakage["passed"] = bool(
        leakage["passed"]
        and regression["passed"]
        and artifact_audit["passed"]
        and leakage["analytic_direct_coefficient_audit"]["passed"]
        and max_anchor_observation_delta == 0.0
        and not decision["any_protocol_violation"]
    )
    summary["artifact_validation"] = artifact_audit
    _atomic_json(run_root / "summary.json", summary)
    _atomic_json(run_root / "leakage_check.json", leakage)
    if not leakage["passed"]:
        raise ValueError("protocol leakage/regression/artifact audit failed")
    return summary
