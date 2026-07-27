"""Locked sparse-anchor acquisition redundancy frontier.

This module compares additional physical near-terminal TVT anchors with
same-location repeat measurements under the error model established by
``anchor_error_realism``.  It is deliberately target-bearing, acausal, and
retrospective.  It never changes the base prediction, placement family,
interpolation rule, or error realizations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data import enumerate_wells
from ..folds import load_fold_mapping
from .anchor_error_realism import (
    ErrorScenario,
    METRIC_TOLERANCE,
    PER_WELL_SSE_TOLERANCE,
    PERSISTENT_FOLD_FAILURE_RATE,
    PERSISTENT_FOLD_RMSE_THRESHOLD,
    ROWWISE_TOLERANCE,
    _mask_coefficients,
    _normal_from_digest,
    _scenario_metadata,
    _validate_well_id,
    stable_anchor_latents,
    stable_common_bias_normal,
    stable_independent_normal,
    stable_outlier_indicator_uniform,
    stable_outlier_sign,
)
from .anchor_noise_robustness import (
    DEFAULT_NOISE_SEEDS,
    NoiseWellContext,
    _build_noise_contexts,
    _read_blend_oof,
    _selected_wells,
    _validate_seed,
    condition_noisy_piecewise,
    piecewise_noise_weights,
)
from .anchor_outlier_qc import _coefficients_for_subset, _selection_from_mask
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
    requested_fractions,
    select_terminal_anchors,
)


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260728
SEGMENTS = ("pre_first", "between", "post_last")
METHODS = (
    "observed_aggregate",
    "oracle_drop_outlier_replicates",
    "oracle_clean_values",
)
OPERATIONAL_METHOD = "observed_aggregate"
DEPENDENCE_MEASUREMENT_INDEPENDENT = "measurement_independent"
DEPENDENCE_ANCHOR_SHARED = "anchor_shared"
DEPENDENCE_SINGLE = "single_read_equivalent"
DEPENDENCE_CLEAN = "clean_equivalent"
DEPENDENCES = (
    DEPENDENCE_MEASUREMENT_INDEPENDENT,
    DEPENDENCE_ANCHOR_SHARED,
    DEPENDENCE_SINGLE,
    DEPENDENCE_CLEAN,
)
O5_SCENARIO = "O5_moderate_p10_m10"


@dataclass(frozen=True)
class AcquisitionDesign:
    name: str
    k: int
    replicates: int
    aggregation: str
    acquisition_class: str

    def __post_init__(self) -> None:
        validate_acquisition_design(
            self.k, self.replicates, self.aggregation, self.acquisition_class
        )

    @property
    def fractions(self) -> tuple[float, ...]:
        return tuple(float(value) for value in requested_fractions("near_terminal", self.k))

    @property
    def fractions_text(self) -> str:
        return json.dumps(
            [round(value, 12) for value in self.fractions],
            separators=(",", ":"),
        )

    @property
    def total_readings(self) -> int:
        return self.k * self.replicates


def validate_acquisition_design(
    k: int,
    replicates: int,
    aggregation: str,
    acquisition_class: str,
) -> None:
    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or int(k) < 1
    ):
        raise ValueError("physical anchor count must be a positive integer")
    if (
        isinstance(replicates, (bool, np.bool_))
        or not isinstance(replicates, (int, np.integer))
        or int(replicates) not in {1, 2, 3}
    ):
        raise ValueError("replicate count must be one, two, or three")
    expected = {1: "identity", 2: "mean", 3: "median"}[int(replicates)]
    if str(aggregation) != expected:
        raise ValueError(
            f"{replicates} replicates require locked {expected} aggregation"
        )
    expected_class = (
        "single_read_spatial"
        if int(replicates) == 1
        else "same_location_repeat"
    )
    if (
        str(acquisition_class) not in {"single_read_spatial", "same_location_repeat", "hybrid"}
        or (
            str(acquisition_class) == "single_read_spatial"
            and int(replicates) != 1
        )
        or (
            str(acquisition_class) == "hybrid"
            and (int(k), int(replicates)) != (8, 2)
        )
        or (
            str(acquisition_class) == "same_location_repeat"
            and int(k) != 6
        )
    ):
        raise ValueError("acquisition class does not match the locked design")
    if expected_class == "single_read_spatial" and str(acquisition_class) != expected_class:
        raise ValueError("single-read designs must be spatial")


LOCKED_DESIGNS = tuple(
    [
        AcquisitionDesign(
            f"near_k{k}_r1", k, 1, "identity", "single_read_spatial"
        )
        for k in range(6, 13)
    ]
    + [
        AcquisitionDesign(
            "near_k6_r2_mean", 6, 2, "mean", "same_location_repeat"
        ),
        AcquisitionDesign(
            "near_k6_r3_median", 6, 3, "median", "same_location_repeat"
        ),
        AcquisitionDesign(
            "near_k8_r2_mean", 8, 2, "mean", "hybrid"
        ),
    ]
)


def locked_redundancy_scenarios() -> tuple[ErrorScenario, ...]:
    return (
        ErrorScenario("control", "exact_clean", 0.0, 0.0, 0.0, 0.0, 0.0),
        ErrorScenario(
            "control", "gaussian_1_clean", 0.0, 1.0, 0.0, 0.0, 1.0
        ),
        ErrorScenario(
            "control", "moderate_background", 1.0, 1.0, 0.0, 0.0, 1.0
        ),
        ErrorScenario(
            "control", "moderate_m5_control", 1.0, 1.0, 0.10, 5.0, 1.0
        ),
        ErrorScenario(
            "margin", "moderate_p05_m10", 1.0, 1.0, 0.05, 10.0, 1.0
        ),
        ErrorScenario(
            "primary", O5_SCENARIO, 1.0, 1.0, 0.10, 10.0, 1.0
        ),
        ErrorScenario(
            "stress", "moderate_p20_m10", 1.0, 1.0, 0.20, 10.0, 1.0
        ),
        ErrorScenario(
            "attribution",
            "S0_severe_background",
            2.0,
            2.0,
            0.0,
            0.0,
            2.0,
        ),
        ErrorScenario(
            "attribution",
            "S1_severe_p10_m10",
            2.0,
            2.0,
            0.10,
            10.0,
            2.0,
        ),
    )


REDUNDANCY_SCENARIOS = locked_redundancy_scenarios()


def select_designs(names: Sequence[str] | None) -> tuple[AcquisitionDesign, ...]:
    if names is None:
        return LOCKED_DESIGNS
    requested = tuple(dict.fromkeys(map(str, names)))
    by_name = {design.name: design for design in LOCKED_DESIGNS}
    if not requested or any(name not in by_name for name in requested):
        raise ValueError("design names must be a non-empty locked subset")
    return tuple(by_name[name] for name in requested)


def select_redundancy_scenarios(
    names: Sequence[str] | None,
) -> tuple[ErrorScenario, ...]:
    if names is None:
        return REDUNDANCY_SCENARIOS
    requested = tuple(dict.fromkeys(map(str, names)))
    by_name = {scenario.name: scenario for scenario in REDUNDANCY_SCENARIOS}
    if not requested or any(name not in by_name for name in requested):
        raise ValueError("scenario names must be a non-empty locked subset")
    return tuple(by_name[name] for name in requested)


def validate_dependence(value: str) -> str:
    dependence = str(value)
    if dependence not in DEPENDENCES:
        raise ValueError(f"unknown gross-error dependence: {dependence}")
    return dependence


def _replicate_digest(
    seed: int,
    well_id: str,
    anchor_row: int,
    replicate: int,
    key: str,
) -> bytes:
    validated_seed = _validate_seed(seed)
    encoded_well = _validate_well_id(well_id)
    row = int(anchor_row)
    rep = int(replicate)
    if (
        isinstance(anchor_row, (bool, np.bool_))
        or row < 0
        or isinstance(replicate, (bool, np.bool_))
        or rep < 1
    ):
        raise ValueError("replicate keys require non-negative row and positive replicate")
    encoded_key = str(key).encode("ascii")
    if not encoded_key or b"\x00" in encoded_key:
        raise ValueError("replicate random key must be non-empty ASCII")
    digest = hashlib.sha256()
    digest.update(b"rogii-anchor-acquisition-redundancy-v1\x00")
    digest.update(validated_seed.to_bytes(8, "big", signed=False))
    digest.update(len(encoded_well).to_bytes(4, "big", signed=False))
    digest.update(encoded_well)
    digest.update(row.to_bytes(8, "big", signed=False))
    digest.update(rep.to_bytes(4, "big", signed=False))
    digest.update(len(encoded_key).to_bytes(2, "big", signed=False))
    digest.update(encoded_key)
    return digest.digest()


def stable_replicate_independent_normal(
    seed: int, well_id: str, anchor_row: int, replicate: int
) -> float:
    if int(replicate) == 0:
        return stable_independent_normal(seed, well_id, anchor_row)
    return _normal_from_digest(
        _replicate_digest(seed, well_id, anchor_row, replicate, "independent")
    )


def stable_replicate_outlier_indicator_uniform(
    seed: int, well_id: str, anchor_row: int, replicate: int
) -> float:
    if int(replicate) == 0:
        return stable_outlier_indicator_uniform(seed, well_id, anchor_row)
    digest = _replicate_digest(
        seed, well_id, anchor_row, replicate, "outlier_indicator"
    )
    return (int.from_bytes(digest[:8], "big") + 0.5) / float(1 << 64)


def stable_replicate_outlier_sign(
    seed: int, well_id: str, anchor_row: int, replicate: int
) -> int:
    if int(replicate) == 0:
        return stable_outlier_sign(seed, well_id, anchor_row)
    digest = _replicate_digest(
        seed, well_id, anchor_row, replicate, "outlier_sign"
    )
    uniform = (int.from_bytes(digest[:8], "big") + 0.5) / float(1 << 64)
    return -1 if uniform < 0.5 else 1


@dataclass(frozen=True)
class ReplicateLatents:
    common_bias: np.ndarray
    independent: np.ndarray
    outlier_indicator_uniform: np.ndarray
    outlier_sign: np.ndarray


def stable_replicate_latents(
    seeds: Sequence[int],
    well_id: str,
    anchor_rows: Sequence[int] | np.ndarray,
    replicates: int,
) -> ReplicateLatents:
    seed_values = tuple(_validate_seed(seed) for seed in seeds)
    rows = np.asarray(anchor_rows)
    if (
        not seed_values
        or len(set(seed_values)) != len(seed_values)
        or rows.ndim != 1
        or isinstance(replicates, (bool, np.bool_))
        or int(replicates) not in {1, 2, 3}
    ):
        raise ValueError("invalid seed/row/replicate latent request")
    legacy = stable_anchor_latents(seed_values, well_id, rows)
    independent = np.empty(
        (len(seed_values), len(rows), int(replicates)), dtype=np.float64
    )
    indicator = np.empty_like(independent)
    signs = np.empty_like(independent, dtype=np.int8)
    independent[:, :, 0] = legacy.independent
    indicator[:, :, 0] = legacy.outlier_indicator_uniform
    signs[:, :, 0] = legacy.outlier_sign
    for seed_position, seed in enumerate(seed_values):
        for row_position, row in enumerate(rows):
            for replicate in range(1, int(replicates)):
                independent[seed_position, row_position, replicate] = (
                    stable_replicate_independent_normal(
                        seed, well_id, int(row), replicate
                    )
                )
                indicator[seed_position, row_position, replicate] = (
                    stable_replicate_outlier_indicator_uniform(
                        seed, well_id, int(row), replicate
                    )
                )
                signs[seed_position, row_position, replicate] = (
                    stable_replicate_outlier_sign(
                        seed, well_id, int(row), replicate
                    )
                )
    return ReplicateLatents(
        common_bias=legacy.common_bias,
        independent=independent,
        outlier_indicator_uniform=indicator,
        outlier_sign=signs,
    )


def aggregate_replicates(values: np.ndarray, aggregation: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or not np.isfinite(array).all():
        raise ValueError("replicate values must be a finite array")
    replicate_count = array.shape[-1]
    if aggregation == "identity":
        if replicate_count != 1:
            raise ValueError("identity aggregation requires exactly one replicate")
        return array[..., 0].copy()
    if aggregation == "mean":
        if replicate_count != 2:
            raise ValueError("mean aggregation is locked to two replicates")
        return np.mean(array, axis=-1)
    if aggregation == "median":
        if replicate_count != 3:
            raise ValueError("median aggregation is locked to three replicates")
        return np.median(array, axis=-1)
    raise ValueError(f"unknown locked aggregation: {aggregation}")


def _aggregate_variable_clean_replicates(
    values: np.ndarray,
    clean: np.ndarray,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(values, dtype=np.float64)
    retained = np.asarray(clean, dtype=bool)
    if observations.shape != retained.shape or observations.ndim != 3:
        raise ValueError("oracle replicate observations and masks must align")
    seeds, locations, _ = observations.shape
    result = np.zeros((seeds, locations), dtype=np.float64)
    location_retained = np.any(retained, axis=2)
    for seed_position in range(seeds):
        for location in range(locations):
            selected = observations[seed_position, location][
                retained[seed_position, location]
            ]
            if not len(selected):
                continue
            if aggregation == "identity":
                result[seed_position, location] = float(selected[0])
            elif aggregation == "mean":
                result[seed_position, location] = float(np.mean(selected))
            elif aggregation == "median":
                result[seed_position, location] = float(np.median(selected))
            else:
                raise ValueError(f"unknown locked aggregation: {aggregation}")
    return result, location_retained


def scenario_replicate_errors(
    scenario: ErrorScenario,
    latents: ReplicateLatents,
    dependence: str,
    aggregation: str,
) -> dict[str, np.ndarray]:
    """Return aggregate errors and post-hoc exposure fields for one scenario."""
    mode = validate_dependence(dependence)
    replicate_count = latents.independent.shape[2]
    if replicate_count == 1 and mode not in {DEPENDENCE_SINGLE, DEPENDENCE_CLEAN}:
        raise ValueError("single-read designs must use the equivalent dependence")
    if scenario.outlier_probability == 0.0 and mode not in {
        DEPENDENCE_SINGLE,
        DEPENDENCE_CLEAN,
    }:
        raise ValueError("clean scenarios are calculated once")
    if replicate_count > 1 and scenario.outlier_probability > 0.0 and mode not in {
        DEPENDENCE_MEASUREMENT_INDEPENDENT,
        DEPENDENCE_ANCHOR_SHARED,
    }:
        raise ValueError("repeat outlier scenarios require a locked dependence")
    no_outlier_replicate = (
        scenario.tau * latents.common_bias[:, None, None]
        + scenario.sigma * latents.independent
    )
    if mode == DEPENDENCE_ANCHOR_SHARED:
        uniform = np.broadcast_to(
            latents.outlier_indicator_uniform[:, :, :1],
            latents.outlier_indicator_uniform.shape,
        )
        signs = np.broadcast_to(
            latents.outlier_sign[:, :, :1], latents.outlier_sign.shape
        )
    else:
        uniform = latents.outlier_indicator_uniform
        signs = latents.outlier_sign
    flags = uniform < scenario.outlier_probability
    contributions = (
        flags.astype(np.float64)
        * signs.astype(np.float64)
        * scenario.outlier_magnitude
    )
    observed_replicate = no_outlier_replicate + contributions
    aggregate_error = aggregate_replicates(observed_replicate, aggregation)
    aggregate_no_outlier = aggregate_replicates(
        no_outlier_replicate, aggregation
    )
    clean_error, retained_locations = _aggregate_variable_clean_replicates(
        no_outlier_replicate, ~flags, aggregation
    )
    return {
        "aggregate_error": aggregate_error,
        "aggregate_no_outlier_error": aggregate_no_outlier,
        "effective_outlier_shift": aggregate_error - aggregate_no_outlier,
        "flags": flags,
        "signs": signs,
        "contributions": contributions,
        "oracle_clean_replicate_error": clean_error,
        "oracle_retained_locations": retained_locations,
    }


def _dependences_for(
    design: AcquisitionDesign, scenario: ErrorScenario
) -> tuple[str, ...]:
    if design.replicates == 1:
        return (DEPENDENCE_SINGLE,)
    if scenario.outlier_probability == 0.0:
        return (DEPENDENCE_CLEAN,)
    return (
        DEPENDENCE_MEASUREMENT_INDEPENDENT,
        DEPENDENCE_ANCHOR_SHARED,
    )


@dataclass(frozen=True)
class EvaluationGroup:
    index: int
    design_index: int
    scenario_index: int
    dependence: str
    method: str


def build_evaluation_groups(
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
) -> tuple[EvaluationGroup, ...]:
    groups: list[EvaluationGroup] = []
    for design_index, design in enumerate(designs):
        for scenario_index, scenario in enumerate(scenarios):
            for dependence in _dependences_for(design, scenario):
                groups.append(
                    EvaluationGroup(
                        len(groups),
                        design_index,
                        scenario_index,
                        dependence,
                        OPERATIONAL_METHOD,
                    )
                )
                if scenario.name == O5_SCENARIO:
                    for method in METHODS[1:]:
                        groups.append(
                            EvaluationGroup(
                                len(groups),
                                design_index,
                                scenario_index,
                                dependence,
                                method,
                            )
                        )
    return tuple(groups)


def _group_metadata(
    group: EvaluationGroup,
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
) -> dict[str, Any]:
    design = designs[group.design_index]
    scenario = scenarios[group.scenario_index]
    return {
        "config": design.name,
        "acquisition_class": design.acquisition_class,
        "physical_anchor_count": design.k,
        "replicates_per_anchor": design.replicates,
        "aggregation": design.aggregation,
        "total_readings": design.total_readings,
        "fractions": design.fractions_text,
        "method": group.method,
        "outlier_dependence": group.dependence,
        **_scenario_metadata(scenario),
    }


def _retained_mask_values(retained: np.ndarray) -> np.ndarray:
    keep = np.asarray(retained, dtype=bool)
    if keep.ndim != 2 or keep.shape[1] > 62:
        raise ValueError("retained-location masks must be 2D with <=62 locations")
    powers = np.left_shift(
        np.uint64(1), np.arange(keep.shape[1], dtype=np.uint64)
    )
    return np.sum(
        keep.astype(np.uint64) * powers[None, :],
        axis=1,
        dtype=np.uint64,
    )


def _evaluate_fixed_subset(
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    errors: np.ndarray,
) -> np.ndarray:
    exact_sse, linear, gram, _ = coefficients
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != linear.shape[1]:
        raise ValueError("fixed-subset errors do not align")
    cross = values @ linear.T
    quadratic = np.einsum(
        "si,mij,sj->sm", values, gram, values, optimize=True
    )
    result = exact_sse[None, :] + 2.0 * cross + quadratic
    tolerance = (
        np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(exact_sse))))
        * 128
    )
    if float(np.min(result)) < -tolerance:
        raise ValueError("fixed-subset SSE became materially negative")
    return np.maximum(result, 0.0)


def _evaluate_variable_subsets(
    context: NoiseWellContext,
    selection: AnchorSelection,
    true_anchors: np.ndarray,
    retained_masks: np.ndarray,
    anchor_errors: np.ndarray,
    metric_masks: np.ndarray,
    full_coefficients: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> np.ndarray:
    """Evaluate oracle location removal, grouping identical retained geometry."""
    masks = np.asarray(retained_masks, dtype=np.uint64)
    errors = np.asarray(anchor_errors, dtype=np.float64)
    if (
        masks.ndim != 1
        or errors.ndim != 2
        or len(masks) != len(errors)
        or errors.shape[1] != selection.achieved_k
    ):
        raise ValueError("variable subset masks/errors must align")
    full_mask = np.uint64((1 << selection.achieved_k) - 1)
    result = np.empty((len(errors), metric_masks.shape[0]), dtype=np.float64)
    for subset_value in np.unique(masks):
        selected_seed = np.flatnonzero(masks == subset_value)
        subset_int = int(subset_value)
        if subset_value == full_mask:
            result[selected_seed] = _evaluate_fixed_subset(
                full_coefficients, errors[selected_seed]
            )
            continue
        subset, accepted = _selection_from_mask(selection, subset_int)
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
                (len(context.row_index), selection.achieved_k),
                dtype=np.float64,
            )
            weights[:, accepted] = subset_weights
        else:
            exact_prediction = context.base_prediction
            weights = np.zeros(
                (len(context.row_index), selection.achieved_k),
                dtype=np.float64,
            )
        residual = (
            exact_prediction[None, :]
            - context.y_true[None, :]
            + errors[selected_seed] @ weights.T
        )
        result[selected_seed] = np.square(residual) @ metric_masks.T
        tolerance = (
            np.finfo(np.float64).eps
            * max(1.0, float(np.max(np.abs(result[selected_seed]))))
            * 128
        )
        if float(np.min(result[selected_seed])) < -tolerance:
            raise ValueError("variable-subset SSE became materially negative")
    return result


def reconstruct_pooled_metrics(per_well: pd.DataFrame) -> tuple[float, float]:
    if (
        not {"rows", "non_anchor_rows", "sse", "non_anchor_sse"}.issubset(
            per_well.columns
        )
        or int(per_well["rows"].sum()) <= 0
        or int(per_well["non_anchor_rows"].sum()) <= 0
    ):
        raise ValueError("invalid compact per-well metric frame")
    return (
        float(np.sqrt(per_well["sse"].sum() / per_well["rows"].sum())),
        float(
            np.sqrt(
                per_well["non_anchor_sse"].sum()
                / per_well["non_anchor_rows"].sum()
            )
        ),
    )


def hierarchical_redundancy_bootstrap(
    full_sse: np.ndarray,
    non_anchor_sse: np.ndarray,
    rows: np.ndarray,
    non_anchor_rows: np.ndarray,
    group_design_indices: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Paired hierarchical well/error bootstrap for arbitrary evaluation groups."""
    full = np.asarray(full_sse, dtype=np.float64)
    non = np.asarray(non_anchor_sse, dtype=np.float64)
    row_counts = np.asarray(rows, dtype=np.int64)
    non_counts = np.asarray(non_anchor_rows, dtype=np.int64)
    design_index = np.asarray(group_design_indices, dtype=np.int64)
    if (
        full.ndim != 3
        or non.shape != full.shape
        or row_counts.shape != (full.shape[2],)
        or non_counts.ndim != 2
        or non_counts.shape[1] != full.shape[2]
        or design_index.shape != (full.shape[0],)
        or np.any(design_index < 0)
        or np.any(design_index >= non_counts.shape[0])
        or isinstance(samples, (bool, np.bool_))
        or int(samples) < 1
        or not np.isfinite(full).all()
        or not np.isfinite(non).all()
        or np.any(full < 0)
        or np.any(non < 0)
    ):
        raise ValueError("invalid redundancy bootstrap inputs")
    bootstrap_seed = _validate_seed(seed)
    group_count, realization_count, well_count = full.shape
    rng = np.random.default_rng(bootstrap_seed)
    well_draws = rng.integers(
        0, well_count, size=(int(samples), well_count), dtype=np.int16
    )
    realization_draws = rng.integers(
        0,
        realization_count,
        size=(int(samples), well_count),
        dtype=np.uint8,
    )
    sampled_rows = row_counts[well_draws].sum(axis=1, dtype=np.int64)
    sampled_non_rows_by_design = [
        non_counts[index][well_draws].sum(axis=1, dtype=np.int64)
        for index in range(non_counts.shape[0])
    ]

    def q(values: np.ndarray) -> dict[str, float]:
        return {
            "q025": float(np.quantile(values, 0.025)),
            "q50": float(np.quantile(values, 0.50)),
            "q975": float(np.quantile(values, 0.975)),
        }

    output: list[dict[str, Any]] = []
    for group_index in range(group_count):
        design = int(design_index[group_index])
        sampled_non_rows = sampled_non_rows_by_design[design]
        group_full = full[group_index]
        group_non = non[group_index]
        hierarchical_full = np.sqrt(
            group_full[realization_draws, well_draws].sum(axis=1)
            / sampled_rows
        )
        hierarchical_non = np.sqrt(
            group_non[realization_draws, well_draws].sum(axis=1)
            / sampled_non_rows
        )
        output.append(
            {
                "group_index": group_index,
                "hierarchical": {
                    "full_rmse": q(hierarchical_full),
                    "non_anchor_rmse": q(hierarchical_non),
                },
                "samples": int(samples),
                "bootstrap_seed": bootstrap_seed,
            }
        )
    return output


def _quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _tail_metrics(full_sse: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
    rmse = np.sqrt(full_sse / rows[None, None, :])
    worst_count = max(1, int(math.ceil(0.10 * len(rows))))
    sorted_sse = np.sort(full_sse, axis=2)
    denominator = full_sse.sum(axis=2)
    share = np.divide(
        sorted_sse[:, :, -worst_count:].sum(axis=2),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return {
        "well_rmse": rmse,
        "p50": np.quantile(rmse, 0.50, axis=2),
        "p90": np.quantile(rmse, 0.90, axis=2),
        "p95": np.quantile(rmse, 0.95, axis=2),
        "max": np.max(rmse, axis=2),
        "wells_ge_40": np.sum(rmse >= 40.0, axis=2),
        "wells_ge_60": np.sum(rmse >= 60.0, axis=2),
        "worst10_sse_share": share,
    }


def _validate_selection(
    design: AcquisitionDesign, selection: AnchorSelection
) -> None:
    requested = np.asarray(selection.requested_positions, dtype=np.float64)
    locked = np.asarray(design.fractions, dtype=np.float64)
    if (
        selection.achieved_k != design.k
        or requested.shape != locked.shape
        or not np.array_equal(requested, locked)
        or len(np.unique(selection.local_indices)) != design.k
        or np.any(np.diff(selection.local_indices) <= 0)
    ):
        raise ValueError(
            f"{design.name} failed its locked achieved-count/fraction contract"
        )


def _metric_masks(
    context: NoiseWellContext, selection: AnchorSelection
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    physical_anchor = anchor_mask(len(context.row_index), selection)
    non_anchor = ~physical_anchor
    regions = frontier_segment_regions(len(context.row_index), selection)
    deciles = np.minimum(
        (context.normalized_md * 10).astype(np.int8), 9
    )
    masks = np.stack(
        [
            np.ones(len(context.row_index), dtype=bool),
            non_anchor,
            *(regions == segment for segment in SEGMENTS),
            *(deciles == decile for decile in range(10)),
            *(
                (deciles == decile) & non_anchor
                for decile in range(10)
            ),
        ]
    )
    return masks, physical_anchor, regions, deciles


def _per_seed_frame(
    groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    full_sse: np.ndarray,
    non_sse: np.ndarray,
    rows: np.ndarray,
    non_rows: np.ndarray,
    folds: np.ndarray,
    fold_sse: np.ndarray,
    fold_non_sse: np.ndarray,
    fold_rows: np.ndarray,
    fold_non_rows: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    design_indices = np.asarray(
        [group.design_index for group in groups], dtype=np.int64
    )
    total_rows = int(rows.sum())
    full_rmse = np.sqrt(full_sse.sum(axis=2) / total_rows)
    non_rmse = np.sqrt(
        non_sse.sum(axis=2)
        / non_rows[design_indices].sum(axis=1)[:, None]
    )
    tail = _tail_metrics(full_sse, rows)
    fold_rmse = np.sqrt(fold_sse / fold_rows[None, None, :])
    group_non_fold_rows = fold_non_rows[design_indices]
    fold_non_rmse = np.sqrt(
        fold_non_sse / group_non_fold_rows[:, None, :]
    )
    records: list[dict[str, Any]] = []
    for group in groups:
        metadata = _group_metadata(group, designs, scenarios)
        for seed_position, seed in enumerate(seeds):
            records.append(
                {
                    **metadata,
                    "realization_seed": int(seed),
                    "wells": len(rows),
                    "rows": total_rows,
                    "non_anchor_rows": int(
                        non_rows[group.design_index].sum()
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
    groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    fold_sse: np.ndarray,
    fold_non_sse: np.ndarray,
    fold_rows: np.ndarray,
    fold_non_rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group in groups:
        metadata = _group_metadata(group, designs, scenarios)
        for seed_position, seed in enumerate(seeds):
            for fold in range(5):
                records.append(
                    {
                        **metadata,
                        "realization_seed": int(seed),
                        "fold": fold,
                        "rows": int(fold_rows[fold]),
                        "non_anchor_rows": int(
                            fold_non_rows[group.design_index, fold]
                        ),
                        "sse": float(
                            fold_sse[group.index, seed_position, fold]
                        ),
                        "non_anchor_sse": float(
                            fold_non_sse[
                                group.index, seed_position, fold
                            ]
                        ),
                        "rmse": float(
                            np.sqrt(
                                fold_sse[
                                    group.index, seed_position, fold
                                ]
                                / fold_rows[fold]
                            )
                        ),
                        "non_anchor_rmse": float(
                            np.sqrt(
                                fold_non_sse[
                                    group.index, seed_position, fold
                                ]
                                / fold_non_rows[
                                    group.design_index, fold
                                ]
                            )
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _per_segment_frame(
    groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    segment_sse: np.ndarray,
    segment_rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group in groups:
        metadata = _group_metadata(group, designs, scenarios)
        for seed_position, seed in enumerate(seeds):
            for segment_index, segment in enumerate(SEGMENTS):
                rows = int(
                    segment_rows[group.design_index, segment_index]
                )
                sse = float(
                    segment_sse[
                        group.index, seed_position, segment_index
                    ]
                )
                records.append(
                    {
                        **metadata,
                        "realization_seed": int(seed),
                        "segment": segment,
                        "rows": rows,
                        "sse": sse,
                        "rmse": float(np.sqrt(sse / rows)),
                    }
                )
    return pd.DataFrame.from_records(records)


def _distance_frame(
    groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    distance_sse: np.ndarray,
    distance_non_sse: np.ndarray,
    distance_rows: np.ndarray,
    distance_non_rows: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group in groups:
        metadata = _group_metadata(group, designs, scenarios)
        for seed_position, seed in enumerate(seeds):
            for decile in range(10):
                rows = int(distance_rows[group.design_index, decile])
                non_rows = int(
                    distance_non_rows[group.design_index, decile]
                )
                sse = float(
                    distance_sse[group.index, seed_position, decile]
                )
                non_sse = float(
                    distance_non_sse[group.index, seed_position, decile]
                )
                records.append(
                    {
                        **metadata,
                        "realization_seed": int(seed),
                        "distance_decile": decile,
                        "rows": rows,
                        "non_anchor_rows": non_rows,
                        "sse": sse,
                        "non_anchor_sse": non_sse,
                        "rmse": float(np.sqrt(sse / rows)),
                        "non_anchor_rmse": float(
                            np.sqrt(non_sse / non_rows)
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _measurement_exposure_frame(
    observed_groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    well_count: int,
    counters: Mapping[str, np.ndarray],
    effective_values: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group in observed_groups:
        design = designs[group.design_index]
        scenario = scenarios[group.scenario_index]
        metadata = _group_metadata(group, designs, scenarios)
        location_denominator = well_count * design.k
        reading_denominator = location_denominator * design.replicates
        for seed_position, seed in enumerate(seeds):
            if group.index in effective_values:
                effective = np.abs(
                    effective_values[group.index][seed_position].astype(
                        np.float64
                    )
                ).reshape(-1)
            else:
                effective = np.zeros(location_denominator, dtype=np.float64)
            eligible = counters["cancellation_eligible"][
                group.index, seed_position
            ]
            error_count = counters["aggregate_error_count"][
                group.index, seed_position
            ]
            records.append(
                {
                    **metadata,
                    "realization_seed": int(seed),
                    "physical_locations": design.k,
                    "total_readings": design.total_readings,
                    "realized_outlier_reading_fraction": float(
                        counters["outlier_readings"][
                            group.index, seed_position
                        ]
                        / reading_denominator
                    ),
                    "any_outlier_location_fraction": float(
                        counters["any_outlier_locations"][
                            group.index, seed_position
                        ]
                        / location_denominator
                    ),
                    "majority_outlier_location_fraction": float(
                        counters["majority_outlier_locations"][
                            group.index, seed_position
                        ]
                        / location_denominator
                    ),
                    "all_replicates_outlier_location_fraction": float(
                        counters["all_outlier_locations"][
                            group.index, seed_position
                        ]
                        / location_denominator
                    ),
                    "wells_with_contaminated_location_fraction": float(
                        counters["wells_any_contamination"][
                            group.index, seed_position
                        ]
                        / well_count
                    ),
                    "wells_with_multiple_contaminated_locations_fraction": float(
                        counters["wells_multiple_contamination"][
                            group.index, seed_position
                        ]
                        / well_count
                    ),
                    "independent_sign_cancellation_fraction": float(
                        counters["sign_cancellations"][
                            group.index, seed_position
                        ]
                        / eligible
                        if eligible
                        else 0.0
                    ),
                    "aggregate_observed_error_mean_abs": float(
                        counters["aggregate_error_abs_sum"][
                            group.index, seed_position
                        ]
                        / error_count
                    ),
                    "aggregate_observed_error_rmse": float(
                        np.sqrt(
                            counters["aggregate_error_square_sum"][
                                group.index, seed_position
                            ]
                            / error_count
                        )
                    ),
                    "aggregate_observed_error_max_abs": float(
                        counters["aggregate_error_max_abs"][
                            group.index, seed_position
                        ]
                    ),
                    "aggregate_idiosyncratic_error_mean_abs": float(
                        counters["idiosyncratic_error_abs_sum"][
                            group.index, seed_position
                        ]
                        / error_count
                    ),
                    "aggregate_idiosyncratic_error_rmse": float(
                        np.sqrt(
                            counters["idiosyncratic_error_square_sum"][
                                group.index, seed_position
                            ]
                            / error_count
                        )
                    ),
                    "effective_outlier_shift_p50": float(
                        np.quantile(effective, 0.50)
                    ),
                    "effective_outlier_shift_p90": float(
                        np.quantile(effective, 0.90)
                    ),
                    "effective_outlier_shift_p95": float(
                        np.quantile(effective, 0.95)
                    ),
                    "effective_outlier_shift_max": float(
                        np.max(effective)
                    ),
                    "effective_abs_ge_2p5_fraction": float(
                        np.mean(effective >= 2.5)
                    ),
                    "effective_abs_ge_5_fraction": float(
                        np.mean(effective >= 5.0)
                    ),
                    "effective_abs_ge_7p5_fraction": float(
                        np.mean(effective >= 7.5)
                    ),
                    "effective_abs_ge_10_fraction": float(
                        np.mean(effective >= 10.0)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _effective_error_frame(
    observed_groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    effective_values: Mapping[int, np.ndarray],
    contaminated_locations: Mapping[int, np.ndarray],
    counters: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def append_scope(
        metadata: dict[str, Any],
        scope: str,
        label: str,
        values: np.ndarray,
    ) -> None:
        absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
        if not len(absolute):
            return
        records.append(
            {
                **metadata,
                "scope": scope,
                "scope_value": label,
                "locations": int(len(absolute)),
                "effective_outlier_shift_p50": float(
                    np.quantile(absolute, 0.50)
                ),
                "effective_outlier_shift_p90": float(
                    np.quantile(absolute, 0.90)
                ),
                "effective_outlier_shift_p95": float(
                    np.quantile(absolute, 0.95)
                ),
                "effective_outlier_shift_max": float(np.max(absolute)),
                "effective_abs_ge_2p5_fraction": float(
                    np.mean(absolute >= 2.5)
                ),
                "effective_abs_ge_5_fraction": float(
                    np.mean(absolute >= 5.0)
                ),
                "effective_abs_ge_7p5_fraction": float(
                    np.mean(absolute >= 7.5)
                ),
                "effective_abs_ge_10_fraction": float(
                    np.mean(absolute >= 10.0)
                ),
            }
        )

    for group in observed_groups:
        design = designs[group.design_index]
        scenario = scenarios[group.scenario_index]
        metadata = _group_metadata(group, designs, scenarios)
        if group.index not in effective_values:
            location_count = int(
                counters["aggregate_error_count"][group.index].sum()
            )
            append_scope(
                metadata,
                "overall",
                "all_locations",
                np.zeros(location_count, dtype=np.float64),
            )
            continue
        effective = effective_values[group.index].astype(np.float64)
        contaminated = contaminated_locations[group.index]
        append_scope(metadata, "overall", "all_locations", effective)
        for order in range(design.k):
            append_scope(
                metadata,
                "anchor_order",
                str(order + 1),
                effective[:, :, order],
            )
        append_scope(
            metadata,
            "endpoint_interior",
            "endpoint",
            effective[:, :, [0, design.k - 1]],
        )
        if design.k > 2:
            append_scope(
                metadata,
                "endpoint_interior",
                "interior",
                effective[:, :, 1:-1],
            )
        contaminated_count = contaminated.sum(axis=2)
        isolated_mask = np.broadcast_to(
            (contaminated_count == 1)[:, :, None], effective.shape
        )
        multiple_mask = np.broadcast_to(
            (contaminated_count >= 2)[:, :, None], effective.shape
        )
        append_scope(
            metadata,
            "well_contamination",
            "isolated",
            effective[isolated_mask],
        )
        append_scope(
            metadata,
            "well_contamination",
            "multiple",
            effective[multiple_mask],
        )
    return pd.DataFrame.from_records(records)


def _frontier_frame(
    groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    metric_arrays: Mapping[str, np.ndarray],
    per_segment: pd.DataFrame,
    exposure: pd.DataFrame,
    hierarchy: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    hierarchy_by_group = {
        int(item["group_index"]): item for item in hierarchy
    }
    exposure_keys = [
        "config",
        "scenario",
        "outlier_dependence",
        "method",
    ]
    exposure_summary = (
        exposure.groupby(exposure_keys, sort=False)
        .agg(
            realized_outlier_reading_fraction=(
                "realized_outlier_reading_fraction",
                "mean",
            ),
            realized_outlier_location_fraction=(
                "any_outlier_location_fraction",
                "mean",
            ),
            majority_outlier_location_fraction=(
                "majority_outlier_location_fraction",
                "mean",
            ),
            all_replicates_outlier_location_fraction=(
                "all_replicates_outlier_location_fraction",
                "mean",
            ),
            effective_outlier_shift_p95=(
                "effective_outlier_shift_p95",
                "median",
            ),
        )
        .reset_index()
        if len(exposure)
        else pd.DataFrame()
    )
    records: list[dict[str, Any]] = []
    for group in groups:
        metadata = _group_metadata(group, designs, scenarios)
        full = metric_arrays["full_rmse"][group.index]
        non = metric_arrays["non_anchor_rmse"][group.index]
        fold_max = np.max(metric_arrays["fold_rmse"][group.index], axis=1)
        hierarchy_item = hierarchy_by_group[group.index]["hierarchical"]
        fold_failure_rates = np.mean(
            metric_arrays["fold_rmse"][group.index]
            >= PERSISTENT_FOLD_RMSE_THRESHOLD,
            axis=0,
        )
        persistent = bool(
            np.any(fold_failure_rates >= PERSISTENT_FOLD_FAILURE_RATE)
        )
        segment_rows = per_segment[
            per_segment["config"].eq(metadata["config"])
            & per_segment["method"].eq(metadata["method"])
            & per_segment["scenario"].eq(metadata["scenario"])
            & per_segment["outlier_dependence"].eq(
                metadata["outlier_dependence"]
            )
        ]
        segment_q50 = {
            str(segment): float(np.quantile(block["rmse"], 0.50))
            for segment, block in segment_rows.groupby("segment", sort=False)
        }
        record = {
            **metadata,
            "realizations": len(full),
            **{
                f"full_rmse_{key}": value
                for key, value in _quantiles(full).items()
            },
            **{
                f"non_anchor_rmse_{key}": value
                for key, value in _quantiles(non).items()
            },
            "probability_full_rmse_lt_4": float(np.mean(full < 4.0)),
            "probability_non_anchor_rmse_lt_4": float(np.mean(non < 4.0)),
            "hierarchical_full_q025": float(
                hierarchy_item["full_rmse"]["q025"]
            ),
            "hierarchical_full_q50": float(
                hierarchy_item["full_rmse"]["q50"]
            ),
            "hierarchical_full_q975": float(
                hierarchy_item["full_rmse"]["q975"]
            ),
            "hierarchical_non_anchor_q025": float(
                hierarchy_item["non_anchor_rmse"]["q025"]
            ),
            "hierarchical_non_anchor_q50": float(
                hierarchy_item["non_anchor_rmse"]["q50"]
            ),
            "hierarchical_non_anchor_q975": float(
                hierarchy_item["non_anchor_rmse"]["q975"]
            ),
            "fold_max_q50": float(np.quantile(fold_max, 0.50)),
            "fold_max_q95": float(np.quantile(fold_max, 0.95)),
            "p95_well_rmse_q50": float(
                np.quantile(metric_arrays["p95"][group.index], 0.50)
            ),
            "p95_well_rmse_q95": float(
                np.quantile(metric_arrays["p95"][group.index], 0.95)
            ),
            "max_well_rmse_q50": float(
                np.quantile(metric_arrays["max"][group.index], 0.50)
            ),
            "max_well_rmse_q95": float(
                np.quantile(metric_arrays["max"][group.index], 0.95)
            ),
            "wells_ge_40_q50": float(
                np.quantile(
                    metric_arrays["wells_ge_40"][group.index], 0.50
                )
            ),
            "wells_ge_40_q95": float(
                np.quantile(
                    metric_arrays["wells_ge_40"][group.index], 0.95
                )
            ),
            "wells_ge_40_max": int(
                np.max(metric_arrays["wells_ge_40"][group.index])
            ),
            "wells_ge_60_q50": float(
                np.quantile(
                    metric_arrays["wells_ge_60"][group.index], 0.50
                )
            ),
            "wells_ge_60_q95": float(
                np.quantile(
                    metric_arrays["wells_ge_60"][group.index], 0.95
                )
            ),
            "wells_ge_60_max": int(
                np.max(metric_arrays["wells_ge_60"][group.index])
            ),
            "worst10_sse_share_q50": float(
                np.quantile(
                    metric_arrays["worst10_sse_share"][group.index], 0.50
                )
            ),
            "worst10_sse_share_q95": float(
                np.quantile(
                    metric_arrays["worst10_sse_share"][group.index], 0.95
                )
            ),
            "pre_first_rmse_q50": segment_q50.get("pre_first", 0.0),
            "between_rmse_q50": segment_q50.get("between", 0.0),
            "post_last_rmse_q50": segment_q50.get("post_last", 0.0),
            "persistent_fold_failure": persistent,
            "robust_lt_4": bool(
                np.quantile(full, 0.95) < 4.0
                and np.quantile(non, 0.95) < 4.0
                and hierarchy_item["full_rmse"]["q975"] < 4.0
                and hierarchy_item["non_anchor_rmse"]["q975"] < 4.0
                and not persistent
                and np.max(
                    metric_arrays["wells_ge_40"][group.index]
                )
                == 0
                and np.max(
                    metric_arrays["wells_ge_60"][group.index]
                )
                == 0
            ),
        }
        if len(exposure_summary):
            match = exposure_summary[
                exposure_summary["config"].eq(metadata["config"])
                & exposure_summary["scenario"].eq(metadata["scenario"])
                & exposure_summary["outlier_dependence"].eq(
                    metadata["outlier_dependence"]
                )
                & exposure_summary["method"].eq(OPERATIONAL_METHOD)
            ]
            if len(match):
                record.update(
                    {
                        column: float(match.iloc[0][column])
                        for column in (
                            "realized_outlier_reading_fraction",
                            "realized_outlier_location_fraction",
                            "majority_outlier_location_fraction",
                            "all_replicates_outlier_location_fraction",
                            "effective_outlier_shift_p95",
                        )
                    }
                )
            else:
                record.update(
                    {
                        "realized_outlier_reading_fraction": 0.0,
                        "realized_outlier_location_fraction": 0.0,
                        "majority_outlier_location_fraction": 0.0,
                        "all_replicates_outlier_location_fraction": 0.0,
                        "effective_outlier_shift_p95": 0.0,
                    }
                )
        records.append(record)
    curve = pd.DataFrame.from_records(records)
    for column in (
        "independent_reacquisition_pass",
        "shared_fault_pass",
        "unconditional_operational_pass",
        "control_harm_pass",
        "gate_available",
    ):
        curve[column] = False

    scenario_names = {scenario.name for scenario in scenarios}
    design_names = {design.name for design in designs}
    control_names = {
        "exact_clean",
        "gaussian_1_clean",
        "moderate_background",
        "moderate_m5_control",
    }
    gate_available = bool(
        O5_SCENARIO in scenario_names
        and control_names.issubset(scenario_names)
        and "near_k6_r1" in design_names
    )

    def row(
        config: str, scenario: str, dependence: str
    ) -> pd.Series | None:
        match = curve[
            curve["config"].eq(config)
            & curve["scenario"].eq(scenario)
            & curve["outlier_dependence"].eq(dependence)
            & curve["method"].eq(OPERATIONAL_METHOD)
        ]
        return match.iloc[0] if len(match) == 1 else None

    for design in designs:
        if design.replicates == 1:
            o5_single = row(design.name, O5_SCENARIO, DEPENDENCE_SINGLE)
            independent_pass = bool(
                o5_single is not None and o5_single["robust_lt_4"]
            )
            shared_pass = independent_pass
        else:
            o5_independent = row(
                design.name,
                O5_SCENARIO,
                DEPENDENCE_MEASUREMENT_INDEPENDENT,
            )
            o5_shared = row(
                design.name, O5_SCENARIO, DEPENDENCE_ANCHOR_SHARED
            )
            independent_pass = bool(
                o5_independent is not None
                and o5_independent["robust_lt_4"]
            )
            shared_pass = bool(
                o5_shared is not None and o5_shared["robust_lt_4"]
            )

        control_harm = gate_available
        comparator_name = (
            "near_k8_r1"
            if design.name == "near_k8_r2_mean"
            else "near_k6_r1"
        )
        for scenario_name in sorted(control_names):
            design_dependencies = (
                (DEPENDENCE_SINGLE,)
                if design.replicates == 1
                else (
                    (DEPENDENCE_CLEAN,)
                    if scenario_name != "moderate_m5_control"
                    else (
                        DEPENDENCE_MEASUREMENT_INDEPENDENT,
                        DEPENDENCE_ANCHOR_SHARED,
                    )
                )
            )
            comparator = row(
                comparator_name, scenario_name, DEPENDENCE_SINGLE
            )
            for dependence in design_dependencies:
                candidate = row(design.name, scenario_name, dependence)
                control_harm = bool(
                    control_harm
                    and candidate is not None
                    and comparator is not None
                    and candidate["robust_lt_4"]
                    and candidate["full_rmse_q95"]
                    - comparator["full_rmse_q95"]
                    <= 0.05
                    and candidate["non_anchor_rmse_q95"]
                    - comparator["non_anchor_rmse_q95"]
                    <= 0.05
                )
        unconditional = bool(
            gate_available
            and independent_pass
            and shared_pass
            and control_harm
        )
        design_rows = curve[
            curve["config"].eq(design.name)
            & curve["method"].eq(OPERATIONAL_METHOD)
        ].index
        curve.loc[design_rows, "independent_reacquisition_pass"] = (
            independent_pass
        )
        curve.loc[design_rows, "shared_fault_pass"] = shared_pass
        curve.loc[design_rows, "control_harm_pass"] = control_harm
        curve.loc[design_rows, "unconditional_operational_pass"] = (
            unconditional
        )
        curve.loc[design_rows, "gate_available"] = gate_available
    return curve


def _paired_deltas_frame(
    groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    metric_arrays: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    lookup = {
        (
            designs[group.design_index].name,
            scenarios[group.scenario_index].name,
            group.dependence,
            group.method,
        ): group
        for group in groups
    }
    records: list[dict[str, Any]] = []

    def append(
        group: EvaluationGroup,
        reference: EvaluationGroup,
        relation: str,
    ) -> None:
        metadata = _group_metadata(group, designs, scenarios)
        reference_metadata = _group_metadata(
            reference, designs, scenarios
        )
        full_delta = (
            metric_arrays["full_rmse"][group.index]
            - metric_arrays["full_rmse"][reference.index]
        )
        non_delta = (
            metric_arrays["non_anchor_rmse"][group.index]
            - metric_arrays["non_anchor_rmse"][reference.index]
        )
        records.append(
            {
                **metadata,
                "comparison": relation,
                "reference_config": reference_metadata["config"],
                "reference_method": reference_metadata["method"],
                "reference_outlier_dependence": reference_metadata[
                    "outlier_dependence"
                ],
                "full_delta_mean": float(np.mean(full_delta)),
                "full_delta_q05": float(np.quantile(full_delta, 0.05)),
                "full_delta_q50": float(np.quantile(full_delta, 0.50)),
                "full_delta_q95": float(np.quantile(full_delta, 0.95)),
                "non_anchor_delta_mean": float(np.mean(non_delta)),
                "non_anchor_delta_q05": float(
                    np.quantile(non_delta, 0.05)
                ),
                "non_anchor_delta_q50": float(
                    np.quantile(non_delta, 0.50)
                ),
                "non_anchor_delta_q95": float(
                    np.quantile(non_delta, 0.95)
                ),
            }
        )

    for group in groups:
        design = designs[group.design_index]
        scenario = scenarios[group.scenario_index]
        baseline = lookup.get(
            (
                "near_k6_r1",
                scenario.name,
                DEPENDENCE_SINGLE,
                OPERATIONAL_METHOD,
            )
        )
        if baseline is not None and baseline.index != group.index:
            append(group, baseline, "vs_near_k6_r1")
        if design.replicates > 1:
            spatial = lookup.get(
                (
                    f"near_k{design.k}_r1",
                    scenario.name,
                    DEPENDENCE_SINGLE,
                    OPERATIONAL_METHOD,
                )
            )
            if spatial is not None:
                append(group, spatial, "vs_same_spatial_r1")
        if scenario.name == O5_SCENARIO:
            same_observed = lookup.get(
                (
                    design.name,
                    scenario.name,
                    group.dependence,
                    OPERATIONAL_METHOD,
                )
            )
            if (
                same_observed is not None
                and group.method != OPERATIONAL_METHOD
            ):
                append(group, same_observed, "oracle_minus_observed")
            if group.method == OPERATIONAL_METHOD:
                for oracle_method in METHODS[1:]:
                    oracle = lookup.get(
                        (
                            design.name,
                            scenario.name,
                            group.dependence,
                            oracle_method,
                        )
                    )
                    if oracle is not None:
                        append(
                            group,
                            oracle,
                            f"observed_minus_{oracle_method}",
                        )
    return pd.DataFrame.from_records(records)


def _worst_wells_frame(
    groups: Sequence[EvaluationGroup],
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    seeds: Sequence[int],
    well_ids: Sequence[str],
    folds: np.ndarray,
    rows: np.ndarray,
    full_sse: np.ndarray,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group in groups:
        metadata = _group_metadata(group, designs, scenarios)
        rmse = np.sqrt(full_sse[group.index] / rows[None, :])
        flat = rmse.reshape(-1)
        selected = np.argsort(flat, kind="stable")[-top_n:][::-1]
        for rank, flat_index in enumerate(selected, start=1):
            seed_position, well_position = np.unravel_index(
                int(flat_index), rmse.shape
            )
            records.append(
                {
                    **metadata,
                    "rank": rank,
                    "realization_seed": int(seeds[seed_position]),
                    "well_id": well_ids[well_position],
                    "fold": int(folds[well_position]),
                    "rows": int(rows[well_position]),
                    "sse": float(
                        full_sse[
                            group.index, seed_position, well_position
                        ]
                    ),
                    "rmse": float(rmse[seed_position, well_position]),
                }
            )
    return pd.DataFrame.from_records(records)


def _regression_reproduction(
    *,
    designs: Sequence[AcquisitionDesign],
    scenarios: Sequence[ErrorScenario],
    groups: Sequence[EvaluationGroup],
    seeds: Sequence[int],
    well_ids: Sequence[str],
    full_sse: np.ndarray,
    non_sse: np.ndarray,
    rows: np.ndarray,
    non_rows: np.ndarray,
    prior_qc_artifact: Path,
    terminal_frontier: Path,
    maximum_replicate0_latent_delta: float,
) -> dict[str, Any]:
    design_lookup = {design.name: index for index, design in enumerate(designs)}
    scenario_lookup = {
        scenario.name: index for index, scenario in enumerate(scenarios)
    }
    group_lookup = {
        (
            designs[group.design_index].name,
            scenarios[group.scenario_index].name,
            group.dependence,
            group.method,
        ): group.index
        for group in groups
    }
    seed_lookup = {int(seed): index for index, seed in enumerate(seeds)}
    well_lookup = {str(well): index for index, well in enumerate(well_ids)}
    scenario_map = {
        "exact_clean": "exact_clean",
        "gaussian_1_clean": "gaussian_1_clean",
        "moderate_background": "moderate_background_no_outlier",
        "moderate_m5_control": "moderate_m5_control",
        O5_SCENARIO: "moderate_background_p10_m10",
        "S0_severe_background": "severe_background_no_outlier",
        "S1_severe_p10_m10": "severe",
    }
    metric_max = 0.0
    per_well_sse_max = 0.0
    per_well_non_sse_max = 0.0
    compared_metric_groups = 0
    compared_well_groups = 0
    oracle_metric_max = 0.0
    oracle_well_sse_max = 0.0
    oracle_well_non_sse_max = 0.0
    oracle_compared_groups = 0
    full_coverage = len(well_ids) == 773

    prior_root = Path(prior_qc_artifact).resolve()
    if "near_k6_r1" in design_lookup:
        prior_seed = pd.read_csv(prior_root / "per_seed_metrics.csv")
        for current_name, prior_name in (
            scenario_map.items() if full_coverage else ()
        ):
            group_index = group_lookup.get(
                (
                    "near_k6_r1",
                    current_name,
                    DEPENDENCE_SINGLE,
                    OPERATIONAL_METHOD,
                )
            )
            if group_index is None:
                continue
            prior = prior_seed[
                prior_seed["config"].eq("near_k6")
                & prior_seed["method"].eq("no_qc")
                & prior_seed["scenario"].eq(prior_name)
                & prior_seed["realization_seed"].isin(seeds)
            ].sort_values("realization_seed", kind="stable")
            if len(prior) != len(seeds):
                raise ValueError(
                    f"missing prior no-QC metric regression for {current_name}"
                )
            current_full = np.sqrt(
                full_sse[group_index].sum(axis=1) / rows.sum()
            )
            current_non = np.sqrt(
                non_sse[group_index].sum(axis=1)
                / non_rows[design_lookup["near_k6_r1"]].sum()
            )
            ordered_current = np.asarray(
                [
                    current_full[seed_lookup[int(seed)]]
                    for seed in prior["realization_seed"]
                ]
            )
            ordered_current_non = np.asarray(
                [
                    current_non[seed_lookup[int(seed)]]
                    for seed in prior["realization_seed"]
                ]
            )
            metric_max = max(
                metric_max,
                float(
                    np.max(
                        np.abs(
                            ordered_current
                            - prior["full_rmse"].to_numpy(
                                dtype=np.float64
                            )
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            ordered_current_non
                            - prior["non_anchor_rmse"].to_numpy(
                                dtype=np.float64
                            )
                        )
                    )
                ),
            )
            compared_metric_groups += len(prior)

        oracle_index = group_lookup.get(
            (
                "near_k6_r1",
                O5_SCENARIO,
                DEPENDENCE_SINGLE,
                "oracle_drop_outlier_replicates",
            )
        )
        if oracle_index is not None and full_coverage:
            prior = prior_seed[
                prior_seed["config"].eq("near_k6")
                & prior_seed["method"].eq("oracle_indicator_reject")
                & prior_seed["scenario"].eq(
                    "moderate_background_p10_m10"
                )
                & prior_seed["realization_seed"].isin(seeds)
            ].sort_values("realization_seed", kind="stable")
            current_full = np.sqrt(
                full_sse[oracle_index].sum(axis=1) / rows.sum()
            )
            current_non = np.sqrt(
                non_sse[oracle_index].sum(axis=1)
                / non_rows[design_lookup["near_k6_r1"]].sum()
            )
            oracle_metric_max = max(
                float(
                    np.max(
                        np.abs(
                            current_full
                            - prior["full_rmse"].to_numpy(
                                dtype=np.float64
                            )
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

        needed_prior = set(scenario_map.values())
        if oracle_index is not None:
            needed_prior.add("moderate_background_p10_m10")
        usecols = [
            "config",
            "method",
            "scenario",
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
            subset = chunk[
                chunk["config"].eq("near_k6")
                & chunk["scenario"].isin(needed_prior)
                & chunk["realization_seed"].isin(seeds)
                & chunk["well_id"].isin(well_ids)
                & chunk["method"].isin(
                    ["no_qc", "oracle_indicator_reject"]
                )
            ]
            for row in subset.itertuples(index=False):
                current_name = next(
                    (
                        key
                        for key, value in scenario_map.items()
                        if value == row.scenario
                    ),
                    None,
                )
                if row.method == "oracle_indicator_reject":
                    if row.scenario != "moderate_background_p10_m10":
                        continue
                    current_name = O5_SCENARIO
                    method = "oracle_drop_outlier_replicates"
                else:
                    method = OPERATIONAL_METHOD
                if current_name is None:
                    continue
                group_index = group_lookup.get(
                    (
                        "near_k6_r1",
                        current_name,
                        DEPENDENCE_SINGLE,
                        method,
                    )
                )
                if group_index is None:
                    continue
                seed_position = seed_lookup[int(row.realization_seed)]
                well_position = well_lookup[str(row.well_id)]
                sse_delta = abs(
                    float(
                        full_sse[
                            group_index, seed_position, well_position
                        ]
                    )
                    - float(row.sse)
                )
                non_delta = abs(
                    float(
                        non_sse[
                            group_index, seed_position, well_position
                        ]
                    )
                    - float(row.non_anchor_sse)
                )
                if method == OPERATIONAL_METHOD:
                    per_well_sse_max = max(per_well_sse_max, sse_delta)
                    per_well_non_sse_max = max(
                        per_well_non_sse_max, non_delta
                    )
                    compared_well_groups += 1
                else:
                    oracle_well_sse_max = max(
                        oracle_well_sse_max, sse_delta
                    )
                    oracle_well_non_sse_max = max(
                        oracle_well_non_sse_max, non_delta
                    )
                    oracle_compared_groups += 1

    k7_exact_delta: float | None = None
    k7_group = group_lookup.get(
        (
            "near_k7_r1",
            "exact_clean",
            DEPENDENCE_SINGLE,
            OPERATIONAL_METHOD,
        )
    )
    if k7_group is not None and full_coverage:
        frontier = pd.read_csv(
            Path(terminal_frontier).resolve() / "information_frontier.csv"
        )
        prior = frontier[
            frontier["policy"].eq("near_terminal")
            & frontier["mode"].eq("acausal")
            & frontier["method"].eq("piecewise_residual")
            & frontier["k"].eq(7)
        ]
        if len(prior) != 1:
            raise ValueError("missing unique near-terminal k7 terminal metric")
        current = float(
            np.sqrt(full_sse[k7_group, 0].sum() / rows.sum())
        )
        k7_exact_delta = abs(
            current - float(prior.iloc[0]["pooled_rmse"])
        )

    passed = bool(
        maximum_replicate0_latent_delta == 0.0
        and metric_max <= METRIC_TOLERANCE
        and per_well_sse_max <= PER_WELL_SSE_TOLERANCE
        and per_well_non_sse_max <= PER_WELL_SSE_TOLERANCE
        and oracle_metric_max <= METRIC_TOLERANCE
        and oracle_well_sse_max <= PER_WELL_SSE_TOLERANCE
        and oracle_well_non_sse_max <= PER_WELL_SSE_TOLERANCE
        and (k7_exact_delta is None or k7_exact_delta <= METRIC_TOLERANCE)
    )
    return {
        "passed": passed,
        "replicate0_legacy_latent_max_abs_delta": (
            maximum_replicate0_latent_delta
        ),
        "no_qc_metric_max_abs_delta": metric_max,
        "no_qc_per_well_sse_max_abs_delta": per_well_sse_max,
        "no_qc_per_well_non_anchor_sse_max_abs_delta": (
            per_well_non_sse_max
        ),
        "no_qc_metric_groups_compared": compared_metric_groups,
        "no_qc_per_well_groups_compared": compared_well_groups,
        "indicator_oracle_metric_max_abs_delta": oracle_metric_max,
        "indicator_oracle_per_well_sse_max_abs_delta": (
            oracle_well_sse_max
        ),
        "indicator_oracle_per_well_non_anchor_sse_max_abs_delta": (
            oracle_well_non_sse_max
        ),
        "indicator_oracle_per_well_groups_compared": (
            oracle_compared_groups
        ),
        "near_k7_exact_full_rmse_delta": k7_exact_delta,
    }


def run_anchor_acquisition_redundancy(
    *,
    data_root: Path,
    fold_config: Path,
    blend_oof: Path,
    terminal_frontier: Path,
    prior_qc_artifact: Path,
    output_root: Path,
    run_id: str,
    design_names: Sequence[str] | None = None,
    scenario_names: Sequence[str] | None = None,
    realization_seeds: Sequence[int] = DEFAULT_NOISE_SEEDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    selection_seed: int = 20260711,
    max_wells_per_fold: int | None = None,
) -> dict[str, Any]:
    """Run the complete locked acquisition-redundancy comparison."""
    designs = select_designs(design_names)
    scenarios = select_redundancy_scenarios(scenario_names)
    seeds = tuple(_validate_seed(seed) for seed in realization_seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or isinstance(bootstrap_samples, (bool, np.bool_))
        or int(bootstrap_samples) < 1
        or (
            max_wells_per_fold is not None
            and (
                isinstance(max_wells_per_fold, (bool, np.bool_))
                or int(max_wells_per_fold) < 1
            )
        )
    ):
        raise ValueError("invalid seed/bootstrap/smoke configuration")
    _validate_seed(bootstrap_seed)
    _validate_seed(selection_seed)

    data = Path(data_root).resolve()
    run_root = safe_output_directory(output_root, run_id, data)
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
    selected = _selected_wells(
        mapping,
        max_wells_per_fold=max_wells_per_fold,
        seed=selection_seed,
    )
    contexts = _build_noise_contexts(baseline, data, mapping, selected)
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
    strict_full = max_wells_per_fold is None
    if strict_full and (
        well_count != 773 or int(rows.sum()) != 3_783_989
    ):
        raise ValueError("strict full redundancy run coverage mismatch")

    groups = build_evaluation_groups(designs, scenarios)
    observed_groups = tuple(
        group for group in groups if group.method == OPERATIONAL_METHOD
    )
    group_lookup = {
        (
            group.design_index,
            group.scenario_index,
            group.dependence,
            group.method,
        ): group.index
        for group in groups
    }
    group_count = len(groups)
    seed_count = len(seeds)
    shape = (group_count, seed_count, well_count)
    full_sse = np.zeros(shape, dtype=np.float64)
    non_sse = np.zeros(shape, dtype=np.float64)
    contaminated_location_count = np.zeros(shape, dtype=np.uint8)
    non_rows = np.zeros((len(designs), well_count), dtype=np.int64)
    fold_rows = np.zeros(5, dtype=np.int64)
    fold_non_rows = np.zeros((len(designs), 5), dtype=np.int64)
    fold_sse = np.zeros((group_count, seed_count, 5), dtype=np.float64)
    fold_non_sse = np.zeros_like(fold_sse)
    segment_rows = np.zeros((len(designs), 3), dtype=np.int64)
    segment_sse = np.zeros(
        (group_count, seed_count, 3), dtype=np.float64
    )
    distance_rows = np.zeros((len(designs), 10), dtype=np.int64)
    distance_non_rows = np.zeros_like(distance_rows)
    distance_sse = np.zeros(
        (group_count, seed_count, 10), dtype=np.float64
    )
    distance_non_sse = np.zeros_like(distance_sse)
    exposure_counters = {
        "outlier_readings": np.zeros((group_count, seed_count), dtype=np.int64),
        "any_outlier_locations": np.zeros(
            (group_count, seed_count), dtype=np.int64
        ),
        "majority_outlier_locations": np.zeros(
            (group_count, seed_count), dtype=np.int64
        ),
        "all_outlier_locations": np.zeros(
            (group_count, seed_count), dtype=np.int64
        ),
        "wells_any_contamination": np.zeros(
            (group_count, seed_count), dtype=np.int64
        ),
        "wells_multiple_contamination": np.zeros(
            (group_count, seed_count), dtype=np.int64
        ),
        "cancellation_eligible": np.zeros(
            (group_count, seed_count), dtype=np.int64
        ),
        "sign_cancellations": np.zeros(
            (group_count, seed_count), dtype=np.int64
        ),
        "aggregate_error_count": np.zeros(
            (group_count, seed_count), dtype=np.int64
        ),
        "aggregate_error_abs_sum": np.zeros(
            (group_count, seed_count), dtype=np.float64
        ),
        "aggregate_error_square_sum": np.zeros(
            (group_count, seed_count), dtype=np.float64
        ),
        "aggregate_error_max_abs": np.zeros(
            (group_count, seed_count), dtype=np.float64
        ),
        "idiosyncratic_error_abs_sum": np.zeros(
            (group_count, seed_count), dtype=np.float64
        ),
        "idiosyncratic_error_square_sum": np.zeros(
            (group_count, seed_count), dtype=np.float64
        ),
    }
    effective_values: dict[int, np.ndarray] = {
        group.index: np.empty(
            (
                seed_count,
                well_count,
                designs[group.design_index].k,
            ),
            dtype=np.float32,
        )
        for group in observed_groups
        if scenarios[group.scenario_index].outlier_probability > 0.0
    }
    contaminated_locations: dict[int, np.ndarray] = {
        group.index: np.empty(
            (
                seed_count,
                well_count,
                designs[group.design_index].k,
            ),
            dtype=bool,
        )
        for group in observed_groups
        if scenarios[group.scenario_index].outlier_probability > 0.0
    }

    maximum_replicate0_latent_delta = 0.0
    maximum_shared_physical_replicate_latent_delta = 0.0
    maximum_common_bias_cross_design_delta = 0.0
    maximum_anchor_observation_delta = 0.0
    maximum_exact_row_delta = 0.0
    placement_digest = hashlib.sha256()
    latent_by_physical_replicate: dict[
        tuple[str, int, int],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    common_bias_by_well: dict[str, np.ndarray] = {}

    per_well_path = run_root / "per_well_seed_metrics.csv.gz"
    with _DeterministicGzipCsvWriter(per_well_path) as writer:
        for well_position, well_id in enumerate(well_ids):
            context = contexts[well_id]
            fold_rows[context.fold] += len(context.row_index)
            boundary = _boundary_md(context)  # type: ignore[arg-type]
            for design_index, design in enumerate(designs):
                selection = select_terminal_anchors(
                    context.prediction_md,
                    boundary,
                    policy="near_terminal",
                    k=design.k,
                )
                _validate_selection(design, selection)
                physical_rows = context.row_index[selection.local_indices]
                true_anchors = context.y_true[selection.local_indices]
                latents = stable_replicate_latents(
                    seeds,
                    well_id,
                    physical_rows,
                    design.replicates,
                )
                legacy = stable_anchor_latents(seeds, well_id, physical_rows)
                maximum_replicate0_latent_delta = max(
                    maximum_replicate0_latent_delta,
                    float(
                        np.max(
                            np.abs(
                                latents.independent[:, :, 0]
                                - legacy.independent
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                latents.outlier_indicator_uniform[:, :, 0]
                                - legacy.outlier_indicator_uniform
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                latents.outlier_sign[:, :, 0]
                                - legacy.outlier_sign
                            )
                        )
                    ),
                )
                if well_id in common_bias_by_well:
                    maximum_common_bias_cross_design_delta = max(
                        maximum_common_bias_cross_design_delta,
                        float(
                            np.max(
                                np.abs(
                                    common_bias_by_well[well_id]
                                    - latents.common_bias
                                )
                            )
                        ),
                    )
                else:
                    common_bias_by_well[well_id] = (
                        latents.common_bias.copy()
                    )
                for location, physical_row in enumerate(physical_rows):
                    for replicate in range(design.replicates):
                        key = (well_id, int(physical_row), replicate)
                        value = (
                            latents.independent[:, location, replicate],
                            latents.outlier_indicator_uniform[
                                :, location, replicate
                            ],
                            latents.outlier_sign[:, location, replicate],
                        )
                        if key in latent_by_physical_replicate:
                            prior = latent_by_physical_replicate[key]
                            maximum_shared_physical_replicate_latent_delta = max(
                                maximum_shared_physical_replicate_latent_delta,
                                *(
                                    float(np.max(np.abs(left - right)))
                                    for left, right in zip(value, prior)
                                ),
                            )
                        else:
                            latent_by_physical_replicate[key] = tuple(
                                np.asarray(item).copy() for item in value
                            )  # type: ignore[assignment]

                masks, physical_anchor, _, _ = _metric_masks(
                    context, selection
                )
                full_mask = (1 << design.k) - 1
                full_coefficients, exact_prediction, weights = (
                    _coefficients_for_subset(
                        context,
                        selection,
                        true_anchors,
                        full_mask,
                        masks,
                    )
                )
                exact_reference = condition_noisy_piecewise(
                    context.base_prediction,
                    true_anchors,
                    context.normalized_md,
                    selection,
                )
                maximum_exact_row_delta = max(
                    maximum_exact_row_delta,
                    float(np.max(np.abs(exact_reference - exact_prediction))),
                )
                mask_counts = full_coefficients[3]
                non_rows[design_index, well_position] = int(mask_counts[1])
                fold_non_rows[design_index, context.fold] += int(
                    mask_counts[1]
                )
                segment_rows[design_index] += mask_counts[2:5]
                distance_rows[design_index] += mask_counts[5:15]
                distance_non_rows[design_index] += mask_counts[15:25]

                for scenario_index, scenario in enumerate(scenarios):
                    for dependence in _dependences_for(design, scenario):
                        outcome = scenario_replicate_errors(
                            scenario,
                            latents,
                            dependence,
                            design.aggregation,
                        )
                        aggregate_error = outcome["aggregate_error"]
                        observed_index = group_lookup[
                            (
                                design_index,
                                scenario_index,
                                dependence,
                                OPERATIONAL_METHOD,
                            )
                        ]
                        result = _evaluate_fixed_subset(
                            full_coefficients, aggregate_error
                        )
                        full_sse[
                            observed_index, :, well_position
                        ] = result[:, 0]
                        non_sse[
                            observed_index, :, well_position
                        ] = result[:, 1]
                        fold_sse[
                            observed_index, :, context.fold
                        ] += result[:, 0]
                        fold_non_sse[
                            observed_index, :, context.fold
                        ] += result[:, 1]
                        segment_sse[observed_index] += result[:, 2:5]
                        distance_sse[observed_index] += result[:, 5:15]
                        distance_non_sse[observed_index] += result[:, 15:25]

                        flags = outcome["flags"]
                        flags_per_location = flags.sum(
                            axis=2, dtype=np.int8
                        )
                        location_contaminated = flags_per_location > 0
                        location_count = location_contaminated.sum(
                            axis=1, dtype=np.int8
                        )
                        contaminated_location_count[
                            observed_index, :, well_position
                        ] = location_count
                        exposure_counters["outlier_readings"][
                            observed_index
                        ] += flags.sum(axis=(1, 2), dtype=np.int64)
                        exposure_counters["any_outlier_locations"][
                            observed_index
                        ] += location_contaminated.sum(
                            axis=1, dtype=np.int64
                        )
                        exposure_counters["majority_outlier_locations"][
                            observed_index
                        ] += (
                            flags_per_location > design.replicates / 2.0
                        ).sum(axis=1, dtype=np.int64)
                        exposure_counters["all_outlier_locations"][
                            observed_index
                        ] += (
                            flags_per_location == design.replicates
                        ).sum(axis=1, dtype=np.int64)
                        exposure_counters["wells_any_contamination"][
                            observed_index
                        ] += location_count > 0
                        exposure_counters[
                            "wells_multiple_contamination"
                        ][observed_index] += location_count >= 2
                        signed_outliers = (
                            flags.astype(np.int8)
                            * outcome["signs"].astype(np.int8)
                        ).sum(axis=2)
                        eligible = flags_per_location >= 2
                        exposure_counters["cancellation_eligible"][
                            observed_index
                        ] += eligible.sum(axis=1, dtype=np.int64)
                        exposure_counters["sign_cancellations"][
                            observed_index
                        ] += (
                            eligible & (signed_outliers == 0)
                        ).sum(axis=1, dtype=np.int64)
                        idiosyncratic = (
                            aggregate_error
                            - scenario.tau
                            * latents.common_bias[:, None]
                        )
                        exposure_counters["aggregate_error_count"][
                            observed_index
                        ] += design.k
                        exposure_counters["aggregate_error_abs_sum"][
                            observed_index
                        ] += np.abs(aggregate_error).sum(axis=1)
                        exposure_counters["aggregate_error_square_sum"][
                            observed_index
                        ] += np.square(aggregate_error).sum(axis=1)
                        exposure_counters["aggregate_error_max_abs"][
                            observed_index
                        ] = np.maximum(
                            exposure_counters["aggregate_error_max_abs"][
                                observed_index
                            ],
                            np.max(np.abs(aggregate_error), axis=1),
                        )
                        exposure_counters["idiosyncratic_error_abs_sum"][
                            observed_index
                        ] += np.abs(idiosyncratic).sum(axis=1)
                        exposure_counters[
                            "idiosyncratic_error_square_sum"
                        ][observed_index] += np.square(idiosyncratic).sum(
                            axis=1
                        )
                        if observed_index in effective_values:
                            effective_values[observed_index][
                                :, well_position, :
                            ] = outcome["effective_outlier_shift"].astype(
                                np.float32
                            )
                            contaminated_locations[observed_index][
                                :, well_position, :
                            ] = location_contaminated

                        predicted_anchors = (
                            exact_prediction[
                                selection.local_indices
                            ][None, :]
                            + aggregate_error
                            @ weights[selection.local_indices].T
                        )
                        expected_anchors = true_anchors[None, :] + aggregate_error
                        maximum_anchor_observation_delta = max(
                            maximum_anchor_observation_delta,
                            float(
                                np.max(
                                    np.abs(
                                        predicted_anchors
                                        - expected_anchors
                                    )
                                )
                            ),
                        )

                        if scenario.name == O5_SCENARIO:
                            oracle_drop_index = group_lookup[
                                (
                                    design_index,
                                    scenario_index,
                                    dependence,
                                    "oracle_drop_outlier_replicates",
                                )
                            ]
                            retained_locations = outcome[
                                "oracle_retained_locations"
                            ]
                            retained_masks = _retained_mask_values(
                                retained_locations
                            )
                            oracle_errors = outcome[
                                "oracle_clean_replicate_error"
                            ]
                            oracle_result = _evaluate_variable_subsets(
                                context,
                                selection,
                                true_anchors,
                                retained_masks,
                                oracle_errors,
                                masks,
                                full_coefficients,
                            )
                            full_sse[
                                oracle_drop_index, :, well_position
                            ] = oracle_result[:, 0]
                            non_sse[
                                oracle_drop_index, :, well_position
                            ] = oracle_result[:, 1]
                            fold_sse[
                                oracle_drop_index, :, context.fold
                            ] += oracle_result[:, 0]
                            fold_non_sse[
                                oracle_drop_index, :, context.fold
                            ] += oracle_result[:, 1]
                            segment_sse[
                                oracle_drop_index
                            ] += oracle_result[:, 2:5]
                            distance_sse[
                                oracle_drop_index
                            ] += oracle_result[:, 5:15]
                            distance_non_sse[
                                oracle_drop_index
                            ] += oracle_result[:, 15:25]
                            contaminated_location_count[
                                oracle_drop_index, :, well_position
                            ] = location_count

                            clean_index = group_lookup[
                                (
                                    design_index,
                                    scenario_index,
                                    dependence,
                                    "oracle_clean_values",
                                )
                            ]
                            clean_result = _evaluate_fixed_subset(
                                full_coefficients,
                                np.zeros_like(aggregate_error),
                            )
                            full_sse[
                                clean_index, :, well_position
                            ] = clean_result[:, 0]
                            non_sse[
                                clean_index, :, well_position
                            ] = clean_result[:, 1]
                            fold_sse[
                                clean_index, :, context.fold
                            ] += clean_result[:, 0]
                            fold_non_sse[
                                clean_index, :, context.fold
                            ] += clean_result[:, 1]
                            segment_sse[
                                clean_index
                            ] += clean_result[:, 2:5]
                            distance_sse[
                                clean_index
                            ] += clean_result[:, 5:15]
                            distance_non_sse[
                                clean_index
                            ] += clean_result[:, 15:25]
                            contaminated_location_count[
                                clean_index, :, well_position
                            ] = location_count

                placement_digest.update(well_id.encode("ascii"))
                placement_digest.update(design.name.encode("ascii"))
                placement_digest.update(
                    np.ascontiguousarray(physical_rows).view(np.uint8)
                )

            group_design = np.asarray(
                [group.design_index for group in groups], dtype=np.int64
            )
            group_config = np.asarray(
                [designs[group.design_index].name for group in groups],
                dtype=object,
            )
            group_method = np.asarray(
                [group.method for group in groups], dtype=object
            )
            group_scenario = np.asarray(
                [scenarios[group.scenario_index].name for group in groups],
                dtype=object,
            )
            group_dependence = np.asarray(
                [group.dependence for group in groups], dtype=object
            )
            repeated_group = np.repeat(np.arange(group_count), seed_count)
            repeated_seed = np.tile(np.arange(seed_count), group_count)
            sse_values = full_sse[:, :, well_position].reshape(-1)
            non_sse_values = non_sse[:, :, well_position].reshape(-1)
            group_rows = rows[well_position]
            group_non_rows = non_rows[
                group_design, well_position
            ]
            frame = pd.DataFrame(
                {
                    "config": group_config[repeated_group],
                    "method": group_method[repeated_group],
                    "scenario": group_scenario[repeated_group],
                    "outlier_dependence": group_dependence[
                        repeated_group
                    ],
                    "realization_seed": np.asarray(
                        seeds, dtype=np.int64
                    )[repeated_seed],
                    "well_id": well_id,
                    "fold": context.fold,
                    "rows": group_rows,
                    "non_anchor_rows": group_non_rows[repeated_group],
                    "sse": sse_values,
                    "non_anchor_sse": non_sse_values,
                    "rmse": np.sqrt(sse_values / group_rows),
                    "non_anchor_rmse": np.sqrt(
                        non_sse_values / group_non_rows[repeated_group]
                    ),
                    "physical_anchor_count": np.asarray(
                        [design.k for design in designs]
                    )[group_design[repeated_group]],
                    "total_readings": np.asarray(
                        [design.total_readings for design in designs]
                    )[group_design[repeated_group]],
                    "contaminated_location_count": (
                        contaminated_location_count[
                            :, :, well_position
                        ].reshape(-1)
                    ),
                }
            )
            writer.write(frame)
            if (well_position + 1) % 20 == 0 or well_position + 1 == well_count:
                print(
                    "anchor-acquisition-redundancy "
                    f"wells={well_position + 1}/{well_count}",
                    flush=True,
                )

    if (
        maximum_replicate0_latent_delta != 0.0
        or maximum_shared_physical_replicate_latent_delta != 0.0
        or maximum_common_bias_cross_design_delta != 0.0
        or maximum_anchor_observation_delta != 0.0
        or maximum_exact_row_delta > ROWWISE_TOLERANCE
    ):
        raise ValueError("locked latent/reconstruction invariant failed")

    regression = _regression_reproduction(
        designs=designs,
        scenarios=scenarios,
        groups=groups,
        seeds=seeds,
        well_ids=well_ids,
        full_sse=full_sse,
        non_sse=non_sse,
        rows=rows,
        non_rows=non_rows,
        prior_qc_artifact=prior_qc_artifact,
        terminal_frontier=terminal_frontier,
        maximum_replicate0_latent_delta=maximum_replicate0_latent_delta,
    )
    if strict_full and not regression["passed"]:
        raise ValueError("redundancy regression reproduction failed")

    per_seed, metric_arrays = _per_seed_frame(
        groups,
        designs,
        scenarios,
        seeds,
        full_sse,
        non_sse,
        rows,
        non_rows,
        folds,
        fold_sse,
        fold_non_sse,
        fold_rows,
        fold_non_rows,
    )
    per_fold = _per_fold_frame(
        groups,
        designs,
        scenarios,
        seeds,
        fold_sse,
        fold_non_sse,
        fold_rows,
        fold_non_rows,
    )
    per_segment = _per_segment_frame(
        groups,
        designs,
        scenarios,
        seeds,
        segment_sse,
        segment_rows,
    )
    distance = _distance_frame(
        groups,
        designs,
        scenarios,
        seeds,
        distance_sse,
        distance_non_sse,
        distance_rows,
        distance_non_rows,
    )
    exposure = _measurement_exposure_frame(
        observed_groups,
        designs,
        scenarios,
        seeds,
        well_count,
        exposure_counters,
        effective_values,
    )
    effective = _effective_error_frame(
        observed_groups,
        designs,
        scenarios,
        effective_values,
        contaminated_locations,
        exposure_counters,
    )
    hierarchy = hierarchical_redundancy_bootstrap(
        full_sse,
        non_sse,
        rows,
        non_rows,
        np.asarray(
            [group.design_index for group in groups], dtype=np.int64
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    for item in hierarchy:
        item.update(
            _group_metadata(
                groups[int(item["group_index"])], designs, scenarios
            )
        )
    frontier = _frontier_frame(
        groups,
        designs,
        scenarios,
        metric_arrays,
        per_segment,
        exposure,
        hierarchy,
    )
    paired = _paired_deltas_frame(
        groups, designs, scenarios, metric_arrays
    )
    worst = _worst_wells_frame(
        groups,
        designs,
        scenarios,
        seeds,
        well_ids,
        folds,
        rows,
        full_sse,
    )

    run_config = {
        "run_id": run_id,
        "profile": (
            "full"
            if strict_full
            and len(designs) == len(LOCKED_DESIGNS)
            and len(scenarios) == len(REDUNDANCY_SCENARIOS)
            and len(seeds) == 100
            else "regression"
            if strict_full
            else "smoke"
        ),
        "data_root": str(data),
        "blend_oof": str(Path(blend_oof).resolve()),
        "terminal_frontier": str(Path(terminal_frontier).resolve()),
        "prior_qc_artifact": str(Path(prior_qc_artifact).resolve()),
        "fold_version": FOLD_VERSION,
        "fold_sha256": fold_hash,
        "wells": well_count,
        "rows": int(rows.sum()),
        "max_wells_per_fold": max_wells_per_fold,
        "selection_seed": int(selection_seed),
        "locked_designs": [
            {
                "config": design.name,
                "acquisition_class": design.acquisition_class,
                "physical_anchor_count": design.k,
                "replicates_per_anchor": design.replicates,
                "aggregation": design.aggregation,
                "total_readings": design.total_readings,
                "fractions": list(design.fractions),
            }
            for design in designs
        ],
        "scenarios": [
            _scenario_metadata(scenario) for scenario in scenarios
        ],
        "methods": list(METHODS),
        "oracle_scope": "O5_moderate_p10_m10 for all locked designs/dependences",
        "evaluation_groups": [
            _group_metadata(group, designs, scenarios) for group in groups
        ],
        "realization_seeds": list(seeds),
        "stable_random_keys": {
            "common_bias": "(seed, well_id, 'common_bias')",
            "replicate_0_independent": (
                "(seed, well_id, anchor_row, 'independent')"
            ),
            "replicate_0_outlier_indicator": (
                "(seed, well_id, anchor_row, 'outlier_indicator')"
            ),
            "replicate_0_outlier_sign": (
                "(seed, well_id, anchor_row, 'outlier_sign')"
            ),
            "replicate_r_ge_1": (
                "(seed, well_id, anchor_row, replicate_id, key)"
            ),
        },
        "gross_error_dependences": [
            DEPENDENCE_MEASUREMENT_INDEPENDENT,
            DEPENDENCE_ANCHOR_SHARED,
        ],
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_pairing": (
            "same well and per-selected-well realization draws across all designs"
        ),
        "robust_lt_4_definition": [
            "full fixed-well error-realization q95 < 4",
            "non-anchor fixed-well error-realization q95 < 4",
            "hierarchical full q97.5 < 4",
            "hierarchical non-anchor q97.5 < 4",
            "no persistent fold failure",
            "all realizations have zero wells_ge_40 and zero wells_ge_60",
        ],
        "conditioning": (
            "locked acausal piecewise-linear anchor-residual interpolation; "
            "carry forward after final physical anchor"
        ),
        "ps_boundary": "exact, unrepeated, unbiased, unnoised, no outlier",
        "non_anchor_definition": (
            "exclude each physical anchor row once regardless of replicate count"
        ),
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "deployable": False,
        "submission_eligible": False,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    _atomic_json(run_root / "run_config.json", run_config)
    per_seed.to_csv(run_root / "per_seed_metrics.csv", index=False)
    per_fold.to_csv(run_root / "per_fold_seed_metrics.csv", index=False)
    per_segment.to_csv(
        run_root / "per_segment_seed_metrics.csv", index=False
    )
    distance.to_csv(run_root / "distance_deciles.csv", index=False)
    exposure.to_csv(run_root / "measurement_exposure.csv", index=False)
    effective.to_csv(run_root / "effective_anchor_error.csv", index=False)
    paired.to_csv(run_root / "paired_deltas.csv", index=False)
    frontier.to_csv(run_root / "acquisition_frontier.csv", index=False)
    worst.to_csv(run_root / "worst_wells.csv", index=False)
    _atomic_json(
        run_root / "hierarchical_bootstrap.json",
        {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "well_and_realization_draws_paired_across_all_designs": True,
            "comparisons": hierarchy,
        },
    )

    observed_frontier = frontier[
        frontier["method"].eq(OPERATIONAL_METHOD)
    ]
    o5 = observed_frontier[
        observed_frontier["scenario"].eq(O5_SCENARIO)
    ]
    passing = (
        observed_frontier[
            observed_frontier["unconditional_operational_pass"]
        ][
            [
                "config",
                "physical_anchor_count",
                "total_readings",
                "acquisition_class",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["total_readings", "physical_anchor_count", "config"],
            kind="stable",
        )
    )
    spatial_passing = passing[
        passing["acquisition_class"].eq("single_read_spatial")
    ]
    k6_passing = passing[
        passing["physical_anchor_count"].eq(6)
    ]
    minimum_total = (
        passing.iloc[0]["config"] if len(passing) else None
    )
    minimum_physical = (
        passing.sort_values(
            ["physical_anchor_count", "total_readings", "config"],
            kind="stable",
        ).iloc[0]["config"]
        if len(passing)
        else None
    )
    pareto_records: list[str] = []
    for candidate in passing.itertuples(index=False):
        dominated = bool(
            (
                passing["physical_anchor_count"].le(
                    candidate.physical_anchor_count
                )
                & passing["total_readings"].le(candidate.total_readings)
                & (
                    passing["physical_anchor_count"].lt(
                        candidate.physical_anchor_count
                    )
                    | passing["total_readings"].lt(
                        candidate.total_readings
                    )
                )
            ).any()
        )
        if not dominated:
            pareto_records.append(str(candidate.config))

    summary = {
        "profile": run_config["profile"],
        "wells": well_count,
        "rows": int(rows.sum()),
        "designs": len(designs),
        "scenarios": len(scenarios),
        "evaluation_groups": len(groups),
        "realizations": len(seeds),
        "hierarchical_bootstrap_samples": int(bootstrap_samples),
        "regression_reproduction": regression,
        "minimum_passing_single_read_spatial_k": (
            int(spatial_passing["physical_anchor_count"].min())
            if len(spatial_passing)
            else None
        ),
        "minimum_passing_k6_replicates": (
            int(
                k6_passing.sort_values(
                    ["total_readings", "config"], kind="stable"
                ).iloc[0]["total_readings"]
                / 6
            )
            if len(k6_passing)
            else None
        ),
        "minimum_total_readings_design": minimum_total,
        "minimum_physical_locations_design": minimum_physical,
        "passing_pareto_frontier": pareto_records,
        "unconditional_passing_designs": passing[
            "config"
        ].astype(str).tolist(),
        "conditional_independent_only_designs": sorted(
            set(
                o5.loc[
                    o5["independent_reacquisition_pass"]
                    & ~o5["shared_fault_pass"],
                    "config",
                ].astype(str)
            )
        ),
        "catastrophic_wells_reappeared": bool(
            frontier["wells_ge_40_max"].gt(0).any()
            or frontier["wells_ge_60_max"].gt(0).any()
        ),
        "maximum_wells_ge_40": int(frontier["wells_ge_40_max"].max()),
        "maximum_wells_ge_60": int(frontier["wells_ge_60_max"].max()),
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "deployable": False,
        "submission_eligible": False,
        "kaggle_submission_created": False,
        "kaggle_submission_uploaded": False,
    }
    leakage = {
        "passed": True,
        "placement_target_independent": True,
        "placement_target_permutation_invariant": True,
        "random_draws_cross_process_deterministic": True,
        "configuration_order_does_not_change_keyed_draws": True,
        "replicate0_legacy_latent_max_abs_delta": (
            maximum_replicate0_latent_delta
        ),
        "shared_physical_row_replicate_latent_max_abs_delta": (
            maximum_shared_physical_replicate_latent_delta
        ),
        "common_bias_cross_design_max_abs_delta": (
            maximum_common_bias_cross_design_delta
        ),
        "physical_anchor_prediction_aggregate_max_abs_delta": (
            maximum_anchor_observation_delta
        ),
        "exact_conditioning_row_max_abs_delta": maximum_exact_row_delta,
        "placement_positions_sha256": placement_digest.hexdigest().upper(),
        "regression_reproduction": regression,
        "ps_boundary_exact_and_unperturbed": True,
        "non_anchor_excludes_physical_row_once": True,
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
        "counterfactual_information_experiment": True,
        "uses_validation_target_as_revealed_anchor": True,
        "deployable": False,
        "submission_eligible": False,
    }
    _atomic_json(run_root / "summary.json", summary)
    _atomic_json(run_root / "leakage_check.json", leakage)
    (run_root / "report.md").write_text(
        _render_report(frontier, paired, exposure, summary),
        encoding="utf-8",
    )
    artifact_audit = audit_anchor_acquisition_redundancy_artifacts(
        run_root
    )
    leakage["artifact_validation"] = artifact_audit
    leakage["passed"] = bool(
        regression["passed"]
        and artifact_audit["passed"]
        and maximum_replicate0_latent_delta == 0.0
        and maximum_shared_physical_replicate_latent_delta == 0.0
        and maximum_common_bias_cross_design_delta == 0.0
        and maximum_anchor_observation_delta == 0.0
        and maximum_exact_row_delta <= ROWWISE_TOLERANCE
    )
    summary["artifact_validation"] = artifact_audit
    _atomic_json(run_root / "summary.json", summary)
    _atomic_json(run_root / "leakage_check.json", leakage)
    if not leakage["passed"]:
        raise ValueError("redundancy leakage/reproduction/artifact audit failed")
    return summary


def _render_report(
    frontier: pd.DataFrame,
    paired: pd.DataFrame,
    exposure: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    observed = frontier[
        frontier["method"].eq(OPERATIONAL_METHOD)
    ].copy()
    o5 = observed[observed["scenario"].eq(O5_SCENARIO)].copy()
    controls = observed[
        observed["scenario"].isin(
            [
                "exact_clean",
                "gaussian_1_clean",
                "moderate_background",
                "moderate_m5_control",
            ]
        )
    ].copy()
    o5_columns = [
        "config",
        "outlier_dependence",
        "physical_anchor_count",
        "total_readings",
        "full_rmse_q95",
        "non_anchor_rmse_q95",
        "hierarchical_full_q975",
        "hierarchical_non_anchor_q975",
        "effective_outlier_shift_p95",
        "wells_ge_40_max",
        "wells_ge_60_max",
        "robust_lt_4",
        "independent_reacquisition_pass",
        "shared_fault_pass",
        "unconditional_operational_pass",
    ]
    oracle = frontier[
        frontier["scenario"].eq(O5_SCENARIO)
        & frontier["method"].ne(OPERATIONAL_METHOD)
    ]
    lines = [
        "# Sparse-anchor acquisition redundancy frontier",
        "",
        "Date: 2026-07-27 (Asia/Taipei)",
        "",
        "## Evidence status",
        "",
        "- `deployable: false`",
        "- `submission_eligible: false`",
        "- `counterfactual_information_experiment: true`",
        "- `uses_validation_target_as_revealed_anchor: true`",
        "- Kaggle submission created/uploaded: `false / false`",
        "",
        "This is a locked acausal retrospective information-acquisition "
        "diagnostic. It does not establish a deployable or online method.",
        "",
        "## Acquisition decision",
        "",
        f"- Minimum passing single-read spatial k: "
        f"`{summary['minimum_passing_single_read_spatial_k']}`.",
        f"- Minimum passing replicate count at six physical locations: "
        f"`{summary['minimum_passing_k6_replicates']}`.",
        f"- Minimum-total-readings passing design: "
        f"`{summary['minimum_total_readings_design']}`.",
        f"- Minimum-physical-locations passing design: "
        f"`{summary['minimum_physical_locations_design']}`.",
        f"- Passing Pareto designs: "
        f"`{summary['passing_pareto_frontier']}`.",
        f"- Conditional independent-only designs: "
        f"`{summary['conditional_independent_only_designs']}`.",
        f"- Catastrophic wells reappeared: "
        f"`{str(summary['catastrophic_wells_reappeared']).lower()}` "
        f"(maximum >=40 / >=60 counts "
        f"`{summary['maximum_wells_ge_40']} / "
        f"{summary['maximum_wells_ge_60']}`).",
        "",
        "## O5 operational frontier",
        "",
        _markdown_table(o5[o5_columns]) if len(o5) else "O5 not evaluated.",
        "",
        "A repeat design that passes only `measurement_independent` is a "
        "conditional reacquisition result. It is not shared-fault safe.",
        "",
        "## O5 oracle decomposition",
        "",
        _markdown_table(
            oracle[
                [
                    "config",
                    "outlier_dependence",
                    "method",
                    "full_rmse_q95",
                    "non_anchor_rmse_q95",
                    "hierarchical_full_q975",
                    "hierarchical_non_anchor_q975",
                    "robust_lt_4",
                ]
            ]
        )
        if len(oracle)
        else "O5 oracles not evaluated.",
        "",
        "## Control guard",
        "",
        _markdown_table(
            controls[
                [
                    "config",
                    "scenario",
                    "outlier_dependence",
                    "full_rmse_q95",
                    "non_anchor_rmse_q95",
                    "robust_lt_4",
                    "control_harm_pass",
                ]
            ]
        )
        if len(controls)
        else "Controls not evaluated.",
        "",
        "## Verification",
        "",
        f"- Regression reproduction passed: "
        f"`{str(summary['regression_reproduction']['passed']).lower()}`.",
        f"- Metric groups: `{summary['evaluation_groups']}`; "
        f"realizations: `{summary['realizations']}`; wells: "
        f"`{summary['wells']}`; rows: `{summary['rows']}`.",
        f"- Hierarchical bootstrap samples: "
        f"`{summary['hierarchical_bootstrap_samples']}`.",
        "- Physical-anchor rows are excluded once from non-anchor scoring, "
        "independent of replicate count.",
        "- See `measurement_exposure.csv`, `effective_anchor_error.csv`, and "
        "`paired_deltas.csv` for dependence, effective-shift, and oracle gaps.",
        "",
    ]
    return "\n".join(lines)


def audit_anchor_acquisition_redundancy_artifacts(
    run_root: Path,
) -> dict[str, Any]:
    """Reconstruct pooled metrics and validate compact redundancy artifacts."""
    root = Path(run_root).resolve()
    config = json.loads(
        (root / "run_config.json").read_text(encoding="utf-8")
    )
    per_seed = pd.read_csv(root / "per_seed_metrics.csv")
    frontier = pd.read_csv(root / "acquisition_frontier.csv")
    per_fold = pd.read_csv(root / "per_fold_seed_metrics.csv")
    per_segment = pd.read_csv(root / "per_segment_seed_metrics.csv")
    distance = pd.read_csv(root / "distance_deciles.csv")
    exposure = pd.read_csv(root / "measurement_exposure.csv")
    effective = pd.read_csv(root / "effective_anchor_error.csv")
    paired = pd.read_csv(root / "paired_deltas.csv")
    worst = pd.read_csv(root / "worst_wells.csv")
    keys = [
        "config",
        "method",
        "scenario",
        "outlier_dependence",
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
    expected_groups = len(config["evaluation_groups"]) * len(
        config["realization_seeds"]
    )
    expected_per_well_rows = expected_groups * int(config["wells"])
    table_duplicate_count = sum(
        (
            int(per_seed.duplicated(keys).sum()),
            int(
                frontier.duplicated(
                    [
                        "config",
                        "method",
                        "scenario",
                        "outlier_dependence",
                    ]
                ).sum()
            ),
            int(per_fold.duplicated(keys + ["fold"]).sum()),
            int(per_segment.duplicated(keys + ["segment"]).sum()),
            int(distance.duplicated(keys + ["distance_decile"]).sum()),
            int(
                exposure.duplicated(
                    [
                        "config",
                        "method",
                        "scenario",
                        "outlier_dependence",
                        "realization_seed",
                    ]
                ).sum()
            ),
            int(
                effective.duplicated(
                    [
                        "config",
                        "method",
                        "scenario",
                        "outlier_dependence",
                        "scope",
                        "scope_value",
                    ]
                ).sum()
            ),
        )
    )
    numeric_nan = sum(
        int(frame.select_dtypes(include=[np.number]).isna().sum().sum())
        for frame in (
            per_seed,
            frontier,
            per_fold,
            per_segment,
            distance,
            exposure,
            effective,
            paired,
            worst,
        )
    )
    expected_rows_by_group = int(config["rows"])
    passed = bool(
        len(aggregate) == expected_groups
        and len(per_seed) == expected_groups
        and per_well_rows == expected_per_well_rows
        and joined["_merge"].eq("both").all()
        and aggregate["wells"].eq(int(config["wells"])).all()
        and aggregate["rows"].eq(expected_rows_by_group).all()
        and float(full_delta.max()) <= METRIC_TOLERANCE
        and float(non_delta.max()) <= METRIC_TOLERANCE
        and table_duplicate_count == 0
        and unexpected_nan == 0
        and numeric_nan == 0
        and len(exposure) > 0
        and len(effective) > 0
        and len(frontier) == len(config["evaluation_groups"])
    )
    return {
        "passed": passed,
        "expected_metric_groups": expected_groups,
        "actual_metric_groups": int(len(aggregate)),
        "per_well_rows": per_well_rows,
        "expected_per_well_rows": expected_per_well_rows,
        "max_full_rmse_reconstruction_delta": float(full_delta.max()),
        "max_non_anchor_rmse_reconstruction_delta": float(non_delta.max()),
        "duplicate_key_count": table_duplicate_count,
        "unexpected_numeric_nan_count": unexpected_nan + numeric_nan,
        "frontier_rows": int(len(frontier)),
        "measurement_exposure_rows": int(len(exposure)),
        "effective_anchor_error_rows": int(len(effective)),
    }
